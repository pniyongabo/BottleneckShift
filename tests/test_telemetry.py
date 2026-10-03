from datetime import datetime, timezone

import pytest

from bottleneckshift import telemetry
from synthetic import gpu_csv, mpstat_text


def utc(text):
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def test_parse_gpu_csv_reads_units_and_missing_values_and_tolerates_truncation():
    text = gpu_csv(samples=3).replace("120.50 W", "[N/A]", 1) + "2026/10/03 03:45:00.000, 9"
    samples = telemetry.parse_gpu_csv(text)
    assert len(samples) == 3
    assert samples[0]["gpu_util"] == 90 and samples[0]["memory_used_mib"] == 13900
    assert samples[0]["power_w"] is None and samples[1]["power_w"] == 120.5
    assert samples[1]["t"] == utc("2026-10-03 03:44:58.200")


def test_parse_gpu_csv_rejects_bad_header_and_malformed_middle_line():
    with pytest.raises(ValueError, match="header"):
        telemetry.parse_gpu_csv("timestamp, name\n")
    lines = gpu_csv(samples=3).splitlines()
    lines[2] = "garbage"
    with pytest.raises(ValueError, match="line 3"):
        telemetry.parse_gpu_csv("\n".join(lines) + "\n")


def test_parse_gpu_csv_accepts_clocks_sm_header_spelling():
    text = gpu_csv(samples=1).replace("clocks.current.sm", "clocks.sm")
    assert len(telemetry.parse_gpu_csv(text)) == 1


def test_parse_mpstat_reads_rows_skips_averages_and_handles_midnight():
    samples = telemetry.parse_mpstat(mpstat_text(start="23:59:58", seconds=4, cpus=2, idle=60.0))
    overall = [sample for sample in samples if sample["cpu"] == "all"]
    assert len(overall) == 4 and len(samples) == 12
    assert overall[0]["t"] == utc("2026-10-03 23:59:58.0")
    assert overall[-1]["t"] == utc("2026-10-04 00:00:01.0")
    assert overall[0]["util"] == pytest.approx(40.0) and overall[0]["usr"] == 20.0


def test_parse_mpstat_tolerates_truncated_last_line_but_not_middle_or_12_hour_clock():
    text = mpstat_text(seconds=2)
    truncated = text.split("Average:")[0] + "03:45:00     all   20.0"
    assert len(telemetry.parse_mpstat(truncated)) == 6
    lines = text.splitlines()
    lines[3] = lines[3][:20]
    with pytest.raises(ValueError, match="mpstat line 4"):
        telemetry.parse_mpstat("\n".join(lines))
    with pytest.raises(ValueError, match="12-hour"):
        telemetry.parse_mpstat(text.replace("03:44:58     all", "03:44:58 AM  all"))
    with pytest.raises(ValueError, match="ISO date"):
        telemetry.parse_mpstat(text.replace("2026-10-03", "10/03/2026"))


def test_window_stats_and_coverage():
    gpu = telemetry.parse_gpu_csv(gpu_csv(samples=10))
    start, end = utc("2026-10-03 03:44:58.0"), utc("2026-10-03 03:44:59.8")
    stats = telemetry.window_stats(gpu, start, end, "gpu")
    assert stats["samples"] == 10 and stats["gpu_util_mean"] == 90
    assert telemetry.coverage(gpu, start, end, "gpu")["max_gap_s"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="does not cover"):
        telemetry.coverage(gpu, start, utc("2026-10-03 03:45:10.0"), "gpu")
    with pytest.raises(ValueError, match="gap"):
        telemetry.coverage(gpu[:2] + gpu[-1:], start, end, "gpu")
    cpu = telemetry.parse_mpstat(mpstat_text(seconds=5, idle=75.0))
    stats = telemetry.window_stats(cpu, utc("2026-10-03 03:44:58.0"), utc("2026-10-03 03:45:02.0"), "cpu")
    assert stats == {"samples": 5, "cpu_util_mean": 25.0, "cpu_core_util_max": 25.0}


class FakeProcess:
    def __init__(self, exit_after_signal=True, returncode=None):
        self.pid, self.returncode, self.exit_after_signal = 4242, returncode, exit_after_signal

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -2
        return self.returncode


def test_sidecar_writes_output_records_lifecycle_and_detects_early_exit(tmp_path, monkeypatch):
    signals = []
    monkeypatch.setattr(telemetry.os, "killpg", lambda pid, sig: signals.append(sig))
    process = FakeProcess()

    def popen(command, stdout, **kwargs):
        stdout.write(gpu_csv(samples=2))
        stdout.flush()
        assert kwargs["env"]["TZ"] == "UTC" and kwargs["start_new_session"]
        return process

    sidecar = telemetry.Sidecar("gpu", telemetry.GPU_COMMAND, telemetry.GPU_ENV, tmp_path / "gpu.csv", popen)
    sidecar.start()
    sidecar.wait_for_data(telemetry.gpu_ready, timeout=1, sleep=lambda _: None)
    sidecar.stop()
    assert signals == [telemetry.signal.SIGINT]
    assert sidecar.record["stop_signal"] == "SIGINT" and "first_data_at_utc" in sidecar.record

    dead = telemetry.Sidecar("cpu", ["mpstat"], {}, tmp_path / "cpu.txt",
                             lambda *a, **k: FakeProcess(returncode=1))
    dead.start()
    with pytest.raises(RuntimeError, match="exited early"):
        dead.wait_for_data(telemetry.cpu_ready, timeout=1, sleep=lambda _: None)
    assert dead.record["exited_early"] and dead.record["returncode"] == 1
