"""GPU and CPU telemetry sidecars: fixed commands, parsers, and window statistics."""

import csv
from datetime import datetime, timedelta, timezone
import io
import os
import re
import signal
import subprocess
import time

GPU_FIELDS = "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm"
GPU_COMMAND = ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv", "-lms", "200"]
GPU_ENV = {"TZ": "UTC"}
GPU_COLUMNS = ("timestamp", "utilization.gpu", "utilization.memory", "memory.used",
               "power.draw", "clocks.current.sm")
CPU_COMMAND = ["mpstat", "-P", "ALL", "1"]
CPU_ENV = {"LC_ALL": "C", "S_TIME_FORMAT": "ISO", "TZ": "UTC"}
MAX_GAP_SECONDS = {"gpu": 1.0, "cpu": 3.0}

_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _number(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text in ("N/A", "-"):
        return None
    return float(text.split()[0])


def parse_gpu_csv(text: str) -> list[dict]:
    """Parse `nvidia-smi --format=csv` output (one GPU); tolerate a truncated last line."""
    rows = list(csv.reader(io.StringIO(text), skipinitialspace=True))
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError("gpu telemetry is empty")
    # nvidia-smi may print the clocks.sm query field as clocks.current.sm.
    header = tuple(column.split(" [")[0].strip().replace("clocks.sm", "clocks.current.sm")
                   for column in rows[0])
    if header != GPU_COLUMNS:
        raise ValueError(f"unexpected nvidia-smi header: {rows[0]}")
    samples = []
    for index, row in enumerate(rows[1:], start=2):
        try:
            if len(row) != len(GPU_COLUMNS):
                raise ValueError(f"expected {len(GPU_COLUMNS)} fields")
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            sample = {"t": stamp, "gpu_util": _number(row[1]), "mem_util": _number(row[2]),
                      "memory_used_mib": _number(row[3]), "power_w": _number(row[4]),
                      "sm_clock_mhz": _number(row[5])}
        except ValueError as exc:
            if index == len(rows):
                break  # sampler stopped mid-line
            raise ValueError(f"gpu telemetry line {index}: {exc}") from exc
        if samples and stamp == samples[-1]["t"]:
            raise ValueError("repeated gpu timestamp: telemetry assumes a single GPU")
        samples.append(sample)
    return samples


def parse_mpstat(text: str) -> list[dict]:
    """Parse `mpstat -P ALL 1` (C locale, ISO time, UTC); `Average:` lines are skipped."""
    lines = text.splitlines()
    date = next((match.group(1) for line in lines[:3] if (match := _DATE.search(line))), None)
    if date is None:
        raise ValueError("mpstat banner has no ISO date; is S_TIME_FORMAT=ISO set?")
    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    columns, samples, previous = None, [], None
    content = [index for index, line in enumerate(lines) if line.strip()]
    last = content[-1] if content else -1
    for index, line in enumerate(lines):
        fields = line.split()
        if not fields or fields[0] in ("Linux", "Average:"):
            continue
        if "AM" in fields[:3] or "PM" in fields[:3]:
            raise ValueError("mpstat output uses a 12-hour clock; set LC_ALL=C")
        if len(fields) > 1 and fields[1] == "CPU":
            columns = [name.lstrip("%") for name in fields[2:]]
            continue
        try:
            if columns is None or not _TIME.match(fields[0]) or len(fields) != len(columns) + 2:
                raise ValueError("unexpected mpstat line")
            clock = datetime.strptime(fields[0], "%H:%M:%S").time()
            stamp = day.replace(hour=clock.hour, minute=clock.minute, second=clock.second)
            if previous and stamp < previous - timedelta(hours=1):
                day += timedelta(days=1)  # midnight rollover
                stamp += timedelta(days=1)
            values = {name: float(value) for name, value in zip(columns, fields[2:])}
        except ValueError as exc:
            if index == last:
                break  # sampler stopped mid-line
            raise ValueError(f"mpstat line {index + 1}: {exc}: {line!r}") from exc
        previous = stamp
        samples.append({"t": stamp, "cpu": fields[1], "util": 100.0 - values["idle"], **values})
    return samples


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def window_stats(samples: list[dict], start: datetime, end: datetime, kind: str) -> dict:
    """Summarize samples whose timestamps fall within [start, end]."""
    inside = [sample for sample in samples if start <= sample["t"] <= end]
    if kind == "gpu":
        def values(key):
            return [s[key] for s in inside if s[key] is not None]
        return {"samples": len(inside), "gpu_util_mean": _mean(values("gpu_util")),
                "gpu_util_max": max(values("gpu_util"), default=None),
                "power_w_mean": _mean(values("power_w")),
                "memory_used_mib_max": max(values("memory_used_mib"), default=None)}
    overall = [s["util"] for s in inside if s["cpu"] == "all"]
    per_cpu = [s["util"] for s in inside if s["cpu"] != "all"]
    return {"samples": len(overall), "cpu_util_mean": _mean(overall),
            "cpu_core_util_max": max(per_cpu, default=None)}


def coverage(samples: list[dict], start: datetime, end: datetime, kind: str) -> dict:
    """Check samples span [start, end] with no gap longer than the sampler allows."""
    times = sorted({sample["t"] for sample in samples})
    limit = MAX_GAP_SECONDS[kind]
    if not times:
        raise ValueError(f"{kind} telemetry has no samples")
    if times[0] > start + timedelta(seconds=limit) or times[-1] < end - timedelta(seconds=limit):
        raise ValueError(f"{kind} telemetry {times[0]}–{times[-1]} does not cover {start}–{end}")
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if start <= b and a <= end]
    largest = max(gaps, default=0.0)
    if largest > limit:
        raise ValueError(f"{kind} telemetry has a {largest:.2f} s gap (limit {limit} s)")
    return {"samples": len(times), "max_gap_s": largest}


class Sidecar:
    """A background sampler whose stdout is written to a file."""

    def __init__(self, name: str, command: list[str], env: dict, path, popen=subprocess.Popen):
        self.name, self.command, self.env, self.path = name, command, env, path
        self.popen, self.process, self.record = popen, None, {
            "command": command, "env": env, "path": str(path), "stderr": str(path) + ".stderr"}

    def start(self) -> None:
        self.record["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "w") as stdout, open(self.record["stderr"], "w") as stderr:
            self.process = self.popen(self.command, stdout=stdout, stderr=stderr,
                                      env={**os.environ, **self.env}, start_new_session=True)

    def wait_for_data(self, ready, timeout: float = 15.0, sleep=time.sleep) -> None:
        """Wait until `ready(text)` accepts the output written so far."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check_alive()
            if ready(open(self.path).read()):
                self.record["first_data_at_utc"] = datetime.now(timezone.utc).isoformat()
                return
            sleep(0.2)
        raise TimeoutError(f"{self.name} sampler produced no data within {timeout} s")

    def check_alive(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            self.record["exited_early"] = True
            self.record["returncode"] = self.process.returncode
            raise RuntimeError(f"{self.name} sampler exited early with status {self.process.returncode}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop with SIGINT (so mpstat flushes), then SIGTERM, then SIGKILL."""
        self.record["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
        if self.process is None or self.process.poll() is not None:
            if self.process is not None:
                self.record.setdefault("returncode", self.process.returncode)
            return
        for stop_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.process.pid, stop_signal)
            except ProcessLookupError:
                break
            try:
                self.process.wait(timeout=timeout)
                self.record["stop_signal"] = stop_signal.name
                break
            except subprocess.TimeoutExpired:
                continue
        self.record["returncode"] = self.process.returncode


def gpu_ready(text: str) -> bool:
    try:
        return bool(parse_gpu_csv(text))
    except ValueError:
        return False


def cpu_ready(text: str) -> bool:
    try:
        return bool(parse_mpstat(text))
    except ValueError:
        return False
