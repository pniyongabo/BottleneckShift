"""Small synthetic artifacts shared by tests: Prometheus scrapes and sampler output."""

from bottleneckshift.prometheus import REQUIRED_FAMILIES, TOKEN_FAMILIES

LABELS = 'engine="0",model_name="synthetic"'


def scrape(requests: int, *, tokens: int | None = None, seconds: float = 0.01,
           start_time: float = 1000.0, running: int = 0, waiting: int = 0) -> str:
    """A scrape whose per-request histograms have `requests` observations of `seconds`."""
    tokens = requests * 3 if tokens is None else tokens
    lines = [f"# HELP process_start_time_seconds synthetic", f"process_start_time_seconds {start_time}",
             f"vllm:num_requests_running{{{LABELS}}} {running}",
             f"vllm:num_requests_waiting{{{LABELS}}} {waiting}"]
    for family in REQUIRED_FAMILIES + TOKEN_FAMILIES:
        count = tokens if family in TOKEN_FAMILIES else requests
        lines += [f"# TYPE {family} histogram",
                  f'{family}_bucket{{{LABELS},le="0.005"}} 0',
                  f'{family}_bucket{{{LABELS},le="0.02"}} {count}',
                  f'{family}_bucket{{{LABELS},le="+Inf"}} {count}',
                  f"{family}_sum{{{LABELS}}} {count * seconds}",
                  f"{family}_count{{{LABELS}}} {count}",
                  f"{family}_created{{{LABELS}}} 999.0"]
    return "\n".join(lines) + "\n"


NVIDIA_HEADER = ("timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], "
                 "power.draw [W], clocks.current.sm [MHz]")


def gpu_csv(start: str = "2026/10/03 03:44:58", samples: int = 10, step_ms: int = 200,
            utilization: int = 90) -> str:
    from datetime import datetime, timedelta
    first = datetime.strptime(start, "%Y/%m/%d %H:%M:%S")
    rows = [NVIDIA_HEADER]
    for index in range(samples):
        stamp = (first + timedelta(milliseconds=index * step_ms)).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        rows.append(f"{stamp}, {utilization} %, 40 %, 13900 MiB, 120.50 W, 1560 MHz")
    return "\n".join(rows) + "\n"


def mpstat_text(start: str = "03:44:58", seconds: int = 5, date: str = "2026-10-03",
                cpus: int = 2, idle: float = 75.0) -> str:
    from datetime import datetime, timedelta
    first = datetime.strptime(f"{date} {start}", "%Y-%m-%d %H:%M:%S")
    lines = [f"Linux 6.14.0-37-generic (synthetic-host) \t{date} \t_x86_64_\t({cpus} CPU)", ""]
    header = "{time}     CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle"
    for index in range(seconds):
        stamp = (first + timedelta(seconds=index)).strftime("%H:%M:%S")
        lines.append(header.format(time=stamp))
        for cpu in ["all", *range(cpus)]:
            lines.append(f"{stamp}     {cpu:>3}   20.00    0.00    5.00    0.00    0.00    0.00    0.00"
                         f"    0.00    0.00   {idle:.2f}")
        lines.append("")
    lines.append(header.format(time="Average:"))
    for cpu in ["all", *range(cpus)]:
        lines.append(f"Average:     {cpu:>3}   20.00    0.00    5.00    0.00    0.00    0.00    0.00"
                     f"    0.00    0.00   {idle:.2f}")
    return "\n".join(lines) + "\n"


def requests_document(prompts: int, input_tokens: int, output_tokens: int, *, ttft: float = 0.02,
                      itl: float = 0.004) -> dict:
    """A vLLM-shaped requests.json with exact token lengths and no errors."""
    intervals = [[itl] * (output_tokens - 1) for _ in range(prompts)]
    e2el = ttft + itl * (output_tokens - 1)
    return {"ttfts": [ttft] * prompts, "itls": intervals, "input_lens": [input_tokens] * prompts,
            "output_lens": [output_tokens] * prompts, "errors": [""] * prompts,
            "duration": e2el * prompts, "median_e2el_ms": e2el * 1000,
            "request_throughput": 1 / e2el, "output_throughput": output_tokens / e2el}


def run_series(tmp_path, monkeypatch, run_experiment, config_text: str, series_id: str = "synthetic",
               metrics_offset: int = 0):
    """Run the real runner main() with fake vLLM, samplers, nvidia-smi, and contention.

    `metrics_offset` adds stray server requests to every after-scrape (to test validation).
    Returns the series directory.
    """
    import json
    import sys
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    config_path = tmp_path / f"{series_id}.toml"
    config_path.write_text(config_text)
    state = {"requests": 0, "tokens": 0, "processes": {}}
    now = datetime.now(timezone.utc) - timedelta(seconds=1)

    class Process:
        def __init__(self, pid, on_stop=None):
            self.pid, self.returncode, self.on_stop = pid, None, on_stop
            state["processes"][pid] = self

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = 0
                if self.on_stop:
                    self.on_stop()
            return self.returncode

    def popen(command, stdout=None, **kwargs):
        if command[0] == "vllm":
            server = Process(100)
            server.returncode = 0  # treated as already exited at teardown
            return server
        if command[0] == "nvidia-smi":
            stdout.write(gpu_csv(start=now.strftime("%Y/%m/%d %H:%M:%S"), samples=60))
            stdout.flush()
            return Process(200)
        if command[0] == "mpstat":
            stdout.write(mpstat_text(start=now.strftime("%H:%M:%S"), seconds=12, date=now.strftime("%Y-%m-%d")))
            stdout.flush()
            return Process(300)
        if command[1].endswith("gpu_contention.py"):
            log = Path(command[command.index("--log") + 1])
            mode = command[command.index("--mode") + 1]
            duty = float(command[command.index("--duty-cycle") + 1]) if "--duty-cycle" in command else 0.0
            log.write_text(json.dumps({"event": "start", "mode": mode}) + "\n"
                           + json.dumps({"event": "ready"}) + "\n")

            def stop():
                with log.open("a") as handle:
                    handle.write(json.dumps({"event": "stop", "achieved_duty": duty, "kernel_duty": duty,
                                             "total_steps": 0 if mode == "sham" else 50}) + "\n")
            return Process(400, on_stop=stop)
        raise AssertionError(f"unexpected Popen {command}")

    def killpg(pid, sig):
        state["processes"][pid].wait()

    class Completed:
        def __init__(self, stdout=""):
            self.returncode, self.stdout, self.stderr = 0, stdout, ""

    def run(command, **kwargs):
        if command[0] == "nvidia-smi":
            if command[1].startswith("--query-compute-apps"):
                return Completed("400, 700\n")
            return Completed("1, 15352\n")
        prompts = int(command[command.index("--num-prompts") + 1])
        inputs = int(command[command.index("--random-input-len") + 1])
        outputs = int(command[command.index("--random-output-len") + 1])
        result_dir = Path(command[command.index("--result-dir") + 1])
        (result_dir / "requests.json").write_text(json.dumps(requests_document(prompts, inputs, outputs)))
        state["requests"] += prompts
        state["tokens"] += prompts * (outputs - 1)
        state["pending_offset"] = metrics_offset
        return Completed()

    def fetch(url):
        extra = state.pop("pending_offset", 0)
        state["requests"] += extra
        return scrape(state["requests"], tokens=state["tokens"])

    monkeypatch.setattr(run_experiment.subprocess, "Popen", popen)
    monkeypatch.setattr(run_experiment.subprocess, "run", run)
    monkeypatch.setattr(run_experiment.os, "killpg", killpg)
    monkeypatch.setattr(run_experiment.telemetry.os, "killpg", killpg)
    monkeypatch.setattr(run_experiment, "fetch_metrics", fetch)
    monkeypatch.setattr(run_experiment.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(run_experiment, "wait_until_healthy", lambda config, process: None)
    monkeypatch.setattr(run_experiment, "preflight", lambda config, root: {"ok": True, "checks": {}})
    monkeypatch.setattr(run_experiment, "environment", lambda repo: {"synthetic": True})
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(config_path),
                                      "--series-id", series_id, "--results-dir", str(tmp_path / "runs")])
    run_experiment.main()
    return tmp_path / "runs" / series_id


def small_config(source: str, *, prompts: int = 4, contention: str = "") -> str:
    """Shrink a Phase B config for tests: 1 repetition, few prompts, no cooldown."""
    import re
    for key, value in {"repetitions": 1, "warmup_prompts": 2, "num_prompts": prompts, "cooldown_seconds": 0}.items():
        source = re.sub(rf"(?m)^({key}\s*=\s*)\d+", rf"\g<1>{value}", source)
    return source + contention
