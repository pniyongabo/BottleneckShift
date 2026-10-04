#!/usr/bin/env python3
"""Run a reviewed vLLM experiment plan and preserve its raw output."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bottleneckshift import contention, prometheus, telemetry  # noqa: E402
from bottleneckshift.config import load_config  # noqa: E402
from bottleneckshift.manifest import environment  # noqa: E402

SERIES_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SETTLE_TIMEOUT_SECONDS = 10
SCRAPE_ATTEMPTS = 3


def server_command(config: dict) -> list[str]:
    server = config["server"]
    command = ["vllm", "serve", server["model"], "--host", server["host"], "--port", str(server["port"]),
            "--tensor-parallel-size", str(server["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(server["gpu_memory_utilization"])]
    if "model_revision" in server:
        command += ["--revision", str(server["model_revision"])]
    if "tokenizer_revision" in server:
        command += ["--tokenizer-revision", str(server["tokenizer_revision"])]
    if "generation_config" in server:
        command += ["--generation-config", str(server["generation_config"])]
    if "enable_prefix_caching" in server:
        command.append("--enable-prefix-caching" if server["enable_prefix_caching"]
                       else "--no-enable-prefix-caching")
    if "max_num_batched_tokens" in server:
        command += ["--max-num-batched-tokens", str(server["max_num_batched_tokens"])]
    return command


def benchmark_command(config: dict, max_concurrency: int, output_dir: Path,
                      num_prompts: int | None = None, request_id_prefix: str | None = None) -> list[str]:
    server, workload = config["server"], config["workload"]
    command = ["vllm", "bench", "serve", "--backend", "vllm", "--model", server["model"],
            "--host", server["host"], "--port", str(server["port"]), "--dataset-name", "random",
            "--random-input-len", str(workload["input_tokens"]), "--random-output-len", str(workload["output_tokens"]),
            "--num-prompts", str(num_prompts if num_prompts is not None else workload["num_prompts"]),
            "--seed", str(workload["seed"]), "--request-rate", "inf",
            "--max-concurrency", str(max_concurrency), "--save-result", "--save-detailed",
            "--result-dir", str(output_dir), "--result-filename", "requests.json"]
    sampling = config.get("sampling", {})
    if "temperature" in sampling:
        command += ["--temperature", str(sampling["temperature"])]
    if "top_p" in sampling:
        command += ["--top-p", str(sampling["top_p"])]
    if "percentile_metrics" in config["execution"]:
        command += ["--percentile-metrics", ",".join(config["execution"]["percentile_metrics"])]
    if request_id_prefix is not None:
        command += ["--request-id-prefix", request_id_prefix]
    return command


def request_id_prefix(config: dict, root: Path, condition: str, repetition: int | None) -> str | None:
    """`<series>-<condition>-[rep-NN-]`, when the config opts in."""
    if not config["execution"].get("request_id_prefix"):
        return None
    suffix = f"rep-{repetition:02d}-" if repetition is not None else ""
    return f"{root.name}-{condition}-{suffix}"


def build_plan(config: dict, root: Path) -> list[dict]:
    """Return the complete, ordered warm-up and measurement plan."""
    plan = []
    execution = config["execution"]
    if execution["warmup_prompts"]:
        if "warmup_concurrencies" in execution:
            warmups = [(f"warmup_c{value}", root / "warmup" / f"c{value}", value)
                       for value in execution["warmup_concurrencies"]]
        else:
            # Configs without warmup_concurrencies keep the historical single warm-up layout.
            warmups = [("warmup", root / "warmup", execution["warmup_max_concurrency"])]
        for condition, directory, concurrency in warmups:
            plan.append({"measurement_role": "warmup", "condition": condition,
                         "repetition": None, "max_concurrency": concurrency, "directory": directory,
                         "command": benchmark_command(
                             config, concurrency, directory, execution["warmup_prompts"],
                             request_id_prefix(config, root, condition, None))})
    for condition in config["condition"]:
        for repetition in range(1, config["experiment"]["repetitions"] + 1):
            directory = root / condition["name"] / f"rep-{repetition:02d}"
            plan.append({"measurement_role": "measured", "condition": condition["name"],
                         "repetition": repetition, "max_concurrency": condition["max_concurrency"],
                         "directory": directory,
                         "command": benchmark_command(
                             config, condition["max_concurrency"], directory, None,
                             request_id_prefix(config, root, condition["name"], repetition))})
    return plan


def sampler_specs(config: dict, root: Path) -> list[tuple]:
    """(name, command, env, output path) for each enabled telemetry sampler."""
    enabled = config.get("telemetry", {})
    specs = []
    if enabled.get("gpu_sampler"):
        specs.append(("gpu", telemetry.GPU_COMMAND, telemetry.GPU_ENV, root / "gpu.csv"))
    if enabled.get("cpu_sampler"):
        specs.append(("cpu", telemetry.CPU_COMMAND, telemetry.CPU_ENV, root / "cpu.txt"))
    return specs


def contention_settings(config: dict) -> dict | None:
    table = config.get("contention", {"mode": "off"})
    return None if table["mode"] == "off" else table


def contention_command(config: dict, root: Path) -> list[str] | None:
    table = contention_settings(config)
    if table is None:
        return None
    command = [sys.executable, str(REPO / "scripts" / "gpu_contention.py"), "--mode", table["mode"],
               "--period-ms", str(table.get("period_ms", 100)),
               "--memory-cap-mib", str(table["memory_cap_mib"]), "--log", str(root / "contention.jsonl")]
    if table["mode"] == "active":
        command += ["--duty-cycle", str(table["duty_cycle"])]
    if "matrix_size" in table:
        command += ["--matrix-size", str(table["matrix_size"])]
    return command


def metrics_url(config: dict) -> str:
    return f"http://{config['server']['host']}:{config['server']['port']}/metrics"


def fetch_metrics(url: str) -> str:
    last_error = None
    for _ in range(SCRAPE_ATTEMPTS):
        try:
            with urlopen(url, timeout=5) as response:
                return response.read().decode()
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"cannot scrape {url}: {last_error}")


def settled_metrics(url: str) -> tuple[str, int]:
    """Scrape until the server is idle and its request count stops changing."""
    deadline, previous, polls = time.monotonic() + SETTLE_TIMEOUT_SECONDS, None, 0
    while True:
        text, polls = fetch_metrics(url), polls + 1
        samples = prometheus.parse(text)
        count = (prometheus.histogram(samples, prometheus.REQUEST_COUNT_FAMILY) or {}).get("count")
        if prometheus.is_idle(samples) and count is not None and count == previous:
            return text, polls
        if time.monotonic() >= deadline:
            raise RuntimeError(f"server did not settle within {SETTLE_TIMEOUT_SECONDS} s after a run")
        previous = count
        time.sleep(0.25)


def query_nvidia(arguments: list[str]) -> list[list[str]]:
    result = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi {' '.join(arguments)} failed: {result.stderr.strip()}")
    return [[field.strip() for field in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


class Instrumentation:
    """Metrics scrapes, telemetry sidecars, and contention for one series."""

    def __init__(self, config: dict, root: Path, manifest: dict):
        self.config, self.root, self.manifest = config, root, manifest
        self.metrics = config.get("telemetry", {}).get("server_metrics", False)
        self.sidecars = [telemetry.Sidecar(name, command, env, path, popen=subprocess.Popen)
                         for name, command, env, path in sampler_specs(config, root)]
        self.contention = contention_settings(config)
        self.contention_process = None
        if self.sidecars:
            manifest["telemetry"] = {sidecar.name: sidecar.record for sidecar in self.sidecars}
        if self.contention:
            manifest["contention"] = {"command": contention_command(config, root),
                                      "log": str(root / "contention.jsonl"), **self.contention}

    def start(self) -> None:
        if self.metrics:
            text = fetch_metrics(metrics_url(self.config))
            prometheus.check_scrape(text)
            (self.root / "metrics-initial.prom").write_text(text)
            self.manifest["server_metrics"] = {"initial": str(self.root / "metrics-initial.prom")}
        if self.sidecars or self.contention:
            used, total = query_nvidia(["--query-gpu=memory.used,memory.total"])[0]
            self.manifest["gpu_memory_after_server_healthy_mib"] = {"used": float(used), "total": float(total)}
        for sidecar in self.sidecars:
            sidecar.start()
            sidecar.wait_for_data(telemetry.gpu_ready if sidecar.name == "gpu" else telemetry.cpu_ready)
        if self.contention:
            self._start_contention()

    def _start_contention(self) -> None:
        record = self.manifest["contention"]
        log_path = Path(record["log"])
        with (self.root / "contention.stderr").open("w") as stderr:
            self.contention_process = subprocess.Popen(record["command"], stdout=stderr, stderr=subprocess.STDOUT,
                                                       start_new_session=True, text=True)
        record["pid"], record["started_at_utc"] = self.contention_process.pid, datetime.now(timezone.utc).isoformat()
        deadline = time.monotonic() + self.contention.get("ready_timeout_seconds", 120)
        while not (log_path.exists() and '"ready"' in log_path.read_text()):
            if self.contention_process.poll() is not None:
                raise RuntimeError(f"contention exited with status {self.contention_process.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("contention generator did not become ready")
            time.sleep(0.2)
        apps = {int(pid): float(used) for pid, used in query_nvidia(["--query-compute-apps=pid,used_memory"])}
        record["vram_mib"] = apps.get(self.contention_process.pid)
        if record["vram_mib"] is None:
            raise RuntimeError("contention process is not listed by nvidia-smi --query-compute-apps")
        if record["vram_mib"] > self.contention["memory_cap_mib"]:
            raise RuntimeError(f"contention uses {record['vram_mib']} MiB, above its "
                               f"{self.contention['memory_cap_mib']} MiB cap")

    def check(self) -> None:
        for sidecar in self.sidecars:
            sidecar.check_alive()
        if self.contention_process is not None and self.contention_process.poll() is not None:
            self.manifest["contention"]["exited_early"] = True
            raise RuntimeError(f"contention exited early with status {self.contention_process.returncode}")

    def before_run(self, directory: Path) -> dict | None:
        if not self.metrics:
            return None
        text = fetch_metrics(metrics_url(self.config))
        (directory / "metrics-before.prom").write_text(text)
        return {"before": str(directory / "metrics-before.prom"),
                "before_scraped_at_utc": datetime.now(timezone.utc).isoformat()}

    def after_run(self, directory: Path, record: dict | None) -> dict | None:
        if record is None:
            return None
        text, polls = settled_metrics(metrics_url(self.config))
        (directory / "metrics-after.prom").write_text(text)
        return {**record, "after": str(directory / "metrics-after.prom"),
                "after_scraped_at_utc": datetime.now(timezone.utc).isoformat(), "settle_polls": polls}

    def stop(self) -> list[str]:
        """Stop contention, then samplers; return errors instead of raising."""
        errors = []
        if self.contention_process is not None:
            record = self.manifest["contention"]
            try:
                if self.contention_process.poll() is None:
                    os.killpg(self.contention_process.pid, signal.SIGTERM)
                    self.contention_process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"contention stop: {exc}")
                os.killpg(self.contention_process.pid, signal.SIGKILL)
            record["returncode"] = self.contention_process.returncode
            record["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
        for sidecar in self.sidecars:
            try:
                sidecar.stop()
            except OSError as exc:
                sidecar.record["error"] = str(exc)
                errors.append(f"{sidecar.name} stop: {exc}")
        return errors


def wait_until_healthy(config: dict, process: subprocess.Popen) -> None:
    server = config["server"]
    deadline = time.monotonic() + server.get("startup_timeout_seconds", 600)
    url = f"http://{server['host']}:{server['port']}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with status {process.returncode}; inspect server.log")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"server did not become healthy within the timeout: {url}")


def preflight(config: dict, root: Path) -> dict:
    """Check local prerequisites without starting vLLM or touching the GPU."""
    host, port = config["server"]["host"], config["server"]["port"]
    checks = {
        "vllm_executable": {"ok": shutil.which("vllm") is not None, "detail": shutil.which("vllm")},
        "nvidia_smi": {"ok": shutil.which("nvidia-smi") is not None, "detail": shutil.which("nvidia-smi")},
        "series_does_not_exist": {"ok": not root.exists(), "detail": str(root)},
    }
    parent = next((path for path in [root.parent, *root.parents] if path.exists()), None)
    checks["results_writable"] = {"ok": bool(parent and os.access(parent, os.W_OK)), "detail": str(parent)}
    try:
        with socket.socket() as probe:
            # SO_REUSEADDR, as servers use: TIME_WAIT sockets left by the previous series'
            # connections must not block, but a live listener still fails the bind.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
        checks["server_port_available"] = {"ok": True, "detail": f"{host}:{port}"}
    except OSError as exc:
        checks["server_port_available"] = {"ok": False, "detail": str(exc)}
    enabled = config.get("telemetry", {})
    if enabled.get("cpu_sampler"):
        checks["mpstat_executable"] = {"ok": shutil.which("mpstat") is not None,
                                       "detail": shutil.which("mpstat") or "apt install sysstat"}
    if enabled.get("gpu_sampler"):
        try:
            rows = query_nvidia([f"--query-gpu={telemetry.GPU_FIELDS}"])
            checks["nvidia_smi_query"] = {"ok": len(rows) == 1 and len(rows[0]) == 6, "detail": rows}
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            checks["nvidia_smi_query"] = {"ok": False, "detail": str(exc)}
    table = contention_settings(config)
    if table is not None:
        checks["torch_importable"] = {"ok": importlib.util.find_spec("torch") is not None, "detail": "torch"}
        try:
            total = float(query_nvidia(["--query-gpu=memory.total"])[0][0])
            headroom = (1 - config["server"]["gpu_memory_utilization"]) * total - table["memory_cap_mib"]
            checks["contention_vram_headroom"] = {"ok": headroom >= 256, "detail": f"{headroom:.0f} MiB"}
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
            checks["contention_vram_headroom"] = {"ok": False, "detail": str(exc)}
    return {"ok": all(check["ok"] for check in checks.values()), "checks": checks}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--series-id", help="stable label: letters, numbers, dot, underscore, or hyphen")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results" / "runs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the complete command plan")
    mode.add_argument("--preflight", action="store_true", help="check local prerequisites without running")
    args = parser.parse_args()
    config = load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    series_id = args.series_id or f"{config['experiment']['name']}-{stamp}"
    if not SERIES_ID_PATTERN.fullmatch(series_id):
        parser.error("--series-id must be 1-80 safe filename characters and start with a letter or number")
    root = args.results_dir / series_id
    plan = build_plan(config, root)
    plan_view = [{key: str(value) if isinstance(value, Path) else value for key, value in item.items()}
                 for item in plan]
    if args.dry_run:
        view = {"series_id": series_id, "server_command": server_command(config),
                "condition_order": [item["name"] for item in config["condition"]], "plan": plan_view}
        if sampler_specs(config, root):
            view["telemetry_commands"] = {name: {"command": command, "env": env, "path": str(path)}
                                          for name, command, env, path in sampler_specs(config, root)}
        if contention_command(config, root):
            view["contention_command"] = contention_command(config, root)
        print(json.dumps(view, indent=2))
        return 0

    checks = preflight(config, root)
    if args.preflight:
        print(json.dumps(checks, indent=2))
        return 0 if checks["ok"] else 1
    if not checks["ok"]:
        print(json.dumps(checks, indent=2), file=sys.stderr)
        raise SystemExit("preflight failed; correct the failed checks before running")

    root.mkdir(parents=True, exist_ok=False)
    (root / "config.toml").write_bytes(args.config.read_bytes())
    protocol_controls = {
        "prefix_caching": config["server"].get("enable_prefix_caching", "vllm_default"),
        "model_revision": config["server"].get("model_revision", "unresolved"),
        "tokenizer_revision": config["server"].get("tokenizer_revision", "unresolved"),
        "generation_config": config["server"].get("generation_config", "vllm_default"),
        "sampling": config.get("sampling", "vllm_benchmark_defaults"),
        "percentile_metrics": config["execution"].get("percentile_metrics", "vllm_default"),
        "warmup_concurrencies": config["execution"].get(
            "warmup_concurrencies", [config["execution"]["warmup_max_concurrency"]]),
    }
    if "request_id_prefix" in config["execution"]:
        protocol_controls["request_id_prefix"] = config["execution"]["request_id_prefix"]
    if "max_num_batched_tokens" in config["server"]:
        protocol_controls["max_num_batched_tokens"] = config["server"]["max_num_batched_tokens"]
    if "telemetry" in config:
        protocol_controls["telemetry"] = config["telemetry"]
    if "contention" in config:
        protocol_controls["contention"] = config["contention"]
    manifest = {"schema_version": 4, "series_id": series_id,
                "plan_name": config["experiment"]["name"], "status": "planned",
                "condition_order": [item["name"] for item in config["condition"]],
                "warmup_enabled": bool(config["execution"]["warmup_prompts"]),
                "protocol_controls": protocol_controls,
                "active_run": None, "config": config, "server_command": server_command(config),
                "environment": environment(REPO), "preflight": checks, "runs": []}
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    process = None
    instruments = Instrumentation(config, root, manifest)
    try:
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen(server_command(config), stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, text=True)
            wait_until_healthy(config, process)
            instruments.start()
            for index, item in enumerate(plan):
                instruments.check()
                manifest["status"] = "warming_up" if item["measurement_role"] == "warmup" else "running"
                manifest["active_run"] = {"condition": item["condition"], "repetition": item["repetition"]}
                write_json(manifest_path, manifest)
                item["directory"].mkdir(parents=True)
                scrape = instruments.before_run(item["directory"])
                started = datetime.now(timezone.utc).isoformat()
                completed = subprocess.run(item["command"], text=True, capture_output=True, check=False)
                finished = datetime.now(timezone.utc).isoformat()
                (item["directory"] / "benchmark.stdout").write_text(completed.stdout)
                (item["directory"] / "benchmark.stderr").write_text(completed.stderr)
                run_record = {"series_id": series_id, "plan_name": config["experiment"]["name"],
                    "measurement_role": item["measurement_role"], "condition": item["condition"],
                    "repetition": item["repetition"], "max_concurrency": item["max_concurrency"],
                    "requested_input_tokens": config["workload"]["input_tokens"],
                    "requested_output_tokens": config["workload"]["output_tokens"],
                    "protocol_controls": protocol_controls,
                    "command": item["command"], "started_at_utc": started,
                    "finished_at_utc": finished,
                    "returncode": completed.returncode, "raw_result": str(item["directory"] / "requests.json")}
                scrape_error = None
                if scrape is not None:
                    prompts = (config["execution"]["warmup_prompts"] if item["measurement_role"] == "warmup"
                               else config["workload"]["num_prompts"])
                    run_record["expected_server_requests"] = prompts
                    try:
                        run_record["server_metrics"] = instruments.after_run(item["directory"], scrape)
                    except Exception as exc:  # record the run before failing the series
                        run_record["server_metrics"], scrape_error = {**scrape, "error": str(exc)}, exc
                write_json(item["directory"] / "run-manifest.json", run_record)
                manifest["runs"].append(run_record)
                write_json(manifest_path, manifest)
                if scrape_error is not None:
                    raise scrape_error
                if completed.returncode:
                    raise RuntimeError(f"benchmark failed for {item['condition']} repetition {item['repetition']}")
                if index < len(plan) - 1 and config["execution"]["cooldown_seconds"]:
                    time.sleep(config["execution"]["cooldown_seconds"])
        manifest["status"] = "complete"
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        stop_errors = instruments.stop()
        if stop_errors:
            manifest["teardown_errors"] = stop_errors
        manifest["active_run"] = None
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main())
