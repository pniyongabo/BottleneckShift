#!/usr/bin/env python3
"""Run a reviewed vLLM experiment plan and preserve its raw output."""

import argparse
from datetime import datetime, timezone
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

from bottleneckshift.config import load_config  # noqa: E402
from bottleneckshift.manifest import environment  # noqa: E402

SERIES_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def server_command(config: dict) -> list[str]:
    server = config["server"]
    return ["vllm", "serve", server["model"], "--host", server["host"], "--port", str(server["port"]),
            "--tensor-parallel-size", str(server["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(server["gpu_memory_utilization"])]


def benchmark_command(config: dict, max_concurrency: int, output_dir: Path,
                      num_prompts: int | None = None) -> list[str]:
    server, workload = config["server"], config["workload"]
    return ["vllm", "bench", "serve", "--backend", "vllm", "--model", server["model"],
            "--host", server["host"], "--port", str(server["port"]), "--dataset-name", "random",
            "--random-input-len", str(workload["input_tokens"]), "--random-output-len", str(workload["output_tokens"]),
            "--num-prompts", str(num_prompts if num_prompts is not None else workload["num_prompts"]),
            "--seed", str(workload["seed"]), "--request-rate", "inf",
            "--max-concurrency", str(max_concurrency), "--save-result", "--save-detailed",
            "--result-dir", str(output_dir), "--result-filename", "requests.json"]


def build_plan(config: dict, root: Path) -> list[dict]:
    """Return the complete, ordered warm-up and measurement plan."""
    plan = []
    execution = config["execution"]
    if execution["warmup_prompts"]:
        directory = root / "warmup"
        plan.append({"measurement_role": "warmup", "condition": "warmup", "repetition": None,
                     "max_concurrency": execution["warmup_max_concurrency"], "directory": directory,
                     "command": benchmark_command(config, execution["warmup_max_concurrency"], directory,
                                                  execution["warmup_prompts"])})
    for condition in config["condition"]:
        for repetition in range(1, config["experiment"]["repetitions"] + 1):
            directory = root / condition["name"] / f"rep-{repetition:02d}"
            plan.append({"measurement_role": "measured", "condition": condition["name"],
                         "repetition": repetition, "max_concurrency": condition["max_concurrency"],
                         "directory": directory,
                         "command": benchmark_command(config, condition["max_concurrency"], directory)})
    return plan


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
            probe.bind((host, port))
        checks["server_port_available"] = {"ok": True, "detail": f"{host}:{port}"}
    except OSError as exc:
        checks["server_port_available"] = {"ok": False, "detail": str(exc)}
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
        print(json.dumps({"series_id": series_id, "server_command": server_command(config),
                          "condition_order": [item["name"] for item in config["condition"]],
                          "plan": plan_view}, indent=2))
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
    manifest = {"schema_version": 2, "series_id": series_id,
                "plan_name": config["experiment"]["name"], "status": "planned",
                "condition_order": [item["name"] for item in config["condition"]],
                "warmup_enabled": bool(config["execution"]["warmup_prompts"]),
                "active_run": None, "config": config, "server_command": server_command(config),
                "environment": environment(REPO), "preflight": checks, "runs": []}
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    process = None
    try:
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen(server_command(config), stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, text=True)
            wait_until_healthy(config, process)
            for index, item in enumerate(plan):
                manifest["status"] = "warming_up" if item["measurement_role"] == "warmup" else "running"
                manifest["active_run"] = {"condition": item["condition"], "repetition": item["repetition"]}
                write_json(manifest_path, manifest)
                item["directory"].mkdir(parents=True)
                started = datetime.now(timezone.utc).isoformat()
                completed = subprocess.run(item["command"], text=True, capture_output=True, check=False)
                (item["directory"] / "benchmark.stdout").write_text(completed.stdout)
                (item["directory"] / "benchmark.stderr").write_text(completed.stderr)
                run_record = {"series_id": series_id, "plan_name": config["experiment"]["name"],
                    "measurement_role": item["measurement_role"], "condition": item["condition"],
                    "repetition": item["repetition"], "max_concurrency": item["max_concurrency"],
                    "requested_input_tokens": config["workload"]["input_tokens"],
                    "requested_output_tokens": config["workload"]["output_tokens"],
                    "command": item["command"], "started_at_utc": started,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": completed.returncode, "raw_result": str(item["directory"] / "requests.json")}
                write_json(item["directory"] / "run-manifest.json", run_record)
                manifest["runs"].append(run_record)
                write_json(manifest_path, manifest)
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
