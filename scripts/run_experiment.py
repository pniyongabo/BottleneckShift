#!/usr/bin/env python3
"""Run vLLM's maintained serving benchmark and preserve its raw output."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bottleneckshift.config import load_config  # noqa: E402
from bottleneckshift.manifest import environment  # noqa: E402


def server_command(config: dict) -> list[str]:
    server = config["server"]
    return ["vllm", "serve", server["model"], "--host", server["host"], "--port", str(server["port"]),
            "--tensor-parallel-size", str(server["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(server["gpu_memory_utilization"])]


def benchmark_command(config: dict, condition: dict, output_dir: Path) -> list[str]:
    server, workload = config["server"], config["workload"]
    return ["vllm", "bench", "serve", "--backend", "vllm", "--model", server["model"],
            "--host", server["host"], "--port", str(server["port"]), "--dataset-name", "random",
            "--random-input-len", str(workload["input_tokens"]), "--random-output-len", str(workload["output_tokens"]),
            "--num-prompts", str(workload["num_prompts"]), "--seed", str(workload["seed"]),
            "--request-rate", "inf", "--max-concurrency", str(condition["max_concurrency"]),
            "--save-result", "--save-detailed", "--result-dir", str(output_dir),
            "--result-filename", "requests.json"]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.results_dir / f"{config['experiment']['name']}_{stamp}"
    planned = []
    for condition in config["condition"]:
        for repetition in range(1, config["experiment"]["repetitions"] + 1):
            directory = root / condition["name"] / f"rep-{repetition:02d}"
            planned.append((condition, repetition, directory, benchmark_command(config, condition, directory)))
    if args.dry_run:
        print(json.dumps({"server_command": server_command(config),
                          "benchmark_commands": [command for _, _, _, command in planned]}, indent=2))
        return 0

    root.mkdir(parents=True, exist_ok=False)
    (root / "config.toml").write_bytes(args.config.read_bytes())
    manifest = {"schema_version": 1, "status": "running", "config": config,
                "server_command": server_command(config), "environment": environment(REPO), "runs": []}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(server_command(config), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, text=True)
        try:
            wait_until_healthy(config, process)
            for condition, repetition, directory, command in planned:
                directory.mkdir(parents=True)
                started = datetime.now(timezone.utc).isoformat()
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                (directory / "benchmark.stdout").write_text(completed.stdout)
                (directory / "benchmark.stderr").write_text(completed.stderr)
                run_record = {"condition": condition["name"], "repetition": repetition,
                    "command": command, "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": completed.returncode, "raw_result": str(directory / "requests.json")}
                (directory / "run-manifest.json").write_text(json.dumps(run_record, indent=2) + "\n")
                manifest["runs"].append(run_record)
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                if completed.returncode:
                    raise RuntimeError(f"benchmark failed for {condition['name']} repetition {repetition}")
            manifest["status"] = "complete"
            return 0
        finally:
            if manifest["status"] != "complete":
                manifest["status"] = "failed_or_interrupted"
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(main())
