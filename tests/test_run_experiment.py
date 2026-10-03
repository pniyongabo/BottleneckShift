import importlib.util
import json
from pathlib import Path
import sys

import pytest

from bottleneckshift.config import load_config

spec = importlib.util.spec_from_file_location("run_experiment", Path("scripts/run_experiment.py"))
run_experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment)

HISTORICAL_SERVER = ["vllm", "serve", "Qwen/Qwen2.5-0.5B-Instruct", "--host", "127.0.0.1",
                     "--port", "8000", "--tensor-parallel-size", "1",
                     "--gpu-memory-utilization", "0.9"]


def historical_benchmark(concurrency, directory, prompts):
    return ["vllm", "bench", "serve", "--backend", "vllm", "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--host", "127.0.0.1", "--port", "8000", "--dataset-name", "random",
            "--random-input-len", "256", "--random-output-len", "128", "--num-prompts", str(prompts),
            "--seed", "2027", "--request-rate", "inf", "--max-concurrency", str(concurrency),
            "--save-result", "--save-detailed", "--result-dir", str(directory),
            "--result-filename", "requests.json"]


def controlled_config():
    config = load_config(Path("configs/milestone1-forward.toml"))
    config["execution"]["warmup_concurrencies"] = [1, 8]
    config["execution"]["percentile_metrics"] = ["ttft", "tpot", "itl", "e2el"]
    config["server"].update({"model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
                             "enable_prefix_caching": False, "generation_config": "vllm"})
    config["sampling"] = {"temperature": 0.0, "top_p": 1.0}
    return config


def test_historical_configs_keep_their_original_command_plan(tmp_path):
    for name, order in (("forward", ["baseline_c1", "perturbation_c8"]),
                        ("reverse", ["perturbation_c8", "baseline_c1"])):
        config = load_config(Path(f"configs/milestone1-{name}.toml"))
        assert run_experiment.server_command(config) == HISTORICAL_SERVER
        plan = run_experiment.build_plan(config, tmp_path)
        warmup, measured = plan[0], plan[1:]
        assert (warmup["condition"], warmup["directory"]) == ("warmup", tmp_path / "warmup")
        assert warmup["command"] == historical_benchmark(1, tmp_path / "warmup", 10)
        assert [item["condition"] for item in measured] == [order[0]] * 3 + [order[1]] * 3
        for item in measured:
            assert item["command"] == historical_benchmark(item["max_concurrency"], item["directory"], 100)


def test_protocol_controls_become_vllm_flags(tmp_path):
    config = controlled_config()
    assert run_experiment.server_command(config) == HISTORICAL_SERVER + [
        "--revision", "a" * 40, "--tokenizer-revision", "b" * 40,
        "--generation-config", "vllm", "--no-enable-prefix-caching"]
    config["server"]["enable_prefix_caching"] = True
    assert run_experiment.server_command(config)[-1] == "--enable-prefix-caching"
    command = run_experiment.benchmark_command(config, 8, tmp_path)
    assert command[-6:] == ["--temperature", "0.0", "--top-p", "1.0",
                            "--percentile-metrics", "ttft,tpot,itl,e2el"]


def test_warmup_concurrencies_create_labelled_warmups(tmp_path):
    plan = run_experiment.build_plan(controlled_config(), tmp_path)
    warmups = [item for item in plan if item["measurement_role"] == "warmup"]
    assert [(item["condition"], item["max_concurrency"], item["directory"]) for item in warmups] == [
        ("warmup_c1", 1, tmp_path / "warmup" / "c1"), ("warmup_c8", 8, tmp_path / "warmup" / "c8")]
    assert plan[:2] == warmups
    assert all(item["repetition"] is None for item in warmups)


def test_protocol_controls_are_recorded_in_series_and_run_manifests(tmp_path, monkeypatch):
    config_path = tmp_path / "controlled.toml"
    source = Path("configs/milestone1-forward.toml").read_text()
    source = source.replace("repetitions = 3", "repetitions = 1")
    source = source.replace("warmup_max_concurrency = 1", "warmup_max_concurrency = 1\nwarmup_concurrencies = [1, 8]")
    source = source.replace("cooldown_seconds = 15", "cooldown_seconds = 0")
    source = source.replace('model = "Qwen/Qwen2.5-0.5B-Instruct"',
                            'model = "Qwen/Qwen2.5-0.5B-Instruct"\nmodel_revision = "' + "a" * 40 + '"\n'
                            'tokenizer_revision = "' + "b" * 40 + '"\nenable_prefix_caching = false')
    config_path.write_text(source + "\n[sampling]\ntemperature = 0.0\ntop_p = 1.0\n")

    class FakeServer:
        pid = 0
        returncode = 0

        def poll(self):
            return 0

    class Completed:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(run_experiment.subprocess, "Popen", lambda *args, **kwargs: FakeServer())
    monkeypatch.setattr(run_experiment.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(run_experiment, "wait_until_healthy", lambda config, process: None)
    monkeypatch.setattr(run_experiment, "preflight", lambda config, root: {"ok": True, "checks": {}})
    monkeypatch.setattr(run_experiment, "environment", lambda repo: {"synthetic": True})
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(config_path),
                                      "--series-id", "synthetic", "--results-dir", str(tmp_path / "runs")])
    assert run_experiment.main() == 0

    root = tmp_path / "runs" / "synthetic"
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {"prefix_caching": False, "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
                "generation_config": "vllm_default", "percentile_metrics": "vllm_default", "sampling": {"temperature": 0.0, "top_p": 1.0},
                "warmup_concurrencies": [1, 8]}
    assert manifest["schema_version"] == 4 and manifest["status"] == "complete"
    assert manifest["protocol_controls"] == expected
    assert [run["condition"] for run in manifest["runs"]] == [
        "warmup_c1", "warmup_c8", "baseline_c1", "perturbation_c8"]
    for run in manifest["runs"]:
        assert run["protocol_controls"] == expected
        run_manifest = json.loads((Path(run["raw_result"]).parent / "run-manifest.json").read_text())
        assert run_manifest["protocol_controls"] == expected


from synthetic import run_series, scrape, small_config  # noqa: E402

PHASE_B = [f"milestone3-{regime}-{order}" for regime in ("prefill_heavy", "decode_heavy")
           for order in ("forward", "reverse")]


def test_controlled_configs_keep_their_milestone_2_command_plan(tmp_path):
    for order in ("forward", "reverse"):
        config = load_config(Path(f"configs/milestone1-controlled-{order}.toml"))
        plan = run_experiment.build_plan(config, tmp_path / "series")
        for item in plan:
            assert "--request-id-prefix" not in item["command"]
            assert item["command"][-6:] == ["--temperature", "0.0", "--top-p", "1.0",
                                            "--percentile-metrics", "ttft,tpot,itl,e2el"]
        assert run_experiment.sampler_specs(config, tmp_path) == []
        assert run_experiment.contention_command(config, tmp_path) is None


def test_phase_b_commands_add_only_the_request_id_prefix(tmp_path):
    controlled = load_config(Path("configs/milestone1-controlled-forward.toml"))
    for name in PHASE_B:
        config = load_config(Path(f"configs/{name}.toml"))
        assert run_experiment.server_command(config) == run_experiment.server_command(controlled)
        root = tmp_path / f"{name}-01"
        for item in run_experiment.build_plan(config, root):
            command = item["command"]
            suffix = f"rep-{item['repetition']:02d}-" if item["repetition"] else ""
            assert command[-2:] == ["--request-id-prefix", f"{root.name}-{item['condition']}-{suffix}"]
            workload = config["workload"]
            prompts = 10 if item["measurement_role"] == "warmup" else workload["num_prompts"]
            assert command[command.index("--num-prompts") + 1] == str(prompts)
            assert command[command.index("--random-input-len") + 1] == str(workload["input_tokens"])
        assert [spec[0] for spec in run_experiment.sampler_specs(config, root)] == ["gpu", "cpu"]


def test_instrumented_run_records_scrapes_telemetry_and_contention(tmp_path, monkeypatch):
    source = small_config(Path("configs/milestone3-prefill_heavy-forward.toml").read_text(),
                          contention='\n[contention]\nmode = "active"\nduty_cycle = 0.5\nmemory_cap_mib = 1024\n')
    root = run_series(tmp_path, monkeypatch, run_experiment, source)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["schema_version"] == 4
    assert manifest["protocol_controls"]["telemetry"] == {"server_metrics": True, "gpu_sampler": True,
                                                         "cpu_sampler": True}
    assert manifest["protocol_controls"]["request_id_prefix"] is True
    assert (root / "metrics-initial.prom").is_file() and (root / "gpu.csv").is_file()
    assert manifest["telemetry"]["cpu"]["stop_signal"] == "SIGINT"
    assert manifest["contention"]["vram_mib"] == 700 and manifest["contention"]["returncode"] == 0
    assert [run["condition"] for run in manifest["runs"]] == [
        "warmup_c1", "warmup_c8", "baseline_c1", "perturbation_c8"]
    for run in manifest["runs"]:
        directory = Path(run["raw_result"]).parent
        assert (directory / "metrics-before.prom").is_file() and (directory / "metrics-after.prom").is_file()
        assert run["expected_server_requests"] == (2 if run["measurement_role"] == "warmup" else 4)
        assert run["server_metrics"]["settle_polls"] >= 2


def test_failed_after_scrape_is_recorded_and_fails_the_series(tmp_path, monkeypatch):
    source = small_config(Path("configs/milestone3-decode_heavy-forward.toml").read_text())
    real_settled = run_experiment.settled_metrics
    calls = []

    def flaky(url):
        calls.append(url)
        if len(calls) == 2:
            raise RuntimeError("server did not settle")
        return real_settled(url)

    monkeypatch.setattr(run_experiment, "settled_metrics", flaky)
    with pytest.raises(RuntimeError, match="did not settle"):
        run_series(tmp_path, monkeypatch, run_experiment, source)
    manifest = json.loads((tmp_path / "runs" / "synthetic" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["runs"][-1]["server_metrics"]["error"] == "server did not settle"
    assert manifest["telemetry"]["gpu"]["returncode"] == 0


def test_missing_metric_family_fails_before_any_warmup(tmp_path, monkeypatch):
    source = small_config(Path("configs/milestone3-decode_heavy-forward.toml").read_text())
    monkeypatch.setattr(run_experiment.prometheus, "REQUIRED_FAMILIES",
                        run_experiment.prometheus.REQUIRED_FAMILIES + ("vllm:not_exposed_seconds",))
    with pytest.raises(ValueError, match="vllm:not_exposed_seconds"):
        run_series(tmp_path, monkeypatch, run_experiment, source)
    manifest = json.loads((tmp_path / "runs" / "synthetic" / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["runs"] == []


def test_preflight_port_check_fails_only_for_a_live_listener(tmp_path):
    import socket
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        config = {"server": {"host": "127.0.0.1", "port": port}}
        assert not run_experiment.preflight(config, tmp_path / "s")["checks"]["server_port_available"]["ok"]
    # A closed port with a just-finished connection (TIME_WAIT) is available again.
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        connection, _ = server.accept()
        connection.close()  # server side closes first and enters TIME_WAIT
        client.close()
    config = {"server": {"host": "127.0.0.1", "port": port}}
    assert run_experiment.preflight(config, tmp_path / "s")["checks"]["server_port_available"]["ok"]
