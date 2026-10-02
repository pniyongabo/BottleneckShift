import importlib.util
import json
from pathlib import Path
import sys

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
    assert command[-4:] == ["--temperature", "0.0", "--top-p", "1.0"]


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
                "generation_config": "vllm_default", "sampling": {"temperature": 0.0, "top_p": 1.0},
                "warmup_concurrencies": [1, 8]}
    assert manifest["schema_version"] == 3 and manifest["status"] == "complete"
    assert manifest["protocol_controls"] == expected
    assert [run["condition"] for run in manifest["runs"]] == [
        "warmup_c1", "warmup_c8", "baseline_c1", "perturbation_c8"]
    for run in manifest["runs"]:
        assert run["protocol_controls"] == expected
        run_manifest = json.loads((Path(run["raw_result"]).parent / "run-manifest.json").read_text())
        assert run_manifest["protocol_controls"] == expected
