from pathlib import Path

import pytest

from bottleneckshift.config import load_config


def test_milestone_config_is_valid():
    config = load_config(Path("configs/milestone1-forward.toml"))
    assert config["experiment"]["repetitions"] == 3
    assert [item["max_concurrency"] for item in config["condition"]] == [1, 8]


def test_reverse_plan_changes_only_name_and_order():
    forward = load_config(Path("configs/milestone1-forward.toml"))
    reverse = load_config(Path("configs/milestone1-reverse.toml"))
    assert forward["server"] == reverse["server"]
    assert forward["workload"] == reverse["workload"]
    assert forward["execution"] == reverse["execution"]
    assert list(reversed(forward["condition"])) == reverse["condition"]


def test_condition_names_must_be_unique(tmp_path):
    path = tmp_path / "bad.toml"
    source = Path("configs/milestone1-forward.toml").read_text()
    path.write_text(source.replace('name = "perturbation_c8"', 'name = "baseline_c1"'))
    with pytest.raises(ValueError, match="unique"):
        load_config(path)


def test_explicit_followup_controls_are_validated(tmp_path):
    path = tmp_path / "controlled.toml"
    source = Path("configs/milestone1-forward.toml").read_text()
    source = source.replace("warmup_max_concurrency = 1", "warmup_max_concurrency = 1\nwarmup_concurrencies = [1, 8]")
    source = source.replace('model = "Qwen/Qwen2.5-0.5B-Instruct"',
                            'model = "Qwen/Qwen2.5-0.5B-Instruct"\n'
                            'model_revision = "immutable-commit"\n'
                            'tokenizer_revision = "immutable-commit"\n'
                            'enable_prefix_caching = false')
    source += "\n[sampling]\ntemperature = 0.0\ntop_p = 1.0\n"
    path.write_text(source)
    config = load_config(path)
    assert config["execution"]["warmup_concurrencies"] == [1, 8]
    assert config["server"]["enable_prefix_caching"] is False


def test_controlled_followup_configs_pin_controls_and_differ_only_in_order():
    forward = load_config(Path("configs/milestone1-controlled-forward.toml"))
    reverse = load_config(Path("configs/milestone1-controlled-reverse.toml"))
    for key in ("server", "workload", "execution", "sampling"):
        assert forward[key] == reverse[key]
    assert list(reversed(forward["condition"])) == reverse["condition"]
    server = forward["server"]
    for key in ("model_revision", "tokenizer_revision"):
        assert len(server[key]) == 40 and all(char in "0123456789abcdef" for char in server[key])
    assert server["enable_prefix_caching"] is False
    assert server["generation_config"] == "vllm"
    assert forward["execution"]["warmup_concurrencies"] == [1, 8]
    assert forward["sampling"] == {"temperature": 0.0, "top_p": 1.0}
    assert forward["execution"]["percentile_metrics"] == ["ttft", "tpot", "itl", "e2el"]
    assert forward["experiment"]["name"] != load_config(Path("configs/milestone1-forward.toml"))["experiment"]["name"]


@pytest.mark.parametrize("value, message", [([], "must list"), (["ttft", "p99"], "must list"),
                                            (["e2el", "e2el"], "unique")])
def test_percentile_metrics_are_validated(tmp_path, value, message):
    path = tmp_path / "bad.toml"
    source = Path("configs/milestone1-forward.toml").read_text()
    path.write_text(source.replace("cooldown_seconds = 15", f"cooldown_seconds = 15\npercentile_metrics = {value!r}".replace("'", '"')))
    with pytest.raises(ValueError, match=message):
        load_config(path)


PHASE_B = {"prefill_heavy": (2048, 32, 100), "decode_heavy": (64, 512, 50)}


def test_phase_b_configs_match_the_protocol_and_milestone_2_controls():
    controlled = load_config(Path("configs/milestone1-controlled-forward.toml"))
    for regime, (inputs, outputs, prompts) in PHASE_B.items():
        forward = load_config(Path(f"configs/milestone3-{regime}-forward.toml"))
        reverse = load_config(Path(f"configs/milestone3-{regime}-reverse.toml"))
        for key in ("server", "workload", "execution", "sampling", "telemetry"):
            assert forward[key] == reverse[key]
        assert list(reversed(forward["condition"])) == reverse["condition"] == list(reversed(controlled["condition"]))
        assert forward["experiment"] == {"name": f"milestone3-{regime}-forward", "repetitions": 3}
        assert forward["server"] == controlled["server"] and forward["sampling"] == controlled["sampling"]
        assert forward["execution"] == {**controlled["execution"], "request_id_prefix": True}
        assert forward["workload"] == {"num_prompts": prompts, "input_tokens": inputs,
                                       "output_tokens": outputs, "seed": 2027}
        assert forward["telemetry"] == {"server_metrics": True, "gpu_sampler": True, "cpu_sampler": True}
        assert "contention" not in forward


@pytest.mark.parametrize("extra, message", [
    ('\n[telemetry]\ngpu_sample = true\n', "unknown \\[telemetry\\]"),
    ('\n[telemetry]\ngpu_sampler = 1\n', "booleans"),
    ('\n[telemetry]\ngpu_sampler = true\n[contention]\nmode = "loud"\n', "contention.mode"),
    ('\n[telemetry]\ngpu_sampler = true\n[contention]\nmode = "active"\nmemory_cap_mib = 1024\n', "duty_cycle"),
    ('\n[telemetry]\ngpu_sampler = true\n[contention]\nmode = "sham"\nduty_cycle = 0.5\nmemory_cap_mib = 1024\n',
     "not allowed in sham"),
    ('\n[telemetry]\ngpu_sampler = true\n[contention]\nmode = "sham"\nmemory_cap_mib = 2048\n', "memory_cap_mib"),
    ('\n[telemetry]\ngpu_sampler = true\n[contention]\nmode = "sham"\nmemory_cap_mib = 600\nmatrix_size = 8192\n',
     "does not fit"),
    ('\n[contention]\nmode = "sham"\nmemory_cap_mib = 1024\n', "requires telemetry.gpu_sampler"),
    ('\n[contention]\nmode = "off"\nduty_cycle = 0.5\n', "takes no other settings"),
])
def test_instrumentation_settings_are_validated(tmp_path, extra, message):
    path = tmp_path / "bad.toml"
    path.write_text(Path("configs/milestone1-forward.toml").read_text() + extra)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_request_id_prefix_must_be_boolean(tmp_path):
    path = tmp_path / "bad.toml"
    source = Path("configs/milestone1-forward.toml").read_text()
    path.write_text(source.replace("cooldown_seconds = 15", 'cooldown_seconds = 15\nrequest_id_prefix = "yes"'))
    with pytest.raises(ValueError, match="request_id_prefix"):
        load_config(path)


def test_phase_c_configs_match_the_addendum():
    base = load_config(Path("configs/milestone3-prefill_heavy-forward.toml"))
    forward = load_config(Path("configs/milestone3-crossover-forward.toml"))
    reverse = load_config(Path("configs/milestone3-crossover-reverse.toml"))
    for config in (forward, reverse):
        for key in ("server", "workload", "sampling", "telemetry"):
            assert config[key] == base[key]
        assert config["execution"] == {**base["execution"], "warmup_prompts": 32,
                                       "warmup_concurrencies": [8, 16, 32]}
        assert "contention" not in config
    assert [(c["name"], c["max_concurrency"]) for c in forward["condition"]] == [
        ("anchor_c8", 8), ("probe_c16", 16), ("probe_c32", 32)]
    assert list(reversed(forward["condition"])) == reverse["condition"]
    expected = {"sham": {"mode": "sham", "memory_cap_mib": 1024},
                "active-025": {"mode": "active", "duty_cycle": 0.25, "memory_cap_mib": 1024},
                "active-050": {"mode": "active", "duty_cycle": 0.5, "memory_cap_mib": 1024}}
    for label, table in expected.items():
        config = load_config(Path(f"configs/milestone3-contention-{label}.toml"))
        assert config["contention"] == table
        assert config["experiment"]["name"] == f"milestone3-contention-{label}-prefill_heavy-forward"
        for key in ("server", "workload", "sampling", "telemetry", "execution", "condition"):
            assert config[key] == base[key]
