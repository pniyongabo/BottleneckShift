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
