from pathlib import Path

import pytest

from bottleneckshift.config import load_config


def test_milestone_config_is_valid():
    config = load_config(Path("configs/milestone1.toml"))
    assert config["experiment"]["repetitions"] == 3
    assert [item["max_concurrency"] for item in config["condition"]] == [1, 8]


def test_condition_names_must_be_unique(tmp_path):
    path = tmp_path / "bad.toml"
    source = Path("configs/milestone1.toml").read_text()
    path.write_text(source.replace('name = "perturbation_c8"', 'name = "baseline_c1"'))
    with pytest.raises(ValueError, match="unique"):
        load_config(path)
