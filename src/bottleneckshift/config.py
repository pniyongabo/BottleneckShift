"""Load and validate the intentionally small experiment configuration."""

from pathlib import Path
import tomllib


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    required = {
        "experiment": ("name", "repetitions"),
        "server": ("model", "host", "port", "tensor_parallel_size", "gpu_memory_utilization"),
        "workload": ("num_prompts", "input_tokens", "output_tokens", "seed"),
    }
    for section, keys in required.items():
        if section not in config:
            raise ValueError(f"missing [{section}] section")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"missing {section}.{key}")
    conditions = config.get("condition", [])
    if not conditions:
        raise ValueError("at least one [[condition]] is required")
    if config["experiment"]["repetitions"] < 1:
        raise ValueError("experiment.repetitions must be positive")
    names = [item.get("name") for item in conditions]
    if None in names or len(names) != len(set(names)):
        raise ValueError("condition names must be present and unique")
    for item in conditions:
        if item.get("max_concurrency", 0) < 1:
            raise ValueError("condition max_concurrency must be positive")
    return config
