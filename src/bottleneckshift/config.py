"""Load and validate the intentionally small experiment configuration."""

from pathlib import Path
import tomllib

from bottleneckshift import contention

TELEMETRY_KEYS = ("server_metrics", "gpu_sampler", "cpu_sampler")
CONTENTION_KEYS = ("mode", "duty_cycle", "period_ms", "memory_cap_mib", "matrix_size",
                   "ready_timeout_seconds")
MAX_CONTENTION_MIB = 1024


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    required = {
        "experiment": ("name", "repetitions"),
        "execution": ("warmup_prompts", "warmup_max_concurrency", "cooldown_seconds"),
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
    if config["execution"]["warmup_prompts"] < 0:
        raise ValueError("execution.warmup_prompts cannot be negative")
    if config["execution"]["warmup_max_concurrency"] < 1:
        raise ValueError("execution.warmup_max_concurrency must be positive")
    if config["execution"]["cooldown_seconds"] < 0:
        raise ValueError("execution.cooldown_seconds cannot be negative")
    names = [item.get("name") for item in conditions]
    if None in names or len(names) != len(set(names)):
        raise ValueError("condition names must be present and unique")
    for item in conditions:
        if item.get("max_concurrency", 0) < 1:
            raise ValueError("condition max_concurrency must be positive")
    warmup_concurrencies = config["execution"].get("warmup_concurrencies")
    if warmup_concurrencies is not None:
        if not warmup_concurrencies or any(value < 1 for value in warmup_concurrencies):
            raise ValueError("execution.warmup_concurrencies must contain positive values")
        if len(warmup_concurrencies) != len(set(warmup_concurrencies)):
            raise ValueError("execution.warmup_concurrencies must be unique")
    percentile_metrics = config["execution"].get("percentile_metrics")
    if percentile_metrics is not None:
        if not percentile_metrics or not set(percentile_metrics) <= {"ttft", "tpot", "itl", "e2el"}:
            raise ValueError("execution.percentile_metrics must list ttft, tpot, itl, or e2el")
        if len(percentile_metrics) != len(set(percentile_metrics)):
            raise ValueError("execution.percentile_metrics must be unique")
    server = config["server"]
    for key in ("model_revision", "tokenizer_revision"):
        if key in server and not str(server[key]).strip():
            raise ValueError(f"server.{key} cannot be empty")
    if "generation_config" in server and not str(server["generation_config"]).strip():
        raise ValueError("server.generation_config cannot be empty")
    if "enable_prefix_caching" in server and not isinstance(server["enable_prefix_caching"], bool):
        raise ValueError("server.enable_prefix_caching must be a boolean")
    if not isinstance(config["execution"].get("request_id_prefix", False), bool):
        raise ValueError("execution.request_id_prefix must be a boolean")
    telemetry = config.get("telemetry", {})
    unknown = set(telemetry) - set(TELEMETRY_KEYS)
    if unknown:
        raise ValueError(f"unknown [telemetry] keys: {sorted(unknown)}")
    if any(not isinstance(value, bool) for value in telemetry.values()):
        raise ValueError("[telemetry] values must be booleans")
    _validate_contention(config.get("contention"), telemetry)
    sampling = config.get("sampling", {})
    if sampling.get("temperature", 0) < 0:
        raise ValueError("sampling.temperature cannot be negative")
    if "top_p" in sampling and not 0 < sampling["top_p"] <= 1:
        raise ValueError("sampling.top_p must be in (0, 1]")
    return config


def _validate_contention(table: dict | None, telemetry: dict) -> None:
    if table is None:
        return
    unknown = set(table) - set(CONTENTION_KEYS)
    if unknown:
        raise ValueError(f"unknown [contention] keys: {sorted(unknown)}")
    mode = table.get("mode")
    if mode not in contention.MODES:
        raise ValueError(f"contention.mode must be one of {contention.MODES}")
    if mode == "off":
        if set(table) - {"mode"}:
            raise ValueError("contention mode 'off' takes no other settings")
        return
    if mode == "active":
        duty = table.get("duty_cycle")
        if not isinstance(duty, (int, float)) or not 0 < duty <= 1:
            raise ValueError("contention.duty_cycle must be in (0, 1] for active mode")
    elif "duty_cycle" in table:
        raise ValueError("contention.duty_cycle is not allowed in sham mode")
    if not 10 <= table.get("period_ms", 100) <= 1000:
        raise ValueError("contention.period_ms must be in [10, 1000]")
    cap = table.get("memory_cap_mib")
    if not isinstance(cap, int) or not contention.CONTEXT_RESERVE_MIB < cap <= MAX_CONTENTION_MIB:
        raise ValueError(f"contention.memory_cap_mib must be an integer in "
                         f"({contention.CONTEXT_RESERVE_MIB}, {MAX_CONTENTION_MIB}]")
    size = table.get("matrix_size", contention.DEFAULT_MATRIX_SIZE)
    if contention.buffer_bytes(size) > contention.tensor_budget_bytes(cap):
        raise ValueError("contention.matrix_size does not fit within memory_cap_mib")
    if table.get("ready_timeout_seconds", 120) <= 0:
        raise ValueError("contention.ready_timeout_seconds must be positive")
    if not telemetry.get("gpu_sampler"):
        raise ValueError("contention requires telemetry.gpu_sampler = true")
