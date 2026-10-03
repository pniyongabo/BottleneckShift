"""Strict, schema-light validation for retained vLLM benchmark series."""

from datetime import datetime
import json
from pathlib import Path

from bottleneckshift import contention, prometheus, telemetry


def _load(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc


def validate_result(path: Path, expected_requests: int, input_tokens: int,
                    output_tokens: int) -> dict:
    """Validate request counts, failures, and configured token lengths."""
    document = _load(path)
    arrays = {}
    for key in ("ttfts", "itls", "input_lens", "output_lens", "errors"):
        value = document.get(key)
        if not isinstance(value, list):
            raise ValueError(f"{path}: missing request array {key}")
        if len(value) != expected_requests:
            raise ValueError(
                f"{path}: {key} has {len(value)} entries; expected {expected_requests}"
            )
        arrays[key] = value
    failures = [index for index, error in enumerate(arrays["errors"]) if error]
    if failures:
        raise ValueError(f"{path}: non-empty errors at request indexes {failures}")
    for key, expected in (("input_lens", input_tokens), ("output_lens", output_tokens)):
        mismatches = [index for index, value in enumerate(arrays[key]) if value != expected]
        if mismatches:
            raise ValueError(f"{path}: {key} differs from {expected} at indexes {mismatches}")
    return {"path": str(path), "requests": expected_requests, "failures": 0,
            "input_tokens": input_tokens, "output_tokens": output_tokens}


def validate_series(root: Path) -> dict:
    """Validate a complete series using its manifest as the source of expectations."""
    manifest = _load(root / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"{root}: manifest status is not complete")
    config = manifest.get("config", {})
    workload = config.get("workload", {})
    expected_runs = len(config.get("condition", [])) * config.get("experiment", {}).get(
        "repetitions", 0
    )
    measured = [run for run in manifest.get("runs", []) if run.get("measurement_role") == "measured"]
    if len(measured) != expected_runs:
        raise ValueError(f"{root}: found {len(measured)} measured runs; expected {expected_runs}")
    validated = []
    identities = set()
    for run in measured:
        identity = (run.get("condition"), run.get("repetition"))
        if identity in identities:
            raise ValueError(f"{root}: duplicate measured run {identity}")
        identities.add(identity)
        if run.get("returncode") != 0:
            raise ValueError(f"{root}: run {identity} has nonzero return code")
        raw_path = Path(run["raw_result"])
        if not raw_path.is_absolute():
            raw_path = root / raw_path
        # Older manifests recorded a path relative to the invocation directory.
        if not raw_path.is_file():
            raw_path = root / run["condition"] / f"rep-{run['repetition']:02d}" / "requests.json"
        validated.append(validate_result(raw_path, workload["num_prompts"],
                                         run["requested_input_tokens"],
                                         run["requested_output_tokens"]))
        run["_directory"] = raw_path.parent
    summary = {"series_id": manifest.get("series_id"), "measured_runs": len(validated),
               "measured_requests": sum(item["requests"] for item in validated), "runs": validated}
    if manifest.get("schema_version", 0) >= 4:
        summary.update(_validate_schema4(root, manifest, measured))
    for run in measured:
        run.pop("_directory", None)
    return summary


def _locate(recorded: str, directory: Path) -> Path:
    """A recorded artifact path, or the same file name in `directory` for moved archives."""
    path = Path(recorded)
    return path if path.is_file() else directory / path.name


def _validate_schema4(root: Path, manifest: dict, measured: list[dict]) -> dict:
    """Checks for manifests written by the instrumented runner (schema 4)."""
    config = manifest["config"]
    execution, enabled = config["execution"], config.get("telemetry", {})
    warmups = [run for run in manifest["runs"] if run.get("measurement_role") == "warmup"]
    expected_warmups = (len(execution.get("warmup_concurrencies", [execution["warmup_max_concurrency"]]))
                        if execution["warmup_prompts"] else 0)
    if len(warmups) != expected_warmups:
        raise ValueError(f"{root}: found {len(warmups)} warm-up runs; expected {expected_warmups}")
    for run in warmups:
        if run.get("returncode") != 0:
            raise ValueError(f"{root}: warm-up {run['condition']} has nonzero return code")
        directory = root / "warmup" / run["condition"].removeprefix("warmup_")
        if not directory.is_dir():
            directory = root / "warmup"
        validate_result(_locate(run["raw_result"], directory), execution["warmup_prompts"],
                        run["requested_input_tokens"], run["requested_output_tokens"])
        run["_directory"] = directory
    runs = warmups + measured
    server_requests = 0
    for run in runs:
        label = f"{root}: {run['condition']} repetition {run['repetition']}"
        if execution.get("request_id_prefix"):
            command = run["command"]
            if "--request-id-prefix" not in command:
                raise ValueError(f"{label}: command has no --request-id-prefix")
            prefix = command[command.index("--request-id-prefix") + 1]
            if not prefix.startswith(f"{manifest['series_id']}-{run['condition']}-"):
                raise ValueError(f"{label}: unexpected request-id prefix {prefix!r}")
        if enabled.get("server_metrics"):
            record = run.get("server_metrics") or {}
            if "error" in record or "after" not in record:
                raise ValueError(f"{label}: server metrics incomplete: {record.get('error', 'no after scrape')}")
            before = _locate(record["before"], run["_directory"]).read_text()
            after = _locate(record["after"], run["_directory"]).read_text()
            try:
                summary = prometheus.summarize_pair(before, after)
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from exc
            expected = run["expected_server_requests"]
            for family in prometheus.REQUIRED_FAMILIES:
                count = summary["families"][family]["count"]
                if count != expected:
                    raise ValueError(f"{label}: server recorded {count:g} {family} observations; "
                                     f"expected {expected} requests")
            server_requests += expected
        run.pop("_directory", None)
    if enabled.get("server_metrics"):
        initial = manifest.get("server_metrics", {}).get("initial")
        if initial is None:
            raise ValueError(f"{root}: no initial metrics scrape recorded")
        prometheus.check_scrape(_locate(initial, root).read_text())
    result = {"warmup_runs": len(warmups), "server_requests": server_requests, "telemetry": {}}
    start = min(datetime.fromisoformat(run["started_at_utc"]) for run in runs)
    end = max(datetime.fromisoformat(run["finished_at_utc"]) for run in runs)
    for kind, flag, parser in (("gpu", "gpu_sampler", telemetry.parse_gpu_csv),
                               ("cpu", "cpu_sampler", telemetry.parse_mpstat)):
        if not enabled.get(flag):
            continue
        record = manifest.get("telemetry", {}).get(kind, {})
        if record.get("exited_early") or record.get("error"):
            raise ValueError(f"{root}: {kind} sampler failed: {record}")
        samples = parser(_locate(record.get("path", ""), root).read_text())
        try:
            result["telemetry"][kind] = telemetry.coverage(samples, start, end, kind)
        except ValueError as exc:
            raise ValueError(f"{root}: {exc}") from exc
    settings = config.get("contention", {"mode": "off"})
    if settings["mode"] != "off":
        record = manifest.get("contention", {})
        if record.get("returncode") != 0 or record.get("exited_early"):
            raise ValueError(f"{root}: contention generator did not exit cleanly: {record.get('returncode')}")
        if record.get("vram_mib") is None or record["vram_mib"] > settings["memory_cap_mib"]:
            raise ValueError(f"{root}: contention VRAM {record.get('vram_mib')} MiB exceeds or lacks a check "
                             f"against the {settings['memory_cap_mib']} MiB cap")
        events = contention.read_log(_locate(record.get("log", ""), root).read_text())
        try:
            result["contention"] = contention.validate_log(events, settings["mode"], settings.get("duty_cycle"))
        except ValueError as exc:
            raise ValueError(f"{root}: {exc}") from exc
    return result
