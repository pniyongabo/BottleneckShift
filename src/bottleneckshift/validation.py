"""Strict, schema-light validation for retained vLLM benchmark series."""

import json
from pathlib import Path


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
    return {"series_id": manifest.get("series_id"), "measured_runs": len(validated),
            "measured_requests": sum(item["requests"] for item in validated), "runs": validated}
