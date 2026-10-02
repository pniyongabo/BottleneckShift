import json
from pathlib import Path

import pytest

from bottleneckshift.validation import validate_series


def synthetic_series(tmp_path: Path, *, output_lens=(4, 4)) -> Path:
    root = tmp_path / "synthetic-series"
    result = root / "baseline" / "rep-01" / "requests.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({
        "ttfts": [0.1, 0.2], "itls": [[0.01] * 3, [0.02] * 3],
        "input_lens": [3, 3], "output_lens": list(output_lens), "errors": ["", ""],
    }))
    manifest = {
        "status": "complete", "series_id": "synthetic-series",
        "config": {"experiment": {"repetitions": 1}, "workload": {"num_prompts": 2},
                   "condition": [{"name": "baseline", "max_concurrency": 1}]},
        "runs": [{"measurement_role": "measured", "condition": "baseline", "repetition": 1,
                  "returncode": 0, "raw_result": str(result), "requested_input_tokens": 3,
                  "requested_output_tokens": 4}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_validate_series_accepts_complete_synthetic_fixture(tmp_path):
    summary = validate_series(synthetic_series(tmp_path))
    assert summary["measured_runs"] == 1
    assert summary["measured_requests"] == 2


def test_validate_series_rejects_wrong_token_length(tmp_path):
    with pytest.raises(ValueError, match="output_lens differs"):
        validate_series(synthetic_series(tmp_path, output_lens=(4, 3)))


def mutate(root: Path, *, manifest=None, result=None) -> Path:
    path = root / "manifest.json"
    document = json.loads(path.read_text())
    if manifest:
        manifest(document)
    path.write_text(json.dumps(document))
    result_path = root / "baseline" / "rep-01" / "requests.json"
    document = json.loads(result_path.read_text())
    if result:
        result(document)
    result_path.write_text(json.dumps(document))
    return root


@pytest.mark.parametrize("manifest, result, message", [
    (lambda m: m.update(status="failed"), None, "status is not complete"),
    (lambda m: m["runs"].clear(), None, "found 0 measured runs"),
    (None, lambda r: r.pop("itls"), "missing request array itls"),
    (None, lambda r: r["ttfts"].append(0.3), "ttfts has 3 entries"),
    (lambda m: m["runs"][0].update(returncode=1), None, "nonzero return code"),
    (None, lambda r: r.update(errors=["", "timeout"]), "non-empty errors at request indexes \\[1\\]"),
    (lambda m: (m["config"]["experiment"].update(repetitions=2), m["runs"].append(dict(m["runs"][0]))),
     None, "duplicate measured run"),
    (None, lambda r: r.update(input_lens=[3, 2]), "input_lens differs"),
])
def test_validate_series_rejects_invalid_series(tmp_path, manifest, result, message):
    with pytest.raises(ValueError, match=message):
        validate_series(mutate(synthetic_series(tmp_path), manifest=manifest, result=result))
