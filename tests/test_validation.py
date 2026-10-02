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
