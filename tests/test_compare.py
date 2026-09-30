import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("compare", Path("analysis/compare.py"))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def write_result(tmp_path, document):
    path = tmp_path / "series" / "baseline_c1" / "rep1" / "requests.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document))
    return path


def test_summary_derives_tpot_and_e2el_and_counts_only_real_errors(tmp_path):
    # Shaped like vLLM 0.29.0 `bench serve --save-detailed` output: no tpots/e2els keys.
    document = {
        "ttfts": [0.1, 0.2, 0.0],
        "itls": [[0.01, 0.03], [0.02, 0.02, 0.02], []],
        "output_lens": [3, 4, 0],
        "errors": ["", "", "Connection reset"],
        "request_throughput": 5.0,
        "output_throughput": 20.0,
    }
    row = compare.summarize(write_result(tmp_path, document))
    assert row["errors"] == 1
    assert row["ttft_ms"] == pytest.approx(150.0)
    assert row["tpot_ms"] == pytest.approx(20.0)
    assert row["e2el_ms"] == pytest.approx(200.0)
    assert row["itl_ms"] == pytest.approx(20.0)


def test_summary_reports_zero_errors_when_all_requests_succeed(tmp_path):
    document = {"ttfts": [0.1], "itls": [[0.01]], "output_lens": [2], "errors": [""]}
    assert compare.summarize(write_result(tmp_path, document))["errors"] == 0
