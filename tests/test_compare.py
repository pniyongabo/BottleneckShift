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
        "median_e2el_ms": 201.5,
    }
    row = compare.summarize(write_result(tmp_path, document))
    assert row["errors"] == 1
    assert row["ttft_ms"] == pytest.approx(150.0)
    assert row["tpot_ms"] == pytest.approx(20.0)
    assert row["e2el_ms"] == pytest.approx(200.0)
    assert row["itl_ms"] == pytest.approx(20.0)
    assert row["vllm_median_e2el_ms"] == 201.5


def test_summary_reports_zero_errors_when_all_requests_succeed(tmp_path):
    document = {"ttfts": [0.1], "itls": [[0.01]], "output_lens": [2], "errors": [""]}
    assert compare.summarize(write_result(tmp_path, document))["errors"] == 0


def test_summary_reports_token_ranges(tmp_path):
    document = {"ttfts": [0.1, 0.2], "itls": [[], []], "input_lens": [3, 4],
                "output_lens": [2, 5], "errors": ["", ""]}
    row = compare.summarize(write_result(tmp_path, document))
    assert (row["input_tokens_min"], row["input_tokens_max"]) == (3, 4)
    assert (row["output_tokens_min"], row["output_tokens_max"]) == (2, 5)


def test_main_excludes_every_warmup_path(tmp_path, monkeypatch, capsys):
    measured = {"ttfts": [0.1, 0.2], "itls": [[0.01], [0.01]], "input_lens": [3, 3],
                "output_lens": [2, 2], "errors": ["", ""]}
    series = tmp_path / "runs" / "synthetic"
    for relative in ("warmup", "warmup/c1", "warmup/c8", "baseline_c1/rep-01"):
        (series / relative).mkdir(parents=True)
        (series / relative / "requests.json").write_text(json.dumps(measured))
    monkeypatch.setattr("sys.argv", ["compare.py", str(tmp_path / "runs"),
                                     "--output", str(tmp_path / "figure.svg")])
    assert compare.main() == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    header, row = lines[0].split(","), dict(zip(lines[0].split(","), lines[1].split(",")))
    assert {"requests", "input_tokens_min", "input_tokens_max",
            "output_tokens_min", "output_tokens_max"} <= set(header)
    assert (row["condition"], row["requests"], row["input_tokens_max"]) == ("baseline_c1", "2", "3")
