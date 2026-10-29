import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path("analysis").resolve()))
spec = importlib.util.spec_from_file_location("milestone5_part2", Path("analysis/milestone5_part2.py"))
part2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(part2)

T0 = 1_793_000_000.0


def write_series(root: Path, budget: int, chunk_ms: dict[int, float]) -> Path:
    """One measured run of one 8192-token prompt, prefilled in 8192 / budget chunks,
    then one decode step; each step's execution time is set by chunk_ms[context]."""
    root.mkdir()
    lines, t, step = [], T0 + 1, 0
    for context in range(0, part2.PROMPT_TOKENS, budget):
        step += 1
        lines.append({"step": step, "t_start": t, "t_end": t, "tokens": budget, "prefill_tokens": budget,
                      "decode_tokens": 0, "requests": 1, "prefill_requests": 1,
                      "chunks": [[context, budget]], "kv_tokens": context})
        t += chunk_ms[context] / 1000
        lines.append({"step": step, "t_output": t})
    for _ in range(2):  # two decode steps of 4 ms
        step += 1
        lines.append({"step": step, "t_start": t, "t_end": t, "tokens": 1, "prefill_tokens": 0,
                      "decode_tokens": 1, "requests": 1, "prefill_requests": 0, "chunks": [], "kv_tokens": 8192})
        t += 0.004
        lines.append({"step": step, "t_output": t})
    (root / "steps.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    manifest = {"series_id": root.name, "config": {"server": {"max_num_batched_tokens": budget}},
                "runs": [{"measurement_role": "measured",
                          "started_at_utc": "2026-10-26T07:33:20+00:00",
                          "finished_at_utc": "2026-10-26T07:34:00+00:00"}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_context_summary_scores_positions_ratio_and_correlation(tmp_path):
    chunk_ms = {context: 20 + context / 1024 for context in range(0, 8192, 1024)}
    summary = part2.context_summary(write_series(tmp_path / "ctx-budget1024", 1024, chunk_ms))
    assert (summary["prefill_steps"], summary["full_chunks"]) == (8, 8)
    assert summary["positions_as_expected"] and summary["steps_without_output_time"] == 0
    assert summary["median_ms_by_context"][0] == pytest.approx(20)
    assert summary["deepest_over_first"] == pytest.approx(27 / 20)
    assert summary["spearman_context_time"] == pytest.approx(1.0)
    assert summary["decode_median_ms"] == pytest.approx(4, abs=1e-3)


def test_timing_summary_compares_execution_time_with_schedule_interval(tmp_path):
    chunk_ms = {context: 20 + context / 1024 for context in range(0, 8192, 2048)}
    result = part2.timing_summary([write_series(tmp_path / "ctx-budget2048", 2048, chunk_ms)])
    assert result["steps_timed"] == 6
    assert result["spearman_tokens_execution"] > 0
