import csv
import importlib.util
import io
import json
from pathlib import Path
import sys

import matplotlib
import pytest

from synthetic import run_series, small_config

# Import pyplot (which may build matplotlib's font cache with real subprocesses) before
# any test monkeypatches subprocess.run.
matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402,F401

_spec = importlib.util.spec_from_file_location("run_experiment", Path("scripts/run_experiment.py"))
run_experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_experiment)
_spec = importlib.util.spec_from_file_location("phases", Path("analysis/phases.py"))
phases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phases)


def series(tmp_path, monkeypatch, regime="prefill_heavy", order="forward"):
    source = small_config(Path(f"configs/milestone3-{regime}-{order}.toml").read_text())
    return run_series(tmp_path / f"{regime}-{order}", monkeypatch, run_experiment, source,
                      series_id=f"{regime}-{order}")


def test_run_rows_combine_client_server_and_telemetry(tmp_path, monkeypatch):
    (tmp_path / "prefill_heavy-forward").mkdir()
    rows = phases.run_rows(series(tmp_path, monkeypatch))
    assert [(row["condition"], row["max_concurrency"]) for row in rows] == [("baseline_c1", 1),
                                                                            ("perturbation_c8", 8)]
    row = rows[0]
    assert (row["regime"], row["order"], row["requests"]) == ("prefill_heavy", "forward", 4)
    # Synthetic client: TTFT 20 ms, 31 intervals of 4 ms -> E2E 144 ms.
    assert row["client_ttft_mean_ms"] == pytest.approx(20.0)
    assert row["client_pre_first_token_share"] == pytest.approx(20 / 144)
    # Synthetic server: every per-request histogram observes 10 ms.
    assert row["server_queue_ms"] == pytest.approx(10.0) and row["server_requests"] == 4
    assert row["server_pre_first_token_share"] == pytest.approx(20 / 20)
    assert row["dominant_component"] == "pre_first_token"
    assert row["ttft_gap_ms"] == pytest.approx(0.0)
    assert row["gpu_util_mean"] == 90 and row["cpu_util_mean"] == pytest.approx(25.0)


def test_server_row_dominance_and_shares(tmp_path):
    from synthetic import scrape
    before, after = tmp_path / "metrics-before.prom", tmp_path / "metrics-after.prom"
    before.write_text(scrape(0))
    text = scrape(2, seconds=0.01)
    # Make decode dominate: 2 requests with 100 ms decode each.
    text = text.replace("vllm:request_decode_time_seconds_sum{engine=\"0\",model_name=\"synthetic\"} 0.02",
                        "vllm:request_decode_time_seconds_sum{engine=\"0\",model_name=\"synthetic\"} 0.2")
    after.write_text(text)
    row = phases.server_row(tmp_path)
    assert row["server_decode_ms"] == pytest.approx(100.0)
    assert row["dominant_component"] == "decode" and row["largest_phase"] == "decode"


def test_main_writes_rows_summary_and_figure(tmp_path, monkeypatch, capsys):
    roots = []
    for regime in ("prefill_heavy", "decode_heavy"):
        for order in ("forward", "reverse"):
            (tmp_path / f"{regime}-{order}").mkdir()
            roots.append(series(tmp_path, monkeypatch, regime, order))
    summary, figure = tmp_path / "summary.csv", tmp_path / "figure.svg"
    monkeypatch.setattr(sys, "argv", ["phases.py", *map(str, roots), "--summary", str(summary),
                                      "--output", str(figure)])
    capsys.readouterr()
    assert phases.main() == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 8
    cells = list(csv.DictReader(summary.open()))
    assert {(cell["regime"], cell["order"], cell["condition"]) for cell in cells} == {
        (regime, order, condition) for regime in ("prefill_heavy", "decode_heavy")
        for order in ("forward", "reverse") for condition in ("baseline_c1", "perturbation_c8")}
    assert figure.read_text().startswith("<?xml")


def test_analysis_refuses_an_invalid_series(tmp_path, monkeypatch):
    (tmp_path / "prefill_heavy-forward").mkdir()
    root = series(tmp_path, monkeypatch)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["status"] = "failed"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not complete"):
        phases.run_rows(root)


def test_summary_keeps_repeated_designs_separate_and_labels_contention(tmp_path, monkeypatch):
    roots = []
    for name, extra in (("off-before", ""), ("sham", '\n[contention]\nmode = "sham"\nmemory_cap_mib = 1024\n'),
                        ("off-after", "")):
        source = small_config(Path("configs/milestone3-prefill_heavy-forward.toml").read_text(), contention=extra)
        (tmp_path / name).mkdir()
        roots.append(run_series(tmp_path / name, monkeypatch, run_experiment, source, series_id=name))
    rows = [row for root in roots for row in phases.run_rows(root)]
    cells = phases.summarize(rows)
    assert [(cell["series_id"], cell["contention"], cell["condition"]) for cell in cells] == [
        ("off-before", "off", "baseline_c1"), ("off-before", "off", "perturbation_c8"),
        ("sham", "sham", "baseline_c1"), ("sham", "sham", "perturbation_c8"),
        ("off-after", "off", "baseline_c1"), ("off-after", "off", "perturbation_c8")]


def test_server_row_reports_engine_steps(tmp_path):
    from synthetic import scrape
    (tmp_path / "metrics-before.prom").write_text(scrape(0, steps={}))
    (tmp_path / "metrics-after.prom").write_text(scrape(4, steps={16: 30, 2048: 10}))
    row = phases.server_row(tmp_path)
    assert row["server_steps"] == 40
    assert row["tokens_per_step_mean"] == pytest.approx((16 * 30 + 2048 * 10) / 40)
    assert row["steps_over_1024_share"] == pytest.approx(0.25)
    assert row["steps_over_2048_share"] == pytest.approx(0.0)
