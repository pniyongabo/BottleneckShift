import importlib.util
from pathlib import Path

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
_spec = importlib.util.spec_from_file_location("figures", Path("analysis/milestone3_figures.py"))
figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figures)

PAIR = "milestone3c-synthetic"


def session(tmp_path, monkeypatch):
    """A synthetic stand-in for the Phase C session: off-before, sham, active, off-after."""
    runs = tmp_path / "runs"
    for name, extra in (("off-before", ""),
                        ("sham", '\n[contention]\nmode = "sham"\nmemory_cap_mib = 1024\n'),
                        ("active-050", '\n[contention]\nmode = "active"\nduty_cycle = 0.5\nmemory_cap_mib = 1024\n'),
                        ("off-after", "")):
        source = small_config(Path("configs/milestone3-prefill_heavy-forward.toml").read_text(), contention=extra)
        work = tmp_path / name
        work.mkdir()
        root = run_series(work, monkeypatch, run_experiment, source, series_id=f"{PAIR}-{name}")
        runs.mkdir(exist_ok=True)
        (runs / root.name).symlink_to(root)
    return runs


def test_percentile_and_by_concurrency():
    assert figures.percentile([3.0, 1.0, 2.0, 4.0], 0.5) == 3.0
    assert figures.percentile([1.0], 0.99) == 1.0
    rows = [{"max_concurrency": 1, "x": 2.0}, {"max_concurrency": 8, "x": 4.0},
            {"max_concurrency": 8, "x": 6.0}]
    assert figures.by_concurrency(rows, "x") == ([1, 8], [2.0, 5.0])


def test_rows_with_itl_adds_percentiles_from_raw_arrays(tmp_path, monkeypatch):
    runs = session(tmp_path, monkeypatch)
    rows = figures.rows_with_itl(runs / f"{PAIR}-off-before")
    assert len(rows) == 2
    # The synthetic fixture uses a constant 4 ms inter-token interval.
    assert all(row["itl_p50_ms"] == pytest.approx(4.0) for row in rows)
    assert all(row["itl_p99_ms"] == pytest.approx(4.0) for row in rows)


def test_main_writes_both_figures(tmp_path, monkeypatch, capsys):
    runs = session(tmp_path, monkeypatch)
    output = tmp_path / "figures"
    monkeypatch.setattr("sys.argv", ["milestone3_figures.py", str(runs), PAIR,
                                     "--output-dir", str(output)])
    assert figures.main() == 0
    written = sorted(path.name for path in output.iterdir())
    assert written == ["milestone3-phase-c-contention.png", "milestone3-phase-c-crossover.png"]
    assert all((output / name).stat().st_size > 1000 for name in written)
    assert "wrote figures" in capsys.readouterr().out
