import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("milestone5_steps", Path("analysis/milestone5_steps.py"))
steps_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(steps_module)


def step(t, tokens, prefill=0):
    return {"t_start": t, "tokens": tokens, "prefill_tokens": prefill}


def test_run_steps_pairs_each_step_with_the_gap_to_the_next_and_drops_idle_gaps():
    steps = [step(9.0, 99), step(10.0, 2048, 2016), step(10.05, 32), step(10.056, 32),
             step(12.0, 2048, 2048), step(12.06, 32), step(25.0, 7)]
    pairs = steps_module.run_steps(steps, 10.0, 20.0)
    assert [(tokens, round(interval, 3), prefill) for tokens, interval, prefill in pairs] == [
        (2048, 0.05, 2016), (32, 0.006, 0), (2048, 0.06, 2048)]  # 10.056 → 12.0 is idle; 12.06 is last


def test_ranks_average_ties():
    assert steps_module.ranks([10, 20, 20, 5]) == [2.0, 3.5, 3.5, 1.0]


def test_spearman_is_one_for_monotone_data_and_handles_ties():
    assert steps_module.spearman([1, 2, 3, 4], [10, 40, 90, 160]) == pytest.approx(1.0)
    assert steps_module.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert -1 <= steps_module.spearman([1, 1, 2, 2], [1, 2, 1, 2]) <= 1
    assert steps_module.spearman([1, 1, 1], [1, 2, 3]) != steps_module.spearman([1, 1, 1], [1, 2, 3])  # NaN


def test_pooled_ratio_uses_large_and_small_step_medians():
    pairs = [(2048, 0.050, 2000), (2048, 0.060, 2000), (32, 0.005, 0), (32, 0.006, 0), (500, 0.02, 400)]
    result = steps_module.pooled(pairs)
    assert (result["large_steps"], result["small_steps"]) == (2, 2)
    assert result["ratio"] == pytest.approx(0.055 / 0.0055)


def test_execution_time_starts_when_the_step_is_scheduled_and_the_previous_step_is_done():
    steps = [{"step": 1, "t_end": 0.000, "t_output": 0.030},
             {"step": 2, "t_end": 0.001, "t_output": 0.050},   # scheduled early: waits for step 1
             {"step": 3, "t_end": 0.080, "t_output": 0.090},   # scheduled after step 2 finished
             {"step": 4, "t_end": 0.100}]                      # no output time recorded
    times = steps_module.execution_times(steps)
    assert {k: round(v, 3) for k, v in times.items()} == {1: 0.030, 2: 0.020, 3: 0.010}
