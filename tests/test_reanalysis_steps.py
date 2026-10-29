import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path("analysis").resolve()))
spec = importlib.util.spec_from_file_location("reanalysis_steps", Path("analysis/reanalysis_steps.py"))
reanalysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reanalysis)


def step(t_start, tokens):
    return {"t_start": t_start, "tokens": tokens}


def test_intervals_drop_the_last_step_and_idle_gaps():
    run = [step(0.0, 10), step(0.05, 2048), step(2.0, 1), step(2.01, 1)]
    assert reanalysis.intervals(run) == [pytest.approx(0.05), None, pytest.approx(0.01), None]


def test_features_split_prior_context_attention_from_decode_kv():
    record = {"tokens": 2052, "chunks": [[1024, 1024], [0, 1024]], "kv_tokens": 1024 + 300 + 500}
    fixed, tokens, attention, decode_kv = reanalysis.features(record)
    assert (fixed, tokens) == (1.0, 2052)
    assert attention == 1024 * (1024 + 512) + 1024 * 512
    assert decode_kv == 800


def test_weighted_percentile_counts_each_step_once_per_decoding_request():
    values = [0.005, 0.050]
    assert reanalysis.weighted_percentile(values, [9, 1], 0.5) == 0.005
    assert reanalysis.weighted_percentile(values, [9, 1], 0.99) == 0.050
    assert reanalysis.weighted_percentile(values, [1, 9], 0.5) == 0.050


def test_next_interval_pairs_take_each_steps_cost_from_the_following_interval():
    run = [step(0.0, 2048), step(0.010, 8), step(0.065, 8), step(0.071, 2048), step(2.0, 8), step(2.005, 8)]
    tokens, costs = reanalysis.next_interval_pairs(run)
    assert tokens == [2048, 8, 2048]
    assert costs == [pytest.approx(0.055), pytest.approx(0.006), pytest.approx(0.005)]
