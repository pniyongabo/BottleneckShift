import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bottleneckshift import vllm_steplog
from bottleneckshift.config import load_config
from bottleneckshift.validation import _validate_step_log

spec = importlib.util.spec_from_file_location("run_experiment", Path("scripts/run_experiment.py"))
run_experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment)

LENGTHS, BUDGETS, LEVELS = (1024, 2048, 4096), (2048, 8192), (32, 8)


class StubScheduler:
    """Mimics the parts of vLLM's v1 Scheduler the plugin reads."""

    def __init__(self, plans):
        self.requests, self.plans = {}, list(plans)

    def schedule(self):
        plan = self.plans.pop(0)
        for request_id, count in plan.items():
            self.requests[request_id].num_computed_tokens += count  # as vLLM does inside schedule()
        return SimpleNamespace(num_scheduled_tokens=dict(plan), total_num_scheduled_tokens=sum(plan.values()))


def request(prompt_tokens):
    return SimpleNamespace(num_prompt_tokens=prompt_tokens, num_computed_tokens=0)


def test_wrapped_schedule_splits_prefill_and_decode_and_skips_empty_steps(tmp_path):
    path = tmp_path / "steps.jsonl"
    scheduler = StubScheduler([{"a": 2048}, {"a": 1024, "b": 1024}, {}, {"a": 1, "b": 1024}, {"a": 1, "b": 1}])
    scheduler.requests = {"a": request(3072), "b": request(2048)}
    schedule = vllm_steplog.wrap_schedule(StubScheduler.schedule, vllm_steplog.StepLog(str(path)))
    for _ in range(5):
        schedule(scheduler)
    steps = vllm_steplog.read_steps(path.read_text())
    assert [(s["tokens"], s["prefill_tokens"], s["decode_tokens"], s["requests"], s["prefill_requests"])
            for s in steps] == [(2048, 2048, 0, 1, 1), (2048, 2048, 0, 2, 2), (1025, 1024, 1, 2, 1),
                                (2, 0, 2, 2, 0)]
    assert all(s["t_start"] <= s["t_end"] for s in steps)
    assert [s["step"] for s in steps] == [1, 2, 3, 4]
    # Prefill chunks carry their context position; kv_tokens sums computed tokens before the step.
    assert [s["chunks"] for s in steps] == [[[0, 2048]], [[2048, 1024], [0, 1024]], [[1024, 1024]], []]
    assert [s["kv_tokens"] for s in steps] == [0, 2048, 4096, 5121]


def test_output_times_are_matched_to_their_step_even_when_the_next_step_is_scheduled_first(tmp_path):
    path = tmp_path / "steps.jsonl"
    log = vllm_steplog.StepLog(str(path))
    scheduler = StubScheduler([{"a": 1024}, {"a": 1024}])
    scheduler.requests = {"a": request(2048)}
    schedule = vllm_steplog.wrap_schedule(StubScheduler.schedule, log)
    update = vllm_steplog.wrap_update(lambda self, output, runner_output: "updated", log)
    first, second = schedule(scheduler), schedule(scheduler)  # batch queue: two steps in flight
    assert update(scheduler, second, None) == "updated"
    update(scheduler, first, None)
    update(scheduler, first, None)  # a repeat for the same output writes nothing
    steps = vllm_steplog.read_steps(path.read_text())
    assert [s["step"] for s in steps] == [1, 2]
    assert steps[0]["t_output"] >= steps[1]["t_output"] >= steps[1]["t_end"]
    assert len(path.read_text().splitlines()) == 4


def test_register_is_inert_without_the_environment_variable(monkeypatch):
    monkeypatch.delenv(vllm_steplog.ENV, raising=False)
    monkeypatch.setitem(sys.modules, "vllm", None)  # importing vLLM would raise
    assert vllm_steplog.register() is None


def test_read_steps_rejects_incomplete_lines():
    with pytest.raises(ValueError, match="lacks"):
        vllm_steplog.read_steps(json.dumps({"tokens": 1}))


def test_read_steps_drops_only_a_truncated_final_line():
    line = json.dumps({field: 1 for field in vllm_steplog.FIELDS})
    assert len(vllm_steplog.read_steps(line + "\n" + line[:10])) == 1
    with pytest.raises(ValueError):
        vllm_steplog.read_steps(line[:10] + "\n" + line + "\n")


def test_milestone5_configs_match_the_protocol():
    base = load_config(Path("configs/milestone3-budget-2048.toml"))
    for level in LEVELS:
        for length in LENGTHS:
            for budget in BUDGETS:
                config = load_config(Path(f"configs/milestone5-len{length:04d}-budget{budget:04d}-c{level}.toml"))
                assert config["experiment"] == {**base["experiment"],
                                                "name": f"milestone5-len{length:04d}-budget{budget:04d}-c{level}"}
                assert config["execution"] == {**base["execution"], "warmup_concurrencies": [level]}
                assert config["server"] == {**base["server"], "max_num_batched_tokens": budget}
                assert config["workload"] == {**base["workload"], "input_tokens": length}
                assert config["telemetry"] == {**base["telemetry"], "step_log": True}
                assert config["sampling"] == base["sampling"]
                assert config["condition"] == [{"name": f"c{level}", "max_concurrency": level}]


def test_part2_context_configs_match_the_addendum():
    base = load_config(Path("configs/milestone5-len2048-budget2048-c32.toml"))
    for budget in (1024, 2048):
        config = load_config(Path(f"configs/milestone5-ctx-len8192-budget{budget:04d}-c1.toml"))
        assert config["experiment"]["name"] == f"milestone5-ctx-len8192-budget{budget:04d}-c1"
        assert config["execution"] == {**base["execution"], "warmup_concurrencies": [1]}
        assert config["server"] == {**base["server"], "max_num_batched_tokens": budget}
        assert config["workload"] == {**base["workload"], "input_tokens": 8192}
        for key in ("telemetry", "sampling"):
            assert config[key] == base[key]
        assert config["condition"] == [{"name": "c1", "max_concurrency": 1}]


def test_step_log_environment_is_opt_in(tmp_path):
    old = load_config(Path("configs/milestone3-budget-2048.toml"))
    new = load_config(Path("configs/milestone5-len4096-budget2048-c32.toml"))
    assert run_experiment.server_env(old, tmp_path) is None
    assert run_experiment.server_env(new, tmp_path) == {vllm_steplog.ENV: str(tmp_path / "steps.jsonl")}


def step_manifest(tmp_path, steps):
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps({field: 0 for field in vllm_steplog.FIELDS} | step) + "\n"
                            for step in steps))
    run = {"condition": "c32", "repetition": 1, "requested_input_tokens": 4,
           "started_at_utc": "2026-10-25T00:00:10+00:00", "finished_at_utc": "2026-10-25T00:00:20+00:00"}
    return {"step_log": {"path": str(path)}, "config": {"workload": {"num_prompts": 2}}}, [run]


def test_step_log_validation_summarizes_each_measured_run(tmp_path):
    t = 1_792_886_400  # 2026-10-25T00:00:00Z
    manifest, runs = step_manifest(tmp_path, [{"t_start": t + 5, "tokens": 9, "prefill_tokens": 9},
                                              {"t_start": t + 11, "tokens": 8, "prefill_tokens": 8},
                                              {"t_start": t + 12, "tokens": 2}])
    result = _validate_step_log(tmp_path, manifest, runs)
    assert result["runs"] == [{"condition": "c32", "repetition": 1, "steps": 2, "max_tokens": 8,
                               "prefill_tokens": 8, "expected_prefill_tokens": 8}]


def test_step_log_validation_rejects_runs_without_steps(tmp_path):
    manifest, runs = step_manifest(tmp_path, [{"t_start": 1_792_886_400 + 30}])
    with pytest.raises(ValueError, match="no logged steps"):
        _validate_step_log(tmp_path, manifest, runs)
