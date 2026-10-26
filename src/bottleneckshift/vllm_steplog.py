"""Opt-in vLLM plugin that logs every engine step's scheduled tokens (Milestone 5).

vLLM loads `vllm.general_plugins` entry points in every process, including the engine
core, before the scheduler is created. `register()` does nothing unless
`BOTTLENECKSHIFT_STEP_LOG` names an output file, so runs that don't opt in are
unaffected.

When enabled, `Scheduler.schedule()` is wrapped. For each step that schedules at least
one token, one JSON line records the wall-clock time before and after scheduling, the
total scheduled tokens, how many of them were prompt (prefill) tokens, and the number
of scheduled requests. A request's tokens count as prefill up to its prompt length,
using its computed-token count from before the step.

Since Milestone 5 Part 2, each step line also has a sequence number (`step`), the
context position and size of each prefill chunk (`chunks`: [computed tokens before the
step, chunk tokens]), and the scheduled requests' computed tokens before the step
(`kv_tokens`). `Scheduler.update_from_output()` is wrapped too: when a step's results
arrive, a short line `{"step": n, "t_output": ...}` records the time. Earlier logs have
only the step lines.
"""

import json
import os
import time

ENV = "BOTTLENECKSHIFT_STEP_LOG"
FIELDS = ("t_start", "t_end", "tokens", "prefill_tokens", "decode_tokens", "requests",
          "prefill_requests")
MARKER = "bottleneckshift_steplog"


def step_record(output, computed_before: dict, requests: dict, t_start: float, t_end: float) -> dict:
    """One step's record, from a `SchedulerOutput`-like object."""
    prefill = prefill_requests = kv_tokens = 0
    chunks = []
    for request_id, count in output.num_scheduled_tokens.items():
        request = requests.get(request_id)
        if request is None:
            continue
        before = computed_before.get(request_id, 0)
        kv_tokens += before
        share = max(0, min(count, request.num_prompt_tokens - before))
        if share:
            prefill += share
            prefill_requests += 1
            chunks.append([before, share])
    total = output.total_num_scheduled_tokens
    return {"t_start": t_start, "t_end": t_end, "tokens": total, "prefill_tokens": prefill,
            "decode_tokens": total - prefill, "requests": len(output.num_scheduled_tokens),
            "prefill_requests": prefill_requests, "chunks": chunks, "kv_tokens": kv_tokens}


class StepLog:
    """Shared state for the two wrappers: the open file, a step counter, and the
    sequence numbers of steps whose results haven't arrived yet (keyed by the
    identity of their `SchedulerOutput`)."""

    def __init__(self, path: str):
        self.handle = open(path, "a", buffering=1)  # line-buffered; one line per event
        self.count = 0
        self.pending = {}

    def write(self, record: dict) -> None:
        self.handle.write(json.dumps(record) + "\n")


def wrap_schedule(original, log: StepLog):
    def schedule(self, *args, **kwargs):
        computed = {request_id: request.num_computed_tokens for request_id, request in self.requests.items()}
        t_start = time.time()
        output = original(self, *args, **kwargs)
        t_end = time.time()
        if output.total_num_scheduled_tokens:
            log.count += 1
            log.pending[id(output)] = log.count
            log.write({"step": log.count, **step_record(output, computed, self.requests, t_start, t_end)})
        return output

    schedule.__wrapped__ = original
    setattr(schedule, MARKER, True)
    return schedule


def wrap_update(original, log: StepLog):
    def update_from_output(self, scheduler_output, *args, **kwargs):
        t_output = time.time()
        step = log.pending.pop(id(scheduler_output), None)
        if step is not None:
            log.write({"step": step, "t_output": t_output})
        return original(self, scheduler_output, *args, **kwargs)

    update_from_output.__wrapped__ = original
    setattr(update_from_output, MARKER, True)
    return update_from_output


def register() -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    if getattr(Scheduler.schedule, MARKER, False):
        return
    log = StepLog(path)
    Scheduler.schedule = wrap_schedule(Scheduler.schedule, log)
    Scheduler.update_from_output = wrap_update(Scheduler.update_from_output, log)


def read_steps(text: str) -> list[dict]:
    """Parse a step log into step records; output-time lines are merged into their step
    as `t_output`. Every step line must carry all FIELDS.

    Only the final line may be cut off (the server stopped mid-write); it is dropped.
    """
    steps, outputs = [], {}
    lines = text.splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines) and not text.endswith("\n"):
                break
            raise
        if "t_output" in record and "tokens" not in record:
            outputs[record["step"]] = record["t_output"]
            continue
        missing = [field for field in FIELDS if field not in record]
        if missing:
            raise ValueError(f"step log line {number} lacks {missing}")
        steps.append(record)
    for record in steps:
        if record.get("step") in outputs:
            record["t_output"] = outputs[record["step"]]
    return steps
