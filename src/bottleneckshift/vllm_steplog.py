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
"""

import json
import os
import time

ENV = "BOTTLENECKSHIFT_STEP_LOG"
FIELDS = ("t_start", "t_end", "tokens", "prefill_tokens", "decode_tokens", "requests",
          "prefill_requests")


def step_record(output, computed_before: dict, requests: dict, t_start: float, t_end: float) -> dict:
    """One step's record, from a `SchedulerOutput`-like object."""
    prefill = prefill_requests = 0
    for request_id, count in output.num_scheduled_tokens.items():
        request = requests.get(request_id)
        if request is None:
            continue
        remaining = request.num_prompt_tokens - computed_before.get(request_id, 0)
        share = max(0, min(count, remaining))
        prefill += share
        prefill_requests += share > 0
    total = output.total_num_scheduled_tokens
    return {"t_start": t_start, "t_end": t_end, "tokens": total, "prefill_tokens": prefill,
            "decode_tokens": total - prefill, "requests": len(output.num_scheduled_tokens),
            "prefill_requests": prefill_requests}


def wrap_schedule(original, path: str):
    handle = open(path, "a", buffering=1)  # line-buffered; one line per step

    def schedule(self, *args, **kwargs):
        computed = {request_id: request.num_computed_tokens for request_id, request in self.requests.items()}
        t_start = time.time()
        output = original(self, *args, **kwargs)
        t_end = time.time()
        if output.total_num_scheduled_tokens:
            handle.write(json.dumps(step_record(output, computed, self.requests, t_start, t_end)) + "\n")
        return output

    schedule.__wrapped__ = original
    schedule.bottleneckshift_steplog = True
    return schedule


def register() -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    if not getattr(Scheduler.schedule, "bottleneckshift_steplog", False):
        Scheduler.schedule = wrap_schedule(Scheduler.schedule, path)


def read_steps(text: str) -> list[dict]:
    """Parse a step log; every line must carry all fields.

    Only the final line may be cut off (the server stopped mid-write); it is dropped.
    """
    steps = []
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
        missing = [field for field in FIELDS if field not in record]
        if missing:
            raise ValueError(f"step log line {number} lacks {missing}")
        steps.append(record)
    return steps
