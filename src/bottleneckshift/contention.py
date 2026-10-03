"""GPU contention generator logic, kept free of torch so it can be unit-tested.

The generator runs fixed-size matrix multiplies for `duty_cycle` of every period and
sleeps for the rest. Sham mode allocates the same memory and holds a CUDA context but
runs no steps. `memory_cap_mib` bounds the whole process, including the CUDA context.
"""

from datetime import datetime, timezone
import json
import time

MIB = 1 << 20
CONTEXT_RESERVE_MIB = 512  # CUDA context + cuBLAS workspace, outside torch's allocator
DEFAULT_MATRIX_SIZE = 2048  # one fp16 matmul is well under a 10 ms period on an A4000
DTYPE_BYTES = 2
MODES = ("off", "sham", "active")


def tensor_budget_bytes(memory_cap_mib: int) -> int:
    budget = (memory_cap_mib - CONTEXT_RESERVE_MIB) * MIB
    if budget <= 0:
        raise ValueError(f"memory_cap_mib must exceed the {CONTEXT_RESERVE_MIB} MiB context reserve")
    return budget


def buffer_bytes(matrix_size: int) -> int:
    return 3 * matrix_size * matrix_size * DTYPE_BYTES  # A, B, and the output


def matrix_size_for_cap(memory_cap_mib: int, preferred: int = DEFAULT_MATRIX_SIZE) -> int:
    """The preferred size, reduced to the largest multiple of 256 the budget allows."""
    budget, size = tensor_budget_bytes(memory_cap_mib), preferred - preferred % 256
    while size >= 256 and buffer_bytes(size) > budget:
        size -= 256
    if size < 256:
        raise ValueError(f"memory_cap_mib {memory_cap_mib} cannot hold a 256×256 matmul")
    return size


def ballast_bytes(memory_cap_mib: int, matrix_size: int) -> int:
    """Extra allocation so sham and active hold the same, fixed tensor footprint."""
    remaining = tensor_budget_bytes(memory_cap_mib) - buffer_bytes(matrix_size)
    if remaining < 0:
        raise ValueError("matrix buffers exceed the memory cap")
    return remaining


class DutyCycleController:
    """Run `step()` for `duty_cycle` of each period and sleep for the remainder."""

    def __init__(self, duty_cycle: float, period_s: float, step, clock=time.monotonic,
                 sleep=time.sleep):
        if not 0 <= duty_cycle <= 1:
            raise ValueError("duty_cycle must be in [0, 1]")
        self.duty_cycle, self.period_s, self.step = duty_cycle, period_s, step
        self.clock, self.sleep = clock, sleep

    def run(self, should_stop, emit, window_s: float = 1.0) -> dict:
        """Loop until `should_stop()`; `emit` receives one record per window."""
        totals = {"periods": 0, "busy_s": 0.0, "elapsed_s": 0.0, "steps": 0, "kernel_s": 0.0}
        window = dict(totals)
        window_start = self.clock()
        while not should_stop():
            period_start = self.clock()
            busy_until = period_start + self.duty_cycle * self.period_s
            steps = kernel = 0
            while self.duty_cycle and self.clock() < busy_until:
                kernel += self.step()
                steps += 1
            busy = self.clock() - period_start if steps else 0.0
            remaining = period_start + self.period_s - self.clock()
            if remaining > 0:
                self.sleep(remaining)
            elapsed = self.clock() - period_start
            for record in (totals, window):
                record["periods"] += 1
                record["busy_s"] += busy
                record["elapsed_s"] += elapsed
                record["steps"] += steps
                record["kernel_s"] += kernel
            if self.clock() - window_start >= window_s:
                emit({"event": "window", "t_utc": utc_now(), **window})
                window, window_start = {key: 0 if key in ("periods", "steps") else 0.0
                                        for key in totals}, self.clock()
        if window["periods"]:
            emit({"event": "window", "t_utc": utc_now(), **window})
        totals["achieved_duty"] = totals["busy_s"] / totals["elapsed_s"] if totals["elapsed_s"] else 0.0
        totals["kernel_duty"] = totals["kernel_s"] / totals["elapsed_s"] if totals["elapsed_s"] else 0.0
        return totals


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_log(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def validate_log(events: list[dict], mode: str, duty_cycle: float | None,
                 tolerance: float = 0.05) -> dict:
    """Check a contention log is complete and its achieved duty matches the config."""
    kinds = [event.get("event") for event in events]
    for required in ("start", "ready", "stop"):
        if required not in kinds:
            raise ValueError(f"contention log has no {required} event")
    start = events[kinds.index("start")]
    stop = events[kinds.index("stop")]
    if start.get("mode") != mode:
        raise ValueError(f"contention log mode {start.get('mode')!r} != configured {mode!r}")
    if mode == "sham" and stop.get("total_steps") != 0:
        raise ValueError("sham contention ran matmul steps")
    if mode == "active" and abs(stop.get("achieved_duty", -1) - duty_cycle) > tolerance:
        raise ValueError(f"achieved duty {stop.get('achieved_duty')} differs from {duty_cycle} "
                         f"by more than {tolerance}")
    return {"mode": mode, "achieved_duty": stop.get("achieved_duty"),
            "kernel_duty": stop.get("kernel_duty"), "total_steps": stop.get("total_steps"),
            "windows": kinds.count("window")}
