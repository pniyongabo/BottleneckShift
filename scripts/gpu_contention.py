#!/usr/bin/env python3
"""Generate controlled GPU contention (or a sham) and log what was achieved.

Started by the experiment runner when a config has a [contention] table with mode
"sham" or "active"; it can also be run by hand for a shakedown. Stops on SIGTERM or
SIGINT and writes JSON-lines events to --log.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bottleneckshift import contention  # noqa: E402


class CudaBackend:
    """Allocates the fixed footprint and runs one timed fp16 matmul per step."""

    def __init__(self, matrix_size: int, memory_cap_mib: int):
        import torch  # imported here so the pure logic and tests do not need torch

        self.torch = torch
        torch.cuda.init()
        free, total = torch.cuda.mem_get_info()
        if free < memory_cap_mib * contention.MIB + 256 * contention.MIB:
            raise RuntimeError(f"only {free // contention.MIB} MiB free; cap is {memory_cap_mib} MiB")
        budget = contention.tensor_budget_bytes(memory_cap_mib)
        torch.cuda.set_per_process_memory_fraction(min(1.0, (budget + 64 * contention.MIB) / total))
        shape = (matrix_size, matrix_size)
        self.a = torch.randn(shape, device="cuda", dtype=torch.float16)
        self.b = torch.randn(shape, device="cuda", dtype=torch.float16)
        self.c = torch.empty(shape, device="cuda", dtype=torch.float16)
        ballast = contention.ballast_bytes(memory_cap_mib, matrix_size)
        self.ballast = torch.empty(ballast, device="cuda", dtype=torch.uint8) if ballast else None
        self.info = {"device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
                     "free_before_bytes": free, "total_bytes": total, "ballast_bytes": ballast,
                     "allocated_bytes": torch.cuda.memory_allocated()}
        self.step()  # warm up cuBLAS before reporting ready

    def step(self) -> float:
        start = self.torch.cuda.Event(enable_timing=True, blocking=True)
        end = self.torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
        self.torch.matmul(self.a, self.b, out=self.c)
        end.record()
        end.synchronize()  # blocking sync: waits without spinning a CPU core
        return start.elapsed_time(end) / 1000


def main(argv=None, backend_factory=CudaBackend) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sham", "active"), required=True)
    parser.add_argument("--duty-cycle", type=float, default=0.0)
    parser.add_argument("--period-ms", type=int, default=100)
    parser.add_argument("--memory-cap-mib", type=int, required=True)
    parser.add_argument("--matrix-size", type=int)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "active" and not 0 < args.duty_cycle <= 1:
        parser.error("--duty-cycle must be in (0, 1] for active mode")
    size = contention.matrix_size_for_cap(args.memory_cap_mib,
                                          args.matrix_size or contention.DEFAULT_MATRIX_SIZE)
    stopping = []
    previous = {stop_signal: signal.signal(stop_signal, lambda *_: stopping.append(True))
                for stop_signal in (signal.SIGTERM, signal.SIGINT)}
    try:
        return _run(args, size, stopping, backend_factory)
    finally:
        for stop_signal, handler in previous.items():
            signal.signal(stop_signal, handler)


def _run(args, size: int, stopping: list, backend_factory) -> int:
    with args.log.open("a") as log:
        def emit(event):
            log.write(json.dumps(event) + "\n")
            log.flush()

        params = {"mode": args.mode, "duty_cycle": args.duty_cycle if args.mode == "active" else 0.0,
                  "period_ms": args.period_ms, "memory_cap_mib": args.memory_cap_mib,
                  "matrix_size": size}
        backend = backend_factory(size, args.memory_cap_mib)
        emit({"event": "start", "t_utc": contention.utc_now(), "pid": os.getpid(),
              **params, **getattr(backend, "info", {})})
        emit({"event": "ready", "t_utc": contention.utc_now()})
        controller = contention.DutyCycleController(params["duty_cycle"], args.period_ms / 1000,
                                                    backend.step)
        totals = controller.run(lambda: bool(stopping), emit)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        emit({"event": "stop", "t_utc": contention.utc_now(), "total_steps": totals["steps"],
              "achieved_duty": totals["achieved_duty"], "kernel_duty": totals["kernel_duty"],
              "elapsed_s": totals["elapsed_s"], "ru_utime": usage.ru_utime, "ru_stime": usage.ru_stime})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
