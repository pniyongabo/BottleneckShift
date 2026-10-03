import importlib.util
from pathlib import Path

import pytest

from bottleneckshift import contention

spec = importlib.util.spec_from_file_location("gpu_contention", Path("scripts/gpu_contention.py"))
gpu_contention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_contention)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def run_controller(duty, periods=50, step_s=0.002, period_s=0.1):
    clock, events = FakeClock(), []

    def step():
        clock.now += step_s
        return step_s * 0.9

    controller = contention.DutyCycleController(duty, period_s, step, clock=clock, sleep=clock.sleep)
    totals = controller.run(lambda: clock.now >= periods * period_s - 1e-9, events.append)
    return totals, events


@pytest.mark.parametrize("duty", [0.25, 0.5, 1.0])
def test_controller_reaches_target_duty_within_one_step(duty):
    totals, events = run_controller(duty)
    assert totals["periods"] == 50
    assert abs(totals["achieved_duty"] - duty) <= 0.002 / 0.1 + 1e-9
    assert totals["kernel_duty"] == pytest.approx(totals["achieved_duty"] * 0.9, rel=0.05)
    assert sum(event["steps"] for event in events) == totals["steps"]
    assert {event["event"] for event in events} == {"window"}


def test_controller_sham_runs_no_steps():
    totals, _ = run_controller(0.0)
    assert totals["steps"] == 0 and totals["achieved_duty"] == 0.0
    assert totals["elapsed_s"] == pytest.approx(5.0)


def test_matrix_size_and_ballast_fit_the_cap():
    assert contention.matrix_size_for_cap(1024) == 2048
    assert contention.buffer_bytes(2048) == 24 * contention.MIB
    assert contention.ballast_bytes(1024, 2048) == (512 - 24) * contention.MIB
    assert contention.matrix_size_for_cap(530) == 1536  # 18 MiB budget; 1792 needs 19.3
    with pytest.raises(ValueError, match="context reserve"):
        contention.matrix_size_for_cap(512)


def test_validate_log_checks_events_mode_and_duty():
    events = [{"event": "start", "mode": "active"}, {"event": "ready"}, {"event": "window"},
              {"event": "stop", "achieved_duty": 0.48, "kernel_duty": 0.4, "total_steps": 10}]
    assert contention.validate_log(events, "active", 0.5)["windows"] == 1
    with pytest.raises(ValueError, match="differs"):
        contention.validate_log(events, "active", 0.6)
    with pytest.raises(ValueError, match="mode"):
        contention.validate_log(events, "sham", None)
    with pytest.raises(ValueError, match="no stop"):
        contention.validate_log(events[:3], "active", 0.5)
    sham = [{"event": "start", "mode": "sham"}, {"event": "ready"},
            {"event": "stop", "total_steps": 3}]
    with pytest.raises(ValueError, match="sham"):
        contention.validate_log(sham, "sham", None)


def test_cli_writes_start_ready_window_stop_with_fake_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(contention.DutyCycleController, "run",
                        lambda self, should_stop, emit: (emit({"event": "window", "steps": 4}),
                                                         {"steps": 4, "achieved_duty": 0.5,
                                                          "kernel_duty": 0.4, "elapsed_s": 1.0})[1])

    class Backend:
        info = {"device_name": "fake"}

        def __init__(self, size, cap):
            assert (size, cap) == (2048, 1024)

        def step(self):
            return 0.001

    log = tmp_path / "contention.jsonl"
    assert gpu_contention.main(["--mode", "active", "--duty-cycle", "0.5", "--memory-cap-mib", "1024",
                                "--log", str(log)], backend_factory=Backend) == 0
    events = contention.read_log(log.read_text())
    assert [event["event"] for event in events] == ["start", "ready", "window", "stop"]
    assert events[0]["matrix_size"] == 2048 and events[0]["device_name"] == "fake"
    assert contention.validate_log(events, "active", 0.5)["achieved_duty"] == 0.5


def fake_torch(free_mib, total_mib=15352):
    import types
    allocated = []
    cuda = types.SimpleNamespace(
        init=lambda: None, mem_get_info=lambda: (free_mib * contention.MIB, total_mib * contention.MIB),
        set_per_process_memory_fraction=lambda fraction: allocated.append(("fraction", fraction)),
        get_device_name=lambda index: "fake", memory_allocated=lambda: 0,
        Event=lambda **kwargs: types.SimpleNamespace(record=lambda: None, synchronize=lambda: None,
                                                     elapsed_time=lambda other: 1.0))
    def tensor(*args, **kwargs):
        allocated.append(("tensor", args))
        return object()
    return types.SimpleNamespace(cuda=cuda, randn=tensor, empty=tensor, matmul=lambda *a, **k: None,
                                 float16="fp16", uint8="u8", __version__="fake"), allocated


@pytest.mark.parametrize("free_mib, fits", [(612, True), (575, False)])
def test_cuda_backend_checks_only_the_tensor_budget_after_context_creation(monkeypatch, free_mib, fits):
    # Seen on the A4000 next to vLLM at 0.90: 612 MiB free after the generator's own context.
    torch, allocated = fake_torch(free_mib)
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    if fits:
        backend = gpu_contention.CudaBackend(2048, 1024)
        assert backend.info["ballast_bytes"] == (512 - 24) * contention.MIB
    else:
        with pytest.raises(RuntimeError, match="tensors need 512 MiB"):
            gpu_contention.CudaBackend(2048, 1024)
