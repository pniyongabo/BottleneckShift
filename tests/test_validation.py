import json
from pathlib import Path

import pytest

from bottleneckshift.validation import validate_series


def synthetic_series(tmp_path: Path, *, output_lens=(4, 4)) -> Path:
    root = tmp_path / "synthetic-series"
    result = root / "baseline" / "rep-01" / "requests.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({
        "ttfts": [0.1, 0.2], "itls": [[0.01] * 3, [0.02] * 3],
        "input_lens": [3, 3], "output_lens": list(output_lens), "errors": ["", ""],
    }))
    manifest = {
        "status": "complete", "series_id": "synthetic-series",
        "config": {"experiment": {"repetitions": 1}, "workload": {"num_prompts": 2},
                   "condition": [{"name": "baseline", "max_concurrency": 1}]},
        "runs": [{"measurement_role": "measured", "condition": "baseline", "repetition": 1,
                  "returncode": 0, "raw_result": str(result), "requested_input_tokens": 3,
                  "requested_output_tokens": 4}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_validate_series_accepts_complete_synthetic_fixture(tmp_path):
    summary = validate_series(synthetic_series(tmp_path))
    assert summary["measured_runs"] == 1
    assert summary["measured_requests"] == 2


def test_validate_series_rejects_wrong_token_length(tmp_path):
    with pytest.raises(ValueError, match="output_lens differs"):
        validate_series(synthetic_series(tmp_path, output_lens=(4, 3)))


def mutate(root: Path, *, manifest=None, result=None) -> Path:
    path = root / "manifest.json"
    document = json.loads(path.read_text())
    if manifest:
        manifest(document)
    path.write_text(json.dumps(document))
    result_path = root / "baseline" / "rep-01" / "requests.json"
    document = json.loads(result_path.read_text())
    if result:
        result(document)
    result_path.write_text(json.dumps(document))
    return root


@pytest.mark.parametrize("manifest, result, message", [
    (lambda m: m.update(status="failed"), None, "status is not complete"),
    (lambda m: m["runs"].clear(), None, "found 0 measured runs"),
    (None, lambda r: r.pop("itls"), "missing request array itls"),
    (None, lambda r: r["ttfts"].append(0.3), "ttfts has 3 entries"),
    (lambda m: m["runs"][0].update(returncode=1), None, "nonzero return code"),
    (None, lambda r: r.update(errors=["", "timeout"]), "non-empty errors at request indexes \\[1\\]"),
    (lambda m: (m["config"]["experiment"].update(repetitions=2), m["runs"].append(dict(m["runs"][0]))),
     None, "duplicate measured run"),
    (None, lambda r: r.update(input_lens=[3, 2]), "input_lens differs"),
])
def test_validate_series_rejects_invalid_series(tmp_path, manifest, result, message):
    with pytest.raises(ValueError, match=message):
        validate_series(mutate(synthetic_series(tmp_path), manifest=manifest, result=result))


import importlib.util  # noqa: E402

from synthetic import run_series, small_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_experiment", Path("scripts/run_experiment.py"))
run_experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_experiment)

ACTIVE = '\n[contention]\nmode = "active"\nduty_cycle = 0.5\nmemory_cap_mib = 1024\n'


def phase_b_series(tmp_path, monkeypatch, *, contention="", metrics_offset=0):
    source = small_config(Path("configs/milestone3-prefill_heavy-forward.toml").read_text(),
                          contention=contention)
    return run_series(tmp_path, monkeypatch, run_experiment, source, metrics_offset=metrics_offset)


def test_schema4_series_validates_warmups_metrics_telemetry_and_contention(tmp_path, monkeypatch):
    summary = validate_series(phase_b_series(tmp_path, monkeypatch, contention=ACTIVE))
    assert summary["measured_runs"] == 2 and summary["warmup_runs"] == 2
    assert summary["server_requests"] == 2 * 2 + 2 * 4
    assert set(summary["telemetry"]) == {"gpu", "cpu"}
    assert summary["contention"]["achieved_duty"] == 0.5


def test_schema4_rejects_server_count_that_differs_from_prompts(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="expected 2 requests"):
        validate_series(phase_b_series(tmp_path, monkeypatch, metrics_offset=1))


def edit_manifest(root, change):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("change, message", [
    (lambda m: m["runs"].pop(0), "found 1 warm-up runs"),
    (lambda m: m["telemetry"]["cpu"].update(exited_early=True), "cpu sampler failed"),
    (lambda m: m["runs"][2]["server_metrics"].pop("after"), "server metrics incomplete"),
    (lambda m: m["runs"][3]["command"].__setitem__(-1, "other-prefix-"), "unexpected request-id prefix"),
    (lambda m: m["contention"].update(vram_mib=1500), "contention VRAM"),
    (lambda m: m["contention"].update(returncode=-9), "did not exit cleanly"),
])
def test_schema4_rejects_incomplete_instrumentation(tmp_path, monkeypatch, change, message):
    root = phase_b_series(tmp_path, monkeypatch, contention=ACTIVE)
    edit_manifest(root, change)
    with pytest.raises(ValueError, match=message):
        validate_series(root)


def test_schema4_rejects_server_restart_bad_warmup_gap_and_duty(tmp_path, monkeypatch):
    root = phase_b_series(tmp_path, monkeypatch, contention=ACTIVE)
    after = root / "baseline_c1" / "rep-01" / "metrics-after.prom"
    original = after.read_text()
    after.write_text(original.replace("process_start_time_seconds 1000.0", "process_start_time_seconds 2000.0"))
    with pytest.raises(ValueError, match="restarted"):
        validate_series(root)
    after.write_text(original)

    warmup = root / "warmup" / "c8" / "requests.json"
    document = json.loads(warmup.read_text())
    document["output_lens"][0] = 31
    warmup.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="output_lens differs"):
        validate_series(root)
    document["output_lens"][0] = 32
    warmup.write_text(json.dumps(document))

    gpu = root / "gpu.csv"
    lines = gpu.read_text().splitlines()
    gpu.write_text("\n".join(lines[:3] + lines[20:]) + "\n")
    with pytest.raises(ValueError, match="gap"):
        validate_series(root)
    gpu.write_text("\n".join(lines) + "\n")

    log = root / "contention.jsonl"
    log.write_text(log.read_text().replace('"achieved_duty": 0.5', '"achieved_duty": 0.3'))
    with pytest.raises(ValueError, match="differs"):
        validate_series(root)
