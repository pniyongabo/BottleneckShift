#!/usr/bin/env python3
"""Post hoc reanalysis of retained client results: uncertainty, steady state, ITL
clusters, and GPU clocks. Not pre-registered; every number here is exploratory.

Definitions (all from the measured runs of validated series; warm-ups are excluded):
- A cell is one series × condition; it has three measured repetitions (runs).
- ITL p50 / p99 of a run: numpy's default (linear-interpolation) percentile of the
  run's pooled per-request `itls`, in ms. This reproduces vLLM's `median_itl_ms` and
  `p99_itl_ms` exactly (`check` reports the largest difference). Mean TTFT is the mean
  of `ttfts`; output throughput is vLLM's `output_throughput`.
- The reported value of a cell is the median of its three run values, as in the reports.
- E2E of a request is TTFT + sum(itls) (requests.json stores no per-request E2E); its
  finish time is `start_times` + E2E, on the client's monotonic clock.
- `uncertainty`: the run range (min–max) and two 95% bootstrap intervals (percentile
  method, 2,000 resamples, seed fixed per cell). Each resample draws each run's requests
  with replacement, keeping each request's whole ITL list. `ci_median_of_runs` is the
  interval of the reported statistic (median of the three resampled run values);
  `ci_pooled` is the interval of the statistic over the three resampled runs pooled,
  whose point estimate is `pooled`. Throughput gets the run range only.
- `steady`: per run, drop the first C requests by start time and the last C by finish
  time (C = the run's max concurrency; a request in both sets is dropped once), then
  recompute ITL p50, p99 and mean TTFT and take the median of runs. Relative change is
  steady / full − 1.
- `clusters`: per run, the share of ITL gaps above 20 ms, mean ITL, p50, p99 and mean
  per-request (E2E − TTFT) / (output_len − 1), each reported as the median of runs;
  `histogram` counts the gaps of all three runs in fixed ms bins.
- `clocks`: per measured run, GPU samples inside the manifest window [started_at_utc,
  finished_at_utc]: median, minimum and maximum SM clock, median and maximum power,
  and the 0% utilization samples (all, leading, and trailing). The manifest window
  brackets the whole benchmark subprocess, so it also holds client start-up and exit;
  `bench_*` repeat every statistic over the benchmark window [finished_at_utc − vLLM's
  duration, finished_at_utc] used by phases.py, and `busy_*` over samples with
  utilization > 0. A session is one archive; a run is flagged (`flagged`,
  `bench_flagged`) when its median SM clock in that window is more than 3% below the
  highest SM clock sampled anywhere in the session's GPU logs.
"""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import statistics
import sys
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bottleneckshift import telemetry  # noqa: E402
from bottleneckshift.validation import validate_series  # noqa: E402

SEED, RESAMPLES, LEVEL = 2027, 2000, 0.95
LONG_GAP_MS, CLOCK_DROP = 20.0, 0.03
BINS_MS = (0, 10, 20, 40, 80, 160, float("inf"))

# Session archives, relative to --archives; each holds results/runs/<series_id>.
SESSIONS = {
    "phase-b": ("20261003T180358Z", "milestone3-20261003T180358Z"),
    "phase-c": ("20261003T201214Z", "milestone3c-20261003T201214Z"),
    "phase-d": ("20261006T022744Z", "milestone3d-20261006T022744Z"),
    "milestone5": ("milestone5-20261025T004640Z/extract", "milestone5-20261025T004640Z"),
    "milestone5p2": ("milestone5p2-20261029T000043Z/extract", "milestone5p2-20261029T000043Z"),
}
# (session, series suffix, condition) for each cell analyzed.
TABLE1 = [("phase-d", f"budget-{budget}", "budget_c32") for budget in ("0512", "2048-before", "8192")]
TABLE2 = ([("milestone5", name, "c32") for name in
           ("len1024-budget2048-c32", "anchor-before", "len4096-budget2048-c32",
            "len1024-budget8192-c32", "len2048-budget8192-c32", "len4096-budget8192-c32")]
          + [("milestone5p2", f"len{length}-budget{budget}-c8", "c8")
             for budget in (2048, 8192) for length in (1024, 2048, 4096)])
CROSSOVER = [("phase-c", f"crossover-{order}", condition) for order in ("forward", "reverse")
             for condition in ("anchor_c8", "probe_c16", "probe_c32")]
SWEEP = [("phase-d", f"sweep-{order}", f"sweep_c{c}") for order in ("forward", "reverse")
         for c in (8, 12, 16, 24, 32)]
BUDGET = [("phase-d", f"budget-{budget}", f"budget_c{c}") for budget in ("0512", "2048-before", "8192",
                                                                      "2048-after") for c in (8, 32)]


def series_root(archives: Path, session: str, suffix: str) -> Path:
    directory, prefix = SESSIONS[session]
    return archives / directory / "BottleneckShift" / "results" / "runs" / f"{prefix}-{suffix}"


_validated: set[Path] = set()


def manifest_of(root: Path) -> dict:
    if root not in _validated:
        validate_series(root)  # refuse incomplete or failing series
        _validated.add(root)
    return json.loads((root / "manifest.json").read_text())


def load_run(root: Path, run: dict) -> dict:
    document = json.loads((root / run["condition"] / f"rep-{run['repetition']:02d}" / "requests.json").read_text())
    itls = [np.asarray(gaps, dtype=float) * 1000 for gaps in document["itls"]]
    ttft = np.asarray(document["ttfts"], dtype=float) * 1000
    decode = np.array([gaps.sum() for gaps in itls])
    start = np.asarray(document["start_times"], dtype=float)
    return {"document": document, "itls": itls, "ttft": ttft, "decode": decode, "start": start,
            "finish": start + (ttft + decode) / 1000, "output_lens": np.asarray(document["output_lens"]),
            "max_concurrency": run["max_concurrency"]}


def cell_runs(archives: Path, session: str, suffix: str, condition: str) -> list[dict]:
    root = series_root(archives, session, suffix)
    manifest = manifest_of(root)
    runs = [run for run in manifest["runs"]
            if run.get("measurement_role") == "measured" and run["condition"] == condition]
    if len(runs) != 3:
        raise ValueError(f"{root}: {len(runs)} measured runs for {condition}; expected 3")
    return [load_run(root, run) for run in sorted(runs, key=lambda r: r["repetition"])]


def label(session: str, suffix: str, condition: str) -> str:
    return f"{session}/{suffix}/{condition}"


def pooled_itls(itls: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(itls) if itls else np.array([])


def latency_stats(itls: list[np.ndarray], ttft: np.ndarray) -> dict:
    gaps = pooled_itls(itls)
    return {"itl_p50_ms": float(np.percentile(gaps, 50)), "itl_p99_ms": float(np.percentile(gaps, 99)),
            "mean_ttft_ms": float(ttft.mean())}


def median_of(values: list[float]) -> float:
    return float(statistics.median(values))


def bootstrap(runs: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    metrics = ("itl_p50_ms", "itl_p99_ms", "mean_ttft_ms")
    of_runs = {key: [] for key in metrics}
    pooled = {key: [] for key in metrics}
    for _ in range(RESAMPLES):
        samples = []
        for run in runs:
            picks = rng.integers(0, len(run["itls"]), len(run["itls"]))
            samples.append(([run["itls"][i] for i in picks], run["ttft"][picks]))
        per_run = [latency_stats(itls, ttft) for itls, ttft in samples]
        together = latency_stats([gaps for itls, _ in samples for gaps in itls],
                                 np.concatenate([ttft for _, ttft in samples]))
        for key in metrics:
            of_runs[key].append(median_of([stats[key] for stats in per_run]))
            pooled[key].append(together[key])
    tails = (100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2)
    return {key: {"ci_median_of_runs": [float(v) for v in np.percentile(of_runs[key], tails)],
                  "ci_pooled": [float(v) for v in np.percentile(pooled[key], tails)]} for key in metrics}


def uncertainty(archives: Path) -> dict:
    cells = []
    for session, suffix, condition in TABLE1 + TABLE2:
        name = label(session, suffix, condition)
        runs = cell_runs(archives, session, suffix, condition)
        per_run = [latency_stats(run["itls"], run["ttft"]) for run in runs]
        pooled = latency_stats([gaps for run in runs for gaps in run["itls"]],
                               np.concatenate([run["ttft"] for run in runs]))
        intervals = bootstrap(runs, zlib.crc32(f"{SEED}:{name}".encode()))
        entry = {"cell": name, "requests_per_run": [len(run["itls"]) for run in runs]}
        for key in ("itl_p50_ms", "itl_p99_ms", "mean_ttft_ms"):
            values = [stats[key] for stats in per_run]
            entry[key] = {"median": median_of(values), "runs": values, "range": [min(values), max(values)],
                          "pooled": pooled[key], **intervals[key]}
        throughput = [run["document"]["output_throughput"] for run in runs]
        entry["output_throughput"] = {"median": median_of(throughput), "runs": throughput,
                                      "range": [min(throughput), max(throughput)]}
        cells.append(entry)
    return {"subcommand": "uncertainty", "resamples": RESAMPLES, "level": LEVEL, "seed": SEED, "cells": cells}


def steady_subset(run: dict) -> np.ndarray:
    count = run["max_concurrency"]
    by_start, by_finish = np.argsort(run["start"], kind="stable"), np.argsort(run["finish"], kind="stable")
    dropped = set(by_start[:count].tolist()) | set(by_finish[len(by_finish) - count:].tolist())
    return np.array([i for i in range(len(run["itls"])) if i not in dropped], dtype=int)


def steady(archives: Path) -> dict:
    cells = []
    for session, suffix, condition in TABLE1 + TABLE2 + CROSSOVER:
        runs = cell_runs(archives, session, suffix, condition)
        full, kept, counts = [], [], []
        for run in runs:
            keep = steady_subset(run)
            counts.append(len(keep))
            full.append(latency_stats(run["itls"], run["ttft"]))
            kept.append(latency_stats([run["itls"][i] for i in keep], run["ttft"][keep]))
        entry = {"cell": label(session, suffix, condition), "max_concurrency": runs[0]["max_concurrency"],
                 "steady_requests_per_run": counts}
        for key in ("itl_p50_ms", "itl_p99_ms", "mean_ttft_ms"):
            before, after = median_of([s[key] for s in full]), median_of([s[key] for s in kept])
            entry[key] = {"full": before, "steady": after, "relative_change": after / before - 1,
                          "steady_runs": [s[key] for s in kept]}
        cells.append(entry)
    return {"subcommand": "steady", "cells": cells}


def histogram(gaps: np.ndarray) -> dict:
    counts, _ = np.histogram(gaps, bins=np.array(BINS_MS))
    names = [f"{low:g}-{high:g}" if high != float("inf") else f">{low:g}" for low, high in zip(BINS_MS, BINS_MS[1:])]
    return dict(zip(names, (int(count) for count in counts)))


def clusters(archives: Path) -> dict:
    cells = []
    for session, suffix, condition in SWEEP + BUDGET:
        runs = cell_runs(archives, session, suffix, condition)
        per_run = []
        for run in runs:
            gaps = pooled_itls(run["itls"])
            per_run.append({"long_gap_share": float((gaps > LONG_GAP_MS).mean()), "itl_mean_ms": float(gaps.mean()),
                            "itl_p50_ms": float(np.percentile(gaps, 50)),
                            "itl_p99_ms": float(np.percentile(gaps, 99)),
                            "tpot_mean_ms": float((run["decode"] / (run["output_lens"] - 1)).mean())})
        entry = {"cell": label(session, suffix, condition), "max_concurrency": runs[0]["max_concurrency"]}
        for key in per_run[0]:
            entry[key] = median_of([stats[key] for stats in per_run])
            entry[f"{key}_runs"] = [stats[key] for stats in per_run]
        entry["histogram"] = histogram(pooled_itls([gaps for run in runs for gaps in run["itls"]]))
        cells.append(entry)
    return {"subcommand": "clusters", "long_gap_ms": LONG_GAP_MS, "bins_ms": [str(b) for b in BINS_MS],
            "cells": cells}


def edge_count(flags: list[bool]) -> int:
    count = 0
    for flag in flags:
        if not flag:
            break
        count += 1
    return count


def window_clocks(inside: list[dict], prefix: str) -> dict:
    clock = [s["sm_clock_mhz"] for s in inside if s["sm_clock_mhz"] is not None]
    busy = [s["sm_clock_mhz"] for s in inside if s["sm_clock_mhz"] is not None and s["gpu_util"]]
    power = [s["power_w"] for s in inside if s["power_w"] is not None]
    idle = [s["gpu_util"] == 0 for s in inside]
    return {f"{prefix}{key}": value for key, value in {
        "samples": len(inside), "sm_clock_median_mhz": median_of(clock), "sm_clock_min_mhz": min(clock),
        "sm_clock_max_mhz": max(clock), "busy_samples": len(busy),
        "busy_sm_clock_median_mhz": median_of(busy) if busy else None,
        "busy_sm_clock_min_mhz": min(busy, default=None),
        "power_median_w": median_of(power), "power_max_w": max(power),
        "idle_samples": sum(idle), "idle_leading": edge_count(idle),
        "idle_trailing": edge_count(idle[::-1])}.items()}


def clocks(archives: Path) -> dict:
    sessions, skipped = [], []
    for session, (directory, prefix) in SESSIONS.items():
        runs_dir = archives / directory / "BottleneckShift" / "results" / "runs"
        rows, session_max = [], 0.0
        for root in sorted(runs_dir.glob(f"{prefix}-*")):
            if not (root / "gpu.csv").is_file():
                skipped.append({"series": root.name, "reason": "no gpu.csv"})
                continue
            samples = telemetry.parse_gpu_csv((root / "gpu.csv").read_text())
            session_max = max([session_max] + [s["sm_clock_mhz"] for s in samples if s["sm_clock_mhz"] is not None])
            try:
                manifest = manifest_of(root)
            except ValueError as exc:
                skipped.append({"series": root.name, "reason": f"validation failed: {exc}"})
                continue
            for run in manifest["runs"]:
                if run.get("measurement_role") != "measured":
                    continue
                start = datetime.fromisoformat(run["started_at_utc"])
                end = datetime.fromisoformat(run["finished_at_utc"])
                duration = json.loads((root / run["condition"] / f"rep-{run['repetition']:02d}"
                                       / "requests.json").read_text())["duration"]
                row = {"series": root.name.removeprefix(f"{prefix}-"), "condition": run["condition"],
                       "repetition": run["repetition"], "window_s": (end - start).total_seconds(),
                       "benchmark_duration_s": duration,
                       **window_clocks([s for s in samples if start <= s["t"] <= end], "")}
                # The benchmark window, as in phases.py: vLLM's duration, ending at finished_at_utc.
                bench_start = end - timedelta(seconds=duration)
                row.update(window_clocks([s for s in samples if bench_start <= s["t"] <= end], "bench_"))
                rows.append(row)
        session_bench_max = max(row["bench_sm_clock_max_mhz"] for row in rows)
        for row in rows:
            for prefix_ in ("", "bench_"):
                row[f"{prefix_}below_session_max"] = 1 - row[f"{prefix_}sm_clock_median_mhz"] / session_max
                row[f"{prefix_}flagged"] = row[f"{prefix_}below_session_max"] > CLOCK_DROP
        sessions.append({"session": session, "session_max_sm_clock_mhz": session_max,
                         "session_bench_max_sm_clock_mhz": session_bench_max, "runs": rows,
                         "flagged_runs": sum(row["flagged"] for row in rows),
                         "bench_flagged_runs": sum(row["bench_flagged"] for row in rows)})
    return {"subcommand": "clocks", "clock_drop_threshold": CLOCK_DROP, "sessions": sessions, "skipped": skipped}


def check(archives: Path) -> dict:
    """Largest |difference| between vLLM's saved summaries and the recomputation, over all
    measured runs of every session's valid series."""
    worst = {"median_itl_ms": 0.0, "p99_itl_ms": 0.0, "mean_ttft_ms": 0.0, "mean_itl_ms": 0.0}
    runs = 0
    for directory, prefix in SESSIONS.values():
        for root in sorted((archives / directory / "BottleneckShift" / "results" / "runs").glob(f"{prefix}-*")):
            try:
                manifest = manifest_of(root)
            except ValueError:
                continue
            for run in manifest["runs"]:
                if run.get("measurement_role") != "measured":
                    continue
                loaded = load_run(root, run)
                gaps, document = pooled_itls(loaded["itls"]), loaded["document"]
                ours = {"median_itl_ms": np.percentile(gaps, 50), "p99_itl_ms": np.percentile(gaps, 99),
                        "mean_ttft_ms": loaded["ttft"].mean(), "mean_itl_ms": gaps.mean()}
                for key, value in ours.items():
                    worst[key] = max(worst[key], abs(float(value) - document[key]))
                runs += 1
    return {"subcommand": "check", "percentile_method": "numpy linear (default)", "runs": runs,
            "max_abs_difference_ms": worst}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "uncertainty", "steady", "clusters", "clocks"))
    parser.add_argument("--archives", type=Path, default=REPO / "results" / "archives",
                        help="directory holding the extracted session archives")
    args = parser.parse_args()
    handlers = {"check": check, "uncertainty": uncertainty, "steady": steady, "clusters": clusters,
                "clocks": clocks}
    print(json.dumps(handlers[args.mode](args.archives), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
