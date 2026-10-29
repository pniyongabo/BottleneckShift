#!/usr/bin/env python3
"""Milestone 5 pilot: per-cell latency and per-step scheduler metrics.

For each series (one prompt-length × budget cell), reads the measured runs' vLLM
results and the plugin's `steps.jsonl`, and writes one CSV row per series with:
- the medians across repetitions of ITL p50 and p99 (vLLM's), mean E2E, and output
  throughput;
- step metrics: steps per run, the share of steps carrying prefill, the median
  tokens of prefill-carrying steps, the largest step, and the median intervals of
  large (>= 1024 tokens) and small (<= 64 tokens) steps.

`--pooled` prints the block-level step metrics of prediction 2: the large/small
median-interval ratio and the Spearman correlation between a step's tokens and its
interval, pooled over the given series. Definitions follow experiments/milestone5.md.
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bottleneckshift.validation import validate_series  # noqa: E402
from bottleneckshift.vllm_steplog import read_steps  # noqa: E402

LARGE, SMALL, IDLE_SECONDS = 1024, 64, 1.0


def run_steps(steps: list[dict], start: float, end: float) -> list[tuple[int, float, int]]:
    """(tokens, interval in seconds, prefill tokens) for each step of one run.

    The interval of a step runs from its t_start to the next step's; the run's last
    step and intervals over IDLE_SECONDS are dropped.
    """
    window = sorted((step for step in steps if start <= step["t_start"] <= end), key=lambda s: s["t_start"])
    pairs = []
    for current, following in zip(window, window[1:]):
        interval = following["t_start"] - current["t_start"]
        if interval <= IDLE_SECONDS:
            pairs.append((current["tokens"], interval, current["prefill_tokens"]))
    return pairs


def execution_times(steps: list[dict]) -> dict[int, float]:
    """Execution time of each step with a recorded output time (Part 2 logs), in seconds.

    A step can start once it has been scheduled and the previous step's results have
    arrived, so its execution time is t_output[n] - max(t_end[n], t_output[n-1]).
    This holds whether or not the engine schedules the next step early.
    """
    timed = sorted((s for s in steps if "t_output" in s and "step" in s), key=lambda s: s["step"])
    result, previous = {}, None
    for record in timed:
        start = record["t_end"] if previous is None else max(record["t_end"], previous)
        result[record["step"]] = record["t_output"] - start
        previous = record["t_output"]
    return result


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        tie = position
        while tie + 1 < len(order) and values[order[tie + 1]] == values[order[position]]:
            tie += 1
        for index in order[position:tie + 1]:
            result[index] = (position + tie) / 2 + 1
        position = tie + 1
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    covariance = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    spread = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return covariance / spread if spread else float("nan")  # undefined when all values tie


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def series_row(root: Path) -> tuple[dict, list[tuple[int, float, int]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    steps = read_steps((root / "steps.jsonl").read_text())
    measured = [run for run in manifest["runs"] if run.get("measurement_role") == "measured"]
    latency, pairs, per_run_steps = [], [], []
    for run in measured:
        result = json.loads((root / run["condition"] / f"rep-{run['repetition']:02d}" / "requests.json").read_text())
        latency.append(result)
        start = datetime.fromisoformat(run["started_at_utc"]).timestamp()
        end = datetime.fromisoformat(run["finished_at_utc"]).timestamp()
        run_pairs = run_steps(steps, start, end)
        pairs += run_pairs
        per_run_steps.append(len(run_pairs) + 1)
    large = [interval for tokens, interval, _ in pairs if tokens >= LARGE]
    small = [interval for tokens, interval, _ in pairs if tokens <= SMALL]
    prefill = [tokens for tokens, _, prefill_tokens in pairs if prefill_tokens]
    row = {
        "series_id": manifest["series_id"],
        "input_tokens": config["workload"]["input_tokens"],
        "token_budget": config["server"]["max_num_batched_tokens"],
        "max_concurrency": config["condition"][0]["max_concurrency"],
        "itl_p50_ms": statistics.median(r["median_itl_ms"] for r in latency),
        "itl_p99_ms": statistics.median(r["p99_itl_ms"] for r in latency),
        "mean_e2el_ms": statistics.median(r["mean_e2el_ms"] for r in latency),
        "mean_ttft_ms": statistics.median(r["mean_ttft_ms"] for r in latency),
        "output_throughput": statistics.median(r["output_throughput"] for r in latency),
        "steps_per_run": statistics.median(per_run_steps),
        "prefill_step_share": len(prefill) / len(pairs),
        "prefill_step_tokens_median": median_or_none(prefill),
        "max_step_tokens": max(tokens for tokens, _, _ in pairs),
        "large_steps": len(large),
        "small_steps": len(small),
        "large_interval_ms": None if not large else statistics.median(large) * 1000,
        "small_interval_ms": None if not small else statistics.median(small) * 1000,
    }
    return row, pairs


def pooled(pairs: list[tuple[int, float, int]]) -> dict:
    large = [interval for tokens, interval, _ in pairs if tokens >= LARGE]
    small = [interval for tokens, interval, _ in pairs if tokens <= SMALL]
    return {"steps": len(pairs), "large_steps": len(large), "small_steps": len(small),
            "large_interval_ms": statistics.median(large) * 1000,
            "small_interval_ms": statistics.median(small) * 1000,
            "ratio": statistics.median(large) / statistics.median(small),
            "spearman": spearman([tokens for tokens, _, _ in pairs], [interval for _, interval, _ in pairs])}


def figure(rows: list[dict], series_pairs: dict[str, list], output: Path) -> None:
    """Left: ITL p50 and p99 by prompt length and budget. Right: step interval against
    scheduled tokens, one point per step."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {2048: "#1665d8", 8192: "#d45d00"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))
    for budget, color in colors.items():
        cells = sorted({row["input_tokens"]: row for row in rows if row["token_budget"] == budget
                        and not row["series_id"].endswith("anchor-after")}.values(), key=lambda r: r["input_tokens"])
        lengths = [row["input_tokens"] for row in cells]
        left.plot(lengths, [row["itl_p99_ms"] for row in cells], color=color, marker="o", label=f"B{budget} p99")
        left.plot(lengths, [row["itl_p50_ms"] for row in cells], color=color, marker="s", linestyle="--",
                  label=f"B{budget} p50")
    left.set(xscale="log", xticks=[1024, 2048, 4096], xticklabels=["1024", "2048", "4096"],
             xlabel="prompt length (tokens)", ylabel="ITL (ms)", title="ITL at C32 (median of 3 repetitions)")
    left.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    left.legend(fontsize=8)
    for row in rows:
        tokens, intervals = zip(*[(t, i * 1000) for t, i, _ in series_pairs[row["series_id"]]])
        right.scatter(tokens, intervals, s=4, alpha=0.25, color=colors[row["token_budget"]])
    for budget, color in colors.items():
        right.scatter([], [], color=color, label=f"B{budget} series")
    right.set(xlabel="scheduled tokens in the step", ylabel="step interval (ms)",
              title="Engine steps at C32 (all measured runs)")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("series", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="write one CSV row per series")
    parser.add_argument("--pooled", action="store_true", help="print prediction 2's pooled step metrics")
    parser.add_argument("--figure", type=Path, help="write the two-panel figure (.png)")
    args = parser.parse_args()
    rows, everything, series_pairs = [], [], {}
    for root in args.series:
        validate_series(root)  # refuse incomplete or failing series
        row, pairs = series_row(root)
        rows.append(row)
        everything += pairs
        series_pairs[row["series_id"]] = pairs
    if args.figure:
        figure(rows, series_pairs, args.figure)
    if args.output:
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"series": rows, **({"pooled": pooled(everything)} if args.pooled else {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
