#!/usr/bin/env python3
"""Milestone 5 Part 2: the context-length test and the executor-timing check.

`context` mode reads the two C1 context series (budgets 1024 and 2048) and reports, per
series: whether every prefill step is one full chunk at the expected positions
(prediction 11), the median execution time of full chunks by context position, the
deepest/first ratio and the Spearman correlation of context with execution time
(prediction 12), and the context-0 chunk-size ratio (prediction 13). `--figure` draws
execution time against context position.

`timing` mode reports prediction 14 over the given series: the Spearman correlation of
a step's tokens with its execution time, and with its schedule interval.
Definitions follow the Part 2 addendum in experiments/milestone5.md.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bottleneckshift.validation import validate_series  # noqa: E402
from bottleneckshift.vllm_steplog import read_steps  # noqa: E402
from milestone5_steps import execution_times, run_steps, spearman  # noqa: E402

PROMPT_TOKENS = 8192


def measured_windows(root: Path) -> list[tuple[float, float]]:
    manifest = json.loads((root / "manifest.json").read_text())
    return [(datetime.fromisoformat(run["started_at_utc"]).timestamp(),
             datetime.fromisoformat(run["finished_at_utc"]).timestamp())
            for run in manifest["runs"] if run.get("measurement_role") == "measured"]


def measured_steps(root: Path) -> tuple[list[dict], dict[int, float]]:
    steps = read_steps((root / "steps.jsonl").read_text())
    windows = measured_windows(root)
    inside = [step for step in steps if any(start <= step["t_start"] <= end for start, end in windows)]
    return inside, execution_times(steps)


def context_summary(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    budget = manifest["config"]["server"]["max_num_batched_tokens"]
    steps, times = measured_steps(root)
    prefill = [step for step in steps if step["prefill_tokens"]]
    full = [step for step in prefill if step["prefill_requests"] == 1 and len(step["chunks"]) == 1
            and step["chunks"][0][1] == budget]
    expected_positions = list(range(0, PROMPT_TOKENS, budget))
    by_position: dict[int, list[float]] = {}
    for step in full:
        by_position.setdefault(step["chunks"][0][0], []).append(times[step["step"]] * 1000)
    medians = {position: statistics.median(values) for position, values in sorted(by_position.items())}
    return {
        "series_id": manifest["series_id"], "budget": budget,
        "prefill_steps": len(prefill), "full_chunks": len(full),
        "positions_as_expected": sorted(by_position) == expected_positions,
        "chunks_per_position": sorted({len(values) for values in by_position.values()}),
        "steps_without_output_time": sum(1 for step in steps if "t_output" not in step),
        "median_ms_by_context": medians,
        "deepest_over_first": medians[expected_positions[-1]] / medians[0],
        "spearman_context_time": spearman([step["chunks"][0][0] for step in full],
                                          [times[step["step"]] for step in full]),
        "decode_median_ms": statistics.median(times[step["step"]] * 1000 for step in steps
                                              if not step["prefill_tokens"] and step["step"] in times),
    }


def timing_summary(roots: list[Path]) -> dict:
    tokens, executions, interval_tokens, intervals = [], [], [], []
    for root in roots:
        steps, times = measured_steps(root)
        for step in steps:
            if step.get("step") in times:
                tokens.append(step["tokens"])
                executions.append(times[step["step"]])
        all_steps = read_steps((root / "steps.jsonl").read_text())
        for start, end in measured_windows(root):
            for count, interval, _ in run_steps(all_steps, start, end):
                interval_tokens.append(count)
                intervals.append(interval)
    return {"steps_timed": len(tokens), "spearman_tokens_execution": spearman(tokens, executions),
            "spearman_tokens_interval": spearman(interval_tokens, intervals)}


def context_figure(summaries: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for summary, color in zip(summaries, ("#1665d8", "#d45d00")):
        positions = list(summary["median_ms_by_context"])
        axis.plot(positions, list(summary["median_ms_by_context"].values()), color=color, marker="o",
                  label=f"{summary['budget']}-token chunks (B{summary['budget']})")
    axis.set(xlabel="context position of the chunk (tokens already computed)",
             ylabel="median execution time (ms)", ylim=(0, None),
             title="Prefill chunk cost by context, one request at a time")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150, **({"metadata": {"CreationDate": None}} if output.suffix == ".pdf" else {}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("context", "timing"))
    parser.add_argument("series", type=Path, nargs="+")
    parser.add_argument("--figure", type=Path, help="context mode: write the figure (.png)")
    args = parser.parse_args()
    for root in args.series:
        validate_series(root)  # refuse incomplete or failing series
    if args.mode == "timing":
        print(json.dumps(timing_summary(args.series), indent=2))
        return 0
    summaries = [context_summary(root) for root in args.series]
    by_budget = {summary["budget"]: summary for summary in summaries}
    result = {"series": summaries}
    if {1024, 2048} <= set(by_budget):
        result["chunk_size_ratio_at_context_0"] = (by_budget[2048]["median_ms_by_context"][0]
                                                   / by_budget[1024]["median_ms_by_context"][0])
    if args.figure:
        context_figure(sorted(summaries, key=lambda s: s["budget"]), args.figure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
