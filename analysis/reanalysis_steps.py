#!/usr/bin/env python3
"""Post hoc reanalysis of the Milestone 5 step logs (not pre-registered).

`lag` mode, over Part 2 series (which log each step's output time): how often the
engine scheduled a step before the previous step's results arrived, and the Spearman
correlation of a step's schedule interval with its own execution time and with the
previous step's. It also checks a re-pairing for logs without output times: a step's
own cost estimated by the *next* step's interval. Over pilot series (no output times)
it reports prediction 2's block metrics under that re-pairing.

`execution` mode, over Part 2 C8 series: prediction 2's metrics computed with
execution time in place of the schedule interval, per series and pooled.

`fit` mode, over Part 2 series: a least-squares step-cost model,

    t = a + b * tokens + c * attention + e * decode_kv,

where `tokens` is every scheduled token, `attention` sums n * (context + n / 2) over a
step's prefill chunks (n tokens at context `context`), and `decode_kv` is the computed
tokens of the step's decoding requests. The coefficients are converted to effective
rates with the model's shapes below. Each series' measured and predicted execution
times are also turned into an ITL p50 and p99, weighting each step by its decode tokens,
and compared with vLLM's client-side ITL. `--train PATTERN` fits only on series whose
name contains PATTERN and evaluates on all, for an out-of-sample check.

`figure` mode, over Part 2 C8 series, draws two panels: each step's execution time
against its scheduled tokens, and each step's schedule interval against the execution
time of the same step and of the previous step.

Definitions: large and small steps are >= 1024 and <= 64 tokens (as in prediction 2);
schedule intervals over IDLE_SECONDS are dropped; execution time is
t_output[n] - max(t_end[n], t_output[n-1]) (see milestone5_steps.execution_times).
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bottleneckshift.validation import validate_series  # noqa: E402
from bottleneckshift.vllm_steplog import read_steps  # noqa: E402
from milestone5_part2 import measured_windows  # noqa: E402
from milestone5_steps import IDLE_SECONDS, LARGE, SMALL, execution_times, spearman  # noqa: E402

# Qwen2.5-0.5B-Instruct: 24 layers, 14 query heads of 64 dimensions, 2 KV heads, bf16.
LINEAR_FLOP_PER_TOKEN = 2 * 357_898_752  # two FLOPs per non-embedding weight
ATTENTION_FLOP_PER_PAIR = 4 * 14 * 64 * 24  # QK^T and AV, per query-key pair, all layers
KV_BYTES_PER_TOKEN = 2 * 2 * 64 * 2 * 24  # K and V, 2 heads of 64, 2 bytes, all layers


def run_sequences(root: Path) -> list[list[dict]]:
    """Each measured run's steps in order, from a series' step log."""
    steps = read_steps((root / "steps.jsonl").read_text())
    return [sorted((s for s in steps if start <= s["t_start"] <= end), key=lambda s: s["t_start"])
            for start, end in measured_windows(root)]


def intervals(run: list[dict]) -> list[float | None]:
    """Schedule interval of each step (None for the last step and idle gaps)."""
    result = [following["t_start"] - current["t_start"] for current, following in zip(run, run[1:])]
    return [value if value <= IDLE_SECONDS else None for value in result] + [None]


def block(tokens: list[int], costs: list[float]) -> dict:
    large = [cost for count, cost in zip(tokens, costs) if count >= LARGE]
    small = [cost for count, cost in zip(tokens, costs) if count <= SMALL]
    return {"steps": len(tokens), "large_ms": statistics.median(large) * 1000,
            "small_ms": statistics.median(small) * 1000,
            "ratio": statistics.median(large) / statistics.median(small),
            "spearman": spearman(tokens, costs)}


def lag_part2(roots: list[Path]) -> dict:
    rows = []
    for root in roots:
        times = execution_times(read_steps((root / "steps.jsonl").read_text()))
        early = total = 0
        own, previous, interval_own, next_tokens, next_interval, execution = [], [], [], [], [], []
        for run in run_sequences(root):
            gaps = intervals(run)
            for index, step in enumerate(run):
                if index and step["t_end"] < run[index - 1]["t_output"]:
                    early += 1
                total += 1
                if gaps[index] is not None and index:
                    own.append(times[step["step"]])
                    previous.append(times[run[index - 1]["step"]])
                    interval_own.append(gaps[index])
                if index + 1 < len(run) and gaps[index + 1] is not None:
                    next_tokens.append(step["tokens"])
                    next_interval.append(gaps[index + 1])
                    execution.append(times[step["step"]])
        rows.append({"series_id": root.name, "scheduled_early_share": early / total,
                     "spearman_interval_own_execution": spearman(interval_own, own),
                     "spearman_interval_previous_execution": spearman(interval_own, previous),
                     "execution": block(next_tokens, execution),
                     "next_interval": block(next_tokens, next_interval)})
    return {"series": rows}


def next_interval_pairs(run: list[dict]) -> tuple[list[int], list[float]]:
    """Each step's tokens, paired with the next step's schedule interval as its cost."""
    gaps = intervals(run)
    pairs = [(step["tokens"], gaps[index + 1]) for index, step in enumerate(run[:-1])
             if gaps[index + 1] is not None]
    return [tokens for tokens, _ in pairs], [cost for _, cost in pairs]


def lag_pilot(roots: list[Path]) -> dict:
    """Prediction 2's metrics with each step's cost taken as the next step's interval."""
    rows, tokens_all, costs_all = [], [], []
    for root in roots:
        tokens, costs = [], []
        for run in run_sequences(root):
            run_tokens, run_costs = next_interval_pairs(run)
            tokens += run_tokens
            costs += run_costs
        rows.append({"series_id": root.name, **block(tokens, costs)})
        tokens_all += tokens
        costs_all += costs
    return {"series": rows, "pooled": block(tokens_all, costs_all)}


def execution_block(roots: list[Path]) -> dict:
    rows, tokens_all, costs_all = [], [], []
    for root in roots:
        times = execution_times(read_steps((root / "steps.jsonl").read_text()))
        tokens, costs = [], []
        for run in run_sequences(root):
            for step in run:
                tokens.append(step["tokens"])
                costs.append(times[step["step"]])
        rows.append({"series_id": root.name, **block(tokens, costs)})
        tokens_all += tokens
        costs_all += costs
    return {"series": rows, "pooled": block(tokens_all, costs_all)}


def features(step: dict) -> list[float]:
    attention = sum(n * (context + n / 2) for context, n in step["chunks"])
    decode_kv = step["kv_tokens"] - sum(context for context, _ in step["chunks"])
    return [1.0, step["tokens"], attention, decode_kv]


def weighted_percentile(values: list[float], weights: list[int], q: float) -> float:
    order = sorted(range(len(values)), key=values.__getitem__)
    cumulative, total = 0, sum(weights)
    for index in order:
        cumulative += weights[index]
        if cumulative >= q * total:
            return values[index]
    return values[order[-1]]


def client_itl(root: Path) -> tuple[float, float]:
    manifest = json.loads((root / "manifest.json").read_text())
    results = [json.loads((root / run["condition"] / f"rep-{run['repetition']:02d}" / "requests.json").read_text())
               for run in manifest["runs"] if run.get("measurement_role") == "measured"]
    return (statistics.median(r["median_itl_ms"] for r in results),
            statistics.median(r["p99_itl_ms"] for r in results))


def fit(roots: list[Path], train: str | None = None) -> dict:
    data = {}
    for root in roots:
        times = execution_times(read_steps((root / "steps.jsonl").read_text()))
        data[root] = [(features(step), times[step["step"]], step["decode_tokens"])
                      for run in run_sequences(root) for step in run]
    training = {root: rows for root, rows in data.items() if train is None or train in root.name}
    x = numpy.array([f for rows in training.values() for f, _, _ in rows])
    y = numpy.array([t for rows in training.values() for _, t, _ in rows])
    coefficients, *_ = numpy.linalg.lstsq(x, y, rcond=None)
    predicted = x @ coefficients
    a, b, c, e = coefficients
    series = []
    for root, rows in data.items():
        xs = numpy.array([f for f, _, _ in rows])
        measured = [t for _, t, _ in rows]
        model = list(xs @ coefficients)
        weights = [w for _, _, w in rows]
        prefill = [(m, p) for (f, _, _), m, p in zip(rows, measured, model) if f[2] > 0]
        row = {"series_id": root.name, "steps": len(rows), "in_training": root in training,
               "median_abs_error_ms": statistics.median(abs(m - p) for m, p in zip(measured, model)) * 1000,
               "prefill_median_measured_ms": statistics.median(m for m, _ in prefill) * 1000 if prefill else None,
               "prefill_median_model_ms": statistics.median(p for _, p in prefill) * 1000 if prefill else None,
               "prefill_median_abs_error_pct": statistics.median(abs(p - m) / m for m, p in prefill) * 100
               if prefill else None}
        if sum(weights):
            client = client_itl(root)
            row.update({"itl_p50_client_ms": client[0], "itl_p99_client_ms": client[1],
                        "itl_p50_from_execution_ms": weighted_percentile(measured, weights, 0.5) * 1000,
                        "itl_p99_from_execution_ms": weighted_percentile(measured, weights, 0.99) * 1000,
                        "itl_p50_from_model_ms": weighted_percentile(model, weights, 0.5) * 1000,
                        "itl_p99_from_model_ms": weighted_percentile(model, weights, 0.99) * 1000})
        series.append(row)
    residual = y - predicted
    return {"steps": len(y), "train": train,
            "coefficients": {"fixed_ms": a * 1000, "per_token_us": b * 1e6, "per_attention_pair_ps": c * 1e12,
                             "per_decode_kv_token_ns": e * 1e9},
            "effective_rates": {"linear_tflops": LINEAR_FLOP_PER_TOKEN / b / 1e12,
                                "attention_tflops": ATTENTION_FLOP_PER_PAIR / c / 1e12,
                                "decode_kv_gb_per_s": KV_BYTES_PER_TOKEN / e / 1e9 if e > 0 else None},
            "r_squared": 1 - float(residual @ residual) / float(((y - y.mean()) ** 2).sum()),
            "series": series}


def lag_figure(roots: list[Path], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {2048: "#1665d8", 8192: "#d45d00"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))
    own, previous, gaps_all = [], [], []
    for root in roots:
        budget = json.loads((root / "manifest.json").read_text())["config"]["server"]["max_num_batched_tokens"]
        times = execution_times(read_steps((root / "steps.jsonl").read_text()))
        for run in run_sequences(root):
            left.scatter([s["tokens"] for s in run], [times[s["step"]] * 1000 for s in run],
                         s=4, alpha=0.25, color=colors[budget])
            gaps = intervals(run)
            for index in range(1, len(run)):
                if gaps[index] is not None:
                    own.append(times[run[index]["step"]] * 1000)
                    previous.append(times[run[index - 1]["step"]] * 1000)
                    gaps_all.append(gaps[index] * 1000)
    for budget, color in colors.items():
        left.scatter([], [], color=color, label=f"B{budget} series")
    left.set(xlabel="scheduled tokens in the step", ylabel="execution time (ms)",
             title="Engine steps at C8 by execution time (all measured runs)")
    left.legend(fontsize=8)
    right.scatter(own, gaps_all, s=4, alpha=0.25, color="#8a8a8a")
    right.scatter(previous, gaps_all, s=4, alpha=0.25, color="#1665d8")
    right.scatter([], [], color="#8a8a8a", label="against the step's own execution time")
    right.scatter([], [], color="#1665d8", label="against the previous step's execution time")
    limit = max(gaps_all + own) * 1.05
    right.plot([0, limit], [0, limit], color="#333333", linewidth=0.8, linestyle=":")
    right.set(xlabel="execution time (ms)", ylabel="schedule interval (ms)", xlim=(0, limit), ylim=(0, limit),
              title="A step's schedule interval tracks the previous step")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150, **({"metadata": {"CreationDate": None}} if output.suffix == ".pdf" else {}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("lag", "execution", "fit", "figure"))
    parser.add_argument("series", type=Path, nargs="+")
    parser.add_argument("--train", help="fit mode: fit only on series whose name contains this")
    parser.add_argument("--output", type=Path, help="figure mode: output file (.png or .pdf)")
    args = parser.parse_args()
    for root in args.series:
        validate_series(root)  # refuse incomplete or failing series
    if args.mode == "lag":
        timed = [root for root in args.series if '"t_output"' in (root / "steps.jsonl").read_text()]
        untimed = [root for root in args.series if root not in timed]
        result = {}
        if timed:
            result["part2"] = lag_part2(timed)
        if untimed:
            result["pilot_next_interval"] = lag_pilot(untimed)
    elif args.mode == "execution":
        result = execution_block(args.series)
    elif args.mode == "fit":
        result = fit(args.series, args.train)
    else:
        lag_figure(args.series, args.output)
        return 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
