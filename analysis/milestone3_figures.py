#!/usr/bin/env python3
"""Figures for the Milestone 3 Phase C and Phase D reports, regenerated from validated series.

Usage:
    python analysis/milestone3_figures.py RUNS_DIR PAIR_ID --output-dir reports/figures [--phase c|d]

RUNS_DIR holds the session's series directories, named `<PAIR_ID>-<name>`.
Phase C (default): crossover-forward, crossover-reverse, off-before, sham, active-025,
active-050, off-after. Phase D: sweep-forward, sweep-reverse, budget-2048-before,
budget-0512, budget-8192, budget-2048-after. Every series is validated before
plotting. Requires matplotlib (`pip install -e '.[analysis]'`).
"""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

_spec = importlib.util.spec_from_file_location("phases", Path(__file__).with_name("phases.py"))
phases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phases)

CROSSOVER = ("crossover-forward", "crossover-reverse", "off-before", "off-after")
CONTENTION = ("off-before", "sham", "active-025", "active-050", "off-after")
SWEEP = ("sweep-forward", "sweep-reverse")
BUDGETS = ("budget-2048-before", "budget-0512", "budget-8192", "budget-2048-after")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def rows_with_itl(root: Path) -> list[dict]:
    """phases.run_rows plus per-run ITL p50/p99 from the raw inter-token arrays."""
    rows = phases.run_rows(root)
    for row in rows:
        document = json.loads((root / row["condition"] / f"rep-{row['repetition']:02d}"
                               / "requests.json").read_text())
        intervals = [value for request in document["itls"] for value in request]
        row["itl_p50_ms"] = percentile(intervals, 0.5) * 1000
        row["itl_p99_ms"] = percentile(intervals, 0.99) * 1000
    return rows


def by_concurrency(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    levels = sorted({row["max_concurrency"] for row in rows})
    return levels, [statistics.median(row[key] for row in rows if row["max_concurrency"] == level)
                    for level in levels]


def crossover_figure(rows: list[dict], output: Path, plt) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = (("output_token_throughput", "Output throughput (tok/s)", False),
              ("itl_p50_ms", "Inter-token latency (ms)", True),
              ("client_pre_first_token_share", "Share of E2E before first token", False))
    for axis, (key, title, log) in zip(axes, panels):
        axis.scatter([row["max_concurrency"] for row in rows], [row[key] for row in rows],
                     color="#1665d8", alpha=0.6, s=18, label="repetition")
        levels, medians = by_concurrency(rows, key)
        axis.plot(levels, medians, color="#1665d8", label="median" if key != "itl_p50_ms" else "p50 median")
        if key == "itl_p50_ms":
            levels, p99 = by_concurrency(rows, "itl_p99_ms")
            axis.plot(levels, p99, color="#d45d00", linestyle="--", label="p99 median")
            axis.set_yscale("log")
        if key == "client_pre_first_token_share":
            axis.axhline(0.5, color="#8c8c8c", linestyle=":", linewidth=1, label="0.5 (pre-registered threshold)")
        axis.set(title=title, xlabel="Max client concurrency", xscale="log", xticks=[1, 8, 16, 32])
        axis.set_xticklabels(["C1", "C8", "C16", "C32"])
        axis.legend(fontsize=7)
    figure.suptitle("prefill_heavy (2048 in / 32 out): throughput saturates by C16; the extra latency moves "
                    "into per-step time, not before the first token", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def contention_figure(series_rows: dict[str, list[dict]], output: Path, plt) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))
    names = [name for name in CONTENTION if name in series_rows]
    labels = {"off-before": "off\n(before)", "sham": "sham", "active-025": "duty\n0.25",
              "active-050": "duty\n0.50", "off-after": "off\n(after)"}
    phases_ = (("server_queue_ms", "queue", "#8c8c8c"), ("server_prefill_ms", "prefill", "#1665d8"),
               ("server_decode_ms", "decode", "#d45d00"))
    for row_axes, condition, title in ((axes[0], "baseline_c1", "C1"), (axes[1], "perturbation_c8", "C8")):
        bottoms = [0.0] * len(names)
        for key, label, color in phases_:
            heights = [statistics.median(r[key] for r in series_rows[name] if r["condition"] == condition)
                       for name in names]
            row_axes[0].bar([labels[n] for n in names], heights, bottom=bottoms, color=color, label=label)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        for index, name in enumerate(names):
            values = [r["client_e2el_mean_ms"] for r in series_rows[name] if r["condition"] == condition]
            row_axes[0].scatter([index] * len(values), values, color="black", marker="x", zorder=3,
                                label="client E2E (mean)" if index == 0 else None)
            utilization = [r["gpu_util_mean"] for r in series_rows[name] if r["condition"] == condition]
            row_axes[1].scatter([labels[name]] * len(utilization), utilization, color="#258750")
        top = max(r["client_e2el_mean_ms"] for name in names for r in series_rows[name]
                  if r["condition"] == condition)
        row_axes[0].set(title=f"{title}: server phases (median) and client E2E", ylabel="ms", ylim=(0, top * 1.3))
        row_axes[0].legend(fontsize=7, loc="upper center", ncol=4)
        row_axes[1].set(title=f"{title}: nvidia-smi GPU utilization (per repetition)", ylabel="%", ylim=(80, 101))
    figure.suptitle("GPU contention (prefill_heavy): latency rises with duty cycle; sham and recovery match "
                    "the baseline; utilization barely moves", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def sweep_figure(rows: list[dict], output: Path, plt) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    levels = sorted({row["max_concurrency"] for row in rows})
    for axis, key, title in ((axes[0], "output_token_throughput", "Output throughput (tok/s)"),
                             (axes[2], "server_step_ms", "Server iteration time (ms)")):
        present = [r for r in rows if r.get(key) is not None]
        axis.set(title=title)
        if not present:  # the server did not expose vllm:iteration_tokens_total
            axis.text(0.5, 0.5, "not exposed by the server", ha="center", transform=axis.transAxes)
            continue
        axis.scatter([r["max_concurrency"] for r in present], [r[key] for r in present], color="#1665d8",
                     alpha=0.6, s=18, label="repetition")
        axis.plot(*by_concurrency(present, key), color="#1665d8", label="median")
    for key, color, style, label in (("itl_p50_ms", "#1665d8", "-", "p50"), ("itl_p99_ms", "#d45d00", "--", "p99")):
        axes[1].scatter([r["max_concurrency"] for r in rows], [r[key] for r in rows], color=color, alpha=0.6, s=18)
        axes[1].plot(*by_concurrency(rows, key), color=color, linestyle=style, label=f"{label} median")
    axes[1].axhline(10, color="#8c8c8c", linestyle=":", linewidth=1, label="10 / 20 ms (pre-registered)")
    axes[1].axhline(20, color="#8c8c8c", linestyle=":", linewidth=1)
    axes[1].set(title="Inter-token latency (ms)", yscale="log")
    for axis in axes:
        axis.set(xlabel="Max client concurrency", xticks=levels)
        axis.set_xticklabels([f"C{level}" for level in levels])
        axis.legend(fontsize=7)
    figure.suptitle("Phase D sweep (prefill_heavy, default 2048-token budget): per-step latency jumps between C16 "
                    "and C24 as throughput plateaus", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def budget_figure(series_rows: dict[str, list[dict]], output: Path, plt) -> None:
    names = [name for name in BUDGETS if name in series_rows]
    labels = {"budget-2048-before": "2048\n(before)", "budget-0512": "512", "budget-8192": "8192",
              "budget-2048-after": "2048\n(after)"}
    figure, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    panels = ((("itl_p50_ms", "p50", "#1665d8"), ("itl_p99_ms", "p99", "#d45d00")),
              (("client_ttft_mean_ms", "mean TTFT", "#258750"),),
              (("output_token_throughput", "output tok/s", "#7d3eb5"),))
    titles = ("Inter-token latency (ms, log)", "Mean TTFT (ms)", "Output throughput (tok/s)")
    for row_axes, condition, cond_label in ((axes[0], "budget_c8", "C8"), (axes[1], "budget_c32", "C32")):
        for axis, metrics, title in zip(row_axes, panels, titles):
            width = 0.8 / len(metrics)
            for offset, (key, label, color) in enumerate(metrics):
                xs = [index + (offset - (len(metrics) - 1) / 2) * width for index in range(len(names))]
                values = [[r[key] for r in series_rows[name] if r["condition"] == condition] for name in names]
                axis.bar(xs, [statistics.median(v) for v in values], width=width, color=color, label=label)
                for x, v in zip(xs, values):
                    axis.scatter([x] * len(v), v, color="black", s=8, zorder=3)
            axis.set_xticks(range(len(names)))
            axis.set_xticklabels([labels[name] for name in names])
            axis.set(title=f"{cond_label}: {title}")
            if len(metrics) > 1:
                axis.set_yscale("log")
                axis.legend(fontsize=7)
    figure.suptitle("Phase D token budget (max_num_batched_tokens): small budgets give frequent small stalls, "
                    "large budgets rare large ones", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("pair_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("c", "d"), default="c")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = CROSSOVER + CONTENTION if args.phase == "c" else SWEEP + BUDGETS
    series = {name: rows_with_itl(args.runs_dir / f"{args.pair_id}-{name}")
              for name in dict.fromkeys(names)
              if (args.runs_dir / f"{args.pair_id}-{name}").is_dir()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "c":
        crossover = [row for name in CROSSOVER if name in series for row in series[name]]
        crossover_figure(crossover, args.output_dir / "milestone3-phase-c-crossover.png", plt)
        contention_figure(series, args.output_dir / "milestone3-phase-c-contention.png", plt)
    else:
        sweep = [row for name in SWEEP if name in series for row in series[name]]
        sweep_figure(sweep, args.output_dir / "milestone3-phase-d-sweep.png", plt)
        budget_figure(series, args.output_dir / "milestone3-phase-d-budget.png", plt)
    print(f"wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
