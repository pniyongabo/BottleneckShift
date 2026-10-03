#!/usr/bin/env python3
"""Milestone 3 analysis: client latency, server-side phases, and resource telemetry.

Every series is validated first. Per-run rows go to stdout as CSV; `--summary` writes
medians and ranges per regime × order × condition, and `--output` writes the combined
figure (requires matplotlib: `pip install -e '.[analysis]'`).
"""

import argparse
import csv
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bottleneckshift import prometheus, telemetry  # noqa: E402
from bottleneckshift.validation import validate_series  # noqa: E402

_spec = importlib.util.spec_from_file_location("compare", Path(__file__).with_name("compare.py"))
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

SERVER = {"queue": "vllm:request_queue_time_seconds", "prefill": "vllm:request_prefill_time_seconds",
          "decode": "vllm:request_decode_time_seconds", "inference": "vllm:request_inference_time_seconds",
          "ttft": "vllm:time_to_first_token_seconds", "itl": "vllm:inter_token_latency_seconds"}
SUMMARY_METRICS = ("ttft_ms", "tpot_ms", "e2el_ms", "output_token_throughput", "client_pre_first_token_share",
                   "server_queue_ms", "server_prefill_ms", "server_decode_ms", "server_pre_first_token_share",
                   "gpu_util_mean", "cpu_util_mean", "cpu_core_util_max")


def regime_of(manifest: dict) -> str:
    name = manifest["plan_name"]
    for regime in ("prefill_heavy", "decode_heavy"):
        if regime in name:
            return regime
    workload = manifest["config"]["workload"]
    return f"{workload['input_tokens']}in_{workload['output_tokens']}out"


def order_of(manifest: dict) -> str:
    name = manifest["plan_name"]
    if name.endswith("-forward") or name.endswith("-reverse"):
        return name.rsplit("-", 1)[1]
    return "-".join(manifest["condition_order"])


def contention_of(manifest: dict) -> str:
    table = manifest["config"].get("contention", {"mode": "off"})
    return f"active-{table['duty_cycle']:g}" if table["mode"] == "active" else table["mode"]


def client_means(document: dict) -> dict:
    requests = compare.successful_requests(document)
    ttft = sum(request["ttft"] for request in requests)
    e2el = sum(request["e2el"] for request in requests)
    itls = [itl for request in requests for itl in request["itls"]]
    return {"client_ttft_mean_ms": ttft / len(requests) * 1000 if requests else None,
            "client_e2el_mean_ms": e2el / len(requests) * 1000 if requests else None,
            "client_itl_mean_ms": sum(itls) / len(itls) * 1000 if itls else None,
            "client_pre_first_token_share": ttft / e2el if e2el else None}


def server_row(directory: Path) -> dict:
    summary = prometheus.summarize_pair((directory / "metrics-before.prom").read_text(),
                                        (directory / "metrics-after.prom").read_text())
    means = {key: summary["families"][family]["mean_s"] for key, family in SERVER.items()}
    row = {f"server_{key}_ms": value * 1000 if value is not None else None for key, value in means.items()}
    row["server_requests"] = summary["server_requests"]
    queue, prefill, decode, inference = (means[key] or 0.0 for key in ("queue", "prefill", "decode", "inference"))
    total = queue + inference
    row["server_pre_first_token_share"] = (queue + prefill) / total if total else None
    row["dominant_component"] = "pre_first_token" if queue + prefill > decode else "decode"
    row["largest_phase"] = max((("queue", queue), ("prefill", prefill), ("decode", decode)),
                               key=lambda item: item[1])[0]
    return row


def run_rows(root: Path) -> list[dict]:
    validate_series(root)  # refuse to analyze anything that fails validation
    manifest = json.loads((root / "manifest.json").read_text())
    enabled = manifest["config"].get("telemetry", {})
    samples = {}
    if enabled.get("gpu_sampler"):
        samples["gpu"] = telemetry.parse_gpu_csv((root / "gpu.csv").read_text())
    if enabled.get("cpu_sampler"):
        samples["cpu"] = telemetry.parse_mpstat((root / "cpu.txt").read_text())
    rows = []
    for run in manifest["runs"]:
        if run["measurement_role"] != "measured":
            continue
        directory = root / run["condition"] / f"rep-{run['repetition']:02d}"
        document = json.loads((directory / "requests.json").read_text())
        client = compare.summarize(directory / "requests.json")
        row = {"series_id": manifest["series_id"], "regime": regime_of(manifest), "order": order_of(manifest),
               "contention": contention_of(manifest), "condition": run["condition"], "max_concurrency": run["max_concurrency"],
               "repetition": run["repetition"],
               **{key: client[key] for key in ("requests", "ttft_ms", "tpot_ms", "e2el_ms", "itl_ms",
                                               "vllm_median_e2el_ms", "output_token_throughput")},
               **client_means(document)}
        if enabled.get("server_metrics"):
            row.update(server_row(directory))
            row["ttft_gap_ms"] = row["client_ttft_mean_ms"] - (row["server_queue_ms"] + row["server_prefill_ms"])
            row["itl_gap_ms"] = (row["client_itl_mean_ms"] - row["server_itl_ms"]
                                 if row["client_itl_mean_ms"] is not None else None)
        # The benchmark window ends when the client finishes; its length is vLLM's duration.
        end = datetime.fromisoformat(run["finished_at_utc"])
        start = end - timedelta(seconds=document["duration"])
        for kind, kind_samples in samples.items():
            stats = telemetry.window_stats(kind_samples, start, end, kind)
            row.update({key: value for key, value in stats.items() if key != "samples"})
            row[f"{kind}_samples"] = stats["samples"]
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    # One cell per series and condition, so repeated designs (e.g. contention off before
    # and after) stay separate; series order follows the input.
    groups = {}
    for row in rows:
        groups.setdefault((row["series_id"], row["condition"]), []).append(row)
    summary = []
    for (series_id, condition), group in groups.items():
        first = group[0]
        entry = {"series_id": series_id, "regime": first["regime"], "order": first["order"],
                 "contention": first["contention"], "condition": condition,
                 "max_concurrency": first["max_concurrency"], "repetitions": len(group)}
        for metric in SUMMARY_METRICS:
            values = [row[metric] for row in group if row.get(metric) is not None]
            entry[f"{metric}_median"] = statistics.median(values) if values else None
            entry[f"{metric}_min"] = min(values, default=None)
            entry[f"{metric}_max"] = max(values, default=None)
        entry["dominant_components"] = "/".join(sorted({row.get("dominant_component", "") for row in group}))
        summary.append(entry)
    return summary


def write_figure(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({row["regime"] for row in rows})
    figure, axes = plt.subplots(len(regimes), 3, figsize=(13, 3.6 * len(regimes)), squeeze=False)
    phases = (("server_queue_ms", "queue", "#8c8c8c"), ("server_prefill_ms", "prefill", "#1665d8"),
              ("server_decode_ms", "decode", "#d45d00"))
    markers = {"forward": "o", "reverse": "s"}
    for row_axes, regime in zip(axes, regimes):
        subset = [row for row in rows if row["regime"] == regime]
        cells = sorted({(row["max_concurrency"], row["order"]) for row in subset})
        labels = [f"C{concurrency}\n{order}" for concurrency, order in cells]
        bottoms = [0.0] * len(cells)
        for key, label, color in phases:
            heights = [statistics.median([row[key] for row in subset if (row["max_concurrency"], row["order"]) == cell
                                          and row.get(key) is not None] or [0.0]) for cell in cells]
            row_axes[0].bar(labels, heights, bottom=bottoms, color=color, label=label)
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        for index, cell in enumerate(cells):
            group = [row for row in subset if (row["max_concurrency"], row["order"]) == cell]
            row_axes[0].scatter([index] * len(group), [row["client_e2el_mean_ms"] for row in group],
                                color="black", marker="x", zorder=3, label="client E2E" if index == 0 else None)
            row_axes[0].scatter([index] * len(group), [row["client_ttft_mean_ms"] for row in group],
                                color="black", marker="_", s=120, zorder=3,
                                label="client TTFT" if index == 0 else None)
        row_axes[0].set(title=f"{regime}: server phases (median) and client means", ylabel="ms")
        row_axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
        for column, (key, title) in ((1, ("output_token_throughput", "output tokens/s")),
                                     (2, ("gpu_util_mean", "GPU utilization (%)"))):
            for order, marker in markers.items():
                group = [row for row in subset if row["order"] == order and row.get(key) is not None]
                row_axes[column].scatter([f"C{row['max_concurrency']}" for row in group], [row[key] for row in group],
                                         marker=marker, label=order, alpha=0.8)
            row_axes[column].set(title=f"{regime}: {title}")
            row_axes[column].legend(fontsize=7)
    figure.suptitle("Milestone 3: each marker is one repetition; bars are medians across repetitions")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def write_csv(rows: list[dict], handle) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("series", type=Path, nargs="+", help="series directories containing manifest.json")
    parser.add_argument("--summary", type=Path, help="write per-cell medians and ranges to this CSV")
    parser.add_argument("--output", type=Path, help="write the combined figure (.svg or .png)")
    args = parser.parse_args()
    rows = [row for root in args.series for row in run_rows(root)]
    write_csv(rows, sys.stdout)
    if args.summary:
        with args.summary.open("w", newline="") as handle:
            write_csv(summarize(rows), handle)
    if args.output:
        write_figure(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
