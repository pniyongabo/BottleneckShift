#!/usr/bin/env python3
"""Summarize vLLM result JSON files and plot variation across repetitions."""

import argparse
import csv
import html
import json
import statistics
from pathlib import Path


def successful_requests(document: dict) -> list[dict]:
    """Return per-request TTFT, TPOT, E2E latency, and ITLs in seconds.

    vLLM's saved result holds per-request ``ttfts``, ``itls``, ``output_lens``,
    and ``errors`` lists but not per-request TPOT or E2E latency, so both are
    derived here: E2E is TTFT plus the request's ITLs, and TPOT matches vLLM's
    definition of decode time divided by ``output_len - 1``.
    """
    ttfts = document.get("ttfts", [])
    itls = document.get("itls", [])
    output_lens = document.get("output_lens", [])
    errors = document.get("errors", [])
    requests = []
    for index, ttft in enumerate(ttfts):
        if index < len(errors) and errors[index]:
            continue
        request_itls = [float(value) for value in (itls[index] if index < len(itls) else [])]
        decode = sum(request_itls)
        output_len = output_lens[index] if index < len(output_lens) else len(request_itls) + 1
        requests.append({
            "ttft": float(ttft),
            "tpot": decode / (output_len - 1) if output_len > 1 else None,
            "e2el": float(ttft) + decode,
            "itls": request_itls,
        })
    return requests


def median_ms(values: list[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.median(values) * 1000 if values else None


def summarize(path: Path) -> dict:
    document = json.loads(path.read_text())
    row = {"series_id": path.parents[2].name, "condition": path.parent.parent.name,
           "repetition": path.parent.name, "path": str(path)}
    requests = successful_requests(document)
    row["ttft_ms"] = median_ms([request["ttft"] for request in requests])
    row["tpot_ms"] = median_ms([request["tpot"] for request in requests])
    row["e2el_ms"] = median_ms([request["e2el"] for request in requests])
    row["itl_ms"] = median_ms([itl for request in requests for itl in request["itls"]])
    row["request_throughput"] = document.get("request_throughput")
    row["output_token_throughput"] = document.get("output_throughput")
    errors = document.get("errors")
    # vLLM records one entry per request; successful requests have an empty string.
    row["errors"] = sum(1 for error in errors if error) if isinstance(errors, list) else document.get("failed", 0)
    return row


def write_svg(rows: list[dict], output: Path) -> None:
    """Write a dependency-free dot plot; each dot is one repetition median."""
    metrics = [("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"), ("e2el_ms", "E2E latency")]
    conditions = sorted({row["condition"] for row in rows})
    colors = ["#1665d8", "#d45d00", "#258750", "#7d3eb5"]
    width, panel_height, margin = 900, 220, 80
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel_height * 3 + margin}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:sans-serif;fill:#222}.axis{stroke:#555}.grid{stroke:#ddd}</style>',
             '<text x="20" y="30" font-size="20">Client-observed run medians (each dot is one repetition)</text>']
    for panel, (metric, title) in enumerate(metrics):
        top = margin + panel * panel_height
        values = [row[metric] for row in rows if row[metric] is not None]
        maximum = max(values, default=1) or 1
        parts += [f'<text x="20" y="{top}" font-size="16">{title} (milliseconds)</text>',
                  f'<line class="axis" x1="180" y1="{top + 25}" x2="850" y2="{top + 25}"/>']
        for tick in range(6):
            x = 180 + tick * 134
            parts += [f'<line class="grid" x1="{x}" y1="{top + 20}" x2="{x}" y2="{top + 150}"/>',
                      f'<text x="{x}" y="{top + 170}" font-size="11" text-anchor="middle">{maximum * tick / 5:.1f}</text>']
        for index, condition in enumerate(conditions):
            y = top + 55 + index * 35
            parts.append(f'<text x="170" y="{y + 4}" font-size="12" text-anchor="end">{html.escape(condition)}</text>')
            for offset, row in enumerate(item for item in rows if item["condition"] == condition and item[metric] is not None):
                x = 180 + 670 * row[metric] / maximum
                parts.append(f'<circle cx="{x:.1f}" cy="{y + (offset % 3 - 1) * 5}" r="5" fill="{colors[index % len(colors)]}"/>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(path for path in args.results_dir.glob("**/requests.json")
                   if path.parent.name != "warmup")
    if not paths:
        parser.error(f"no requests.json files found below {args.results_dir}")
    rows = [summarize(path) for path in paths]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

    write_svg(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
