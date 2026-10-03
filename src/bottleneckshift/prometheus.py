"""Parse vLLM Prometheus scrapes and compute per-run histogram deltas.

The server stays alive across a series, so its histograms are cumulative; a run's
server-side timings are the difference between scrapes taken just before and just
after that run.
"""

from dataclasses import dataclass
import math

# Per-request histograms that every scrape must contain (vLLM 0.29.0).
REQUIRED_FAMILIES = (
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:time_to_first_token_seconds",
)
# Observed once per output token rather than once per request.
TOKEN_FAMILIES = ("vllm:inter_token_latency_seconds",)
REQUEST_COUNT_FAMILY = "vllm:request_queue_time_seconds"
START_TIME = "process_start_time_seconds"
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple
    value: float


def _parse_value(text: str) -> float:
    lowered = text.lower()
    if lowered in ("+inf", "inf"):
        return math.inf
    if lowered == "-inf":
        return -math.inf
    if lowered == "nan":
        return math.nan
    return float(text)


def _parse_labels(text: str, line: str) -> tuple:
    labels, index = [], 0
    while index < len(text):
        while index < len(text) and text[index] in " ,":
            index += 1
        if index >= len(text):
            break
        equals = text.find("=", index)
        if equals < 0 or equals + 1 >= len(text) or text[equals + 1] != '"':
            raise ValueError(f"malformed labels: {line}")
        key, index, value = text[index:equals].strip(), equals + 2, []
        while index < len(text) and text[index] != '"':
            if text[index] == "\\" and index + 1 < len(text):
                value.append({"n": "\n", "\\": "\\", '"': '"'}.get(text[index + 1], text[index + 1]))
                index += 2
            else:
                value.append(text[index])
                index += 1
        if index >= len(text):
            raise ValueError(f"unterminated label value: {line}")
        labels.append((key, "".join(value)))
        index += 1
    return tuple(sorted(labels))


def parse(text: str) -> list[Sample]:
    """Parse Prometheus text exposition format; `_created` samples are ignored."""
    samples = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            close = rest.rfind("}")
            if close < 0:
                raise ValueError(f"malformed sample: {line}")
            labels, value_text = _parse_labels(rest[:close], line), rest[close + 1:]
        else:
            name, _, value_text = line.partition(" ")
            labels = ()
        fields = value_text.split()
        if not fields:
            raise ValueError(f"missing value: {line}")
        if name.endswith("_created"):
            continue
        samples.append(Sample(name.strip(), labels, _parse_value(fields[0])))
    return samples


def scalar(samples: list[Sample], name: str) -> float | None:
    """Sum a counter or gauge across label sets; None if absent."""
    values = [sample.value for sample in samples if sample.name == name]
    return sum(values) if values else None


def histogram(samples: list[Sample], family: str) -> dict | None:
    """Aggregate a histogram across all label sets except `le`; None if absent."""
    total, count, buckets, seen = 0.0, 0.0, {}, False
    for sample in samples:
        if sample.name == f"{family}_sum":
            total, seen = total + sample.value, True
        elif sample.name == f"{family}_count":
            count, seen = count + sample.value, True
        elif sample.name == f"{family}_bucket":
            le = dict(sample.labels).get("le")
            if le is None:
                raise ValueError(f"{family} bucket without le label")
            bound = _parse_value(le)
            buckets[bound] = buckets.get(bound, 0.0) + sample.value
            seen = True
    if not seen:
        return None
    return {"sum": total, "count": count, "buckets": dict(sorted(buckets.items()))}


def delta(before: dict, after: dict, family: str) -> dict:
    """Difference two aggregated histograms, rejecting resets and inconsistencies."""
    if after["count"] < before["count"] or after["sum"] < before["sum"] - 1e-9:
        raise ValueError(f"{family}: counter decreased (server restart or reset)")
    if set(before["buckets"]) != set(after["buckets"]):
        raise ValueError(f"{family}: bucket boundaries changed between scrapes")
    buckets, previous = {}, 0.0
    for bound in after["buckets"]:
        value = after["buckets"][bound] - before["buckets"][bound]
        if value < 0:
            raise ValueError(f"{family}: bucket {bound} decreased")
        if value < previous:
            raise ValueError(f"{family}: cumulative buckets are not monotonic")
        buckets[bound], previous = value, value
    count = after["count"] - before["count"]
    if buckets and math.inf in buckets and buckets[math.inf] != count:
        raise ValueError(f"{family}: +Inf bucket delta {buckets[math.inf]} != count delta {count}")
    total = after["sum"] - before["sum"]
    return {"sum": total, "count": count, "buckets": buckets,
            "mean": total / count if count else None}


def quantile(hist: dict, q: float) -> float | None:
    """Approximate a quantile by linear interpolation within buckets (report-only)."""
    count = hist["count"]
    if not count:
        return None
    target, lower, cumulative_before = q * count, 0.0, 0.0
    for bound, cumulative in hist["buckets"].items():
        if cumulative >= target:
            if math.isinf(bound):
                return lower
            in_bucket = cumulative - cumulative_before
            fraction = (target - cumulative_before) / in_bucket if in_bucket else 0.0
            return lower + (bound - lower) * fraction
        lower, cumulative_before = bound, cumulative
    return lower


def check_scrape(text: str) -> list[Sample]:
    """Parse a scrape and require every family this study depends on."""
    samples = parse(text)
    missing = [family for family in REQUIRED_FAMILIES + TOKEN_FAMILIES
               if histogram(samples, family) is None]
    if scalar(samples, START_TIME) is None:
        missing.append(START_TIME)
    if missing:
        raise ValueError(f"scrape is missing required metrics: {', '.join(missing)}")
    return samples


def summarize_pair(before_text: str, after_text: str) -> dict:
    """Per-run server-side summary from scrapes taken before and after the run."""
    before, after = check_scrape(before_text), check_scrape(after_text)
    if scalar(before, START_TIME) != scalar(after, START_TIME):
        raise ValueError("server restarted between scrapes")
    families = {}
    for family in REQUIRED_FAMILIES + TOKEN_FAMILIES:
        change = delta(histogram(before, family), histogram(after, family), family)
        families[family] = {"count": change["count"], "mean_s": change["mean"],
                            "p50_s": quantile(change, 0.5), "p99_s": quantile(change, 0.99)}
    return {"families": families, "server_requests": families[REQUEST_COUNT_FAMILY]["count"]}


def is_idle(samples: list[Sample]) -> bool:
    """True when the server reports no running or waiting requests."""
    return all((scalar(samples, gauge) or 0) == 0 for gauge in GAUGES)
