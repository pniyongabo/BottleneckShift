import math

import pytest

from bottleneckshift import prometheus
from synthetic import scrape


def test_parse_handles_labels_escapes_special_values_and_skips_created():
    samples = prometheus.parse(
        '# HELP x help\n'
        'x_total{path="a\\"b\\\\c",le="+Inf"} 3 1700000000\n'
        'x_created{path="a"} 99\n'
        'y NaN\n'
        'z -Inf\n')
    assert [sample.name for sample in samples] == ["x_total", "y", "z"]
    assert dict(samples[0].labels) == {"path": 'a"b\\c', "le": "+Inf"}
    assert samples[0].value == 3
    assert math.isnan(samples[1].value) and samples[2].value == -math.inf


def test_histogram_sums_across_label_sets():
    text = ('h_bucket{engine="0",le="1"} 2\nh_bucket{engine="1",le="1"} 3\n'
            'h_bucket{engine="0",le="+Inf"} 2\nh_bucket{engine="1",le="+Inf"} 4\n'
            'h_sum{engine="0"} 1.0\nh_sum{engine="1"} 2.0\nh_count{engine="0"} 2\nh_count{engine="1"} 4\n')
    hist = prometheus.histogram(prometheus.parse(text), "h")
    assert hist == {"sum": 3.0, "count": 6, "buckets": {1.0: 5, math.inf: 6}}
    assert prometheus.histogram(prometheus.parse(text), "missing") is None


def test_summarize_pair_reports_per_run_means_and_request_count():
    summary = prometheus.summarize_pair(scrape(10, seconds=0.01), scrape(110, seconds=0.01))
    assert summary["server_requests"] == 100
    queue = summary["families"]["vllm:request_queue_time_seconds"]
    assert queue["count"] == 100 and queue["mean_s"] == pytest.approx(0.01)
    assert 0.005 <= queue["p50_s"] <= 0.02
    assert summary["families"]["vllm:inter_token_latency_seconds"]["count"] == 300


@pytest.mark.parametrize("before, after, message", [
    (scrape(10, start_time=1.0), scrape(20, start_time=2.0), "restarted"),
    (scrape(20), scrape(10), "decreased"),
])
def test_summarize_pair_rejects_restarts_and_resets(before, after, message):
    with pytest.raises(ValueError, match=message):
        prometheus.summarize_pair(before, after)


def test_check_scrape_requires_every_family():
    text = "\n".join(line for line in scrape(1).splitlines() if "prefill" not in line)
    with pytest.raises(ValueError, match="vllm:request_prefill_time_seconds"):
        prometheus.check_scrape(text)


def test_delta_rejects_inf_bucket_mismatch():
    before = {"sum": 0.0, "count": 0, "buckets": {1.0: 0, math.inf: 0}}
    after = {"sum": 1.0, "count": 3, "buckets": {1.0: 1, math.inf: 2}}
    with pytest.raises(ValueError, match="Inf bucket"):
        prometheus.delta(before, after, "h")


def test_quantile_interpolates_within_buckets():
    hist = {"count": 4, "buckets": {1.0: 2, 2.0: 4, math.inf: 4}}
    assert prometheus.quantile(hist, 0.5) == pytest.approx(1.0)
    assert prometheus.quantile(hist, 0.75) == pytest.approx(1.5)
    assert prometheus.quantile({"count": 0, "buckets": {}}, 0.5) is None


def test_is_idle_reads_running_and_waiting_gauges():
    assert prometheus.is_idle(prometheus.parse(scrape(1)))
    assert not prometheus.is_idle(prometheus.parse(scrape(1, running=2)))


def test_step_histogram_summarizes_tokens_per_engine_step():
    before = scrape(0, steps={})
    after = scrape(10, steps={32: 60, 512: 20, 2048: 20})
    steps = prometheus.summarize_pair(before, after)["steps"]
    assert steps["count"] == 100
    assert steps["mean_tokens"] == pytest.approx((32 * 60 + 512 * 20 + 2048 * 20) / 100)
    assert steps["share_over_512"] == pytest.approx(0.20)
    assert steps["share_over_1024"] == pytest.approx(0.20)
    assert steps["share_over_2048"] == pytest.approx(0.0)


def test_step_summary_is_absent_when_the_server_does_not_expose_it():
    assert "steps" not in prometheus.summarize_pair(scrape(0), scrape(5))
