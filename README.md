# BottleneckShift

**A pre-registered study of bottleneck shifts in LLM inference serving**

> When an LLM inference service slows down, can controlled experiments tell workload
> effects apart from resource contention, and identify when the dominant bottleneck
> shifts?

BottleneckShift runs controlled experiments on [vLLM](https://github.com/vllm-project/vllm)
and measures every run three ways: from the client, from the server's own phase
timings, and from GPU/CPU telemetry. The goal is experimental diagnosis. Everything
here uses public software, generated prompts, and independent experiments, with no
proprietary code, data, or results.

## Why it matters

Serving systems are tuned for different goals. DistServe, for example, sets separate
time-to-first-token (TTFT) and time-per-output-token (TPOT) targets for chatbot,
code-completion, and summarization services [[3]](#related-work). Knowing which phase of
a request dominates, and when that changes, decides what to tune:

- **Interactive chat and coding assistants.** Users feel both the wait for the first
  token and how smoothly tokens stream. Under load, the delay moves into per-step
  generation, and the scheduler's token budget trades median against tail latency
  (finding 2).
- **Long-prompt workloads such as retrieval-augmented generation.** A workload that
  looks prefill-heavy can still be dominated by decode (finding 1).
- **Shared GPUs.** Multi-tenant clusters co-locate work with MPS or time slicing
  [[8]](#related-work). A co-located job slows inference in ways that GPU utilization
  barely shows (finding 3).
- **Capacity planning and autoscaling.** The concurrency at which throughput saturates
  depends partly on a scheduler default, so it's a configuration choice, not a fixed
  hardware limit (finding 2).

## Findings so far

All on one GPU with a 0.5B model (Qwen2.5-0.5B-Instruct on an RTX A4000).

1. **Decode dominates end-to-end latency, even with long prompts.** With 2048-token
   prompts and 32-token outputs, prefill takes about 40 ms and the 31 generation steps
   about 122 ms. ([Phase B](reports/milestone3-20261003T180358Z-phase-b.md))
2. **Past saturation, added latency moves into per-step time, not into the wait for the
   first token.** From 8 to 32 concurrent requests, median inter-token latency (ITL)
   rises from 4.9 to 55 ms, while the share of latency before the first token falls
   from 0.44 to 0.24, against the pre-registered prediction. Changing vLLM's per-step
   token budget reshapes this cost: at 32 requests, 8192 tokens gives a 7 ms median ITL
   and 18% more throughput but a 169 ms p99, while 512 evens out ITL but triples TTFT.
   ([Phase C](reports/milestone3-20261003T201214Z-phase-c.md),
   [Phase D](reports/milestone3-20261006T022744Z-phase-d.md))
3. **GPU contention leaves a signature distinct from added load.** A co-located process
   at a 50% duty cycle costs 13–15% end-to-end latency. It slows single requests too,
   leaves queueing flat, and slows prefill more than decode, while `nvidia-smi`
   utilization moves only 1–6 points.
   ([Phase C](reports/milestone3-20261003T201214Z-phase-c.md))

![Phase C crossover: throughput, inter-token latency, and pre-first-token share against concurrency](reports/figures/milestone3-phase-c-crossover.png)

![Phase D token budget: inter-token latency, TTFT, and throughput per budget](reports/figures/milestone3-phase-d-budget.png)

More figures, including the contention series: [`reports/figures/`](reports/figures/README.md).

## Method

- **Pre-registration.** Each protocol, with numbered predictions and falsification
  criteria, is merged before its data are collected
  ([`experiments/`](experiments/roadmap.md)). Every report scores every prediction,
  including the ones that failed.
- **One factor at a time,** in forward and reverse order, with three repetitions, a
  pinned model revision, prefix caching off, and explicit sampling.
- **Three views of every run:** client request timings, vLLM's server-side `/metrics`
  histograms (queue, prefill, decode), and GPU/CPU telemetry.
- **Validation before analysis.** Request counts, failures, token lengths, server
  counts, and telemetry coverage are checked, and analysis refuses an unvalidated
  series. Raw data are archived with checksums.
- **Controls for interventions:** a sham condition and a recovery baseline.

## Related work

1. Yu et al., [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI
   2022: iteration-level scheduling, the batching model vLLM builds on.
2. Kwon et al., [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023:
   the serving system under study.
3. Zhong et al., [DistServe](https://arxiv.org/abs/2401.09670), OSDI 2024: prefill and
   decode interfere, and services set different TTFT and TPOT targets. This study
   measures that interference on one GPU.
4. Agrawal et al., [Sarathi-Serve](https://arxiv.org/abs/2403.02310), OSDI 2024: chunked
   prefill and the per-step token budget. Finding 2 measures that trade-off directly.
5. Agrawal et al., [Etalon](https://arxiv.org/abs/2407.07000), 2024: averaged metrics
   such as TPOT hide jitter. Here, median and tail ITL rank the token budgets in
   opposite orders.
6. Dean and Barroso, [The Tail at Scale](https://doi.org/10.1145/2408776.2408794), CACM
   2013: why tail latency, not the average, shapes user experience.
7. Hoefler and Belli,
   [Scientific Benchmarking of Parallel Computing Systems](https://doi.org/10.1145/2807591.2807644),
   SC 2015: the reporting and repetition practices this study follows.
8. NVIDIA, [Multi-Process Service](https://docs.nvidia.com/deploy/mps/index.html) and
   [GPU time slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html):
   the GPU-sharing mechanisms behind the contention scenario.

Each report has a related-evidence section that checks its findings against these and
other sources; the [Phase C report](reports/milestone3-20261003T201214Z-phase-c.md) is
the most complete.

## Status and next steps

| Milestone | Question | Status |
|---|---|---|
| 1–2 | Concurrency effect, then repeated under explicit cache, sampling, and model controls | Complete ([1](reports/milestone1-20261001T205137Z.md), [2](reports/milestone2-20261003T034456Z.md)) |
| 3 | Workload regimes, concurrency crossover, GPU contention, token-budget intervention | Complete ([B](reports/milestone3-20261003T180358Z-phase-b.md), [C](reports/milestone3-20261003T201214Z-phase-c.md), [D](reports/milestone3-20261006T022744Z-phase-d.md)) |
| 4 | Measurement-driven backend selection | Proposed |
| 5–9 | Workload shape, network, host contention, GPU contention, crossed factors | Proposed |

Proposed milestones have no data yet, and each needs its own pre-registered protocol
first. Scope and status: [roadmap](experiments/roadmap.md).

## Reproduce

Linux, Python 3.11, and a CUDA GPU supported by the pinned vLLM release:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e '.[test]'
python scripts/run_experiment.py --config configs/milestone1-forward.toml \
  --series-id DEVICE-MODEL-forward-01 --dry-run
```

The full guide (preflight, runs, validation, analysis, and instrumentation) is in
[`docs/reproduce.md`](docs/reproduce.md). Raw data aren't in this repository; each
report's artifact record gives the archive name and checksum.

## Limitations

One model, one GPU, and one framework version limit generality. Load is closed-loop,
client and server share a host (so there's no network realism), and prompts are
generated. Findings are measured associations under controlled changes; server
metrics and telemetry corroborate them but don't prove causation on their own.

## Repository layout

`configs/` experiment inputs · `scripts/` orchestration around the vLLM CLI ·
`src/bottleneckshift/` validation, manifests, and parsers · `analysis/` post-processing
and figures · `experiments/` protocols written before running · `reports/` scored
results · `docs/` [glossary](docs/glossary.md), [reproduction guide](docs/reproduce.md),
and [workflow](docs/milestone-workflow.md).
