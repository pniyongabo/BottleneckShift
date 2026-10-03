# BottleneckShift Milestone 3 Phase C: crossover probe and GPU contention

## Result

**Crossover probe:** the primary pre-registered prediction failed, in the opposite
direction. Raising concurrency on `prefill_heavy` (2048 input / 32 output tokens) from
C8 to C16 and C32 did not make pre-first-token time dominant. Its share of E2E latency
fell from 0.44 to 0.35 to 0.24. The added delay appeared in token generation instead:

* TPOT rose from 8.4 to 17.6 to 50.4 ms.
* The median inter-token interval rose from 4.9 ms at C8 and 5.5 ms at C16 to 55 ms
  at C32.
* Output throughput peaked at C16 (583 tok/s) and fell slightly at C32 (555 tok/s).

So the server saturates between C8 and C16. Beyond that point, prefill work is
plausibly folded into every generation step by chunked prefill, so it shows up as
decode time, not as time before the first token. This reading is post hoc; see the
interpretation section.

**GPU contention:** almost every prediction held.

* The sham and recovery series match the uncontended baseline within 1%.
* Latency rises and throughput falls monotonically with duty cycle at both C1 and C8:
  at duty 0.5, E2E is +13% at C1 and +15% at C8, and throughput is −12% and −13%.
* Server prefill and decode times both rise. Prefill rises the most at C1 (+23% vs
  +11% for decode).
* The client–server TTFT gap is unchanged.

Contention leaves a signature that is distinct from added load. Service times inflate
at C1 as well as C8, and queue time barely moves (+6% at C8). Raising concurrency from
C8 to C16 instead roughly doubles queue and decode time and does nothing at C1.

## Provenance and protocol

- **Session:** `milestone3c-20261003T201214Z`, October 3, 2026, 20:12–21:54 UTC. It
  ran on the same VM and host as Phase B (Hyperstack RTX A4000, 15,352 MiB,
  4 vCPU, 20 GiB), with driver 580.178.04, vLLM 0.29.0, and PyTorch 2.13.0.
- **Protocol:** the Phase C addendum in
  [`experiments/milestone3.md`](../experiments/milestone3.md), pre-registered by the
  merge of PR #14 at 2026-10-03 20:05:16 UTC, before any Phase C data.
- **Commits:**
  - `5d16a1d` for the probe in both orders and `off-before`;
  - `d9578ae` (PR #15, a fix to the generator's startup VRAM check; see the execution
    notes) for `sham`, both active series, and `off-after`.

  The commits differ only in `scripts/gpu_contention.py`, which the probe and the off
  series do not use.
- **Controls:** the same as Milestone 2 and Phase B, confirmed in every `server.log`:
  - model and tokenizer pinned to `7ae5576…`;
  - prefix caching off;
  - `--generation-config vllm`;
  - greedy sampling;
  - 15 s cooldowns;
  - three repetitions.

  Chunked prefill is enabled (vLLM's default; logged as `enable_chunked_prefill=True`).
- **Probe:** `prefill_heavy` at `anchor_c8`, `probe_c16`, and `probe_c32`, 100 prompts
  per run, with 32-prompt warm-ups at C8, C16, and C32. Forward order is C8, C16, C32;
  reverse is C32, C16, C8.
- **Contention:** the Phase B `prefill_heavy` design (C1 then C8) with the generator:
  fp16 2048×2048 matrix multiplies, a 100 ms period, and a 1024 MiB cap on total
  VRAM. Series ran in pre-registered order: `off-before`, `sham`, `active-025`,
  `active-050`, `off-after`.

## Validation

`scripts/validate_results.py` passed for each series on the host right after it ran,
and again locally from the transferred archive. The archive SHA-256 was verified after
transfer.

| Series | Measured requests | Warm-up requests | Server count = prompts | Telemetry max gap (GPU / CPU) | Contention log |
|---|---:|---:|---|---|---|
| crossover-forward | 900 | 96 | yes | 0.21 / 1.0 s | — |
| crossover-reverse | 900 | 96 | yes | 0.22 / 1.0 s | — |
| off-before | 600 | 20 | yes | 0.21 / 1.0 s | — |
| sham | 600 | 20 | yes | 0.20 / 1.0 s | 0 steps; 706 MiB |
| active-025 | 600 | 20 | yes | 0.20 / 2.0 s | achieved duty 0.255 (kernel 0.142); 706 MiB |
| active-050 | 600 | 20 | yes | 0.21 / 1.0 s | achieved duty 0.504 (kernel 0.282); 706 MiB |
| off-after | 600 | 20 | yes | 0.21 / 1.0 s | — |

In total, 4,800 measured requests had zero failures and exact 2048/32 token lengths.
No server log has `ERROR` lines. The generator's total VRAM (706 MiB, including its
CUDA context) stayed under the 1024 MiB cap.

## Measurements

Medians across three repetitions, with the repetition range where it matters.
TTFT/TPOT/E2E are per-run medians across requests. Shares, server phases, and
"mean TTFT" are per-run means.

### Part 1 — crossover probe (`prefill_heavy`, 2048/32)

| Order | Cond. | Median TTFT (ms) | Mean TTFT (ms) | TPOT (ms) | Median E2E (ms) | Output tok/s | Pre-1st share (client / server) | Server queue / prefill / decode (ms) | ITL p50 / p99 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fwd | C8 | 219.6 | 215.2 | 8.42 | 488.2 | 517.5 | 0.444 / 0.392 | 59.0 / 116.4 / 270.9 | 4.89 / 45.1 |
| fwd | C16 | 307.0 (280.0–308.5) | 296.9 | 17.57 | 858.9 | 582.8 | 0.348 / 0.305 | 114.2 / 130.9 / 559.7 | 5.46 / 48.7 |
| fwd | C32 | 278.6 | 430.9 | 50.34 | 1892.7 (1810.4–1892.8) | 555.1 | 0.238 / 0.198 | 188.2 / 153.7 / 1384.8 | 54.99 / 58.2 |
| rev | C32 | 278.5 | 430.2 | 50.47 | 1906.4 (1806.5–1909.5) | 552.8 | 0.236 / 0.197 | 188.9 / 153.3 / 1392.4 | 54.93 / 58.3 |
| rev | C16 | 307.9 | 296.3 | 17.53 | 858.9 | 582.3 | 0.346 / 0.304 | 114.5 / 131.0 / 561.5 | 5.46 / 48.3 |
| rev | C8 | 218.3 | 215.3 | 8.54 | 489.4 | 516.3 | 0.445 / 0.393 | 59.1 / 116.6 / 270.4 | 4.91 / 45.4 |

At C16 and C32 the per-run *medians* of TTFT and E2E are unstable. Requests arrive
in waves of the concurrency limit, so the distributions are multimodal. For example,
C32 median TTFT is 278 ms while the mean is 431 ms and p90 is about 1,150 ms. The
means and the shares (which are mean-based) are stable across repetitions and orders.

### Part 2 — GPU contention (`prefill_heavy`, C1 and C8)

| Series | Cond. | TTFT (ms) | TPOT (ms) | E2E (ms) | Output tok/s | Server queue / prefill / decode (ms) | Pre-1st share (client) | Client − server TTFT (ms) | GPU util |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off-before | C1 | 54.00 | 3.88 | 174.53 | 183.3 | 0.01 / 40.0 / 122.5 | 0.309 | 3.7 | 91.7% |
| sham | C1 | 54.07 | 3.88 | 174.50 | 183.4 | 0.01 / 40.1 / 122.5 | 0.310 | 3.8 | 88.3% |
| active-025 | C1 | 57.91 | 4.08 | 185.46 | 172.3 | 0.02 / 43.6 / 129.8 | 0.311 | 3.8 | 91.7% |
| active-050 | C1 | 64.26 | 4.30 | 197.96 | 161.6 | 0.02 / 49.0 / 136.4 | 0.322 | 4.0 | 93.1% |
| off-after | C1 | 54.16 | 3.88 | 174.48 | 183.4 | 0.01 / 40.2 / 122.5 | 0.309 | 3.7 | 91.8% |
| off-before | C8 | 220.18 | 8.40 | 490.44 | 514.4 | 60.7 / 117.0 / 269.8 | 0.448 | 4.6 | 90.3% |
| sham | C8 | 222.21 | 8.35 | 490.20 | 514.8 | 61.4 / 116.9 / 268.9 | 0.450 | 4.9 | 93.5% |
| active-025 | C8 | 237.90 | 9.07 | 525.69 | 479.5 | 63.0 / 126.3 / 292.4 | 0.441 | 4.9 | 93.8% |
| active-050 | C8 | 255.78 | 9.88 | 565.48 | 445.7 | 64.1 / 136.8 / 318.5 | 0.435 | 5.1 | 96.3% |
| off-after | C8 | 221.68 | 8.42 | 490.65 | 514.4 | 60.8 / 117.2 / 269.7 | 0.447 | 4.7 | 93.8% |

Changes at duty 0.50 relative to `off-before`:

| Cond. | E2E | TTFT | TPOT | Throughput | Queue | Prefill | Decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | +13.4% | +19.0% | +10.8% | −11.8% | — | +22.5% | +11.3% |
| C8 | +15.3% | +16.2% | +17.6% | −13.4% | +5.6% | +17.0% | +18.0% |

At duty 0.25 the changes are about half as large.

## Predictions vs observed

### Part 1 — crossover probe

| # | Prediction | Observed | Verdict |
|---|---|---|---|
| 1 (primary) | Client pre-first-token share > 0.5 at C16 or C32; pre-first-token becomes dominant | Share falls: 0.44 → 0.35 → 0.24; decode stays dominant | **Not confirmed (opposite direction)** |
| 2 | Queue at C32 > 4× queue at C8 | 188 / 59 ms = 3.2× (1.9× at C16) | **Not confirmed** |
| 3 | TPOT ≥ ~8.3 ms; ITL p99 > 40 ms | TPOT 17.6 and 50.4 ms; p99 48.7 and 58.2 ms | **Confirmed** |
| 4 | C32 throughput < 1.5× C8 | 1.07×; it peaks at C16 (1.13×) | **Confirmed** |
| 5 | C8 anchor reproduces Phase B within 5% (E2E, throughput) | E2E within 0.5% (488.2/489.4 vs 489.2/486.8 ms); throughput within 0.5% | **Confirmed** |

### Part 2 — GPU contention

| # | Prediction | Observed | Verdict |
|---|---|---|---|
| 1 | Sham within the `off-before` range or 2% | E2E within 0.05%; C8 TTFT +0.9%; throughput identical | **Confirmed** |
| 2 | Monotonic dose response (E2E, TTFT, TPOT up; throughput down) at C1 and C8 | Monotonic for all four metrics in both conditions | **Confirmed** |
| 3 | Server prefill and decode both rise; at C8 queue rises proportionally more than prefill | Both rise. At C8 queue rises *less* than prefill (+5.6% vs +17.0%) | **Mixed:** first part confirmed, second not |
| 4 | Client − server TTFT gap within 2 ms of `off-before` | +0.3 ms (C1) and +0.5 ms (C8) | **Confirmed** |
| 5 | Mean GPU utilization changes < 5 points (off-before → active-0.50) | +1.4 (C1), +6.0 (C8); the off and sham series themselves vary by ±3 points | **Mixed:** within noise at C1, just over at C8 |
| 6 | (Secondary) the pre-first-token share at C8 rises under contention | 0.448 → 0.441 → 0.435 (slightly down) | **Not confirmed** |
| 7 | `off-after` within 2% of `off-before` (E2E, throughput) | Within 0.05% | **Confirmed** |

## Interpretation

- **Where the extra latency goes past saturation (post hoc; plausible, not proven).**
  From C8 to C32, server decode time per request grows from 271 to 1,385 ms. For 31
  generation steps that is ~8.7 → 44.7 ms per step, and the median inter-token
  interval jumps from 4.9 to 55 ms between C16 and C32.
  - With chunked prefill (enabled), new requests' prompt chunks share iterations with
    ongoing decodes, so a generation step's duration is set by the prefill tokens it
    carries.
  - At C32 the steady state always has some prefill in flight, so nearly every step
    becomes a prefill-sized step. The ITL median rising to roughly the ITL p99 at C8
    (~45–55 ms) fits this.
  - This matches the mechanism the literature describes: chunked prefill trades
    higher TTFT for "generation stalls", i.e. inflated time between tokens (see the
    [Phase B report's](milestone3-20261003T180358Z-phase-b.md) related-evidence
    section, references 3–5).

  For this study, the consequence is that a **phase-level share (pre-first-token vs
  decode) cannot reveal this bottleneck shift**. The shift is from memory-bound
  decode steps at low load to compute-heavy mixed steps at high load. It shows up in
  the per-step time distribution (ITL p50) and in throughput saturation, not in which
  phase "dominates". The pre-registered metric was the wrong instrument for this
  question. This is reported as a finding, not as a reason to re-score the
  prediction.
- **Saturation.** Throughput rises 2.8× from C1 to C8 (Phase B), a further 13% to
  C16, then falls 5% at C32. Queue time grows sub-linearly with concurrency (1.9× and
  3.2×) because the scheduler admits more requests into running batches rather than
  holding them waiting. The cost lands on per-step time.
- **Contention signature.** GPU contention and added load leave different
  fingerprints:

  | | C1 | C8 queue | C8 prefill / decode | Client−server gap |
  |---|---|---|---|---|
  | Contention (duty 0.5) | slows (+13% E2E) | ~flat (+6%) | +17% / +18% | unchanged |
  | Load (C8 → C16) | unaffected | +88% | +12% / +107% | +2.8 ms |

  At C1, contention slows prefill (+23%) about twice as much as decode (+11%). That
  fits a compute-bound matmul generator competing mostly with compute-bound prefill,
  with memory-bound decode less exposed. No published source tests this asymmetry,
  so it remains this study's observation.

  The flat queue is partly a property of the load generator. With closed-loop load
  (`--max-concurrency`), requests in flight are capped, so slower service cannot
  build an unbounded queue. Under open-loop arrivals, queueing models predict that
  queue time grows with service time [P2 §3.1]. The signature should therefore be
  stated as "under closed-loop load"; an open-loop test would check it.
- **GPU utilization is not a reliable detector.** The `nvidia-smi` percentage moves by
  1–6 points under contention that costs 13–15% in E2E, and it varies by about 3
  points between identical off and sham series. The generator's own log (achieved and
  kernel duty) and the server phase times are what identify the contention.
- **Front end unaffected by GPU contention, but not by load.** The client–server
  TTFT gap is flat under contention, as predicted. It grows with concurrency instead
  (4.6–5 ms at C8, 7.4 at C16, 15 at C32), which fits API-server work per request
  growing under load. It was not isolated.

## Related evidence from published research and engineering sources

This section checks the Phase C findings against published papers, preprints, and
engineering documentation and blogs. Every source below was fetched and read on
October 3, 2026. The preprint IDs, titles, and first authors were also checked
against arXiv, and the key quotes from the vLLM and `nvidia-smi` documentation were
re-checked verbatim. Blogs and documentation are not peer-reviewed.

Limits on comparison: most sources use 7B–180B models on A100/H100 GPUs, or
pre-V1 or custom schedulers. None used an RTX A4000, a 0.5B model, or a co-located
synthetic matmul load.

| Finding | Verdict | Evidence |
|---|---|---|
| Throughput saturates between C8 and C16; queue time grows sub-linearly | **Supports** | Throughput "generally saturates around the maximum batch size" while latency keeps rising [B8, B9]. Past the compute-bound point, more users "just delays everyone with no benefit of throughput" [B19, see also B18]. Step time "grows roughly with B" above saturation [B3]. Sub-linear queue growth is expected under closed-loop load, which caps requests in flight [P3, B7]. |
| Past saturation the cost moves into per-step time; a phase-share metric cannot show it | **Supports** | Mixed batches make step time rise with prefill tokens: "generation stalls", and time between tokens up to 28.3× higher [P1 §1, §4.2–4.3]. "Adding a single prefill job to a batch of decoding requests significantly slows down both" [P2 §2.3]. vLLM V1 schedules prefill and decode within one token budget, with no prefill/decode distinction [B1, B3]. Its documentation states that smaller `max_num_batched_tokens` "achieve better ITL because there are fewer prefills slowing down decodes" [B4]. Averaged metrics such as TPOT "hide jitters in token generation" [R1 §3.1], matching the conclusion that per-step distributions are needed. |
| Closed-loop waves make per-run medians unstable | **Qualifies** | Closed-loop load, sending requests in "rounds", and coordinated omission are all documented [P3, B7, B17, B20]. No source reports multimodal medians specifically; that is this study's observation. Closed-loop testing also understates tails [B20], and common metrics can be gamed [R2]. |
| GPU contention: C1 slows, queue flat, prefill slows more than decode at C1 | **Qualifies** | Time-slicing gives "an equal share of time to all GPU processes" with no isolation [B13], at "a cost for context-switching" that brings "increased latency, jitter" [B14]. This fits the similar prefill and decode slowdowns at C8. Interference from co-executing workloads on inference tails is documented [R3, R4]. MPS would run the kernels concurrently instead of time-slicing them [B12]. No source covers the prefill > decode asymmetry at C1, and the flat queue is partly a closed-loop effect (see interpretation) [P2 §3.1]. |
| `nvidia-smi` utilization is a weak signal | **Supports** | NVML utilization is the "percent of time … during which one or more kernels was executing" [B10], so the generator's kernels count as busy time too. DCGM's SM activity, "0.8 or greater is necessary, but not sufficient", is the documented finer metric [B11]. Practitioners call the utilization counter "misleading" [B15, B16]. |
| Client–server TTFT gap grows with load and is flat under GPU contention | **Supports, with a caveat** | vLLM's TTFT starts when tokenization begins in the frontend, while queue time is measured in the engine core, which "the frontend does not have visibility into" [B5]. Tokenization, detokenization, and streaming run outside the engine core [B1]. Frontend cost has been large: the HTTP server was "33% of the total execution time" before the V1 process split [B2], and "the API server process can become a bottleneck" [B6]. Caveat: the client shares the host's CPUs, and CPU interference can degrade vLLM "by up to two orders of magnitude" [R5]. |

**Taken together:** the saturation and per-step findings match established results. This
study adds measurements on a small model with a bandwidth-limited workstation GPU, and
shows that a pre-registered phase-share metric misses the shift. The contention
signature is only partly anticipated. The C1 prefill/decode asymmetry and the
utilization blind spot under a co-located load are this study's observations. The flat
queue should be reported as specific to closed-loop load. The planned chunk-budget
intervention is the test vLLM's own documentation implies [B4].

**Peer-reviewed**

- [P1] A. Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI 2024. https://arxiv.org/abs/2403.02310
- [P2] Y. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024. https://arxiv.org/abs/2401.09670
- [P3] B. Schroeder, A. Wierman, M. Harchol-Balter, "Open Versus Closed: A Cautionary Tale," NSDI 2006. https://www.usenix.org/legacy/event/nsdi06/tech/schroeder.html

**Preprints (arXiv)**

- [R1] A. Agrawal et al., "Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems," arXiv 2407.07000, 2024. https://arxiv.org/abs/2407.07000
- [R2] Z. Wang et al., "Revisiting Service Level Objectives and System Level Metrics in Large Language Model Serving," arXiv 2410.14257, 2024. https://arxiv.org/abs/2410.14257
- [R3] W. Zhao et al., "Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads," arXiv 2410.07381, 2024. https://arxiv.org/abs/2410.07381
- [R4] J. J. Martín et al., "Performance Isolation for Inference Processes in Edge GPU Systems," arXiv 2601.07600, 2026. https://arxiv.org/abs/2601.07600
- [R5] M. Siavashi et al., "Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC," arXiv 2604.07609, 2026. https://arxiv.org/abs/2604.07609

**Engineering blogs and documentation (not peer-reviewed)**

- [B1] vLLM Team, "vLLM V1: A Major Upgrade to vLLM's Core Architecture," vLLM blog, Jan 27, 2025. https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- [B2] vLLM Team, "vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction," vLLM blog, Sep 5, 2024. https://blog.vllm.ai/2024/09/05/perf-update.html
- [B3] A. Gordić, "Inside vLLM: Anatomy of a High-Throughput LLM Inference System," vLLM blog, Sep 5, 2025. https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
- [B4] vLLM documentation, "Optimization and Tuning: Chunked Prefill." https://docs.vllm.ai/en/latest/configuration/optimization.html
- [B5] vLLM documentation, "Metrics" (design). https://docs.vllm.ai/en/latest/design/metrics.html
- [B6] vLLM documentation, "Data Parallel Deployment." https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html
- [B7] vLLM documentation, "vllm bench serve." https://docs.vllm.ai/en/latest/cli/bench/serve.html
- [B8] NVIDIA Technical Blog, "LLM Inference Benchmarking: Fundamental Concepts," Apr 2, 2025. https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
- [B9] NVIDIA NIM documentation, "Metrics and Parameters" (benchmarking). https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
- [B10] NVIDIA, "nvidia-smi" documentation (Utilization). https://docs.nvidia.com/deploy/nvidia-smi/index.html
- [B11] NVIDIA DCGM documentation, "Profiling Metrics." https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html
- [B12] NVIDIA, "Multi-Process Service" documentation ("When to Use MPS"). https://docs.nvidia.com/deploy/mps/index.html
- [B13] NVIDIA GPU Operator documentation, "Time-Slicing GPUs in Kubernetes." https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html
- [B14] K. Klues et al., "Improving GPU Utilization in Kubernetes," NVIDIA Technical Blog, Jun 16, 2022. https://developer.nvidia.com/blog/improving-gpu-utilization-in-kubernetes/
- [B15] R. Baviskar, "GPU Utilization is a Misleading Metric," Trainy blog, Feb 24, 2025. https://trainy.ai/blog/gpu-utilization-misleading-metric
- [B16] C. Frye, "GPU utilization guide," Modal, Feb 24, 2025. https://modal.com/blog/gpu-utilization-guide
- [B17] W. Kadous et al., "Reproducible Performance Metrics for LLM inference," Anyscale blog, Nov 1, 2023. https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference
- [B18] Databricks/MosaicML, "LLM Inference Performance Engineering: Best Practices," Databricks blog, Oct 12, 2023. https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
- [B19] D. Thomas, "Benchmarking Text Generation Inference," Hugging Face blog, May 29, 2024. https://huggingface.co/blog/tgi-benchmarking
- [B20] G. Tene, "wrk2" README (coordinated omission). https://github.com/giltene/wrk2

## Limitations

- The probe's primary metric was not suited to chunked-prefill scheduling, as above.
  The ITL-based reading of the shift is exploratory and needs its own pre-registered
  test, for example reporting per-step time or prefill tokens per iteration directly.
- At C16 and C32, per-run medians are unstable because requests arrive in waves. Mean
  and percentile summaries are more reliable there. Closed-loop load also has a
  documented bias of its own: coordinated omission understates tail latency [B20,
  P3]. Percentiles from these runs are therefore not artifact-free either.
- The client runs on the server's 4 vCPUs. The growing client–server TTFT gap
  therefore cannot separate client-side overhead from API-server overhead. Host CPU
  contention is known to hurt vLLM badly [R5].
- Contention ran in a fixed dose order. The sham/recovery bracketing and the
  monotonic pattern argue against drift, but order and dose are not fully separated.
- One host and one model. Sampler overhead is still unmeasured (no telemetry-off
  series).
- The contention series ran ~53 minutes after `off-before`, because of the generator
  fix. `off-after` matching `off-before` within 0.05% suggests no drift.

## Execution notes

- **Host:** the run reused the Phase B VM. Phase B's `~/pair-id` file was moved aside
  first, so Phase C got its own ID.
- **Failed first sham attempt** (`sham-failed1`, retained in the archive, no
  measurements):
  - **Error:** the generator exited at startup with "only 612 MiB free; cap is
    1024 MiB".
  - **Cause:** its free-VRAM check required cap + 256 MiB *after* its own CUDA
    context already existed, so it counted the context twice.
  - **Fix:** PR #15 (`d9578ae`) requires only the tensor budget plus 64 MiB, and adds
    a test.
  - **Resumption:** `run.sh` skipped the three completed series and resumed. The
    pre-registered parameters are unchanged. The restart overwrote `run.log`; the
    traceback is in `sham-failed1/contention.stderr`.
- **Generator timing:** achieved duty was 0.255 and 0.504. Kernel duty (GPU time
  inside the matmuls) was 0.142 and 0.282, because each step's blocking sync adds
  idle time inside the busy window.

## Recommended next step

- **Make the bottleneck-shift test match the scheduler.** The chunk-budget intervention
  below is the test vLLM's own documentation implies [B4]. Add DCGM SM activity
  [B11], if available, alongside `nvidia-smi` utilization, and consider an
  open-loop arrival condition to check the flat-queue signature [P2, P3].
  - Pre-register a per-iteration metric: ITL p50 or step-time distribution vs
    concurrency, with the saturation point (C8–C16) as the predicted transition.
  - Optionally, vLLM's iteration-level statistics, if they can be exposed cheaply.
  - This would fit Milestone 5 (workload shape) or a short Milestone 3 addendum
    before more data.
- **Package the faculty-facing story** from what is now solid:
  - a saturation and per-step shift between C8 and C32;
  - a contention signature distinct from load (C1 sensitivity, flat queue,
    prefill-first slowdown);
  - utilization as a weak signal.
- **Milestone 3 status:** Phases B and C are complete. Mark Milestone 3 complete, with
  the bottleneck-shift question carried forward.

## Artifact record

- **Archive:** `milestone3c-20261003T201214Z.tar.gz` (2.9 MB).
- **SHA-256:** `4513c1309f3b078963cb34d770305c300839321e789d1a0183564c36d05f0aa9`
- **Contents:**
  - all seven series and `sham-failed1`: raw `requests.json`, manifests, server logs,
    metrics scrapes, `gpu.csv`, `cpu.txt`, `contention.jsonl` and `contention.stderr`,
    and per-series `validation.json`;
  - the host scripts and logs.
- **Analysis outputs** (`runs.csv`, `summary.csv`) were produced locally from the
  verified archive with `analysis/phases.py` at `d9578ae`.
- **Before publishing:** remove identifying data. `cpu.txt` contains the hostname, and
  the manifests contain the GPU UUID and host paths. The archive is retained outside
  this repository.
