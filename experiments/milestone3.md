# Milestone 3 protocol: diagnose one bottleneck shift

> **Status: Phase B pre-registered October 3, 2026** (decisions below made by the
> maintainer; this file is committed before any Milestone 3 data are collected).
> Phase A may refine measurement details while building the tooling. Any change to
> Phase B's regimes, conditions, or predicted signatures after this commit must be
> recorded below as a dated amendment, committed before Phase B data are collected.
> Phase C (intervention) is pre-registered separately, by an addendum committed
> before Phase C runs.

## Question

Under which workload conditions does the dominant source of client-visible delay
change, and what measurements distinguish the causes?

## Fixed controls (inherited from Milestone 2)

Same model and pinned revision (`7ae557604adf67be50417f59c2c2f167def9a775`), vLLM
0.29.0, prefix caching off, `--generation-config vllm`, greedy sampling
(`temperature 0.0`, `top_p 1.0`), `percentile_metrics` including `e2el`, server
settings (TP 1, GPU memory utilization 0.90), seed 2027, three repetitions,
15-second cooldowns, forward and reverse condition orders, and one stable host for
all phases. The preferred host is the Milestone 2 shape: Hyperstack RTX A4000,
4 vCPU, 20 GiB, driver 580.178.04. Warm-ups run at both concurrency levels, using
each regime's token lengths.

## Phases

### Phase A — tooling (no measured data)

Implement and test the prerequisites below, then run one disposable GPU shakedown of
the tooling. No Phase A output is analyzed.

### Phase B — baseline regimes (pre-registered by this file)

Two contrasting workload regimes, each at C1 and C8, each in forward and reverse
order:

| Regime | Input tokens | Output tokens | Prompts per run | Emphasis |
|---|---:|---:|---:|---|
| `prefill_heavy` | 2048 | 32 | 100 | prompt processing (prefill) |
| `decode_heavy` | 64 | 512 | 50 | token generation (decode) |

Rationale: both fit easily in the model's 32,768-token context and the A4000's KV
cache at C8. Their ratios of input to output differ by ~2,000×, so prefill and decode
respectively should dominate end-to-end latency. `decode_heavy` uses 50 prompts per
run to halve its slowest runs (C1 runs take ~100 s); each request still yields 511
inter-token intervals. Its per-run request counts are therefore not directly
comparable with Milestones 1–2 or with `prefill_heavy`. Estimated measured time is
~3 minutes per `prefill_heavy` series and ~8 minutes per `decode_heavy` series, so
~22 minutes for all four series.

**Configs:** one config per regime and order, so that no per-condition workload
override is needed: `configs/milestone3-{prefill_heavy,decode_heavy}-{forward,reverse}.toml`.
Each pair differs only in plan name and condition order. The server restarts between
series, as in Milestone 2.

### Phase C — one intervention (pre-registered by a later addendum)

Chosen after Phase B shows which regime is informative. The addendum must state the
intervention, its fixed intensity, the predicted signature, and the controls below,
and be committed before Phase C data are collected.

* **Candidate 1 (preferred; Phase A builds its tooling): GPU contention.** A separate CUDA process runs a fixed
  matrix-multiply loop at a fixed duty cycle and memory footprint. It starts after the
  vLLM server and stays within the VRAM vLLM leaves free (≤ ~1 GiB at
  `gpu_memory_utilization = 0.90`). The intensity is recorded in the manifest.
* **Candidate 2: loopback network delay.** Apply `tc netem` delay to `lo` for the
  client–server port. This is a clean discriminator (client-visible delay with
  unchanged server-side timings) but is not representative of a real network;
  Milestone 6 does network properly.
* Host CPU contention is not a Phase C candidate on the 4-vCPU host; it is left to
  Milestone 7 with pinning or a separate client host.
* **Required controls:** a *sham* condition (the intervention set up but inactive,
  e.g. the CUDA process holding its context while sleeping, or a 0 ms netem rule) and
  a *recovery* baseline after the intervention is removed.

## Predicted signatures (Phase B)

Stated before collection; a different signature is informative and must be reported.

| Measurement | `prefill_heavy`, C1 → C8 | `decode_heavy`, C1 → C8 |
|---|---|---|
| TTFT | rises strongly (prefills queue and batch) | small absolute rise |
| Share of E2E latency spent before the first token | high at both C1 and C8 | low at both |
| TPOT / ITL | rises; decode steps are interleaved with other requests' prefill chunks, so ITL tail widens | modest rise, similar to Milestone 2's +12% |
| Output throughput gain | well below the ~6.5× seen in Milestone 2 | near or above Milestone 2's ~6.5× |
| Server `vllm:request_queue_time_seconds` | rises at C8 | small at both |
| Server `vllm:request_prefill_time_seconds` | dominant component | minor component |
| Server `vllm:request_decode_time_seconds` | minor component | dominant component |
| GPU utilization | high during prefill bursts | high and steady |

**Stable in all cells:** zero errors, exact realized token lengths, reconstructed E2E
equal to vLLM's own within 1 ms, and forward/reverse agreement within repetition
ranges.

**The shift being tested:** the dominant component of end-to-end latency changes from
pre-first-token time (queue + prefill) in `prefill_heavy` to decode time in
`decode_heavy`, and concurrency amplifies different components in each.

## Measurements

* **Client** (existing): per-request TTFT, ITLs, output lengths, errors, start times;
  run-level throughput; vLLM's `median_e2el_ms`.
* **Server** (new): Prometheus `/metrics` scraped immediately before and after each
  benchmark invocation. Per-run histogram deltas (`_sum` / `_count`) give mean queue,
  prefill, decode, and inference time, plus TTFT and ITL histograms for cross-checks
  against the client.
* **Resources** (new): a GPU sampler (`nvidia-smi --query-gpu=timestamp,
  utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm --format=csv
  -lms 200`) and a CPU sampler (`mpstat 1`) per series, with timestamps.
* **Matching** (new): `--request-id-prefix <series>-<condition>-<rep>-` on every
  benchmark invocation, so server-side records and logs can be attributed to runs.

## Tooling prerequisites (Phase A)

Each item needs tests with small synthetic fixtures, and opt-in config fields so that
the Milestone 1 and Milestone 2 command plans stay byte-identical.

1. **Metrics scrapes:** runner fetches `http://host:port/metrics` before and after each
   invocation into `metrics-before.prom` / `metrics-after.prom` next to
   `requests.json`. Paths are listed in the run manifest.
2. **Telemetry sidecar:** runner starts and stops the GPU and CPU samplers around each
   series, writing `gpu.csv` and `cpu.txt` in the series directory. Paths go in the
   series manifest. A sampler failure is recorded, not silently ignored.
3. **Request-ID prefix:** an opt-in field that adds `--request-id-prefix`, recorded in
   `protocol_controls`.
4. **Analysis:** per-run server-side means from histogram deltas; the share of E2E
   latency before the first token; and a figure showing latency components,
   throughput, and GPU utilization together for each regime × concurrency.
5. **Validation:** `validate_results.py` additionally checks that the metrics scrapes
   and telemetry files exist and parse when the config enables them.
6. **Configs:** the four Phase B configs, validated by tests (pairs differ only in
   name and order; controls match Milestone 2; regime lengths and prompt counts match
   the table above).
7. **GPU contention generator (for Phase C):** a small script that runs a fixed-size
   CUDA matrix-multiply loop at a configured duty cycle, with a hard memory cap, and a
   sham mode that holds a CUDA context while idle. It logs its parameters and achieved
   duty cycle. It is tested on CPU-only paths where possible and on the GPU in the
   disposable shakedown. Its use and intensity remain a Phase C addendum decision.

## Analysis plan

1. Validate every series (measured and warm-up) before any analysis.
2. Report median and range across repetitions per regime × concurrency × order, as in
   Milestones 1 and 2.
3. Compare client and server views: TTFT against server queue + prefill time, and
   client ITL against server ITL histograms. Disagreement beyond measurement overhead
   is reported, not adjusted.
4. State for each regime which component dominates E2E latency and how C8 changes it,
   against the predicted signatures.

## Interpretation guardrails

* The comparison supports statements about these two regimes on this host, model, and
  framework.
* Utilization and counters corroborate a mechanism; they do not prove cause.
* Measurement overhead (scrapes, samplers) must be estimated from the disposable
  shakedown, e.g. by running with and without the samplers, and reported.
* Crossed conditions run only if single-factor results motivate them.

## Decisions

Made by the maintainer on October 3, 2026:

1. Regime token lengths: `prefill_heavy` 2048/32, `decode_heavy` 64/512.
2. Prompts per run: 100 for `prefill_heavy`, 50 for `decode_heavy`.
3. Phase A prepares tooling for GPU contention (Candidate 1).

Still open, for the Phase C addendum after Phase B: whether to use GPU contention or
another intervention, its intensity and memory cap, and the sham and recovery
schedule.

## Amendments

### 2026-10-03 — Phase A measurement details (before any Milestone 3 data)

These refine how Phase A measures, as the status banner allows. Phase B's regimes,
conditions, prompt counts, and predicted signatures are unchanged.

1. **Sampler commands.** GPU: the command above, run with `TZ=UTC` so timestamps are
   UTC. CPU: `mpstat -P ALL 1` with `LC_ALL=C S_TIME_FORMAT=ISO TZ=UTC`, adding per-CPU
   rows so that saturation of a single client core is visible on the 4-vCPU host.
   The host needs `apt install sysstat`.
2. **Server-side attribution.** Each run's server timings are histogram deltas
   between a scrape taken just before the run and one taken after the server
   settles (no running or waiting requests, and an unchanged request count across two
   scrapes). Validation requires every per-request family's count delta to equal the
   run's prompt count. vLLM 0.29.0's `vllm bench serve` sends no extra requests by
   default (`--ready-check-timeout-sec 0`, `--num-warmups 0`). The per-token ITL
   histogram is reported, not count-checked.
3. **Request-ID prefix.** `--request-id-prefix <series>-<condition>-[rep-NN-]` provides
   client-side traceability only. Prometheus histograms carry no request IDs, and
   `--enable-log-requests` is not enabled: it is not a Milestone 2 control and it
   writes prompts to the log. Server records are attributed by scrape bracketing.
4. **Definitions.**
   - Client pre-first-token share = Σ TTFT / Σ E2E over a run's successful requests.
   - Server share = (queue + prefill) / (queue + inference), from per-run means.
   - Dominant component = pre-first-token if mean queue + prefill > mean decode,
     otherwise decode.
   - Telemetry run window = [finish − vLLM's `duration`, finish], so client startup
     is excluded.
5. **Validation.** Warm-ups are validated automatically. Telemetry must cover the
   series with no gap over 1 s (GPU) or 3 s (CPU). A sampler that exits early, or a
   failed scrape, fails the series.
6. **Contention tooling (for Phase C).**
   - `memory_cap_mib` bounds the generator's total VRAM, including its CUDA context
     (512 MiB reserved). It is checked with `nvidia-smi --query-compute-apps`.
   - The default fp16 matrix is 2048×2048, and the duty cycle is fixed-time within
     each period (default 100 ms).
   - Achieved duty must be within ±0.05 of the configured value.
   - Sham allocates the same memory but runs zero steps.
   - Whether to use contention, and at what intensity, remain Phase C addendum
     decisions.

## Deliverables and acceptance

**Deliverables:** this protocol and the Phase C addendum, the configs, validated raw
data, two or three figures, and a technical note. The note covers supported
attribution, competing explanations, measurement overhead, and the limits of the
single-GPU setup.

**Acceptance:** the report names at least one regime transition with an
intervention, a predicted signature, observed evidence, and a plausible falsification
check. If attribution remains ambiguous, report that clearly.

**Effort:** approximately 15–22 focused hours, including the tooling.
