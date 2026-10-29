# Milestone 5 pilot protocol: prompt length versus token budget

> **Status: pre-registered October 24, 2026,** before any Milestone 5 data. Any change
> to the conditions, series order, predictions, or thresholds after this file merges
> must be recorded below as a dated amendment, committed before the affected data are
> collected.

## Question

Does the number of tokens an engine step schedules set that step's cost, and does the
scheduler's token budget therefore cap the tail of inter-token latency (ITL) as prompts
grow longer?

## Motivation

This pilot follows directly from Milestone 3 Phase D
([report](../reports/milestone3-20261006T022744Z-phase-d.md)). There, the token budget
moved ITL, TTFT, and throughput by large margins at C32, but the pre-registered
step-size explanation was not supported. Two problems prevented a test:

1. **A confound.** Every Phase D prompt was 2048 tokens, equal to the default budget,
   so prompt length and budget were never varied separately.
2. **The wrong instrument.** vLLM's `vllm:iteration_tokens_total` histogram credits a
   request's whole prompt at its first token, so it cannot show per-step chunk sizes,
   and the manipulation check failed as specified.

This pilot varies prompt length at two budgets and records each engine step's
scheduled tokens directly. The thresholds below are informed by Phase D's numbers; the
hypothesis and its predictions were written before any Milestone 5 data.

## Hypothesis

**H (step size):**
1. An engine step's wall time is set mainly by the number of tokens it schedules.
2. The budget caps the tokens per step. A prompt longer than B is split into chunks
   of about B tokens, while a prompt shorter than B adds about one prompt's worth of
   tokens to the step that admits it.

Hence, at a fixed budget, the ITL tail grows with prompt length up to B and then stops
growing. Concurrency changes how many steps carry prefill, which moves the median, but
not the size of the largest steps, which sets the tail. (In steady state each request
lives about 32 steps, a prefill plus 31 decodes, so at C32 about one new prompt enters
per step and nearly every step carries prefill; at C8 about one step in four does.)

## Fixed controls (inherited from Phase D)

The same model and pinned revision (`7ae557604adf67be50417f59c2c2f167def9a775`), vLLM
0.29.0, prefix caching off, `--generation-config vllm`, greedy sampling (`temperature
0.0`, `top_p 1.0`), `percentile_metrics` including `e2el`, TP 1, GPU memory utilization
0.90, seed 2027, 100 prompts per run, 32 output tokens, three repetitions, 15-second
cooldowns, 32-prompt warm-ups at the measured concurrency, request-ID prefixes,
`/metrics` scrapes, and GPU and CPU samplers. The host is a Hyperstack RTX A4000 with
driver 580, as in Phase D.

## Design

* **Factors:** prompt length L ∈ {1024, 2048, 4096} tokens × token budget
  B = `max_num_batched_tokens` ∈ {2048, 8192}, at concurrency C32 (primary block) and
  C8 (secondary block).
* **Configs:** `configs/milestone5-len{L}-budget{B}-c{C}.toml`, one per cell, copied
  from `configs/milestone3-budget-2048.toml`. Only the name, warm-up level, budget,
  input length, step log, and single condition (`c32` or `c8`) differ.
* **One server per series.** A series is one cell: a warm-up, then three measured
  repetitions.

### Instrument: per-step scheduler log

`src/bottleneckshift/vllm_steplog.py` is a vLLM general plugin, registered through the
`vllm.general_plugins` entry point. vLLM 0.29.0 loads general plugins in the engine
process before it creates the scheduler. The plugin does nothing unless
`BOTTLENECKSHIFT_STEP_LOG` is set, which the runner does only for configs with
`[telemetry] step_log = true`. Earlier configs keep byte-identical plans.

When enabled, it wraps `Scheduler.schedule()` and writes one JSON line to
`steps.jsonl` for every step that schedules at least one token:

| Field | Meaning |
|---|---|
| `t_start`, `t_end` | Wall-clock time (Unix seconds) before and after scheduling |
| `tokens` | Total scheduled tokens (`SchedulerOutput.total_num_scheduled_tokens`) |
| `prefill_tokens`, `decode_tokens` | Tokens counted as prompt tokens (up to each request's prompt length, from its computed-token count before the step), and the rest |
| `requests`, `prefill_requests` | Scheduled requests, and how many of them received prefill tokens |

Steps are attributed to runs by `t_start` falling between a run's start and finish
times. `scripts/validate_results.py` requires a readable log with at least one step in
every measured run, and reports each run's step count, largest step, and logged
prefill-token total against the expected 100 × L.

### Series order

**C32 block (primary):**

1. `m5-anchor-before`: L2048, B2048 (also reproduces Phase D)
2. L1024, B2048
3. L4096, B8192
4. L4096, B2048
5. L1024, B8192
6. L2048, B8192
7. `m5-anchor-after`: L2048, B2048 (brackets drift)

**C8 block (secondary):** run only after all seven C32 series have validated, and
only if GPU time allows.

8. L1024, B2048
9. L4096, B8192
10. L4096, B2048
11. L1024, B8192
12. L2048, B8192
13. L2048, B2048 (also reproduces Phase D at C8)

## Metrics

* **ITL p50 and p99:** per run, from the raw inter-token arrays of all requests; the
  cell value is the median across the three repetitions (as in Phase D).
* **Mean E2E and output throughput:** per run as in Phase D; the cell value is the
  median across repetitions.
* **Step interval:** for consecutive logged steps within a run, the difference of
  their `t_start` values. The last step of each run is dropped, and intervals over 1 s
  are excluded as idle gaps.
* **Large and small steps:** steps scheduling at least 1024 tokens, and steps
  scheduling at most 64 tokens.

## Predictions

C32 block:

1. **Manipulation and instrument check.** In every series, no step schedules more than
   B tokens, and each measured run's logged prefill-token total is within 1% of
   100 × L.
2. **Primary: step cost tracks step size.** Pooling all validated series in the C32
   block, the median interval of large steps (≥1024 tokens) is at least 3× the median
   interval of small steps (≤64 tokens), and the Spearman correlation between a step's
   tokens and its interval is at least 0.7. In each series with at least 30 large and
   30 small steps, the 3× ratio also holds.
3. **The tail grows with prompt length up to the budget.** At C32 and B2048, ITL p99
   for L1024 is at least 25% lower than for L2048, and ITL p99 for L4096 is within
   ±20% of L2048.
4. **A larger budget raises the tail.** At C32, for L2048 and L4096, ITL p99 at B8192
   is at least 1.5× ITL p99 at B2048.
5. **The median is low at the larger budget, and high at the default.** At C32, for
   L2048 and L4096, ITL p50 is below 10 ms at B8192 and at least 20 ms at B2048.
6. **Reproduction.** `m5-anchor-before` is within 5% of Phase D's `budget-2048-before`
   at C32 on ITL p50, mean E2E, and throughput. Because the anchor runs with the step
   log on, this also bounds the plugin's overhead. (Known differences: Phase D ran C8
   before C32 on the same server and warmed up at both levels; the anchor runs C32
   alone with a C32 warm-up.)
7. **Drift.** `m5-anchor-after` is within 2% of `m5-anchor-before` on mean E2E and
   throughput.

C8 block (scored only if all six C8 series complete and validate):

8. **The same tail shape at C8.** At C8 and B2048, ITL p99 for L1024 is at least 25%
   lower than for L2048, and ITL p99 for L4096 is within ±20% of L2048.
9. **Concurrency moves the median.** At C8 and B2048, ITL p50 is below 10 ms for every
   L, while prediction 5 expects at least 20 ms at C32 for L2048 and L4096.
10. **Reproduction at C8.** The C8 L2048/B2048 series is within 5% of Phase D's
    `budget-2048-before` at C8 on ITL p50, mean E2E, and throughput.

Prediction 1 applies to every series run. Prediction 2 is also evaluated, pooled, on
the C8 block if it completes. L1024 at B8192 is exploratory: Phase D did not test it,
and H makes no clear prediction for it.

## Falsification

- **If prediction 1 fails** in a series, that series is excluded from predictions
  2–10 and reported. If it fails in either anchor or in most series, the pilot is
  inconclusive.
- **If prediction 1 holds but prediction 2 fails,** H is rejected: the number of
  scheduled tokens does not explain per-step cost on this setup.
- **If predictions 1 and 2 hold but prediction 3 or 8 fails,** step size explains
  per-step cost, but the budget cap does not set the ITL tail as H predicts. The report
  says which part failed. Predictions 4, 5, and 9 test secondary consequences, and a
  failure there qualifies H without rejecting it.

## Session rules

* One GPU session. Run the C32 block in the order above; then the C8 block in its
  order, if time allows. Pre-registration merges before any measured run.
* Validate each series as soon as it finishes. A failed series reruns under a new
  series ID; completed series are kept, and nothing is spliced.
* If the C8 block is stopped partway, its completed series are kept and reported, but
  predictions 8–10 are not scored.
* Archive and checksum before teardown, and verify the checksum on the laptop.
* Estimated time: about 3–5 minutes per series (L4096 runs take about twice as long
  as L2048), so about 30 minutes per block, plus about 30 minutes of host setup.

## Analysis plan

Per-cell ITL p50/p99, mean E2E, throughput, server phases, and TTFT come from the
existing analysis (`analysis/phases.py`). Step metrics (interval distributions by
token bin, the large/small ratio, the Spearman correlation, and the largest step) are
computed from `steps.jsonl` by an analysis addition committed before the report.
Exploratory checks, labeled post hoc in the report: whether step interval depends on
prompt length at a fixed step size (attention over longer contexts), and how the
share of prefill-carrying steps differs by L and B.

## Interpretation guardrails

* One model, one GPU, synthetic prompts, closed-loop load. The results describe this
  setup and do not generalize on their own.
* Step interval includes scheduler and output-processing time, not only GPU kernel
  time.
* Repetitions replay the same seed-2027 prompts, so they measure run-to-run variation,
  not prompt-to-prompt variation.

## Part 2 addendum (pre-registered 2026-10-26, before any Part 2 data)

### Why

Part 2 follows from the pilot's C32 results
([report](../reports/milestone5-20261025T004640Z-pilot.md)), which raised three
questions:
1. **Packing.** The scheduler fills a step with several short prompts, so prompt length
   below the budget changed how often steps carried prefill, not how large they were.
   The C8 block (predictions 8–10) tests the same tail and median predictions where
   fewer prompts wait.
2. **Context length (post hoc in the pilot).** Steps of the same size cost more for
   longer prompts (45, 55.5, and 57.7 ms for 2048-token steps at L1024, L2048, and
   L4096). Part 2B tests that directly, with fixed chunk sizes at known context
   positions.
3. **Timing.** Step intervals correlated slightly better with the previous step than
   with the current one, which fits vLLM 0.29.0 scheduling the next step before the
   current step's results are processed. Part 2 adds executor-side timing.

### Part 2A: the C8 block

Run as pre-registered above (series 8–13, in that order). Predictions 1, 2, and 8–10
and their thresholds are unchanged, and predictions 1 and 2 keep their definitions
(step interval as schedule-to-schedule time).

### Part 2B: context-length test

* **Configs:** `configs/milestone5-ctx-len8192-budget{1024,2048}-c1.toml`, copied from
  `configs/milestone5-len2048-budget2048-c32.toml`. Only the name, warm-up level,
  budget, input length (8192), and condition (C1) differ.
* **Design:** one request at a time (C1), so every prefill step carries one chunk of
  one prompt. Each 8192-token prompt is prefilled in 8192 / B chunks of exactly B
  tokens, at context positions 0, B, 2B, …, 8192 − B. With 32-token outputs, 100
  prompts per run, and three repetitions, each series gives 300 prompts × 8192 / B
  chunks.
* **Series:** `ctx-budget1024`, then `ctx-budget2048`.

### Instrument additions

The plugin (`src/bottleneckshift/vllm_steplog.py`) now also records, for each step:
- a sequence number (`step`);
- each prefill chunk's context position (computed tokens before the step) and size
  (`chunks`);
- the scheduled requests' computed tokens before the step (`kv_tokens`).

It also wraps `Scheduler.update_from_output()`, logging the time a step's results
arrive (`t_output`), matched to the step by its scheduler output. The pilot's fields
and their meanings are unchanged.

**Execution time:** t_output[n] − max(t_end[n], t_output[n − 1]). A step can start once
it has been scheduled and the previous step's results have arrived, so this holds
whether or not the engine schedules the next step early. When it does, the GPU may
start a step slightly before the previous step's results reach the scheduler, so in
that case this slightly underestimates execution time.

**Full chunk:** a step with exactly one prefill chunk of exactly B tokens.

### Predictions

11. **Instrument check (Part 2B).** In both context series:
    - every step that carries prefill holds exactly one full chunk;
    - each prompt's chunks sit at context positions 0, B, …, 8192 − B;
    - each run's logged prefill total equals 100 × 8192;
    - every logged step has an output time.
12. **Primary for Part 2: context raises the cost of a fixed-size chunk.** In each
    context series, the median execution time of full chunks at the deepest position
    (context 8192 − B) is at least 1.2× that at context 0, and the Spearman correlation
    between a full chunk's context position and its execution time is at least 0.6.
13. **Cost grows with chunk size.** At context 0, the median execution time of a
    2048-token chunk (B2048 series) is between 1.4× and 2.5× that of a 1024-token
    chunk (B1024 series). The range allows for a fixed per-step overhead (below 2×)
    and for attention within the chunk, which grows faster than linearly (above 2×).
14. **Executor timing removes the lag.** Pooled over the C8 block, the Spearman
    correlation between a step's tokens and its execution time is at least as high as
    the correlation between its tokens and its schedule interval. Scored only if all
    six C8 series complete and validate.

### Falsification

- **If prediction 11 fails** in a context series, predictions 12 and 13 are not scored
  for it, and the report says why.
- **If prediction 11 holds but prediction 12 fails,** the pilot's post hoc reading
  (same-size steps cost more at longer contexts) is not supported on this setup.
- **Prediction 13** tests whether cost grows roughly in proportion to chunk size; a
  failure qualifies the step-size account without rejecting it.
- The C8 block is scored by the original rules above.

### Session rules

* One GPU session, in this order: `ctx-budget1024`, `ctx-budget2048`, then the C8 block
  (series 8–13). Pre-registration merges before any Part 2 run.
* Validate each series as it finishes. A failed series reruns under a new series ID;
  nothing is spliced. If the session stops partway, completed series are kept and
  reported, and predictions that depend on missing series are not scored.
* Archive and checksum before teardown, and verify the checksum on the laptop.
* Estimated time: about 3–4 minutes per context series and about 4 minutes per C8
  series, so about 35 minutes of collection plus about 30 minutes of host setup.

## Amendments

- **2026-10-26 (before any Part 2 data):** the Part 2 addendum above. The step log
  gains the fields listed there; the pilot's fields and definitions are unchanged. The
  C8 block will run with the extended log.

## Outcome

**C32 block completed October 25, 2026:** session `milestone5-20261025T004640Z`, seven
series, 2,100 measured requests validated. See the
[report](../reports/milestone5-20261025T004640Z-pilot.md).

- Predictions 1, 2, 4, 5, 6, and 7 were confirmed. Step cost tracks step size: the
  pooled large-step median interval was 9.8× the small-step one, with a Spearman
  correlation of 0.87.
- Prediction 3 was not confirmed. L4096 was within +11% of L2048 at B2048, as
  predicted, but L1024's ITL p99 was only 18.7% lower (threshold 25%), because the
  scheduler packed two 1024-token prompts into one 2048-token step. Under the
  falsification rule, step size explains per-step cost, but H's account of the tail for
  prompts shorter than the budget is not supported.
- The C8 block was not run in this session, so predictions 8–10 were not scored here;
  they were scored in Part 2.

**Part 2 completed October 29, 2026:** session `milestone5p2-20261029T000043Z`, eight
series, 2,400 measured requests validated. See the [Part 2 report](../reports/milestone5p2-20261029T000043Z-part2.md).

- Predictions 11, 12 (primary for Part 2), 13, and 14 were confirmed. A fixed-size
  chunk's cost grows with its context (deepest/first 1.44× at B1024 and 1.62× at B2048;
  Spearman 0.96), and a 2048-token chunk costs 1.67× a 1024-token chunk at context 0.
- In the C8 block, predictions 1, 9, and 10 were confirmed. Prediction 8 was not: the
  L1024 tail was only 11.2% lower, for the same packing reason as at C32. Prediction 2
  was not confirmed on the C8 block: the large/small ratio held (9.0×), but the pooled
  Spearman correlation was 0.57, against 0.7, so by the pre-registered rule H's first
  clause is not supported at C8 by that statistic.
