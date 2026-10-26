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

## Amendments

None.

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
- The C8 block was not run, so predictions 8–10 are not scored.
