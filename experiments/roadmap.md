# BottleneckShift roadmap: Milestones 2–9

**Drafted October 2, 2026**
**Reviewed state:** `main` at `20602610df9e30401c1d193ae668c5b6d55c0a48`
**Updated October 3, 2026:** Milestone 2 completed; per-milestone protocols moved to
`experiments/milestoneN.md`.

This is a plan, not a result; completed milestones link to their reports. Each
milestone needs a reviewed protocol and configs before any measured run.

## Starting point

Milestone 1 ran `Qwen/Qwen2.5-0.5B-Instruct` with vLLM 0.29.0 on one RTX A4000. It
compared maximum client concurrency 1 and 8 in forward and reverse order, with three
repetitions per condition. The [published report](../reports/milestone1-20261001T205137Z.md)
found approximately 6.6× higher output-token throughput at concurrency 8 and
approximately 16% higher median request latency. Prefix caching, a single
concurrency-1 warm-up, an unpinned model revision, and implicit sampling defaults
limit how much of the effect can be attributed to concurrency alone.

The historical pair sent no request-level temperature, so the server applied the
model's `generation_config.json` defaults (temperature 0.7, top_p 0.8, top_k 20,
repetition_penalty 1.1). Its TPOT and end-to-end latency are therefore not
greedy-decoding measurements.

The repository already contains a runner, manifests, validation, analysis, tests,
the Milestone 1 report, and the controlled configs
`configs/milestone1-controlled-forward.toml` and
`configs/milestone1-controlled-reverse.toml`. These pin the model and tokenizer
revision, disable prefix caching, set explicit sampling and generation behavior, and
warm both concurrency levels. They produced the Milestone 2 pair, reported in
[`reports/milestone2-20261003T034456Z.md`](../reports/milestone2-20261003T034456Z.md).

## Milestone 2 — Confirm the concurrency result under explicit controls

**Status: completed October 3, 2026.** Protocol:
[`experiments/milestone2.md`](milestone2.md). Report:
[`reports/milestone2-20261003T034456Z.md`](../reports/milestone2-20261003T034456Z.md).

**Question:** Which Milestone 1 effects persist when cache and generation behavior
are controlled?

Pair `milestone2-20261003T034456Z`: 1,200 measured requests validated. C8 gives ~6.5×
output throughput at ~18% higher median E2E latency in both orders; forward and
reverse agree within 0.1% except C8 TTFT. Relative to Milestone 1, C8 TTFT is 21–42%
higher; TPOT is 3–4% lower and throughput 2–4% higher. Reconstructed E2E matches
vLLM's own within 0.001 ms.

## Milestone 3 — Diagnose one bottleneck shift

**Status: Phases B–C completed October 3, 2026; Phase D pre-registered October 4.**
Protocol: [`experiments/milestone3.md`](milestone3.md). Reports:
[Phase B](../reports/milestone3-20261003T180358Z-phase-b.md),
[Phase C](../reports/milestone3-20261003T201214Z-phase-c.md).

- **Phase B:** decode dominates E2E latency in both regimes, so the pre-registered
  shift did not occur.
- **Phase C crossover probe:** past C8 the pre-first-token share *falls*
  (0.44 → 0.24 at C32). Throughput peaks at C16, and the cost moves into per-step
  time (ITL p50 4.9 → 55 ms), consistent with chunked prefill inflating generation
  steps.
- **Phase C contention:** GPU contention has a distinct signature: C1 slows too,
  queue time stays flat, prefill slows more than decode, and the front end is
  unaffected.
- **Phase D (pre-registered):** a C8–C32 concurrency sweep plus a scheduler token-budget
  intervention (512, 2048, 8192 tokens per step), measuring step sizes from vLLM's
  per-step histogram to test the chunked-prefill explanation directly.

**Question:** Under which workload conditions does the dominant source of
client-visible delay change, and what measurements distinguish the causes?

**Outline:** (A) tooling: `/metrics` scrapes, GPU and CPU telemetry, request-ID
matching, analysis, and validation; (B) two pre-registered workload regimes,
prefill-heavy (2048/32, 100 prompts) and decode-heavy (64/512, 50 prompts), each at
C1 and C8 in both orders, with Milestone 2's controls; (C) one contention
intervention, chosen after Phase B and pre-registered in an addendum, with sham and
recovery controls. GPU contention is preferred, and Phase A builds its load
generator; CPU contention is unsuitable on the 4-vCPU host.

**Acceptance:** the report names at least one regime transition with an
intervention, a predicted signature, observed evidence, and a plausible
falsification check. If attribution remains ambiguous, report that clearly.

**Effort:** approximately 15–22 focused hours, including the tooling.

## Milestone 4 — Test adaptation only if Milestone 3 supports it

**Question:** Can measurements of changing resource conditions improve backend
selection for a user-facing inference objective?

**Entry gate:** Milestone 3 yields reproducible, distinguishable operating states.
There must be two credible backend choices or controlled backend states, and enough
observations to evaluate a selection policy without inventing rewards.

**Work**

1. Define a concrete reward, such as tail TTFT under a throughput constraint, with a
   stated switching cost. Record what a selector observes before deciding and what it
   learns after a request.
2. Start with a trace replay that preserves temporal order and the limits of
   observed feedback. Compare a fixed choice, a random choice, and one simple
   adaptive policy. Identify any counterfactual rewards the trace cannot supply.
3. If replay cannot support a fair comparison, either collect a small two-backend
   experiment or stop the extension. Evaluate changing conditions, fairness across
   clients if applicable, and switching overhead.
4. Report both gains and failure cases. Do not treat peer or server selection in
   content delivery as equivalent to LLM routing: cache affinity, queueing, request
   lengths, and token streaming change the decision.

**Deliverables:** a small reproducible policy comparison or a documented decision to
defer it; a concise results section with limitations.

**Acceptance:** every policy sees the same permitted information and is evaluated on
the same workload sequence. The result makes no novelty claim without a focused
prior-work check.

**Effort:** approximately 8–12 focused hours if the entry gate passes. Otherwise use
this time to strengthen Milestone 3 and the final report.

## Milestones 5–9 — Full factor studies

These are the original planned stages, kept as later milestones. Milestone 3 takes a
deliberately narrow slice of them: two workload regimes and one contention
intervention. Each milestone below widens one factor into its own pre-registered
study, reusing the Milestone 2 controls and the Milestone 3 tooling (server-side
metrics, telemetry, request matching). None is scheduled within the current budget.

| Milestone | Factor | Conditions |
|---|---|---|
| 5 | Workload shape | short/long input and output factorial |
| 6 | Network | controlled latency/bandwidth/loss shaping |
| 7 | Host contention | controlled CPU and memory pressure |
| 8 | GPU contention | isolated, controlled competing GPU work |
| 9 | Bottleneck shift | selected crossed factors from milestones 5–8 |

Notes for planning:

* **Milestone 5** generalizes Milestone 3's two regimes into a full factorial; it
  one config per cell (as in Milestone 3) works for a small factorial, but a larger
  one needs an optional per-condition workload override in the runner.
* **Milestone 6** needs the client off the server host, or shaping applied to a
  network path the client actually uses; loopback shaping on one host is not
  representative.
* **Milestone 7** inherits the 4-vCPU constraint: pin client, server, and stressor,
  or use a separate client host.
* **Milestone 8** needs a competing GPU workload with a fixed, recorded intensity, and
  a check that it does not exhaust VRAM for the vLLM server.
* **Milestone 9** crosses only the factors that milestones 5–8 show to matter, rather
  than running the full grid.

Whichever factor Milestone 3 chooses as its intervention gives that milestone a head
start. Its Milestone 3 data are a pilot, not a substitute for the full study.

## Priority, budget, and final artifact

Milestone 2 is complete; make Milestone 3 the core contribution. Milestone 4 is
optional. Milestones 5–9 are beyond the current budget.

Planned effort is roughly 27–40 focused hours: 4–6 for Milestone 2, 15–22 for
Milestone 3, and 8–12 for Milestone 4. Hours already spent on Milestone 1 count
against the original 40–50-hour project budget. If the combined total would exceed
it, drop Milestone 4 first.

The final public artifact should contain reproducible configs, raw-data provenance,
analysis scripts, approximately four strong figures, a clear limitations section,
and a 5–7 page technical report if time permits. Use only public resources and
independent results.
