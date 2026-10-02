# BottleneckShift roadmap: Milestones 2–9

**Drafted October 2, 2026**
**Reviewed state:** `main` at `20602610df9e30401c1d193ae668c5b6d55c0a48`

This is a plan, not a result. Each milestone needs a reviewed protocol and configs
before any measured run.

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
warm both concurrency levels. They have **not yet produced a reported measured pair**.

## Milestone 2 — Confirm the concurrency result under explicit controls

**Question:** Which Milestone 1 effects persist when cache and generation behavior
are controlled?

Milestone 2 runs the merged `milestone1-controlled-*` configs. They keep their names,
which are recorded in manifests.

**Work**

1. Review the exact commands from both controlled configs; freeze the commit, GPU
   setup, and configs.
2. Run both series on one stable GPU instance, with fresh series IDs and the same
   host. Preserve warm-ups, server logs, raw JSON, and any failed requests. The runner
   stops a series on a nonzero benchmark exit, and the validator rejects any
   non-empty error string. A series with failed requests is therefore recorded and
   reported, not analyzed as valid.
3. Validate before analysis:
   * archive checksum with `shasum -a 256 -c`, separately from the validator;
   * manifest completeness, return codes, error strings, array lengths, and realized
     token lengths with `scripts/validate_results.py`;
   * warm-ups separately, because the validator checks only measured runs: apply
     `validate_result` from `src/bottleneckshift/validation.py` to each
     `warmup/c*/requests.json`, as was done for the Milestone 1 archive.

   Compare repetitions and execution orders. Keep anomalous observations, including
   the historical reverse C8 outlier.
4. Compare controlled results with the historical pair as **different protocols**.
   Report changes in TTFT, TPOT, end-to-end latency, and output throughput without
   pooling their repetitions.
5. Audit request timing before making fine-grained phase claims.
   `analysis/compare.py` reconstructs end-to-end latency as TTFT plus summed
   inter-token intervals, ending at the last streamed chunk. The Milestone 1 report
   already notes that streamed chunks can hold several tokens, so an ITL is not a
   per-token decode step. Check the reconstruction against vLLM's own end-to-end
   latency. vLLM writes `median_e2el_ms` only when `e2el` is included in
   `--percentile-metrics`, and the runner does not pass that flag (the default is
   `ttft,tpot,itl`). This needs an opt-in config field with a test, so the
   historical and controlled command plans stay unchanged.

**Deliverables:** two validated raw series and checksums; one comparison figure; a
short report stating replicated effects, changed effects, uncertainty, and remaining
confounders.

**Acceptance:** both series complete, all measured requests validate, and each
reported number traces to raw results and a fixed protocol. A mismatch with
Milestone 1 is a result to explain, not grounds to discard the run.

**Effort:** approximately 4–6 focused hours after GPU setup.

## Milestone 3 — Diagnose one bottleneck shift

**Question:** Under which workload conditions does the dominant source of
client-visible delay change, and what measurements distinguish the causes?

**Design**

1. Pre-register two contrasting workload regimes: a longer-input/shorter-output case
   emphasizing prefill and a shorter-input/longer-output case emphasizing decode. Run
   each at C1 and C8, with the Milestone 2 model, GPU, framework, cache policy, and
   sampling controls held fixed.
2. Select **one** contention intervention after baseline data indicates an
   informative regime. Host CPU contention is a candidate only if client and server
   effects can be separated; use process-level measurements or isolate the client.
   GPU contention or network shaping can replace it if the setup affords a cleaner
   intervention. Do not run a broad grid of all factors.
3. State predicted signatures before collection: which of TTFT, inter-token timing,
   queue/processing time, throughput, and resource counters should move, and which
   should remain stable? Include a negative control or recovery baseline.
4. Collect timestamped client and server measurements and resource telemetry.
   Repeat and vary order. Match intervals and request IDs where the instrumentation
   allows it. Treat utilization and counters as corroboration, not proof of cause.
5. Run a small set of crossed conditions only where the single-factor results
   motivate them. Present the transition in a figure that shows latency
   contributions, throughput, and supporting telemetry together.

**Tooling prerequisites** (gaps in the current code; implement with tests before
collection):

* **Per-condition workload.** `benchmark_command` in `scripts/run_experiment.py` reads
  one `[workload]` table per config; conditions vary only `max_concurrency`. Either
  use one config per regime or add an optional per-condition workload override.
* **Server-side timing.** Nothing server-side is captured yet. Scrape vLLM's
  Prometheus `/metrics` endpoint before and after each run, covering
  `vllm:request_queue_time_seconds`, `vllm:request_prefill_time_seconds`, and
  `vllm:request_decode_time_seconds`, and store the scrapes next to `requests.json`.
* **Resource telemetry.** Add a sidecar sampler (`nvidia-smi --query-gpu=… -lms`,
  `pidstat`/`mpstat`) whose output files are listed in the run manifest.
* **Request matching.** `vllm bench serve --request-id-prefix` exists in 0.29.0, and
  `requests.json` includes `start_times`. Use them to align client requests with
  server logs and telemetry intervals.
* **Host constraint.** The Milestone 1 host had 4 vCPUs, with client and server on
  the same host. CPU contention there needs `taskset` pinning of client, server, and
  stressor, or a separate client host. Otherwise prefer GPU contention or network
  shaping.

**Deliverables:** reviewed protocol and configs; validated raw data; two or three
figures; a technical note identifying supported attribution, competing explanations,
measurement overhead, and limits of the single-GPU setup.

**Acceptance:** the report names at least one regime transition with an
intervention, a predicted signature, observed evidence, and a plausible
falsification check. If attribution remains ambiguous, report that clearly.

**Effort:** approximately 15–22 focused hours, including the tooling prerequisites.

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
  needs the per-condition workload support listed under Milestone 3.
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

Finish Milestone 2, then make Milestone 3 the core contribution. Milestone 4 is
optional. Milestones 5–9 are beyond the current budget.

Planned effort is roughly 27–40 focused hours: 4–6 for Milestone 2, 15–22 for
Milestone 3, and 8–12 for Milestone 4. Hours already spent on Milestone 1 count
against the original 40–50-hour project budget. If the combined total would exceed
it, drop Milestone 4 first.

The final public artifact should contain reproducible configs, raw-data provenance,
analysis scripts, approximately four strong figures, a clear limitations section,
and a 5–7 page technical report if time permits. Use only public resources and
independent results.
