# Milestone 2 protocol: concurrency under explicit controls

> **Provenance.** This protocol was pre-registered in `experiments/roadmap.md`
> (Milestone 2 section) and in the README's "Controlled follow-up protocol" section.
> It was last changed before collection in commit `bfa0af1` (October 2, 2026, 18:09
> UTC); the measured pair started at 03:44:56 UTC on October 3. This file moves that
> text here after the run, without substantive change, so each milestone has one
> protocol document. The cache-policy decision (below) was made before the run and
> recorded in the private run plan.

## Question and hypothesis

Which Milestone 1 effects persist when cache and generation behavior are controlled?

Milestone 1 found ~6.6× output throughput and ~16% higher median end-to-end latency
at concurrency 8. Prefix caching (hit rate rising to ~75–78% across reused seed-2027
prompts), a single concurrency-1 warm-up, an unpinned model revision, and implicit
sampling defaults limited attribution to concurrency alone. The historical pair sent
no request temperature, so the model's `generation_config.json` defaults applied
(temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1).

The expected signature is the Milestone 1 hypothesis: higher TTFT and end-to-end
latency and higher throughput at C8, stable output lengths, and zero errors. A
mismatch with Milestone 1 is a result to explain, not grounds to discard the run.

## Configs and controls

Milestone 2 runs `configs/milestone1-controlled-forward.toml` and
`configs/milestone1-controlled-reverse.toml`. They keep their names, which are
recorded in manifests. The workload, server, repetitions, and cooldown are unchanged
from Milestone 1 (256/128 tokens, 100 prompts, seed 2027, three repetitions, 15 s).
Forward and reverse differ only in plan name and condition order.

| Control | Setting |
|---|---|
| Model and tokenizer | `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775` |
| Prefix caching | disabled (`--no-enable-prefix-caching`) |
| Generation config | `--generation-config vllm` (model defaults ignored) |
| Request sampling | `temperature = 0.0`, `top_p = 1.0` (greedy) |
| Warm-ups | 10 prompts at C1, then 10 at C8 |
| Percentile metrics | `ttft,tpot,itl,e2el`, so vLLM saves `median_e2el_ms` |

**Cache policy (Option A).** Prefix caching is off, so every request does a full
prefill and repetitions cannot reuse cached work from earlier runs. Reusing seed 2027
is then intentional: every repetition sees identical prompts. TTFT may be higher than
in Milestone 1, which was partly served from cache. Cache-on alternatives were set
aside: a fresh seed per run (B), deliberate pre-warming (C), and uncontrolled
caching (D). Cache behavior is a later factor, not part of the concurrency question.

## Procedure

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
     `warmup/c*/requests.json`;
   * each control confirmed in both `server.log` files.

   Compare repetitions and execution orders. Keep anomalous observations, including
   the historical reverse C8 outlier.
4. Compare controlled results with the historical pair as **different protocols**.
   Report changes in TTFT, TPOT, end-to-end latency, and output throughput without
   pooling their repetitions.
5. Audit request timing before making fine-grained phase claims.
   `analysis/compare.py` reconstructs end-to-end latency as TTFT plus summed
   inter-token intervals; streamed chunks can hold several tokens, so an ITL is not a
   per-token decode step. Compare the reconstruction with vLLM's own
   `median_e2el_ms`, reported as `vllm_median_e2el_ms`.

## Deliverables and acceptance

**Deliverables:** two validated raw series and checksums; one comparison figure; a
short report stating replicated effects, changed effects, uncertainty, and remaining
confounders.

**Acceptance:** both series complete, all measured requests validate, and each
reported number traces to raw results and a fixed protocol.

**Effort:** approximately 4–6 focused hours after GPU setup.

## Interpretation guardrails

The controls change several things at once (caching, sampling, warm-ups, model
pinning), and the VM allocation differs from Milestone 1. Differences from Milestone 1
are protocol differences; do not attribute them to a single control. No resource
contention is manipulated.

## Outcome

Completed October 3, 2026: pair `milestone2-20261003T034456Z`, 1,200 measured
requests validated. C8 gives ~6.5× output throughput at ~18% higher median E2E
latency in both orders. See the
[Milestone 2 report](../reports/milestone2-20261003T034456Z.md) for measurements,
interpretation, limitations, and the archive checksum. Raw artifacts remain external.
