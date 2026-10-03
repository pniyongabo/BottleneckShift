# Milestone workflow: what we repeat, what went wrong, what to change

Written October 3, 2026, after Milestones 1–3 Phase B (October 1–3). It summarizes the
loop each milestone went through, where the time went, and the adjustments now
adopted. The rules themselves live in [`AGENTS.md`](../AGENTS.md); this file explains
them.

## The loop we repeated

Each milestone went through the same seven stages.

| # | Stage | Where | Typical artifacts |
|---|---|---|---|
| 1 | **Plan and decide.** Question, regimes, controls, open decisions put to the maintainer. | laptop | `experiments/roadmap.md` entry |
| 2 | **Pre-register.** Protocol with predictions, committed before any data. | laptop, PR | `experiments/milestoneN.md` |
| 3 | **Build.** Runner, config, validator, and analysis changes, with synthetic tests. | laptop, PR | code, `configs/`, `tests/` |
| 4 | **Run plan.** Host steps for the GPU session. | laptop (local only) | `results/archives/.../*-run-plan.md` |
| 5 | **GPU session.** Driver fix, setup, measured series, validation, archive. | rented VM | raw series, `*.tar.gz` + SHA-256 |
| 6 | **Verify and analyze.** Checksum, extract, re-validate, analysis, figure. | laptop | CSVs, figure |
| 7 | **Report and sync docs.** Report against the predictions; update README, roadmap, and protocol outcome together. | laptop, PR | `reports/`, README, roadmap |

### What each milestone took

| Milestone | Pull requests | GPU session | Notes |
|---|---|---|---|
| 1 | several (baseline build, report, validation hardening) | one pair | Cache-enabled; sampling implicit; outlier in reverse C8 |
| 2 | #6–#9 (controls audit, roadmap ×2, percentile metrics, report) | one pair, ~14 min measured | Run plan grew to 12 verbatim steps |
| 3 | #10–#12 + report (protocol, Phase A tooling, flaky-test fix) | four series, ~47 min | Lean three-script plan; two small failures debugged live |

## Where the time went

- **Over-preparation.** The first Milestone 3 run plan was 13 verbatim steps
  (~800 lines) and included separate smoke, contention, and overhead shakedowns. It
  also re-checked vLLM source details and simulated the plan's own scripts against
  synthetic data. The lean replacement was three scripts (~90 lines) and still caught
  every real problem.
- **PR fragmentation.** Roadmap edits, small runner options, and test fixes each went
  into their own PR (#7, #8, #12). Every PR costs a review and merge round-trip.
- **Interactive round-trips on the host.** Each step meant paste output, wait, then
  run the next step. Steps run without `tee` lost their console logs.
- **Re-deriving the same host fixes.** The Hyperstack A4000 image ships driver R570,
  which cannot initialize CUDA 13. Every session needs the same compute-only upgrade
  to 580.178.04 and a reboot.

## Problems that recurred or surfaced late

| Problem | Seen in | Fix now in place |
|---|---|---|
| Image ships R570; CUDA 13 fails | M1, M2, M3 | `setup.sh` installs 580 headless packages, reboots once, and refuses to loop |
| `nvidia-smi \| grep -q` under `pipefail` misread the driver | M3 | query `--query-gpu=driver_version` directly |
| Smoke directory picked by name sort, not time | M2 | no separate smoke runs; series IDs are explicit |
| Console logs not captured | M2 | every step runs via `tee` or detached with a log file |
| Flaky synthetic test (sub-second window vs 1 s mpstat) | M3 | synthetic runs last ≥ 2 s (PR #12) |
| matplotlib font cache built under a monkeypatched `subprocess.run` | M3 host | pyplot imported before any patching |
| Preflight blocked by TIME_WAIT from `/metrics` scrapes | M3 | `SO_REUSEADDR` probe; a live listener still fails |
| Model-default sampling silently applied | M1 | `generation_config = "vllm"` and explicit sampling |
| Unpinned model revision; prefix cache across repetitions | M1 | pinned revision; caching off |

## Adjustments (adopted)

1. **Keep four non-negotiables, and relax the rest.**
   - Pre-register before collecting data.
   - Validate each series before analysis.
   - Archive and checksum before teardown.
   - Never splice repetitions.

   Everything else is "run, and debug on failure."
2. **One PR per unit of work,** containing the implementation, tests, and the docs it
   touches. A pre-registration is still its own commit, made before any data, because
   its timestamp is the evidence.
3. **Lean GPU sessions.**
   - Three host scripts: `setup` (driver, clone, install, tests), `run` (series, each
     validated as it finishes), `archive`. Plus a one-page plan.
   - No shakedown phase: the first measured series is the smoke test. A failed
     attempt restarts under a new ID.
   - Paste only last lines or errors.
4. **Rent the GPU only for collection.** Analysis, figures, and reports happen locally
   from the verified archive. Delete the VM in the console as soon as the archive's
   checksum is verified locally. An OS shutdown timer is unnecessary.
5. **Defer tooling checks to the phase that uses them.** For example, the contention
   generator is first exercised at the start of Phase C, not before Phase B.
6. **Report against the predictions table,** row by row (confirmed, not confirmed,
   mixed). Record execution problems in the report rather than re-running for a
   cleaner story.
7. **Docs move together.** A status change updates the README matrix and staged plan,
   the roadmap, and the protocol's outcome in the same PR.

## Next adjustments (proposed)

- **Commit the host scripts.** Move `setup.sh`, `run.sh`, and `archive.sh` into
  `scripts/host/`, parameterized by config list and commit. Each milestone's plan then
  becomes a list of configs and a pair-ID prefix, not new scripts.
- **A report skeleton.** A `reports/TEMPLATE.md` with the standing sections:
  - result;
  - provenance and protocol;
  - validation;
  - measurements;
  - predictions vs observed;
  - interpretation;
  - limitations;
  - execution notes;
  - next step;
  - artifact record.

  Analysis output could fill the measurement table directly.
- **A one-command local verification.** For example, `scripts/verify_archive.sh
  ARCHIVE`: checksum, extract, validate, and run analysis into `analysis/`.
- **Resume support in `run.sh`.** Skip series that already validated, so an
  interrupted pair continues without a hand-written `run-rest.sh`.
- **A lighter roadmap.** Keep milestone summaries to a few lines, and let protocols and
  reports carry the detail.

## Per-milestone checklist (target)

1. Roadmap entry and decisions: one conversation.
2. `experiments/milestoneN.md` with predictions → PR (merge = pre-registration).
3. Code, configs, and tests for anything new → one PR, if needed.
4. GPU session: `setup` → `run` → `archive` → `scp` → checksum → delete VM.
5. Locally: validate, analyze, and write the report from the template, with the
   README, roadmap, and protocol outcome in the same PR.
