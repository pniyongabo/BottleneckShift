# Agent instructions

## Git

- Do not commit, push, or open pull requests without asking the maintainer first.
  Make the change, run the checks, report the result, and wait for approval.
- Keep a unit of work in one pull request: its implementation, tests, and the docs it
  affects (README, roadmap, protocol) go together rather than in separate PRs.

## Execution style

- Keep milestone prep lean: run, and debug when something fails, instead of
  pre-validating every detail.
- GPU sessions use a few combined host scripts (setup, run, archive) and a one-page
  plan. There are no separate shakedown phases; the first measured series is the smoke
  test, and a failed attempt restarts under a new series or pair ID.
- Rent the GPU only for collection. Analysis and reports happen locally after
  transfer.
- Non-negotiables: pre-register the protocol before collecting data, validate each
  series before analysis, archive and checksum before teardown, and never splice
  repetitions.

## Documentation

- Document layout: `experiments/roadmap.md` is the plan and status index;
  `experiments/milestoneN.md` is each milestone's protocol, committed before its data
  are collected and not rewritten afterwards except to link the outcome;
  `reports/` holds results.

- When a milestone's status, scope, or numbering changes (for example, a new report in
  `reports/`), update `README.md` (experimental matrix and staged plan) and
  `experiments/roadmap.md` in the same change, so they never disagree. Link the report
  from both.
