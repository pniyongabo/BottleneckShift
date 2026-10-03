# Agent instructions

## Git

- Do not commit, push, or open pull requests without asking the maintainer first.
  Make the change, run the checks, report the result, and wait for approval.

## Documentation

- Document layout: `experiments/roadmap.md` is the plan and status index;
  `experiments/milestoneN.md` is each milestone's protocol, committed before its data
  are collected and not rewritten afterwards except to link the outcome;
  `reports/` holds results.

- When a milestone's status, scope, or numbering changes (for example, a new report in
  `reports/`), update `README.md` (experimental matrix and staged plan) and
  `experiments/roadmap.md` in the same change, so they never disagree. Link the report
  from both.
