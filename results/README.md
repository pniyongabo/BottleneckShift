# Results policy

`results/runs/` and `results/figures/` are generated and ignored by Git by default.
Each run directory contains the copied config, `manifest.json`, server log, benchmark
stdout/stderr, and vLLM's raw `requests.json` for every repetition.

Commit only small, reviewed, machine-readable results needed to reproduce a published
figure, plus the figure and a provenance note. Never commit model weights, caches,
large traces, secrets, machine credentials, or identifying host metadata. Store large
artifacts in a versioned external archive and commit its checksum and durable locator.

No example or synthetic benchmark result is included. If one is added for tests or
documentation, its filename and surrounding text must explicitly say `synthetic` and
it must never be presented as a measured finding.
