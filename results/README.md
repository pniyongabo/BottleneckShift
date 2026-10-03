# Results policy

`results/runs/` and `results/figures/` are generated and ignored by Git by default.
Each series directory contains the copied config, `manifest.json`, server log, a
separate `warmup/` directory, and measured condition/repetition directories. Each
benchmark directory contains stdout/stderr, a run manifest, and vLLM's raw
`requests.json`. Warm-up observations are retained for diagnosis but are never study
measurements and are excluded by `analysis/compare.py`.

Use `scripts/validate_results.py SERIES [SERIES ...]` after extracting retained raw
artifacts. It requires a complete series manifest and checks measured-run completeness,
return codes, aligned request arrays, empty error strings, and exact configured token
lengths. It does not authenticate an archive: verify the published SHA-256 separately
before extraction.

Commit only small, reviewed, machine-readable results needed to reproduce a published
figure, plus the figure and a provenance note. Never commit model weights, caches,
large traces, secrets, machine credentials, or identifying host metadata. Store large
artifacts in a versioned external archive and commit its checksum and durable locator.

Milestone 3 series (manifest schema 4) may also contain `metrics-initial.prom`, per-run
`metrics-before.prom`/`metrics-after.prom`, `gpu.csv`, `cpu.txt`, and
`contention.jsonl`. For these series the validator also checks:

* warm-ups;
* that each run's server request count delta equals its prompt count;
* that telemetry covers the series without gaps;
* the contention log.

Before publishing anything, check that these files carry no identifying data:
`cpu.txt` starts with the hostname, and manifests record the GPU UUID and host paths.

No example or synthetic benchmark result is included. If one is added for tests or
documentation, its filename and surrounding text must explicitly say `synthetic` and
it must never be presented as a measured finding.
