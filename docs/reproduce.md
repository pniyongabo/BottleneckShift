# Reproducing the experiments

How to set up, plan, run, validate, and analyze BottleneckShift series. The landing
page is the [README](../README.md); protocols and predictions are in
[`experiments/`](../experiments/roadmap.md); metric definitions are in the
[glossary](glossary.md).

## Milestone 1: the concurrency baseline

The first milestone holds model, generation lengths, server configuration, request
count, and host fixed, then changes maximum client concurrency from 1 to 8. Each
condition is repeated three times and retains vLLM's raw request-level output plus an
exact run manifest. The model and tensor parallelism are parameters. The default,
`Qwen/Qwen2.5-0.5B-Instruct`, was chosen to fit on many single-GPU systems; it is a
starting point, not a claimed universal configuration.

### Pre-registered hypothesis

Increasing concurrency should increase offered load and aggregate throughput until
the server approaches saturation. Queueing and batching should increase
client-observed TTFT and end-to-end latency; TPOT may improve initially through
batching and then degrade. Output lengths should remain near the requested target,
and errors should remain zero. A different signature is informative and must not be
silently discarded. This comparison establishes an association with the controlled
workload change. By itself, it does not attribute delay to GPU utilization or any
other resource.

## Metrics and evidence

The vLLM benchmark client records request-level:

* time to first token (TTFT), inter-token latency (ITL), time per output token
  (TPOT), and end-to-end latency;
* input/output token counts and errors; and
* aggregate request/output-token throughput.

The [glossary](glossary.md) defines these and the other terms used in reports. These
are client-observed timings; server logs are retained separately. Milestone 3
adds explicitly labelled server-side metrics (see below), but utilization is
corroborating evidence, not causal attribution. Every repetition gets a manifest
containing the resolved config, commands, timestamps, Git revision, Python/platform
metadata, package inventory, and best-effort GPU inventory.

## Reproduce the experiments

Prerequisites are Linux, Python 3.11, a CUDA-capable GPU supported by the pinned
vLLM release, and enough VRAM for the selected model.

For a cost-controlled workflow on AWS, GCP, Azure, or public GPU clouds, read the
[GPU cloud development and execution plan](../experiments/gpu-cloud-development-plan.md)
before provisioning. It separates disposable spot-instance debugging from
non-interruptible measured runs and includes cost gates, artifact handling, and a
failure/retry policy.

Create a clean environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

The `test` extra includes matplotlib, used by the Milestone 3 figure test.

### Milestone 1 protocol

A dry run prints the full forward execution plan without needing vLLM or a GPU:

```bash
python scripts/run_experiment.py --config configs/milestone1-forward.toml \
  --series-id DEVICE-MODEL-forward-01 --dry-run
```

On the GPU host, verify that vLLM, `nvidia-smi`, the output location, and server port
are available. Preflight does not start the server or run inference:

```bash
python scripts/run_experiment.py --config configs/milestone1-forward.toml \
  --series-id DEVICE-MODEL-forward-01 --preflight
```

Then run the forward and reverse series with distinct, meaningful series IDs:

```bash
python scripts/run_experiment.py --config configs/milestone1-forward.toml \
  --series-id DEVICE-MODEL-forward-01
python scripts/run_experiment.py --config configs/milestone1-reverse.toml \
  --series-id DEVICE-MODEL-reverse-01
```

Replace `DEVICE-MODEL` with a non-identifying label for the selected setup. A series
ID may contain letters, numbers, dots, underscores, and hyphens and cannot be reused.
If omitted, a timestamped ID is generated.

The runner starts one vLLM server, waits for its health endpoint, performs a labelled
warm-up, executes all measured repetitions in the configured order, and stops the
server. A cooldown separates benchmark invocations. Forward and reverse plans have
identical server, workload, and execution settings; only their plan name and condition
order differ. Do not commit results until inspecting manifests for identifying host
metadata.

### Run lifecycle and artifacts

The series manifest progresses through `planned`, `warming_up`, `running`, and
`complete`; exceptions produce `failed`, while a keyboard interrupt produces
`interrupted`. Before each invocation, `active_run` identifies the condition and
repetition. Each raw result has a colocated run manifest containing the series and
plan IDs, measurement role, requested token targets, concurrency, command, timestamps,
and exit status. Warm-up data is retained under `warmup/` for troubleshooting but is
not a measured observation and is ignored by analysis.

Analyze all completed repetitions:

```bash
python analysis/compare.py results/runs --output results/figures/milestone1.svg
```

The analysis refuses to invent missing observations. It prints a series-labelled,
run-level CSV summary, plus request counts and realized input/output-token ranges.
When valid measured results exist, it also draws a figure of repetition-level
distributions, so no single run is presented as definitive. Before analysis, validate
complete extracted series against their manifests:

```bash
python scripts/validate_results.py \
  results/runs/milestone1-20261001T205137Z-forward \
  results/runs/milestone1-20261001T205137Z-reverse
```

This command requires the raw result directories; they are intentionally not in this
repository. For Milestone 1, it passes on both extracted series: all 12 measured runs and 1,200
measured requests, with zero failures and exact configured token lengths. See the
[Milestone 1 report](../reports/milestone1-20261001T205137Z.md) for its scope.

### Controlled follow-up protocol

This is the milestone 2 protocol; its measured pair is reported in
[`reports/milestone2-20261003T034456Z.md`](../reports/milestone2-20261003T034456Z.md).
The historical configs remain unchanged. The reviewed follow-up plans
`configs/milestone1-controlled-forward.toml` and
`configs/milestone1-controlled-reverse.toml` keep the milestone 1 model, workload,
server, and execution settings and add these controls:

* model and tokenizer pinned to Hugging Face commit
  `7ae557604adf67be50417f59c2c2f167def9a775`, the commit `main` resolved to on
  October 1, 2026 (last modified September 25, 2024);
* prefix caching explicitly disabled, so the repeated seed-2027 prompts cannot reuse
  KV cache across warm-ups and repetitions;
* `generation_config = "vllm"`, so the model's `generation_config.json` defaults
  (`repetition_penalty` 1.1, `top_k` 20, logged by the milestone 1 server) no
  longer silently apply;
* request-level greedy sampling: `temperature = 0.0`, `top_p = 1.0`;
* separate 10-prompt warm-ups at concurrency 1 and 8; and
* `percentile_metrics` including `e2el`, so each result also saves vLLM's own
  `median_e2el_ms`, reported by the analysis as `vllm_median_e2el_ms` to audit the
  reconstructed `e2el_ms`.

These values are passed to the vLLM CLIs and copied into both series and run
manifests. `vllm bench serve` has no tokenizer-revision option, so the client
tokenizes generated prompts with `main`; validation of exact 256-token input lengths
guards against a silent tokenizer change. Use new series IDs, for example:

```bash
python scripts/run_experiment.py --config configs/milestone1-controlled-forward.toml \
  --series-id DEVICE-MODEL-controlled-forward-01 --dry-run
```

### Milestone 3 instrumentation

Milestone 3 adds opt-in instrumentation. Configs without these settings, including
every milestone 1 and 2 config, produce byte-identical command plans. The Phase B
configs are `configs/milestone3-{prefill_heavy,decode_heavy}-{forward,reverse}.toml`.
See [`experiments/milestone3.md`](../experiments/milestone3.md) for the protocol and its
amendments.

```toml
[execution]
request_id_prefix = true   # --request-id-prefix <series>-<condition>-[rep-NN-]

[telemetry]
server_metrics = true      # /metrics before each run and after the server settles
gpu_sampler = true         # nvidia-smi every 200 ms -> gpu.csv
cpu_sampler = true         # mpstat -P ALL 1 -> cpu.txt (apt install sysstat)

[contention]               # Phase C only; requires gpu_sampler
mode = "active"            # off | sham | active
duty_cycle = 0.5           # active only
memory_cap_mib = 1024      # total generator VRAM, including its CUDA context
```

Instrumented series write manifest schema 4. Next to each `requests.json` the runner
writes `metrics-before.prom` and `metrics-after.prom`. At series level it writes
`metrics-initial.prom`, `gpu.csv`, `cpu.txt`, and, with contention,
`contention.jsonl`. `scripts/validate_results.py` additionally checks warm-ups, that
the server's request count equals each run's prompt count, telemetry coverage, and the
contention log.

Analyze validated series with:

```bash
python -m pip install -e '.[analysis]'   # matplotlib, for the figure
python analysis/phases.py results/runs/SERIES-ID [...] \
  --summary results/figures/milestone3-summary.csv --output results/figures/milestone3.svg
```

The step columns (`server_steps`, `tokens_per_step_mean`, `steps_over_*_share`,
`server_step_ms`) come from `vllm:iteration_tokens_total`. Its count is output
iterations, but its token values credit a request's whole prompt at its first token, so
they do not measure per-step chunk sizes (see the [glossary](glossary.md)).

`scripts/gpu_contention.py` can also be run by hand during a disposable shakedown.

## Build the technical report

The report's source is in `paper/` (LaTeX, one file per section under
`paper/sections/`). Its figures are vector PDFs written by the same analysis scripts
as the report figures, with `--format pdf` (`analysis/milestone3_figures.py`) or a
`.pdf` path for `--figure` (`analysis/milestone5_part2.py`) or `--output`
(`analysis/reanalysis_steps.py figure`).
Build it with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
paper/build.sh      # writes paper/bottleneckshift-report.pdf
```
