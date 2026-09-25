# BottleneckShift

**A reproducible study of network and resource-contention effects on LLM inference**

> When an LLM inference service slows down, can controlled experiments distinguish
> workload effects from host, GPU, and network contention, and identify when the
> dominant bottleneck shifts?

BottleneckShift is a small, staged research project about experimental diagnosis,
not a leaderboard. It uses only public resources, generated prompts, and independent
experiments. It contains no proprietary code, data, infrastructure description,
workload, result, or conclusion.

## First milestone

The implemented milestone holds model, generation lengths, server configuration,
request count, and host fixed, then changes **maximum client concurrency from 1 to
8**. Each condition is repeated three times and retains vLLM's raw request-level
output plus an exact run manifest. The model and tensor parallelism are parameters;
the default `Qwen/Qwen2.5-0.5B-Instruct` is intentionally small enough for many
single-GPU systems, but is a starting choice rather than a claimed universal
configuration.

### Pre-registered hypothesis

Increasing concurrency should increase offered load and aggregate throughput until
the server approaches saturation. Queueing and batching should increase
client-observed TTFT and end-to-end latency; TPOT may improve initially through
batching and then degrade. Output lengths should remain near the requested target,
and errors should remain zero. A different signature is informative and must not be
silently discarded. This comparison establishes an association with the controlled
workload change—it does **not** by itself attribute delay to GPU utilization or any
other resource.

## Metrics and evidence

The vLLM benchmark client records request-level:

* time to first token (TTFT), inter-token latency (ITL), time per output token
  (TPOT), and end-to-end latency;
* input/output token counts and errors; and
* aggregate request/output-token throughput.

These are **client-observed timings**. Server logs are retained separately. Future
work may collect explicitly labelled server-side metrics, but utilization is
corroborating evidence—not causal attribution. Every repetition gets a manifest
containing the resolved config, commands, timestamps, Git revision, Python/platform
metadata, package inventory, and best-effort GPU inventory.

## Experimental matrix

| Stage | Factor | Conditions | Status |
|---|---|---|---|
| 1 | Workload concurrency | 1 vs 8, fixed generated token targets | **Implemented; not executed here** |
| 2 | Workload shape | short/long input and output factorial | Planned |
| 3 | Network | controlled latency/bandwidth/loss shaping | Planned |
| 4 | Host contention | controlled CPU and memory pressure | Planned |
| 5 | GPU contention | isolated, controlled competing GPU work | Planned |
| 6 | Bottleneck shift | selected crossed factors from stages 2–5 | Planned |

The later rows are proposals, not completed experiments or findings. Each will need
a written hypothesis, randomized run order, controls, safety checks, and repeated
runs before implementation.

## Reproduce milestone 1

Prerequisites are Linux, Python 3.11, a CUDA-capable GPU supported by the pinned
vLLM release, and enough VRAM for the selected model. Create a clean environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

Review the full forward execution plan without requiring vLLM or a GPU:

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
run-level CSV summary and, when valid measured results exist, makes a figure showing
repetition-level distributions rather than presenting a single run as definitive.

## Repository layout

* `configs/` — reviewed experiment inputs.
* `scripts/` — thin orchestration around the official vLLM CLI.
* `src/bottleneckshift/` — config validation and manifest capture.
* `analysis/` — transparent post-processing of raw JSON.
* `experiments/` — hypotheses and protocols written before running.
* `results/` — retention policy and ignored local run artifacts.
* `report/` — future study narrative (currently only a scope note).

## Limitations and validity threats

No GPU benchmark results ship with this milestone. Generated prompts improve
repeatability but are not representative of all applications. Tokenizer behavior can
make realized lengths differ from targets. A single model/GPU/framework combination
limits external validity. Client and server on one host remove network realism and
can introduce client interference. Fixed condition order permits drift and warm-up
effects; repetitions help but do not eliminate them. Framework/version changes can
alter scheduling and benchmark schemas, hence the dependency pin and raw retention.

## Staged plan (40–50 focused hours total)

1. **Baseline + concurrency perturbation (this repository state):** reproducible
   runner, manifests, raw retention, and comparison plot.
2. **Workload-shape study:** pre-register a small factorial design and validate
   measurement stability.
3. **One contention factor at a time:** add network, host, then GPU perturbations
   with negative controls and server-side corroboration.
4. **Bottleneck shifts:** run only motivated crossed conditions and reason from
   metric signatures, not utilization alone.
5. **Synthesis:** uncertainty-aware plots, limitations, and a concise report.

This scope deliberately excludes Kubernetes, distributed deployment, dashboards,
databases, a custom load generator, and an adaptive router.
