# Glossary

Terms used across the BottleneckShift protocols and reports. Each report also has a
short "Terms used in this report" table near the top.

## Latency and throughput metrics

These are measured client-side by `vllm bench serve` unless marked "server". vLLM's
summary percentiles default to `--percentile-metrics ttft,tpot,itl` and
`--metric-percentiles 99`. The custom metrics (marked *custom*) are computed by
`analysis/compare.py` and `analysis/phases.py` from raw per-request arrays.

| Term | Meaning |
|---|---|
| **TTFT** (time to first token) | Time from sending a request until the first output token arrives at the client. It includes queueing, prompt processing (prefill), and front-end and network time. |
| **TPOT** (time per output token) | For one request, (E2E − TTFT) / (output tokens − 1): the average time per generated token after the first. |
| **ITL** (inter-token latency) | The gap between two consecutive streamed tokens of one request. One request with 32 output tokens yields up to 31 ITLs. A streamed chunk can carry several tokens, so an ITL is not exactly one generation step. |
| **E2E** (end-to-end latency) | Time from sending a request until its last token arrives. vLLM saves `median_e2el_ms` only when `e2el` is in `--percentile-metrics`, which is not the default; the controlled configs add it. *Custom* `e2el_ms` is reconstructed as TTFT + Σ ITL. |
| **Output tok/s** (output token throughput) | Output tokens generated per second across all concurrent requests in a run. |
| **p50 / p90 / p99** | The 50th, 90th, and 99th percentiles of a distribution. p50 is the median; p99 is the value 99% of observations fall below, i.e. the tail. |
| **Median (in tables)** | Unless stated otherwise, the median across a run's requests, then the median of those values across the three repetitions. Ranges in parentheses are the minimum and maximum across repetitions. |
| **Pre-first-token share** (*custom*) | The fraction of E2E latency spent before the first token. Client-side: Σ TTFT / Σ E2E over a run's requests. Server-side: (queue + prefill) / (queue + inference), from mean server phase times. |
| **Dominant component** (*custom*) | "Pre-first-token" if mean server queue + prefill time exceeds mean decode time; otherwise "decode". |
| **Client − server TTFT gap** (*custom*) | Client mean TTFT minus the server's own TTFT measurement. It covers the front end (HTTP, tokenization, streaming) and, here, client overhead on the shared host. |

## Server-side phases

From vLLM's Prometheus `/metrics` histograms. Per-run values are differences between
scrapes taken before and after the run.

| Term | Meaning |
|---|---|
| **Queue time** | Time a request waits in the engine before it is first scheduled. |
| **Prefill** | Processing the prompt (input tokens) to produce the first output token. It is compute-heavy. |
| **Decode** | Generating output tokens one step at a time after the first. At low load it is limited by memory bandwidth. |
| **Inference time** | Prefill plus decode time, i.e. from first scheduled to finished. |
| **Chunked prefill** | vLLM's default scheduling: a long prompt's prefill is split into chunks that share each batch step with other requests' decode work, within a token budget (`max_num_batched_tokens`). |
| **Generation stall** | A decode step slowed because prefill work was scheduled in the same or an intervening step. |
| **Engine step** | One scheduler iteration of the vLLM engine. It processes up to `max_num_batched_tokens` tokens: one new token per decoding request plus prompt chunks. |
| **Tokens per step** (*server*) | From the `vllm:iteration_tokens_total` histogram: prompt tokens computed plus tokens generated in each engine step. Phase D reports the mean and the share of steps over 512, 1024, 2048, and 8192 tokens. |
| **Server step time** (*custom*) | Benchmark duration divided by engine steps in the run: the mean time per step, including idle gaps. |

## Load and experimental design

| Term | Meaning |
|---|---|
| **C1, C8, C16, C32** | Maximum client concurrency (`--max-concurrency`): at most 1, 8, 16, or 32 requests in flight at once. |
| **Closed-loop load** | A new request is sent only when one finishes, so requests in flight never exceed the concurrency limit. All runs here use closed-loop load with `--request-rate inf`. |
| **Open-loop load** | Requests arrive on a schedule independent of completions, so a queue can grow without bound. Not used here. |
| **Coordinated omission** | A known closed-loop bias: a slow server also slows request sending, so measured tail latency understates what users would see. |
| **Regime** | A workload shape. `prefill_heavy` is 2048 input / 32 output tokens; `decode_heavy` is 64 input / 512 output tokens. |
| **Run** | One `vllm bench serve` invocation: one condition and one repetition, e.g. 100 prompts. |
| **Repetition** | One of three runs of the same condition within a series. |
| **Series** | One server start running a config's warm-ups and all of its measured runs, in a fixed condition order, with one `manifest.json`. |
| **Pair / session ID** | The ID shared by the series of one GPU session, e.g. `milestone3c-20261003T201214Z`. Series IDs append a name, e.g. `…-off-before`. |
| **Forward / reverse** | The same conditions run in opposite orders (for example C1 then C8, versus C8 then C1), to expose order and history effects. |
| **Warm-up** | Unmeasured runs before the measured runs, at each concurrency level, so that model loading, compilation, and allocator effects are excluded from measurements. |
| **Cooldown** | A 15 s pause between runs. |
| **Anchor** | A condition repeated from an earlier session (e.g. `anchor_c8`), used to check that results reproduce across sessions. |
| **Token-budget intervention** | Phase D: setting `max_num_batched_tokens` to 512, 2048, or 8192 to change how many prompt tokens can share a step with decodes. Series `budget-0512`, `budget-8192`, and `budget-2048-before`/`-after`, each at C8 and C32 (`budget_c8`, `budget_c32`). |
| **Sweep** | Phase D: `prefill_heavy` at C8, C12, C16, C24, and C32 (`sweep_c8` … `sweep_c32`) at the default budget. |
| **Pre-registration** | Committing the protocol and its predictions before any data are collected. The merge time is the evidence. |
| **Addendum / amendment** | A dated section added to a protocol before the data it governs. An addendum adds a phase; an amendment refines measurement details. |

## Controls

| Term | Meaning |
|---|---|
| **Prefix caching** | vLLM reusing stored work for repeated prompt prefixes. It is turned off from Milestone 2 onward, so repetitions cannot share cached work. |
| **`generation_config = "vllm"`** | Ignore the model's own default sampling settings, such as repetition penalty and top-k, so that only the explicit settings apply. |
| **Greedy decoding** | `temperature = 0`: always pick the most likely next token, so outputs are deterministic. |
| **Pinned revision** | A fixed Hugging Face commit for the model and tokenizer (`7ae5576…`), instead of the moving `main`. |

## GPU contention (Milestone 3 Phase C)

| Term | Meaning |
|---|---|
| **Contention generator** | `scripts/gpu_contention.py`: a separate process on the same GPU that runs fp16 2048×2048 matrix multiplies to compete with vLLM for GPU time. |
| **Duty cycle** | The fraction of each 100 ms period in which the generator runs matrix multiplies; it sleeps for the rest. Duty 0.25 means busy for 25 ms of every 100 ms, and duty 0.5 means 50 ms of every 100 ms. |
| **Achieved duty** | The measured fraction of time the generator was in its busy phase. It should match the configured duty within ±0.05. |
| **Kernel duty** | The fraction of time the generator's matrix-multiply kernels were actually executing on the GPU. It is lower than achieved duty because each step waits for completion. |
| **Sham** | The generator started with its CUDA context and full memory footprint, but running no kernels. It separates the effect of the extra process and memory from the effect of compute contention. |
| **`off-before` / `off-after`** | The same design with no generator, run before and after the contention series. `off-after` is the recovery check. |
| **`active-025` / `active-050`** | Contention series with the generator active at duty 0.25 and 0.50 (configs `configs/milestone3-contention-active-025.toml` and `-active-050.toml`). The protocol calls them `active-0.25` and `active-0.50`. |
| **Time-slicing** | The GPU's default way of sharing between processes: it alternates between them rather than running their kernels at the same time. |
| **`memory_cap_mib`** | The generator's total VRAM budget, including its CUDA context, checked with `nvidia-smi --query-compute-apps`. |

## Settings and their defaults

"Default" is what applies when a setting is omitted. "Used here" is what the
experiment configs set. Runner defaults come from `scripts/run_experiment.py`,
`src/bottleneckshift/config.py`, `telemetry.py`, `contention.py`, and
`validation.py`. vLLM defaults come from the v0.29.0 source: `vllm/config/*.py`,
`vllm/engine/arg_utils.py`, and `vllm/benchmarks/`.

### Experiment config (TOML) — custom fields

| Field | Default | Used here | Meaning |
|---|---|---|---|
| `experiment.repetitions` | required | 3 | Measured runs per condition. |
| `execution.warmup_prompts` | required | 10 (32 in the crossover probe) | Prompts per warm-up run; 0 disables warm-ups. |
| `execution.warmup_max_concurrency` | required | 1 | Warm-up concurrency when `warmup_concurrencies` is absent (historical layout). |
| `execution.warmup_concurrencies` | `[warmup_max_concurrency]` | `[1, 8]`; `[8, 16, 32]` in the probe | One warm-up run per listed concurrency. |
| `execution.cooldown_seconds` | required | 15 | Pause between runs. |
| `execution.percentile_metrics` | not passed (vLLM default `ttft,tpot,itl`) | `["ttft","tpot","itl","e2el"]` | Passed as `--percentile-metrics`. |
| `execution.request_id_prefix` | `false` (vLLM's own prefix `bench-<8 hex>-`) | `true` from Milestone 3 | Adds `--request-id-prefix <series>-<condition>-[rep-NN-]`. |
| `server.model_revision` / `tokenizer_revision` | not passed (vLLM `None`, i.e. the Hugging Face default branch `main`); manifest records `unresolved` | `7ae5576…` from Milestone 2 | Passed as `--revision` / `--tokenizer-revision`. |
| `server.enable_prefix_caching` | not passed (vLLM default `true`) | `false` from Milestone 2 | Passed as `--[no-]enable-prefix-caching`. |
| `server.generation_config` | not passed (vLLM default `auto`: use the model's `generation_config.json`) | `vllm` from Milestone 2 | Passed as `--generation-config`. |
| `server.gpu_memory_utilization` | required | 0.90 | Passed through; vLLM's own default is 0.92. |
| `server.tensor_parallel_size` | required | 1 | Passed through. |
| `server.startup_timeout_seconds` | 600 | 600 | Wait for `/health` before failing. |
| `server.max_num_batched_tokens` | not passed (vLLM default 2048 on this GPU) | 512, 2048, 8192 in Phase D's budget series | Passed as `--max-num-batched-tokens`; the token budget per engine step. |
| `sampling.temperature` / `top_p` | not sent (the server's default applies, from the model's generation config unless `generation_config = "vllm"`) | 0.0 / 1.0 from Milestone 2 | Passed as `--temperature` / `--top-p`. |
| `workload.num_prompts` | required (vLLM default 1000) | 100; 50 for `decode_heavy` | Prompts per measured run. |
| `workload.input_tokens` / `output_tokens` | required (vLLM defaults 1024 / 128) | 256/128, 2048/32, 64/512 | `--random-input-len` / `--random-output-len`. |
| `workload.seed` | required (vLLM default 0) | 2027 | Prompt-generation seed. |
| `condition.max_concurrency` | required (vLLM default: unlimited) | 1, 8, 16, 32 | `--max-concurrency`. |
| `telemetry.server_metrics` | `false` | `true` from Milestone 3 | `/metrics` scrapes around every run. |
| `telemetry.gpu_sampler` | `false` | `true` from Milestone 3 | `nvidia-smi` sampler, every 200 ms. |
| `telemetry.cpu_sampler` | `false` | `true` from Milestone 3 | `mpstat -P ALL 1`, every 1 s; needs `sysstat`. |
| `contention.mode` | `off` (table absent) | `sham`, `active` | Generator mode. |
| `contention.duty_cycle` | none; required for `active`, forbidden for `sham` | 0.25, 0.5 | Busy fraction of each period. |
| `contention.period_ms` | 100 | 100 | Period length (10–1000 ms). |
| `contention.memory_cap_mib` | none; required for `sham`/`active` | 1024 | Total generator VRAM (513–1024 MiB), including the context. |
| `contention.matrix_size` | 2048 (reduced to fit the cap) | 2048 | fp16 matrix dimension. |
| `contention.ready_timeout_seconds` | 120 | 120 | Wait for the generator's `ready` event. |

### vLLM settings used at their defaults (v0.29.0, RTX A4000)

| Setting | Default | Meaning |
|---|---|---|
| `enable_chunked_prefill` | `true` | Prefill split into chunks that share steps with decodes. |
| `max_num_batched_tokens` | 2048 on this GPU (the API-server default for GPUs under 70 GiB; 8192 on H100/H200) | Token budget per scheduler step. One 2048-token prompt fills a whole step. Left at the default except in Phase D's budget series. |
| `max_num_seqs` | 256 | Maximum requests in a running batch. |
| `vllm bench serve --request-rate` | `inf` (also set explicitly) | Send all requests at time 0, limited only by `--max-concurrency`. |
| `--ready-check-timeout-sec` / `--num-warmups` | 0 / 0 | No extra bench requests, so server request counts equal prompts. |
| `--metric-percentiles` | `99` | Percentile reported in vLLM's summary. |

### Runner, validator, and generator constants

| Constant | Value | Meaning |
|---|---|---|
| Settle timeout | 10 s | After a run, wait for zero running/waiting requests and a stable count before the after-scrape. |
| Scrape attempts | 3 | Retries for each `/metrics` fetch. |
| Sampler first-data timeout | 15 s | A sampler must produce a parseable row in this time. |
| Telemetry max gap | 1.0 s (GPU), 3.0 s (CPU) | Validation fails above these. |
| Duty tolerance | ±0.05 | Allowed difference between achieved and configured duty. |
| Context reserve | 512 MiB | Part of `memory_cap_mib` set aside for the generator's CUDA context. |
| Generator start margin | 64 MiB | Free VRAM required beyond the tensor budget at startup. |

## Terms from the cited literature

| Term | Meaning |
|---|---|
| **TBT** (time between tokens) | Another name for ITL, used in the Sarathi-Serve and DistServe papers. |
| **SLO** (service-level objective) | A latency target, e.g. "p99 TTFT under 500 ms". |
| **Goodput** | Throughput counting only requests that meet their SLOs. |
| **MPS** (Multi-Process Service) | NVIDIA's mode for running kernels from different processes concurrently on one GPU, instead of time-slicing them. Not used here. |
| **MIG** (Multi-Instance GPU) | Hardware partitioning of a GPU into isolated instances. Not available on the RTX A4000. |

## Telemetry

| Term | Meaning |
|---|---|
| **GPU utilization (`nvidia-smi`)** | The percent of time in the sample period during which at least one kernel was running. It does not measure how busy the GPU's compute units are. |
| **SM activity (DCGM)** | A finer NVIDIA metric: the fraction of time at least one warp was active on each streaming multiprocessor. Not collected here. |
| **`/metrics` scrape** | A snapshot of vLLM's Prometheus metrics, taken before and after each run. |

## Hardware, software, and artifacts

| Term | Meaning |
|---|---|
| **VRAM** | GPU memory: 15,352 MiB on the RTX A4000 used here. |
| **BF16** | The 16-bit floating-point format the model runs in. |
| **KV cache** | Per-request attention state that vLLM keeps in VRAM while generating. |
| **vLLM V1** | vLLM's current engine architecture. Its scheduler mixes prefill and decode within one token budget, and the API server runs in a separate process from the engine core. |
| **NVML** | NVIDIA's management library; `nvidia-smi` and the utilization counter come from it. |
| **R570 / 580** | NVIDIA driver branches. The cloud image ships R570, which cannot run the CUDA 13 PyTorch build, so each session upgrades to 580.178.04. |
| **A100 / H100 / H200** | NVIDIA data-center GPUs used by most cited work. They have much more compute and memory bandwidth than the RTX A4000. |
| **SHA-256** | The archive checksum, verified after transfer and before the VM is deleted. |
| **GPU UUID** | The GPU's unique ID, recorded in manifests; removed before publishing. |

## Citation tags in reports

- **[P#]**: peer-reviewed paper.
- **[R#]**: arXiv preprint, not peer-reviewed.
- **[B#]**: engineering blog or documentation, not peer-reviewed.
- **Numbered references [1]–[15]** in the Phase B report mix these kinds, and its
  reference list marks preprints.
- **Conference abbreviations:** OSDI, SOSP, NSDI, ISCA, MLSys, and NeurIPS are
  peer-reviewed venues.
