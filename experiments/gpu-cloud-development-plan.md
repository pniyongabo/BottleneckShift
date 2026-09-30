# Cheap public-GPU development and execution plan

## Purpose and boundary

This plan describes how to develop, debug, and run BottleneckShift on rented GPU
instances while keeping the work inexpensive, repeatable, and easy to repair. It is
an operations plan for the already implemented Milestone 1 concurrency experiment;
it does not add or claim network, host, or GPU-contention experiments.

The central rule is to separate **cheap, interruptible development** from
**controlled measurement**:

* use local CPU work and short-lived/interruptible GPUs for installation checks,
  model-download checks, and disposable shakedowns;
* use one explicitly recorded, non-interruptible GPU shape for both halves of a
  measured forward/reverse pair; and
* treat any code, dependency, image, hardware, or configuration change as a new
  series, never as an in-place repair of a partially collected series.

Prices, quotas, accelerator availability, and spot rules change frequently. Do not
copy a price from this document into a budget. Record a dated quote from the
provider's public pricing page or calculator immediately before provisioning.

## Recommended default strategy

Start with the least operationally complex option available in the chosen region:

1. Develop and test everything that does not require CUDA on the local machine.
2. Rent a single NVIDIA GPU with at least 16 GB VRAM for a 30--60 minute disposable
   shakedown. An L4, A10/A10G, T4, RTX 3090, or RTX 4090 class device is sufficient
   to evaluate the repository's small default model; this is a compatibility screen,
   not a claim that different GPU models are scientifically interchangeable.
3. Prefer an interruptible/spot instance for shakedowns only. Prefer a provider with
   a simple per-second or per-minute lifecycle and persistent storage or snapshots.
4. After the workflow passes, provision one on-demand instance and run the forward
   and reverse series back-to-back on that same instance. Keep the exact instance
   type, GPU count, image, disk, region, Git commit, and configuration fixed.
5. Download and inspect artifacts, destroy the GPU instance, and only then decide
   whether a second independent pair is worth the cost.

For the first iteration, a managed GPU VM or pod from Runpod or Lambda Cloud is
usually the shortest path if an appropriate fixed GPU is available. GCP G2/L4 or
AWS G5/A10G are strong fallbacks when account access and quota already exist. Azure
is equally valid when quota is already approved. Do not choose a provider solely on
the lowest advertised hourly number: setup time, storage, egress, quota delay, and
marketplace-host variability can cost more than the GPU.

## Provider decision matrix

Use the matrix to shortlist providers, then fill in the dated selection record below.
The terms “spot” and “interruptible” are provider-specific; consult the linked
current documentation before each rental.

| Provider | Sensible first shapes to check | Best use here | Important cautions |
|---|---|---|---|
| AWS EC2 | G4dn/T4, G5/A10G, G6/L4 | Existing AWS account, durable VM/image workflow, on-demand measured pair | GPU quota may be zero; Spot can be interrupted; EBS, snapshots, public IPv4, and egress can add cost |
| Google Cloud | N1+T4, G2/L4 | Simple VM workflow and transparent Spot semantics | GPU and VM quotas are separate; Spot can be preempted; regional stock varies |
| Microsoft Azure | NCasT4_v3/T4 or another single-GPU NC size | Existing Azure credits/account and a fixed VM shape | Spot eviction policy and quota/region availability must be checked; OS disk and public IP can add cost |
| Runpod | Secure Cloud fixed GPU; Community Cloud only for shakedown | Fast disposable pods and broad low-cost GPU choice | Community hosts can differ; record host/GPU data; inspect storage and stopped-pod billing; do not use a changed host across a measured pair |
| Lambda Cloud | Available single-GPU instance | Low-setup fixed instances when stocked | Limited region/shape inventory; verify persistent storage and termination behavior |
| Vast.ai | Verified single-GPU offers | Very cheap exploratory compatibility checks | Marketplace offers vary in CPU, disk, network, reliability, and GPU configuration; unsuitable for a controlled pair unless the same reserved offer remains allocated |
| CoreWeave | Single fixed GPU VM where directly available | Repeat studies needing stable infrastructure | More operational setup than this milestone needs; avoid Kubernetes for this project |

The project does not need multi-GPU hardware. Keep `tensor_parallel_size = 1` unless
a separately pre-registered future experiment justifies a change. Do not compare
measurements across providers or GPU models as though provider were the only changed
factor.

### Current official references

* [AWS EC2 accelerated-computing instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
  and [Spot interruption behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html)
* [Google Cloud GPU platforms](https://cloud.google.com/compute/docs/gpus)
  and [Spot VMs](https://cloud.google.com/compute/docs/instances/spot)
* [Azure GPU VM sizes](https://learn.microsoft.com/azure/virtual-machines/sizes-gpu)
  and [Azure Spot VMs](https://learn.microsoft.com/azure/virtual-machines/spot-vms)
* [Runpod GPU pricing](https://www.runpod.io/pricing) and
  [Pods documentation](https://docs.runpod.io/pods)
* [Lambda GPU Cloud](https://lambda.ai/service/gpu-cloud)
* [Vast.ai pricing documentation](https://docs.vast.ai/documentation/instances/pricing)
* [CoreWeave compute documentation](https://docs.coreweave.com/docs/products/compute/overview)

These links are inputs to a provisioning decision, not endorsements. Save the date,
quoted currency, billing granularity, and ancillary charges; do not scrape a live
price into an experiment result.

## Selection record and cost gate

Before launching, copy this block into a private run note or reviewed provenance
record. Do not record account IDs, instance IDs, IP addresses, or credentials in Git.

```text
quote_checked_at_utc:
provider:
service_tier: on-demand | spot/interruptible | marketplace
region_or_nonidentifying_location_label:
instance_shape:
gpu_model:
gpu_count:
gpu_memory_gib:
vcpu_and_ram:
base_image_and_version:
storage_type_and_size_gib:
gpu_price_per_billing_unit:
storage_ip_egress_estimate:
hard_budget_cap:
automatic_shutdown_method:
quota_and_capacity_confirmed: yes | no
measured_or_disposable: measured | disposable
```

Use two cost gates:

* **Shakedown gate:** cap the first rental at one hour. Stop immediately after one
  successful tiny request/benchmark output, or after collecting enough logs to fix
  the first blocker.
* **Measurement gate:** estimate model download, server start, warm-up, six measured
  invocations per plan, cooldowns, artifact verification, and a 25% time margin.
  Approve that whole forward/reverse window before starting. Do not move one half to
  a cheaper or different machine.

Set a provider budget alert as a backstop, but also schedule an OS-level shutdown
(`shutdown -h`) for the approved deadline and set an external timer. Provider budget
alerts are not hard spending caps. Delete the instance and unneeded disks/snapshots
after artifacts are safe.

## Reproducible machine bootstrap

### Base image policy

Prefer an official Ubuntu LTS GPU image with the NVIDIA driver already installed.
Record the exact image identifier and driver reported by `nvidia-smi`. Do not casually
upgrade the driver, CUDA components, Python, or vLLM between forward and reverse
series. The project currently requires Python 3.11+ and pins vLLM in
`requirements.txt`.

Keep two layers distinct:

* **Machine layer:** provider, region label, shape, base-image identifier, disk,
  NVIDIA driver, and GPU inventory.
* **Project layer:** clean Git commit, Python virtual environment, pinned Python
  dependencies, reviewed TOML config, and commands captured by the manifests.

A container can reduce setup variance later, but building and publishing a bespoke
image is unnecessary for the first milestone. First prove the documented virtualenv
workflow. If a provider image cannot supply Python 3.11 without invasive changes,
choose a newer official image or another provider rather than silently changing the
project requirements.

### Bootstrap checklist

Run commands interactively the first time so failures are visible:

```bash
set -euxo pipefail
nvidia-smi
python3.11 --version
git clone https://github.com/pniyongabo/BottleneckShift.git
cd BottleneckShift
git fetch origin main
git switch --detach <reviewed-commit-sha>
test -z "$(git status --porcelain)"

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'

python -m compileall -q src scripts analysis tests
python -m pytest -q
python scripts/run_experiment.py \
  --config configs/milestone1-forward.toml \
  --series-id disposable-dry-run \
  --dry-run
```

Using a detached commit for measured runs prevents the branch from moving under an
experiment. The repository must be clean. Capture installation failures in a local
text file outside `results/runs/`; do not call them benchmark findings.

Model downloads can dominate the first rental. Reuse a persistent cache only between
disposable shakedowns and the final pair when its contents are known and unchanged.
Never commit model files. Record whether the cache was cold or warm, and keep the
cache state consistent within the forward/reverse pair.

## Iterative development loop

### Phase A: CPU-only development

Use the local development machine for documentation, config validation, unit tests,
dry runs, and synthetic test fixtures that are explicitly labelled synthetic. Push a
small reviewed branch before renting a GPU. This minimizes paid debugging time.

### Phase B: disposable GPU shakedown

1. Provision an interruptible instance and bootstrap the reviewed commit.
2. Run preflight with a unique disposable ID:

   ```bash
   python scripts/run_experiment.py \
     --config configs/milestone1-forward.toml \
     --series-id disposable-<provider>-<gpu>-01 \
     --preflight
   ```

3. Confirm `nvidia-smi`, model loading, `/health`, the vLLM CLI flags, one benchmark
   result schema, manifest creation, log capture, and process shutdown. A full six-run
   plan is unnecessary to discover compatibility errors.
4. Mark all output disposable. Download only logs needed for diagnosis and terminate
   the instance.

### Phase C: fix, test, and push

Prefer fixing code on the normal development machine:

```bash
git switch work
git fetch origin main
git rebase origin/main
# make the smallest repair
python -m compileall -q src scripts analysis tests
python -m pytest -q
git diff --check
git add <reviewed-files>
git commit -m "Describe the compatibility fix"
git push -u origin work
```

If diagnosis truly requires editing on the GPU host, create a named branch first and
never edit a detached measured checkout:

```bash
git switch -c fix/<short-description> <reviewed-commit-sha>
# edit, run focused checks, inspect git diff
git commit -am "Describe the GPU compatibility fix"
git push -u origin HEAD
```

Use SSH agent forwarding or a short-lived GitHub authentication flow if a push from
the GPU is unavoidable. Do not place tokens in clone URLs, shell history, files,
images, startup scripts, manifests, or logs. The safer default is to copy the minimal
diagnostic log home, terminate the rental, reproduce/fix locally, and push locally.

Every repair restarts the loop on a fresh disposable series ID. Never overwrite a
series directory. Once the fix is merged, fetch it on a clean GPU checkout and record
the new commit SHA. Do not cherry-pick unreviewed working-tree changes immediately
before a measured run.

### Phase D: release candidate shakedown

On the exact GPU shape intended for measurement:

1. bootstrap the exact candidate commit from scratch or from a documented clean
   snapshot;
2. pass compile, unit, dry-run, and preflight checks;
3. run the smallest disposable inference check needed to prove compatibility;
4. inspect its raw JSON schema with `analysis/compare.py`;
5. stop and fix any discrepancy rather than adapting the analyzer after seeing the
   measured series.

Freeze the commit, configs, dependency pin, base image, and hardware after this gate.

## Measured Milestone 1 run

Use an on-demand/non-preemptible instance unless spot capacity is guaranteed for the
whole window (ordinary Spot products do not provide that guarantee). Disable unrelated
jobs and use one instance for the entire pair.

```bash
cd BottleneckShift
source .venv/bin/activate
test -z "$(git status --porcelain)"
git rev-parse HEAD
nvidia-smi

python scripts/run_experiment.py \
  --config configs/milestone1-forward.toml \
  --series-id <provider>-<gpu>-forward-01 \
  --preflight
python scripts/run_experiment.py \
  --config configs/milestone1-reverse.toml \
  --series-id <provider>-<gpu>-reverse-01 \
  --preflight

python scripts/run_experiment.py \
  --config configs/milestone1-forward.toml \
  --series-id <provider>-<gpu>-forward-01
python scripts/run_experiment.py \
  --config configs/milestone1-reverse.toml \
  --series-id <provider>-<gpu>-reverse-01

python analysis/compare.py results/runs \
  --output results/figures/milestone1.svg
```

Run inside `tmux` or an equivalent session so an SSH disconnect does not kill the
process. This protects against client disconnects, not provider eviction. Do not run
forward and reverse concurrently.

If any measured invocation fails, preserve the incomplete series and logs, mark the
attempt invalid/incomplete in the study notes, fix the cause, and begin a completely
new forward/reverse pair with new series IDs. Do not resume by filling only missing
repetitions after changing code or environment.

## Artifact handling and teardown

Before termination:

1. Confirm both series manifests say `complete`, `active_run` is null, and every
   measured directory contains `requests.json`, benchmark stdout/stderr, and a run
   manifest.
2. Generate a checksum archive locally on the instance:

   ```bash
   tar -C results -czf /tmp/bottleneckshift-results.tgz runs figures
   sha256sum /tmp/bottleneckshift-results.tgz \
     > /tmp/bottleneckshift-results.tgz.sha256
   ```

3. Download the archive and checksum over SSH or upload them to a private,
   access-controlled object store. Verify the checksum after transfer.
4. Inspect manifests for hostnames, usernames, paths, GPU UUIDs, IP addresses, or
   other identifying metadata before publishing anything.
5. Retain raw results outside Git. Commit only small reviewed data required for a
   published figure, plus a provenance note and durable archive checksum/locator.
6. Delete credentials from the host, terminate the GPU, and delete unused disks,
   snapshots, public IPs, and object-store staging files. Confirm in the provider
   console that billing resources are gone.

GPU UUIDs captured by the current manifest are useful for confirming a stable device
within a pair, but should be redacted or replaced with a consistent non-identifying
label in any public artifact.

## Failure playbook

| Symptom | Cheap diagnostic action | Rule before retrying measurement |
|---|---|---|
| GPU quota/capacity unavailable | Check another region or shortlisted provider; do not resize silently | Record the final shape and start both plans there |
| Spot/preemptible eviction | Preserve any logs available and terminate leftovers | Treat attempt as disposable/incomplete; use on-demand for the full pair |
| `nvidia-smi` failure | Recreate from an official GPU image before changing project code | Driver/image must be fixed and recorded |
| vLLM install failure | Save Python/pip/platform details; reproduce with the pinned requirements | Commit and review dependency changes; use a new series |
| Model fails to load/OOM | Verify cache, VRAM, and server log; try the configured small model on a clean host | Any model/server-setting change requires a reviewed config and new series |
| vLLM CLI/schema mismatch | Capture `vllm --version`, help output, and a disposable raw result | Fix runner/analyzer with tests before collecting data |
| SSH disconnect | Reconnect to `tmux`; inspect manifest state | Do not mistake a disconnected client for provider eviction |
| Benchmark failure midway | Preserve the failed series exactly | Never splice repaired repetitions into it; restart the pair |
| Artifact transfer failure | Keep the instance stopped only if stopped storage billing is understood | Verify two copies plus checksum before deletion |
| Unexpected metric signature | Check errors and realized token counts first | Preserve it; do not tune or discard based on the desired hypothesis |

## Definition of ready and definition of done

### Ready for measured execution

* A dated provider quote and hard cost cap are recorded.
* Quota and capacity are confirmed for the chosen on-demand shape.
* The exact commit passes compile, unit, dry-run, preflight, and disposable inference
  checks on the target GPU shape.
* Forward and reverse configs are reviewed and unchanged.
* The checkout is clean and detached at the reviewed commit.
* Artifact transfer, checksum verification, and automatic shutdown have been tested.
* No long-lived repository or cloud credential is stored on the instance.

### Done with the cloud iteration

* Both measured series are complete, or a failed attempt is explicitly retained as
  incomplete without being presented as evidence.
* Raw artifacts and checksums exist in two controlled locations.
* Public artifacts have been reviewed for identifying metadata and secrets.
* The analysis runs without inventing missing observations.
* Provider resources have been deleted and billing cessation confirmed.
* Any fixes are committed, reviewed, and pushed separately from measurement data.

## Later optimization, only if repeated rentals justify it

After at least one successful pair, consider a provider-neutral bootstrap script or
an immutable image with a recorded digest. Add it only if it reduces observed setup
errors. Do not introduce Terraform, Kubernetes, distributed serving, dashboards, or
a general benchmark platform for this milestone. The lean workflow above keeps the
scientific variables visible and stays within the project's intended scope.
