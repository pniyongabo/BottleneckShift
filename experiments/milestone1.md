# Milestone 1 protocol: concurrency perturbation

## Question and hypothesis

Does increasing maximum client concurrency from 1 to 8 produce the queueing and
batching signature described in the repository README while all configured workload
and server factors remain fixed?

This hypothesis was written before collecting results. Expected signatures are
higher TTFT and end-to-end latency, rising throughput until saturation, potentially
non-monotonic TPOT, stable output lengths, and no errors. These expectations are not
findings.

## Controls and procedure

1. Use a dedicated, otherwise-idle GPU host and record its environment automatically.
2. Run `configs/milestone1.toml` without editing it after a series begins.
3. Keep the server alive across conditions so model loading is not part of timings.
4. Execute three repetitions per condition and preserve every raw result and failure.
5. Repeat the entire series with reversed condition order before drawing conclusions.
6. Compare repetition distributions; report errors and realized token counts before
   interpreting latency or throughput.

The initial runner uses fixed order to remain transparent and minimal. A second,
reversed config is intentionally not checked in until the first GPU/model choice is
confirmed. Any deviations belong in the report and alongside the archived results.

## Interpretation guardrails

The intervention supports statements about this workload setting on this system.
GPU/CPU utilization may help explain a signature but cannot establish causation.
Network, host, and GPU contention have not been manipulated in this milestone.
