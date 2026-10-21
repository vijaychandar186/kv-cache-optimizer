# Benchmark methodology

## Workload

The primary sweep uses **input token counts**, not total sequence lengths: 2,048, 4,096, and 8,192. Output is fixed at 256 tokens, EOS stopping is disabled, temperature is zero, and each request has a reproducible seed. Input IDs are sent directly to `/v1/completions` to avoid chat-template overhead changing lengths. Chat API smoke tests are separate.

Each concurrency level is a closed-loop load: a worker submits its next request after the previous request completes. It measures saturation behavior, not an externally scheduled arrival-rate SLO. A worker's request timer begins at dispatch; there is no artificial client queue delay included. At least as many requests as maximum concurrency are required.

Setup, tokenization, and warmup are excluded from timed regions. All requested profiles use identical prompts and settings. Warmup exercises the same shape/concurrency before every measured repetition. Use randomized profile order and multiple complete sessions if thermal effects, other jobs, or startup order are concerns.

## Metrics

| Metric | Definition |
|---|---|
| Aggregate output tokens/s | Successful completion tokens divided by the full cell wall time, including time spent on failures |
| Request latency | Dispatch to stream completion or failure |
| TTFT | Dispatch to the first nonempty content chunk observed by the client |
| TPOT | `(last_content_time - first_content_time) / (completion_tokens - 1)` for outputs longer than one token |
| Latency percentiles | Linear interpolation over successful requests, per cell and repetition |
| Sampled GPU memory | Highest observed whole-device `nvidia-smi memory.used`; simultaneous sum is computed per sample |
| Research cache bytes | Persistently retained K/V, scales, positions, and importance scores, after compaction |
| Research allocator peak | PyTorch allocator's maximum allocated/reserved bytes per device, reset before each measured arm |

Network chunks may contain multiple tokens, especially with speculation. TTFT is client-observed and excludes empty chunks; TPOT is a client approximation. Chunk timestamps are saved as chunk observations and are not reported as per-token inter-token latency. Server usage counters provide token counts; the benchmark does not retokenize output text to guess them.

Missing `[DONE]`, missing usage, server errors, HTTP failures, and mismatched input/output token counts are failures. They remain in raw records and the throughput denominator. The command exits nonzero if any measured request fails. Warmup failures abort the cell. Successful-only latency percentiles must be interpreted alongside failure counts.

The GPU sampler runs on the benchmark host. It includes other processes and CUDA memory outside PyTorch. A 250 ms sampled maximum can miss shorter peaks, and vLLM preallocation can keep it constant across cache dtypes. Never interpret it as exact process-only allocation or persistent-cache size. The research allocator peak and persistent-cache bytes answer different questions.

## Quality and conclusions

FP8 and INT8 need quality checks. Eviction removes information and requires long-context retrieval and downstream task accuracy checks. N-gram verification preserves the target distribution algorithmically, but hardware reduction order and floating-point effects can prevent exact text equality across configurations.

The research runner teacher-forces an identical held-out continuation after the same input. Compare NLL by arm, using natural text. Generated outputs afterward are free-running and can diverge; text mismatch by itself is not an answer-quality score. Validate reasoning answers with enough output budget for the thinking model.

Report each context/concurrency cell, repeats, hardware, model SHA, software lock hash, sampling settings, and failures. A speedup is `optimized_tok_per_sec / baseline_tok_per_sec` for otherwise matched runs. Do not substitute a kernel microbenchmark improvement for an end-to-end serving improvement. Keep representative workloads distinct from synthetic pressure tests.
