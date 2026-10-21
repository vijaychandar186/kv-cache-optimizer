# Troubleshooting

| Symptom | Action |
|---|---|
| Azure allocation failure | Check regional family quota, SKU restrictions, and actual capacity. Try an available approved region/zone. |
| Marketplace image missing | List `microsoft-dsvm:ubuntu-hpc:2404` versions in the selected region and pin an available version. |
| CUDA initialization or unsupported PTX | Check driver version against installed wheel requirements; use a current HPC image/driver, reboot, and run `doctor` again. |
| NVLink/NCCL failure | Inspect `nvidia-smi topo -m`, `nvidia-smi -q`, Fabric Manager status, and HPC image versions. Temporarily set `NCCL_DEBUG=INFO` and retain server logs. |
| Model loading OOM | Stop other GPU jobs. Check actual free GPU memory, TP size, stale serving workers, and dtype. Reduce `max_num_seqs` or `max_model_len`; record every configuration change. |
| HF research model offloads to CPU | Increase `--max-gpu-memory-gib` only if memory exists, or expose more GPUs. The runner rejects CPU/disk offload. |
| FP8 backend unsupported | Verify Hopper hardware and the pinned environment. Run the baseline first. Do not silently fall back and label that result FP8. |
| Speculative profile slower | Inspect proposal acceptance, use a representative corpus, and compare lower concurrency. Low repetition and scheduler overhead can erase gains. |
| Benchmark returns 401 | Export the same `VLLM_API_KEY` in the benchmark and serving shells. |
| Benchmark reports manifest mismatch | Start the intended profile and use that server's current manifest/model lock. |
| Corpus too short | Supply a corpus with at least the requested context plus research quality continuation tokens. |
| Triton import fails on Windows | Run Triton on Linux. Windows can run client/CPU tests; research CPU tests use Torch's reference implementation. |
| uv removes GPU packages | Include `--extra gpu` in every related `uv run`, or use a separate project environment for client commands. |
| GPU memory unchanged with FP8 | vLLM reserves a pool near the configured utilization target; compare token capacity and throughput as well as sampled memory. |
| Windows blocks a local PowerShell script | Review the script, then run it with `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/provision.ps1` followed by its parameters. This applies only to that process and does not change your machine policy. Organization policies can still block it. |

Only stop processes you own. After Ctrl+C, inspect `nvidia-smi` before restarting; failed workers can retain memory. Server manifest metadata describes the launch configuration, so archive it together with server logs for each experiment.
