# Validation status

Checked locally on Windows with Python 3.12.10 and dependencies installed through uv's lockfile.

| Check | Result |
|---|---|
| `uv lock` / locked dependency installation | Passed; Linux and Windows dependency graphs resolved |
| Ruff lint and formatting | Passed |
| Pytest with research dependencies | 17 passed; 1 GPU module skipped because this host is Windows |
| Native OLMo3 versus research-cache logits | Passed on a tiny random model, including sliding-window crossings and chunked prefill |
| INT8 cache append and eviction policy | Passed against the Torch reference backend |
| Benchmark artifact flow | Passed with an in-process HTTP server mock, including missing GPU telemetry |
| PowerShell provisioning contract | Passed with Azure CLI mocked; checks preview, deployment arguments, image pinning, and auto-shutdown |
| Azure Bicep compilation | Passed with Bicep v0.48.1, no compiler warnings; official release asset SHA256 verified |
| Bash scripts | Syntax checks passed with Git Bash |
| CLI entry point | `kv-lab --help` passed |

The local sandbox required placing uv's cache and pytest's temporary files inside the workspace. Equivalent verification commands:

```powershell
uv --cache-dir .uv-cache sync --locked --extra research
uv --cache-dir .uv-cache run --locked --extra research ruff check .
uv --cache-dir .uv-cache run --locked --extra research ruff format --check .
uv --cache-dir .uv-cache run --locked --extra research pytest -q --basetemp .pytest-tmp/verified
powershell -NoProfile -ExecutionPolicy Bypass -File tests/provision.tests.ps1
```

CI checks Windows with the research dependencies and Linux with the default client dependencies. The GPU tests additionally check Triton/Torch equivalence and integration when run on the VM.

Azure provisioning, multi-GPU NCCL, Triton GPU kernels, 32B model loading, FP8 serving, speculative serving, and throughput measurements require the Linux GPU VM. No fabricated benchmark results are included.
