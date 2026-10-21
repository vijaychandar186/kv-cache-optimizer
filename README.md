# Distributed LLM Inference — KV-Cache & Decode Optimization

A runnable experiment project for `allenai/Olmo-3-32B-Think` on Azure NVIDIA GPU VMs. All Python packages and commands use **uv**. Start with the walkthrough below; the default hardware is one `Standard_ND96isr_H100_v5` VM with eight H100 80 GB GPUs.

The project includes:

- Azure Bicep infrastructure and a PowerShell deployment script with a preview mode.
- vLLM tensor parallelism, continuous batching, chunked prefill, and four serving profiles.
- Benchmarks over 2,048 / 4,096 / 8,192 input tokens, concurrency sweeps, raw request traces, and GPU memory sampling.
- OpenAI Triton INT8 KV quantization, dequantization, and physical token-compaction kernels.
- An OLMo3 Hugging Face research decoder that actually uses these kernels during generation and evicts tokens using observed attention importance.
- Controlled cache ablations, teacher-forced quality checks, CPU tests, GPU tests, and CI.

**Validation status:** the code can be checked on Windows, while vLLM and Triton execute on Linux with NVIDIA GPUs. GPU throughput and Azure deployment must be validated on your VM. No performance improvements or cloud deployment are claimed merely from creating this project. See [VALIDATION.md](VALIDATION.md) for the checks performed in this workspace.

## 1. Understand the experiment paths

| Path | Placement / batching | Optimizations | Outputs |
|---|---|---|---|
| vLLM server | Tensor parallel across 8 GPUs; continuous batching | BF16 or FP8 KV; optional n-gram speculative decoding | Aggregate output tokens/s, latency, TTFT, TPOT, sampled device memory |
| Research decoder | Hugging Face sequential layer placement; one request | Custom Triton INT8 KV, importance eviction, or both | Decode tokens/s, persistent cache bytes, allocator peaks, teacher-forced NLL |

The custom kernels are integrated into the research decoder. They do not alter vLLM's paged-cache internals. Compare optimizations **within each path**. The Hugging Face path has Python, scoring, dequantization, allocation, and compaction overhead, so its speed cannot be directly attributed to cache policy when compared with vLLM.

N-gram speculation proposes tokens from matching prior n-grams and verifies them with the target model. It needs no unverified OLMo draft checkpoint. Benefits depend on token repetition and concurrency; open-ended reasoning may show little benefit or a regression. Eviction changes model behavior and needs a quality evaluation.

## 2. Install local tools and verify the project

On your Windows machine, install [uv](https://docs.astral.sh/uv/getting-started/installation/), [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), Git, and OpenSSH. Use Python 3.12, which uv selects from `.python-version`.

```powershell
cd C:\Users\Vijay\Downloads\kv-cache
uv sync --locked
uv run --locked ruff check .
uv run --locked pytest -m "not gpu and not research"
uv run --locked kv-lab --help
az login
```

`uv.lock` locks the Linux GPU and Windows client dependency graphs. Default installation avoids downloading GPU packages. `--extra gpu` installs pinned vLLM 0.18.0 / PyTorch 2.10.0 and Triton on Linux; `--extra research` installs PyTorch and Triton for the research path. Use the same extra in each `uv run` command to keep uv from removing packages needed by that command.

Optional CPU correctness tests for the actual OLMo cache, using a tiny randomly initialized model and no model downloads:

```powershell
uv sync --locked --extra research
uv run --locked --extra research pytest -m research
```

## 3. Choose Azure quota, region, and image

The default H100 VM allocates **all eight GPUs** even when `--tp 2` uses only two of them. Use TP=8 initially and measure TP=2/4/8 later if you want a scaling study. A100 configurations can run BF16, but this project's preflight intentionally targets Hopper or newer for the FP8 experiment.

In Azure Portal, request the regional GPU-family vCPU quota for your chosen size. Check both quota and regional capacity; passing one does not guarantee the other. Review the current regional price in Azure before creating the VM. Storage and static public IP charges can remain after VM deallocation.

The deployment uses Microsoft's Ubuntu HPC 24.04 image with NVIDIA drivers and GPU fabric support. It resolves an explicit image version and prints it for repeatability. Inspect the current image's component versions; older images may have drivers too old for the pinned PyTorch/vLLM wheels. The `doctor` command executes a CUDA operation to detect this.

## 4. Preview and create the VM

Generate an SSH key if you do not already have one:

```powershell
ssh-keygen -t ed25519
```

Find your current public IPv4 address using your network provider or a trusted IP lookup. Substitute your actual subscription, region, and **single-IP `/32` CIDR**:

```powershell
.\scripts\provision.ps1 `
  -Subscription 'YOUR_SUBSCRIPTION_ID' `
  -Location 'YOUR_GPU_REGION' `
  -AdminSourceCidr 'YOUR_PUBLIC_IP/32' `
  -SshPublicKeyPath "$HOME\.ssh\id_ed25519.pub"
```

This lists quota/SKU restrictions, resolves the image, creates a resource group, and runs an ARM `what-if` preview. It does not allocate a VM. Save the printed image version and rerun with `-ImageVersion 'PRINTED_VERSION' -Deploy` when ready to create the resources.

```powershell
.\scripts\provision.ps1 `
  -Subscription 'YOUR_SUBSCRIPTION_ID' `
  -Location 'YOUR_GPU_REGION' `
  -AdminSourceCidr 'YOUR_PUBLIC_IP/32' `
  -SshPublicKeyPath "$HOME\.ssh\id_ed25519.pub" `
  -ImageVersion 'PRINTED_VERSION' `
  -Deploy
```

SSH is limited to your `/32`; port 8000 is closed. Password login is disabled. The script configures daily auto-shutdown at 03:00 UTC. For your October Eastern Daylight Time schedule, that is 11:00 PM on the preceding local date; UTC remains fixed when daylight saving time changes.

The template provisions a 512 GiB persistent OS disk for the environment, roughly 64 GB of BF16 model weights, and artifacts. It does not format local NVMe devices. See [infra/main.bicep](infra/main.bicep) before deployment.

## 5. Copy the project and bootstrap the GPU environment

Get the public IP:

```powershell
az vm show -d -g rg-olmo-kv-lab -n olmo-kv-lab --query publicIps -o tsv
```

From the project directory, package source without your environment or secrets, then upload it:

```powershell
tar -czf "$env:TEMP\kv-cache-lab.tar.gz" README.md VALIDATION.md pyproject.toml uv.lock .python-version .env.example src configs scripts infra tests docs .github .gitignore .gitattributes
scp "$env:TEMP\kv-cache-lab.tar.gz" azureuser@VM_PUBLIC_IP:~/
ssh azureuser@VM_PUBLIC_IP
```

On the **Linux VM**:

```bash
mkdir -p ~/kv-cache
tar -xzf ~/kv-cache-lab.tar.gz -C ~/kv-cache
cd ~/kv-cache
# Review/install uv using its official installer if not already installed.
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install-uv.sh
sh /tmp/install-uv.sh
source "$HOME/.local/bin/env"
bash scripts/bootstrap.sh
```

Inspect GPU topology and the installed HPC components:

```bash
nvidia-smi topo -m
cat /opt/azurehpc/component_versions.txt
uv run --locked --extra gpu kv-lab doctor --tp 8
```

If a driver check fails, update the driver and matching NVIDIA Fabric Manager using the Azure HPC image guidance, reboot, and rerun `doctor`. Do not replace the working GPU fabric stack with an unrelated desktop driver. See [docs/troubleshooting.md](docs/troubleshooting.md).

## 6. Pin and download the exact model revision

```bash
uv run --locked --extra gpu kv-lab download
cat models/model-lock.json
```

The command resolves `main` to a full Hugging Face commit SHA, downloads safetensors/tokenizer files, and writes the SHA to `models/model-lock.json`. Later runs reuse this SHA. The server, tokenizer, research decoder, and benchmark all use it. Copy that lock into your experiment archive. Model files remain in the Hugging Face cache; do not commit weights.

OLMo is public. If rate limits or a proxy require authentication, export `HF_TOKEN` in your shell without committing it. To deliberately change model versions, supply an explicit `--revision COMMIT_SHA`. Do not change revision between benchmark arms.

## 7. Start the BF16 baseline

Use `tmux` or a persistent SSH session for serving. Create one API key for the experiment and use the same value in the benchmark shell:

```bash
umask 077
printf 'VLLM_API_KEY=%s\n' "$(openssl rand -hex 32)" > .env
set -a
source .env
set +a
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
uv run --locked --extra gpu kv-lab serve --profile baseline
```

Keep that terminal open. The key is saved in your git-ignored `.env` with owner-only permissions, then exported into the environment. It is omitted from command arguments and result artifacts. In each additional shell, run `set -a`, `source .env`, and `set +a` from the project directory to load the same key. Create this file once per experiment; keep its value unchanged while the server is running.

Configuration in [configs/serving.toml](configs/serving.toml):

- TP=8 with multiprocessing on one VM, BF16 model weights, 85% GPU memory target.
- 12,288 total model tokens, allowing 8K input plus generated tokens.
- Up to 32 simultaneous sequences and 8,192 scheduled tokens per iteration.
- vLLM continuous batching and chunked prefill; simultaneous benchmark requests exercise batching.
- Prefix caching disabled to prevent shared benchmark prefixes from changing the comparison.
- vLLM generation defaults explicitly selected; benchmark sampling parameters are explicit.

Inspect a command without starting a server:

```bash
uv run --locked --extra gpu kv-lab serve --profile baseline --dry-run
```

The server manifest is saved to `results/server.json`. Keep model, TP, memory target, batching settings, hardware, and input workload identical between optimization profiles. Restart the server for each profile.

## 8. Smoke-test the API

In a second shell on the VM, export the same `VLLM_API_KEY` and run:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"allenai/Olmo-3-32B-Think","messages":[{"role":"user","content":"Compute 17 times 19."}],"max_tokens":256,"temperature":0}'
```

The thinking model can consume the whole output budget in reasoning. Increase `max_tokens` when evaluating final answer correctness.

To reach the API from Windows, keep it bound to the VM's loopback interface and create an SSH tunnel:

```powershell
ssh -N -L 8000:127.0.0.1:8000 azureuser@VM_PUBLIC_IP
```

Run measured GPU-memory benchmarks on the VM. A Windows client cannot measure remote GPU memory using its local `nvidia-smi`.

## 9. Run a short benchmark, then the full matrix

On the VM in the second shell:

```bash
cd ~/kv-cache
set -a
source .env
set +a
uv run --locked --extra gpu kv-lab benchmark --profile baseline \
  --contexts 2048 --concurrency 1 4 --requests 8 \
  --output-tokens 64 --repeats 1 --output results/smoke/baseline
```

After the smoke run succeeds, use one run ID across all profiles:

```bash
bash scripts/benchmark-matrix.sh baseline experiment-001
```

The full matrix uses 3 context sizes × 3 concurrency levels × 3 repeats × 64 measured requests, plus warmups. Start with the short run to estimate runtime and cost.

Default prompts contain deterministic random non-special token IDs. This gives controlled input lengths and cache pressure; it is a **systems workload**, and it will not establish reasoning quality or representative n-gram speedups. For natural text, provide your own sufficiently long UTF-8 corpus:

```bash
uv run --locked --extra gpu kv-lab benchmark --profile baseline \
  --corpus /path/to/long-corpus.txt --output results/natural-text/baseline
```

The corpus must contain at least as many tokens as the largest requested context. Each request takes a seeded contiguous slice; it is not silently repeated. Archive its SHA256 and use the same file for every profile. Include a representative reasoning/coding workload before drawing application-level conclusions.

## 10. Compare FP8 and speculative decoding

Stop the previous server with Ctrl+C and ensure its GPU processes exit. In the serving shell:

```bash
uv run --locked --extra gpu kv-lab serve --profile fp8
```

Then in the benchmark shell:

```bash
bash scripts/benchmark-matrix.sh fp8 experiment-001
```

Repeat the server restart and benchmark for `speculative` and `fp8-speculative`:

```bash
# Serving shell, one at a time:
uv run --locked --extra gpu kv-lab serve --profile speculative
# Benchmark shell:
bash scripts/benchmark-matrix.sh speculative experiment-001

# After stopping that server, serving shell:
uv run --locked --extra gpu kv-lab serve --profile fp8-speculative
# Benchmark shell:
bash scripts/benchmark-matrix.sh fp8-speculative experiment-001
```

The FP8 profiles use vLLM's built-in `fp8_e4m3` cache with dynamic scale calculation supported by the pinned version. This flag is deprecated in newer vLLM versions; upgrade it deliberately together with its tests. FP8 storage savings do not imply lower sampled device memory: vLLM preallocates a cache pool according to the configured memory budget. Record available cache capacity and scheduler metrics from server logs/`/metrics` as well.

Inspect speculative acceptance counters exposed by your pinned server:

```bash
curl --fail -H "Authorization: Bearer $VLLM_API_KEY" \
  http://127.0.0.1:8000/metrics > results/metrics-snapshot.txt
```

Compare acceptance, throughput, and latency at each concurrency. Run the same natural-text corpus when studying lookup-based speculation. Do not assume that combining optimizations improves performance.

## 11. Run the custom Triton cache experiments

Stop vLLM first to free GPU memory. Verify the kernels and the cache integration:

```bash
uv run --locked --extra gpu pytest -m research
uv run --locked --extra gpu pytest -m gpu
uv run --locked --extra gpu kv-lab kernels --tokens 8192
```

Run all four cache arms on the actual 32B model:

```bash
uv run --locked --extra gpu kv-lab research --context 2048 \
  --budget 1024 --sink 4 --recent 128 --output results/research-2k.json
bash scripts/research-matrix.sh experiment-001
```

For meaningful quality measurements:

```bash
uv run --locked --extra gpu kv-lab research --context 8192 \
  --corpus /path/to/long-corpus.txt --quality-tokens 128 \
  --output results/research-natural-8k.json
```

`baseline`, `int8`, `evict`, and `int8-evict` use the same research attention/decoder. K and V have independent per-token/per-head FP32 scales. New tokens are quantized once. The attention calculation dequantizes the prior layer cache; persistent INT8 storage therefore does not remove all temporary BF16 allocations. Physical compaction uses Triton gathers, including scales.

Full-attention layers share chronological retained indices selected by mean attention importance across those layers. Four initial sink tokens and 128 recent tokens are protected by default. Importance is an exponential moving accumulation of the last query's attention probabilities. The policy is an experimental heuristic, not a reproduction of all H2O details. Sliding layers keep their native recent window. Original absolute positions drive RoPE and masks after eviction.

Prefill is processed in chunks of 128 to limit temporary attention allocations. The research decoder uses sequential Hugging Face layer placement across available GPUs, not vLLM tensor parallelism. It dequantizes, concatenates, scores, and compacts cache tensors at Python boundaries; expect overhead. See [docs/design.md](docs/design.md) for constraints and integration details.

Quality output includes NLL for an identical teacher-forced continuation in each arm. Random-token NLL is only a numerical diagnostic. Use natural text plus downstream answer accuracy/retrieval tasks to choose an eviction budget. No quality threshold is pre-certified here.

## 12. Read results and export plots

```bash
uv run --locked --extra gpu kv-lab report results/experiment-001
# Optional plot dependency; keep gpu selected if the server is using this environment.
uv run --locked --extra gpu --extra plots kv-lab report results/experiment-001 --plot
```

`comparison.csv` contains individual repetitions. Optional plots report means with min/max ranges; they are not statistical confidence intervals. Preserve raw results. See [docs/benchmarking.md](docs/benchmarking.md) for definitions, failure handling, and comparison rules.

Useful artifacts:

| Artifact | Purpose |
|---|---|
| `models/model-lock.json` | Exact model commit |
| `results/server.json` | Serving command/configuration and package versions |
| `<run>/<profile>/metadata.json` | Workload, model SHA, environment, and copied server manifest |
| `*-requests.jsonl` | Text, token counts, errors, request latency, content-chunk timestamps |
| `*-summary.json` | Throughput, latency percentiles, and sampled memory peaks |
| `*-gpu-samples.json` | Raw simultaneous whole-device memory/utilization samples |
| `research*.json` | Cache bytes, allocator peaks, generated IDs, NLL, and decode timing |

## 13. Shut down the GPU VM

From Windows when finished:

```powershell
az vm deallocate -g rg-olmo-kv-lab -n olmo-kv-lab
az vm get-instance-view -g rg-olmo-kv-lab -n olmo-kv-lab --query instanceView.statuses -o table
```

Verify the status is `VM deallocated`. Download needed artifacts first. When the entire experiment is disposable, remove its resource group through Azure Portal or Azure CLI; this also deletes the VM, disks, network, and remaining artifacts in that group.

## Sources and version choices

- [Ai2 model card](https://huggingface.co/allenai/Olmo-3-32B-Think): Transformers 4.57+ support and model usage.
- [Model configuration](https://huggingface.co/allenai/Olmo-3-32B-Think/blob/main/config.json): 40 query heads, 8 KV heads, 64 layers, mixed sliding/full attention, and a 4,096-token sliding window.
- [Azure H100 VM specifications](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndh100v5-series): eight H100 GPUs and intra-VM NVLink.
- [Azure HPC images](https://github.com/Azure/azhpc-images): available Ubuntu HPC images and installed GPU stack.
- [vLLM 0.18.0 model registry](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/model_executor/models/registry.py): OLMo3 support.
- [vLLM quantized KV cache](https://docs.vllm.ai/en/v0.18.0/features/quantization/quantized_kvcache/): FP8 configuration.
- [vLLM n-gram speculation](https://docs.vllm.ai/en/v0.18.2/features/speculative_decoding/n_gram/): prompt-lookup proposal configuration.
- [vLLM pinned CUDA dependencies](https://github.com/vllm-project/vllm/blob/v0.18.0/requirements/cuda.txt): PyTorch 2.10.0 dependency alignment.
- [Triton documentation](https://triton-lang.org/main/index.html): GPU kernel programming.
