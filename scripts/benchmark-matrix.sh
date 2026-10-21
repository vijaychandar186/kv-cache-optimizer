#!/usr/bin/env bash
# Run on the same VM as an already-running server. One profile per invocation.
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:?Usage: bash scripts/benchmark-matrix.sh PROFILE [RUN_ID]}"
run_id="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${VLLM_API_KEY:?Export VLLM_API_KEY in this shell}"
uv run --locked --extra gpu kv-lab benchmark \
  --profile "$profile" --contexts 2048 4096 8192 --concurrency 1 4 16 \
  --requests 64 --output-tokens 256 --repeats 3 \
  --output "results/$run_id/$profile"
uv run --locked --extra gpu kv-lab report "results/$run_id"
