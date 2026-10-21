#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
for context in 2048 4096 8192; do
  uv run --locked --extra gpu kv-lab research --context "$context" \
    --output "results/research-$run_id/context-$context.json"
done
