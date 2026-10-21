#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this on the Azure Ubuntu HPC VM.' >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun.' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'NVIDIA driver missing. Use the Ubuntu HPC 2404 image and check its component versions.' >&2
  exit 1
fi
nvidia-smi
uv sync --locked --extra gpu
uv run --locked --extra gpu kv-lab doctor --tp "${TP_SIZE:-8}"
echo 'Ready. Next: uv run --locked --extra gpu kv-lab download'
