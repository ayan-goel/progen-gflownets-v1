#!/usr/bin/env bash
# Use this project's venv and preserve the user's CUDA_VISIBLE_DEVICES selection.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
exec "$project_root/.venv/bin/python" -m stability_gfn "$@"
