#!/usr/bin/env bash
# Never expose server GPUs to development tests.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
exec "$project_root/.venv/bin/python" -m unittest discover -s "$project_root/tests" -v
