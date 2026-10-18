#!/usr/bin/env bash
# User-run CPU fitting experiments on the saved exact-task cache; no ProGen reload.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
run="${1:-runs/exact-first}"
output="${2:-runs/exact-optimization}"
# A fresh parent protects all arms from accidental reuse, including partial runs.
mkdir -- "$output"

scripts/run.sh diagnose-capacity --run "$run" --output "$output/original-lr" \
  --steps 10000 --learning-rate 0.001 --seed 0
scripts/run.sh inspect-fit --run "$run" --fit "$output/original-lr" \
  --output "$output/original-lr-audit"

scripts/run.sh diagnose-capacity --run "$run" --output "$output/lower-lr" \
  --steps 10000 --learning-rate 0.0001 --seed 0
scripts/run.sh inspect-fit --run "$run" --fit "$output/lower-lr" \
  --output "$output/lower-lr-audit"

# Summarize fixed-final checkpoints separately from the best logged diagnostic TV.
.venv/bin/python - "$output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = [
    "# Exact optimization comparison",
    "",
    "User-run privileged CPU fits: same cached inputs, seed 0, head, objective, and 10,000 updates. Only the learning rate differs. This does not pass the three-seed TB gate.",
    "",
    "| Arm | TV at step 2,000 | Final TV | Lowest logged TV | Step of lowest TV |",
    "|---|---:|---:|---:|---:|",
]
for arm in ("original-lr", "lower-lr"):
    result = json.loads((root / arm / "diagnostics.json").read_text())
    at_2000 = next(row for row in result["history"] if row["step"] == 2000)
    best = result["best_observed_tv"]
    lines.append(
        f"| {arm} | {at_2000['total_variation']:.6f} | "
        f"{result['total_variation']:.6f} | {best['total_variation']:.6f} | {best['step']} |"
    )
lines.extend([
    "",
    "The engineering reference is TV <0.05. All checkpoints remain fixed-final; lowest logged TV is observational and can miss between-log minima. See each arm's progress.jsonl and audit for KL, gradient size, and error by prefix length.",
    "",
    "If longer training fits, the original budget was insufficient for this optimizer. If only the smaller rate fits, the result supports a learning-rate issue. If both stall, neither result proves insufficient capacity: inspect feature conditioning and an alternative optimizer next. A privileged fit passing still requires a fresh three-seed TB experiment before biology.",
])
report = root / "comparison.md"
report.write_text("\n".join(lines) + "\n")
print(report.resolve())
PY
