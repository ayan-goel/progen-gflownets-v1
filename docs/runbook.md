# Experiment commands

Run from the project root. The launcher uses the existing .venv and preserves
your CUDA_VISIBLE_DEVICES selection. Outputs must be new directories.

## CPU checks

~~~bash
source .venv/bin/activate
scripts/test_cpu.sh
CUDA_VISIBLE_DEVICES='' scripts/run.sh doctor --config configs/biological-pilot.json
~~~

These check implementation and files; they do not rerun the biological experiment.

## Reevaluate the completed pilot without training

~~~bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 scripts/run.sh evaluate \
  --run runs/biological-first --output runs/biological-first-review
~~~

This uses saved frozen features and heads on CPU. It does not load ProGen or
create a fresh evaluation set.

## Original real experiments, retained for reproducibility

You select a free GPU and set CUDA_VISIBLE_DEVICES before running these yourself:

~~~bash
export CUDA_VISIBLE_DEVICES=0
scripts/run.sh run --config configs/exact-decay-long.json \
  --output runs/exact-reproduction --device cuda
scripts/run.sh run --config configs/biological-pilot.json \
  --output runs/biological-reproduction --device cuda
~~~

The biological configuration requires the preserved passing runs/exact-decay-long
checkpoint identity. These reproduce existing procedures; they are not a new
linear-baseline/reward study and are not the next recommended experiment.

## Preparation and exact-task diagnostics

- scripts/fetch_megascale.py: original data downloader.
- scripts/run.sh prepare: deterministic local-task preparation.
- scripts/run.sh diagnose-capacity: privileged finite-task conditional fitting.
- scripts/run.sh inspect-fit: read-only saved-fit analysis.
- scripts/diagnose_exact.sh: historical exact-optimization diagnostics.

Do not overwrite the original data task or runs. Consult --help for arguments.
The new predictor/reward study is described below.

## Predictor and reward study

Read the [fixed protocol](experiments/predictor_reward_study_2026-11-03.md).
After choosing an available GPU and setting CUDA_VISIBLE_DEVICES yourself:

~~~bash
source .venv/bin/activate
scripts/run.sh reward-study --output runs/predictor-reward-v1 --device cuda
~~~

This extracts frozen pooled features once, fits constant/ridge/MLP comparisons
and evaluates exact reward targets. It does not train ProGen or another GFlowNet.
The launcher preserves your GPU selection. Results: report.md and results.json
inside the chosen output directory. Outputs also retain features, predictions,
models, split identities, configuration, source, protocol and hashes. Existing
outputs are never overwritten; use a new directory after a failed run.

Saved original-reward analysis can be reproduced without GPU or fitting:

~~~bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh audit-reward --output runs/reward-audit-review
~~~

The completed analysis is at runs/reward-audit-v1/audit.json.


## Ridge-reward GFlowNet

The [9 November protocol](experiments/ridge_gfn_2026-11-09.md) fixes the saved
ridge reward, beta 1, original 204 candidates and three guidance-training seeds.
Select your GPU with CUDA_VISIBLE_DEVICES, then run:

```bash
source .venv/bin/activate
scripts/run.sh ridge-gfn --output runs/ridge-gfn-v1 --device cuda
```

One command trains and evaluates. No ProGen loading, feature extraction or new
predictor fit is needed. The main report is runs/ridge-gfn-v1/report.md; it shows
expected predicted and expected measured stability as separate columns, with
exact target-matching metrics. Per-seed progress is flushed during training.

Optional CPU-only readiness check:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh ridge-gfn --check-only
```

Outputs must be new directories. Real training remains user-run; CPU smoke tests
use only small synthetic fixtures. Evaluation draws do not need another GPU run.


## Fixed reward-noise study

[Protocol and rationale](experiments/reward_noise_2026-11-09.md). After choosing
CUDA_VISIBLE_DEVICES yourself:

```bash
source .venv/bin/activate
scripts/run.sh reward-noise --output runs/reward-noise-v1 --device cuda
```

Runs 18 fits sequentially (about five hours on comparable hardware), then exact
CPU evaluation. Report: runs/reward-noise-v1/evaluation/report.md. Completed clean
ridge controls are reused. ProGen is never loaded or updated. A new output
directory is required; failure records and completed fit checkpoints are retained.
There is no automatic resume.

Optional read-only preflight:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh reward-noise --check-only
```

Reevaluate a completed study without fitting:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-noise \
  --run runs/reward-noise-v1 --output runs/reward-noise-review
```
