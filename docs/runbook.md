# Experiment commands

## Direct charge-context comparison — current

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # choose your GPU
scripts/run.sh idr-context --output runs/idr-context-v1 --device cuda
```

Runs the unchanged head and the direct-context head from scratch for all three
seeds, then evaluates both automatically. Uses one shared prior pool. The
[protocol](experiments/idr_context_2026-11-12.md) owns the matched budgets and
interpretation. Preflight only: `scripts/run.sh idr-context --check-only`.

If generation completed but scoring failed, replay scoring on CPU:

```bash
scripts/run.sh evaluate-idr-context --run runs/idr-context-v1 --output runs/idr-context-v1-evaluation
```

All outputs need new directories; source runs remain unchanged. Training does
not automatically resume after interruption.

## Real-state charge-learning diagnostic — completed

Completed in 4 min 46 s. Commands are retained for reproducibility; use a new
output directory for a deliberate rerun. The current training command is above.

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
scripts/run.sh idr-diagnose --output runs/idr-learning-diagnostic-v1 --device cuda
```

Captures 768 continuations from initial/final heads; automatically analyzes gradient
variability, token corrections and disposable updates. No sustained training.
[Protocol and rationale](experiments/idr_learning_diagnostic_2026-11-12.md).
Preflight without inference: `scripts/run.sh idr-diagnose --check-only`.

If capture finished, replay analysis on CPU without ProGen:

```bash
scripts/run.sh analyze-idr-diagnostic --run runs/idr-learning-diagnostic-v1 --output runs/idr-learning-diagnostic-v1-analysis
```

Use fresh output directories. Source models and completed experiments are unchanged.

## IDR charge guidance — completed pilot

The pilot and its diagnostic are complete. The command below is
retained for reproducibility; choose a new output directory for any deliberate rerun.
From the project root:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
scripts/run.sh idr-charge --output runs/idr-charge-v1 --device cuda
```

This trains only the GFN head, then automatically generates comparisons and scores
charge/disorder/diversity on CPU. [Protocol and budgets](experiments/idr_charge_2026-11-12.md).
Read-only preflight: `scripts/run.sh idr-charge --check-only`. If all generation
completed but scoring failed, recover without loading ProGen:

```bash
scripts/run.sh evaluate-idr-charge --run runs/idr-charge-v1 --output runs/idr-charge-v1-evaluation
```

Existing output directories are refused. Partial-training automatic resume is not
implemented; intermediate head/optimizer/RNG checkpoints remain available for diagnosis.

## IDR baseline generation and evaluation

The reference panel and pinned metapredict package are prepared in this workspace.
Activate the existing `.venv`, select your GPU, then run:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
scripts/run.sh idr-baseline --output runs/idr-baseline-v1 --device cuda
```

No training: 384 fixed-length ProGen samples, automatic CPU disorder/descriptor
scoring, and `report.md` plus raw sequences and detailed metrics. The
[protocol](experiments/idr_baseline_2026-11-12.md) owns settings and interpretation.
CPU-only readiness check: `scripts/run.sh idr-baseline --check-only`.

If generation completed but evaluation failed, or to replay evaluation:

```bash
scripts/run.sh evaluate-idr --run runs/idr-baseline-v1 --output runs/idr-baseline-v1-evaluation
```

This verifies saved samples/scorer provenance and uses CPU only. It never loads
ProGen. Existing output directories are refused; use a fresh output name.

## Retained stability commands

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


## Local split robustness

[Protocol](experiments/split_robustness_2026-11-09.md). After setting
CUDA_VISIBLE_DEVICES yourself:

```bash
source .venv/bin/activate
scripts/run.sh split-study --output runs/split-robustness-v1 --device cuda
```

This computes frozen ProGen priors for three new evaluation sets, fits matched
ridge/MLP/constant comparisons, and evaluates exact targets at beta 1. It reuses
pooled full-sequence features and does not train a GFlowNet. Results appear in
runs/split-robustness-v1/report.md, with complete per-split artifacts underneath.
Use a new output directory; no automatic overwriting or resume is performed.

CPU-only input verification:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh split-study --check-only
```


## Mutation-position holdout

[Protocol](experiments/position_holdout_2026-11-09.md). Activate .venv and set
CUDA_VISIBLE_DEVICES yourself, then run:

```bash
scripts/run.sh position-study --output runs/position-holdout-v1 --device cuda
```

The command uses configs/position-holdout.json. It fits constant, ridge and MLP
on three grouped-position splits and evaluates their exact targets automatically.
Frozen ProGen computes each new prior; no GFN training is performed. Read
runs/position-holdout-v1/report.md for mean gains, concentration, success and
diversity. Existing output directories are rejected.

Optional read-only CPU check:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh position-study --check-only
```


## GFN on position-held-out rewards

[Protocol](experiments/position_gfn_2026-11-09.md). With .venv active and your chosen
CUDA_VISIBLE_DEVICES exported:

```bash
scripts/run.sh position-gfn --output runs/position-gfn-v1 --device cuda
```

This builds frozen prefix caches and trains nine heads (three seeds per fixed
position split), then evaluates automatically. Read runs/position-gfn-v1/report.md;
evaluation/results.json contains all metrics and evaluation/summary.json summarizes
seed replication within each split. Expect hours rather than the earlier three-
minute predictor-only run; no exact runtime has been measured yet. The command
preserves your GPU selection. Existing outputs are rejected, not overwritten.

Optional CPU preflight (no backbone inference or fitting):

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh position-gfn --check-only
```

CPU-only reevaluation into a fresh directory, without retraining:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-position-gfn   --run runs/position-gfn-v1 --output runs/position-gfn-v1-recheck
```


## Multi-protein stability predictors

[Protocol](experiments/multiprotein_2026-11-10.md). Data preparation is complete.
After selecting CUDA_VISIBLE_DEVICES yourself and activating .venv:

```bash
scripts/run.sh multiprotein --output runs/multiprotein-v1 --device cuda
```

The command extracts frozen ProGen features for 4,160 sequences, fits ridge and
three MLP seeds plus the constant control, and evaluates automatically. See
runs/multiprotein-v1/report.md and evaluation/results.json. No GFlowNet training
or generation is included. Existing output paths are rejected. The first completed
run took 57.875 seconds on the user's GPU.

CPU preflight:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh multiprotein --check-only
```

CPU saved-prediction metric replay:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-multiprotein   --run runs/multiprotein-v1 --output runs/multiprotein-v1-recheck
```

Data re-preparation into a fresh directory (original pinned panel stays unchanged):

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh prepare-multiprotein   --output data/processed/multiprotein-recheck
```

## Ridge regularization only

[Protocol](experiments/regularization_2026-11-10.md). Activate .venv and select
CUDA_VISIBLE_DEVICES yourself, then run:

```bash
scripts/run.sh regularization --output runs/regularization-v1 --device cuda
```

Reuses the completed multi-protein feature cache; no ProGen execution. Performs
25 ridge fits across five training-group folds and one selected-strength refit,
then evaluates on the existing validation proteins. No test evaluation or protein
set expansion. Results: runs/regularization-v1/report.md, cross_validation.json,
selection.json and validation_results.json. Each output directory must be new.
The first completed run took 2.003 seconds on the user's GPU using cached features.

Optional CPU preflight, with no fitting:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh regularization --check-only
```

## Training-protein coverage

[Protocol](experiments/coverage_2026-11-10.md). Activate .venv and select your GPU:

```bash
scripts/run.sh coverage --output runs/coverage-v1 --device cuda
```

Compares nested 44/64/86-parent training sets on the same validation sequences,
using the original ridge strength per training example. Reuses the original
features and extracts only added training features. Evaluation is automatic;
report.md, validation_results.json and validation_predictions.json are in the
new output directory. No test evaluation or GFN training. The first completed run
took 32.321 seconds, including 31.959 seconds for loading/extracting added features.

CPU preflight, with no inference or fitting:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh coverage --check-only
```

CPU preparation is already complete. To reproduce it without changing inputs:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh prepare-coverage \
  --output data/processed/coverage-recheck
```
