# Running stability-guidance experiments

The v1 environment and Mega-scale local task are already prepared in this workspace.
The agent has run only small CPU tests and data preparation. **You run all real
experiments and every GPU operation.** No scheduler submission happens automatically.

## Next user-run experiment: development diversity ablation

[Protocol](experiments/diversity_plan_2026-10-17.md) and
[CPU findings / handoff](experiments/diversity_findings_2026-10-17.md).
This compares original and descendant-count-corrected priors on the three existing
development parents. It is not another confirmation and does not open fresh test
labels. No real ablation has run. About five hours is a planning estimate.

```bash
cd /usr/scratch/agoel320/progen-gflownets-v1
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2  # replace with the GPU you selected
scripts/run.sh diversity --output runs/diversity-v1 --device cuda
```

Evaluation runs automatically. Read `runs/diversity-v1/evaluation/report.md`.
Keep all failed/complete run directories; use a fresh output for a deliberate rerun.
CPU readiness and later saved replay:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh check-diversity
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-diversity \
  --run runs/diversity-v1 --output runs/diversity-v1-replay
```

The second command requires completed, sealed fits. It never retrains or loads
ProGen. Diagnostic commands also require fresh outputs:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh diagnose-diversity \
  --output runs/diversity-diagnostic-repeat
CUDA_VISIBLE_DEVICES='' scripts/run.sh inspect-unique \
  --run runs/confirmation-v1 --output runs/confirmation-unique-repeat
```

The historical unique review is post-hoc; it cannot become new confirmation.

## Completed confirmation: historical launch and CPU replay

This experiment completed in 4 h 8 min; see [current status](status.md) and the
[result analysis](experiments/confirmation_results_2026-10-17.md). The command below is
retained for reproducibility, **not an instruction to start a duplicate run**.
For an intentional future rerun, choose a fresh output directory and select the GPU
yourself. The [protocol](experiments/confirmation_plan_2026-10-16.md) specifies the
fixed choices.

```bash
cd /usr/scratch/agoel320/progen-gflownets-v1
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=2  # replace with your chosen GPU
scripts/run.sh confirm --output runs/confirmation-v1 --device cuda
```

One command fits a single MLP per reserved parent, trains GFN and supervised
guidance for seeds 0–2, seals all fits, then evaluates against ProGen,
generate-and-select and an exact-target reference. Every policy gets 256 proposals
and 32 selection slots. All five parents remain in the report, including failed
scorer screens. The main output is `runs/confirmation-v1/evaluation/report.md`.
Roughly four hours is a planning estimate from the previous single-parent pilot;
actual runtime depends on the GPU and load.

An interrupted training run does not reveal test outcomes. There is no automatic
resume; retain the run and use a new output for any deliberate restart. After all
fits are sealed, evaluation can be completed/replayed on CPU with a fresh output:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-confirmation \
  --run runs/confirmation-v1 --output runs/confirmation-v1-reevaluation
```

This checks the seal before revealing/reading measurements and never refits models.
Existing output directories cannot be reused. The launcher preserves your GPU
selection and does not inspect GPUs or submit scheduler jobs. Candidate identities
have already been prepared; no additional setup command is needed.

## Completed development diagnostic

The previous biological pilot learned its target accurately but did not improve
measured stability. The [new diagnostic](experiments/target_diagnostic_2026-10-15.md)
ran successfully in 62.08 seconds. Exact targets improve measured development
stability on all three proteins at alpha=1, but this does not establish confirmation
performance or consistent superiority over selection. See the
[result and recommended confirmation setup](experiments/target_diagnostic_results_2026-10-15.md).
Those five confirmation parents were subsequently evaluated; see the completed
confirmation section above. They are no longer fresh evaluation data.

### Completed diagnostic command (historical; output already exists)

The command below records how the completed experiment was launched. Do not reuse
its existing output directory. For an intentional rerun, choose a new name:

```bash
cd /usr/scratch/agoel320/progen-gflownets-v1
source .venv/bin/activate
# After choosing a free GPU, e.g. physical GPU 2:
export CUDA_VISIBLE_DEVICES=2
scripts/run.sh diagnose-target --output runs/target-diagnostic-v1 --device cuda
```

The launcher preserves your GPU selection. This command loads frozen ProGen2,
fits ridge and five small MLPs per development protein, and evaluates exact targets
and matched sampling references. It stops before any GFN/confirmation training.
Read `runs/target-diagnostic-v1/evaluation/report.md` and the per-parent JSON files.
No automatic model/prior selection occurs. Output directories cannot be reused;
a failed/interrupted run requires a fresh name. There is no automatic resume.

For later saved-output evaluation only, no GPU or model fit is needed:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-target \
  --run runs/target-diagnostic-v1 --output runs/target-diagnostic-v1-reevaluation
```

Panel preparation has already completed. `prepare-panel` is a CPU-only command for
recreating the configured panel into a new directory if needed; it refuses overwrite.
The readiness record and parameter rationale are in the [diagnostic record](experiments/target_diagnostic_2026-10-15.md).

## Previous biological result

Existing results are in `runs/biological-first/evaluation/report.md`, with the
post-run exact-expectation analysis in `docs/verification/biological-first-analysis.json`.
The original output must not be reused. To independently replay saved evaluation
without training or loading ProGen, choose a fresh output:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate \
  --run runs/biological-first --output runs/biological-first-reevaluation
```

The following historical commands explain the experiment stages; they are not
instructions to rerun completed training or overwrite existing directories.

## Original first GPU command (already run)

The example below is retained for reference; `runs/exact-first` now exists and
must not be reused. A later TB experiment needs a new output and documented settings.


Activate the v1 environment, inspect GPU availability yourself, then select your
GPU. For example, if you choose GPU 2:

```bash
cd /usr/scratch/agoel320/progen-gflownets-v1
source .venv/bin/activate
nvidia-smi
export CUDA_VISIBLE_DEVICES=2
scripts/run.sh run --config configs/exact.json --output runs/exact-first --device cuda
```

This one command loads the frozen local ProGen2-medium checkpoint, checks the
adapter, caches the 85 toy prefixes, trains TB and supervised heads for three
seeds, and evaluates all four methods. It saves the entire run to `runs/exact-first`.
Replace `2` with your selected GPU's index. The launcher never sets or overrides
`CUDA_VISIBLE_DEVICES`, runs `nvidia-smi`, selects a GPU, or submits a scheduler job.
With one GPU exposed, `--device cuda` uses that GPU (PyTorch sees it as logical
device 0). You do not need to put the physical GPU index in the experiment config.

`scripts/run.sh` is a small wrapper: it locates v1, makes the source package
importable, sets reproducibility/tokenizer environment options, and forwards your
arguments to Python in v1's `.venv`. Activation works normally; the wrapper also
uses the correct interpreter when called without activation.

Inspect:

- `runs/exact-first/evaluation/report.md`: readable raw/selected comparison.
- `runs/exact-first/gate.json`: TV, both KL directions and log-partition errors per seed.
- `runs/exact-first/status.json`: completion/gate/failure status.
- `runs/exact-first/seed-*/gfn_progress.jsonl`: progress written during training.
- `runs/exact-first/costs.json`: setup, training, lookup and memory accounting.

The real exact gate requires TV <0.05 in each seed. If it fails, outputs remain
available and the command exits with code 2. Do not proceed to biological guidance
by changing the saved gate file. No real convergence result is claimed in advance.

## Evaluation only, without retraining

Saved-run evaluation uses the cached frozen features and saved heads; it needs no
GPU or loaded ProGen checkpoint:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate \
  --run runs/exact-first --output runs/exact-first-evaluation
```

Every output directory must be new. Choose a new name for another evaluation.
The same saved seeds reproduce raw samples, selection indices and metrics on the
same CPU software environment; timing naturally changes.

## If the exact learning gate fails

The privileged capacity diagnostic fits the same head to exact conditional targets:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh diagnose-capacity \
  --run runs/exact-first --output runs/exact-capacity
```

This is a user-run diagnostic experiment, not an agent development smoke test.
It uses the finite cache on CPU. If it also fits poorly, investigate capacity or
optimization; if it fits well, investigate TB optimization and exploration. Its
results are explicitly privileged and cannot serve as a fair sample-efficiency
baseline. Settings are in [hyperparameters.md](hyperparameters.md).

## Biological pilot, after the exact gate passes

```bash
scripts/run.sh run --config configs/biological-pilot.json --output runs/biological-first --device cuda
```

The current pilot configuration requires `runs/exact-decay-long` to have passed
on the same checkpoint. The historical `biological.json` remains unchanged and
still references the failed first exact run; use `biological-pilot.json`. It then extracts Mega-scale features, validates a ridge/MLP scorer,
and starts guidance only if the scorer gate passes. It uses the prepared
`data/processed/local-v1` task, with test values reserved for final evaluation.
All initial methods share the same selected scorer.

If scorer validation fails, inspect `scorer_validation.json` and `costs.json`.
No guidance training or test-label evaluation will have started. Preserve the
failed run; revise the scorer/task using training/validation evidence and make a
new run. Do not select a model using the withheld stability outcomes.

For saved biological evaluation:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate \
  --run runs/biological-first --output runs/biological-first-evaluation
```

## Development checks

```bash
scripts/test_cpu.sh
CUDA_VISIBLE_DEVICES='' scripts/run.sh doctor
CUDA_VISIBLE_DEVICES='' scripts/run.sh run \
  --config configs/exact.json --output runs/my-cpu-smoke --smoke
```

Smoke mode is explicitly CPU-only: tiny random ProGen, one seed, 20 updates and
small samples. It never loads pretrained weights and never passes the real gate.
Biological smoke mode rejects real Mega-scale manifests: its synthetic branch is
exercised by the unit tests. Real measurements are reserved for your actual scorer
experiment.

Optional lint/format checks (Ruff 0.14.0 is installed in this workspace):

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

## Data preparation and environment recreation

Already completed; no need to repeat for the first run. To retrieve the pinned
CSV in a fresh workspace, then create the prepared task:

```bash
.venv/bin/python scripts/fetch_megascale.py
CUDA_VISIBLE_DEVICES='' scripts/run.sh prepare
```

The downloader resumes partial compressed bytes and verifies them before publishing
the CSV. Preparation refuses to overwrite a task; use `--output` for a new manifest
and update a copied biological configuration accordingly. See [data.md](data.md).

The current `.venv` is based on Python 3.10 at
`/usr/scratch/agoel320/miniconda3/envs/abmlm/bin/python`, with system packages exposed
for the existing PyTorch installation. Transformers 4.27.4 and Tokenizers 0.13.3
are installed inside v1. The launcher invokes `.venv/bin/python` directly and does
not rely on stale activation paths from the renamed test project.

For a fresh isolated environment, use Python 3.10+ to create `.venv`, then install
this package and its pinned dependencies with `.venv/bin/python -m pip install -e .`.
Choose an appropriate PyTorch wheel for your machine; run `doctor` before any real
experiment. Environment installation downloads dependencies but launches no model.
Do not delete/recreate the existing environment while an experiment is active.

## Reading results correctly

`complete` on a main experiment means configured execution and gates completed.
For a capacity diagnostic, `complete` means fitting finished; it has no gate and
does not certify convergence. An audit completing certifies analysis execution only. It does not mean the
method improves biological stability. `exact_gate_failed`, `scorer_gate_failed`,
`failed`, `interrupted` and `smoke_complete` remain distinct. A directory or
`running` marker alone is not evidence of completion.

Raw-policy distributions and selected batches answer different questions. The
baseline selection pool is larger (1,024 versus 256); selected batches contain 32
slots, including duplicates. Compare quality with its recorded cost, and retain
all failures and all seeds. See [architecture](architecture.md) and
[hyperparameters](hyperparameters.md) for exact definitions and limitations.
