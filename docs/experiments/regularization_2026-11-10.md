# Ridge regularization with fixed protein coverage

Status: completed by the user; results verified on CPU on 10 November 2026.

## Question and motivating evidence

Does stronger ridge regularization improve transfer to other proteins when the
training set remains fixed? The user requested this **before** increasing protein
coverage. The [previous diagnosis](multiprotein_2026-11-10.md#follow-up-diagnosis--10-november-2026)
found that ridge's validation disadvantage versus constant came from errors in
protein means, despite useful within-protein ranking. Its selected strength was
1000, the original grid's upper boundary. This motivates an expanded strength
comparison, not a claim that it will solve the problem.

Keep the existing 44 training, 10 validation and 10 test protein groups, the same
65 sequences per parent and all preprocessing. Reuse the completed
runs/multiprotein-v1 frozen features and its ridge checkpoint. No additional
proteins, features, MLPs, ProGen updates, GFlowNets or generation are included.

## Fixed choices and rationale

| Choice | Value and reason |
|---|---|
| Ridge penalties | 1000, 10000, 100000, 1000000, 10000000. Include the current setting and four decade increases to probe a broad shrinkage range; engineering grid, not literature-derived optimum. |
| Ridge objective | Same summed squared error on training-standardized features/labels plus penalty times squared coefficient norm. Bias is zero after centering. No change to the existing primal/dual solver. |
| Penalty convention | Same numerical penalty in folds and full-data refit, matching conventional raw ridge strength and the original fit. With fewer fold-training rows, the penalty is somewhat stronger relative to the average data loss; no hidden rescaling. |
| Cross-validation | Five folds within the original 44 training groups. Sorted group IDs, NumPy default_rng(42) permutation, array_split: 9/9/9/9/8 held-out groups. All variants in a group stay together. Five folds balances coverage and fit cost; one fixed partition, not five independent studies. |
| Normalization | Recompute feature means/scales and label means/scales using each fold's training rows only. Never normalize on the full dataset before cross-validation. |
| Selection | Minimum root mean of per-parent MSE from all out-of-fold predictions. Each training parent has equal weight. At current equal row counts this equals pooled OOF RMSE. Exact ties favor stronger regularization. |
| Constant control | Each fold predicts its own training-label mean. If constant matches/beats the best ridge CV score, report that explicitly; do not promote ridge based on validation alone. Still refit/report the best ridge to answer the regularization comparison. |
| Final fit | Refit only the selected ridge strength on all original 2,860 training rows. No model selection or fitting on outer validation. |
| Comparators | Original saved ridge (penalty 1000), selected ridge, full-training-mean constant, on identical outer-validation identities. |
| Runtime/device | User runs real fitting on their selected CUDA device. No backbone loading/inference. 25 small ridge fits for CV plus one full-data refit; no gradient-update budget. |

The expanded search uses grouped CV rather than reusing outer-validation labels
to select the strength. Because 1000 is included in the same folds, its CV result
is directly comparable to every stronger candidate. A larger selected strength
is not automatically a success: judge its fixed-validation behavior as well.
If 1000 wins, the experiment provides no support for stronger shrinkage under this
selection rule. If the upper boundary wins, record the limitation without
automatically expanding the grid after seeing validation.

## Evaluation boundaries and interpretation

Write folds, out-of-fold predictions, all CV metrics and the selected strength
before refitting and evaluating the selected model on outer validation. The
selection function receives only original training features and labels. Outer
validation labels are loaded for input checks/reporting but never enter CV or
refitting. Test labels and test predictions are not evaluated at all. Existing
test identities/splits remain unchanged. Outer validation was already inspected
in the preceding diagnosis, so it is a development check, not pristine confirmation.

Primary comparison: selected versus original ridge validation RMSE. Also compare
with constant and report mean-error versus centered-variant MSE, pooled and
within-parent Spearman, per-parent errors, and top-10% measured selection lift.
Lift selects seven of each parent's 65 candidates (ties averaged) and compares
with uniform selection of those same candidates, not a ProGen baseline. Better
absolute predictions can come with worse ranking; report that tradeoff. Group
mean centering is diagnostic only, not an oracle correction available in generation.

This experiment does not establish biological function or increased novelty.
After results, decide whether a separately specified protein-coverage study is
warranted. Do not change both training coverage and ridge strength in this run.

## Execution and artifacts

After the user selects CUDA_VISIBLE_DEVICES and activates .venv:

```bash
scripts/run.sh regularization --output runs/regularization-v1 --device cuda
```

CPU preflight (checks saved inputs, no fitting):

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh regularization --check-only
```

The new run saves config, source/protocol snapshots, training/validation rows,
fold assignments, all OOF predictions/metrics, selection, selected and reference
checkpoints, validation predictions/metrics, progress and report.md. Configuration
pins source hashes; data identities/order and data hashes are checked before
execution. Existing output directories are rejected. Original run outputs and
split manifests remain unchanged.

## Readiness verification

All 52 CPU tests pass. Three new synthetic tests cover whole-group folds,
fold-training normalization and constant baselines, out-of-fold scoring,
invariance of selection to outer-validation label changes, checkpoint prediction
replay, and input/output/real-CPU guards. Synthetic runs have no test data files
and execute without them. Source hashes, real feature/identity alignment and
fold sizes (9/9/9/9/8) pass CPU preflight. No real fitting, test-label analysis or
GPU execution was performed by the assistant.

Evidence: [CPU tests](../verification/regularization-cpu-tests-2026-11-10.txt),
[readiness](../verification/regularization-readiness-2026-11-10.json).

## Completed results — 10 November 2026

runs/regularization-v1 completed in **2.003 seconds**, reusing frozen features.
Five-fold training-group CV selected penalty **1000000**, an interior candidate
and 1000 times the original strength. All training/validation identities remained
unchanged. No test outcomes were evaluated.

| Penalty/control | Parent-balanced CV RMSE | Mean within-parent selection lift |
|---|---:|---:|
| Constant | 1.403104 | 0.0000 |
| 1000 (original) | 1.505240 | +0.2836 |
| 10000 | 1.403105 | +0.0779 |
| 100000 | 1.391743 | -0.0174 |
| 1000000 (selected) | 1.390673 | -0.1248 |
| 10000000 | 1.399344 | -0.2182 |

The 10000 score is slightly worse than constant despite rounding to the same
four-decimal number. The selected ridge beats constant CV RMSE by only 0.01243.
Selection followed the prespecified RMSE rule; the unfavorable ranking metrics
were not used to retrospectively choose another penalty.

| Fixed-validation predictor | RMSE | Within-parent Spearman | Mean selection lift |
|---|---:|---:|---:|
| Constant | 1.447310 | undefined | 0.0000 |
| Original ridge (1000) | 1.638994 | +0.2854 | +0.2907 |
| Selected ridge (1000000) | 1.433029 | -0.1539 | -0.3009 |

RMSE and lift are kcal/mol. Lift compares seven highest-predicted candidates per
parent against the mean of that same parent's 65 measured candidates. These are
the ten **validation** proteins, not the prior report's ten test proteins. The
selected model improves validation RMSE by 0.20596 versus original ridge but only
0.01428 versus constant (about 1%). No statistical significance is established.

The intended calibration improvement occurred, but **stable-candidate selection
got worse**. Original ridge has positive lift on 7/10 validation parents; selected
ridge on 5/10, with losses outweighing gains. The negative CV lift for the selected
strength agrees with this concern; it is not just a validation-only observation.

Validation MSE decomposes into parent-mean and centered-variant error:

| Predictor | Parent-mean MSE | Centered-variant MSE |
|---|---:|---:|
| Constant | 1.06048 | 1.03423 |
| Original ridge | 1.72869 | 0.95761 |
| Selected ridge | 1.00913 | 1.04444 |

Selected predictions have standard deviation 0.118 versus 0.580 for original
ridge. Strong shrinkage reduces large parent-level mistakes while losing useful
variant discrimination. Ridge changes relative feature coefficients, not merely
the overall output scale; a positive uniform rescaling alone would preserve ranks.
Centered errors remain a diagnostic using measured parent means, not deployable
oracle corrections.

### Decision and next step

Stronger regularization alone is **not a satisfactory fix for the reward**.
Retain the selected model as the RMSE-winning diagnostic, but do not adopt it as
the generation reward merely because of that win. This exposes a mismatch between
absolute prediction-error selection and our goal of selecting stable sequences.
No new GFlowNet training is warranted by this result alone.

The next proposed study is the separately requested increase in protein coverage.
Use original ridge as a fixed, ranking-useful development comparator, not as a
validated broad-generation reward. Before fitting, specify nested training groups,
unchanged evaluation identities, variants per protein, and penalty scaling with
training-set size so increasing data does not silently change regularization per
example. Track both prediction error and measured selection lift; do not retune
strengths and change coverage simultaneously. A coverage benefit remains a
hypothesis. The current validation/test panels are already inspected development
data, so success here would still need subsequent independent assessment.

Verification: source hashes and split rows match the original; all saved OOF
metrics and the selected penalty reconstruct exactly, and fold constants use
only each fold's training labels. Validation metrics reconstruct exactly from
saved predictions. CPU checkpoint replay differs by at most 1.44e-6 kcal/mol.
[Machine-readable verification](../verification/regularization-results-2026-11-10.json).
No real fitting, backbone execution, GPU use or new test analysis was performed
by the assistant; original run artifacts remain unchanged.
