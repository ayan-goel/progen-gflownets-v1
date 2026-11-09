# Current project status

**3 November 2026 — scope reset to the first biological pilot (slide 7).**

[Specification](../spec.md) · [Pilot result](experiments/biological_first_results_2026-10-15.md)
· [Runbook](runbook.md)

## Retained starting point

- Standalone ProGen2 and TF-Bind-8 work remain in the original test workspace.
- Frozen ProGen2 + trajectory-balance guidance passed the 256-sequence synthetic
  test: final total variation distances 0.04090 / 0.04686 / 0.04472.
- The first biological run is preserved at runs/biological-first.
- Its task is 1O6X.pdb, a 72-residue reference: 950 scorer-training sequences,
  204 validation sequences and 204 held-out single-substitution candidates.

| Distribution | Expected predicted stability | Expected measured stability |
|---|---:|---:|
| Constrained ProGen | 3.89403 | 4.28005 |
| Exact predicted-reward target | 3.97291 | 4.26993 |
| GFlowNet, mean over three seeds | 3.97310 | 4.26975 |

Values are kcal/mol. These are exact probability-weighted expectations over the
finite candidate set, not new laboratory measurements. GFlowNet's close target
match does not make the predictor's ranking correct.

## Completed predictor/reward investigation

The user completed runs/predictor-reward-v1: 27 predictor conditions and 216 exact
target evaluations. [Protocol and results](experiments/predictor_reward_study_2026-11-03.md)
own the detailed comparisons and limitations.

- Original reward arithmetic reconstructs correctly; predictor errors matter.
- More training data improves both predictors. At 950 examples, ridge and MLP
  evaluation RMSE are 0.524 and 0.529 (MLP mean over three seeds).
- At beta 1, ridge's exact target raises measured expected stability from 4.28005
  to 4.31965; MLP targets average 4.26144. These are target calculations, not new
  GFN results.
- MLP has better validation RMSE, but worse errors on some high-probability
  variants. Global regression selection alone misses the design issue.
- Capped MLP targets modestly improve mean dG but reduce reference-level success.

## Completed ridge-reward GFlowNet rerun

The user completed runs/ridge-gfn-v1 in 48 min 6 s. All three final policies
closely match the ridge target and improve expected measured stability over
ProGen on the original 204 candidates:

| Distribution | Expected measured stability (kcal/mol) |
|---|---:|
| ProGen | 4.28005 |
| Historical MLP-reward GFN, mean | 4.26975 |
| Exact ridge target | 4.31965 |
| Ridge-reward GFN, mean | **4.31953** |

Final TVs to ridge target: 0.000412 / 0.000227 / 0.001367. Reference-level
success probability rises from 83.95% to 87.06%; expected distinct variants in
256 draws falls from 10.67 to 9.87. See [protocol and results](experiments/ridge_gfn_2026-11-09.md)
for predicted-versus-measured distinctions, training curves and limitations.

## Next run: fixed reward-noise sensitivity

The [noise protocol](experiments/reward_noise_2026-11-09.md) is fixed and the
[user-run command](runbook.md#fixed-reward-noise-study) is ready. Two score-noise
SDs (0.25/0.50), three independent noise vectors and three training seeds per
vector give 18 new fits. Reuse the clean ridge control. Keep the original
20,000-step TB schedule, frozen ProGen, candidate support and beta 1.

Report noisy-target damage separately from GFN error against that noisy target.
All 33 CPU tests pass, including a synthetic crossed-panel run and evaluation
replay; clean-input preflight passes. No actual noisy-target experiment or GPU
work has been run by the assistant. Estimated user runtime is about five hours
on the same hardware as the clean run. This is fixed score corruption, not noise
resampled on every query or a calibrated model of assay errors.

## Evaluation boundaries

The original labels have already been inspected. This study and future resplits
of these variants are development, not fresh confirmation. Full-data ridge seeds
use identical training data and do not provide independent replication. The
improvement is small, support is narrow, and broad stable-protein generation is
not established. Retired outcomes remain inspected as well.

Verification: original inputs and feature-cache hashes match provenance; CPU
reconstruction reproduces all 216 expected measured dG values within 4.45e-15.
No GPU execution or retraining was performed during analysis. The saved ridge-GFN
checkpoints independently reproduce the final measured expectations on CPU.

## Recovery

Later work is outside the active project in
/usr/scratch/agoel320/progen-gflownets-archive/post-pilot-2026-11-03.
The [decision record](decisions.md#2026-11-03--return-to-the-first-biological-pilot)
explains the reset. That archive is not part of the current workflow.
