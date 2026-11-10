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

## Completed fixed reward-noise study

The user completed all 18 fits in 5 h 2 min. Every GFN closely matches its noisy
target (maximum final TV 0.000504). Noise changes target biological utility:
trained means range 4.29858–4.33340 at sigma 0.25 and 4.25404–4.34721 at sigma
0.50. One higher-noise realization falls below ProGen (4.28005), despite accurate
fitting. Some perturbations help by chance; three noise draws do not establish
noise tolerance or a useful noise-injection method.

See [protocol and completed results](experiments/reward_noise_2026-11-09.md).
Pinned inputs match and full CPU evaluation replay reproduces saved results.
The assistant ran no GPU work or training during analysis.

## Completed local split robustness

All three predefined splits completed in 3 min 21 s. Ridge and MLP targets both
improve expected measured stability over their own ProGen baseline on all three
splits. Mean gains are +0.01512 and +0.02416 kcal/mol respectively. MLP is better
on two splits and essentially tied on the third: the original ridge advantage
is not a general model-ranking result. No new GFlowNet was trained.

Split 44 barely improves (+0.0006 for either model); its prior puts 98.36% on one
variant, and both targets remain similarly concentrated. Absolute baseline values
vary with candidate support and must not be compared to the old 4.28005 baseline.
See the [completed protocol/results](experiments/split_robustness_2026-11-09.md)
for all splits, secondary metrics and limitations. CPU checkpoint/prior/target
reconstruction passed; completed run outputs remain unchanged.

## Completed mutation-position holdout study

All three grouped splits completed in 3 min 14 s. Both predictors improve target
utility on substitutions at positions absent from their training/validation
mutant sets. Ridge target gains are +0.78574 / +0.37963 / +1.36634 kcal/mol;
MLP gains are +0.63510 / +0.29585 / +0.32468. No GFN was trained.

These gains use substantially different candidate supports and weaker baseline
means (2.97034 / 3.94440 / 1.81102); they are not a direct improvement over the
random-split experiment. Prediction errors remain substantial. On split 45, even
the ridge target has only 24.82% probability of reaching reference stability.
See the [completed record](experiments/position_holdout_2026-11-09.md) for all
metrics, selection details and CPU artifact verification.

## Completed GFN follow-up — verified 10 November 2026

All nine fits completed in 2 h 30 min 29 s. Each closely matches its fixed ridge
target and realizes its stability gain. Across splits 43/44/45, expected measured
stability averages 3.75585 / 4.32386 / 3.17742, versus own-support ProGen baselines
2.97034 / 3.94440 / 1.81102. Maximum final TV is 0.001972 and maximum absolute
measured-target gap is 0.002265 kcal/mol. CPU replay reproduces saved evaluation
exactly; the [result record](experiments/position_gfn_2026-11-09.md#completed-results--10-november-2026)
owns full metrics and limitations.

The local sampler check is complete. Diversity follows the target rather than
uniformly increasing, and split 45 still reaches reference stability only about
25% of the time. These are finite, known-candidate results on one parent protein,
not broad generation or new biological measurements.

## Ready: multi-protein stability predictor benchmark

The [fixed protocol](experiments/multiprotein_2026-11-10.md) and
[user command](runbook.md#multi-protein-stability-predictors) are ready. CPU data
preparation retained 414 eligible parents, grouped them into 115 components, and
excluded 9 components matching known historical identities. From 106 remaining
components, a fixed identity-only rule selects 64 parents, each with 65 sequences:
44 training groups, 10 validation and 10 evaluation (2,860 / 650 / 650 sequences).
Maximum cross-partition parent-reference identity is 31.94% under the documented
alignment. This heuristic is not a guarantee of unseen biological families.

The user GPU command extracts frozen features and compares constant, ridge and
three MLP seeds. Evaluation reports pooled, per-parent and reference-only errors
and measured top-score selection lift over uniform candidates. There is no GFN
sampling prior, new guidance fitting or generation in this benchmark. All models
share fixed evaluation identities. Test labels remain unopened by fitting code;
CPU preparation used them only for validity, replicate averaging and copying.

All 47 CPU tests pass, including synthetic end-to-end execution and equivalent
primal/dual ridge solving. Real-input preflight passes. No real predictor fitting,
backbone execution or GPU work was run by the assistant. Historical exposure
inventory is incomplete by nature; this remains developmental evidence. Broad
generation still needs an explicit sequence space and independent assessment.

## Evaluation boundaries

The original labels have already been inspected. This study and future resplits
of these variants are development, not fresh confirmation. Full-data ridge seeds
use identical training data and do not provide independent replication. The
gains depend on support, which remains narrow; broad stable-protein generation is
not established. Retired outcomes remain inspected as well.

Verification: original inputs and feature-cache hashes match provenance; CPU
reconstruction reproduces all 216 expected measured dG values within 4.45e-15.
No GPU execution or retraining was performed during analysis. The saved ridge-GFN
checkpoints independently reproduce the final measured expectations on CPU.

## Retired work

The retired project archive was deleted at the user's request on 10 November
2026. Its files are no longer available for recovery from that location.
The [decision record](decisions.md#2026-11-03--return-to-the-first-biological-pilot)
explains the reset. Active experiment artifacts and the frozen identity inventory
remain intact; the prepared multi-protein benchmark does not depend on the archive.
