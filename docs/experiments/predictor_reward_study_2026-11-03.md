# Local stability predictor and reward study

**Status: completed user-run experiment; analyzed 3 November 2026.**
Run: `runs/predictor-reward-v1`. The original saved-reward audit below was CPU
analysis; the subsequent comparison was executed by the user on CUDA.

## Question and motivation

The [first pilot](biological_first_results_2026-10-15.md) fit its specified target
accurately but did not improve measured stability. Tom requested a focused
linear baseline and reward investigation. We therefore separate predictor error
from GFlowNet optimization before trying another training objective.

The scientific basis remains the original Mega-scale dG_ML task and frozen ProGen
features; see [data contract](../data.md), [specification](../../spec.md) and
[original pilot settings](../hyperparameters.md). This protocol adds engineering
choices, not claims that the literature recommends these particular budgets.

## Completed reward audit

Command: `CUDA_VISIBLE_DEVICES='' scripts/run.sh audit-reward --output runs/reward-audit-v1`.
Numerical evidence and input hashes: `runs/reward-audit-v1/audit.json`.

- Label: absolute dG_ML in kcal/mol, higher means more stable.
- Original score: (predicted variant dG − predicted reference dG) / training-label
  population standard deviation. Scale = 1.177569; predicted reference = 4.105916.
- Beta = 1. Reconstructing scores differs by at most 2.17e-7 (floating-point
  rounding). No arithmetic mismatch was found; this does not prove predictor validity.
- Exact expected measured dG: baseline 4.280046, predicted-reward target 4.269933.
  Probability of measured dG at least reference: 0.839459 → 0.810609.
- G2I is predicted at 4.266718 but measured at 3.830234. Its probability grows
  from 0.104312 to 0.138636. V1G is predicted at 3.691396 but measured at 4.328701;
  its probability falls from 0.385167 to 0.314059. These are consequential local
  ranking errors despite overall Spearman 0.8285.
- Baseline top-five mass is 0.9142. Global regression accuracy alone therefore
  cannot characterize error in the region where the prior puts most probability.

Reference subtraction is a constant shift and cancels in the normalized linear
exponential target. Multiplying every terminal reward by a constant also cancels.
Changing score scale changes effective beta; it is not merely a numerical fix.
Negative scores are allowed because exponentiating them yields positive rewards.

## Fixed protocol before fitting

Use the original 950/204/204 training/validation/evaluation partitions, with the
same pooled frozen ProGen2-medium features for every predictor. Extract features
once and save them. No prefix-cache rebuilding, GFN fitting, or backbone updates.

| Choice | Setting and rationale |
|---|---|
| Training sizes | 40, 190, 950 total labeled sequences. Small-data probe, 20% of original training pool, full original budget. Each includes the reference, hence 39/189/949 mutants. |
| Subset seeds | 0, 1, 2; nested seeded subsets within the original training pool. Identical subsets across predictor families. Full-size ridge repeats are identical controls, not independent evidence. |
| Held-out data | Keep all 204 validation and 204 evaluation sequences fixed. Tom's suggested ~40 examples motivates the small condition; we do **not** repartition the 204 evaluation variants. |
| Constant | Mean of the selected training labels. Detect trivial prediction improvements; its reward recovers the prior. |
| Linear | Ridge on the same standardized features. Penalties 0.1, 1, 10, 100; choose by validation RMSE. |
| Neural | Original 128-unit SiLU MLP; 500 Adam updates, batch 64, lr 0.001, weight decay 0.0001, gradient clip 1; select checkpoint every 25 updates by validation RMSE. |
| Normalization | Feature means/scales and label mean/scale from the selected training subset only; population std with 1e-6 floor, matching pilot. |
| Feature extraction | Original float32 checkpoint, batch 8; checkpoint hashes must match pilot. Inherited mean pooling excludes direction marker. |
| Top selection | Best 20 distinct evaluation identities by prediction (~10%); fixed before fitting, stable sequence-order ties. Constant-model top-20 is an arbitrary tie control. |
| Reward strengths | Beta 0, 0.5, 1, 2: no guidance, half, original, double. Small diagnostic grid, not optimized against evaluation labels. |
| Reward utility | Original linear improvement and capped improvement (defined below). No new arbitrary biological threshold. |

The small-data conditions still have 204 validation labels for hyperparameter
selection: they are **not** experiments with only 40 total accessible labels.
Models are fit separately rather than discarding the losing family. All models
are reported even if the old scorer gate fails. The 500-step MLP budget is a
controlled inherited baseline, not a claim of optimal neural fitting. Standalone
MLP initialization does not consume the ridge initializations' random draws, so
it need not reproduce the original pilot checkpoint exactly.

## Reward definitions and interpretation

For each predictor and training subset:

\[
 u(x)=\frac{\widehat g(x)-\widehat g(x_{ref})}{a},\qquad
 u_{cap}(x)=\min(u(x),0),\qquad
 p_\beta(x)=\frac{p_0(x)\exp(\beta u(x))}{\sum_y p_0(y)\exp(\beta u(y))}.
\]

- x and y: candidate protein sequences within the same 204-variant support.
- x_ref: the original reference sequence, present in training.
- g-hat: the fitted predictor's absolute stability estimate in kcal/mol.
- a: training-subset label standard deviation, shared across predictor families
  within the same subset.
- u: dimensionless predicted improvement; u_cap: same improvement with no extra
  reward above predicted reference stability. Substitute u_cap for u for capped runs.
- p_0: original locally constrained ProGen probability on this candidate support.
- beta: nonnegative guidance strength; p_beta: normalized exact target probability.

Capped reward tests “reference-level is sufficient,” not a calibrated probability
of success. Its threshold is the model's predicted reference dG. Evaluation uses
the measured reference dG (4.179632), already available in training. Prediction
miscalibration can still hurt capped rewards. The hypothesis is diagnostic, not
an assumed improvement. Across training sizes, changing a changes the effective
strength in kcal/mol; report a and interpret these curves accordingly.

## Outputs and decision rule

Report each seed/size/model separately: validation and evaluation MAE, RMSE,
Spearman, measured top-20 mean and reference-level success fraction. For every
reward/beta pair, compute exact expected predicted and measured dG, reference-level
success probability, entropy-effective support, top-five probability mass and
expected number of unique sequences in 256 draws. Preserve predictions, subset
identities, scorer checkpoints, candidate selection histories, feature cache,
source snapshot, configuration, protocol and input/checkpoint hashes.

These are paired development diagnostics on **previously inspected** outcomes.
They do not establish broad generation, unseen-family generalization or fresh
confirmation; three seeds do not create three independent evaluation cohorts.
Do not select a “winning” reward from this table and call it confirmed. Review
whether a predictor improves both ranking and exact target utility; if none does,
continue predictor/data investigation. If a useful target emerges, define a new
validation/confirmation protocol before spending on another GFN fit. Keep TB
unchanged for now because target matching was already accurate.

## Execution and checks

User command is in the [runbook](../runbook.md#predictor-and-reward-study).
Only synthetic CPU checks are run during implementation: reward normalization,
beta-zero recovery, constant-shift/scale invariants, capped behavior, separate
family fitting, nested subsets and evaluation-label isolation. Real feature
extraction and training remain user-run.


## Results — 3 November 2026

All 27 predictor conditions and 216 exact target evaluations completed. Original
input hashes and the saved feature-cache hash match provenance. Independent CPU
reconstruction from saved predictions reproduces all expected measured stability
values within 4.45e-15. No fitting or GPU work was performed for this analysis.
Aggregate values, validation choices and influential variants are retained in
[analysis evidence](../verification/predictor-reward-results-2026-11-03.json).
Original outputs and source/protocol snapshots remain unchanged.

### Prediction improves with data; linear is competitive

Means across three seeds; RMSE in kcal/mol, lower is better. Spearman measures
ranking, higher is better. Full-data ridge repeats share the same training data
and are effectively the same fitted model, not independent replications.

| Training examples | Ridge RMSE | MLP RMSE | Ridge Spearman | MLP Spearman |
|---|---:|---:|---:|---:|
| 40 | 1.001 | 0.982 | 0.433 | 0.444 |
| 190 | 0.738 | 0.712 | 0.640 | 0.654 |
| 950 | 0.524 | 0.529 | 0.799 | 0.814 |

The mean control has RMSE about 1.16. Forty training examples are insufficient
for strong overall predictions here, and their target gains change sign across
subsets. A positive target result from one small subset is not evidence of a
reliable predictor. At 950 examples, neural complexity provides no clear RMSE
advantage; its modest ranking advantage does not guarantee better design utility.

### Exact reward targets at the original beta = 1

These are expectations over 204 existing measured variants, **not newly trained
GFN results or laboratory measurements of generated proteins**. Values for fitted
predictors are seed means. Success means measured dG at least the reference's
4.179632 kcal/mol.

| Distribution | Expected measured dG | Change from ProGen | Success probability |
|---|---:|---:|---:|
| Constrained ProGen | 4.28005 | — | 83.95% |
| Full-data ridge, original linear utility | 4.31965 | +0.03960 | 87.09% |
| Full-data MLP, original linear utility | 4.26144 | −0.01861 | 78.87% |
| Full-data ridge, capped utility | 4.29214 | +0.01210 | 83.87% |
| Full-data MLP, capped utility | 4.28555 | +0.00550 | 82.97% |

Original linear utility means exponential reward of the unbounded scaled
prediction, as defined above; it is distinct from choosing a linear predictor.
All full-data MLP seeds lose measured utility with the original reward. Capping
recovers a small mean-dG gain but still lowers reference-level success. It is
therefore not a general fix for the intended stable-enough objective.

Ridge linear utility gains +0.02033 / +0.03960 / +0.07343 kcal/mol at beta
0.5 / 1 / 2. At beta 2, effective support drops to 4.10 from ProGen's 4.93;
stronger guidance has a concentration cost. Do not select beta 2 simply because
it has the largest observed gain on this inspected set.

### Why validation RMSE missed the design problem

Full-data ridge selects penalty 10 by validation RMSE. The three MLPs achieve
validation RMSE 0.476 / 0.447 / 0.424, each better than ridge's 0.486. The original
RMSE-based family selection was consequently defensible for regression, but
regression quality and prior-weighted target utility are different criteria.

The prior assigns 91.4% of its probability to five variants. For example:

| Variant | Prior probability | Measured dG | Ridge prediction, seed 0 | MLP prediction, seed 0 |
|---|---:|---:|---:|---:|
| V1G | 0.3852 | 4.329 | 3.638 | 3.544 |
| V1R | 0.3392 | 4.438 | 4.684 | 4.183 |
| G2I | 0.1043 | 3.830 | 3.904 | 4.816 |
| D3G | 0.0438 | 4.419 | 4.542 | 3.157 |
| D3E | 0.0417 | 3.808 | 4.307 | 3.484 |

Ridge better orders several consequential variants, but still has important
errors (including V1G and D3E). This is a local explanation, not proof that ridge
is generally superior or that the reward problem is solved. Full-data top-20
measured means are 4.315 for ridge and 4.288 for MLP; this uniform top-20 selection
metric differs from a probability-weighted target expectation.

### Next decision

Carry full-data ridge and the original beta 1 forward as the **provisional
baseline**, retaining MLP as a comparator. This choice is informed by inspected
outcomes and must be labeled development. Before another GFN fit, write a small
fixed protocol for additional within-parent train/validation/evaluation splits,
with the same budget, ridge grid, MLP settings and beta 1. Report global errors,
prior-weighted errors and exact target utility on each split; select predictor
hyperparameters only on validation. This tests whether the gain depends on this
particular candidate support and split. It is robustness analysis, not fresh
confirmation, since all these variants were previously accessible.

If useful targets persist, a controlled ridge-reward GFN run can test whether the
sampler realizes that gain. There is currently no evidence to switch away from
TB, expand to broad generation or resume retired pipelines. No follow-up run
has been implemented or launched as part of this results analysis.
