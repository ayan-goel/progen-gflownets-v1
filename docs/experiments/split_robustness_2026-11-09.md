# Local split robustness: is the ridge target advantage repeatable?

**Status: completed by the user on 9 November 2026; all three splits verified.**

## Question and motivation

The [ridge GFN](ridge_gfn_2026-11-09.md) realized a modest stability gain on one
fixed evaluation set. The [noise study](reward_noise_2026-11-09.md) showed accurate
fitting even when corrupted rewards erased that gain. The immediate question is
therefore reward generalization: does ridge still define a useful target when
*different sequence variants* are held out?

These three resplits reuse the same 1,358 measured sequences from 1O6X. They are
not new proteins, new assays or independent confirmation. All outcomes have been
accessible in earlier training, validation or evaluation. The split design and
settings below are fixed before the new predictor fits; no evaluation outcomes
are used to select the partitions or a preferred run.

## Fixed design

| Setting | Choice and reason |
|---|---|
| Split seeds | 43, 44, 45; next three seeds after the original 42, chosen without outcome screening. |
| Partition rule | Original rule: sort unique mutant identities, NumPy default_rng(seed) permutation, floor(70%) training, through floor(85%) validation, remainder evaluation; add reference to training. |
| Counts | 950 training (reference + 949 mutants), 204 validation, 204 evaluation per split. Reference always in training. |
| Predictors | Constant training mean, ridge, MLP; identical features, identities and validation rules within each split. |
| Predictor seed | Fix seed 0 across splits. This isolates partition variation; it does not characterize MLP initialization variability. |
| Features | Reuse the original frozen ProGen pooled full-sequence feature cache for all 1,358 identities. No supervised fitting was involved in constructing these features. |
| Normalization | Recompute feature and label moments using each new training partition only; same floors and population-standard-deviation conventions as original. |
| Ridge | Original penalty grid 0.1, 1, 10, 100; select separately by validation RMSE. Do not fix 10 because it won on the original split. |
| MLP | Inherited 128-unit SiLU network, 500 Adam updates, batch 64, lr 0.001, weight decay 0.0001, gradient clipping 1; validate every 25 updates, select minimum validation RMSE. |
| Reward | Original uncapped dimensionless predicted improvement, beta 1. Beta 0 is an identity check, not a sweep. No noise or new GFN fitting. |
| Top selection | Top 20 predicted evaluation identities (not selected sampling slots), inherited diagnostic. |
| ProGen | Same frozen checkpoint and tokenizer, float32, batch 8 for prefix calculations; no weight updates. |

The split identities depend only on sequences and random seed, not measured dG.
The original seed-42 membership must reconstruct from the combined cache before
new experiments can run. All rows/features are indexed together. Raw data,
original partitions, trained predictors and completed run outputs remain unchanged.
New split files are saved inside the new output directory before fitting begins.

## Each split needs its own baseline

The original 204-candidate prior cannot be reused when the candidate set changes.
For each new evaluation set, construct its legal prefix tree and compute frozen
ProGen logits using the existing cache_tree implementation. Normalize over legal
next residues, exactly as in the original pilot, and multiply conditional
probabilities to obtain each sequence probability. Save the logits, ordered
candidate identities and sequence log probabilities. Forced transitions still
have probability one. No original 4.28005 baseline is transplanted to new support.

The comparison remains a finite, known-candidate local task. Because different
splits change both predictor data and permitted candidate support, interpret
results as robustness of the *whole local protocol*, not a pure causal estimate
of changing training data alone. Additional splits overlap each other and the
original evaluation set; they are not independent biological replicates.

## Metrics and interpretation

For every split and predictor, report global MAE/RMSE/Spearman, ProGen-probability-
weighted MAE/RMSE, measured top-20 selection quality, and exact target utility.
Report expected **measured** stability, reference-level success, effective support,
top-five mass and expected unique counts in 256 draws. Keep predicted expectations
explicitly distinguished from measured ones.

Primary comparison: each predictor's beta-1 target minus **that split's own
ProGen baseline** in expected measured dG. Compare ridge and MLP gains within
split. Present every split, mean/range and the number with positive gain; do not
select the best split or claim statistical confirmation from three overlapping
partitions. Beta-zero and constant-predictor targets should recover the prior.

The reference label is available in every training partition. Validation chooses
ridge penalty and MLP checkpoint; evaluation labels enter only metrics, not
fitting, normalization or selection. No gate suppresses a losing predictor.
Exact enumeration makes new GFN training unnecessary for this question.

A consistent positive ridge gain would motivate a subsequent controlled move
beyond the fixed list; it would not establish accuracy on mutation combinations
or novel folds. Mixed/negative gains would motivate further reward investigation
before expanding generation. Partial unfreezing remains a separate architectural
question, not a repair for incorrect scores.

## Execution and artifacts

After selecting CUDA_VISIBLE_DEVICES yourself and activating .venv:

```bash
scripts/run.sh split-study --output runs/split-robustness-v1 --device cuda
```

This loads frozen ProGen to compute three new priors, fits three ridge and three
MLP models plus constants, and evaluates targets automatically. It does not train
any GFlowNet or extract full-sequence features again. The completed run took 200.97 seconds (3 min 21 s).

The top-level report.md summarizes gains and prior-weighted errors. Per-split
artifacts contain partition identities/labels, prior tensors, predictor
checkpoints, validation histories, predictions, target metrics and individual
reports. Top-level config, source/protocol snapshots and hashes preserve provenance.
An existing output directory is rejected; failed outputs are retained.

Read-only CPU preflight:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh split-study --check-only
```

Sources: the completed local predictor/reward, ridge-GFN and noise studies;
original split construction in src/stability_gfn/data.py and the
[data contract](../data.md). Seeds and size of this diagnostic are engineering
choices. No new literature claims or external datasets are introduced.


## Completed results and interpretation

Run: runs/split-robustness-v1. All nine predictor conditions and their beta-zero
and beta-one targets completed. Values below are exact expected **measured**
stability in kcal/mol, using each split's own candidate set and ProGen baseline.
No GFlowNet was trained in this study.

| Split | ProGen | Ridge target | Ridge gain | MLP target | MLP gain |
|---|---:|---:|---:|---:|---:|
| 43 | 4.04954 | 4.05998 | +0.01044 | 4.07712 | +0.02758 |
| 44 | 4.11012 | 4.11073 | +0.00061 | 4.11072 | +0.00060 |
| 45 | 4.29341 | 4.32774 | +0.03433 | 4.33771 | +0.04430 |

Both predictors improve the mean on all three splits, although split 44's gain
is negligible. Mean gains are +0.01512 for ridge and +0.02416 for MLP. Constants
recover the baseline exactly. Ridge is therefore a useful local baseline, but
its superiority on the original split **does not generalize** to these resplits:
MLP is better on two and essentially tied on the third. This does not establish
MLP as a universal winner either. Keep both comparisons; do not choose a model
or hyperparameter by these inspected evaluation outcomes.

MLP also has lower global evaluation RMSE on all three splits: 0.5372 / 0.4577 /
0.4304 versus ridge 0.5674 / 0.5303 / 0.4717. Prior-weighted RMSE favors ridge on
43 and MLP on 44/45. Neither error statistic alone completely orders target
utility. Different objectives also disagree: on 43 and 45, ridge has greater
reference-level success than MLP despite lower mean stability. On 44, ridge
slightly raises the mean but reduces the already tiny reference-level success.

### Post-hoc concentration diagnostic

Inspection of saved probabilities, without new model execution, shows:

| Split | Most probable mutation | Prior probability | Measured dG of that variant |
|---|---|---:|---:|
| 43 | V1L | 59.14% | 4.02959 |
| 44 | V1M | 98.36% | 4.11177 |
| 45 | V1S | 39.85% | 4.28557 |

At beta 1, the split-44 ridge and MLP targets put 98.87% and 98.97% on V1M.
Both barely move the mean; this is not evidence of a GFN training failure.
Variants with measured dG up to 4.80879 exist in that evaluation set, but maximum
available dG is not the target expectation or a requirement of proportional
sampling. Strong concentration and reward/prior tradeoffs matter. This analysis
explains the observed concentration, not a causal attribution solely to masking
or solely to predictor error.

The baseline is the documented locally masked prefix-tree distribution. Changing
which variants are allowed changes its probabilities. Accordingly, differences
between absolute means across splits are not gains or losses against the old
4.28005 baseline. This finite-support protocol still does not establish broad
generation, diversity, or stability of new mutation combinations.

### Next decision

The original ridge preference should remain provisional. The current results
support modest local target utility for both predictor families and remove the
rationale for assuming ridge is generally superior. Repeating GFN fitting on
these lists would primarily retest sampling, which the earlier clean/noisy runs
already established for their respective supports.

Recommended next prerequisite to expanding generation: evaluate both predictors
with entire mutation positions held out from training, rather than randomly
holding out substitutions at otherwise represented positions. Predefine grouped
train/validation/evaluation rules and compare every target with its own prior;
report prior concentration alongside regression and target utility. This tests
one harder local generalization question before proposing new mutation
combinations. It does not by itself test epistasis or unseen protein families.
This is a recommendation, not an implemented or launched experiment. A separate
protocol must fix groups, seeds and selection rules before execution. Keep ProGen
frozen and TB unchanged until a new task shows a fitting limitation. The eventual
move beyond the candidate list also needs an explicit allowed sequence space and
prior convention; do not silently carry over the old tree probabilities.

### Verification

[CPU verification](../verification/split-study-results-2026-11-09.json) checks
pinned source inputs, reproduces all split identities, reconstructs every saved
prior from masked logits and reconstructs all 18 expected measured target values
exactly. All six saved scorer checkpoints reproduce saved predictions within
2.39e-6 kcal/mol using cached features. Prior probability sums differ from one
by at most 6.68e-8. Original and completed run artifacts were not modified; no
training or GPU work was performed during analysis.
