# Mutation-position holdouts: harder local reward generalization

**Status: completed by the user on 9 November 2026; all three grouped splits verified.**

## Question and rationale

Can ridge and MLP predict stability and define useful sampling targets for single
substitutions at positions with no mutant labels in their training or validation
sets? The [completed random resplits](split_robustness_2026-11-09.md) found modest
positive gains for both predictors, without consistent ridge superiority. Random
variant splits can expose other substitutions at the same position during fitting.
Holding out entire positions tests a harder, specific generalization question.

This is still developmental analysis of the same previously inspected 1O6X
variants. Held-out means absent from this fit, not historically unseen, absent
from ProGen pretraining, or a new protein family. The reference sequence includes
all wild-type residues and remains available as a training example/reward anchor.
No claim of true out-of-distribution stability or epistasis follows from success.

## Fixed design and hyperparameters

| Setting | Choice and rationale |
|---|---|
| Dataset/features | Same 1,358 sequences and frozen pooled ProGen features, with pinned source/model/tokenizer hashes. Avoid another feature-extraction confound. |
| Group identity | One-based changed residue position relative to the reference. Reject indels, duplicate identities and multi-mutants. Labels do not enter grouping. |
| Partition rule | Sort observed positions; permute with NumPy default_rng(seed). First floor(70%) train, through floor(85%) validation, remainder evaluation. Keep all substitutions in their assigned group. |
| Position counts | 72 observed positions: 50 train, 11 validation, 11 evaluation. Add the reference only to training. |
| Seeds | 43, 44, 45, retained as the same three predefined seeds used in the random-split diagnostic. The partitioning unit changes, so these are not matched candidate sets. No screening by position identity or outcomes. |
| Sequence ordering | Sort identities within each group partition, reference first in training. Ordering is independent of input cache order. |
| Predictor seed | 0 for each fit, matching the prior diagnostic; does not estimate initialization variance. |
| Comparators | Constant training mean, ridge, MLP on identical partitions within each seed. Keep both learned families regardless of observed results. |
| Normalization | Training-only feature/label moments, inherited numerical floors; recompute separately per split. |
| Ridge penalties | 0.1, 1, 10, 100; select by grouped-validation RMSE. No evaluation-based selection. |
| MLP | 128-unit SiLU, 500 Adam updates, batch 64, lr 0.001, weight decay 0.0001, gradient clip 1; validate every 25 updates and retain minimum validation RMSE. Inherited settings, not newly optimized. |
| Reward | Uncapped predicted improvement divided by training-label standard deviation; beta 1. Beta 0 and constant predictor check prior recovery. No caps, noise or beta sweep. |
| Frozen ProGen | Original checkpoint/tokenizer, float32, prefix batch size 8; recompute the locally masked prior on each evaluation candidate set. No backbone updates. |
| GFN budget | None. Exact targets answer this predictor question without another sampler fit. |
| Top selection | Top 20 predicted variants, inherited secondary diagnostic; no sampled best-of selection. |

Identity-only preflight gives the following sizes; unequal group coverage means
sequence counts need not equal the previous 950/204/204. Do not rebalance by moving
individual mutants across groups.

| Seed | Train sequences (including reference) | Validation | Evaluation |
|---|---:|---:|---:|
| 43 | 942 | 207 | 209 |
| 44 | 942 | 209 | 207 |
| 45 | 945 | 204 | 209 |

Full group identities are in [readiness](../verification/position-study-readiness-2026-11-09.json)
and are saved before fitting in each run's mutation_positions.json and
partitions.json. No real predictor outcomes are computed during preflight.

## Evaluation and decision rules

Primary comparison: each beta-1 target's exact expected **measured** dG minus its
own constrained-ProGen baseline (kcal/mol, higher is more stable). Report every
split and family, mean/range gains and count of positive gains. No automatic
pass/fail or winner selection from three overlapping partitions.

Also report global MAE/RMSE/Spearman, prior-weighted MAE/RMSE, predicted versus
measured expectations, reference-level success, top-one/top-five mass, effective
support, expected distinct variants in 256 draws, and top-20 selection quality.
Concentration/success/diversity appear in the main report to expose a nearly
single-variant prior or a mean gain accompanied by reduced success/diversity.

Evaluation labels enter metrics only, never normalization, fitting or model
selection. Grouped validation selects ridge penalty/MLP checkpoint. Frozen pooled
features can be reused because their extraction did not fit stability labels.
Original run outputs and all earlier partitions remain immutable.

Prior and allowed support change when evaluation positions change. Differences
from random-split absolute means do not isolate training-position generalization:
training size, partition identity and candidate support also differ. Evaluate
within-support improvements and prediction errors together. Do not transplant
4.28005 or any other old baseline. Position groups overlap across seeds, and all
three runs share a parent; these are not independent biological replications.

If both predictors fail to define useful targets, investigate reward reliability
before new mutation combinations. If usefulness persists, specify a controlled
expansion with an explicit sequence space, prior and independent assessment.
Success here alone does not validate combinations. A concentrated prior is a
separate modeling consideration, not a reason to change beta after seeing labels.
Keep ProGen frozen and TB unchanged; this experiment does not test unfreezing.

## Run and artifacts

After activating .venv and selecting CUDA_VISIBLE_DEVICES yourself:

```bash
scripts/run.sh position-study --output runs/position-holdout-v1 --device cuda
```

The command computes three new priors, fits three ridge/MLP pairs and constant
controls, then evaluates automatically. See runs/position-holdout-v1/report.md.
Per-split artifacts retain predictions, checkpoints, validation histories, priors,
sequence partitions and position lists. Config/source/protocol/provenance snapshots
record exactly what ran. Existing output directories are rejected; failures keep
partial artifacts and do not resume or overwrite automatically.

CPU-only input check (no predictor/backbone execution):

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh position-study --check-only
```

The completed position-held-out run took 194.35 seconds (3 min 14 s).

## Sources and verification scope

This design is motivated by the local [random-split results](split_robustness_2026-11-09.md)
and [noise results](reward_noise_2026-11-09.md), with inherited fitting choices from
[predictor comparison](predictor_reward_study_2026-11-03.md). Grouping and seed count
are engineering choices to address the stated question, not literature-derived
optimal settings. Dataset provenance remains in the [data contract](../data.md).

Synthetic CPU checks verify disjoint mutation groups, full coverage, input-order
invariance, malformed-input rejection and the complete three-split fit/evaluate/
reload path. Real-data checks verify hashes and group identities only. No real
training, inference or GPU work is performed by the assistant.


## Completed results

Run: runs/position-holdout-v1. All nine predictor conditions and 18 exact target
calculations completed. No GFlowNet was fitted. These are exact probability-
weighted expectations of previously measured dG, in kcal/mol, not new assays.

| Position split | ProGen | Ridge target | Ridge gain | MLP target | MLP gain |
|---|---:|---:|---:|---:|---:|
| 43 | 2.97034 | 3.75608 | +0.78574 | 3.60544 | +0.63510 |
| 44 | 3.94440 | 4.32403 | +0.37963 | 4.24025 | +0.29585 |
| 45 | 1.81102 | 3.17736 | +1.36634 | 2.13570 | +0.32468 |

Both predictors define positive-gain targets on all three grouped splits. Ridge
has the larger gain on each, with descriptive mean +0.84390 versus MLP +0.41854.
Constants recover their priors exactly. This supports useful reward information
at positions whose mutant labels were absent from the current fit; it does not
establish accurate absolute predictions, combination generalization or a universal
ridge advantage. These outcomes are inspected development evidence.

### Prediction quality and selection

| Split | Ridge evaluation RMSE / Spearman | MLP evaluation RMSE / Spearman |
|---|---|---|
| 43 | 0.7812 / 0.6539 | 0.6995 / 0.6882 |
| 44 | 0.7971 / 0.6627 | 0.8170 / 0.6088 |
| 45 | 0.7565 / 0.5947 | 0.9951 / 0.4810 |

RMSEs exceed those in the random-variant experiment, although changed candidate
sets prevent isolating grouping as the sole cause. Ridge's larger target gain on
43 despite worse global error again separates regression accuracy from design
utility. Prior-weighted RMSEs are ridge 0.6674/0.9299/0.7066 and MLP
0.5823/0.7063/2.0524; weighted error alone also does not order utility perfectly.

Grouped validation selected ridge penalty 100 on every split and MLP updates
250/150/50 respectively. MLP had lower validation RMSE on all three, so choosing
by overall validation RMSE alone would not have selected the highest-utility
family observed here. Do not reinterpret that as permission to select models on
evaluation outcomes. The penalty-grid boundary is recorded, not used to expand
or retune this completed experiment.

### Concentration, success and limitations

Prior top-one probabilities are 20.66%, 13.02% and 10.28%. These supports differ
substantially from the random-split case that put 98.36% on V1M. None of the new
evaluation position sets includes position 1, as fixed by the original seed rule;
no groups were changed after inspection. Changed support and locally masked
probabilities mean that large gains here are **not** a direct improvement over
the old random-split results, evidence of stronger ProGen, or proof that grouping
improves the predictor. Each row is meaningful against its own prior.

| Split | ProGen reference-level success | Ridge success | MLP success | ProGen / ridge / MLP expected unique in 256 draws |
|---|---:|---:|---:|---|
| 43 | 20.89% | 49.66% | 40.28% | 26.24 / 21.34 / 22.69 |
| 44 | 68.34% | 82.34% | 77.61% | 27.27 / 18.80 / 21.99 |
| 45 | 8.77% | 24.82% | 12.97% | 29.85 / 35.36 / 31.54 |

All targets increase reference-level success, but most ridge probability in split
45 remains below the reference's measured 4.17963 kcal/mol. A large gain over a
weak baseline does not mean most outputs meet the desired stability level.
Diversity decreases on 43/44 and increases on 45; better mean utility need not
imply greater diversity. All candidates remain known single mutants of one parent.

### Recommended next experiment

Use these fixed ridge rewards to test GFN training on all three position-held-out
supports, retaining the same beta, frozen ProGen and TB settings. The targets are
useful here and differ materially in concentration from the original GFN task;
check whether trained policies realize their measured gains. Record TV/KL to the
exact target separately from measured stability, success and diversity. Preserve
the exact MLP targets as comparators and report all splits rather than selecting
the largest gain. Choosing ridge for this follow-up is an explicitly post-hoc
development choice, not independent biological confirmation or a claim that MLP
is generally inferior.

No follow-up run has been implemented or launched. Specify its budget and seed
panel before execution. The subsequent move to new mutation combinations still
requires an explicit sequence space, prior convention and independent stability
assessment; the current findings alone cannot validate combination rewards.

### Completed-artifact verification

[CPU verification](../verification/position-study-results-2026-11-09.json) reproduces
all partitions and confirms disjoint position groups, checks pinned inputs,
reconstructs priors exactly from saved masked logits, and reconstructs all 18
predicted/measured expectations and reference-success values. Six saved scorer
checkpoints reproduce predictions within 2.15e-6 kcal/mol; prior probability
sums differ from one by at most 6.70e-8. No original or completed run files were
modified, and no fitting, backbone execution or GPU work was performed in analysis.
