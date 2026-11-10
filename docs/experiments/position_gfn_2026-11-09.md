# GFlowNet follow-up on position-held-out ridge rewards

**Status: completed by the user; results verified on 10 November 2026.**

## Question and fixed scope

Does the same frozen-ProGen TB sampler realize the useful ridge targets from all
three [completed position-held-out splits](position_holdout_2026-11-09.md)? Those
exact targets improved measured expectations by 0.78574 / 0.37963 / 1.36634 kcal/mol
over their own priors. They have not yet been learned by a GFlowNet.

This follow-up holds each split's candidate identities, logits, predictions and
reward normalization fixed. Ridge is a post-hoc development choice motivated by
inspected outcomes; no independent confirmation or universal superiority is
claimed. Retain the exact MLP targets as distribution comparators, not trained
MLP-reward GFNs. Do not choose a winning split or tune the reward after fitting.

Changing train/validation/test splits in the preceding study also changed the
allowed sampling candidates because the evaluation set defines the prefix tree.
That changed locally normalized ProGen probabilities. The previous random and
position-held-out experiments therefore did not isolate training-data effects.
Within this follow-up, compare each GFN with the exact target and baseline on
its own unchanged support. A fixed-support predictor comparison would answer a
separate causal question; it is not silently added here.

## Training choices and rationale

| Setting | Fixed choice and reason |
|---|---|
| Source | runs/position-holdout-v1; all three saved position splits 43/44/45. Pin source config, completion, provenance and each fold's partitions/groups, prior, results, ridge checkpoint and ridge/MLP predictions. |
| Candidates | Original 209 / 207 / 209 evaluation identities, in saved order. No new partitions or mutations. |
| Predictor | Saved ridge checkpoint and predictions for each split. No refitting, model selection or changed normalization. |
| Reward | Saved constrained ProGen prior times exp(beta × score), beta 1; score is (ridge prediction − reference prediction) / that fold's training-label standard deviation. |
| Numeric convention | Stored score uses float32, as in existing TB training. Verify its target differs from the float64 study target by TV at most 1e-6; record actual rounding discrepancy. |
| Trainable model | Original residual head with LayerNorm, 128-unit SiLU hidden layer and zero-initialized output correction, plus scalar log Z. ProGen and ridge stay frozen. |
| Training seeds | 0, 1, 2 on each support, inherited from the original GFN panel: nine fits. Do not confuse training seeds with split seeds. |
| Objective | Existing on-policy trajectory balance, no objective or sampler changes. |
| Budget | 20,000 updates per fit, batch 64. Fixed final checkpoint, no early stopping or best-seed selection. |
| Optimizer | Existing Adam, head lr 0.001, log-Z lr 0.01, gradient clip 1. Multiply both rates by 0.1 after update 8,000, preserving Adam state. |
| Monitoring | Initialization, step 1, every 100 updates, transition 8,000/8,001 and final. Exact TV/KL use predicted rewards only; they do not alter updates or select checkpoints. |
| Backbone setup | Recompute frozen prefix features once per support, same checkpoint/tokenizer, float32 and batch 8. Earlier position study saved logits but not prefix features. |
| Prior preservation | Check newly computed logits against saved logits (atol 5e-5, rtol 1e-5), then use saved logits exactly. Abort on mismatch before any head training. |
| Memory | Cache all three supports, release ProGen, train each small head in turn. Backbone features remain detached. |

These settings are inherited from the [ridge-GFN protocol](ridge_gfn_2026-11-09.md)
and [fixed-noise study](reward_noise_2026-11-09.md) to avoid a simultaneous optimizer
change. Nine fits retain training-seed replication on every new support. This is
not a budget optimization experiment. Only synthetic CPU smoke tests shorten
training to four updates, a small head and two seeds.

## Evaluation and boundaries

Automatic exact evaluation after all fits reports per split:

- ProGen baseline, exact ridge target, exact MLP target and each of three GFNs.
- TV and both KL directions to the **ridge** target, probability sum and learned
  log-Z error. MLP-versus-ridge distance is not a measure of MLP fitting failure.
- Expected measured dG, gain over the same-support ProGen baseline and signed gap
  from the exact ridge target.
- Reference-level success, top-one/top-five mass, entropy/effective support and
  expected distinct variants in 256 draws.
- Ridge-predicted and MLP-predicted expectations, explicitly separate from measured
  stability. All methods are scored with both predictors for comparable fields.

The report shows every fit, with mean gain and worst fitting gap per split saved
in summary.json. Exact enumeration removes Monte Carlo noise; it is evaluation
of a trained sampling policy, not a best-of or post-selection procedure. No new
sample-selection baseline is introduced. No threshold-based pass/fail or automatic
next experiment is launched.

Held-out measurements are not included in task.pt or passed to train_tb, monitoring
or checkpoint selection. After every fit finishes, labels are copied for evaluation.
The source outcomes have already been inspected; this code separation is about
training inputs, not a claim that the study is blind. Reference thresholds retain
the source study's float32 label convention. Completed run outputs are immutable.

Small TV plus realized gains would establish sampling on these new finite supports.
Large TV would motivate a fitting investigation. Neither outcome validates new
mutation combinations or protein families. A gain over a weak baseline may still
leave most probability below reference stability; inspect success and diversity.
No ProGen unfreezing, SubTB or reward retuning is included.

## Commands and artifacts

After activating .venv and selecting CUDA_VISIBLE_DEVICES yourself:

```bash
scripts/run.sh position-gfn --output runs/position-gfn-v1 --device cuda
```

Automatic evaluation writes runs/position-gfn-v1/report.md and evaluation/ with
results.json and summary.json. Per-split directories retain task.pt, scorer.pt,
position groups, per-seed checkpoints/history/progress, and post-training labels.
Config/source/protocol/readiness/provenance snapshots and hashes preserve the run;
costs.json separates setup and each fit. Existing output directories are rejected.
Failures preserve completed checkpoints and status; there is no automatic resume.

CPU input verification (no backbone execution or training):

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh position-gfn --check-only
```

CPU reevaluation into a new directory needs only the saved run, not ProGen or its
source study, and verifies hashes of tasks, labels, checkpoints and configuration:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-position-gfn \
  --run runs/position-gfn-v1 --output runs/position-gfn-v1-recheck
```

Completed runtime: 9,029.39 seconds (2 h 30 min 29 s), including setup, all nine
fits and automatic evaluation.

Sources are the completed local grouped-position, ridge-GFN and noise studies
linked above. New code only connects saved rewards/prefix caching to existing TB;
no new literature-derived algorithm or biological claim is introduced.


## Readiness verification

[All 43 CPU tests pass](../verification/position-gfn-cpu-tests-2026-11-09.txt).
The new synthetic path exercises six four-update fits, zero-guidance prior
identity, source immutability, label separation, report replay, changed-input
rejection and mismatch abort before training. [Real-input preflight](../verification/position-gfn-readiness-2026-11-09.json)
confirms pinned sources and ridge predictions to 2.15e-6 kcal/mol. Score-rounding
TV is at most 8.83e-9. Real backbone execution and training remain user-run.


## Completed results — 10 November 2026

Run: runs/position-gfn-v1. All nine fits finished 20,000 updates, with finite
logged losses and 203 monitoring records per fit. Values below are exact expected
**measured** dG in kcal/mol; GFN values average the three final training seeds.
They use existing assay labels, not newly measured generated proteins.

| Split | ProGen | Exact ridge target | Trained GFN mean | GFN gain over ProGen | Maximum final TV |
|---|---:|---:|---:|---:|---:|
| 43 | 2.97034 | 3.75608 | 3.75585 | +0.78550 | 0.000472 |
| 44 | 3.94440 | 4.32403 | 4.32386 | +0.37946 | 0.001972 |
| 45 | 1.81102 | 3.17736 | 3.17742 | +1.36639 | 0.001111 |

Every fit improves measured expectation over its own baseline. Final TV ranges
from 0.00003582 to 0.00197160; largest absolute measured-expectation gap from the
ridge target is 0.00226465 kcal/mol. The largest logged TV during updates
16,000–20,000 is 0.00427998. Small positive gaps above the exact target reflect
finite fitting error: the reward-proportional target is not a maximum-measured-
stability policy. No best seed was selected.

| Split | ProGen reference-level success | GFN mean success | ProGen expected unique / GFN mean unique in 256 draws |
|---|---:|---:|---|
| 43 | 20.89% | 49.65% | 26.24 / 21.34 |
| 44 | 68.34% | 82.33% | 27.27 / 18.77 |
| 45 | 8.77% | 24.82% | 29.85 / 35.34 |

All three supports gain reference-level success. Diversity falls on 43/44 and
rises on 45, following the reward targets. Split 45 still assigns about 75% of
probability below reference stability despite its large gain. A successful sampler
can therefore realize a useful but biologically limited target.

The exact MLP targets remain 3.60544 / 4.24025 / 2.13570, as recorded in the
predictor study. They are mathematical comparators, not trained MLP-reward heads.
Ridge's local design advantage is an inspected development result; it does not
establish superiority across proteins or an independently confirmed selection rule.

### What is established, and what is next

This completes the local chain on the tested supports: fixed stability predictions
from held-out-position evaluation define useful rewards, and the frozen-ProGen
TB guidance reliably realizes their probability distributions and stability gains.
There is no demonstrated local fitting bottleneck motivating a change of objective
or partial ProGen unfreezing. The comparison is within each fixed support; gains
cannot be read as direct improvements over earlier random-split experiments with
different candidate sets and constrained priors.

The user's goal remains generation of new stable proteins, not more local mutation
benchmarks. Recommend moving the next investigation to **stability rewards across
multiple parent proteins**. First audit usable data and prior outcome exposure,
then predefine protein/similarity-group-separated training, validation and evaluation
so related variants cannot straddle partitions. Compare simple predictor baselines
without assuming the local ridge winner transfers, assess calibration/ranking and
stable-enough selection, and specify independent assessment for eventual generated
sequences. Group definitions, dataset scope, budgets and pass criteria still need
a concrete protocol; no new data outcomes have been inspected for this recommendation.

This is a proposed next scope, not an implemented experiment or authorization to
restore retired workflows. Broader generation still needs a clearly specified
sequence space and prior; the present finite candidate-tree prior must not silently
be treated as unrestricted ProGen. Generalization to new families or new sequences
and experimentally verified stability remain unestablished.

### Verification

[Saved-artifact verification](../verification/position-gfn-results-2026-11-10.json)
checks pinned source inputs and evaluation-input hashes. Independent CPU replay
from saved tasks and all nine heads reproduces results.json, summary.json and the
report exactly. Training histories confirm all scheduled final steps and finite
logged losses. Original and completed run artifacts remain unchanged. No training,
backbone execution or GPU work was performed during this analysis.
