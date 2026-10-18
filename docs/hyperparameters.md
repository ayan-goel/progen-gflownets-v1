# Hyperparameters and numerical choices

[exact.json](../configs/exact.json) and [biological.json](../configs/biological.json)
are the executable settings. These values define the historical initial pilots. Subsequent changes and
real-run outcomes are recorded in dated sections below. Changes require a new configuration/run and
an entry in [decisions.md](decisions.md). Full upstream model settings are preserved
in each run's `backbone_config.json`.

## Shared guidance choices

| Setting | Initial value | Reason |
|---|---|---|
| ProGen checkpoint | Existing medium checkpoint, frozen | Preserve the working backbone |
| Precision | float32 | Establish correctness before reduced-precision comparisons |
| Temperature / top-k / top-p | 1 / disabled / disabled | Full legal support, consistent probability accounting |
| Head | LayerNorm → concatenate context → Linear(128) → SiLU → Linear(actions) | Small nonlinear correction with shared features |
| Context | position/length, remaining-budget/max(1,budget), one-hot reference residue | Make legal task state explicit |
| Head output initialization | All zeros | Exact baseline recovery |
| Other head initialization | PyTorch Linear defaults; LayerNorm affine ones/zeros, epsilon 1e-5 | Standard reproducible initialization under the recorded seed |
| Learned log Z initialization | 0 | Neutral partition estimate; exact at beta zero |
| Head optimizer | Adam, lr 0.001 | Conservative initial head learning rate |
| Normalizer learning rate | 0.01 | Allow the scalar normalization estimate to adapt faster |
| Adam defaults | betas (0.9, 0.999), epsilon 1e-8, no AMSGrad, weight decay 0 | Explicitly retain standard optimizer behavior |
| Updates / batch size | 2,000 / 64 trajectories | Bounded first pilot; no convergence claim |
| Gradient clip | Global norm 1.0 | Limit unstable updates; non-finite gradients fail |
| Guidance strength | beta 1.0 | Initial unit tilt; later sweeps require validation |
| Seeds | 0, 1, 2 | Engineering replication; not strong statistical evidence |
| Log interval | Every 100 updates, plus first/final | Progress and learning-curve inspection |
| Dropout, replay, exploration mixture | None | Isolate the initial on-policy mechanism |
| Checkpoint selection | Fixed final guidance update | No oracle/test-based choice or hidden early stopping |
| CPU threads | 1 | Bound development checks and reduce small-task overhead |
| Enumeration cap | 20,000 completions | Refuse unexpectedly large finite tasks |

Deterministic algorithms are enabled; CUDA TF32 is disabled. The launcher sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Reproducibility is within the same recorded
software/device/precision environment, not a claim of CPU/GPU bitwise equivalence.
Real GPU numerical checks remain for the user to execute.

## Exact task

Reference `AAAA`, alphabet `AGLS`, budget four, length four: 256 leaves and 85
internal prefixes. Score is `-(number_of_G - 2)^2`, with no learned scale or scorer.
Beta-zero identity is checked; training uses beta one. Feature-extraction batch
size is 16. The real three-seed TV gate is <0.05 per seed; likelihood normalization
tolerance is 1e-5. Cached/full float32 logits/states use absolute/relative tolerance
1e-4. Optional non-float32 checks use 0.08 and are a separate configuration choice.

## Supervised future-reward baseline

Same head width, optimizer rate, update count, batch size and gradient clip as TB.
Use 4,096 baseline rollouts, then estimate log mean exponentiated terminal score
for observed prefix/action pairs. Sum in float64 after subtracting a common maximum
log reward; add it back after taking logs. Regress observed-edge targets with
rollout multiplicities. Unvisited edges receive no direct target. These labels
are predicted reward, not experimental test outcomes. Saved cost records distinguish
rollout score lookups from new scorer execution.

## Biological task and scorer

See [data.md](data.md) for all selection/filter/split choices. Initial biological
feature batch size is 8; task is one reference and a one-substitution budget.
Scorer seed is 0 and the fitted scorer is shared across the three guidance seeds.

| Setting | Initial value | Reason |
|---|---|---|
| Features | Mean final-layer residue states, excluding boundary | Simple fixed sequence representation |
| Input/label standardization | Training mean and population std; floor 1e-6 | No held-out fitting; avoid division by zero |
| Ridge penalty candidates | 0.1, 1, 10, 100 | Small log-spaced validation search |
| Ridge objective | Sum squared standardized errors + penalty × squared weights | Unambiguous penalty convention; zero bias after centering |
| Ridge solver | Dual formulation in float64 | Fewer training rows than feature dimensions in pilot |
| MLP | Linear(128) → SiLU → Linear(1), no dropout | Small nonlinear comparison |
| MLP updates / batch | 500 / 64 | Bounded scorer pilot |
| MLP Adam | lr 0.001, weight decay 0.0001; other defaults above | Mild regularization |
| MLP gradient clip | 1.0 | Numerical control |
| Validation frequency | Every 25 updates and final | 20 candidate MLP checkpoints |
| Model selection | Lowest validation RMSE across ridge/MLP candidates | No test-label selection |
| Validation gate | MAE below training-mean predictor and Spearman ≥0.2 | Engineering screening criterion, not a scientific guarantee |
| Reward | (predicted candidate − predicted reference) / training-label std | Dimensionless stability improvement |
| Reward clipping / uncertainty bonus | None | Fixed initial target without additional transformations |

The gate is selected before real fitting. Failure requires reviewing the scorer/task,
not bypassing the gate or selecting against test outcomes. Scorer selection considers
four ridge candidates and twenty MLP checkpoints; this tuning cost is part of setup.

## Evaluation

- Raw comparison batch: 256 proposals per method.
- Selected batch: 32 slots, retaining duplicates.
- Generate-and-select: 1,024 baseline proposals; GFN/supervised select from 256.
  Also report the baseline's first 32 unranked proposals. Larger baseline pool cost
  is explicit; selected-size matching does not mean compute matching.
- Rank by the same frozen predicted stability; ties keep original proposal order.
- Evaluation RNG seed: 100000 + training seed, restarted per method.
- Evaluation runs on CPU from cached inputs and saved heads, with no ProGen reload.
- Default all-prefix head evaluation chunk: 512 nodes; this only bounds memory.
- Pairwise Hamming includes all unordered pairs of sample slots, including duplicates.
  Novelty uses the explicit scorer-training set, not the unknown PLM pretraining set.
- Stability improvement uses the measured reference for measured values and the
  predicted reference for predicted values. Positive change defines stabilization.
- No coverage means null measured metrics, never a zero biological score.

## CPU fixtures and capacity diagnostic

Tiny backbone: seed 0, two layers, width 64, eight heads, rotary width 8, MLP width
64, context 128, vocabulary 32, float32. Other architecture defaults are upstream
and saved in the model configuration. No pretrained weights are loaded for tests.

CLI smoke overrides: one seed (0), 20 updates, batch 16, head width 32, logging every
10, 32 supervised rollouts; evaluation raw/selected/filter counts 16/4/32 and
feature batch 8. Biological smoke scorer uses width 8, ten updates, batch 8 and
validation every five. Unit tests use smaller hand-checkable spaces and record
explicit fixture choices in their source. They do not tune real hyperparameters.

Privileged capacity diagnostic: default 2,000 updates, Adam lr 0.001, seed 0,
gradient clip 1.0 and the saved run's head size. Fit cross-entropy against exact
conditional targets over branching prefixes. This changes the training information
and is labeled privileged; it is not another fair baseline or a biological result.

## Exact-fit follow-up (14 October 2026)

The original settings above remain unchanged in the experiment configurations.
The failed real runs and all follow-up choices are recorded in
[the diagnostic record](experiments/exact_diagnostics_2026-10-14.md).
`scripts/diagnose_exact.sh` compares two fresh seed-0 privileged fits: Adam 0.001
versus 0.0001, each 10,000 updates, one CPU thread. Everything else matches the
original capacity objective/head. Both fits have now been run by the user; neither was run by the agent.

Diagnostic logging now includes initialization, update 1, every 100 updates and
final. Metrics are post-update, with entropy floor, excess conditional CE, TV,
both KL directions and pre-clipping gradient norm. The fixed-final head is saved;
best logged TV is observational only. Sources and input hashes are saved per run.
The previous diagnostic's cross-entropy history was pre-update; its saved final
head is unchanged. Numerical analysis uses float64 and compares frozen float32
policy outputs; oracle agreement tolerances are checked on synthetic fixtures.

## Longer TB follow-up after the privileged fit passed

[exact-long.json](../configs/exact-long.json) changes only `training.steps` and
`supervised.steps` from 2,000 to 10,000. The original-rate privileged fit reached
TV 0.042336 at 10,000 updates; this motivates a bounded test of the longer budget,
not a guarantee that sampled TB converges. Keep learning rates 0.001 for heads,
0.01 for log Z, batch 64, width 128, seeds 0–2 and all other original values.
Both methods receive the same update count; Monte Carlo baseline rollouts remain
4,096. This is a fresh run with fixed-final checkpoints, not a privileged warm start.

## Late learning-rate experiment (completed by the user)

[exact-decay.json](../configs/exact-decay.json) adds the following to each learner:
`lr_decay.after_fraction = 0.8`, `lr_decay.factor = 0.1` and
`monitor_exact_distribution = true`. All values from the 10,000-update configuration
remain unchanged. Updates 1–8,000 use head/log-Z rates 0.001/0.01; updates
8,001–10,000 use 0.0001/0.001. Supervised head rates follow the same schedule.
The 80% boundary and tenfold reduction test late settling and are not tuned optima.
Adam moments persist. Old configurations have no schedule and no intermediate
exact monitor. Invalid fractions/factors and biological monitoring are rejected.

Record exact distribution metrics at initialization, step 1, every 100 updates,
final and both sides of the boundary: 103 observations per head per seed. Metrics
are post-update; sampled loss/gradient norm remain pre-update. Step-0 loss is null.
Monitoring never selects checkpoints, changes gradients or samples. Monitoring
wall time and all-support score reads are separate fields in the cost report.
The [experiment record](experiments/exact_decay_2026-10-14.md) documents timing
semantics, safeguards, numerical verification and the command.

## Extended low-rate phase after the two-seed pass

The completed scheduled run reached TV 0.053937 / 0.045554 / 0.049461.
[exact-decay-long.json](../configs/exact-decay-long.json) changes only `steps` to
20,000 and `lr_decay.after_fraction` to 0.4 in both learners. The absolute drop
remains after update 8,000; the 0.1 factor and all other settings are unchanged.
Monitor 203 points per learner/seed, with fixed-final selection. Doubling the
budget tests additional fitting at the lower rate; it is not evidence that more
steps will necessarily remove the remaining error. The user runs this experiment.

## Biological pilot after the exact gate passed — 15 October 2026

[biological-pilot.json](../configs/biological-pilot.json) preserves the original
biological scorer, splits, reward and evaluation settings. It changes the required
exact run to `runs/exact-decay-long` and transfers 20,000 updates plus the
0.4-fraction/0.1-factor schedule to both guidance learners. This is an initial
engineering default derived from the exact task, not a validated biological optimum.
Training-time exact monitoring is absent/disabled. Final checkpoints only; no toy
or privileged weights are reused. The pipeline still stops if the scorer gate fails.

## 2026-10-15 — Development target diagnostic

The current executable configuration is [target-diagnostic.json](../configs/target-diagnostic.json).
The [complete numerical-choice table](experiments/target_diagnostic_2026-10-15.md#numerical-choices)
records panel selection/alignment, inner splits, ridge/MLP settings, ensemble seeds,
sequence-prior exponents, sampling budgets and metric conventions. These choices
are frozen before the first diagnostic run; they were not tuned on its outcomes.
No guidance hyperparameter or previous configuration changed. The main added
choices are a 20% inner validation partition (seed 43), five MLP seeds 0–4,
alpha={1,0.5,0.25} at beta=1, and matched 256-proposal/32-slot evaluation.

## 2026-10-16 — Fixed confirmation configuration

[confirmation.json](../configs/confirmation.json) preserves the biological-pilot
head architecture and 20,000-update schedule, with the accepted single MLP seed
zero and original prior/beta=1. Ridge candidates are disabled. Both learners retain
seeds 0–2, batch 64, width 128 and the rate drop after update 8,000. The intentional
evaluation change is equal 256-proposal budgets for all policies, with 32 selected
slots and duplicates retained. The [confirmation protocol](experiments/confirmation_plan_2026-10-16.md#frozen-numerical-choices)
records every value, inherited default, rationale and failure rule. No parameters
were chosen using confirmation outcomes. No new scientific core or optimizer was
introduced.

## 17 October 2026 — Development diversity ablation

[Configuration](../configs/diversity.json), [protocol](experiments/diversity_plan_2026-10-17.md)
and [findings](experiments/diversity_findings_2026-10-17.md).

| Choice | Fixed value and rationale |
|---|---|
| Data/scorer | All three original development supports; saved MLP-0 predictions and inner screens; SHA-256 freezes inputs and model metadata |
| Priors | `original` and `count_corrected`; add log descendant count with coefficient exactly 1 so uniform logits give uniform leaves; derived identity, no tuned exponent |
| Guidance | Same 20,000 updates, width 128, seeds 0/1/2, batch 64, Adam defaults, gradient clip 1 and rate schedule as confirmation; beta=1 for both learners |
| Supervised rollouts | 4,096, inherited; equal update counts do not mean equal compute |
| Proposal budget | 256 per policy/batch, no refill; old 32 repeated slots plus up to 32 distinct predicted-best sequences |
| Ties | First appearance in proposals, independent of outcome labels |
| Sampling repeats | 100 per final policy; engineering choice to describe sampling noise, not biological sample size or a power calculation |
| Evaluation seed | 200000 + 1000 × training seed + repeat index 0–99; fixed before new fitting |
| Trajectory timing seed | 900000 + training seed; one additional 256-trajectory timing batch, not used to choose or report the selection outcome |
| Approximation check | TV <=0.01 per final GFN fit; engineering diagnostic, not stopping rule or biological gate |
| Cache consistency | CPU float64 leaf-law comparison with atol=rtol=2e-4 between new prefix cache and saved causal path logits; fail before head fitting if different |
| Law normalization | Absolute probability sum error <=1e-4; exact metrics use double-precision normalized probabilities |
| Threads / precision | One CPU thread; frozen ProGen float32 with TF32 disabled; unchanged setup choices |
| Smoke overrides | Synthetic inputs only, random tiny backbone, one seed, width 8, four updates/learner, batch 4, log every 2, eight supervised rollouts, eight proposals, four selections, two repeats |

Shared original cache/scorer costs (including the original diagnostic's extra
scorers) are retained as reused source work, not charged as new fitting or silently
deleted. New prefix caching, each learner's fitting, exact-law evaluation and actual
trajectory decoding are reported separately. Uniform and exact-target categorical
sampling exploit enumeration and are not scalable online-generation comparators.
No new GPU experiment was run while selecting these settings.
