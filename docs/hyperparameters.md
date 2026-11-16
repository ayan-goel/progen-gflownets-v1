# Hyperparameters and numerical choices

## Replay TB and supervised pilot

[The protocol](experiments/idr_replay_2026-11-13.md#objective-architecture-and-numerical-settings)
and `configs/idr-replay.json` own the settings: seed 0, 800 updates, batch 16,
inherited rates/decay/clipping, 50% fresh / 25% stored success / 25% stored near-miss
for replay, and 128 final draws per prompt/arm. Its fixed bank uses only historical
training rows with a tail-grouped diagnostic holdout. The supervised objective is
mean token NLL on successful examples, with the same head rates and no log Z.
The protocol records every new RNG, bank limit and progression threshold.

## Joint-acceptance reward comparison — completed

The [reward record](experiments/idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1)
owns the complete formulation and rationale: inherited charge tolerances and V3
screen, disorder-deficit scale 0.2, pass/near-failure ratio 20, failure floor fraction
0.25, and `beta=1` with `S=log(property factor)`. These are declared engineering
defaults, not optimized values. The audit uses 2,000 paired within-group bootstrap
resamples with seed 20261113. The [training protocol](experiments/idr_acceptance_2026-11-13.md#fixed-settings-and-rationale)
inherits 800 updates, seeds 0/1/2, batch eight, rates and clipping from completed TB.
It fixes fresh prior/evaluation streams and reuses old heads without refitting.

## Historical experiment settings

All values below belong to existing stability/synthetic experiments. The IDR baseline
settings and rationale live in its [protocol](experiments/idr_baseline_2026-11-12.md)
and `configs/idr-baseline.json`. The new [charge-pilot protocol](experiments/idr_charge_2026-11-12.md)
and `configs/idr-charge.json` own charge reward scales, training and comparison budgets.
The [real-state diagnostic protocol](experiments/idr_learning_diagnostic_2026-11-12.md)
owns its fixed capture and gradient-averaging budget; it inherits the pilot reward.
The [direct-context comparison](experiments/idr_context_2026-11-12.md) adds only a
zero-initialized 20-by-5 bias-free matrix under the same head optimizer group.
`configs/idr-context.json` pins the completed pilot configuration; all reward,
training and evaluation settings are inherited. Six fresh fits share one prior
pool; no batch, rate or loss tuning is bundled with the architecture test.
The subsequent [bounded branch-rate protocol](experiments/idr_branch_rate_2026-11-12.md)
sets only the added matrix rate to 0.001 (0.0001 after step 300), retaining old-head
0.0001/0.00001 and log-normalizer 0.01/0.001. `configs/idr-branch-rate.json` pins
control evidence and multiplier ten. No other setting is tuned; a negative/mixed
result ends this branch-specific follow-up.

The [matched TB/SubTB protocol](experiments/idr_subtb_2026-11-13.md#fixed-settings-and-rationale)
and `configs/idr-subtb.json` own the current settings: 800 steps per fit, three
seeds, unchanged policy/branch/normalizer rates, lambda 1, separate 128-unit flow
MLP at rate 0.001, common decay after step 600 and 64 evaluation draws per prompt.
These are bounded engineering choices, not paper-optimal values. Both methods
are trained afresh; only SubTB has the auxiliary flow network.

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

## Predictor/reward diagnostic study

The [fixed study protocol](experiments/predictor_reward_study_2026-11-03.md) owns the new training sizes, seeds, beta grid and capped-utility rationale. Predictor optimizer settings are inherited from the original pilot; its artifacts are unchanged.

## Ridge-reward GFN rerun

The [9 November protocol](experiments/ridge_gfn_2026-11-09.md) inherits the original TB optimizer, architecture, schedule and beta 1. Only the predictor changes; observational exact-target monitoring is enabled.

## Fixed reward-noise study

The [noise protocol](experiments/reward_noise_2026-11-09.md) specifies sigma 0.25/0.50, noise seeds 100/101/102 and training seeds 0/1/2. Original TB settings and beta 1 are retained; the clean baseline is reused.

## Local split robustness

The [three-split protocol](experiments/split_robustness_2026-11-09.md) fixes split seeds 43/44/45, predictor seed 0, original partition sizes and validation grids, and beta 1. No new GFN budget is introduced.


## Mutation-position holdout

The [grouped-position protocol](experiments/position_holdout_2026-11-09.md) retains
seeds 43/44/45, predictor seed 0, original scorer settings and beta 1. It assigns
50/11/11 positions, producing 942–945 training examples including the reference.
Group integrity takes precedence over equal sequence counts; no GFN budget is added.


## GFN follow-up on position-held-out rewards

The [protocol](experiments/position_gfn_2026-11-09.md) retains split seeds 43/44/45
and crosses training seeds 0/1/2 with them (nine fits). Original 20,000-step TB,
beta 1, head 128, batch 64 and the step-8,000 rate reduction are unchanged. No
scorer fitting or reward sweep; exact monitoring observes predicted-target fit.


## Multi-protein stability predictors

The [10 November protocol](experiments/multiprotein_2026-11-10.md) fixes the
64-group panel, 65 sequences per parent, 40% reference-identity component rule,
selection seed 20261110 and group-split seed 42. It retains the frozen backbone
and scorer architecture, extends ridge penalties through 1000 and MLP budget to
2,000 updates, batch 128, validation every 50. All other scorer optimizer settings
are inherited. MLP seeds 0/1/2; deterministic ridge once. Top 10% selection is
compared with uniform candidates, without beta or GFN training.

## Fixed-coverage ridge regularization

The [protocol](experiments/regularization_2026-11-10.md) owns the grid and selection
rules: raw ridge penalties 1000/10000/100000/1000000/10000000; five group folds,
seed 42, within the same 44 training groups. Every fold normalizes on its own
training subset. Choose parent-balanced out-of-fold RMSE, refit once on all
training rows, and retain the original validation set for reporting. No optimizer
iterations, new feature extraction, new proteins or GFN hyperparameters.

## Training-protein coverage

[Protocol](experiments/coverage_2026-11-10.md): nested 44/64/86 training parents,
65 sequences each; original selection seed 20261110 and fixed outer splits.
Use the original ridge rule, with penalty `1000 * training_rows / 2860` to hold
strength per training example fixed. Refit normalization on each stage's training
rows. Frozen float32 ProGen features, batch eight for the added sequences only.
No selection sweep, MLP, GFN or test-label evaluation.

## Supervised-start comparison — 15 November 2026

The [protocol](experiments/idr_warmstart_2026-11-15.md#budgets-optimizer-and-random-streams)
owns all new choices and rationale. Both continuation arms start from the same
step-800 supervised head, reset Adam and receive 400 updates of 16 examples.
Restart inherited head rates (`1e-4`, direct branch `1e-3`), TB normalizer `1e-2`,
with ×0.1 after update 300 and clipping at 1. TB uses a training-only least-squares
normalizer initialization. The bank/reward stay unchanged. Evaluate 128 fresh
outputs per prompt per arm; new RNG seeds and the dual-control progression rule
are fixed in the protocol. These are engineering defaults, not tuned outcomes.

## Continuous-reward comparison — 15 November 2026

The [protocol](experiments/idr_continuous_2026-11-15.md) owns the full rationale.
Penalty 2.5 is rounded from 2.5099, calibrated on unselected seed-1 historical
training proposals to approximately match aggregate target success preference;
seed 2 is diagnostic only. Importance weighting includes prior/behavior correction.
Keep TB/replay, 400 updates, all optimizer settings, seeds 92000/95000 and the
same starting head/bank as the completed control. Evaluation seed 103000 is fresh;
128 outputs per prompt per arm. Only one new TB fit; no curriculum or sweep.


## Exact-task update diagnosis — 16 November 2026

The [diagnostic protocol](experiments/idr_update_diagnosis_2026-11-16.md) owns
fixed leaf weighting, 400-gradient-evaluation budgets, Adam/clipping controls,
L-BFGS choices and the post-hoc invertible input transform. These settings apply
to a selected finite CPU problem only; production generation parameters remain
unchanged. The successful fit does not select a stochastic training configuration.


## Sampled conditioning — 16 November 2026

The [protocol](experiments/idr_sampled_conditioning_2026-11-16.md) owns three paired
RNG seeds and the inherited 400-update sampled Adam/replay budget. It fixes the
same starting function, reward and rates across ordinary/conditioned inputs;
no hyperparameter sweep follows the failed progression gate.


## Expected-gradient conditioning control — 16 November 2026

The [protocol](experiments/idr_expected_conditioning_2026-11-16.md) owns one
400-update deterministic fit, inheriting all sampled-comparison rates, clipping
and initialization. Only the finite-sample gradient estimator changes to its exact
conditional mean. No sampling-seed repetitions or settings search are needed.


## Averaged sampled TB — 16 November 2026

The [protocol](experiments/idr_averaged_sampling_2026-11-16.md) preselects 16 draws
per fresh/replay group, three fixed seeds and 400 inherited Adam updates. This
spends 16 times as many trajectories to approximate the successful expectation;
no additional batch size or training length was selected after observing results.


## Variance-reduced fixed-bank TB — 16 November 2026

The [adaptive CPU protocol](experiments/idr_variance_repair_2026-11-16.md) owns the
variance audit, fixed proposal, 32 draws per prompt, snapshot refresh every 20
steps and inherited 400-update Adam schedule. Three matched development seeds
are followed by three predefined fresh confirmation seeds without retuning.
No reward or gate is changed. Counts include snapshot samples and full-bank passes.
