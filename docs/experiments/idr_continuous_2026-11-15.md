# Continuous acceptance reward: controlled GFlowNet comparison

**Stage:** user-run experiment completed and analyzed on 15 November 2026.
The assistant used saved outputs only; no model inference or GPU work. [Command](../runbook.md#continuous-reward-gflownet-comparison)
· [Configuration](../../configs/idr-continuous.json).

## Question and decision

Does removing compressed failure weights and the joint-pass bonus help TB learn
useful charge/disorder guidance? Keep GFlowNet training and ProGen frozen. The
[completed warm-start study](idr_warmstart_2026-11-15.md#completed-results--15-november-2026)
showed poorer generation despite improved selected-bank relative target fit;
57.5% of fresh failures were within 25% above the failure-weight floor. Stored
near misses were not saturated. This motivates a specific reward-only comparison,
not a claim that reward shape explains all failures.

The [RL review](../../literature/RL_REWARD_DESIGN_2026-11-15.md) distinguishes changing
terminal preferences from providing local credit. This experiment does only the
former. No SubTB, local flow estimator, curriculum, unfreezing, adaptive replay,
longer training or new supervised fitting is added.

## Reward and acceptance criteria

\[
R_\lambda(x\mid c)=p_0(x\mid c)\exp[-\lambda V(x)],\qquad \lambda=2.5.
\]

- `x`: generated 64-residue continuation; `c`: supplied 16-residue prefix.
- `p0`: frozen ProGen probability under the existing canonical-token support and
  temperature-one sampling; `R_lambda`: unnormalized terminal GFN reward.
- `V`: sum of normalized violations **outside** the existing acceptable ranges.
  It uses exactly the four terms in the [current reward](idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1):
  first-block charge, second-block charge, fraction charged and disorder deficit.
- `lambda`: positive violation penalty, fixed at 2.5 by the development audit below.

Every acceptable sequence receives property factor 1; worse violations get
progressively smaller positive factors. There is no constant failure floor or
extra acceptance jump. The term is continuous in `V`; sequence descriptors and
the per-residue disorder threshold remain discrete. No gradient flows through
the scorer. Compute `log w = -2.5 V` directly; use effective beta one and include
the prior exactly once. Multiplying by historical beta two would be incorrect.

The old factor is 1 on a joint pass and `(0.25 + 0.75 exp(-V))/20` otherwise.
This comparison therefore **changes the terminal distribution**, including the
relative preference for near misses and severe failures. It is not a neutral
rescaling or proof of better credit assignment. Large violations now yield larger
negative log weights; the calibration minimum is −67.19 versus −4.38 previously.
This may increase gradient clipping/noise. Preserve the existing optimizer and
clipping so the reward intervention remains isolated; do not silently cap penalties.

Success definitions stay fixed: two 32-residue charge means target +0.125/−0.125,
tolerance 0.0625; charged fraction target 0.25, tolerance 0.125; at least 80% of
isolated-tail V3 scores strictly above 0.5. Disorder deficit scale remains 0.2.
Nominal charge K/R +1, D/E −1, others zero. These are engineering benchmark choices.

## Penalty calibration and limitations

Use **unselected historical training trajectories**, never the generation evaluation
pools or success-selected replay bank. Source: `runs/idr-acceptance-v1/seed-1/`
and `seed-2/reward_inputs.jsonl`, 6,400 rows each (800 per prompt), with duplicates
retained. Seed 1 calibrates; seed 2 is a diagnostic and does not change the value.
Neither pool overlaps the fixed diagnostic bank's tails. Both contain previously
generated development data; this is not a new independent validation experiment.

These trajectories came from changing policies. Estimate each prompt's target
using importance weights `exp(log p0 - log q_behavior + log w)` normalized over
its pool. `q_behavior` is the recorded sampling probability of that trajectory.
Using property weights alone would incorrectly treat these proposals as prior
draws. Adaptive proposals and finite self-normalization still limit the estimates.

Solve for the penalty that matches the current reward's mean estimated joint
mass across equally weighted prompts on seed 1: 2.5099; round to one decimal,
**2.5**. This is a development calibration rule chosen for comparable aggregate
target preference, not a literature-prescribed constant or pre-registered result.
Do not choose a penalty by which trained generator performs best.

| Pool | Current estimated joint mass | Continuous estimated joint mass |
|---|---:|---:|
| Calibration, seed 1 | 50.02% | 49.84% |
| Diagnostic, seed 2 | 47.49% | 47.79% |

These are finite-pool target estimates, **not expected trained success guarantees**.
Prompt-level preferences differ. DP00181 has only 2/1 observed successes; its
candidate effective sample sizes are 12.41/12.35 of 800 and maximum normalized
weights 18.79%/24.24%. Retain that prompt and report it separately, without treating
its estimate as a reliable performance ceiling. Other prompts retain broader
coverage. The audit makes this a bounded mechanism test, not a well-validated target.

[Audit source](../../scripts/audit_idr_continuous_reward.py) and
[full per-prompt results](../verification/idr-continuous-reward-audit-2026-11-15.json)
record input hashes, weighting, calibration and diagnostic separation. The run
pins and copies the audit JSON; no online calibration or parameter sweep occurs.

## Arms and budget

| Arm | Weights | New training |
|---|---|---|
| `unchanged` | Original supervised step-800 head, shared starting point | None |
| `current_tb` | Completed warm-start TB step-400 checkpoint | None; reuse pinned control |
| `continuous_tb` | Same original supervised step-800 head | 400 TB/replay updates with the continuous reward |

Using the original common start isolates reward design. Do not start the new arm
from the later, stronger supervised continuation. The two TB arms have identical
head architecture, initial weights, training seed, replay RNG, update/example
budget and optimizer schedule. Their online trajectories can diverge after their
first update. The unchanged and current-TB heads are sampled afresh; old percentage
estimates are not substituted for new controls. The best observed supervised
continuation is retained as historical context, not a freshly evaluated fourth arm;
this experiment cannot establish superiority to that method.

| Setting | Value and rationale |
|---|---|
| Training method | TB with replay, inherited unchanged |
| Backbone/head | Frozen ProGen2-medium; direct-context guidance head |
| Initialization | `runs/idr-replay-v1/supervised/step-800.pt`, exactly the control's starting head |
| Updates | 400; match completed control, no automatic extension |
| Batch | 8 fresh + 4 stored successes + 4 stored near misses |
| Replay bank | Same 408 training / 95 diagnostic rows and frozen-feature cache |
| Bank relabeling | Replace terminal log factors for candidate only; no reselection or split changes |
| Optimizer | Fresh Adam, base head 1e-4 / direct branch 1e-3 / log normalizers 1e-2 |
| Schedule/clipping | All rates ×0.1 after update 300; norm clip 1, as control |
| Training/replay seeds | 92000 / 95000, matching control; evaluation verifies stored-example order/rates |
| Normalizer | Same equal-pass/fail training-bank least-squares rule, applied to each reward; not an estimate of true partition function |
| Checkpoints/logging | Every 100/10 steps; evaluate final checkpoint only |
| Evaluation seed | 103000 reset for each arm; fresh common stream distinct from earlier experiments |
| Evaluation count | 128 per prompt × 8 prompts × 3 arms = 3,072 raw continuations |

Only one new fit: 3,200 fresh training trajectories and 6,400 total presentations,
including 1,600 stored-success presentations. No new supervised fit occurs. The
shared initialization was previously trained by supervision; the learned candidate
is trained with the GFlowNet TB objective. Cost records separate reused training
from newly performed training/generation and retain control costs.

## Outcomes and stopping

Use the unchanged primary joint charge/V3 success endpoint on raw, unfiltered
samples. The candidate must gain **at least five percentage points over both
current TB and the shared starting head**, in raw and noncopy joint success,
while losing no more than five points in uniqueness against either. This inherited
engineering screen is not a significance test. Merely beating a degraded TB head
without improving on the start is insufficient.

Report charge, V3 and V2 sensitivity, per-prompt outcomes, uniqueness, repeated
trimers, exact copies against inherited plus actually consumed training examples,
and nearest-bank identity. V3 is optimized; V2 is related sensitivity, not independent
biological validation. Copies are not removed from the denominator. No novelty
claim against all known proteins follows from these checks.

Saved-bank relative log-ratio errors are evaluated against **both** rewards for
every arm, at initial/final stages. Exclude the union of diagnostic tails encountered
by either TB arm. These are selected finite-bank diagnostics, not global target
matching. Own-reward TB residuals use each arm's actual terminal weight; cross-arm
loss magnitudes with different rewards are not directly comparable. Training logs
retain success exposure and clipping. Do not declare success from lower TB loss.

If the screen passes, consider confirming robustness before adding properties.
If it fails, stop this reward-only route; inspect whether generation, target
preference or fitting failed and revisit local prefix-flow credit. No automatic
longer run, reward sweep, loss swap or backbone update.

## Verification and recovery

The [verification record](../verification.md#continuous-reward-gflownet--15-november-2026)
links CPU tests and real-input preflight. Tests use a tiny random backbone and
deterministic scores; they establish correctness, not efficacy. Checkpoint hashes,
control settings, identical starts, replay/rate schedules, unchanged backbone,
acceptance invariance, copying and cross-target diagnostics are checked.

The run snapshots its source/control manifests, checkpoints, bank/cache, histories,
reward audit, reward definition, inputs and outputs. Existing directories are
refused. Final scoring can be recovered on CPU from the snapshot after hiding
source runs/data; corrupted artifacts are rejected. Interrupted training does not
automatically resume. Historical source modules and completed run files remain intact.


## Completed results — 15 November 2026

`runs/idr-continuous-v1` completed in **39 min 5 s**. All 400 candidate updates
and 3,072 fresh evaluation outputs are present. Existing controls were reused as
planned, with fresh sampling; the matched replay identities/rates verify.

| Arm | Joint charge/V3 | Charge | V3 disorder | Unique tails | Exact training copies |
|---|---:|---:|---:|---:|---:|
| Shared supervised starting head | 101/1,024 (9.86%) | 13.57% | 78.71% | 100.00% | 0 |
| Previous-reward TB | 77/1,024 (7.52%) | 12.30% | 73.73% | 97.46% | 30 (2.93%) |
| Continuous-reward TB | 98/1,024 (9.57%) | 13.67% | 78.91% | 97.56% | 28 (2.73%) |

**The predeclared practical screen fails.** Continuous TB gains 2.05 percentage
points over previous TB but is 0.29 points below the starting head. Neither
comparison reaches the required +5 points. All exact training copies fail joint
acceptance, so noncopy joint success (with the original denominator) is unchanged.
These are sampled results from one training seed, not evidence of a significant
0.29-point decline or a reliable 2.05-point improvement.

Per prompt, the candidate beats previous TB on six and ties on two; versus its
starting head it improves four, ties one and worsens three. The low-coverage
DP00181 prompt has 11/128 passes unchanged and 1/128 for each TB model. Keeping
this prompt was prespecified; it is not excluded after seeing results. V2 joint
rates are 9.96%, 7.42% and 9.38%, respectively, with the same qualitative ordering.
V2 is related predictor sensitivity, not independent biological validation.

### What training learned

On the selected, unexposed diagnostic bank, mean within-prompt centered log-ratio
error against the **continuous target** goes from 1.832 at initialization to 1.123
for continuous TB. Previous TB gives 1.019 against that same target. Against the
**previous target**, errors are 1.973 initially, 1.370 after previous TB and 1.385
after continuous TB. No diagnostic rows were excluded for fresh-training exposure.
This indicates limited fitting progress, not global target matching, and does not
show that the candidate fits the continuous target better than the control.

Continuous TB encounters 234 fresh training successes out of 3,200, versus 225
for previous TB; both consume 1,600 stored-success presentations. Every update
clips gradients in both fits. Clipping alone does not identify the failure cause.
Mean new log property factor on fresh outputs is essentially unchanged from the
start (−5.7236 versus −5.7241), consistent with no aggregate improvement in the
summed violation score. Different-reward TB loss magnitudes are not compared as
if they measured the same objective.

### Interpretation and next decision

Removing the floor and pass jump recovers some of the previous TB model's lost
success, but does not improve on the shared starting head. The reward-only change
is insufficient **under this architecture and training budget**. This neither
proves the reward is optimal nor isolates credit assignment as the sole cause.
Finite-pool target estimates around 50% remain uncertain estimates, not an achieved
or verified attainable generation rate. The historical continued-supervision result
(16.11%) remains context, not a freshly evaluated comparator here.

Follow the stopping rule: **end this reward-only route; no penalty sweep, longer
run or seed confirmation is recommended.** Keep the GFlowNet component. The next
proposal is to assess a concrete prefix-flow parameterization that estimates the
remaining terminal property weight from partial sequences, with the terminal
target fixed and a tiny exactly enumerable validation before another GPU run.
This needs a mechanism and ablation distinct from our already unsuccessful generic
SubTB experiment; simply switching the loss again is not sufficient justification.
The [existing review](../../literature/RL_REWARD_DESIGN_2026-11-15.md) provides the
forward-looking-flow/soft-RL rationale and limitations. This is a proposed design
step, not an implemented trainer or a promise of improvement.

[CPU reconstruction](../verification/idr-continuous-result-analysis-2026-11-15.py)
verifies captured artifacts/source hashes and reproduces all five evaluation JSON
files exactly using cached V2/V3 predictions. The [audit](../verification/idr-continuous-result-audit-2026-11-15.json)
records provenance. No inference, training or GPU execution was needed; original
run artifacts remain unchanged. This verifies arithmetic, not biological validity.
