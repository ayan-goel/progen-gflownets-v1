# Joint-acceptance reward training — 13 November 2026

**Stage: completed; fails the practical improvement screen.** The user ran all
GPU work. This protocol fixes one reward intervention, with no hyperparameter
search, replay, SubTB, backbone updates or extra training extension.

## Question and hypothesis

Does the revised reward translate its better-aligned target into substantially
more raw generations satisfying both charge and predicted-disorder requirements?
The [reward audit](idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1)
found that the old smooth score could favor charge failures over passes and omitted
disorder. Its separate `joint_acceptance_v1` replacement fixes those definitions
and limits the log property factor's range. The saved-pool 47.86% joint estimate
motivates this test; it is not a forecast of the trained sampler.

The intervention changes acceptance preference, failure-score scale and inclusion
of disorder together. It tests the replacement reward as a package, not which
individual term caused any improvement. Bounded reward range may reduce extreme
signals, but a larger probability mass on rare joint passes may remain hard to learn.

## Methods and controls

| Method | Training | Final sampling |
|---|---|---|
| New TB | Three fresh direct-context heads, revised reward | 64 raw outputs per prompt/seed |
| Old TB | Reuse final step-800 TB heads from `runs/idr-subtb-v1` | 64 fresh raw outputs per prompt/seed |
| ProGen | Frozen prior, no head | First 64 outputs of each fresh prior pool |
| Generate-and-select | No head training | Best 64 of 256 prior proposals per prompt/seed by the **new** property factor |

Reusing old checkpoints avoids repeating unchanged fits. These are the original
three training replicates, not additional old-method replications. Evaluation is
new for all methods. Shared prompts, sample counts, prior definition and scorer
make this a controlled development comparison. Old/new evaluation uses the same
random-number seed per replicate; outputs diverge as policy probabilities differ.
Selection shares the prior pool, includes discarded-proposal cost and breaks ties
by original generation order. It does not secretly use the old center score or
deduplicate passing sequences. No disorder filtering applies to raw methods.

The scope is this reward comparison. Composition swaps/shuffles from earlier
experiments remain historical controls and are not rerun here. This omission does
not establish superiority to direct sequence design. The previously inspected
eight prompts remain development contexts; new random draws are not a held-out
biological benchmark or proof of global sequence novelty.

## Fixed settings and rationale

| Setting | Value / rationale |
|---|---|
| Backbone | Same frozen ProGen2-medium checkpoint, float32, canonical20-renormalized logits, temperature 1 |
| Context/output | Same 8 prefixes, 16 supplied + 64 generated residues; no EOS sampling |
| Head | Same zero-output residual 128-hidden-unit MLP and 5-to-20 direct task branch |
| Training seeds | 0, 1, 2; same initial heads as old TB |
| Training draws | `30000 + seed`, matching old TB's starting stream; trajectories diverge during learning |
| Updates / batch | 800 / 8; one complete trajectory per prompt, one update per batch |
| Adam learning rates | Original head `1e-4`, direct branch `1e-3`, prompt log normalizers `1e-2` |
| Schedule / clipping | All rates ×0.1 after update 600; joint head/normalizer norm clipping at 1 |
| Normalizer initialization | Zero, inherited from the old control |
| New prior draws | `60000 + seed`; 256 per prompt/seed; 6,144 proposals total |
| New final-policy draws | `70000 + seed`; 64 per prompt/seed/method; 1,536 outputs each |
| New training trajectories | 19,200 total; same number of V3 isolated-tail scoring requests |
| Saved checkpoints | Every 100 updates; evaluate **only final**, no outcome-selected checkpoint |
| Training reward predictor | Pinned metapredict 3.0.2, V3, unrounded isolated-tail scores, explicitly CPU |
| Final assessment | V3 and V2, full-context and isolated-tail scores; V3 joint success is primary |
| CPU threads | 1, inherited; scoring cost measured separately within total training cost |

These optimization settings are inherited to isolate the reward intervention.
Fresh evaluation streams avoid reusing the samples used to design the reward.
No additional reward constants are selected from the resulting evaluation.

The [audited reward contract](idr_reward_audit_2026-11-13.md#definition-and-every-parameter)
fixes all tolerances, scales and symbols. In brief, the property multiplier is
1 for a joint pass and between 1/80 and 1/20 for a failure, improving with proximity
to the acceptable region. This runner supplies **log(multiplier)** directly to TB
with effective beta 1; it does not multiply again by historical beta 2.

For a generated tail `x` and prompt `c`, the residual is

`log Z(c) + log q_theta(x|c) - log p0(x|c) - log w(x)`.

Here `Z` is the learned prompt normalizer, `q_theta` the policy from the head with
parameters `theta`, `p0` the frozen prior, and `w` the new property multiplier.
TB minimizes the batch mean squared residual. Only the head and normalizer get
gradients; sampled sequences, ProGen features, disorder predictions and terminal
rewards are detached. The predictor is never fitted or differentiated through.

The run saves `pilot_config.json` with effective beta 1 and the new reward version,
`config.json` with the complete new reward settings, and `source_config.json` with
the old beta-2 protocol. The inherited `reward` descriptor block in the pilot
config supplies charge context; it does not override the new terminal scorer.

## Evaluation and decision rule

Primary: raw joint charge/V3 pass rate, compared with both old TB and ProGen.
Report charge alone, V3 disorder, V2 joint/disorder sensitivity, unique tails,
repeated trimers, entropy, mean pairwise Hamming distance, charge means, prior
log probability and similarity to the limited reference panel. Aggregate results
are balanced because every prompt/seed has equal counts; also report every seed,
prompt and prompt/seed combination. V3 is an optimized proxy. V2 is a related-model
check, not independent biological validation; no experimental claim follows.

Fix the practical continuation screen before the real run:

- At least **+5 percentage points** in aggregate raw joint success over **fresh
  outputs from the old TB heads**, not over the historical 4.3% sample estimate.
- Improvement in at least **two of three seeds**.
- No more than a **5-percentage-point loss** in aggregate unique-tail fraction.

These thresholds are engineering judgments, not significance tests. Passing them
supports further guidance work, not broad biological success. Inspect per-prompt
and V2 results for concentration or proxy disagreement even if the screen passes.
If gains are still practically small, stop this bounded test and diagnose encounter
frequency versus target fitting before choosing an intervention. No automatic
reward/rate sweep, longer run or unfreezing follows.

Use the fresh 256-proposal groups to estimate old/new targets with normalized
property factors, then average groups equally. Because proposals come from `p0`,
do **not** multiply by `p0` again. Report ESS, largest weight, observed pass counts
and zero-pass groups. These estimates cannot reveal unseen modes or verify exact
sampling. A learned normalizer and residual are not substitutes for these limits.
Each policy's held-out TB residual uses **its own** reward and normalizer; the raw
loss scales differ, so do not rank methods by which residual is smaller.

## Artifacts, integrity and recovery

- Pin the completed control generation manifest in `configs/idr-acceptance.json`.
  Verify captured artifacts, checkpoint metadata, references, packages, scorer
  files and frozen model identity before running. Tiny fixtures cannot be used as
  real controls. Preserve original files and copy old heads into the new run.
- Each seed retains update logs, checkpoints and `reward_inputs.jsonl`: tail,
  per-residue V3 scores, violations, pass flag, log factor, prior probability and
  sampling-policy probability. These allow reward reconstruction and counting
  observed/unique joint passes without another predictor or ProGen pass.
- `samples.jsonl` contains 9,216 raw evaluation/proposal records. Evaluation adds
  only derived prior/selection rows in `sequence_metrics.json`; raw methods are
  unchanged. `metrics.json`, `target_estimates.json`, `decision.json` and
  `training_diagnostics.json` feed the concise `report.md`.
- Save `generation.json` with hashes before final scoring. If final CPU scoring
  fails, `evaluate-idr-acceptance` can recover from this snapshot without training,
  ProGen, original run directories or original reference files. It still verifies
  scorer/software identity. Training interruption has no automatic resume.
- `costs.json` separates new fitting, training scorer time and fresh generation;
  old fitting is labeled reused, with original costs in `source_costs.json`.
  `evaluation_cost.json` records final scoring cost. This is matched training
  trajectory/update budget, **not matched wall-clock cost**.

## Verification and sources

CPU integration uses a random tiny ProGen architecture and deterministic fixture
scores, never the pretrained model or real metapredict inference. It verifies
updates/frozen weights, reward-input reconstruction, inherited rates, original
control integrity, method-specific residuals, raw/selection counts, prior-correct
importance weights and exact CPU evaluation replay after hiding the source run.
Real-input preflight passes without model execution or GPU queries.

Motivation is the [completed TB/SubTB result](idr_subtb_2026-11-13.md),
[reward audit](idr_reward_audit_2026-11-13.md), and
[targeted literature review](../../literature/IDP_IDR_REVIEW_2026-11-12.md#11-bounded-joint-acceptance-reward--13-november-2026).
The bounded reward and decision thresholds are our design choices, not prescribed
hyperparameters from those papers. [Commands](../runbook.md#joint-acceptance-tb--completed)
cover reproduction and evaluation recovery.

## Completed results and next decision — 13 November 2026

`runs/idr-acceptance-v1` completed all three fits and evaluation in **2 h 56 min
17 s**. The revised reward did **not** produce a meaningful generation improvement.
All raw methods below have 1,536 outputs: 64 per prompt/seed over eight prompts
and three seeds. These are fresh samples, so the old-TB counts differ from its
previous experiment's counts.

| Method | Charge pass | Joint charge / V3 pass | V3 disorder pass | Unique tails |
|---|---:|---:|---:|---:|
| ProGen | 70 (4.56%) | 34 (2.21%) | 65.76% | 95.05% |
| Old TB | 122 (7.94%) | 56 (3.65%) | 64.84% | 95.64% |
| New reward TB | 88 (5.73%) | 48 (3.13%) | 68.10% | 95.64% |
| Generate-and-select, four proposals/output | 222 (14.45%) | 160 (10.42%) | 87.37% | 98.24% |

New-minus-old joint success is **−0.52 percentage points**, with improvement in
only one of three seeds and no change in unique-tail fraction. It fails the
predeclared +5-point / two-seed screen. Eight fewer passes do not establish
statistically reliable harm; the conclusion is absence of a practically useful
gain. Disorder improves modestly while charge success falls. V2 joint success is
3.26% for new TB versus 3.65% for old TB, so it does not reverse that conclusion.

| Seed | ProGen joint / 512 | Old TB joint / 512 | New TB joint / 512 |
|---|---:|---:|---:|
| 0 | 9 | 22 | 16 |
| 1 | 9 | 16 | 17 |
| 2 | 16 | 18 | 15 |

Selection retains all 160 joint passes in the 6,144-proposal pool. This is a
useful sampling-cost baseline, not a trained sampler or evidence of exact target
sampling. All methods use the previously inspected development prompts; the V3
screen is an optimized prediction, not experimental disorder validation.

### Target alignment versus learning

Reweighting the **fresh** prior pool estimates 51.46% joint success for the new
target versus 13.00% for the old target. This agrees directionally with the earlier
reward-design audit, but **51.46% is neither an observed GFN result nor an exact
or guaranteed attainable rate**. New-target median effective sample size is only
17.86 of 256 proposals per group, minimum 14.47; three groups contain no joint
pass. Finite proposals cannot identify missing modes, and zero observed passes
do not imply zero true probability. The largest normalized weight is 0.165.

The target still appears to favor acceptable sequences, while the trained policy
barely shifts their frequency. Better alignment with our pass criteria did not
solve fitting. The acceptance jump and failure floor may themselves affect
learnability; these results do not establish that the reward is now optimal or
that the current head has sufficient capacity.

### Post-hoc diagnosis from saved trajectories

The run encountered **480 successful trajectories out of 19,200 (2.5%)**. Counts
are 183 / 149 / 148 by seed, each distinct within its seed. Every prompt had at
least one successful training example in every seed, but encounters were uneven.
**1,964 of 2,400 batches (81.83%) had no success.** Each trajectory received only
one update; the run used no replay. This is sparse and weakly reused evidence,
not a complete failure to discover successes.

Mean TB loss falls from about 14 in the first 100 updates to 0.45–0.65 in the last
100. Yet on fresh new-TB samples, residual RMSE is **0.215 for the 1,488 failures
versus 3.883 for the 48 passes**. Those passes account for **91.31% of total squared
residual**. This post-hoc stratification uses this policy's own reward throughout;
loss scales across old and new rewards are not directly comparable. The model
fits common failures much better than rare successes despite a low training loss.
Gradients are clipped on 99.42% of updates; this is a diagnostic, not proof that
changing clipping alone would fix learning.

Saved checkpoints are finite, their task weights changed, final normalizers match
training logs, and reused old heads match the pinned original files. Independent
reward reconstruction found no arithmetic or pass-threshold discrepancy. The
observed failure is not explained by those implementation errors.

### Follow-up motivation and subsequent result

The subsequent [comparison of alternatives](../decisions.md#2026-11-13--compare-alternatives-before-committing-to-replay)
retains replay as the next GFN intervention and adds a small supervised same-head
control to distinguish learning-objective trouble from broader head/data limits.
The subsequently authorized [pilot](idr_replay_2026-11-13.md#completed-results--14-november-2026)
is complete: replay gives a small gain and supervised training gives stronger
steering, though both miss the progression screen. The proposal below records its motivation.

Try **one bounded replay-assisted TB pilot**: repeatedly train on successful and
informative near-miss training trajectories alongside fresh policy samples, while
keeping this reward, frozen backbone and head fixed. This directly addresses the
observed imbalance in learning opportunities. Reuse training trajectories only;
keep final evaluation samples out of the replay buffer. Evaluate fresh generations
and success-stratified residuals, not just replay loss or reproduction of stored
sequences. Include unique outputs and distance to replay examples to detect memorization.

Before implementation, fix the mixture, update/trajectory budget, seed and stopping
rule. A single-seed pilot can screen the mechanism, not establish robustness. Use
the existing +5-point joint-success threshold as a practical progression screen
against a matched control, with no more than a 5-point uniqueness loss; multi-seed
confirmation follows only if the pilot shows a substantive effect. If successful
trajectories remain poorly fitted or fresh generation barely improves, stop this
intervention and reassess head expressivity/credit assignment instead of extending
training or sweeping minor settings. Reward scaling remains a competing explanation.

Replay is motivated by this run and the previously reviewed
[Hu et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/bc667ac84ef58f2b5022da97a465cbab-Paper-Conference.pdf)
and [Shen et al., ICML 2023](https://proceedings.mlr.press/v202/shen23a/shen23a.pdf).
Their settings differ; neither promises improvement for this frozen-ProGen head.
See [review section 9](../../literature/IDP_IDR_REVIEW_2026-11-12.md).

### Result verification

The [saved-artifact audit](../verification/idr-acceptance-result-2026-11-13.json)
and [analysis source](../verification/idr-acceptance-result-analysis-2026-11-13.py)
record 42 captured-file hash checks, 12,288 scored-row reconstructions, 19,200
training reward reconstructions, 2,400 update/rate checks, target reweighting,
stable-tie selection, method-specific residuals and final checkpoint checks.
No pretrained model, real disorder scorer, training or GPU operation was run
during analysis. Completed run artifacts remain unchanged.
