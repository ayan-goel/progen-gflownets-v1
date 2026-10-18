# Late learning-rate reduction experiment — 14 October 2026

## Status and purpose

**Subsequent milestone, 15 October:** the extended low-rate run passed all three
final seeds. See the [milestone and biological pilot](exact_pass_and_biological_pilot_2026-10-15.md)
for the current next step. The results and planning below preserve the earlier run.

**Completed by the user:** final TB TV is 0.053937 / 0.045554 / 0.049461.
Two of three seeds pass; the all-seed gate remains failed. The lower-rate phase
substantially reduces observed TV fluctuations. The next prepared experiment
extends only that phase; see the results and command at the end of this record.
No real fitting or GPU operation was performed by the agent.

The previous 10,000-update TB run had TV 0.097115 / 0.047138 / 0.077344, so only
one seed passed. Its sampled losses fluctuated late and all 21 logged checkpoints
from 8,000–10,000 had pre-clipping gradient norm above 1. Those observations motivate
a smaller late learning rate, but do not prove excessive step size caused failure.
The old run did not save intermediate exact TV. See the
[previous result](exact_long_2026-10-14.md).

This experiment asks whether a predeclared lower-rate final phase improves
three-seed final accuracy, while adding the missing distribution-learning curve.
Architecture, fixed reward, sampling, and the TV <0.05 gate remain unchanged.

## Configuration and exact schedule

Executable configuration: [exact-decay.json](../../configs/exact-decay.json).
Relative to [exact-long.json](../../configs/exact-long.json), each of `training`
and `supervised` gains only:

```json
{
  "lr_decay": {"after_fraction": 0.8, "factor": 0.1},
  "monitor_exact_distribution": true
}
```

| Quantity | Updates 1–8,000 | Updates 8,001–10,000 |
|---|---:|---:|
| TB head Adam learning rate | 0.001 | 0.0001 |
| TB log-normalizer Adam learning rate | 0.01 | 0.001 |
| Supervised head Adam learning rate | 0.001 | 0.0001 |

The boundary is `floor(total_updates * after_fraction)`; reduction applies **before**
the first update after that boundary. Adam moments are retained, not reset. Rates
are set from their configured initial values, so reduction does not compound.
The optional schedule requires both values strictly between zero and one, with
at least one update in each phase. Non-finite values are rejected.

The 80% boundary and tenfold reduction are bounded engineering choices, not
validated optima. Keep the earlier learning phase intact and reduce the late rate
while retaining the head/log-Z rate ratio. Apply the same head schedule to the
supervised baseline. This is a coupled rate-schedule intervention; it does not
separately identify the effect of the head rate versus the log-Z rate.

All other settings match the completed longer experiment: 10,000 updates,
seeds 0–2, batch 64, width 128, beta 1, clipping norm 1, 4,096 baseline rollouts,
float32 cached ProGen features and head computations, and the same evaluation
sample counts. Start every seed fresh; no privileged weights, continuation,
replay, early stopping or best-checkpoint selection. Old configurations omit
these options and retain constant-rate behavior and no intermediate exact monitor.

## Distribution monitoring and cost

For both heads, record before training (step 0), after update 1, every 100 updates,
final update, and both sides of the rate transition (8,000 and 8,001). This gives
**103 observations per head per seed** in the real configuration. Initial loss
and gradient norm are null because no update has occurred. The progress printer
shows `n/a` for that loss, and displays the current learning rate and TV.

Each row's nested `distribution` contains post-update TV, both KL directions,
probability sum, expected score and target expected score, entropy and exact log Z.
TB rows additionally include absolute learned log-Z error. The existing `loss`
and `gradient_norm` remain pre-update sampled quantities; `log_z` and distribution
metrics are post-update. `learning_rate` and `normalizer_learning_rate` identify
the rates used for that update. Do not equate a sampled loss fluctuation with an
exact TV fluctuation.

The monitor enumerates cached probabilities under no-grad, uses no random numbers
and sends no metrics back into training, sampling or checkpoint selection. The CLI
rejects training-time exact monitoring for biological experiments. The toy's exact
labels are observational diagnostics, not extra learning targets.

`costs.json` records per-method:

- `training_seconds`: inclusive elapsed learner time (including monitoring; the
  supervised learner also includes its existing rollout-label construction).
- `exact_monitoring_seconds`: the measured monitoring portion. Outstanding CUDA
  work is synchronized before its timer; CPU copies finish metric evaluation.
- `training_seconds_excluding_exact_monitoring`: inclusive time minus that portion;
  this is approximate wall-time attribution, not a kernel profiler. The synchronization
  before timing remains charged to the learner.
- `exact_monitoring_calls` and `exact_monitoring_cached_score_reads`: separate from
  the unchanged training reward-lookup counts. Each observation reads all 256 toy
  scores: 26,368 reads per head per seed. No new backbone/scorer execution occurs.

Final post-training evaluation still has its existing accounting; these new fields
cover only the added training-time observations. Historical cost totals cannot be
compared while silently ignoring this monitoring overhead.

## Run and inspect

From the project root, with `.venv` active and your chosen `CUDA_VISIBLE_DEVICES`
exported:

```bash
scripts/run.sh run --config configs/exact-decay.json \
  --output runs/exact-decay --device cuda
```

The launcher preserves manual GPU selection. The run automatically trains and
evaluates; the output directory must be new. Inspect:

- `runs/exact-decay/gate.json`: final three-seed TV and the unchanged gate.
- `runs/exact-decay/evaluation/report.md`: raw and selected comparison.
- `runs/exact-decay/seed-*/gfn_progress.jsonl`: flushed TV/KL/rate history during TB.
- `runs/exact-decay/seed-*/supervised_progress.jsonl`: corresponding baseline history.
- `runs/exact-decay/seed-*/*_history.json`: completed histories.
- `runs/exact-decay/costs.json` and `status.json`: costs and completion/gate status.

If all three final TB seeds reach TV <0.05, this establishes the exact engineering
gate for this task, not protein stability. If errors remain, the new curve can
separate persistent approximation error from late excursions at the recorded
checkpoints; it cannot establish what happened between logged points. No threshold
relaxation or oracle-selected checkpoint is permitted. Biology remains gated until
the exact criterion and subsequent scorer validation pass.

## Verification

**26 CPU tests pass**; [test log](../verification/schedule-tests.txt).
The new tests check the off-by-one boundary in both learners, actual weight changes
under scheduling, invalid options, and forbidden biological monitoring. Monitoring
on/off produces exactly identical final weights, normalizer, sampled losses and
CPU random-number state on the four-leaf fixture. Costs and callback records agree.
An end-to-end 20-update tiny random-ProGen smoke verifies persisted progress,
step-0 handling, boundary updates 16/17, final metrics and a deliberately failed
real-experiment gate.

The first integrated test exposed a formatter that assumed every progress row had
a numeric loss; it was fixed to handle initialization explicitly before rerunning
the suite. Ruff lint and formatting pass. A separate ten-update four-leaf check
compared the constant-rate path against the saved original training implementation:
weights, normalizer and original history fields match exactly. See
[compatibility/configuration check](../verification/schedule-compatibility.json).
Original run input hashes are unchanged. CPU verification does not assert GPU
bitwise equivalence or that this schedule will make all real seeds converge.

## Completed scheduled run and interpretation

The user-run [status](../../runs/exact-decay/status.json) reports `exact_gate_failed`
after **234.53 seconds**, about 3 minutes 55 seconds. The
[gate](../../runs/exact-decay/gate.json) agrees exactly with the final monitored
TB distributions; all six learners logged 103 observations and the specified rates.
The [read-only analysis](../verification/exact-decay-analysis.json) records input
hashes and curve summaries. No model execution or training was needed for this audit.

| Seed | Constant-rate final TV | Scheduled final TV | Final criterion |
|---|---:|---:|---|
| 0 | 0.097115 | 0.053937 | Fail |
| 1 | 0.047138 | 0.045554 | Pass |
| 2 | 0.077344 | 0.049461 | Pass |

The final head, not a best intermediate point, determines the gate. Seed 0's lowest
logged TV was 0.049373 at update 9,800; it finished at 0.053937. Its four logged
passing points do not make this a passing final run. Seed 2 also moves across the
boundary late; these are small margins, not strong convergence across repetitions.

Within the scheduled run, observed TV population standard deviations fell from
0.02729 / 0.02177 / 0.04176 in updates 6,000–8,000 to
0.00295 / 0.00539 / 0.00324 in updates 8,100–10,000. Means in those windows fell
from 0.09622 / 0.08772 / 0.11033 to 0.05311 / 0.05026 / 0.05249.
The first window has 21 logged points, the second 20. These descriptive windows
combine progress and fluctuations, are temporally correlated, and are not an
estimate of independent sampling uncertainty or a formal causal test. Together
with the constant-rate final comparison, they support the late-rate intervention.

The three final supervised TVs are 0.071190 / 0.066430 / 0.071970. As in earlier
runs, this is the Monte Carlo baseline, not exact conditional supervision.
No biological conclusion or statistically established superiority follows from
these small synthetic-task comparisons. Monitoring itself used about 0.47 seconds
summed over all six learners; inclusive elapsed runtime includes other work too.

## Next prepared run: extend only the low-rate phase

[exact-decay-long.json](../../configs/exact-decay-long.json) changes each learner's
`steps` from 10,000 to 20,000 and `lr_decay.after_fraction` from 0.8 to 0.4.
The absolute boundary is therefore still update **8,000**, with the original rates
through 8,000 and reduced rates from **8,001 through 20,000**. This prevents an
accidental extension of the high-rate phase when increasing the total budget.
The lower-rate phase grows from 2,000 to 12,000 updates. The factor remains 0.1;
no architecture, batch, seed, reward, sampling, gate or scorer setting changes.
Both TB and supervised budgets increase together; the baseline still uses the same
4,096 rollout-label count, so this is more fitting rather than more labels.

The doubled total budget is a bounded engineering test, not a promise of passing.
The failing seed's late curve is near a plateau with fluctuations around 0.05;
more low-rate optimization may help, but the evidence does not guarantee it. If it
remains near that floor, use the curves and residual errors to investigate input
conditioning or stochastic optimization instead of repeatedly doubling the budget.
No threshold is changed and no best intermediate checkpoint is substituted.

Run fresh from the project root after selecting your GPU and activating `.venv`:

```bash
scripts/run.sh run --config configs/exact-decay-long.json \
  --output runs/exact-decay-long --device cuda
```

This is a fresh three-seed run, not a resume from privileged or previous weights.
Monitoring follows the same interval and absolute transition, now 203 observations
per head per seed (51,968 cached score reads each). Final acceptance still requires
TB TV <0.05 in all seeds. The existing biological configuration remains gated.
The agent has prepared and validated the configuration but has not launched it.
No training code changed for this follow-up. Configuration validation and an exact
diff check verify only the two declared values in each learner change; the
existing 26-test implementation verification still applies.
