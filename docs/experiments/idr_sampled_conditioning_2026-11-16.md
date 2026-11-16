# Sampled TB with conditioned inputs — 16 November 2026

## Question and pre-run protocol

Does the function-preserving input conditioning that repaired deterministic TB
also help the inherited sampled TB/Adam updates? The motivating result is the
[fixed-objective diagnosis](idr_update_diagnosis_2026-11-16.md): at 400 gradient
calls, ordinary L-BFGS gives mean TV 0.10796 and conditioned L-BFGS 0.00163.
This comparison changes both sampling and optimizer relative to that successful
fit; it tests transfer to the existing sampled trainer, not noise in isolation.

Use the same 408 training leaves and 168 branching states across eight locally
masked trees from `runs/idr-continuous-v1`. No holdout, new sequence generation,
ProGen execution, disorder inference or GPU execution. Reuse the terminal reward,
initial supervised head, normalizers and exact evaluation from the prior diagnosis.
The finite prior is locally renormalized at allowed branches; it is not the full
ProGen prior globally conditioned on membership in the bank. Most residues are
forced. These are fitting diagnostics on selected known trajectories.

Compare ordinary and conditioned coordinates from the **same original start**,
not from the successful L-BFGS weights. Conditioning is computed once from initial
features/context and LayerNorm, without reward labels; it is fixed during fitting,
while LayerNorm remains trainable. Same trainable parameter count and initial
function (maximum prompt TV difference below 1e-4). The bank itself was selected
using outcomes; reward-blind transformation does not make the bank unbiased.

- Three paired sampling seeds: fresh draws 92000/92001/92002 and replay draws
  95000/95001/95002. Seed 0 reproduces the previous sampled control. Three seeds
  expose obvious sampling sensitivity without a search over favorable seeds.
- Exactly 400 updates per arm/seed, six bounded CPU fits. Each update averages
  eight prompt losses; each prompt contributes one current-policy leaf and one
  stored leaf with equal weight. Replay alternates pass/fail strata by prompt and
  step, exactly as before. Paired arms use identical replay draws and RNG seeds;
  current-policy draws may differ as their policies diverge.
- Inherit Adam: head rate 1e-4, direct task branch 1e-3, log-normalizer 1e-2,
  global gradient clipping 1, rates multiplied by 0.1 after step 300. Fresh optimizer
  per fit. Same numerical settings do not imply the same functional step under a
  change of coordinates; that is the intended intervention.
- Differentiate only the sampled TB residuals; sampling probabilities are detached.
  No teacher probabilities or target labels enter the loss beyond terminal rewards.
  Exact target probabilities are used only for reporting.
- Report final step 400, not best checkpoint: per-prompt and mean TV, joint-pass
  probability, normalizer error, clipping frequency, and matched seed differences.
  Save sample indices, trajectories' IDs, checkpoints, history and source hashes.
- Retain the local gate: mean and worst-prompt TV both <0.05 and mean joint-pass
  probability within 0.02 of the exact target. Require all three conditioned fits
  to pass before calling this a robust sampled repair. This is an engineering
  progression gate, not a statistical significance test.
- Stop at this budget regardless of outcome. No rate/batch sweep, automatic longer
  training or full GPU run. Failure means conditioning alone has not repaired the
  inherited sampled optimizer at these settings, not that TB cannot fit the target.
  Success justifies proposing a separate short fully branching capture/check.

## Reproduction

From the project root, activate `.venv`, then use a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/compare_idr_sampled_conditioning_cpu.py \
  --output runs/idr-sampled-conditioning-v1
```

This record was written before running the comparison. Implementation uses the
existing finite-tree, conditioning and optimizer helpers; no production trainer
or earlier experiment definition changes. Sources and rationale are the linked
local diagnosis; no additional literature claim or search is needed.

## Completed results

All six fits completed on CPU; actual fit times were 3.4–3.5 seconds for ordinary
inputs and 5.5–5.6 seconds for conditioned inputs (excluding loading/verification).
Outputs: `runs/idr-sampled-conditioning-v1`. Final step 400 results:

| Paired seed | Ordinary mean TV | Conditioned mean TV | Ordinary joint pass | Conditioned joint pass | Conditioned worst-prompt TV |
|---|---:|---:|---:|---:|---:|
| 0 | 0.30865 | 0.25073 | 36.03% | 41.39% | 0.34286 |
| 1 | 0.30814 | 0.25444 | 35.90% | 40.86% | 0.34600 |
| 2 | 0.30981 | 0.25494 | 35.98% | 40.84% | 0.35511 |
| Mean | **0.30887** | **0.25337** | **35.97%** | **41.03%** | — |

The exact target has 62.019% joint-pass probability; the common starting head has
34.17% and mean TV 0.32547. These are exact probability masses on known finite
leaves, not empirical success rates on newly generated sequences. The reward
combines charge criteria with predicted disorder, not measured biological function.

Conditioning reduces mean TV by about 18% and increases joint mass by 5.06 percentage
points relative to the ordinary sampled control, consistently across these seeds.
**All six fits fail the local gate.** It is a partial improvement, not a sampled
learning repair. All updates clip gradients in both arms; that observation alone
does not show clipping causes the remaining error. Removing clipping already failed
to help the earlier ordinary fixed-objective control.

The ordinary seed-0 TV reproduces the earlier sampled control exactly. Replay
indices match within each pair. Saved-checkpoint reconstruction reproduces every
per-prompt metric and normalizer error; saved folded heads preserve policies within
1e-4 TV. Twelve targeted CPU tests pass, including sampled-residual gradients,
legacy control equivalence, paired replay, unchanged source weights and the fact
that exact target probabilities do not train the head. See the
[test log](../verification/idr-sampled-conditioning-tests-2026-11-16.txt) and
[independent saved-artifact audit](../verification/idr-sampled-conditioning-audit-2026-11-16.json).
Source/input hashes match; no pretrained model, scorer or GPU was executed.

## Interpretation and next decision

The same reward and head can fit this task under deterministic conditioned L-BFGS,
but conditioning alone does not transfer that success to inherited sampled Adam
within 400 updates. This comparison changes optimizer as well as sampling relative
to the successful fit, so **we cannot attribute the gap to sampling noise alone**.
Nor does it prove the reward is adequate for unrestricted generation: the target
is deliberately easier and most residues are forced.

Stop at the declared budget. Keep full-generation runs paused. The most informative
next proposed control is **conditioned expected-TB with the same Adam updates**:
average over every leaf using one-half detached current-policy probabilities and
one-half uniform mass in that step's replay stratum. This is the exact conditional
expectation of the sampled loss gradient, not the different fixed proposal used
by L-BFGS. Hold initialization, conditioning, reward, clipping, rates and 400 updates
fixed. It removes finite-sample noise without changing the GFN training objective
or introducing teacher labels. If it fits, investigate sampling variance; if it
also underfits, investigate optimizer/coordinate step behavior first. The proposal
is not implemented or run by this comparison and needs its own bounded protocol.
It is more diagnostic than another reward change or blind rate/length increase.

**Subsequent control:** the [expected-gradient fit](idr_expected_conditioning_2026-11-16.md)
is now complete and passes the local gate with the same Adam settings. The proposal
above describes the decision at completion of this sampled comparison; follow the
linked record for the new result and next decision.
