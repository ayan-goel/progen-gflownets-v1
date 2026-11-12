# Why charge guidance barely changed generation

**12 November 2026 — completed post-hoc diagnosis, CPU only.**
Source: [first charge pilot](idr_charge_2026-11-12.md#completed-results--12-november-2026).
This investigation inspects saved checkpoints and runs tiny synthetic learning
checks. It does not run pretrained ProGen, retrain real heads, or alter the pilot.

## Main conclusion

The failure is **weak policy learning in the current setup**, not demonstrated
failure of the exact charge reward. The policy barely moves from ProGen, scarcely
uses the explicit half/charge context in controlled probes, and has noisy,
consistently clipped updates. Its loss can decrease substantially as normalization
estimates change without learning useful sequence steering.

No missing-gradient, wrong-sign or token-alignment bug was identified. We cannot
attribute the real-run failure to one hyperparameter from observational logs.
The most plausible combined explanation is a small/noisy training batch, weak
use of task context, and an inadequately validated optimization budget on a much
larger sequence space. These are design limitations in our pilot, not evidence
that GFlowNets inherently cannot guide IDRs.

## Checks on the real saved run

1. **The requested settings actually ran.** Each parameter's Adam step is 400,
   intermediate checkpoints exist at 100/200/300/400, and the final head/normalizer
   rates are 1e-5/0.001, matching the planned tenfold decay after step 300.
2. **Gradients were not disconnected.** Head output weights started at zero and
   end with norms 0.0659, 0.0701 and 0.0749. Hidden and normalization weights also
   change. The frozen-backbone/cached likelihood code agrees with teacher forcing
   in the existing CPU correctness tests. Actual source hashes match run provenance.
3. **The policy barely changes sampled paths.** Saved final log(q/p0) has standard
   deviations 0.176/0.207/0.165. Its pooled mean is 0.00285 nats; it is a noisy
   Monte Carlo KL estimate, not exact KL. Prior and GFN both get 20/384 charge
   passes and nearly neutral mean charge in both halves.
4. **Loss decrease overstates steering.** The prior results record shows that
   removing learned path corrections barely changes final TB MSE, while removing
   the learned log-normalizer roughly doubles it or more. This algebraic check
   identifies which term changes the residual; it is not a trained causal ablation
   and does not mean learning log Z is wrong.
5. **Every update clips.** All 1,200 logged gradient norms exceed one. Median
   pre-Adam scaling factors are 0.0169/0.0183/0.0145. These are gradient scaling
   factors, **not equivalent reductions in the Adam learning rate**: Adam also
   rescales by accumulated second moments. Changing clip thresholds is therefore
   not guaranteed to fix learning.
6. **Late updates add little output-head magnitude.** Output-weight norms at step
   300→400 are 0.0653→0.0659, 0.0710→0.0701 and 0.0751→0.0749. The preset decay
   was applied despite unresolved residuals and weak steering. This warrants
   testing the schedule, but does not prove that keeping the rate high would help.

The reward arithmetic and probability conventions are independently checked in
the [completed pilot verification](idr_charge_2026-11-12.md#verification-of-completed-outputs).
The prior-pool importance diagnostic estimates target charge success near 27.8%,
well above observed GFN success, with ESS 47.8–75.5/192 per prompt. This finite-pool
estimate is approximate; it nevertheless argues against assuming beta=2 creates
no meaningful tilt. Selection's 79/384 actual charge passes provides a separate
practical indication of available candidates.

## Controlled context probe: a weak route for the charge instruction

The head has 1,536 frozen-feature inputs and 13 explicit context inputs (five
progress/charge values plus eight prompt indicators). This count alone does not
prove a problem. We tested its actual sensitivity using each saved final head:

- Generate 32 fixed Gaussian feature vectors, seed 456, and keep each vector and
  prompt identity identical across the pair.
- Compare the empty-tail context with a context containing 32 neutral residues:
  progress changes from 0 to 0.5 and desired block sign from +0.125 to −0.125;
  accumulated charge/FCR remain zero.
- Measure centered next-token logit changes. Maxima are only **0.000212, 0.000760,
  0.000781** across the three heads.
- On an illustrative uniform prior, the mean resulting first-versus-second-context
  signed-charge difference is only **0.0000113, 0.0000650, 0.0000458**.

The direct context route has learned little response to the instruction that must
reverse the desired charge along the sequence. However these are synthetic feature
vectors, not actual ProGen prefix states. The frozen representation itself changes
with position and could carry task information; this probe cannot measure that
route or prove a dimension-imbalance cause. Real prefix states were not saved,
so actual feature/gradient sensitivity cannot be reconstructed from this run alone.

## Tiny exact learning controls

A two-residue, three-action tree (K/E/A) has nine complete sequences and four
nonterminal states. Use the same 1,536-input, 128-hidden residual head, six context
inputs for one prompt, fixed Gaussian state features and prior action probabilities
(0.1, 0.1, 0.8). The toy reward requests +1 then −1 with charged fraction 1,
scales 1 and beta 2. This deliberately changes the target to one realizable by two
residues; it is **not an IDR experiment or a reproduction of the 64-residue task**.

Run 300 CPU updates, seeds 0/1/2. All conditions share feature values,
initialization and target. Adam normalizer rate is 0.01; head rate is 1e-4 unless
stated. No decay is used in this diagnostic. Exact evaluation computes TV to the
normalized target over all nine sequences; initial TV is 0.6119.

| Toy condition | Final TV, seeds 0 / 1 / 2 | Mean |
|---|---|---:|
| Exact expected TB update | 0.000063 / 0.000061 / 0.000063 | 0.000062 |
| Eight sampled trajectories, clip norm 1 | 0.0607 / 0.0613 / 0.0550 | 0.0590 |
| Eight samples, head rate increased to 1e-3 | 0.1917 / 0.1559 / 0.0686 | 0.1387 |
| Eight samples, no gradient clipping | 0.0344 / 0.0505 / 0.0317 | 0.0389 |

The exact expected update weights each path's TB gradient by its current policy
probability and detaches those weights, matching the expectation of the sampled
update. It does not differentiate through sampling probabilities. It uses privileged
enumeration unavailable for real ProGen; it is a low-variance implementation control.

**Interpretation:** the head/TB mechanism can learn a nontrivial exact target when
updates have low noise. Sampling only eight examples is worse on this fixture;
raising the rate tenfold is worse still. Removing clipping helps this tiny case
but does not establish safety or benefit for real ProGen. These outcomes justify
measuring gradient noise rather than prescribing an arbitrary higher rate.
Finite differences agree with autograd on the TB residual expression to a maximum
absolute error of **2.68e-11**. The previous production rollout tests cover cached
probability alignment and frozen-backbone behavior; this check covers the loss math.

## What changed relative to earlier successful GFN work?

The stability studies sampled a small, finite candidate tree with reusable prefix
features and many forced moves. This pilot has 64 freely generated residues,
changing prefix states on every update, only one trajectory per prompt per update,
and a single terminal score for all 64 decisions. The pilot used 400 updates and
a 1e-4 head rate, reduced to 1e-5 for the last quarter. Reusing the old head/loss
was reasonable for a first check, but convergence on the finite tree did not validate
these choices for open generation. The CPU smoke tests established correctness,
not sufficient learning signal or real-task convergence.

## Recommended next action

Run a **short instrumented learning diagnostic before another full pilot**. Keep
the 64-residue support, target and prefixes unchanged. Capture a small number of
real frozen prefix-state batches, with per-prompt rewards, token corrections,
separate head/normalizer gradient norms and pre/post-clip statistics. Use these
fixed batches to measure how gradient estimates change when averaging multiple
batches, and compare actual first-/second-half correction behavior.

This fills the main evidence gap: the real run did not retain features or gradient
vectors. It avoids using synthetic probes to pick a supposedly optimal rate,
context scale or clip threshold. After that measurement, change **one** factor
(effective batch averaging, explicit context path, or learning schedule), retaining
reward and evaluation. Do not combine larger batches, unfreezing, stronger beta and
SubTB in one run. No new real-training configuration is selected or executed here.

SubTB remains a justified later comparison if noisy long-trajectory updates persist.
The [TB paper](https://arxiv.org/abs/2201.13259) motivates the current objective;
the [SubTB paper](https://arxiv.org/abs/2209.12782) explicitly studies gradient
bias/variance and convergence on longer sequences. These papers motivate questions;
they do not establish the cause of our particular failure or guarantee a remedy.

## Reproducibility

[Diagnostic script](../../scripts/diagnose_idr_charge_cpu.py):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/diagnose_idr_charge_cpu.py \
  --run runs/idr-charge-v1 --output runs/idr-charge-diagnosis-v1
```

This command was already run. Use a new output path for deliberate repetition.
[Saved checks](../../runs/idr-charge-diagnosis-v1/checks.json),
[checkpoint audit](../../runs/idr-charge-diagnosis-v1/checkpoint_audit.json),
[toy results](../../runs/idr-charge-diagnosis-v1/synthetic_learning.json), and input/script
hashes are retained. The diagnostic never loads the pretrained backbone and confirms
CUDA was not initialized. Production source/configs and original outputs are unchanged.
