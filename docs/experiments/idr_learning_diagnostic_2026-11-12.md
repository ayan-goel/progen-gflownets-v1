# Real-state charge-learning diagnostic

**12 November 2026 — completed and analyzed.**
This follows the [saved-checkpoint/CPU diagnosis](idr_charge_diagnosis_2026-11-12.md).
It captures real ProGen states and measures updates. It is not a new sustained
training experiment, a hyperparameter selection run, or biological validation.

## Questions and fixed choices

1. Do eight-sequence gradient estimates disagree, and does averaging more of them
   make their directions more consistent on independent samples?
2. On actual prefix states, does the trained head shift charge positively in the
   first half and negatively in the second? Does its explicit context route respond?
3. Does one inherited Adam update help the head's path probabilities, or mostly
   adjust its normalization estimates?

The task, reward, beta, eight prompts, 16-residue prefix, 64-residue tail, sampling
support and architecture are exactly those of the [completed pilot](idr_charge_2026-11-12.md).
No new charge/disorder reward is added and no metapredict evaluation is needed:
this diagnostic investigates learning signals, not a new biological outcome.

## Source states and capture budget

Use seeds 0/1/2 and two states for each: the exactly reconstructed initialized head
and its saved step-400 head. Reconstruct initialization with the original CPU seed,
head dimensions and zero output projection; initial log Z is zero. The final state
includes the original optimizer moments and post-decay learning rates.

| Setting | Choice and rationale |
|---|---|
| Source | `runs/idr-charge-v1`, pinned configuration and all three final-head checksums |
| Cases | Three seeds × initial/final = six; retain every original seed |
| Microbatch | Eight trajectories, one per prompt; inherits balanced prompt weighting |
| Batches per case | 16; enough for two independent halves and several averaging groups within a small capture budget |
| Proposal half | First eight batches = 64 trajectories |
| Reference half | Last eight batches = 64 separate trajectories from the same frozen policy |
| Averaging groups | 1, 2 and 4 microbatches = 8, 16 and 32 trajectories; 8/4/2 disjoint estimates in the proposal half |
| One-step probes | First 1 and first 4 proposal batches; compare one small versus one averaged update, not a rate sweep |
| Sampling seeds | 80000 + 100 × original seed + stage (initial 0, final 1); separate from pilot and CPU probes |
| Numerical settings | Float32, deterministic algorithms, TF32 disabled, one CPU thread; inherited original model/tokenization |

Total new trajectories: **6 × 16 × 8 = 768**. Each case samples its own current
policy with the original incremental rollout function; the heads remain fixed
throughout capture. Initial/final cases have different trajectories, so their
metric differences are not matched-sequence causal effects. Repeated seeds use
the same biological contexts, not independent protein datasets.

Save each batch's sequences, frozen prefix features, explicit contexts, canonical
logits, sampled actions, prior/policy path probabilities and canonical mass.
Capture approximately 0.4 GB including gradients/checkpoints. At the previous
run's roughly 0.38 seconds per trajectory, backbone capture alone is about five
minutes; loading, hashing and gradient analysis add overhead. This is an estimate,
not a measured duration for the new diagnostic.

## Gradient measurements

Use the same unique-path TB residual and exact charge score as the original pilot.
Differentiate each batch loss with respect to the head and per-prompt log-normalizers,
without updating either. Record:

- Raw head, normalizer and joint gradient norms.
- The inherited norm-one clipping multiplier and resulting component norms.
- Gradients in the hidden layer's frozen-feature, task-context and prompt columns.
- Every full raw gradient vector, with parameter names/shapes in model order.
- Per-prompt charge scores, block charges, charged fractions and charge-pass flags.

For each proposal group, average **raw gradients before clipping**, as for a larger
balanced batch. Compare with the mean raw gradient from the separate reference
half, using cosine, norms and squared distance. Head and normalizer measurements
are separate. Undefined cosine for a zero vector is reported as null.

The reference is an empirical estimate from 64 trajectories, **not the true gradient**.
Larger proposal groups should reduce sampling variation in expectation, but this
small study cannot guarantee monotonic improvement or identify an optimal batch
size. The reference samples are separate within this diagnostic, not an unseen
protein or biological test set. Do not compare proposal gradients to their own
pooled mean and interpret the resulting alignment as independent evidence.

## Actual-state and context measurements

For every captured prefix, compare the complete next-token distribution under the
head with the corresponding frozen prior. Report token KL and the change in expected
nominal residue charge, separately for first/second halves and each prompt.
The desired direction is positive in the first half and negative in the second.
These are conditional token effects at sampled prefixes, not new whole-sequence
success rates or claims that arbitrary alternate states would be reached.

A counterfactual pair holds the **actual frozen features**, accumulated charges,
FCR and prompt identity fixed, setting progress/target to (0,+0.125) versus
(0.5,−0.125). Measure centered-logit and expected-charge differences. This isolates
the explicit progress/target channel; it intentionally may create inconsistent
histories and cannot quantify the information already encoded by ProGen.

Separately record the norms of frozen-feature, five task-context and prompt-identity
contributions to the hidden preactivation, excluding the bias. Norm imbalance is
a descriptive scaling observation, not proof that one channel causes failure.
Initial heads should produce exactly zero policy correction regardless of context;
this is also a diagnostic control.

## Disposable Adam updates

For each case, create separate copies for the first one-batch and first four-batch
proposal groups. Use averaged raw gradients, inherited joint clipping, and the
snapshot's optimizer moments and rates. Initial rates are 1e-4/0.01; final rates
are 1e-5/0.001. These match their source stages; this is not a controlled comparison
of initial versus final learning rates.

Apply **one** update to each copy, then evaluate its TB residual on the disjoint
reference trajectories using saved frozen states. Reusing features is valid because
ProGen and sequence-dependent contexts are unchanged. Record the joint residual,
head-only and normalizer-only contributions algebraically, parameter-update norms,
clip factor and mean absolute change in sequence log probability. All copies are
discarded. Source heads, normalizers, optimizer states and checkpoints remain intact.

The contribution columns decompose the *same joint update*. They are not separately
trained models. This is one selected proposal group per averaging size, not repeated
confirmation that a setting is best. Reference trajectories were sampled before
the update, so lower residual does not establish better generation or biological
properties. No candidate is automatically selected for a full run.

## Execution and recovery

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # select your available GPU
scripts/run.sh idr-diagnose --output runs/idr-learning-diagnostic-v1 --device cuda
```

CPU read-only preflight: `scripts/run.sh idr-diagnose --check-only`.
Capture and head analysis execute on the user-selected device. The run saves
`capture.json`, per-case `head.pt` and `batch-*.pt`, gradients, `metrics.json` and
`report.md`. Capture completion hashes all inputs and records source implementation.
It also retains source provenance, prompt identities and timing/work counts.

If capture completes but analysis fails, analysis can be replayed entirely on CPU,
without loading ProGen or generating more sequences:

```bash
scripts/run.sh analyze-idr-diagnostic \
  --run runs/idr-learning-diagnostic-v1 \
  --output runs/idr-learning-diagnostic-v1-analysis
```

All outputs require new directories. Partial capture cannot be replayed as a
complete diagnostic. Corrupted captures, changed pinned heads/configs and incompatible
analysis code are rejected rather than silently mixed. Original pilot files and
production rollout/training code are unchanged.

## Interpreting the result

- If larger gradient groups align more consistently with the reference and the
  corresponding copied update improves held-aside residuals, consider a controlled
  gradient-accumulation experiment. Keep reward, architecture and support fixed;
  define the compute/update budget before fitting.
- If actual-state correction and context sensitivity remain negligible despite
  consistent gradients, examine the explicit context path or schedule separately.
- If only normalization improves, investigate policy gradient signal and its
  scaling; do not remove learned normalization merely because it is adapting.
- If head updates behave sensibly here but long-run learning still fails, an
  instrumented training curve or controlled TB/SubTB comparison may follow.

No outcomes from this diagnostic have been observed when choosing these settings.
The motivation is the completed failed pilot and CPU controls, not a new literature
claim. [TB/SubTB references and interpretation](idr_charge_diagnosis_2026-11-12.md#recommended-next-action)
remain the source for the optimization questions.

## Verification before handoff

Five new tiny CPU tests check gradient averaging against a combined balanced batch,
zero-output identity, separation of proposal/reference gradients, end-to-end capture
and replay, preservation of source/backbone weights, and tamper rejection. No
pretrained backbone was executed and no GPU was queried or initialized by the
assistant. Real-input preflight passes. The full CPU suite passes all 72 tests
(6.56 seconds), including the five new diagnostic tests.


## Completed results — 12 November 2026

The user completed `runs/idr-learning-diagnostic-v1` in **286.20 seconds
(4 min 46 s)**. All six fixed-policy cases and 768 trajectories completed.
The [generated report](../../runs/idr-learning-diagnostic-v1/report.md) and
[metrics](../../runs/idr-learning-diagnostic-v1/metrics.json) retain full results.
No sustained training or improved-generation experiment occurred.

### What the head actually does

Changes below are expected nominal charge **per next residue**, relative to ProGen
at the captured prefixes, averaged over prompts and positions. They are not whole
sequence charges, success rates, or distances from an exact target distribution.

| Final head | First-half change (desired positive) | Second-half change (desired negative) | Explicit-context charge contrast |
|---|---:|---:|---:|
| Seed 0 | -0.002816 | -0.002515 | +0.000012 |
| Seed 1 | -0.005111 | -0.004900 | +0.000050 |
| Seed 2 | -0.000991 | -0.001284 | +0.000050 |

All three heads push average charge slightly negative in both halves; they have
not learned the intended opposing corrections. Initial heads give exactly zero
correction, as required. Final mean per-token KL from ProGen is only
0.000237 / 0.000356 / 0.000212 nats. These observations reinforce the failed
pilot's generation results rather than demonstrate a new success rate.

The mean hidden preactivation contribution norms are 20.02 / 12.80 / 21.14 for
frozen features versus 0.089 / 0.100 / 0.096 for the five task-context inputs.
Feature contributions are roughly 128–225 times larger in this particular norm.
The explicit progress/target intervention barely changes predicted charge even
on real frozen features. This makes context routing/scaling a plausible limitation,
not a proven cause: different dimensionalities affect norms, ProGen features
already encode history, and counterfactual inputs need not describe reachable states.

### Larger gradient groups are not an established fix

Head-gradient cosine against the separate 64-trajectory reference is weak or
negative. With 8 versus 32 proposal trajectories, initial cases change from
0.145 to 0.308, -0.037 to -0.117, and -0.161 to -0.308; final cases change from
0.118 to 0.160, -0.066 to -0.212, and 0.151 to 0.271. Thus alignment improves in
three of six cases and worsens in three. Mean gradient norms fall in every case,
but smaller norms alone do not establish a better direction. The finite reference
is noisy too; negative cosine does not mean the true optimization direction is
wrong or that larger batches are intrinsically worse.

The one-step head-only residual improves in **3/6 cases at either proposal size**.
The larger proposal improves the head contribution relative to the smaller in
3/6 cases, not consistently. Joint residual decreases in all twelve probes, but
normalizer changes explain much of this: at initialization, normalizer-only MSE
reductions are 0.075 / 0.057 / 0.070, while head contributions can help or hurt.
Final updates are small and inherit the decayed rate and optimizer history.
This is not evidence that learning normalization is incorrect or should be removed.

### Next decision (proposed; not implemented)

Test context routing as one controlled architectural change: retain the current
feature-based residual head and add a small, zero-initialized output branch driven
by the existing task context. Its purpose is to let progress, accumulated charge
and the requested block target affect token logits without first mixing with the
much larger ProGen feature vector. This is our engineering hypothesis motivated
by the measurements above, not a literature-established remedy.

Compare against the unchanged head with matched prompts, seeds, reward, beta,
trajectory budget, optimizer schedule and evaluation. Define the exact branch and
parameter initialization in the implementation protocol before running. Report
actual raw-generation charge and joint disorder success, diversity, and charge
corrections by half; a lower TB loss alone is insufficient. The added branch changes
capacity as well as routing, so a positive outcome would support this intervention,
not uniquely prove feature-scale imbalance caused the original failure.

Do not simultaneously increase batch size, change TB to SubTB, or unfreeze ProGen.
The gradient evidence is mixed and does not select an optimal batch size. If the
controlled context change fails, revisit optimization/credit assignment separately.
The overall objective remains property guidance of generated IDRs, with disorder
assessed separately; no biological-property validation is claimed.

### Verification

CPU-only replay into `runs/idr-learning-diagnostic-v1-analysis` validates every
capture checksum and analysis-source hash and reconstructs all six cases without
loading ProGen. Across 6,774 numeric fields, maximum difference scaled by
`max(1, abs(original))` is 0.000170; maximum absolute difference in the one-step
TB MSE fields is 0.0000211. Small CPU/GPU numerical differences do not change the
reported directions or conclusions. Source-policy immutability and captured path
likelihood checks pass. Original run artifacts and code remain unchanged; the
assistant executed no pretrained model or GPU work.
