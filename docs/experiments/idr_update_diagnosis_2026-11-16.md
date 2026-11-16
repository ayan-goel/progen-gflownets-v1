# Inspecting and repairing learning on the exact IDR task

**16 November 2026 — completed CPU diagnosis and bounded repair.**
The first measurement protocol preceded its audit; optimizer settings preceded
the three control fits; the conditioning check was specified after those outcomes.
The [preceding diagnosis](idr_exact_diagnosis_2026-11-16.md) showed a representable
finite target that the inherited optimizer underfits. Keep the 408 training leaves,
168 cached branching states, saved initialization, locally masked prior, continuous
reward and 128-unit direct-context architecture. No new backbone/scorer execution,
held-out training, full-generation fit or GPU work. All source runs stay untouched.

## First measurements

Measure shared-input/hidden-activation concentration, initial gradient norms by
parameter group, and the cosine between positive/negative alternating-replay
expected gradients. These can expose state coupling or stratum oscillation; neither
correlation nor a large norm by itself proves causality. Compare exact teacher
and TB gradients without ranking raw loss scales across objectives.

For a stable line probe, fix a full-support per-prompt leaf weighting to one-half
the initial policy, one-quarter uniform successes and one-quarter uniform failures.
This requires only the known finite support, saved policy and terminal rewards,
not oracle next-token targets. Its squared TB residual has the same zero-residual
solution as production TB. It is a different training distribution, explicitly
introduced to inspect a deterministic objective. At initialization, inspect loss
and TV at 0, 0.25, 1, 4 and 16 times one inherited Adam update. These are read-only
copies to describe local behavior, not a selected training-rate sweep or evidence
that the best probe scale is safe over a run. Record the normalizer/head update
contributions and full-support normalization.

After these measurements, specify one bounded optimizer comparison if justified.
Keep target, support and initial policy unchanged, and include a control with the
same deterministic weighting to separate optimizer from sampling changes. Do not
infer an open-generation fix from a finite deterministic fit. Record actual
function/gradient evaluations and preserve final-iterate results, not best TV.

## Initial audit and selected comparison (before fitting)

The [initial audit](../../runs/idr-update-audit-v1/audit.json) finds 96.69% of
input energy in the state-shared mean and median pairwise cosine 0.9814. Hidden
activation values have 84.83% mean energy and cosine 0.9553. This motivates
examining conditioning, not concluding that features contain no information.
Even/odd replay gradients have positive cosine 0.2342; they differ substantially
but are not globally opposite at initialization. The dominant gradient norm is
in output weights (3.724), followed by hidden weights (0.952); normalizers are
0.252. The global norm is 3.858. Thus normalizer domination is not supported here.

One Adam step reduces fixed TB loss 1.2393 → 1.2042 and mean TV 0.32547 → 0.32313.
Four times the step reduces loss more but barely improves TV; sixteen times
increases both loss (1.5759) and TV (0.37447). This is not evidence for a globally
useful larger learning rate. Correlated inputs and this directional sensitivity
motivate a deterministic curvature/line-search diagnostic.

Use three fits, with **at most 400 loss/gradient evaluations each**:

- **Fixed-weight Adam:** inherited rates, decay after update 300, global clip 1.
  This controls the change from alternating/on-policy weights to the same fixed
  full-support weights used in the candidate.
- **Fixed-weight Adam without clipping:** identical schedule and weights; isolate
  whether removing clipping alone explains a gain. This is a diagnostic control,
  not a proposed production change.
- **Fixed-weight L-BFGS TB:** same starting head/normalizers, terminal reward and
  fixed TB objective. PyTorch 2.6 implementation, LR 1, history 20, strong-Wolfe
  line search, gradient tolerance 1e-7 and parameter/loss tolerance 1e-9. Use raw
  gradients because line search must see derivatives of the evaluated loss.
  All parameters share one group; there are no inherited per-group Adam rates.
  This is an optimizer package comparison, not proof that one internal component
  (curvature, line search, parameter scaling or clipping) alone causes the result.

These are fixed engineering choices, not optimized on outcomes. History 20 bounds
memory; LR 1 and tolerances follow the inspected API defaults. Line-search calls
count against the budget, including rejected proposals. Each outer call permits
one quasi-Newton iteration; if budget exhaustion interrupts a trial, restore the
last accepted parameter vector and stop. Report the final accepted iterate and
actual gradient counts. No best-TV checkpoint, second seed or extension.

A candidate passes this local diagnostic if mean **and maximum prompt TV < 0.05**
and mean joint success is within 0.02 of the exact target. This is a finite-task
engineering gate, not statistical significance or authorization for full generation.
Do not deploy a deterministic full-batch solver directly into noisy on-policy
training. If it fits, the result establishes learned fitting with a different
optimizer; it still does not isolate the unrestricted generation failure.

Sources: [official PyTorch L-BFGS API](https://docs.pytorch.org/docs/2.15/generated/torch.optim.LBFGS.html)
(the accessible current manual; the archived 2.6 manual was inaccessible)
and [version-pinned source](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/lbfgs.py)
and the installed `torch/optim/lbfgs.py`, version 2.6.0+cu124, SHA256
`f181e9224a72596f0f9027cfc42c0bf3819fd136828aef0fdc002a4e94495367`.
The implementation was inspected for closure/line-search evaluation accounting.
This is an implementation-motivated diagnostic, not a literature claim that L-BFGS
solves protein GFlowNet generation. Keep existing reward/local-credit reviews intact.

## First fit outcomes and post-hoc conditioning check

The fixed-weight comparison completed without exceeding 400 gradient evaluations
per arm. Mean TV is 0.18783 for clipped Adam, 0.21397 for unclipped Adam and
0.10796 for L-BFGS; all fail the local gate. Removing clipping alone is not supported
as a fix. L-BFGS improves the same fixed objective, but does not fit closely enough.
This is not permission for extra iterations or optimizer hyperparameter tuning.

The initial-state geometry remains a concrete separate hypothesis. Before further
fitting, specify **one function-preserving input reparameterization**, then test it
with the same bounded L-BFGS procedure from the same initial policy. This additional
check is post hoc, motivated by the failed unconditioned solver; it is not part of
the original three-arm comparison and is not selected from a transform sweep.

Form the matrix `X` of normalized features, context and an affine constant for the
168 training branching states. Its thin SVD is `X=U diag(s) V^T`, where `s` are
positive singular values and `V` contains the corresponding orthonormal input
directions. Define the invertible input transform
`P=I+V diag(sqrt(N)/s - 1) V^T`, with `N=168` states and `I` the identity.
Then `X P` has equal singular values on the observed row span; directions outside
that span remain unchanged. This is our deterministic linear-algebra construction,
not a literature-selected protein method. It uses no reward/target labels.

Replace the original hidden affine weights `W` (including bias as their last
column) with `W P^-1`, and feed transformed inputs into that layer. All other
weights are unchanged. The original function and model class are preserved
algebraically; check initial policies and arbitrary-input logits numerically before
fitting. Keep LayerNorm trainable as before. The transform is fixed, not learned,
and changes optimizer coordinates, not the terminal target or trainable parameter
count. It may amplify weak/off-distribution feature directions; finite-state
interpolation is not a production generalization claim. No state IDs are inputs.

Use float64 SVD, fixed rank cutoff 1e-12 to reject deficient input matrices, and
float32 forward/parameters matching controls. No scale tuning or truncation after
seeing outcomes. Same fixed objective, optimizer settings, 400-gradient budget and
local gate as the unconditioned L-BFGS control. Fold final transformed weights back
into the original head and verify equivalent output. A pass supports conditioning
as a tractable finite-task bottleneck; a failure ends this bounded conditioning
check. Neither outcome authorizes switching the full generator to this solver.


## Completed conditioning result

The [conditioned fit](../../runs/idr-exact-conditioned-v1/results.json) passes
both mean and worst-prompt TV gates. All fits below use the **same fixed TB loss,
terminal reward, selected support and initial policy**, with 400 loss/gradient
evaluations each. Comparisons across optimizers include their declared differences
in parameter rates, clipping and update rule; the last two rows isolate the input
coordinate change under the same L-BFGS setup.

| Method | Mean TV ↓ | Worst-prompt TV ↓ | Exact joint success on this tree | Fixed TB loss |
|---|---:|---:|---:|---:|
| Adam, inherited clipping | 0.18783 | 0.25943 | 47.79% | 0.36248 |
| Adam, no clipping | 0.21397 | 0.27503 | 45.10% | 0.43902 |
| L-BFGS, original coordinates | 0.10796 | 0.16460 | 58.33% | 0.12598 |
| L-BFGS, conditioned coordinates | **0.00163** | **0.00223** | **62.04%** | **0.0000315** |
| Exact target | 0 | 0 | 62.02% | 0 |

TV is half the sum of absolute differences in leaf probabilities. All eight
conditioned prompt TVs are below 0.0023. Mean success differs from the exact target
by +0.0202 percentage points. This is trained TB using terminal rewards; oracle
next-token targets and the previous algebraic solution are not used in fitting.
The selected bank itself is outcome-dependent; saying the transform is label-blind
means conditional on that bank, not that bank selection was reward-independent.

Conditioned L-BFGS uses 194 accepted updates versus 197 unconditioned, both within
400 gradient evaluations; rejected line-search trials also count. Neither uses
extra steps, the best intermediate TV or a second hyperparameter attempt. The
CPU fitting times are approximately 3.1 s / 3.1 s / 3.9 s / 6.0 s respectively,
excluding source loading/audits. This is a tiny cached-state diagnosis, not a real
protein-generation training run. No ProGen or disorder model executed.

The initial [equivalence check](../../runs/idr-exact-conditioned-v1/initialization.json)
finds maximum per-prompt policy TV 4.08e-6 versus the saved initial head, cached
logit difference 6.74e-6 and random-input logit difference 1.73e-6. Thus the gain
is not an advantageous new initialization. The input matrix condition number
changes from 2,937.14 to 1 in double precision. Direction scales span 0.0258–75.67;
this large amplification is a reason to test unseen states before production use.
The transform is fixed from initial states; later LayerNorm updates do not imply
that subsequent input covariance stays exactly white.

Folding the trained weights into an ordinary production-class head preserves the
final policy to maximum prompt TV 3.39e-6. No extra trainable parameters or state
identity inputs were added. The final folded checkpoint is marked diagnostic-only
and has not replaced any generation checkpoint.

## What this establishes and the next boundary

**We found and repaired a learning failure on the reduced task.** The same TB
objective can fit this target closely when the input coordinates and optimizer
make distinctions among similar cached states easier to learn. This is stronger
than the prior algebraic capacity witness: the weights were learned from the TB
loss, not calculated from exact desired token probabilities.

Do not describe this as proof that correlated features are the sole cause of the
full-generation failures. The strongest controlled conclusion is that this invertible
reparameterization greatly improves the fixed-task L-BFGS fit. We have not shown
that conditioning alone fixes Adam, that L-BFGS works with noisy rolling trajectories,
or that this finite-state transform generalizes. Removing clipping alone was worse.
Changing to a fixed training distribution also helped before the optimizer change;
that result bundles frozen proposal weights and removal of alternating strata.

**Next proposed check:** retain this tiny support, reward and shared start, then
compare ordinary and conditioned coordinates under *sampled* TB with a matched
stochastic optimizer. This asks whether the useful conditioning transfers when
exact full-batch updates are unavailable. Do not use L-BFGS line search directly
on changing minibatches. Define that comparison separately; no additional fit was
run after the local gate passed. If stochastic fitting works, move to a short
fully branching, user-captured ProGen task before trying 64-residue generation.
No new GPU command or production change is warranted yet.

## Verification and reproduction

Ten [CPU tests](../verification/idr-update-diagnostic-tests-2026-11-16.txt) cover
the prior exact task plus deterministic TB gradients, clipping controls, strict
line-search budget accounting/rollback, source-weight immutability, invertible
function preservation, unchanged parameter count, folding after weight changes,
and independence of the transform from reward labels.

The [saved-weight reconstruction](../../scripts/verify_idr_optimizer_diagnostic.py)
exactly reproduces final per-prompt metrics and fixed TB losses for all four fits,
checks captured source hashes, and verifies the folded checkpoint. Its
[audit](../verification/idr-update-result-audit-2026-11-16.json) records artifact
hashes and confirms CUDA remained uninitialized. Original source-run manifests
verify before and after analyses. No full model/scorer or GPU execution occurred.

The standalone scripts preserve hash-pinned production modules. Reproduction uses
fresh output paths; these are CPU-only commands, not additional recommended runs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/diagnose_idr_updates_cpu.py \
  --output runs/idr-update-audit-reproduction
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/fit_idr_exact_optimizer_cpu.py \
  --output runs/idr-exact-optimizer-reproduction
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/condition_idr_exact_cpu.py \
  --output runs/idr-exact-conditioned-reproduction
```

The fitted runs capture their analysis source files and hashes. Reproduction of
current saved artifacts without fitting uses:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/verify_idr_optimizer_diagnostic.py \
  --output /tmp/idr-update-verification.json
```
