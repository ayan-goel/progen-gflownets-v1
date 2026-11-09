# Implementation decisions

Substantive implementation choices and the problems they address. Scientific
rationale and parameter choices are in the [decision record](decisions.md);
protocols and outcomes are in the [experiment index](experiments/README.md).

## 2026-10-14 — Finite sequence environment and frozen backbone

Represent each candidate set as a prefix tree. Every sequence has one construction
path, constraints exclude dead ends, and forced suffix residues remain explicit.
This enables exact target/learned-distribution comparisons before attempting larger
spaces. The four-position toy has 85 internal prefixes and 256 leaves. Cache frozen
prefix features once and charge that work to setup; this does not establish
scalability beyond enumerable supports.

Reuse the working Python/ProGen package versions and vendor upstream model code,
tokenizer and license to preserve checkpoint compatibility. Use the literal `1`
direction marker, float32 logits, causal prefix features for guidance and pooled
complete-sequence features for the stability predictor. Tiny CPU fixtures use
64 hidden dimensions and eight heads because upstream attention packs eight ways.
Cached/full-forward and shared-prefix comparisons verify causal alignment.
See [architecture](architecture.md) and [backbone decisions](decisions.md#2026-10-14--backbone-and-feature-definitions).

## 2026-10-14 — Guidance and comparison methods

Initialize the residual head's output to zero so generation starts exactly at the
locally constrained prior. Use log-space trajectory balance with a learned log
normalizer. Implement Monte Carlo future-reward supervision as the cheaper guidance
comparison, with the same frozen backbone and reward. Verify zero-guidance,
beta-zero, forced-action and likelihood identities before real training.

Save frozen numerical inputs and final heads so evaluation can be reproduced on
CPU without reloading ProGen or refitting. Preserve raw proposals and selection
indices to distinguish generation from selection. Use immutable run directories,
source snapshots and hashes to keep comparisons reproducible.
See [learning mechanics](decisions.md#2026-10-14--learning-mechanics).

## 2026-10-14 — Stability data and predictor

Separate candidate identities from held-out labels so support construction need
not consume evaluation outcomes. Apply canonical-sequence and label filters,
resolve reference identities, average sequence-level replicates and preserve
seeded 70/15/15 splits. The real release audit selected 1O6X by coverage, yielding
950/204/204 train/validation/test sequences. See [data](data.md).

Fit scaling on training data only. Compare ridge and a small MLP, select by
validation, and require the predictor to beat a training-mean baseline and meet
the declared ranking screen before guidance. Construct rewards from frozen
predicted improvement relative to the reference. Biological smoke tests require
explicitly synthetic data; screen failure blocks guidance and held-out evaluation.
See [scorer/data decisions](decisions.md#2026-10-14--mega-scale-preprocessing-and-scorer).

## 2026-10-14 — Diagnose exact-task optimization before changing capacity

Initial exact-task and privileged conditional fits missed the TV criterion.
Independent leaf-mass calculations and a prefix-depth KL decomposition localized
most error at length-two prefixes. Add conditional-fit diagnostics and gradient,
TV/KL and entropy-floor logging to distinguish insufficient optimization from
limited head capacity. Preserve the architecture and compare longer fits at two
rates. The longer original-rate conditional fit passed, motivating a longer TB run
rather than a larger model. See [diagnostics](experiments/exact_diagnostics_2026-10-14.md).

Longer constant-rate TB passed only one seed. Add a deterministic late rate drop
and observational exact-distribution monitoring to test late settling. Monitoring
must not change RNG, updates or checkpoint selection, and its cost is separate.
CPU comparisons establish parity with the original constant-rate implementation.
The scheduled run passed two seeds; extend the low-rate phase to 20,000 total
updates while preserving the drop after 8,000 and the final-checkpoint criterion.
See [scheduled experiment](experiments/exact_decay_2026-10-14.md).

## 2026-10-15 — Transfer the verified schedule to biology

All three final exact-task seeds passed. Transfer the tested schedule to both
biological guidance learners, require matching checkpoint hashes and remove
training-time exact monitoring from the biological run. Keep predictor validation
before guidance. This verifies mechanics before biology without treating toy
convergence as evidence of improved stability.
See [gate and pilot setup](experiments/exact_pass_and_biological_pilot_2026-10-15.md).


## 2026-11-03 — Matched predictor and exact-reward diagnostics

Add an optional family selector to the existing scorer fitter; default combined
selection remains unchanged. The new study calls ridge and MLP separately so
both receive downstream evaluation, reusing normalization and optimization code.
Extract complete-sequence features once (the original prefix cache cannot serve
as pooled predictor features), preserve them, and reuse the original finite prior.
An independent saved-run audit reconstructs score arithmetic and reports
probability-shift contributions to measured utility. No GFN training changes.
See the [study protocol](experiments/predictor_reward_study_2026-11-03.md).


## 2026-11-09 — Reuse cached ProGen inputs with the frozen ridge reward

A dedicated ridge-guidance runner verifies the original cache and saved full-data
ridge artifacts, reconstructs predictor outputs without fitting, and replaces
only the task's scores/predictions. Original inputs and historical heads remain
unchanged. Existing TB training, progress reporting and sampling evaluation are
reused. The evaluator accepts an explicit method list so this run needs no
unrelated supervised-head fit. Default evaluation still includes both original
learners. A final exact report distinguishes predicted from measured expectations
and compares the trained policy with its target and historical MLP policies.

CPU checks exercise the complete synthetic training/evaluation/reload path,
score substitution with fixed cache tensors, stale input rejection and output
immutability. All 30 tests pass. See the [protocol](experiments/ridge_gfn_2026-11-09.md).


## 2026-11-09 — Fixed reward-noise panel

Add a dedicated runner around the existing TB trainer and frozen ridge task.
A local CPU RNG produces immutable per-sequence noise, independent of training
RNG. Store the clean task once and only noise tensors/heads per condition; preserve
clean controls, source and input hashes. Existing optimizer code is unchanged.
Evaluate full-support distributions after all fits, distinguishing clean-target
shift from fitting gap to the noisy target. Exact evaluation avoids introducing
sample/selection noise into this diagnostic. Reevaluate saved heads via a separate
CPU command. Synthetic checks cover paired perturbations, zero-noise identity,
RNG isolation, invalid inputs, crossed training/evaluation, decomposition identity,
replay equality and immutable source artifacts; all 33 tests pass.
