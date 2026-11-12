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


## 2026-11-09 — Reuse pooled features, rebuild candidate-support probabilities

The split-study runner verifies and reassembles the original 1,358 frozen pooled
features by identity. It reproduces original seed-42 membership as a provenance
check, then saves label-independent new partitions. Existing prefix-tree caching
and predictor comparison functions provide the new priors and separate-family
fits. No generation/training probability conventions change. A summary adds
prior-weighted errors and compares gains within support; per-split predictions,
validation selection history and exact probabilities remain inspectable.
Synthetic end-to-end and integrity tests pass; real experiment remains user-run.


## 2026-11-09 — Position-group partitions reuse the local comparison pipeline

Add an explicit position split mode to split_study.py and a position-study
command with its own fixed config/protocol. A sequence-only grouping function
rejects ambiguous multi-mutants/indels, keeps groups intact and preserves the
reference anchor. The original sequence-split mode remains the default. All
partitions and one-based position lists are saved before model execution. Reuse
the frozen-prior and predictor code unchanged, allowing each split's actual
training count instead of forcing 950. Add top-one probability, success and
expected diversity to the main report so a concentrated prior is visible.
Synthetic checks cover disjointness, ordering, complete coverage, invalid inputs,
end-to-end fits, report reconstruction and source immutability; all 40 tests pass.


## 2026-11-09 — Reuse grouped rewards and the existing TB learner

position_guidance.py connects verified grouped-position artifacts to the existing
prefix cache and train_tb implementations. CPU preflight reproduces partitions,
prior probabilities and saved ridge predictions; pinned input hashes prevent
silent changes. Prefix representations must be recomputed because the previous
study retained logits only. Compare new logits to saved ones, keep the saved prior,
and check float32-score target rounding before training. Save tasks/heads and
separate post-training labels so CPU reevaluation is independent of ProGen and
the source study. Exact MLP probabilities remain a comparator, not a fitted head.
Three synthetic tests cover crossed fitting/replay and label separation, source
hash/CPU guards and frozen-backbone mismatch rejection. All 43 CPU tests pass.


## 2026-11-10 — Grouped multi-protein data and frozen-feature comparison

multiprotein_data.py reuses the release filters/replicate policy, builds transitive
parent components, excludes documented exposed identities and freezes a bounded
panel. multiprotein.py batches variable-length sequences by length while preserving
row order, caches frozen features once and reuses the existing scorer fitting.
Evaluation labels are parsed only after all model checkpoints and predictions are
saved. Metrics distinguish across-parent prediction from within-parent ranking;
cutoff ties are averaged so constant predictions cannot exploit row ordering.

Ridge now solves the smaller primal or dual system, preserving its objective and
the original dual path for local pilots while avoiding a large row-by-row kernel
for 2,860 training examples. An explicit equivalence test covers this change.
Biopython 1.86 was already installed and is pinned for reproducible alignment.
All 47 CPU tests pass, including grouping/exposure/label independence, mixed-length
features, synthetic complete execution, exact metric replay and input guards.

## 2026-11-10 — Cached-feature grouped regularization comparison

regularization.py reuses fit_scorer with one fixed penalty per fit, preserving
the existing ridge objective and solver. Its selection function receives only
original training rows; deterministic whole-group folds recompute normalization
and constant baselines separately. OOF predictions select a strength before a
single full-data refit and fixed-validation comparison. Reuse the original saved
ridge rather than silently replacing the comparator. Source/data hashes and row
identities are verified, and all outputs go to a new directory. Synthetic checks
cover validation-label perturbation invariance, fold-local normalization and
constant predictions, checkpoint replay, and input/CPU/output guards. The
[protocol](experiments/regularization_2026-11-10.md) specifies scientific choices.

## 2026-11-10 — Nested training coverage without changing evaluation support

coverage_data.py consumes the pinned original component inventory and selects
unused groups before processing measurements. audit_table gains an optional parent
filter before label parsing; existing callers retain identical behavior.
coverage.py extracts only added training features and reuses the fixed-penalty
ridge solver. It saves every stage before validation evaluation and requires the
44-parent refit to reproduce original predictions. New rows, memberships and
checkpoints live in separate artifacts. Tests exercise deterministic preparation,
group/sequence boundaries, training-only fitting, constant per-example penalty,
missing test labels, saved-model replay and original-baseline reproduction.


## 2026-11-12 — IDR baseline sampling and prefix-safe evaluation

Added a separate `stability_gfn.idr` package, reusing the frozen ProGen tokenizer
and incremental decoder rather than modifying the completed stability trainer.
Canonical masking, fixed-length stopping and sampled-token likelihoods are explicit.
A matched tail region prevents the supplied natural prefix from inflating the main
success score; isolated-tail scoring checks contextual dependence. No deduplication
or score filtering changes raw sample denominators.

Prepared references retain source annotation/coordinate provenance and are pinned
by hashes. Runs snapshot data, record package/code/weight provenance, flush samples
batchwise and mark failures. Saved-sample reevaluation requires a completed,
checksum-matching generation stage and matching scorer files; it does not load ProGen.
The tiny CPU integration test uses actual V3/V2 scorers and a random tiny ProGen,
not the pretrained checkpoint. See the [protocol](experiments/idr_baseline_2026-11-12.md).

## 2026-11-12 — Open continuation guidance without a sequence-tree cache

The stability implementation cached an enumerable mutation tree. Open 64-residue
continuations cannot use that representation. The new `idr/charge.py` samples
on-policy with the existing frozen ProGen KV adapter, retaining detached hidden
states, canonical logits and actions for only one update. Recomputing the head's
path likelihood on those states is equivalent to a fresh backbone pass because
backbone weights are fixed; CPU teacher-forcing comparisons check token alignment.
Only head parameters and eight prompt-specific log-normalizers receive gradients.

`idr/charge_experiment.py` owns orchestration and evaluation, reusing baseline
reference validation and disorder/descriptor scoring. Generation and charge-only
selection are kept separate; selection does not inherit a misleading output-policy
density from its proposal. Swaps/shuffles retain source identities and omit
uncomputed transformed-sequence likelihoods. Generation completion hashes allow
CPU evaluation recovery without ProGen; output directories cannot be reused.
Per-seed and per-prompt reports expose collapse hidden by pooled metrics. The
[charge protocol](experiments/idr_charge_2026-11-12.md) owns the settings and budgets.


## 2026-11-12 — Reusable real-state diagnostic captures

`idr/learning_diagnostic.py` reuses the unchanged pilot rollout and path-likelihood
functions. It snapshots each head on CPU, stores detached balanced feature batches,
and computes head/normalizer gradients separately. Gradient averages equal the
combined-batch derivative; clipping occurs after averaging for the disposable
Adam probes. Source heads/normalizers remain unchanged, with CPU tests checking
file and weight preservation.

Capture hashes cover config, references, head states and all batches; analysis
source hashes prevent silently reinterpreting captures under different code. The
CLI provides a separate CPU replay route. No historical pilot code is modified,
so its provenance checks continue to pass. See the
[diagnostic protocol](experiments/idr_learning_diagnostic_2026-11-12.md) for what the
measurements mean and which comparisons are deliberately not causal training tests.


## 2026-11-12 — Paired context-path experiment with preserved pilot code

`idr/context_experiment.py` subclasses the original guidance head with a zero
20-by-5 Parameter, avoiding extra RNG draws and keeping every shared parameter
identically initialized. The paired runner calls the original rollout, TB update
and scorer directly; historical modules are unchanged. The tiny fixture verifies
that the standard arm reproduces the original trainer's weights and samples.

Generate one shared prior pool per seed, use independent named generators for
training/evaluation, and retain detached evaluation states only long enough to
measure block corrections. Save architecture-tagged checkpoints, stepwise losses,
raw samples, token probes, costs and source/model/scorer provenance. CPU-only
replay validates saved input hashes and scorer/source compatibility before scoring;
no ProGen load is needed. Both complete and partial runs refuse overwriting.
The [protocol](experiments/idr_context_2026-11-12.md) owns the research rationale
and fixed experimental choices.
