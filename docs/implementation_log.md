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


## 2026-11-12 — Isolated task-matrix rate and self-contained control reuse

`idr/branch_rate.py` reuses `DirectContextHead`, rollout, TB updates, final sampling
and charge/disorder scoring while leaving prior experiment modules unchanged.
Three disjoint Adam groups isolate the matrix's multiplier; clipping still acts
jointly and the decay boundary matches the original. Tiny CPU parity at multiplier
one checks that the separate group and loop do not change the original algorithm.

The runner pins the completed comparison manifest and metric files, verifies
references/packages/source/scorer/model identity, and copies controls needed for
analysis. It records zero new prior generation and retains original costs as
reused evidence. CPU evaluation checks shared control counts/pass rates exactly (continuous metrics
allow 1e-6 relative / 1e-7 absolute scorer roundoff), and can be
replayed after the source directory is unavailable. Training logs/checkpoints name
all rates and the multiplier, preventing a base-config rate from obscuring the
matrix's actual update setting. The [protocol](experiments/idr_branch_rate_2026-11-12.md)
owns rationale, budgets and the bounded continuation rule.


## 2026-11-13 — Prior-factored SubTB with a matched frozen-policy comparator

New `idr/subtb.py` computes all partial-trajectory residuals from cumulative
policy/prior log-ratios. Cached pre-action features index interior states; root
log-normalizers and terminal beta-times-score replace network outputs at the two
boundaries. This preserves the original terminal target and avoids extra ProGen
passes. A separate zero-output scalar MLP learns interior flows; independent
initialization cannot consume the token policy or sampling RNG stream.

`idr/subtb_experiment.py` reuses historical rollout, TB optimizer/update and
charge/disorder evaluation unchanged. Both arms get fresh identical policy
initialization, matched budgets and new shared prior controls. Artifacts record
per-group rates, both objective and common TB residuals, flow/policy gradients,
raw final sequences, token effects, checkpoints, learning curves and computation
counts. CPU replay checks provenance and works without the source run or model.
The [protocol](experiments/idr_subtb_2026-11-13.md) owns hyperparameters and rationale.
Exact-tree value/gradient tests and tiny optimization protect the changed loss;
a random CPU-backbone fixture verifies the complete run/replay and freezing.

## 13 November 2026 — Versioned joint-acceptance scorer and offline audit

`idr/acceptance_reward.py` computes a bounded property log factor from complete
tails and supplied V3 residue scores, exposing each constraint violation and pass
flag. It runs no predictor and detaches rewards from autograd. Separating it from
historical charge modules preserves completed run hashes and avoids silently
changing old replay/evaluation semantics. Its output replaces the old beta-times-
score term in TB; tests reject the conceptual error of applying beta 2 again.

`scripts/audit_idr_acceptance_reward.py` pins the already inspected prior pool,
checks recomputed flags, uses proposal-correct importance weights, and reports
per-group coverage plus a paired within-group bootstrap. A proximity-only ablation
checks whether the acceptance step is consequential. These are saved-data checks,
not real fitting. The [reward record](experiments/idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1)
owns the design and interpretation; integration into a training runner is pending.

## 13 November 2026 — Integrate acceptance scoring into TB and recoverable evaluation

`idr/acceptance_experiment.py` reuses frozen rollout, the direct-context head and
the three-group Adam schedule. It scores completed tails on CPU, passes the
detached log property factor to TB with beta 1, and saves per-sequence reward
inputs alongside update logs. Completed TB checkpoints are copied and verified;
fresh samples from those heads provide the old-reward comparator without refitting.
Historical implementations/configurations are unchanged.

`idr/acceptance_evaluation.py` computes raw outcomes, new-reward selection,
prompt-balanced importance estimates with coverage diagnostics, and each policy's
residual under its own reward. Exact pairwise mean Hamming distance uses residue
counts rather than a quadratic pair loop, avoiding unnecessary evaluation cost
for the prior pool. A pre-scoring generation manifest supports CPU-only recovery
without the source run or pretrained model. The [protocol](experiments/idr_acceptance_2026-11-13.md)
owns the experimental choices and limitations; no real training was run here.

## 13 November 2026 — Replay bank, matched TB control and supervised capability check

The new `idr/replay_data.py` selects only prior training trajectories, partitions
identical tails together and caches causally aligned teacher-forced frozen features.
Cached float32 states let repeated head updates avoid repeated ProGen execution;
prior-probability agreement and a tiny incremental-versus-teacher-forced test guard
alignment. Saved behavior likelihoods are provenance, never the current-policy
likelihood used in a replay loss. The fixed bank and fresh rollouts keep the first
intervention limited to training-example reuse rather than buffer-refresh design.

`idr/replay_experiment.py` fits fresh on-policy TB, mixed replay TB and supervised
heads with equal updates/trajectory-example counts. Reusing the existing head,
optimizer grouping, rate schedule and TB update preserves their definitions.
The supervised arm removes the normalizer optimizer group and uses mean token NLL.
Per-trajectory IDs and fresh reward inputs expose actual training membership and
copying. Initial/final fixed-bank fits separate successful and failed examples;
fresh encounters of reserved tails invalidate those holdout rows explicitly.

`idr/replay_evaluation.py` evaluates unfiltered generation, common-bank similarity,
actual training copies, per-prompt success and fit diagnostics. A captured snapshot
supports CPU scoring recovery even after removing the source run/data from view.
Historical scientific modules/configurations remain unchanged; new CLI commands
select this separate protocol. [Protocol and rationale](experiments/idr_replay_2026-11-13.md).

## 2026-11-15 — Supervised-start continuation with preserved source artifacts

New `warmstart_data`, `warmstart_experiment` and `warmstart_evaluation` modules
implement the [protocol](experiments/idr_warmstart_2026-11-15.md) while reusing the
existing head updates, sampling and metric primitives. Historical source modules
stay unchanged because completed manifests pin their hashes. The new driver loads
the same source weights independently for every arm and reuses its frozen-feature
cache; it does not refit or reconstruct the original bank.

Capture original supervised training IDs along with the initial checkpoint so
memorization checks include pre-continuation exposure. Fit diagnostics exclude
online-exposed holdout tails across all arms/stages, and the relative-target spread
is calculated within each prompt rather than across different normalizers. Final
scoring recovery uses a self-contained manifest without source/data directories.
The training budget, optimizer reset and normalizer rationale live in the protocol.


## 15 November 2026 — Versioned continuous-reward TB comparison

Added separate scorer, input validation, training, evaluation and driver modules
under `idr/continuous_*`. Historical source modules are hash-pinned by completed
runs, so their functions are reused without modifying them or introducing runtime
patching. The new scorer reuses the exact existing violations and acceptance labels,
then replaces the terminal log factor. Cached replay features and row identities
stay unchanged; only candidate training factors and normalizer targets are relabeled.

The driver verifies the matched control's initial weights/settings and reuses its
checkpoint. Evaluation asserts replay order/rate equality, includes inherited
training in copy checks, and compares each policy against both reward definitions
on a common bank excluding all exposed diagnostic tails. Its snapshot supports
CPU-only evaluation recovery without original sources/data. The [protocol](experiments/idr_continuous_2026-11-15.md)
owns scientific choices; [verification](verification.md#continuous-reward-gflownet--15-november-2026)
records tests and preflight.


## 2026-11-16 — Exact cached-state diagnostic and capacity witness

Two standalone CPU scripts reuse saved states, the production head, reward and
optimizer without changing hash-pinned production modules. A prefix trie removes
forced edges from computation and checks shared cached states; exact leaf sums
and an independent tabular oracle verify normalization. Exact and sampled TB use
the same alternating replay strata. A separate post-hoc SiLU-inversion/linear-solve
construction tests finite-state representability without changing the architecture
or relying on gradient fitting. It is never deployed. The
[protocol/results](experiments/idr_exact_diagnosis_2026-11-16.md) own settings,
conditioning distinctions, controls and the limits of this deliberately easier task.


## 2026-11-16 — Controlled update audit and input-coordinate repair

Standalone CPU scripts inspect exact-task gradients and compare fixed-weight TB
optimizers while preserving production modules and prior outputs. Closure calls
count against a hard gradient budget; interrupted line-search trials restore the
last accepted weights. The subsequent input transform reparameterizes the hidden
affine layer invertibly, including its bias, so model class, parameter count and
initial function are preserved. It uses feature geometry only, and final weights
fold back into the ordinary head for a numerical equivalence check. All resulting
checkpoints are diagnostic-only. The [protocol/results](experiments/idr_update_diagnosis_2026-11-16.md)
own choices, reasoning and the distinction from noisy open-generation training.


## 2026-11-16 — Matched sampled conditioning diagnostic

A standalone CPU script reuses the existing finite trees, conditioning transform
and Adam/replay helpers, preserving hash-pinned production code. Each fit starts
from the original supervised weights with fresh optimizer state. Separate paired
RNGs preserve replay draws while allowing current-policy samples to diverge.
Checkpoints, sample indices and source/protocol snapshots support reconstruction;
all saved heads remain diagnostic-only. Tests verify legacy-control equivalence,
residual-only differentiation and no training dependence on exact target weights.
[Protocol/results](experiments/idr_sampled_conditioning_2026-11-16.md) own settings,
partial improvement, failure gate and next-control rationale.


## 2026-11-16 — Expected sampled-TB gradient control

A standalone CPU script averages squared residual gradients with detached normalized
current-policy weights and the unchanged alternating replay stratum. It applies
clipping and Adam afterward; it does not differentiate through proposal weights or
train against oracle next-token targets. Independent sample-pair enumeration tests
that distinction. Source snapshots and reconstructed conditioned/folded checkpoints
preserve reproducibility without editing pinned production modules.
[Protocol/results](experiments/idr_expected_conditioning_2026-11-16.md) own the
fixed budget, local pass and practical sampling follow-up.


## 2026-11-16 — Sample averaging before the optimizer update

A standalone cached-state diagnostic draws 16 fresh and 16 replay leaves per
prompt with replacement, averages residual losses, and clips/steps once per
aggregate batch. Duplicate multiplicity is retained. Separate RNG streams and
saved indices support reconstruction; one-draw mode exactly matches the prior
trainer in a fixture test. Source snapshots and input/control hashes protect
completed experiments. [Protocol/results](experiments/idr_averaged_sampling_2026-11-16.md)
own the bounded budget, additional sample cost and limits of the partial gain.


## 2026-11-16 — Variance attribution and corrected fixed-bank TB

The audit enumerates raw leaf gradients in actual conditioned coordinates and
combines fresh/replay covariance traces using squared prompt/group weights.
A separate trainer uses a fixed full-support proposal, refreshing a snapshot every
20 updates; current-minus-snapshot sampled gradients plus the full snapshot mean
are clipped once before inherited Adam. All terms include log-normalizers.
Matched controls share exact sampled indices. Tests enumerate unbiasedness and
verify zero estimator noise at the snapshot; checkpoints remain diagnostic-only.
A standalone verifier reconstructs all fits and checks pairing, folding, source
hashes and input preservation. [Protocol/results](experiments/idr_variance_repair_2026-11-16.md)
own budgets, source rationale, cost accounting, successful confirmation and limits.


## 2026-11-16 — Full-trajectory TB transfer and evaluation

The separate `idr/transfer_*` modules reuse frozen cached features and existing
rollout/scoring helpers. Training evaluates full 20-residue softmax probabilities
at every cached position; only the example-selection proposal is normalized over
the finite bank. Training/conditioning exclude historical holdout rows, and
preflight verifies the pinned manifest, inherited training identities and prior
probability reconstruction before expensive execution.

A bounded, invertible eigenspace transform handles rank-deficient full-trajectory
inputs without changing the initial function or trainable parameter count.
Multiplying the hidden weight by the transform avoids a large transform per token;
folded heads use ordinary generation code. Equivalence checks and fresh-state
amplification measurements expose numerical and extrapolation problems.

Current, snapshot and full-reference gradients include the normalizer and use
global example weights across microbatches; clipping occurs once per optimizer
step. Matched row IDs, rates and gradient costs are persisted. Self-contained
snapshots support scoring recovery without source runs or ProGen execution.
Fresh-prefix coverage, common-bank copies and bank probability mass complement
raw generation success without presenting finite-bank metrics as global TV.
[Protocol](experiments/idr_transfer_2026-11-16.md) and
[verification](verification.md#full-support-transfer--16-november-2026).

## 2026-11-16 — Shared-data GFlowNet training with one adapted ProGen block

The separate `idr/lastblock_*` implementation shares the original frozen lower
layers and keeps a trainable copy of the final transformer block. Frozen final
normalization and output weights transmit gradients into that copy; the original
block still defines the reward reference. Teacher forcing recomputes current
policy likelihoods after sampling, with causal alignment tested against incremental
decoding. Adapted rollout base probabilities are explicitly replaced with original
reference probabilities before recording rewards and drift diagnostics.

Both trainers consume the identical ordered refresh batch. Shared lower-layer
inference feeds disjoint objectives, globally averaged across microbatches before
one optimizer step per arm. Separate block clipping preserves the matched head
clipping rule. Historical cached final features initialize training-only offsets;
they never substitute for the adapted computation. No conditioning or SVRG
machinery enters this trainer.

Fresh four-arm evaluation audits common data exposure, excludes accidental online
holdout overlap consistently, and counts noncopy successes over all raw outputs.
The original reference, source artifacts and unchanged supervised controls are
preserved. Source/config/checkpoint snapshots and sequence logs allow scoring
recovery without loading ProGen. [Protocol](experiments/idr_lastblock_2026-11-16.md)
owns budgets, choices and limits; [verification](verification.md#last-block-adaptation--16-november-2026)
owns correctness evidence.

## 2026-11-17 — Preserve the trainer while testing reproducibility

The confirmation orchestrates two fresh invocations of the existing last-block
trainer and adds separate plain-ProGen final evaluation. It derives settings from
the pinned completed pilot, permitting only the three declared RNG changes. This
keeps the successful training mechanism and old source hashes intact, while
preventing accidental continuation of the adapted checkpoint or drift in budgets.
The original per-row seed label remains local to each fit; enclosing replicate
metadata records the actual independent random streams.

Plain-ProGen evaluation uses no guidance head and shares support, prompts, scoring
and copy definitions with the four existing arms. It cannot reuse plain-ProGen
training draws as final evidence. The enclosing report preserves every replicate,
checks the old progression screens plus a separate plain-baseline screen, and
requires both fresh repeats to pass. This prevents favorable averaging or selecting
the better run from becoming a confirmation claim. After both fits and both plain
baseline generation stages finish, seal the enclosing generation manifest before
final plain-baseline scoring. This preserves recovery after a scorer failure,
without loading ProGen or repeating completed fits. Training is not resumed
automatically. The [protocol](experiments/idr_lastblock_confirmation_2026-11-17.md)
owns seeds, budgets and decision rules.

## 2026-11-17 — Compare training budgets without changing the successful trainer

The `idr/budget_*` wrapper derives both longer fits from the completed confirmation,
changing only total updates and evaluation seeds. Reusing the unchanged paired
trainer retains its online frozen/adapted/prior/replay mixture and protects past
source hashes. The inherited relative decay naturally moves from update 300 to
600; this is an explicit budget-and-schedule intervention. Saved 400-update copied
blocks and heads are loaded only for fresh control generation, never as training
initialization or as optimizer resumptions.

The parent evaluator uses the union of original-bank, old-fit and new-fit training
tails for every arm's copy-aware success. This avoids asymmetric novelty rules
when the new model has seen more data. Parent sequence identities include the
repeat name; child reports preserve their original definitions. The two-repeat
decision compares final 800-update models directly with freshly sampled original
400-update controls; inherited architecture screens remain secondary.

All generation and input snapshots are sealed before final control scoring.
Scoring recovery uses those snapshots without old source folders, ProGen loading
or training. It cannot recover missing generation or resume incomplete training.
The [protocol](experiments/idr_lastblock_budget_2026-11-17.md) owns settings and
stopping rules; [verification](verification.md#last-block-training-budget-comparison--17-november-2026)
owns correctness evidence.

## 2026-11-18 — Extend the budget comparison while preserving completed provenance

The `idr/budget1600_*` wrapper reuses the existing paired trainer and saved-record
scoring helpers. Historical budget modules are source-pinned by completed runs;
changing them into a generic runner would invalidate those provenance checks.
The new wrapper validates the completed 800-update derivation, permits only the
doubled budget and fresh evaluation seeds, and loads old adapted weights solely
for control generation. It does not introduce another training algorithm.

Snapshot the old 400-update training history as well as the 800-update control
history, so the common copy boundary persists and scoring recovery needs no
ancestor folders. Seal generation before final comparison scoring; incomplete
training is not resumable. The [protocol](experiments/idr_lastblock_1600_2026-11-18.md)
owns scientific settings and the stopping rule.

## 2026-11-18 — Match GFN and supervised capacity without changing historical runs

New `idr/objective_*` modules reuse the pinned final-block wrapper, probability
accounting, original initialization and sequence scoring. Each learner owns a
separate copied block/head, sharing only the frozen original model. Both gradients
are accumulated and checked before either optimizer steps. Supervised weights
use detached full-batch within-prompt softmax of log property factors, then split
across microbatches; renormalizing inside a microbatch would change the objective.

The shared collector saves every GFN/supervised/prior discovery and its detached
reward, plus exact replay IDs in both update logs. This makes data-access matching
auditable. The evaluator reconstructs reward labels, raw/noncopy endpoints and
fixed-threshold sequence clusters from saved sequences; it does not infer GFN
distributional accuracy from property success. New repeat exposure defines a
common copy union, while old adapted training histories are not treated as inputs
to these freshly initialized policies.

Checkpoint writes use an atomic temporary replacement and only one rolling latest
file per arm; completion renames it to the final checkpoint. This bounds retained
weights and protects the previous checkpoint from write failure without adding
an optimizer/RNG resume path. Seal generation and provenance before final scoring;
recovery uses saved inputs without ProGen or ancestor folders. Keep measured setup,
collection, scoring and per-arm update costs separate, labeling the original-prior
audit overhead included in GFN updates. The
[protocol](experiments/idr_objective_2026-11-18.md) owns scientific choices.

## 2026-11-19 — Version the acceptance bonus across training and evaluation

The `idr/success_bonus_*` modules preserve source-pinned historical implementations
while reusing the existing last-block policy, samplers, objective updates and metric
helpers. The new target adds log(9) only to joint passes and retains all failure
weights; both objectives consume the same final factor. Relabel cloned bank rows
before training-only normalizer initialization and replay, and label fresh shared
samples before either learner sees them. Keep base factor, bonus and final factor
in saved records so evaluation can reconstruct the intended labels independently.

Three fresh seed plans exclude prior sampling streams. Checkpoints and the separate
reward definition carry the new version; inherited pilot configuration describes
original probability/acceptance conventions and is not the authoritative reward
version. The evaluator uses all three repeats for the conservative training-copy
union and applies the prospective ten-point/distinct-success gate. Atomic rolling
checkpoint storage and sealed generation recovery are retained. The
[protocol](experiments/idr_success_bonus_2026-11-19.md) owns budgets, reward rationale,
baseline fairness and limits; no new algorithm is hidden in the shared trainer.
