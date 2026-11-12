# Decision record

This is a chronological record; newer entries are appended below. For the active
stage use [current status](status.md); use the [experiment index](experiments/README.md)
to find each protocol, result and supporting evidence.

## 2026-10-14 — Scope and execution

- Implement in v1 only. Keep the old test project unchanged; reuse its verified
  checkpoint as a read-only input and vendor the minimal upstream ProGen source
  with attribution. No dependence on old runnable scripts.
- Stage the exact four-position experiment before the biological experiment.
  GPU convergence and stability-model quality are gates the user must run, not
  results we can establish from toy CPU tests.
- Use Python 3.10, NumPy 1.26.4, PyTorch 2.6.0, Transformers 4.27.4 and Tokenizers
  0.13.3, matching the working ProGen environment. Keep a v1 environment and
  explicit runtime validation to avoid the old renamed environment's entrypoints.
- Use a small src-layout Python package, JSON configurations, standard-library
  unittest, and simple commands. Avoid a new training framework or scheduler.
- Download the published Mega-scale archive into v1 for a data-only audit. This
  is setup, not an experiment. Verify its published MD5 and record SHA-256.
  Source: [Zenodo release 7992926](https://zenodo.org/records/7992926),
  `Processed_K50_dG_datasets.zip`, published MD5
  `f7e8c553efee734cf161ee6f2b0a09cf`, 1,013,678,473 bytes.

## 2026-10-14 — Finite task representation

- Compile each initial task into a prefix tree. This enforces measured-candidate
  support using identities only and makes exact diagnostics possible. Mutation
  neighborhoods remain available through explicit enumeration, capped at 20,000
  sequences to prevent accidentally starting a huge task.
- Cache frozen prefix features once, then train only the small guidance head.
  This is mathematically the same fixed-backbone policy on the declared finite
  support. Charge cache construction to setup cost. It does not demonstrate
  scalable unrestricted generation; expanding beyond enumerable tasks is later work.
- Tree order is deterministic (prefix length then lexicographic order; sorted
  leaves). Hamming distances reject unequal lengths. Budget means *at most* k,
  so the unmutated reference is legal unless a finite support explicitly excludes it.
- Head context uses position/length, remaining budget/max(1,budget), and one-hot
  reference residue. Full reference identity is constant within the first run.

## 2026-10-14 — Backbone and feature definitions

- Preserve the checkpoint's original output projection. Use its literal forward
  marker `1`, not special-token ID 1. Verify all 20 residue round trips.
- Compute logits in float32 after the original LM projection; store cached features
  and guidance computations in float32. Configure real inference precision explicitly.
- Complete-sequence scorer features are the mean final-layer states at residue
  positions, excluding the initial marker. These contain full-completion information
  collectively; the guidance head receives only the state at its current prefix.
- Prefix-cache construction groups equal-length prefixes, avoiding padding/position
  ambiguity. Incremental decoding consumes every residue, including forced actions.
- CPU fixture: two layers, hidden width 64, eight attention heads, rotary width 8,
  MLP width 64, context 128, seed 0. This is a mechanical test, not a protein model.

## 2026-10-14 — Learning mechanics

- Head: layer-normalize frozen prefix features, append deterministic context,
  hidden linear layer + SiLU + zero-initialized output projection. No dropout.
- Train TB on current-policy complete trajectories. Discrete sampling has no
  gradient; recompute path likelihoods for the squared TB loss. The scorer and
  cached baseline remain detached. Adam updates the head and scalar log Z only.
- Supervised baseline estimates log mean exponentiated terminal reward for each
  observed prefix/action under baseline rollouts. Sample regression examples with
  rollout multiplicities. Unobserved edges have no direct training target. This
  is a continuous future-reward adaptation, not a reproduction of FUDGE's classifier.
- Report both KL directions explicitly. No truncation, temperature scaling,
  replay buffer, moving scorer, or implicit deduplication is introduced.

## 2026-10-14 — Mega-scale preprocessing and scorer

- Use `aa_seq` (domain only), `WT_name`, `WT_cluster`, and finite `dG_ML`; retain
  original `ddG_ML`, confidence-interval and stabilizing-flag metadata for audit.
  Do not use linkers in `aa_seq_full`. Retain 40–72 canonical residues.
- Require one unambiguous measured wild-type sequence per parent. Restrict initial
  candidates to same-length Hamming distance ≤1. Average valid duplicate-sequence
  measurements before splitting; record replicate counts and label ranges.
- Choose the eligible parent with most unique valid local variants, breaking ties
  lexicographically. Minimum 300 substitutions. Selection uses measurement
  availability, not high stability values. This conditions the benchmark on assay
  coverage, which must be acknowledged in interpretation.
- Split unique substitutions 70/15/15 with NumPy seed 42; force the reference into
  training. The biological benchmark support consists only of held-out test variant
  identities. There is no within-parent sequence overlap across splits. This is
  local prediction, not unseen-family generalization. Grouped-domain splits are
  deferred to the later expansion milestone.
- Predict absolute dG_ML. Normalize features and labels with training moments,
  population standard deviations and a 1e-6 numerical floor. Reward is predicted
  candidate minus predicted reference, divided by training-label standard deviation.
  No clipping in the pilot; non-finite values fail rather than silently changing rewards.
- Select ridge penalty or MLP checkpoint by validation RMSE. Gate: validation MAE
  must beat a constant training-mean predictor and Spearman must be at least 0.2.
  This is an engineering screen, not proof of reliable design or a literature cutoff.
- Slow archive delivery motivated a range fetch of only the relevant CSV. Verify
  its central-directory metadata, decompressed size and ZIP CRC; record CSV SHA-256.
  The published whole-archive MD5 is recorded but cannot be verified from a partial
  archive. Do not report whole-archive checksum verification.

## 2026-10-14 — Reproducibility, comparisons and handoff

- Pilot settings are fully listed in hyperparameters.md and JSON configs. Default
  float32, fixed update counts and final head checkpoints avoid unreported tuning.
  Scorer candidates alone use validation checkpoint selection. Seeds 0–2 are
  engineering replication; the data split uses seed 42 and scorer fitting seed 0.
- Keep raw sample slots and top-score selection indices. Use 256 raw comparisons,
  32 selected slots, and an explicitly larger 1,024 baseline selection pool. Do not
  equate selected-size matching with equal compute. No deduplication is performed.
- Both targets and methods operate on the same finite support. Biological novelty
  is restricted by the held-out single-mutant support and cannot establish novelty
  relative to ProGen's pretraining corpus or general natural proteins.
- Store source snapshots, original model configuration, model/tokenizer/data hashes,
  progress records and separate failure states. Saved evaluation runs on CPU from
  frozen inputs and heads. An interrupted directory is never automatically reused.
- The exact capacity diagnostic uses full enumerated conditional targets and is
  explicitly privileged. It is available to diagnose a failed user-run gate, not
  counted as a fair competitor or run as a real experiment by the agent.
- A real biological run requires a completed exact run with the same checkpoint
  hashes. Scorer-gate failure returns before guidance and test-value evaluation.
- Distinguish cached terminal-reward lookups from new scorer calls. Count all
  frozen-backbone checks and prefix/sequence feature extraction in setup. Record
  host-process high-water memory separately from CUDA peak allocation.
- Ruff 0.14.0 is a development-only pinned formatter/linter. Preserve vendored
  source formatting and licenses; check our code and tests without modifying it.
- Biological CPU smoke mode requires an explicitly synthetic manifest; actual
  Mega-scale data cannot accidentally become a tiny random-model experiment.
  Real-data manifests created before that optional flag are interpreted as measured.

## 2026-10-14 — Manual GPU selection and environment activation

The user inspects `nvidia-smi`, exports `CUDA_VISIBLE_DEVICES` and launches the
experiment manually. The launcher preserves that variable and does not choose GPUs
or submit jobs. This keeps resource allocation under user control.

## 2026-10-14 — Failed exact gate and controlled fitting diagnostics

- Preserve `exact-first` and `exact-capacity`; record their failures in the
  [experiment record](experiments/exact_diagnostics_2026-10-14.md). Do not relax the
  gate or begin biology. Neither mean-score agreement nor completion status is a pass.
- Audit saved inputs on CPU without fitting. Verify exact conditional targets by
  an independent leaf-mass calculation; decompose target-to-policy KL by prefix
  depth and inspect distinctness/nearby feature rows and required residuals.
- Keep the original head, reward and uniform-prefix objective for the next
  comparison. Compare 10,000 updates at 0.001 and 0.0001, seed 0, to separate longer
  optimization from a lower rate. This is an engineering diagnostic, not a fair
  baseline or a new biological result. The user runs both arms.
- Add observational TV/KL/entropy-floor and gradient logging, not oracle checkpoint
  selection. Retain final heads, source/input provenance and failure markers.
  The new post-update logging convention fixes historical loss/checkpoint timing
  ambiguity without changing original parameter updates; a fixture verifies parity.
- No architecture, baseline, scientific formulation, threshold or experimental
  split changed. Input conditioning is a hypothesis, not established causality.

## 2026-10-14 — Longer privileged fit passes; return to TB

The user-run 10,000-update original-rate fit reached TV 0.042336, while the lower
rate reached 0.055953. Keep the existing architecture and learning rate. Prepare
`configs/exact-long.json` with 10,000 updates for both TB and the supervised baseline;
all other settings are unchanged. This isolates update budget within the existing
comparison and avoids reusing oracle-trained weights. Require fresh final TB TV
<0.05 in all three seeds. The one-seed privileged pass is evidence of attainable
accuracy on this cache, not proof that more updates resolve TB or biological tasks.
Input hashes were verified; raw outputs are preserved. Details and the historical
step-2,000 reproduction caveat are in the experiment record.

## 2026-10-14 — Longer TB result: one seed passes

The user-run longer TB experiment improved all seeds but passed only seed 1
(TV 0.0971 / 0.0471 / 0.0773). Preserve the all-seed criterion and biological gate.
Saved-head CPU checks reproduce results; logged late gradients are clipped and
sampled losses fluctuate, but there is no intermediate exact-TV record. Propose
an explicit late learning-rate reduction and observational exact monitoring as
the next controlled test; do not claim instability is established or change the
architecture from this evidence. The proposal is not implemented or launched.
See [the result record](experiments/exact_long_2026-10-14.md).

## 2026-10-14 — Implement the approved late-rate experiment

The user approved the proposed schedule and monitoring. Add optional settings to
both learners; in the new exact-decay configuration, preserve rates through update
8,000 and multiply both TB optimizer-group rates and the supervised head rate by
0.1 starting at 8,001. Preserve Adam state and all other hyperparameters. These
choices test late settling and are not evidence of optimal tuning.

Add exact metrics at initialization, regular logging points and both sides of the
boundary, without feeding them into learning or checkpoint selection. Log actual
rates; keep sampled loss semantics explicit. Charge added monitoring time and score
reads separately and restrict it to the exact toy via configuration validation.
Old configurations retain their behavior; source snapshots and raw outputs remain
unchanged. CPU parity tests confirm monitoring is observational. No scientific
specification or acceptance threshold changed. See the
[experiment record](experiments/exact_decay_2026-10-14.md).

## 2026-10-14 — Scheduled run: two seeds pass

The user-run scheduled experiment passes seeds 1 and 2, but seed 0 finishes at
TV 0.053937. Keep the unchanged all-seed gate and preserve final checkpoints.
Logged TV variability falls markedly after the rate drop; seed 0 briefly crossed
0.05 but that does not count as a final pass. Prepare one bounded low-rate-budget
extension: 20,000 total updates, drop still after 8,000 (`after_fraction=0.4`),
both learners extended equally. The near-plateau may persist; do not claim success
in advance or substitute a best intermediate checkpoint. No training code changed.
See the [result and next plan](experiments/exact_decay_2026-10-14.md).

## 2026-10-15 — Exact gate passes; prepare biological pilot

The user-run 20,000-update scheduled exact experiment passes all three final
seeds (TV 0.040901 / 0.046860 / 0.044724). Accept the unchanged engineering gate
and stop further toy tuning for this milestone; late fluctuations and tuning on
these same seeds limit generalization claims. No biological improvement is implied.
Prepare `biological-pilot.json` referencing the passing exact run and transferring
the tested 20,000-update schedule to both heads. Preserve scorer/split/reward/gate
settings; omit exact training-time monitoring. Validation precedes guidance and
failure still stops it. Keep historical configurations and raw results unchanged.
The spec gains only a factual milestone-status bullet. The user runs the pilot.

## 2026-10-15 — Biological pilot completes without measured gain

Accept the completed biological pilot as an end-to-end implementation result,
not evidence of improved measured stability. Exact finite-support expectations show
the predicted target itself lowers mean measured stability slightly while improving
predicted stability. GFN already matches that target closely; additional fitting
is not the next priority. Preserve all results and analyze reward ranking and
candidate-prior concentration. Any revised choices must use development evidence
and fresh confirmation, not hidden repeated tuning against this inspected test set.
No new hyperparameters, training configuration or model changes were introduced.
See the [pilot results](experiments/biological_first_results_2026-10-15.md).

## 2026-11-03 — Return to the first biological pilot

The user's meeting direction is to investigate the local stability reward and
simple predictor baselines in depth before expanding scope. The active project
therefore returns to the first 1O6X pilot. Retain the frozen backbone, original
data/splits, reward definition, trajectory-balance learner, supervised future-reward
baseline, exact tests and completed pilot artifacts unchanged.

Remove the subsequent workflows from the active CLI, code, tests, configurations
and documentation. They are recoverable outside the project at
/usr/scratch/agoel320/progen-gflownets-archive/post-pilot-2026-11-03.
This is a scope reset, not a claim that later results were scientifically invalid.
The archive is not an active dependency; Git history remains intact.

The pilot already matched its predicted target but did not improve measured
stability. That motivates predictor/reward diagnosis before another optimizer
comparison. See [pilot results](experiments/biological_first_results_2026-10-15.md).
The next comparison must give linear and neural predictors identical data,
features and evaluation rules, with training-set size explicitly controlled.
No new split, threshold, reward transform or hyperparameter sweep is fixed here.

Previously inspected outcomes remain inspected. In particular, the original 204
test variants and any retired evaluation panels cannot be relabeled as fresh
confirmation after tuning. New analyses on the original task are developmental.


## 2026-11-03 — Separate predictor quality from target fitting

The saved reward arithmetic agrees within floating-point tolerance, while
influential local ranking errors reduce exact measured utility. Compare both
predictor families explicitly on nested original-training subsets rather than
only reporting the validation winner. Keep fixed validation/evaluation identities;
40 examples is a small-data diagnostic, not a relabeled original training count.
Use original settings to limit new tuning, and test only half/original/double
strength plus zero and a reference-capped utility. Exact enumeration isolates
reward usefulness without another guidance fit. The [protocol and audit](experiments/predictor_reward_study_2026-11-03.md)
own settings, rationale, evidence and interpretation boundaries.


## 2026-11-03 — Ridge is provisional; test split dependence before new guidance

The completed [predictor/reward study](experiments/predictor_reward_study_2026-11-03.md)
finds comparable full-data regression errors but opposite exact-target utility:
ridge gains 0.03960 kcal/mol at beta 1; MLP loses 0.01861 on average. MLP still
wins validation RMSE, so the issue is not a broken family selector. Errors on the
small high-probability part of the candidate set can dominate guidance utility.
Treat ridge/beta 1 as a provisional development baseline and predefine a small
within-parent split-robustness comparison before new GFN training. Keep the neural
control and track prior-weighted error as well as global prediction accuracy.
Do not interpret repeated full-data ridge seeds as replication or select the
largest beta using inspected outcomes. No optimizer change is motivated yet.


## 2026-11-09 — Test the ridge target with unchanged trajectory balance

Tom's meeting feedback prioritizes an actual GFN rerun using the better reward
before more split studies. Reuse the full-data ridge checkpoint (seed 0 as an
arbitrary representative), original cached prior and beta 1. This isolates reward
source from optimizer, architecture, candidate-support and backbone changes.
Add observational predicted-target monitoring to check training reliability,
while retaining fixed steps and final-only checkpoints. The original MLP GFN is
a saved historical comparator, not a new neural-predictor retraining. The
[protocol](experiments/ridge_gfn_2026-11-09.md) owns settings and interpretation.

Reward-noise experiments are a subsequent controlled study; distinguish fixed
corruption of each sequence's score from fresh noise on each query. Attribute
factorization, partial ProGen training and SubTB require separate review and
experiments. No such changes are combined with the first ridge run.


## 2026-11-09 — Use the completed ridge GFN as the clean noise-study baseline

All three ridge-reward GFNs match the specified target and improve measured
expectation over ProGen on the fixed candidate set. Mean gain is +0.03948 kcal/mol,
with expected unique count in 256 draws decreasing from 10.67 to 9.87. The
[completed experiment](experiments/ridge_gfn_2026-11-09.md) supports proceeding to
Tom's reward-noise question while keeping TB and frozen ProGen unchanged.
Separate noisy-target degradation from optimization error against that target.
Start by specifying fixed per-sequence score corruption; resampled query noise
would require a separate interpretation. No noise settings have been chosen yet.


## 2026-11-09 — Fixed score corruption with crossed noise and training seeds

Implement Tom's noise suggestion as fixed independent Gaussian perturbations of
dimensionless terminal scores. Preserve beta, reward scale, prior and optimization.
Two amplitudes (0.25/0.50) crossed with three noise seeds and three training seeds
separate corrupted-target variability from training variability; reuse the clean
baseline. Pair each noise vector across amplitudes without recentering or clipping.
Keep the full training budget because shorter noisy runs would confound the
comparison. This is a sensitivity model, not an inferred assay-noise distribution.
Exact evaluation separates target damage from policy-fitting error and reports
clean predictor expectations distinctly from measured outcomes. The
[fixed protocol](experiments/reward_noise_2026-11-09.md) owns the complete choices.


## 2026-11-09 — Fixed noise affects target quality more than fitting

All 18 noisy-target GFNs converge closely, while the higher-noise seed-101 target
falls below the ProGen stability baseline. The [results](experiments/reward_noise_2026-11-09.md)
therefore prioritize reward-generalization checks; they do not identify a TB
optimization failure. Favor a controlled local evaluation before opening generation
beyond known candidates. Partial ProGen unfreezing remains separate, aimed at speed
or harder-task capacity with a fixed reference target. Noise that happens to help
on inspected outcomes is not a justification for selecting perturbations or
claiming noise injection improves biological design.


## 2026-11-09 — Three predefined local resplits before wider generation

The user authorized checking whether the ridge advantage survives held-out
identity changes. Use split seeds 43/44/45 with original partition sizes and
predictor seed 0, fixed beta 1 and original validation-based predictor selection.
Reassemble cached frozen pooled features without fitting on evaluation labels.
Recompute each evaluation support's constrained ProGen prior: a baseline from the
original 204 candidates would not be the correct comparator. Report global and
prior-weighted error plus each target's gain over its own baseline. This varies
both predictor partitions and candidate support; overlapping resplits remain
local development evidence, not independent confirmation. Full choices are in the
[protocol](experiments/split_robustness_2026-11-09.md).


## 2026-11-09 — Local gains persist; ridge superiority does not

The [three completed resplits](experiments/split_robustness_2026-11-09.md) give
positive target gains for ridge and MLP, with MLP better on two and essentially
tied on the third. Keep both baselines rather than treating the original ridge
win as a general conclusion or selecting MLP using these test outcomes. One
split's prior assigns 98.36% to a single variant, so utility analysis must retain
prior concentration and support definitions. Mean stability and reference-level
success do not always rank predictors identically.

Recommend a predefined mutation-position holdout study before new mutation
combinations: random substitution holdouts can include other substitutions at
the same positions in training. This addresses a harder predictor-generalization
question; it does not establish accuracy on combinations or new families.
Protocol/implementation remain pending. Keep the current sampler and frozen
backbone while assessing rewards, with no additional GFN rerun for this analysis.


## 2026-11-09 — Group all substitutions by mutation position

The user authorized the harder local generalization check. Keep the same three
seed values (43/44/45), but permute observed positions rather than variants.
Allocate 50/11/11 position groups using the inherited 70/15/15 rounding rule;
retain the reference only in training. Accept slightly unequal sequence counts
rather than splitting groups to force the previous counts. No position is moved
based on sequence identity or stability outcomes. Reuse unsupervised frozen
features and the original scorer settings, with group-disjoint validation for
model selection. Recompute each support's locally masked prior and report its
concentration alongside mean utility. These choices address generalization at
unseen mutation positions without claiming an isolated causal effect of grouping
or accuracy on combinations. The [protocol](experiments/position_holdout_2026-11-09.md)
owns full settings, outcome boundaries and limitations.


## 2026-11-09 — Useful position-held-out rewards justify a sampler follow-up

The [completed grouped-position study](experiments/position_holdout_2026-11-09.md)
finds positive target gains for both predictors and larger ridge gains on all
three splits. These supports have different, less concentrated priors and often
weaker starting means; the gains are not directly comparable to random-split
improvements. Prediction error and poor reference-level success on split 45
remain limitations. Recommend testing the same frozen-backbone TB sampler on all
three fixed ridge targets, preserving exact MLP baselines and measuring policy
fit separately from biological utility. This ridge choice is post-hoc development,
not independent confirmation. Do not tune beta, penalties or choose only a winning
split from these outcomes. Follow-up protocol/implementation remain pending.


## 2026-11-09 — Fixed-support ridge GFN follow-up across all position folds

The user authorized the sampler follow-up. Retain all three fixed grouped supports
and saved ridge rewards; use inherited training seeds 0/1/2 and 20,000-step TB,
with no beta, architecture or optimization changes. Reconstruct missing prefix
features with frozen ProGen, verify matching logits, then reuse saved logits
exactly to prevent prior drift. Release the backbone before small-head training.
Keep MLP as an exact distribution comparator; do not fit another MLP sampler.
Exact evaluation of all policies distinguishes target utility from fitting error
without best-of selection. Labels are excluded from training tasks and copied for
evaluation after all fits. The [protocol](experiments/position_gfn_2026-11-09.md)
owns all settings, tolerances and post-hoc-development interpretation.


## 2026-11-10 — Local sampler validation complete; broaden the stability question

The [nine completed grouped-support GFNs](experiments/position_gfn_2026-11-09.md)
all match their targets closely (maximum final TV 0.001972) and realize measured
stability gains. Treat this as completion of the local mechanism check, with no
observed fitting failure justifying a new objective or partial unfreezing. It
still uses one parent's finite known variants; a low starting baseline and limited
reference-level success must not be confused with broad biological capability.

The user's stated goal is new stable proteins. Recommend a next protocol for
stability prediction across multiple proteins, beginning with dataset/exposure
review and protein/similarity-group separation, retaining simple baselines rather
than presuming ridge superiority. No new outcomes, groups or training budgets are
selected here; this is proposed scope, not an implemented expansion or restoration
of retired workflows. Novel generation also needs an explicit prior/support and
independent stability assessment.


## 2026-11-10 — A bounded grouped stability benchmark before broader generation

The user authorized the multi-protein step. Build components from source cluster,
shared-sequence and >=40% globally aligned reference-identity links, including
unselected intermediate parents. Exclude components matching known prior-use
metadata, without claiming complete exposure or pretraining independence. Choose
one parent per component and cap the panel at 64 groups with 65 measured sequences
each, using identity-only hash ranks. Equal parent counts limit scan-size dominance;
44/10/10 groups provide a first cross-protein feasibility test within the assay's
short-domain scope. Grouping and selection are fixed before real fitting.

Keep frozen ProGen mean-pooled features, compare constant/ridge/MLP on identical
candidates, and assess pooled, per-parent and reference-only errors/selection.
Uniform candidate selection is the explicit comparator; there is no constrained
ProGen prior in this study. This avoids attributing a changed sampling support to
better predictor accuracy. Extend the ridge grid to 1000 and MLP budget to 2,000
updates for this larger panel; validation alone selects checkpoints. The
[protocol](experiments/multiprotein_2026-11-10.md) owns all choices, rationale,
primary-source links and limitations. No old workflows or outcomes are restored.

## 2026-11-10 — Separate regularization from protein-coverage expansion

At the user's direction, vary ridge strength first while retaining all existing
data identities, frozen features and outer partitions. The earlier validation
failure was dominated by parent-mean prediction errors; the chosen penalty 1000
was at the grid boundary. Extend strength by four decades and select with grouped
CV only inside the 44 training parents, then report against the original saved
ridge and constant on outer validation. This is an engineering diagnostic of
shrinkage, not an established remedy. No test evaluation, MLP changes or additional
training proteins. Constant winning CV or loss of within-parent ranking remains a
negative finding, even if one metric improves. [Protocol and rationale](experiments/regularization_2026-11-10.md).

## 2026-11-10 — Expand training coverage after the shrinkage tradeoff

Stronger shrinkage lowered error but reversed average selection lift. At the
user's direction, return to original ridge as a fixed development comparator and
test more protein groups separately. Preserve evaluation identities, the original
44 training parents, sequence counts per parent, and original grouping/exclusions.
Add unused groups by the prior identity-only hash order, producing 44/64/86-parent
nested sets. Scale summed-loss penalty with training rows to preserve the original
strength per example, rather than silently weakening regularization as data grows.
Refit training-only normalization and reuse all original features. Report all
stages and both error/ranking metrics; do not select a favorable expansion from
validation. One expansion does not establish a universal learning curve. The
[protocol](experiments/coverage_2026-11-10.md) records controls and limitations.


## 2026-11-10 — Investigate the scorer representation before broader guidance

After coverage expansion left absolute errors above constant, the user requested
literature on improving the predictor. The [focused review](../literature/STABILITY_PREDICTOR_REVIEW_2026-11-10.md)
finds relevant precedents for stability-specific encoders and structural inputs,
while distinguishing absolute stability from mutation-effect prediction.

Recommend a frozen ESM-2/ridge comparison against ProGen/ridge on identical data,
followed by a separate scorer-adaptation experiment. The 35M generic ESM-2 base is
an inexpensive diagnostic choice supported by the EsmTherm author's training
example, not a claim of optimal capacity or a reproduction of fine-tuned ESMtherm.
Retain all 86 training parents; do not select the favorable observed 64-parent
stage. Preserve the original ridge rule for the initial representation comparison;
any subsequent tuning must be fair to both encoders and use training groups.

This revises the proposed next improvement from coverage resplits alone to a
representation test with group-robustness checks. Require both useful selection and
absolute transfer. Published stability checkpoints need supervised-overlap review,
and current inspected panels remain development. ProGen generation stays frozen;
GFN objective changes do not repair reward errors. This is a recommendation only:
no architecture, configuration, split or completed artifact changes in this review.
The [plan](../literature/EXPERIMENT_PLAN.md) owns the proposed sequence of work.


## 2026-11-12 — Pivot from stability to IDP/IDR property guidance

Source: the user's [meeting transcript](</nethome/agoel320/.codex/attachments/da1c453b-9d01-4172-9fff-e5df1ae4ab60/Pasted text.txt>)
and explicit request to analyze and document the pivot without implementation.
The [review](../literature/IDP_IDR_REVIEW_2026-11-12.md) records the supporting sources
and candidate tools; the [specification](../spec.md) now owns the active direction.

**Why stop the stability branch here:** Tom framed the central problem as guiding
an evolutionary sequence prior with a reliable property reward, not establishing
the limits of ridge/MLP regression. Folding stability depends on a completed
sequence's collective interactions and conformational behavior; attributing its
terminal signal to early token decisions may require substantial data/model
capacity. Building a general cross-protein stability predictor was becoming a
separate research problem that obscured the guidance question. The local GFN
checks already provide a useful foundation for changing the biological task.

**What our evidence establishes:** the local GFNs learned their targets accurately;
reward quality and cross-protein prediction were the demonstrated bottlenecks.
The additional long-range credit-assignment explanation is a plausible difficulty,
not a diagnosed cause of a GFN failure in our completed runs. Terminal-reward
GFlowNets are not inherently unable to optimize stability. A prefix-tree policy
can use full-prefix PLM representations, and normalized sequence probabilities,
not unnormalized flow, sum to one. Preserve this distinction in future slides.

Tom's desire for a highly accurate reward model (roughly 0.9 Spearman in the
conversation) is an aspiration for a useful proxy, not a universal justified gate
or evidence that such performance is available. Our fixed-score Gaussian-noise
study perturbed predicted rewards; it did not perturb DMS labels and refit scorers.
It showed that noisy targets can remain accurately sampled while losing biological
utility, not robustness to every measurement-noise process.

**New direction:** select IDP/IDR properties with transparent sequence scoring and
potential local profiles. Investigate IDiom's data and post-training, reuse author
scoring tools, and compare against direct design. Do not choose flexibility for
the first experiment: Tom explicitly used it as an example and preferred another
property. Start with isolated IDRs rather than assume all whole IDPs are covered
by the same dataset. No final functional property was selected in the meeting.

**Proposed first test:** local charge-profile matching, followed by an ordering
comparison at fixed composition. The score is exact under declared conventions,
removing fitted-reward error from the first question. It is not an exact biological
function label. Disorder/function predictions require separate evaluation. GOOSE
is an essential baseline because descriptor-targeted IDR design already exists.
Acidic activation-domain activity is a later measured-data lead, not yet selected.

The previous ESM stability-representation proposal is superseded as immediate work.
Historical specifications/results remain accessible; no completed run or code is
modified by this decision. The current authorization covers literature, scientific
framing and a plan only. Implementation and a GPU command follow a fixed protocol,
not this meeting analysis.


## 2026-11-12 — Implement the generation-only IDR feasibility baseline

The user authorized implementation after agreeing on baseline IDR generation,
charge-profile guidance, then a functional target. Begin with 80-residue outputs,
16-residue natural prefixes and matched 64-residue evaluation tails. Three seeds
and eight distinct prompts help expose dependence on one convenient prefix. Use
fixed-length canonical ancestral sampling with no outcome filtering or tuning.

Prepare experimentally annotated DisProt windows with source statements and NMR/CD
evidence; exclude missing-density-only evidence and the audited apo-calsequestrin
case. Identity-only selection precedes predictor scoring. The assay context remains
part of the evidence; cropped windows are not individually measured synthetic IDRs.

Pin metapredict 3.0.2 to preserve the existing numerical environment. V3 is primary;
V2 is a related sensitivity check because IUPred3 requires academic acquisition.
Do not label V2 independent verification. Both full-context and isolated-tail
predictions expose prompt/context dependence. No stability predictor or GFN is fit.
The [pre-run protocol](experiments/idr_baseline_2026-11-12.md) owns every fixed budget,
threshold, source and decision rule. Charge guidance depends on baseline results.


## 2026-11-12 — Baseline supports charge-guidance planning, with prompt-level checks

The [completed baseline](experiments/idr_baseline_2026-11-12.md#completed-results--12-november-2026)
shows usable predicted-IDR yield without backbone adaptation. Proceed to specifying
charge-profile guidance on the prompted prior; preserve unguided and generate-and-select
comparators. Fix numeric guidance choices before implementation/training rather
than infer them from a favorable outcome. This is a proposed next experiment.

The RPB1 prompt shows why pooled uniqueness and homopolymer checks are insufficient:
repeated multi-residue motifs can score highly without long identical-residue runs.
Retain per-prompt motif/diversity diagnostics and the full baseline panel instead
of selecting only high-scoring contexts. The post-hoc non-RPB1 calculation supports
that the aggregate prompting gain is not entirely due to this case; it does not
replace the predefined main analysis. Predicted disorder remains a proxy, and
this pilot establishes no GFN advantage or biological function.

## 2026-11-12 — Authorize a coarse charge-profile guidance pilot

The user approved implementing the next step after understanding the baseline and
its disorder screen. Reuse all eight prefixes and the same frozen, locally masked
ProGen prior. Use a two-block signed-charge target with a soft charged-content term;
this isolates sequence ordering at modest charge before fine sliding-window targets.
Numeric targets/scales and the inspection of existing development profiles are
recorded in the [pre-run protocol](experiments/idr_charge_2026-11-12.md), not presented
as literature defaults or independent confirmation.

Keep disorder outside the training reward and candidate selection. The next question
is whether direct descriptor guidance preserves disorder, not whether a generator
can maximize its evaluation predictor. Reuse TB first and change the sequence space
from finite mutation candidates to open continuations, without simultaneously
unfreezing ProGen or switching losses. Use one learned normalizer per prompt.

Retain raw prior and raw GFN outputs, and compare charge-only top-quarter selection.
Keep generation/training budgets explicit rather than claim equal compute from equal
output counts. Include cheap composition-preserving swaps and shuffles: matching
this simple target is not enough to establish a need for GFlowNets. Fixed final
checkpoints and all-prompt reporting guard against selecting an easy/repetitive
context. The protocol owns inference limits and next-decision branches.


## 2026-11-12 — Diagnose charge policy learning before expanding the task

The [completed charge pilot](experiments/idr_charge_2026-11-12.md#completed-results--12-november-2026)
shows no aggregate charge-pass gain for GFN over prior (20/384 each), while selection
and composition-preserving editing improve charge success. Preserve this negative
result. Do not proceed to a learned functional reward or claim charge-guidance success.

Saved-path diagnostics show small policy corrections and large residual reductions
associated with normalizer shifts; all updates trigger clipping. These motivate
controlled optimization diagnosis, not a causal claim against clipping or learned
normalizers. Approximate prior-pool reweighting suggests that the fixed target has
headroom, with finite-sample limitations. Keep the objective and comparison data
fixed while isolating learning behavior; change optimization and horizon separately.
No new experiment configuration is selected in this analysis. Evidence and exact
post-hoc calculation definitions are recorded with the experiment.


## 2026-11-12 — Measure real gradient behavior before changing charge training

The [CPU diagnosis](experiments/idr_charge_diagnosis_2026-11-12.md) finds that the
same residual head learns an exact tiny target with low-noise updates, while eight
sampled trajectories learn less accurately and a tenfold rate increase worsens
that control. All real updates clip, but Adam makes clipping fractions insufficient
to infer effective update size. Saved heads also respond weakly to explicit block
context when synthetic features are held fixed; real states were not retained.

Recommend a small instrumented real-feature diagnostic before selecting a new rate,
context path or clipping rule. Keep objective/support fixed, measure gradient
variance and actual context dependence, then change one factor. Do not claim that
normalizer learning caused the failure, that removing clipping is established as
better for ProGen, or that a synthetic exact solve proves real convergence. The
original pilot and production code remain unchanged.


## 2026-11-12 — Capture real gradient evidence without another full fit

The user approved the short diagnostic. Keep all original prompts, charge settings
and sequence support, and inspect both initialized and final heads for every seed.
Six fixed-policy cases with sixteen balanced microbatches each provide separate
proposal/reference halves and 8/16/32-example gradient comparisons at a modest
768-trajectory capture cost. The [protocol](experiments/idr_learning_diagnostic_2026-11-12.md)
owns grouping, seeds, resource estimates and interpretation limits.

Average raw gradients before clipping; compare against a disjoint finite reference
rather than the same samples' pooled mean. Preserve optimizer moments/rates for
one-step copied updates, and decompose head/normalizer effects on fixed reference
trajectories. Do not claim improved generation from lower fixed-data residuals or
select a production setting automatically. Counterfactual context checks hold
actual ProGen features fixed but need not be reachable sequence histories.

Keep original source heads and code intact. Save detached batches and hashes so
analysis can be repeated without another GPU sampling run. No biological scorer,
new reward, backbone fine-tuning or sustained training is introduced.


## 2026-11-12 — Test context routing after mixed gradient diagnostics

The [completed real-state diagnostic](experiments/idr_learning_diagnostic_2026-11-12.md#completed-results--12-november-2026)
shows nearly identical-sign charge corrections across the two blocks, tiny explicit
context response and much larger feature than task-context preactivation norms.
Larger gradient groups improve independent-reference alignment in only three of
six cases, and one-step head effects are mixed. Do not promote gradient accumulation
as a demonstrated fix or interpret normalization progress as better generation.

Propose a controlled task-context output branch against the unchanged head, holding
the scientific target and training budget fixed. This isolates one plausible
intervention before changing loss, batch size or ProGen weights. Routing and added
capacity remain intertwined; results must be judged on generation and not solely
on TB loss. This is a local engineering hypothesis, not an externally established
remedy. Implementation and its exact parameter choices remain pending.


## 2026-11-12 — Minimal direct task-context path, matched fresh fits

The user approved the architecture comparison. Add a bias-free 20-by-5 matrix to
the original residual logits, using the existing five scalar task inputs and no
prompt-identity shortcut, rescaling or extra hidden layer. Zero initialization
preserves the prior and shared initialization without consuming new random draws.
Use the original head optimizer group and schedule: a separate branch rate would
confound the architecture test with tuning. The added path changes capacity as
well as routing, and success would not isolate a unique causal mechanism.

Run fresh standard/direct-context fits at all three seeds with equal update and
trajectory budgets. Share the prior pool and evaluate all raw final-head samples,
including disorder, diversity and block-specific corrections. Reuse unchanged
rollout, reward, update and scorer functions to protect comparability and historical
replay. No results are available yet; the [protocol](experiments/idr_context_2026-11-12.md)
owns settings, evidence, costs and decision criteria.
