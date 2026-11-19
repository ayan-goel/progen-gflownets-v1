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


## 2026-11-12 — Test the direct branch's update scale after weak generation

The [completed direct-context comparison](experiments/idr_context_2026-11-12.md#completed-results--12-november-2026)
shows greater explicit-context sensitivity and correctly signed target coefficients,
but only 17/384 charge and 9/384 joint passes versus the standard head's 20/384
and 11/384. No seed improves either pass count. The direct matrix learned small
weights; it is neither disconnected nor an established generation improvement.

Propose one tenfold rate increase for the added matrix alone, keeping the original
head rate, normalizer, clipping, 400-step budget, reward and frozen ProGen fixed.
This tests update scale without treating a synthetic whole-head rate experiment
as equivalent. The multiplier is an engineering hypothesis, not a paper optimum.
TB/SubTB comparison remains a separate follow-up, with primary motivation linked
in the result record. The new rate test is not implemented or run at this stage.


## 2026-11-12 — One bounded branch-rate follow-up with reused controls

The user authorized the [branch-rate protocol](experiments/idr_branch_rate_2026-11-12.md).
Train three fresh direct-context heads with an added-matrix rate of 0.001 and
retain the original head's 0.0001 and normalizers' 0.01. All rates decay tenfold
after step 300. Use separate nonduplicated Adam groups and the existing joint
norm clipping; retain seeds, initialization, support and reward. No warm start.

Reuse checksum-pinned completed standard/lower-rate controls and the prior pool:
those deterministic fits already supply the intended comparison. Save copied
evidence for CPU replay and distinguish reused proposal costs from new generation.
Judge raw charge/joint success, disorder and diversity across all seeds/prompts;
larger weights or lower loss alone are insufficient. A mixed/negative result ends
this branch-specific test and motivates reconsidering learning, potentially in a
separate training-objective investigation. No further rate sweep or budget extension
is authorized by this protocol. No real follow-up outcomes have been observed.


## 2026-11-13 — Verify the small branch-rate gain without further tuning

The [completed bounded test](experiments/idr_branch_rate_2026-11-12.md#completed-results--verified-13-november-2026)
raises charge passes from 17 to 23 and joint passes from 9 to 13 out of 384, with
charge increases in every seed and no joint-count decreases. Mean score improves
on all eight prompts; disorder/diversity remain similar. This satisfies the
qualitative rationale for further verification, not a claim of statistical or
biological success. The margin over ProGen is only three passes on either endpoint.

Freeze checkpoints/settings and propose larger fresh-sample evaluation of both
branch rates and ProGen, using all three heads and newly fixed random streams.
This isolates sampling variability without more training or a 100× rate sweep.
The comparison remains on the inspected prompt panel and cannot establish new
protein generalization. If the gain does not repeat or remains unhelpfully small,
stop branch-specific tuning and revisit credit assignment separately. The new
evaluation protocol/command remains pending; this is a decision from local
experimental evidence, not a new literature claim.


## 2026-11-13 — Prioritize a training-method comparison over sampling-only verification

The user questioned whether more samples would improve the model. They would not:
the earlier proposal measured uncertainty around a small gain. Retain that reasoning
as history, but defer the standalone sampling run because the active objective is
substantially better guidance. The [targeted SubTB review](../literature/IDP_IDR_REVIEW_2026-11-12.md#8-training-method-priority-after-the-branch-rate-result--13-november-2026)
supports a controlled TB/SubTB comparison as the next proposal, not a guarantee.

Keep the current token policy and reward fixed, add the state-flow estimator needed
for partial-trajectory constraints, and match trajectory/update budgets. Existing
400-step TB results do not establish convergence or an inherent failure of TB.
Use fresh evaluation and learning curves to separate useful generation from lower
training loss. Resolve exact parameterization/weighting/budgets and test correctness
on CPU before any user GPU run. Do not simultaneously fine-tune ProGen, change beta,
or perform another branch-rate sweep. No implementation was started in this review.


## 2026-11-13 — Fix and implement the matched SubTB charge protocol

The user authorized the training-method comparison. The
[protocol](experiments/idr_subtb_2026-11-13.md) fixes all choices before real results:
800 updates in both arms, identical high-rate direct-context token heads, separate
scalar interior flows, prior-factored constraints, uniform segment weighting and
fresh evaluation. The extra budget is matched because 400 steps did not establish
TB convergence. Prior factoring avoids asking a new network to reproduce known
prefix log likelihoods. Separate flow clipping preserves the policy's existing
clipping budget. Neither choice changes the terminal distribution.

This is a bounded test of a plausible training-signal improvement, motivated by
our weak/noisy learning evidence and the primary SubTB paper. The new practical
continuation screen demands useful charge and joint gains, preserved disorder/
diversity and seed consistency. A lower loss or a handful of extra passes is not
sufficient. No automatic extension, rate/lambda sweep or ProGen unfreezing follows.
Tiny CPU correctness checks passed; no real outcome has been observed.


## 2026-11-13 — Stop the SubTB branch; propose training-data reuse

The [matched result](experiments/idr_subtb_2026-11-13.md#completed-results-and-next-decision--13-november-2026)
favors TB over SubTB in every seed and fails the predeclared SubTB screen.
Do not extend that method or sweep its weighting/rate by default. TB still moves
the charge profile weakly, while selection and a finite-pool target estimate
suggest useful outputs and remaining target-fitting headroom.

Propose replay-assisted TB: reuse informative trajectories alongside fresh draws,
with prompt balance, fixed reward/prior/head and fresh final evaluation. This
addresses the current one-update-per-batch training rule; it does not assume the
backbone must be unfrozen. GFlowNet LM replay literature motivates the mechanism,
but replay's benefit here is untested. The result record owns diagnostics and the
next protocol requirements. No new training method was implemented or run.


## 2026-11-13 — Resolve reward alignment before implementing replay

The user interrupted the replay implementation to question whether the reward
itself fits the objective and supplied an RL theory monograph. No replay code had
been written. The [CPU reward audit](experiments/idr_reward_audit_2026-11-13.md)
finds same-prompt score/pass inversions in 21/24 pools, no disorder reward, and a
fixed-target estimate that remains far from mostly passing outputs. It also
retains evidence of imperfect fitting: TB trails the current target estimate.
Replay may address the latter, but not the former.

Defer replay and specify acceptance-aware utility before another training run.
The [reading note](../literature/RL_THEORY_REWARD_AUDIT_2026-11-13.md) distinguishes
target choice, KL regularization and data coverage; it does not claim a recipe or
guarantee for our neural sampler. Beta/bonus calculations are post-hoc target
sensitivity, not selected settings. Keep any future reward comparison on TB with
the same direct-context head, without simultaneous unfreezing or replay changes.
Current executed reward/configuration and historical evidence remain unchanged.

## 2026-11-13 — Replace center matching with bounded joint acceptance

The user authorized investigating and fixing the reward. Adopt a versioned
candidate that prioritizes all charge tolerances plus the existing V3 isolated-tail
screen, with full credit inside the joint acceptable set and positive graded
credit outside it. This resolves the demonstrated score/pass inversion and the
missing disorder objective. It changes the scientific target; it is not a fix to
a charge-calculation bug. Historical commands/runs retain their original reward.

The [reward record](experiments/idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1)
owns every term, default and alternative. The pass/near-failure ratio 20 and
failure floor fraction 0.25 bound the property multiplier to [1/80, 1]; they are
engineering choices fixed before evaluating this candidate, not tuned optima.
Passes preserve relative ProGen probabilities. V3 becomes an optimized proxy,
so it loses independent-evaluation status; V2 remains sensitivity only.

Saved proposals suggest a better aligned target (47.86% joint versus 11.35%),
with a smaller log-factor range but limited proposal coverage. This does not
demonstrate trained improvement. The [targeted literature update](../literature/IDP_IDR_REVIEW_2026-11-12.md#11-bounded-joint-acceptance-reward--13-november-2026)
supports separating objective choice from imperfect fitting and cautions against
assuming stronger rewards solve learning. Implement the scorer and reproducible
CPU audit now; next integrate a bounded TB comparison with the same frozen head
architecture. Keep replay, SubTB and unfreezing out of that comparison.

## 2026-11-13 — Train the revised reward; reuse old fits with fresh evaluation

The user authorized the [joint-acceptance TB comparison](experiments/idr_acceptance_2026-11-13.md).
Train three new heads under the inherited 800-update budget, rates, clipping and
initialization. Reuse the completed old TB heads rather than refitting unchanged
controls. Generate fresh evaluation from both policies and a new prior pool so
the reward-design samples are not reused as final outputs. This remains the same
inspected development prompt panel and reuses old training replicates.

V3 scores only generated tails during training and never receives gradients.
Record its added CPU cost and per-trajectory reward inputs, since bounded rewards
may still fail through scarce successful trajectories. Each policy's residual
uses its own target. A fresh selection baseline uses the new property factor with
stable ties; selection density is not reported as the original proposal density.
Fix the prior audit's practical screen before execution: +5 points joint success,
two improved seeds, no more than 5 points lost uniqueness. The protocol owns all
RNG streams, budgets and limitations. The completed outcome is recorded below.

## 2026-11-13 — Reward alignment alone fails; test reuse of rare successes next

The [completed comparison](experiments/idr_acceptance_2026-11-13.md#completed-results-and-next-decision--13-november-2026)
fails the predeclared practical screen: new TB 3.13% joint success versus old TB
3.65%, with only one improved seed. Fresh prior-pool estimates still favor the
new target, but limited coverage prevents treating them as exact achievable rates.
Do not promote this reward intervention as successful guidance or extend its
training automatically.

Post-hoc saved-trajectory analysis finds 480 successful encounters in 19,200
trajectories; 81.83% of batches contain none, and each example receives one update.
The final model fits failures far better than successes. This supports a bounded
replay pilot as a more targeted intervention than another small rate change:
reuse informative training trajectories alongside fresh draws, holding reward and
architecture fixed. The completed experiment record owns the proposed progression
screen and the previously reviewed Hu/Shen sources. Replay is not implemented;
its mixture, budget and control must be fixed before a new run. Do not use final
evaluation outputs as replay data or confuse memorizing stored successes with
improved generation. A negative pilot should trigger reassessment, not a sweep.

This does not prove replay will work, that head capacity is sufficient, or that
the acceptance reward is easy to learn. The terminal acceptance jump and failure
floor remain potential contributors. V3 remains an optimized proxy, and no
biological validation follows from these results.

## 2026-11-13 — Compare alternatives before committing to replay

**Recommendation refined:** replay remains the best next GFlowNet intervention,
but use a bounded pilot with a simple supervised control on the same frozen-ProGen
guidance architecture. This is a proposal, not implementation authorization. The
question is whether better exposure to useful examples makes TB steer fresh
generation, and whether that head can learn useful steering under simpler training.

The [completed result](experiments/idr_acceptance_2026-11-13.md#completed-results-and-next-decision--13-november-2026)
shows scarce successful encounters and poor fit on successful outputs. It does
not establish that scarcity is the sole cause. Training already exposes position,
accumulated block charge and charged fraction to the head; the missing feature is
not simply a charge counter. Earlier synthetic fitting checks do not establish
capacity on real open-ended continuations. Heavy gradient clipping and a nearly
constant failure reward remain competing optimization explanations.

| Option | What it addresses | Decision |
|---|---|---|
| TB with success-aware replay plus fresh draws | Rare successful trajectories receive repeated training instead of one update | First GFN intervention; directly tests the observed imbalance without changing the scientific target |
| Supervised imitation of successful training sequences, same head | Gives explicit next-residue targets without trajectory-level TB credit assignment | Include as a small diagnostic control, not as a replacement GFN or exact target sampler |
| Another smoother reward / curriculum | Acceptance discontinuity, weak distinctions between failures, difficult early learning | Plausible, but changes the target or training schedule again; reserve until informative examples have actually been learned from |
| Bigger or nonlinear task branch; unfreeze final ProGen layers / LoRA | Representation or policy expressivity could limit steering | Serious alternative if simpler learning also fails; first test the existing head. Backbone adaptation changes scope and must keep a separate frozen reward prior |
| Another SubTB setting, larger batch, learning-rate/clipping sweep, or longer TB | Variance, optimization and credit assignment | Lower priority: earlier loss/rate interventions gave little practical gain; clipping alone does not identify the cause |
| Prefix-success/value predictor, following FUDGE or controlled decoding | Predict whether each partial sequence can lead to a desirable completion and use it to guide decoding | Strong alternate formulation, but introduces another estimator and calibration/coverage problem; broader than the first replay test |
| Generate-and-select or explicit charge constraints | Obtain acceptable candidates directly | Keep practical controls. Selection already achieves 10.42% joint success at four proposals/output, but neither establishes learned sampling of the specified soft target |

Replay has unusually relevant precedent: [Jain et al.](https://proceedings.mlr.press/v162/jain22a/jain22a.pdf),
the original biological-sequence paper, uses TB with mixed offline and fresh
trajectories (§3.2, Algorithm 2). [Shen et al.](https://proceedings.mlr.press/v202/shen23a/shen23a.pdf)
specifically studies oversampling low-reward objects and prioritized replay;
benefits are task-dependent. [Hu et al.](https://proceedings.iclr.cc/paper_files/paper/2024/file/bc667ac84ef58f2b5022da97a465cbab-Paper-Conference.pdf)
uses replay in autoregressive language-model GFlowNets. These support a test, not
a promise of improvement. The [literature update](../literature/IDP_IDR_REVIEW_2026-11-12.md#12-alternatives-after-the-joint-acceptance-result--13-november-2026)
records reading limits and the prefix-guidance alternative.

### What the proposed comparison must distinguish

Keep the reward, frozen backbone and head architecture fixed. Start with one
preselected training seed. Use only that seed's training examples to seed replay
and the supervised control, balanced across prompts where possible. Never put
final evaluation outputs into either training set. The supervised arm maximizes
the likelihood of residues in successful training sequences using the same head;
it is deliberately an easier learning objective, not a target-matching claim.
Do not combine it with TB as an auxiliary loss in this first comparison.

Replayed TB must recompute **current** policy probabilities, preserve the original
terminal reward and mix stored data with fresh full-support draws. Uniform replay
of the predominantly failed buffer would not address the proposed mechanism;
explicit success/near-miss coverage and prompt diversity matter. Prioritizing by
success is our choice, not literal ranking by the full prior-times-property reward.

Fix replay composition, data split, update/example budgets, checkpoints and matched
control before execution. The historical on-policy run is a reference, not a
compute-matched control if this adds updates or training examples. Account for
feature extraction/reconstruction too: saved reward rows are not a complete cache
of frozen per-token features. Cached states may be reused only while ProGen is
frozen. Preserve a training-only holdout for fit diagnostics and use fresh final
generations; this remains development on the same eight prompts.

- **Replay improves fresh success and diversity is retained:** proceed to seed
  confirmation using the existing practical +5-point progression screen.
- **Supervised control improves fresh success, replay does not:** the existing
  head can support useful steering, so prioritize TB optimization/credit assignment
  or a supervised warm start. This does not prove capacity for exact target fitting.
- **Both only fit stored sequences, without improving fresh generation:** the
  failure is generalization/coverage, not solved by low training loss. Reassess the
  state representation and available useful data before increasing compute.
- **Neither fits even its training examples:** inspect head capacity and optimizer
  behavior on the fixed data. It does not by itself prove unfreezing is necessary.

This is one bounded mechanism comparison, not a new sweep. Its numerical budget
and settings remain to be fixed in an implementation protocol. Do not launch it
or change completed code/configuration as part of this literature reassessment.

## 2026-11-13 — Implement a bounded replay comparison with a matched fresh control

The user authorized the [replay/supervised pilot](experiments/idr_replay_2026-11-13.md).
Fix seed 0 and train three fresh heads for 800 updates, with the acceptance reward
and architecture unchanged. Use batch 16 so replay can include eight fresh examples,
four successes and four near-misses while preserving prompt balance. Refit ordinary
TB at batch 16; comparing only with the earlier batch-eight run would confound the
replay intervention with the number of examples per update. Updates and trajectory
presentations match, but fresh generation, historical data exposure and cost do not.

Use a fixed bank from earlier seed-0 training trajectories. This deliberately tests
repeated exposure to a shared informative dataset before adding adaptive buffer
replacement. A tail-grouped hash holdout prevents duplicate leakage. Select near-
misses only after splitting; reserve final generated samples for evaluation. Saved
V3 scores define the same reward without rescoring the historical training data.
The supervised control draws the same successful bank rows and updates only the
same head with mean token NLL; it neither trains a predictor nor unfreezes ProGen.

The protocol owns the bank limits, split repair, random streams, numerical defaults
and cost accounting. Add a predeclared +5-point gain in joint success excluding exact
training copies alongside the existing +5-point raw gain and uniqueness guard.
This protects the mechanism screen from pure memorization; near-copy and per-prompt
checks still matter. One seed can justify confirmation, not establish robustness.
No real outcome was used to choose these settings and no real execution was run
by the assistant. CPU fixtures and read-only preflight establish readiness only.

## 2026-11-14 — Stop replay-only tuning; retain supervised steering as the next control

The [completed comparison](experiments/idr_replay_2026-11-13.md#completed-results--14-november-2026)
shows a small replay gain (+0.88 percentage points) and a larger supervised gain
(+4.79 points) over matched on-policy TB. Both miss the predeclared +5-point
screen. Do not relabel the supervised near-miss as a pass or extend evaluation
until it crosses the threshold. Its fresh success, high uniqueness and absence
of exact training copies nevertheless establish useful steering with the same
frozen backbone/head. Different objectives and successful-example exposure prevent
attributing the entire difference to TB alone.

Every replay batch already contained successes, yet policy changes remained small;
the held-out signed-residual decomposition mostly reflects normalizer differences.
This weakens the case for more replay-only tuning. The next candidate is a bounded
supervised initialization followed by TB with replay, against the unchanged
supervised checkpoint. It asks whether TB can preserve and improve existing
steering. This is an anticipated protocol branch and local hypothesis, not a new
literature-backed guarantee or evidence of exact target matching.

Before implementation, specify the update budget, normalizer initialization and
success/diversity rule. If TB removes the gain or adds no useful steering, revisit
the objective/parameterization rather than extend training. No new experiment is
implemented or launched by this result analysis; ProGen remains frozen. One seed,
previously inspected prompts and optimized disorder proxies limit the conclusion.


## 2026-11-14 — Compare alternatives to supervised initialization

The user asked whether the proposed next step is actually preferable to other
options. The [completed result](experiments/idr_replay_2026-11-13.md#completed-results--14-november-2026)
supports useful supervised steering, not a claim that TB will improve it. Neither
arm passed the prior progression screen. The [source review](../literature/IDP_IDR_REVIEW_2026-11-12.md#13-alternatives-after-replay-and-supervised-control--14-november-2026)
records new reading and its limits.

| Alternative | Mechanism and limitation | Priority |
|---|---|---|
| Supervised initialization, then unchanged TB/replay objective | Starts with more useful generation; cannot itself fix poor TB credit assignment and may lose the gain | One bounded next diagnostic, with continued supervision as a control |
| Continue supervised learning alone | Builds on the best observed steering; still imitation of selected successes, not our soft target or a GFN | Required comparator, not evidence that GFN training works |
| Prefix expected-property guidance | Predict completion quality at each residue and steer locally; new estimator needs representative data and calibration | Strongest structural alternative if warm-start TB adds no value |
| Reward-weighted likelihood / distributional fitting | Fit the same target using direct token likelihood signals; needs correct proposal weighting and adequate coverage | Strong non-GFN reference, not silently a GFlowNet replacement |
| Log-partition variance loss | Avoid jointly fitting prompt normalizers; needs within-prompt samples, and our normalizer is not proven causal | Reserve for evidence of a specific normalizer failure |
| Unfreeze ProGen / enlarge the head | May improve expressivity; changes scope and does not fix a poor objective | Not first: existing frozen features already support some steering, though exact-target capacity remains unknown |
| Another reward, rate sweep or longer replay run | Could change optimization, but lacks a newly identified failure mechanism | Do not prioritize |

**Recommendation:** retain the warm-start idea as a small test of GFN value, not
as the most confidently effective generator improvement. Start from the same
completed supervised head and compare (1) leaving it unchanged, (2) continuing
supervised training, and (3) TB with replay. Evaluate all three afresh. Match the
two continuation arms' update/example budgets, report different success exposure,
fresh sampling and wall time, and preserve reward, backbone and head architecture.
A gain over the starting checkpoint alone cannot establish that TB adds value.

Before implementation, fix a short budget, optimizer/normalizer initialization,
checkpoint selection, meaningful gain threshold and diversity/copying guards.
Record the full optimizer reset/continuation choice. Assess success-stratified
residuals and policy changes as well as fresh property success; matching the soft
target and maximizing pass rate are different claims. A weaker pass rate would
not prove TB has no possible distributional benefit, but absent independently
supported target-fit benefit it would not justify continued development here.

If TB damages the useful policy or provides no practical advantage over continued
supervision, stop this warm-start route and reconsider prefix-level guidance.
Do not automatically extend training or start another loss sweep. This updates
only the proposal: no experiment code, configurations or completed artifacts change.

## 2026-11-15 — Implement the controlled supervised-start comparison

The user authorized the [three-arm continuation](experiments/idr_warmstart_2026-11-15.md).
Use the completed supervised head in every arm; compare no updates, continued
supervision and TB/replay. Preserve the fixed bank, frozen backbone, architecture
and acceptance reward. A 400-update budget is half the preceding pilot, with no
checkpoint selection or automatic extension. Match example/update counts and
explicitly disclose unequal successful-example exposure and computation.

Reset Adam and restart the inherited rate schedule for both trained arms. Carrying
supervised moments into TB would confound the objective switch; asymmetrically
keeping them only in the control would introduce another difference. This is an
intentional restart, not exact optimizer resumption. The supervised head has no
normalizer: initialize TB's prompt offsets by least squares on equally weighted
positive/negative training-bank strata, then train them normally. This avoids an
arbitrary zero offset without treating the selected bank as an unbiased estimate
of the true target normalizer. It is an initialization choice, not a claimed fix
for the previous failures; holdout data are excluded.

Require meaningful raw and noncopy joint gains over **both** fresh controls, with
uniqueness preserved. The protocol fixes the numerical screen and seeds before
real execution. Include inherited training examples in copying checks; compare
fixed-bank relative-target discrepancies without a normalizer as a complementary,
limited diagnostic. This distinguishes useful generation from fitted scalar offsets
without claiming exact distribution matching. Sources and alternatives remain in
the [14 November review](../literature/IDP_IDR_REVIEW_2026-11-12.md#13-alternatives-after-replay-and-supervised-control--14-november-2026).

## 2026-11-15 — Reassess reward design separately from credit assignment

The [RL/GFN review](../literature/RL_REWARD_DESIGN_2026-11-15.md) identifies a concrete
tradeoff in the current acceptance reward: bounded failure weights limit extremes
but compress failure discrimination; the pass bonus aligns the target but creates
a large boundary jump. The earlier redesign did not improve generation, so target
alignment must not be described as a solved learning problem. Replay's limited
gain despite successes in every batch also weakens a pure lack-of-examples account.

Keep the user-running supervised-start experiment unchanged. After its result,
inspect reward discrimination/concentration on designated training artifacts.
If failure weights are largely saturated and a continuous outside-tolerance
candidate preserves useful target mass, consider one matched terminal-weight test.
Otherwise, if TB fails to retain/add steering, prioritize local prefix-flow credit
with the terminal target preserved. Forward-looking GFlowNets and soft-RL/GFN
equivalence provide specific mechanisms; merely adding telescoping bonuses to
full-trajectory TB does not provide them. Do not bundle these changes, promise
improvement, or substitute another loss/rate sweep. The review owns sources,
mathematical distinctions, limitations and stopping logic; no follow-up trainer
or hyperparameter is selected by this research update.

## 2026-11-15 — End warm-start TB; isolate the reward question

The [completed comparison](experiments/idr_warmstart_2026-11-15.md#completed-results--15-november-2026)
gives 9.38% joint success unchanged, 16.11% continued supervision and 7.03% warm TB.
The predeclared TB screen fails. Do not extend this training or confirm more seeds;
retain the supervised continuation as the best observed generation baseline.
Matched starting weights do not remove differing success exposure or objectives.

Unlike earlier normalizer-shift diagnostics, the selected-bank relative-target
error improves independently of the normalizer. Thus this is not evidence that
TB learned nothing. It is evidence that improved limited target fitting failed
to improve the desired generation criterion; exact global target matching remains
unverified. Do not automatically attribute the result to capacity or credit alone.

Post-hoc training reward inspection supports a specific alternative: many fresh
failures lie near the floor, although selected replay near misses do not. Assess
one continuous outside-tolerance candidate for target preference and concentration
on designated training proposals before committing to a matched reward-only test.
The selected/nonstationary bank is not an unbiased prior sample. The [review](../literature/RL_REWARD_DESIGN_2026-11-15.md)
supplies mechanisms and limitations; local prefix-flow learning remains the
structural alternative. No new penalty, trainer or GPU run is selected by this
result analysis. Keep the two interventions separate and preserve all results.


## 2026-11-15 — Isolate continuous terminal reward under unchanged TB

The user explicitly retained GFlowNet training and authorized the reward-only
comparison. The [protocol](experiments/idr_continuous_2026-11-15.md) owns design,
calibration, exact parameters and stopping. Remove the failure floor and pass
bonus using log factor `-2.5 V`; retain outside-tolerance penalties and every
acceptance criterion. This changes the target, not its differentiability to TB.

Calibrate on unselected historical training proposals with prior/behavior ratios,
not selected replay or generated evaluation samples. A one-dimensional matching
calculation gives 2.5099; round to 2.5. A separate training seed gives similar
aggregate target preference, but one scarce prompt has low effective coverage.
Retain that limitation rather than claim a reliable attainable success rate.
Larger negative log weights are an explicit optimization risk of removing the floor.

Reuse the completed current-TB checkpoint and original common starting head;
train only one new candidate. Matching the original initialization, training seed,
replay order, optimizer and budget isolates the reward intervention. Fresh common
sampling of all arms avoids comparing new outcomes against old sample estimates.
Use the same normalizer initialization rule with each reward's labels; equal
numeric offsets would not have the same meaning across targets. Require gains
over both controls, not merely recovery from a degraded TB head. Keep local-flow
changes separate, and stop after the declared comparison if it fails.


## 2026-11-15 — Stop reward-only tuning after the continuous comparison

The [completed comparison](experiments/idr_continuous_2026-11-15.md#completed-results--15-november-2026)
finds 9.57% joint success versus 9.86% at initialization and 7.52% for previous TB.
It fails the prespecified practical screen. Better results than a degraded control
are insufficient; neither another penalty sweep nor more of the same training has
credible support from this result. Limited target-fit improvement is not useful
generation improvement or proof of exact target matching.

Retain the GFlowNet goal. Next assess prefix-flow/local-credit parameterization
under a fixed terminal target, with a small exact CPU check before a real run.
The mechanism must differ explicitly from the prior unsuccessful generic SubTB
comparison. The [existing review](../literature/RL_REWARD_DESIGN_2026-11-15.md)
supplies motivation, not a performance guarantee. Representation, exploration,
optimization and target coverage remain competing explanations; the observed
failure does not prove reward correctness or uniquely identify credit assignment.
No new experiment implementation or GPU command follows automatically.


## 2026-11-16 — Diagnose a representable finite task before another full run

The user authorized separating target quality, head capacity and learning. The
[exact cached-IDR diagnosis](experiments/idr_exact_diagnosis_2026-11-16.md) replaces
an immediate prefix-flow proposal with a controlled failure to investigate.
Actual cached features and the unchanged reward define a selected finite tree;
its exact target improves success. The ordinary teacher and both TB procedures
underfit after the fixed budget. A post-hoc algebraic same-architecture fit reaches
mean TV 1.90e-6, establishing representation on these states without relying on
optimizer convergence. This is a capacity witness, not a new generator.

Prioritize explaining update/conditioning behavior on this fixed CPU task before
another full-length GPU experiment. Expected TB also underfits, so sampling noise
is not sufficient; most tree decisions are early, so a purely long-horizon account
is insufficient here. Do not infer a unique production failure cause. Local masking,
selected support and forced suffixes preclude treating these as open-generation
results. Sources are the linked protocol/results and implementation, not a new
literature claim. No additional sweep or production change was executed.


## 2026-11-16 — Repair finite-task optimization before returning to generation

The user authorized inspecting why exact-task learning fails. The
[update diagnosis](experiments/idr_update_diagnosis_2026-11-16.md) measures highly
correlated inputs, output-layer-dominated gradients and sensitivity to update scale.
A fixed full-support TB objective isolates optimizer behavior from changing proposals.
Removing clipping alone is worse. L-BFGS improves but misses the local gate; a
subsequent explicitly post-hoc invertible input-coordinate change preserves the
starting function and lets the same bounded L-BFGS fit pass on all eight prompts.
The transform uses initial training features, not oracle next-token targets.

Retain the GFlowNet/TB objective. Treat the repair as a finite deterministic
optimization result, not improved IDR generation or proof of a unique failure cause.
Next test conditioning under sampled TB on the same small task before a short
fully branching ProGen capture. Do not port fixed-batch line search to noisy changing
batches or launch another full-length run yet. The linked protocol owns parameters,
controls, stopping criteria, sources and measured outcomes; no sweep was performed.


## 2026-11-16 — Conditioning alone does not repair sampled Adam

The [bounded paired comparison](experiments/idr_sampled_conditioning_2026-11-16.md)
finds lower error in all three conditioned seeds, but none passes the local gate.
Keep the positive deterministic-TB evidence separate from stochastic fitting and
open generation. Before a new GPU run, the proposed next control averages the
same TB update exactly under conditioned Adam. This separates finite-sample noise
from update behavior without changing reward or supervising next-token targets.
Do not infer a noise-only cause from the L-BFGS-versus-sampled-Adam gap or extend
this result into an unplanned rate/budget sweep. The protocol owns numbers and
conditional decisions; sources are the linked local controls.


## 2026-11-16 — Exact gradient averaging identifies a useful sampled-learning direction

The [expected-gradient control](experiments/idr_expected_conditioning_2026-11-16.md)
passes the finite gate under the same conditioned head, Adam and 400-step budget.
Removing draw noise while retaining the current-policy/replay weighting rule is
sufficient here. Prioritize a bounded sampled gradient-averaging approximation;
account for extra trajectories rather than claiming equal compute. This is stronger
motivation than another reward change or blind longer fit. Keep local success
separate from open generation, and do not attribute it solely to clipping or
assume an adequate batch size without testing. The protocol owns metrics and
limitations; the proposed sampled approximation is not yet implemented.


## 2026-11-16 — Assess variance sources before increasing rollout cost again

The [16-pair sampled comparison](experiments/idr_averaged_sampling_2026-11-16.md)
shows substantial consistent improvement but fails the local gate at 16-fold
sample cost. Keep the successful exact expectation as a reference and stop the
predeclared fits. Next propose a fixed-checkpoint decomposition of fresh/replay
raw-gradient variance, to decide whether cached replay can reduce noise without
proportional new rollouts. The current results do not identify that source or
establish sample efficiency; do not automatically enlarge batches or launch GPUs.


## 2026-11-16 — Correct sampled gradients and stop tuning the solved finite task

The [variance audit and repair](experiments/idr_variance_repair_2026-11-16.md)
finds substantial fresh and replay noise. A fixed-proposal TB control with ordinary
Adam still underfits, while a snapshot gradient correction passes in three seeds
and three predeclared fresh confirmation seeds. This supports the correction as a
local repair beyond merely changing visitation. Keep the GFlowNet objective;
record the change to fixed-bank proposals and extra gradient cost explicitly.
The protocol links the SVRG primary source and limits of applying it to Adam.

Stop CPU tuning on the known task. The next meaningful boundary is new-state,
fully branching transfer with fresh generated evaluation and coverage checks.
Neither fixed-bank success nor a folded checkpoint establishes that transfer.
All GPU execution remains user-run; no production model was changed.


## 2026-11-16 — Test the local repair on full-support generation

Six successful finite fits justify one transfer pilot, not further local tuning.
The [protocol](experiments/idr_transfer_2026-11-16.md) compares unchanged initialization,
ordinary offline TB and snapshot-corrected offline TB. Match proposal, sampled
indices, conditioning and optimizer budget so the trained-arm difference isolates
the gradient correction; record its extra computation. Reuse the 408 training
trajectories, frozen backbone and continuous reward. Do not add online replay,
unfreezing or another reward change before this mechanism is assessed.

Remove every finite-tree action mask and score all 64 decisions over 20 residues.
Adapt conditioning to dependent full-path states using a training-only second
moment with a predeclared gain cap; this adaptation is not the original finite
transform. Require fresh generation, noncopy success and diversity alongside bank
residuals. The learned normalizer and unobserved paths make offline bank fit
insufficient to identify the full target. This pilot tests that limitation directly;
it does not promise transfer or compare against every historical baseline.

The linked protocol owns budgets, practical progression thresholds and the SVRG
source. If bank fitting improves without fresh success, investigate coverage and
extrapolation; if full-path bank fitting itself fails, investigate the adaptation.
Neither outcome automatically justifies longer training or another reward sweep.


## 2026-11-16 — Stop fixed-bank transfer after failed fresh generation

The [completed transfer pilot](experiments/idr_transfer_2026-11-16.md#completed-results--16-november-2026)
fits the training paths almost exactly with both ordinary and corrected TB, yet
fresh joint success stays below initialization. The common cached holdout improves
partly but retains a large error gap. This changes the diagnosis: further reduction
of training-bank loss or draw variance is not a supported next intervention.
Prioritize coverage/generalization with newly visited prefixes and a held-out
trajectory gate, retaining the GFN objective. The evidence does not uniquely
identify the cause or establish globally adequate reward design. Earlier online
TB failures mean that refreshing data alone is not a proven fix. A new bounded
protocol is needed before another real experiment; no extension is automatic.


## 2026-11-16 — Separate reference search from learning its sampler

Tom's GPA paper motivates a [training-free reference proposal](../literature/GPA_AND_IDR_NEXT_STEPS_2026-11-16.md)
before another head-training variant. The latest bank fit is already near exact;
more steps cannot establish useful unseen-path behavior. GPA keeps generator and
scorer fixed and searches a population; its practical recipes are not all exact
target samplers, and its DNA results do not establish IDR efficacy. Keep the GFN
as the eventual amortized sampler, with search as a comparator and possible source
of refreshed trajectories, not a silent project replacement.

Exact rejection is available for our bounded property factor, but a post-hoc
saved-prior check suggests poor acceptance for some prompts. The review records
that alternative, low effective sample sizes and cost uncertainty. A bounded
reference experiment must distinguish raw outputs from selected designs, assess
cost/diversity, and not use final evaluation scores for selection. No sampler or
new training protocol is implemented; existing execution restrictions remain.


## 2026-11-16 — Recenter on GFlowNet IDR generation and test representation adaptation

The user clarified that the next intervention must directly advance GFlowNet-based
IDR generation. The preceding GPA-first proposal is therefore deprioritized;
GPA remains background/comparator literature, not the prerequisite or main path.
The user subsequently authorized implementing this bounded comparison; real
execution remains user-run. Existing completed runs and their frozen-backbone
contract are intact. The [implementation protocol](experiments/idr_lastblock_2026-11-16.md)
owns the budget, selected settings and practical gates.

**Recommended next comparison:** retain the continuous charge/disorder reward and
ordinary TB, and compare the existing frozen-ProGen guidance head against the
same head trained jointly with **the last ProGen transformer block**. Freeze the
earlier blocks and original output projection in this first design. Updating the
last block changes contextual prefix representations, whereas LM-head-only tuning
mainly changes the readout of the same representations. Full last-block updates
are the simplest proposed intervention; a low-rank adapter is a memory/regularization
alternative to specify before implementation, not a simultaneous extra arm or sweep.

This is a testable representation/generalization hypothesis, not a diagnosed
capacity failure. The 128-unit residual head already fits the bank; the historical
same-head supervised result also shows some useful steering with frozen features.
More parameters on the same 408 trajectories could worsen overfitting. Earlier
continuous TB already used fresh trajectories, so online data alone is not new.
Use one shared, bounded, refreshed trajectory/replay pool for both comparison
arms, with declared contributions from their policies and fixed reference; pair
training examples and budgets. This tests architecture under that common training
regime, not architecture versus the previous static-bank result.

Keep an immutable original ProGen reference for terminal prior probabilities.
Only the sampling policy's upper block/head change, so the reward target does not
move with the policy. Sampling itself can use no-grad; recompute trajectory log
probabilities with gradients through the trainable block. Detached cached final
features cannot provide that gradient. No new auxiliary loss, reward change,
SubTB swap, SVRG extension or whole-backbone unfreezing is bundled into this test.

Use fresh unselected generation as the main gate: joint charge/predicted-disorder
success, disorder retention, diversity/copies and compute cost. Inspect common
held-out trajectories and prior drift as diagnostics. Freshly evaluate both the
unchanged starting head and the saved continued-supervised head (historical16.11%)
as controls, without retraining them. Training loss alone cannot qualify success.
Fix the run budget and practical effect threshold before any execution; no automatic
longer fits or additional layer sweep follows failure.

**Alternatives considered:** structured local prefix-flow learning remains a
GFN-native option if representation adaptation fails or credit diagnostics motivate
it. Earlier generic SubTB did not implement forward-looking prefix energies, but
local constraints alone on the same bank would not resolve unseen-prefix coverage.
Do not combine this new objective with backbone adaptation in the first test.
A larger readout is a weaker test of sequence representation; another reward sweep
or reference-search detour does not directly test the chosen mechanism.

Primary precedent: [Hu et al., Amortizing intractable inference in large language models](https://arxiv.org/html/2310.04363v3),
§3.3, its LoRA footnote and Appendix B, inspected 16 November. It trains a language
model policy with a GFN objective, using parameter-efficient adaptation and mixed
policy/replay trajectories. This supports adapting a PLM within a GFN, not guaranteed
IDR gains or our last-block-only choice. [Pan et al.](https://proceedings.mlr.press/v202/pan23c.html)
and the existing [local-credit review](../literature/RL_REWARD_DESIGN_2026-11-15.md#b-keep-the-terminal-target-teach-completion-quality-at-each-prefix)
own the alternative's evidence and assumptions.

## 2026-11-16 — Confirm the positive last-block generation result

The [completed pilot](experiments/idr_lastblock_2026-11-16.md#completed-results--16-november-2026)
passes the predefined screen. Adapted TB improves joint success by 8.01 percentage
points over the matched frozen arm and 6.54 over saved supervision, with all eight
prompts favoring adaptation over frozen TB. The 209 successful outputs are not
exact training copies. This supports a useful effect of limited policy adaptation
under the shared refreshed-data curriculum; comparison to earlier fixed-bank
experiments does not isolate unfreezing alone.

Propose **two fresh training/replay/evaluation seed sets with the same protocol**,
each restarting from the original supervised head. Keep RPB1 in the panel despite
its repetition, and retain the fixed final checkpoint and every outcome. This
tests whether the material gain survives fresh data-collection randomness, not
unseen prefixes or different supervised initializations. The two additional runs
were proposed as a bounded confirmation, not an automatic extension of the
completed run. The user authorized implementation on 17 November; the
[confirmation protocol](experiments/idr_lastblock_confirmation_2026-11-17.md)
fixes both new seed sets and the decision rules before real execution.

Do not change the reward, loss, layer count or learning rates now. The result
already supplies a credible effect to confirm; more interventions would obscure
whether it reproduces. The full result record also preserves the contrary
common-holdout target-fit diagnostic, ongoing clipping and prompt-specific
repetition. Passing generation criteria is not proof of exact reward sampling,
experimental disorder or a general biological property.

The user's baseline question also identifies a missing final-evaluation control:
the unchanged arm was a supervised guidance head, not plain ProGen. Historical
plain-ProGen evaluation gives 1.66% joint success under the same acceptance and
generation conventions, but it is not a new matched evaluation. Include a fresh
plain-ProGen arm in the confirmation, keeping training and original gates fixed.
The added baseline has its own practical screen, with the same five-point
improvement and quality-preservation conventions. Require both new repeats to
pass both screens; do not pool them to conceal a failed replicate. The existing
frozen-GFN control answers the unfreezing question;
the saved head-only supervised comparator cannot establish superiority over
capacity-matched non-GFN last-block fine-tuning. The
[comparison clarification](experiments/idr_lastblock_2026-11-16.md#plain-progen-comparison--post-hoc-clarification)
owns sample counts and reuse limits.


## 2026-11-17 — Retain last-block adaptation after both confirmations pass

Both [fresh-seed confirmations](experiments/idr_lastblock_confirmation_2026-11-17.md#completed-results--17-november-2026)
pass the unchanged progression screen and the separate fresh-prior screen.
Adapted joint success is 20.41% and 23.14%, versus 10.45% and 11.04% matched
frozen TB and 2.25% and 2.15% plain ProGen. This supplies the missing repeatability
and current baseline evidence for a useful effect on the fixed task. Preserve both
results; the old pilot does not count toward this confirmation decision.

Retain the copied-final-block GFN method with its fixed original prior. Propose
a separately trained second charge profile on the same eight prompts, under a
bounded protocol that fixes the new target, acceptance criteria, starting head
and replay bank. This tests transfer of the training recipe to a different request,
not arbitrary target control by the current checkpoint. The head currently uses
eight explicit prompt identities, so replacing prefixes is not a clean new-prompt
test; define unseen-identity handling separately before that experiment.

This follows the successful branch of the predeclared rule and does not authorize
new real experiments automatically. Further reward/loss/layer sweeps are not the
next question. Repetition on RPB1, imperfect target fitting, the fixed supervised
initialization and development prompt reuse constrain the claim. Independent
biological assessment and a capacity-matched non-GFN adaptation comparator remain
separate unresolved questions.

## 2026-11-17 — Prioritize improving current charge success before a second profile

The user asked whether the confirmed 20–23% result can improve meaningfully.
The [post-hoc diagnosis](experiments/idr_lastblock_confirmation_2026-11-17.md#post-hoc-headroom-analysis--17-november-2026)
finds charge failure, not predicted disorder, dominates. Two of three training
curves improve late, while one is mixed. This supports one bounded larger-budget
test of the successful architecture, rather than treating 400 updates as converged.

Propose 800 updates from the original initialization, keeping the same reward,
architecture and 75%-of-run learning-rate decay convention; freshly compare with
preserved 400-update controls. This changes time at the initial rates as well as
total updates. The linked analysis owns rationale, provisional +5-point practical
screen, quality checks and the stop at 800. No trainer/configuration is changed
or real experiment authorized by this recommendation.

If that plateaus, consider a separate 2.5-versus-5 property-penalty comparison.
Stronger penalties can favor accepted sequences but cannot guarantee better
learning. Diverse adaptive replay is plausible but not an established bottleneck;
more unfreezing lacks a demonstrated capacity need. A second profile remains a
later generalization question. This supersedes its priority in the preceding
decision, preserving the completed evidence and the GFlowNet generation goal.


## 2026-11-17 — Authorize one stretched-budget comparison on the working generator

The user accepted the preceding recommendation. The
[budget protocol](experiments/idr_lastblock_budget_2026-11-17.md) fixes two new
800-update paired fits, the original confirmation training/replay seeds and fresh
final-evaluation streams. Keep reward and architecture fixed; stretch the existing
75%-of-run decay convention to update 600. This tests additional training with
more time at the initial rates, rather than only extending the low-rate tail.

Use the preserved 400-update adapted checkpoints as fresh-evaluation baselines.
New update-400 checkpoints cannot replace them because schedules differ after
300. Restart from the original initialization, since saved weights lack optimizer
and RNG state for exact resumption. The frozen arm remains necessary as a source
of the common refreshed data. Give all six evaluation arms one union of original,
old-run and new-run training tails for copy-aware success, retaining failures in
the denominator. Require a five-point raw/noncopy gain and bounded quality loss
in both repeats; stop at 800 even if results disappoint. This is a bounded test
motivated by the observed late learning, not a claim that more training must help.

## 2026-11-17 — Benchmark the objective and prepare one functional IDR task

The user's research question is novel IDRs with requested properties and an
advantage over existing methods, not charge success alone. The
[focused review](../literature/IDR_BENCHMARK_ROADMAP_2026-11-17.md) identifies
GOOSE direct design and IDiom post-training as substantive alternatives. Our
successful method already fine-tunes ProGen's final block; its head-only
supervised control cannot isolate a GFN-objective advantage at equal capacity.

After the authorized budget decision, propose matched final-block supervision and
costed generate-and-select before more tuning. Follow with one contrasting charge
target to test reuse of the training recipe, rather than accumulating charge
variants. One matched KL-regularized RL comparator is needed for the broader
method claim; a GFN diversity advantage must be measured, not assumed. This
prioritization replaces the earlier conditional penalty increase as the default
follow-up without changing the current protocol or authorizing new runs.

Prepare the functional route in parallel: acidic transcriptional activation has
measured tile assays and public PADDLE/TADA predictors. The review owns source
citations, input-length constraints and proposed data/scorer checks. Advance only
after validating the reward's usefulness on appropriately held-out measurements
and defining evaluation not optimized during generation. No arbitrary charge-pass
threshold is required. Tool agreement supports predicted function; actual new
biological function requires measurement. External-reference novelty and diversity
must also be assessed separately from passing the reward. This is a research
roadmap, not a functional-task implementation or a claim of method superiority.

## 2026-11-18 — Retain the 800-update recipe after both budget screens pass

The [completed comparison](experiments/idr_lastblock_budget_2026-11-17.md#completed-results--18-november-2026)
increases raw and common-union noncopy joint success by 9.67 and 6.05 percentage
points, with disorder/uniqueness changes within the predefined limits. Retain
800 updates with decay after 600 for the current task, following the protocol's
successful branch. This improves mean success from 21.78% to 29.64%; it does not
isolate duration from schedule or establish cost efficiency after doubled training.

The gains are broad across prompts and are not explained by exact training copies.
Charge remains limiting; RPB1 repetition and imperfect diagnostic target fitting
remain. No inference of biological function or arbitrary property control follows.
The user requested evaluating this result before choosing further work, so the
earlier literature roadmap remains optional rather than a committed experiment
queue. No automatic longer run, reward change, new property or baseline is selected.

## 2026-11-18 — Consider one further budget test after inspecting late learning

At the user's request, the [late-learning analysis](experiments/idr_lastblock_budget_2026-11-17.md#post-hoc-late-learning-analysis--18-november-2026)
checks saved fresh adapted-policy draws. Success improves in both final 200-update
windows, but the final 100-update trends disagree and mean violations flatten.
Further gains are plausible, not established. Falling TB loss alone is insufficient.

Propose one bounded 1,600-versus-800 comparison rather than assume convergence or
launch an open-ended extension. The preceding material gain and broad late trend
motivate this option; doubling and the inherited practical screen are engineering
choices. Preserve 800-update controls and the reward/architecture, declare the
stretched schedule and stop if the paired improvement/quality checks fail. This
proposal awaits authorization and its own protocol; 800 remains the retained
recipe and no further training or implementation has begun.

## 2026-11-18 — Authorize the bounded 1,600-versus-800 comparison

The user accepted the preceding proposal. The
[new protocol](experiments/idr_lastblock_1600_2026-11-18.md) fixes two paired fits,
the original initialization, unchanged reward/architecture and fresh evaluation
streams. Preserve the relative decay convention, moving it to after update 1,200.
The completed 800-update checkpoints remain fresh-evaluation controls; no weight-only
continuation is presented as exact optimizer resumption. Require the same five-point
raw/noncopy gain and bounded quality losses in both repeats, and stop at 1,600.

Keep the earlier 400-update training tails in the common copy union alongside
800/new histories. This preserves the previous exposure boundary and avoids giving
new outputs an easier novelty definition. It is conservative common-exposure
accounting, not evidence that all models trained on the entire union. 800 remains
the retained recipe until results justify replacing it; the wider benchmark and
functional-property roadmap remains deferred.

## 2026-11-18 — Stop extending the budget after modest 1,600-update gains

The [completed comparison](experiments/idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026)
raises joint success by 4.79 and 1.66 points over freshly evaluated preserved
800-update controls. Quality checks pass; mean success increases from 29.39%
to 32.62%, with no successful exact training copies. Both repeats miss the
predeclared five-point gain. Retain 800 as the default and stop budget extensions,
while preserving the higher-scoring 1,600-update models and all outcomes.

This is diminishing marginal improvement for doubled training exposure, not zero
benefit or proof of a hard ceiling. Do not revise the threshold for the first
repeat's near miss or use mixed post-hoc training windows to choose a checkpoint.
The next proposed comparison is supervised fine-tuning of the same final block
and guidance head with equivalent scored-data access; the current head-only
supervised control confounds objective with capacity. A contrasting profile would
then test reuse of the recipe. These remain proposals, not authorized new runs.

## 2026-11-18 — Keep 1,600 as the working budget; prioritize the missing objective control

The user chose to retain the completed 1,600-update models and budget and defer
further budget/checkpoint tests. This supersedes the initial 800-update retention
recommendation above. Preserve the reported five-point gate failures: choosing the
best observed models for subsequent work does not make that earlier test pass.
No additional training or checkpoint selection follows this decision.

Recommend matched supervised fine-tuning on the existing charge target before
another profile. The current GFN adapts the final ProGen block and guidance head;
the existing supervised control adapts only the head. Its lower success therefore
cannot establish an objective advantage. A new comparison should match adaptable
parameters, initialization and scored-data access, including refreshed examples,
and predeclare supervised selection/weighting. Report raw joint success, diversity,
copies and actual scoring/training/generation costs; equal updates alone are not
equal computation. A retrospective fit to GFN-discovered examples is a useful
diagnostic but would not establish independent-pipeline efficiency.

The [benchmark roadmap](../literature/IDR_BENCHMARK_ROADMAP_2026-11-17.md#direct-benchmark-design--follow-up-clarification)
owns these proposed controls. [Hu et al.](https://arxiv.org/html/2310.04363v2)
provide precedent for comparing GFN language-model adaptation with supervised
and reward-based objectives, not evidence of an advantage on our IDR task.
Costed generate-and-select supplies a complementary practical baseline. Then
one reversed two-block charge profile can test reuse of the recipe under another
request; it would require retraining and does not establish zero-shot control.
GOOSE and one audited reward/preference implementation remain broader comparator
proposals. No new comparator or profile experiment is implemented or authorized
by the user's request for advice.

## 2026-11-18 — Authorize a capacity- and data-matched objective comparison

The user selected the matched supervised comparison. The
[protocol](experiments/idr_objective_2026-11-18.md) fixes two new GFN/supervised
pairs at 1,600 updates, each starting from the original common initialization.
Sharing discoveries from both policies prevents one objective receiving a better
or newer scored dataset. Replacing the old frozen-head proposer changes the
training mixture, so the old 32.62% is historical context, not a matched endpoint.

Use reward-weighted token likelihood for the supervised arm rather than restricting
it to rare perfect successes. Normalize the existing property factor within prompt
and before microbatching, maintaining equal prompt weight even when some are harder.
The proposal distribution is shared and changing; this baseline does not estimate
the exact prior-weighted target or provide an independent-pipeline cost comparison.
These are predeclared engineering choices; inherited matched rates do not establish
optimal tuning for either objective. Keep the full raw-success/diversity/cost tradeoff.

Save the same block/head capacity in both arms, with GFN-only normalizers. Atomic
replacement of one latest checkpoint per arm limits storage and avoids truncating
the previous checkpoint on write failure. Retain only final weights when training
completes; checkpoint sweeps and automatic training resumption are out of scope.
No new target or external RL/direct-design implementation is bundled into this test.
