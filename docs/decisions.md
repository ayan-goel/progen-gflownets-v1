# Decision record

This is a chronological record; newer entries are appended below. For the active
stage use [current status](status.md); use the [experiment index](experiments/README.md)
to find each protocol, result and supporting evidence.

## 2026-10-14 — Scope and execution

- Implement in v1 only. Keep the old test project unchanged; reuse its verified
  checkpoint as a read-only input and vendor the minimal upstream ProGen source
  with attribution. No dependence on old runnable scripts.
- The old AGENTS milestone described TF-Bind-8. Updated it to the user's current
  integration request; preserved the prohibition on agent GPU execution.
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

## 2026-10-14 — Documentation authority and alignment

- The spec defines the scientific contract; README summarizes it; the literature
  review explains its evidential basis; the experiment plan expands its A–E sequence.
  None is an alternative project proposal. Concrete numerical settings belong in
  configurations and the implementation documentation.
- Replaced superseded GB1 recommendations and linear-only first-head instructions.
  Preserved external-paper facts, original retrieval dates, reading limits and
  standalone evidence separately. Added Mega-scale to the bibliography rather than
  implying it was included in the original 28-reference search.
- Implementation remains authorized; only real experiments/GPU work are user-run.
  Documentation alignment does not reset completed CPU checks or establish pending
  real-checkpoint convergence and biological-validation results.

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
or submit jobs. Verified the v1 activation path and pinned imports with CUDA hidden;
updated the runbook to show this workflow explicitly. No training behavior changed.

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

## 2026-10-15 — Investigate the target before additional guidance training

Performed literature research and saved-tensor CPU analysis only. The baseline
places 96.02% of mass on mutations at positions 1–3; a uniform-logit control on the
same legal-action tree places 96.67% there. A complete-path 20-amino-acid likelihood
counterfactual is even more concentrated overall (95.74% on one candidate).
Therefore neither interpreting early-position mass as learned biology nor simply
switching likelihood definitions is justified. Preserve the implemented target.

Record a proposal for development-only scorer/prior comparisons, followed by fresh
multi-parent confirmation with matched proposal budgets. Proposed numerical
choices and their rationale are in [the next-step plan](next_steps_2026-10-15.md).
They are not implemented settings. Keep frozen ProGen, fixed per-fit rewards and
the existing GFN head; no spec/configuration, training code or run outputs changed.
The [literature update](../literature/POST_PILOT_REVIEW_2026-10-15.md) adds eight
sources with explicit reading limits. No GPU execution or real training occurred.

## 2026-10-15 — Implement the authorized development diagnostic

Implement the next bounded stage, not automatic follow-on confirmation training.
Freeze three development and five reserved parents by coverage, disjoint clusters
and global sequence-identity screening. Record exact alignment settings, all
selected/rejected identities and split hashes. Pin the already-installed Biopython
1.86 dependency. Preserve the old outer split; add a seed-43 20% inner split so
checkpoint selection does not consume the outer development assessment. Keep the
reference in fitting and do not refit on inner validation after selection.

Reuse the existing scorer fitter through ridge-only and MLP-only configurations;
leave its implementation and previous behavior unchanged. Compare ridge, MLP seed
zero and a five-member mean. Cache complete-sequence causal logits instead of
recomputing each prefix; verify equality with the existing local prior on a
synthetic ProGen fixture. Compute alpha={1,.5,.25} exact targets, uniform diagnostic
and construction controls; do not modify the GFN's implemented prior. All
comparisons use 256 proposal draws and 32 slots with duplicates reported.

Keep failed scorer screens visible instead of suppressing domains. Save weights,
permitted split rows, caches, predictions, input/source hashes, timing and CPU
reevaluation. No automatic best setting, positive biological gate or confirmation
job is introduced. The [implementation record](experiments/target_diagnostic_2026-10-15.md)
documents numerical choices and their limits; the spec only adds this diagnostic
stage and corrects stale status. All prior run artifacts remain untouched.

## 2026-10-15 — Development diagnostic completed and reviewed

Verified successful non-smoke completion (62.08 s), six passing inner scorer fits
per parent, immutable input/source hashes, and byte-identical saved evaluation
reproduction on CPU. At the original prior, single-MLP exact targets improve
measured development means by +0.28386 / +0.04515 / +0.14938 kcal/mol. The ensemble's
average gain is nearly identical; flatter priors broaden coverage but can lower
stability versus original ProGen. Matched generate-and-select remains competitive.
The 1O6X development support and scorer split differ from the earlier failed test
pilot, so this does not establish that the earlier failure was fixed.

Recommend a single-MLP alpha=1/beta=1 primary for the next reserved confirmation
implementation. Preserve inner selection and all five parents; no new training
configuration or run was created in this review. No confirmation row files were
read, real fits performed or GPU operations launched by the agent. See the
[result record](experiments/target_diagnostic_results_2026-10-15.md) and
[CPU audit](verification/target-diagnostic-result-analysis.json).

## 2026-10-16 — Implement fixed five-parent confirmation

The user authorized the recommended single-MLP/original-prior confirmation. Reuse
all scientific core functions and freeze alpha=1, beta=1, scorer seed zero and the
existing guidance schedule/seeds. Prepare identity-only candidate supports from
source availability/identity metadata to avoid opening reserved label files during
fitting. Keep original panel roles/splits and every parent.

Preserve the existing scorer validation gate: failure skips both guidance learners
for that parent, but prior/target diagnostic rows and failure counts remain visible.
Require every parent's fitting phase to finish before recording hashed final fit
artifacts and releasing test outcomes. Numerical interruption before completion
keeps outcomes unrevealed. No automatic resume, new tuning loop or follow-on run.
Use matched 256-proposal/32-slot evaluation with explicit duplicates, exact target
reference, raw versus selected results and parent-level aggregation. Record cached
setup and CPU evaluation costs separately. The source protocol and all numerical
choices are in [the handoff](experiments/confirmation_plan_2026-10-16.md).

All work executed by the agent was CPU data preparation or synthetic correctness
testing. Thirty-seven tests pass; no real confirmation fit or GPU operation ran.

## 2026-10-17 — Mandatory documentation and stable navigation

At the user's request, make documentation part of every experiment, investigation,
implementation change and decision. Add explicit rules to [AGENTS.md](../AGENTS.md#research-documentation-required)
for pre-run protocols, commands/provenance, numerical choices, results/failures,
motivating primary sources or local evidence, honest reading limits and current
indexes. Engineering defaults must be identified rather than given invented
literature support. This policy is motivated by the user's reproducibility and
organization requirement, not by an external paper.

Use one current-status page and grouped documentation, experiment and verification
indexes. Retain existing record/artifact paths to avoid breaking links from saved
runs. Add a compact reusable record template. Separate historical plans from current
execution and preserve negative/partial results. Correct stale entry-point wording
and record the user-reported confirmation launch/NVML observation without claiming
completion or diagnosing the unknown host issue.

No scientific definitions or experimental parameters changed. The spec's metadata
now points to current status instead of duplicating changing execution summaries.
Do not rerun model tests for documentation-only work: validate links and compare
protected code/config/snapshot hashes. The active experiment remains untouched.

## 2026-10-17 — Mechanism confirmed; comparative utility remains open

The [completed confirmation](experiments/confirmation_results_2026-10-17.md)
supports the target and its GFN approximation on all five finite supports. Retain
this positive result alongside mixed selection comparisons, severe concentration
and higher fitting cost than supervised guidance. More training on this same
support is not the priority: approximation error is already small. The next
proposed development design should address distinct-sequence utility and total
compute, audit prior position bias, and establish scorer validity before expanding
mutation distance. Fresh evaluation splits are necessary before new tuning; these
confirmation outcomes have been inspected. No scientific definition, hyperparameter
or implementation changed, and no new real experiment was launched. Motivation is
local measured evidence, not an unverified external literature claim.

## 2026-10-17 — Diagnose branching and implement a two-prior development ablation

User authorized all proposed diagnostics and the next experiment implementation.
The CPU [development diagnostic](../runs/diversity-diagnostic-v2/report.md) shows
that local branching alone puts 92–93% of mass on the first three mutation positions,
versus 2.5–3.0% for uniform candidate identities. Derived a label-free descendant-count
correction: with uniform logits its edge probabilities telescope to uniform leaves.
With actual ProGen logits, two targets broaden substantially and one remains narrow;
raw measured stability declines on one parent. This evidence supports a controlled
comparison, not adoption as the universally better protein prior.

Explicitly added the alternate arm to the spec; kept original prior as control and
held scorers, support, beta and learner settings fixed. Chose unique selection from
256 proposals without refill, retaining old repeated-slot outputs. Uniform selection
is strong and remains an explicit comparator. Chose 100 sampling repeats to reduce
noise; a descriptive engineering choice, not biological power. A proposed TV .01
check concerns approximation only. Frozen source hashes prevent accidental drift.

Reuse all three development parents; do not select a favorable subset or retune on
the completed confirmation. This GPU handoff is development-only. Fresh evaluation
and multi-mutant measurement/scorer validation remain prerequisites for later
expansion. The next run addresses whether heads can learn the broader law and whether
unique-design quality/cost justify them; it cannot show non-enumerable efficiency.
No external source is claimed for this derived correction; evidence and alternatives
are in the [protocol](experiments/diversity_plan_2026-10-17.md) and
[findings](experiments/diversity_findings_2026-10-17.md). No real/GPU run by the agent.
