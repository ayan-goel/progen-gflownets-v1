# Verification

## Success-bonus comparison — 19 November 2026

[Protocol](experiments/idr_success_bonus_2026-11-19.md),
[32 focused CPU tests](verification/idr-success-bonus-tests-2026-11-19.txt),
[real-input preflight](verification/idr-success-bonus-preflight-2026-11-19.json).

Thirteen new tests and nineteen existing objective-comparison tests pass. Tiny
fixtures run three paired four-update fits and 36 final outputs, verifying the
same bonus on bank/fresh/evaluation rows, the `log(9)/2` normalizer initialization
shift, shared examples, independent trained blocks and an immutable reference.
Saved-data checks reject corrupted reward metadata, labels, checkpoints and
evaluation records. Atomic-write failure preserves the preceding checkpoint;
source-independent recovery reproduces five scientific output files byte-for-byte.
The real model constructor and disorder scorer are blocked in these fixtures.

The CLI preflight validates protected source artifacts and declares three repeats,
1,600 updates per arm and 9,216 final outputs, without running ProGen or a scorer.
The completed objective run's 86 captured files and pinned implementation also
verify unchanged. CUDA remained uninitialized and the new real run was not
started. These are correctness checks, not evidence that the bonus improves
generation. Source/test lint and formatting checks pass.

## Target and learning mechanism analysis — 19 November 2026

The [saved-array script](verification/idr-objective-mechanism-2026-11-19.py) and
[results](verification/idr-objective-mechanism-2026-11-19.json) separate 6,400
fixed-prior training draws from adaptive/replay data, calculate target-strength
sensitivity, and decompose saved final TB residuals by prompt and repeat.
Input hashes remain unchanged; no neural model, scoring, training or GPU runs.
Finite-pool target estimates and reused final outcomes are explicitly post-hoc,
with per-stream variation and effective sample size reported. They do not verify
global target probabilities or predict retraining outcomes. The
[research assessment](../literature/IDR_GFN_IMPROVEMENT_2026-11-19.md) owns the
proposed intervention, alternatives and limitations.

## Objective-comparison result audit — 19 November 2026

[Result record](experiments/idr_objective_2026-11-18.md#completed-results--19-november-2026).
The [cached-array reconstruction](verification/idr-objective-results-2026-11-19.json)
([script](verification/idr-objective-results-2026-11-19.py)) reproduces five files
byte-for-byte: sequence metrics, aggregate metrics, decision, training summary
and report. It uses saved V2/V3 arrays, blocks model/scorer execution, and confirms
all 95 completed run files remain unchanged with CUDA uninitialized.

The [independent arithmetic audit](verification/idr-objective-independent-2026-11-19.json)
([script](verification/idr-objective-independent-2026-11-19.py)) recomputes charge,
disorder acceptance, copy definitions, distinct-success and cluster counts from
saved sequences/arrays. It also verifies all 19,200 fresh training reward rows.
The [protocol audit](verification/idr-objective-protocol-2026-11-19.json)
([script](verification/idr-objective-protocol-2026-11-19.py)) checks manifests,
source identity, matched training batches/schedules and saved final checkpoints.
No training tests were repeated for this results-only review, and no real neural
inference was executed. These checks verify recorded computational evidence,
not biological validity or independent statistical replication.

## Matched GFN versus supervised comparison — 18 November 2026

[Protocol](experiments/idr_objective_2026-11-18.md),
[19 focused CPU tests](verification/idr-objective-tests-2026-11-18.txt),
[real input-only preflight](verification/idr-objective-preflight-2026-11-18.json).

Tiny fixtures verify independent copied-block/head learning with an unchanged
reference, original-prior TB, stable per-prompt reward-weighted token likelihood,
microbatch equivalence, replay isolation, matched batches, fixed reward/evaluation
budgets and corruption rejection. The integration test runs both tiny training
repeats through fresh generation and evaluation, then removes ancestor folders
and recovers identical scientific JSON outputs. Cached-array scoring also reproduces
the report; variable timing is isolated in separate cost records. Simulated write
failure preserves the preceding atomic checkpoint. No real pretrained model or
disorder scorer executes, and CUDA stays uninitialized.

CLI preflight verifies existing source/model/scorer hashes and confirms the planned
1,600 updates per arm, 9,600 fresh training paths per repeat and 6,144 final outputs.
Ruff and `git diff --check` pass. These checks establish implementation readiness,
not a real performance result; all real execution remains user-run.

Historical verification records follow.

The preserved source snapshot in runs/biological-first/source is the reference
for this reset. All 13 top-level Python modules match that snapshot after
restoring the original command-line interface. The frozen backbone, scorer,
reward, guidance and evaluation implementations are unchanged.

The retained suite covers backbone/token alignment, finite environments,
zero-head/beta-zero identities, scorer normalization, metrics, trajectory balance,
optimization scheduling and end-to-end synthetic CPU runs with evaluation replay.

## Reset verification — 3 November 2026

- All **26 retained CPU tests pass**, including tiny synthetic end-to-end tests.
- All **13 core Python modules** and **six configurations** match the saved pilot
  source snapshot byte-for-byte.
- Saved biological evaluation reproduces all **10 JSON files exactly**, excluding
  only sampling_seconds. The report is byte-identical.
- The existing .venv has the pinned Torch/Transformers/tokenizers versions;
  original checkpoint/tokenizer files exist and CUDA remained uninitialized.
- Retained Markdown file links resolve, and the CLI exposes only the original
  six commands. No later workflow imports remain in the active core.

Replay uses saved features/heads and inspected outcomes on CPU, with one compute
thread. No ProGen inference, GPU execution or real fitting was performed.
[Machine-readable verification](verification/reset-verification-2026-11-03.json)
and [CPU test output](verification/reset-cpu-tests-2026-11-03.txt) retain evidence.

Historical machine-readable evidence is indexed [here](verification/README.md).
The [exact-test result](experiments/exact_pass_and_biological_pilot_2026-10-15.md)
and [biological result](experiments/biological_first_results_2026-10-15.md)
remain the scientific evidence.


## Predictor/reward study

All 28 CPU tests pass (including the 26 retained tests). New synthetic checks
verify beta-zero recovery, shift/scale identities, capped-reward behavior,
concentration metrics, isolated family fitting, nested subsets and unchanged
predictions after perturbing evaluation labels. Saved pilot scores reconstruct
within 2.17e-7; the [protocol](experiments/predictor_reward_study_2026-11-03.md)
records the numerical findings. No real predictor retraining or GPU run was
performed during implementation.


## Ridge-reward GFN readiness — 9 November

All 30 CPU tests pass. New synthetic checks cover reward substitution with
unchanged prior features/logits, a four-update training run, exact and sampled
evaluation, saved-evaluation replay, immutable inputs, stale-hash rejection and
rejection of accidental real CPU training. Actual saved-input preflight passes:
950 training examples, 204 candidates, beta 1, scale 1.177569, maximum ridge
prediction reconstruction difference 3.34e-6. No GPU or real training was run.


## Fixed reward-noise readiness — 9 November

All 33 CPU tests pass. New synthetic tests verify zero-noise identity, pairing
across sigma, independent noise seeds, isolation from global training RNG, a
crossed panel of four-update fits, exact decomposition of utility change, saved
replay parity, source immutability and changed-input/real-CPU-training rejection.
Clean-run input hashes pass preflight. No actual noisy-target experiment or GPU
execution was performed during implementation.


## Local split-study readiness — 9 November

All 37 CPU tests pass. New checks cover deterministic identity-only partitions,
reference placement, no overlap, sizes, source-order invariance, original-split
exclusion, changed-input rejection, row/feature alignment, synthetic full pipeline,
normalized new priors, zero-gain constant controls, report reconstruction, source
immutability and rejection of real CPU fitting. Real-input preflight reconstructs
original split membership and verifies cached features and checkpoint hashes;
new split sizes are 950/204/204. No real fit or GPU operation was run.


## Mutation-position holdout readiness — 9 November 2026

All 40 CPU tests pass, including malformed-group rejection, complete/disjoint
position partitions, input-order invariance and a synthetic three-split end-to-end
run. [Test output](verification/position-study-cpu-tests-2026-11-09.txt).
[Real-input preflight](verification/position-study-readiness-2026-11-09.json)
verifies pinned hashes, original split reconstruction and new position identities/
counts. No real training or backbone execution was performed. This establishes
implementation readiness, not scientific outcomes.


## Position-held-out GFN readiness — 9 November 2026

[CPU test output](verification/position-gfn-cpu-tests-2026-11-09.txt): all 43 tests
pass, including six tiny synthetic GFN fits over three position folds, exact replay,
label separation and input/cache guards. No real training or GPU work was performed.
[Real-input preflight](verification/position-gfn-readiness-2026-11-09.json) checks
pinned sources, group membership, priors and ridge checkpoint reconstruction.
Maximum prediction discrepancy is 2.15e-6 kcal/mol; float32 score rounding changes
the target by TV below 8.83e-9. Real prefix-feature rebuilding and GFN fitting await
the user's command; preflight does not execute ProGen.


## Multi-protein predictor readiness — 10 November 2026

[All 47 CPU tests pass](verification/multiprotein-cpu-tests-2026-11-10.txt), including
transitive grouping/exposure exclusion, outcome-independent selection, mixed-length
feature ordering, synthetic fitting/label separation/replay and primal/dual ridge
equivalence. [Real preflight](verification/multiprotein-readiness-2026-11-10.json)
checks prepared-data/model hashes and partition integrity without evaluation-label
parsing or model execution. CPU preparation completed the documented metadata/QC
and grouped panel; no real training or GPU execution occurred.


## Matched IDR TB/SubTB — 13 November 2026

All **85 CPU tests pass**, including the new exact-tree loss/gradient checks,
tiny 8-leaf optimization and random-backbone run/evaluation replay. Both methods
start from matched policies and draw the same first batch; the fixture confirms
equal ProGen forward budgets, flow/policy updates, scheduled rates and unchanged
backbone weights. Real-input preflight validates the pinned protocol without
model execution. Lint passes. No real training or GPU execution was performed.
The [protocol](experiments/idr_subtb_2026-11-13.md) describes coverage and limits.


## Completed IDR TB/SubTB result — 13 November 2026

[Saved-artifact verification](verification/idr-subtb-result-2026-11-13.json) checks
63 captured hashes, 9,216 raw sample identities/counts, all six update/rate schedules
and independent charge/threshold arithmetic for 27,648 scored rows. It retains
post-hoc token/gradient summaries and finite-pool target estimates. No metapredict
replay, pretrained inference, retraining or GPU work was performed. The
[result record](experiments/idr_subtb_2026-11-13.md#completed-results-and-next-decision--13-november-2026)
owns interpretation and limitations.

## Joint acceptance reward — 13 November 2026

All **90 CPU tests pass**. Five new tests check the reward/pass ordering, exact
acceptance boundaries, failure proximity and bounded factors, malformed inputs,
and zero TB residual at an enumerated target with the prior included once and
beta 1. These are arithmetic checks, not a real guidance-training result.

The [reproducible saved-pool audit](verification/idr-acceptance-reward-2026-11-13.json)
pins its source and code, reconstructs charge and disorder flags from 6,144 prior
proposals, and records each target's group estimates, ESS and conditional bootstrap.
The [scientific record](experiments/idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1)
owns interpretation. No pretrained predictor or ProGen ran. The separate scorer
does not alter historical run modules/configurations; the subsequent integration
and completed-run checks are recorded below.

## Joint-acceptance TB integration — 13 November 2026

All **92 CPU tests pass**. The new random-backbone integration fixture completes
training, fresh control sampling, scoring and exact evaluation replay after the
source run is hidden. It checks frozen weights/absent backbone gradients, updated
task weights, inherited rates, source integrity, scored tail lengths, saved reward
reconstruction, method-specific residuals, raw/selected counts, selection-density
labeling, efficient Hamming parity, and corrupt-snapshot rejection. A separate
importance-weight test protects against counting the prior twice and dropping
zero-pass groups. No pretrained model or real scorer inference was executed.

Real-input `idr-acceptance --check-only` passes without GPU queries or model
execution. Lint passes. [Preflight evidence](verification/idr-acceptance-preflight-2026-11-13.json)
and [CPU test output](verification/idr-acceptance-training-cpu-tests-2026-11-13.txt)
are retained. The [protocol](experiments/idr_acceptance_2026-11-13.md) owns the
GPU experiment; these smoke checks do not establish reward-learning success.

## Completed joint-acceptance result — 13 November 2026

The [CPU saved-artifact audit](verification/idr-acceptance-result-2026-11-13.json)
verifies 42 captured hashes, package/scorer identity, 9,216 raw records and 12,288
scored/derived records. Independent arithmetic reconstructs charge, disorder pass
flags and both reward definitions; it also reconstructs 19,200 training rewards,
2,400 update/rate records, exact reported counts, stable-tie selection and all
24 groups' importance estimates. Final new checkpoints are finite and agree with
logged normalizers; copied old controls match original pinned hashes. The audit
also records the post-hoc successful/failed-output residual split.

Reproduce without model/scorer execution or GPU access:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python docs/verification/idr-acceptance-result-analysis-2026-11-13.py
```

The [analysis source](verification/idr-acceptance-result-analysis-2026-11-13.py)
is preserved with its hash. Only the audit JSON is regenerated; completed run
artifacts are untouched. These checks establish internal consistency, not exact
target coverage or biological validity. The [result record](experiments/idr_acceptance_2026-11-13.md#completed-results-and-next-decision--13-november-2026)
owns interpretation. No training or pretrained inference was run during analysis.

## Replay and supervised pilot — 13 November 2026

All **96 CPU tests pass**, including four new tests for tail-grouped bank splitting,
stratified sampling, causal-cache parity with incremental rollout, recomputation of
current policy likelihoods, and complete tiny execution/recovery. The integration
fixture checks frozen weights/absent backbone gradients, updated task weights,
matched example budgets, no normalizer in supervised training, inherited schedules,
raw counts, method-specific residuals, unchanged sources and corrupted-snapshot
rejection. Recovery reproduces metrics/reports after hiding source/data directories.
A separate check excludes online-exposed holdout rows from both initial/final fit
summaries. Fixtures use random tiny ProGen weights and deterministic disorder scores.

[CPU test output](verification/idr-replay-cpu-tests-2026-11-13.txt) and
[real-input preflight](verification/idr-replay-preflight-2026-11-13.json) are retained.
Preflight verifies the completed source and bank without pretrained execution or
GPU queries: 408 training rows and 95 diagnostic rows, no split repair needed.
The [protocol](experiments/idr_replay_2026-11-13.md) owns the scientific interpretation.
No real training, frozen pretrained feature extraction or real scorer execution
was performed by the assistant. Real efficacy remains untested.

## Completed replay / supervised result — 14 November 2026

The [saved-artifact audit](verification/idr-replay-result-2026-11-14.json) verifies
44 captured hashes, package/scorer identity, 4,096 evaluation rewards, 19,200
fresh-training rewards and 2,400 updates. Independent scalar arithmetic reconstructs
the reward and pass flags; evaluation replay using saved per-residue scores exactly
reproduces metrics, decisions, fit summaries and report text. Bank partitions are
disjoint, checkpoints are finite, normalizers match logs and initial cached policy
probabilities equal the prior. No diagnostic holdout tail appeared in fresh training.

Reproduce without model/scorer inference or GPU access:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python docs/verification/idr-replay-result-analysis-2026-11-14.py
```

The [analysis source](verification/idr-replay-result-analysis-2026-11-14.py) records
its own hash and the post-hoc residual decomposition. Reconstruction writes only
under `docs/verification/`; completed run artifacts remain unchanged. These checks
establish artifact consistency, not biological validity or exact target coverage.
The [completed experiment record](experiments/idr_replay_2026-11-13.md#completed-results--14-november-2026)
owns interpretation. No real training, pretrained inference or GPU work ran.

## Supervised-start comparison — 15 November 2026

All **100 CPU tests pass**, including four new tests covering identical starting
policies, updated continuation weights, untouched frozen weights/absent backbone
gradients, reset Adam state, matched example counts, the inherited rate schedule,
training-only normalizer centering and progression against both controls. Fixed-bank
shape diagnostics are invariant to a prompt-wide offset and exclude common exposed
holdout rows. The integration fixture preserves source hashes and reproduces all
metrics/reports after hiding the source run and data. Corrupt snapshots are rejected.

[CPU test output](verification/idr-warmstart-cpu-tests-2026-11-15.txt) and
[real-input preflight](verification/idr-warmstart-preflight-2026-11-15.json) are retained.
Preflight confirms the step-800 supervised checkpoint, 503 cached rows, 400 updates
per continuation and 3,072 planned evaluation outputs without pretrained/scorer
execution or CUDA queries. Tests use a random tiny backbone and deterministic
scores. No real training or GPU work was run by the assistant. The
[protocol](experiments/idr_warmstart_2026-11-15.md) owns scientific interpretation.

### Completed result reconstruction

The [CPU analysis](verification/idr-warmstart-result-analysis-2026-11-15.py) verifies
captured artifact/source hashes and exactly reconstructs six saved evaluation JSON
files using cached V2/V3 arrays. It recomputes training rewards and summarizes
post-hoc failure-weight compression. The [audit output](verification/idr-warmstart-result-audit-2026-11-15.json)
pins analysis and generation-manifest hashes; it records the descriptive thresholds.
This verifies arithmetic, not predictor validity or global target matching.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python docs/verification/idr-warmstart-result-analysis-2026-11-15.py
```

No training, model/scorer inference or GPU work is performed. Reconstruction uses
a temporary directory and writes only the documentation audit, preserving the run.


## Continuous-reward GFlowNet — 15 November 2026

All **104 CPU tests pass**, including four new tests for unchanged acceptance,
monotonic log penalties without a floor, invalid penalty rejection, both-control
progression, cross-target diagnostics with common holdout exclusions, and full
fixture execution/recovery. The tiny CPU comparison verifies identical initial
policies and initial samples, matched replay order/rates, updated candidate weights,
unchanged backbone weights/gradients, untouched source artifacts, recovery after
source runs/data are hidden, and rejection of corrupted snapshots.

[CPU test output](verification/idr-continuous-cpu-tests-2026-11-15.txt) and
[real-input preflight](verification/idr-continuous-preflight-2026-11-15.json) are saved.
Preflight validates pinned controls, bank/cache, calibration record and frozen
checkpoint files without executing models or querying CUDA. It confirms one new
400-update TB fit, 3,200 fresh training trajectories and 3,072 evaluation outputs.
The [audit](verification/idr-continuous-reward-audit-2026-11-15.json) records source
hashes and finite-sample calibration limits. These checks establish correctness,
not training efficacy. No real training, pretrained/scorer inference or GPU job
was run by the assistant.


### Continuous-reward result reconstruction

The [saved-score analysis](verification/idr-continuous-result-analysis-2026-11-15.py)
verifies captured artifact/implementation hashes and exactly reproduces five
result JSON files using cached disorder predictions. The [audit](verification/idr-continuous-result-audit-2026-11-15.json)
pins the analysis and generation manifest. No inference, training or GPU work is
performed; outputs are reconstructed in a temporary directory without modifying
the original run.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python docs/verification/idr-continuous-result-analysis-2026-11-15.py
```


## Exact cached-IDR diagnosis — 16 November 2026

Five focused [CPU tests](verification/idr-exact-diagnostic-tests-2026-11-16.txt)
pass for the finite tree, independent oracle, conditional-prior distinction,
expected gradients, immutable initial heads and algebraic capacity construction.
The [analysis audit](verification/idr-exact-analysis-audit-2026-11-16.json) pins
results and executed source; a preserved source snapshot matches run hashes,
and formatted current scripts have identical executable function logic.
Both analyses validate source artifacts before/after and confirm CUDA remains
uninitialized. No pretrained or disorder model executes. The
[protocol/results](experiments/idr_exact_diagnosis_2026-11-16.md) own interpretation
and reproduction commands. These checks do not establish open-generation success.


## Exact-task update repair — 16 November 2026

Ten focused [CPU tests](verification/idr-update-diagnostic-tests-2026-11-16.txt)
pass for the finite task and optimizer/conditioning controls. The
[saved-weight audit](verification/idr-update-result-audit-2026-11-16.json) exactly
reproduces all four final per-prompt metrics and fixed TB losses, verifies source
snapshots and checks folding into an ordinary head. No ProGen, scorer or GPU
execution occurs. [Protocol/results](experiments/idr_update_diagnosis_2026-11-16.md)
own settings, limits and reproduction commands. A local diagnostic pass is not
verification of sampled/open-generation performance.


## Sampled conditioning — 16 November 2026

Twelve focused [CPU tests](verification/idr-sampled-conditioning-tests-2026-11-16.txt)
pass. The [saved-artifact audit](verification/idr-sampled-conditioning-audit-2026-11-16.json)
reconstructs every final metric and normalizer error, verifies folded policies
and source/input hashes, and confirms exact reproduction of historical seed 0.
Paired replay samples match; CUDA remains uninitialized. These checks verify the
restricted diagnostic, not generation performance. [Protocol/results](experiments/idr_sampled_conditioning_2026-11-16.md).


## Expected-gradient conditioning — 16 November 2026

Fourteen focused [CPU tests](verification/idr-expected-conditioning-tests-2026-11-16.txt)
pass, including exact sample-pair gradient enumeration and target-probability
independence of training. The [saved-artifact audit](verification/idr-expected-conditioning-audit-2026-11-16.json)
reproduces per-prompt metrics and normalizers, checks the saved folded head and
hashes, and verifies shared initialization/input definitions against the sampled
control. CUDA remains uninitialized. [Protocol/results](experiments/idr_expected_conditioning_2026-11-16.md).


## Averaged sampled TB — 16 November 2026

Sixteen focused [CPU tests](verification/idr-averaged-sampling-tests-2026-11-16.txt)
pass. The [saved-artifact audit](verification/idr-averaged-sampling-audit-2026-11-16.json)
reproduces all final metrics and normalizers, validates all 307,200 draw indices
and scheduled replay strata, checks folded policies and source/input hashes, and
confirms matched, unchanged control artifacts. No GPU was initialized.
[Protocol/results](experiments/idr_averaged_sampling_2026-11-16.md).


## CPU variance repair — 16 November 2026

Nineteen focused [CPU tests](verification/idr-variance-repair-tests-2026-11-16.txt)
pass, including covariance identity, estimator unbiasedness, exact snapshot
correction and oracle-independence of fitting. The [artifact audit](verification/idr-variance-repair-audit-2026-11-16.json)
reconstructs all nine fits, normalizers and gates; checks folded policies, exact
control/treatment sample pairing, source hashes and variance-audit inputs.
No GPU is initialized. [Protocol and reproduction](experiments/idr_variance_repair_2026-11-16.md#verification-and-reproduction).


## Full-support transfer — 16 November 2026

Twenty-nine focused [CPU tests](verification/idr-transfer-tests-2026-11-16.txt)
pass: six new transfer tests, four continuous-reward tests and nineteen finite
learning/variance tests. Coverage includes full-support probabilities,
rank-deficient conditioning/folding, globally weighted microbatch gradients,
raw-estimator unbiasedness and exactness at the snapshot, matched training draws,
frozen/source preservation and end-to-end synthetic generation/evaluation recovery.
Recovery reproduces metrics/decisions after hiding source runs and rejects a
corrupted generation artifact. CUDA remains uninitialized in the fixture test.

The [read-only preflight](verification/idr-transfer-preflight-2026-11-16.json)
passes against the real pinned source: 408 training rows, eight prompts, two
400-update fits and 3,072 planned evaluation outputs. It checks files, checkpoint
hashes and cached prior arithmetic without loading ProGen or running the disorder
scorer. Ruff passes for the new modules, CLI and tests. No real training or GPU
execution was performed. [Protocol](experiments/idr_transfer_2026-11-16.md).


## Transfer result audit — 16 November 2026

The [saved-artifact audit](verification/idr-transfer-results-2026-11-16.json)
verifies all 43 captured manifest files, implementation provenance, paired
400-update training draws and schedules, reward recomputation from saved scores,
aggregate metrics and the progression decision. CPU evaluation of saved heads
reconstructs recorded training log probabilities within 0.005 and evaluates the
same 95 previously available diagnostic holdout trajectories without refitting.
No pretrained model or scorer execution occurs, and CUDA stays uninitialized.
The [result record](experiments/idr_transfer_2026-11-16.md#completed-results--16-november-2026)
labels this as post-hoc diagnostic evidence, not new sealed validation.

## Last-block adaptation — 16 November 2026

Twelve focused [CPU tests](verification/idr-lastblock-tests-2026-11-16.txt) cover
initial identity, causal/incremental alignment after adaptation, gradients confined
to the copied block, exact original-model preservation, fixed reward reference,
global microbatch accumulation, paired training draws, holdout protection and the
four-arm success gate. A full tiny random-model pipeline performs four matched
updates with an injected synthetic scorer, then reproduces evaluation after hiding
source folders and rejects corrupted outputs. CUDA remains uninitialized.

The [real-input preflight](verification/idr-lastblock-preflight-2026-11-16.json)
verifies source/control manifests, checkpoint hashes and cached prior arithmetic
without model/scorer execution. It confirms eight prompts, 408 training replay
rows, 95 diagnostic rows, 400 updates per arm, 2,400 fresh training trajectories
and 4,096 planned evaluation outputs. New modules, CLI and tests pass Ruff.
These checks establish implementation correctness on fixtures, not an expected
generation gain. [Protocol](experiments/idr_lastblock_2026-11-16.md).

## Last-block result audit — 16 November 2026

The [saved-artifact audit](verification/idr-lastblock-results-2026-11-16.json)
and its [reproduction script](verification/idr-lastblock-results-2026-11-16.py)
verify 45 generation-manifest files and 71 provenance files, including the original
checkpoint file, without loading ProGen. Saved V2/V3 score arrays replace the
scorer, reconstructing sequence metrics, aggregates, decisions, bank diagnostics,
training summaries and the report byte-for-byte. Control heads match their pinned
sources; trained final checkpoints match step 400.

The audit checks all 400 paired batches, schedules and training-only replay, 2,400 fresh
training paths, 12,800 presentations per arm, frozen-reference arithmetic and no
holdout exposure. Both practical gates pass. It does not re-evaluate the neural
scorer or recompute pretrained likelihoods; original-reference consistency is
checked from saved likelihoods and protected artifacts. CUDA stays uninitialized
and run outputs remain unchanged. The [result record](experiments/idr_lastblock_2026-11-16.md#completed-results--16-november-2026)
owns interpretation and post-hoc prompt sensitivity.


## Last-block confirmation — 17 November 2026

All 26 focused [CPU tests](verification/idr-lastblock-confirmation-tests-2026-11-17.txt)
pass. Tiny random-model fixtures with injected synthetic scorers verify fresh
initialization, unchanged training settings, independent RNG streams, head-free
plain-ProGen probabilities, shared copy denominators and preservation of every
repeat/prompt. One failing repeat cannot be hidden by averaging. The integration
checks also recover after an injected final-scoring failure, with source folders
removed and no ProGen loading or retraining. Retained last-block checks cover
original-reference preservation and paired training.

The [read-only real-input preflight](verification/idr-lastblock-confirmation-preflight-2026-11-17.json)
passes: two paired fits, 400 updates per trained arm and 10,240 planned final
outputs across five arms. It checks pinned inputs and budgets without executing
real models or scorers; CUDA remains uninitialized. Ruff passes on the affected
code/tests. These checks validate implementation behavior, not a generation gain.
Real confirmation remains user-run. [Protocol](experiments/idr_lastblock_confirmation_2026-11-17.md).

## Confirmation result audit — 17 November 2026

The completed user-run confirmation passes its [saved-artifact audit](verification/idr-lastblock-confirmation-results-2026-11-17.json).
The [reproduction script](verification/idr-lastblock-confirmation-results-2026-11-17.py)
checks all 127 enclosing-manifest entries, four child manifests and 74 unique
provenance files, including the original ProGen checkpoint hash. Package versions
match. Replacing predictor calls with the run's saved V2/V3 arrays reproduces
23 result files byte-for-byte: sequence metrics, arm/prompt summaries, copy
definitions, bank diagnostics, training summaries, decisions and reports.

Both fits preserve every non-seed setting and the original supervised starts.
The controls' heads match their sources; trained checkpoints match update 400.
All 400 paired batches per repeat, learning-rate schedules, 2,400 fresh training
paths and 12,800 presentations per trained arm check out. Plain-ProGen saved
policy and reference log probabilities agree exactly; continuous reward arithmetic
reconstructs and neither repeat encountered a held-out bank tail during training.
Both predefined confirmation screens reproduce as passing.

All 145 completed-run files remain unchanged. No ProGen likelihood or neural
disorder score was recomputed, and CUDA remains uninitialized. Reported block
changes and reference-gradient flags are saved diagnostics; the audit does not
independently reload the full original model to recompute those changes. This
verifies the recorded computation, not biological validity or global target
matching. The [result record](experiments/idr_lastblock_confirmation_2026-11-17.md#completed-results--17-november-2026)
owns the scientific interpretation and next decision.

## Last-block training-budget comparison — 17 November 2026

The [focused test log](verification/idr-lastblock-budget-tests-2026-11-17.txt)
records **14 passing CPU tests** using tiny injected backbones and deterministic
fixture disorder scores. No pretrained model or real disorder scorer executes.
Coverage includes source/provenance rejection, checkpoint dimensions and finite
values, doubled budgets with only fresh evaluation seeds changed, all six arms,
both-repeat decision thresholds and symmetric training-copy controls.

The end-to-end fixture reproduces the old training prefix until its earlier
decay, then verifies that the longer schedule diverges as declared. It checks
original initialization, unchanged saved-control weights and historical files,
an untouched frozen reference, and final checkpoint steps. Both normal completion
and an injected final-scoring failure recover from saved sequences after source
folders are removed. Corrupt snapshots are rejected. CUDA remains uninitialized.

The [real-input preflight](verification/idr-lastblock-budget-preflight-2026-11-17.json)
passes through the actual CLI with `CUDA_VISIBLE_DEVICES=''`. It checks saved
lineage, original model-file hashes and checkpoint tensors on CPU; it does not
load ProGen or execute a scorer. It confirms two paired 800-update fits, decay
after 600, 4,800 fresh training paths per repeat, 25,600 presentations per trained
arm and 12,288 final outputs. This establishes readiness of the implementation,
not improved scientific performance. All real experiments remain user-run.

## Training-budget result audit — 18 November 2026

The completed user-run experiment passes its
[saved-score audit](verification/idr-lastblock-budget-results-2026-11-18.json).
The [reproduction script](verification/idr-lastblock-budget-results-2026-11-18.py)
uses cached V2/V3 arrays, copied child score records and temporary output to
reproduce the five parent result files byte-for-byte: sequence metrics, summaries,
copy definitions, decision and report. All 182 completed-run files remain unchanged;
CUDA is uninitialized. ProGen loading and real scorer calls are blocked.

The evaluator rechecks protected generation/provenance inputs, preserved 400-step
controls, support, raw/scored row alignment, continuous-reward arithmetic and the
common training-tail union. Separate independent saved-data arithmetic agrees on
all 12,288 rows, charge/disorder/joint flags, copy membership and prompt/arm rates.
The two primary gains are +9.67 and +6.05 points; both gates pass.

Additional read-only compliance checks verify 160 outer manifest entries, 53 per
fit, four per prior and 77 provenance files without drift. Both starts match the
original initialization; the first 300 adapted log entries match the old fits.
All 800 steps preserve paired example IDs and the declared rates, decaying at
601. Final checkpoint tensors match step 800, and copied controls match protected
step 400. No neural disorder score or pretrained likelihood is recomputed, so the
audit verifies recorded computation, not biological validity or exact target
sampling. The [result record](experiments/idr_lastblock_budget_2026-11-17.md#completed-results--18-november-2026)
owns scientific interpretation.

## 1,600-versus-800 comparison — 18 November 2026

All **28 focused CPU tests pass**: 14 for the new comparison and 14 regressions
for the preserved 800-versus-400 wrapper. The [test output](verification/idr-lastblock-1600-tests-2026-11-18.txt)
records a 25.710-second run with CUDA hidden and one compute thread. All model
execution uses the tiny synthetic backbone and injected fixture disorder scores;
real ProGen loading is blocked in integration tests.

Checks cover unchanged initialization/reference/source artifacts; the 8→16 fixture
budget with decay after 6→12; paired streams and fresh evaluation seeds; all six
arms and raw output counts; the common union retaining historical-only training
copies; both-repeat progression/quality thresholds; and byte-identical scoring
recovery after ancestor folders become unavailable. An injected final-scoring
failure retains sealed generation and recovers without training. Corrupt histories,
checkpoints and settings are rejected. Ruff lint and format checks pass for the
new modules/tests and changed CLI.

The [real-input CLI preflight](verification/idr-lastblock-1600-preflight-2026-11-18.json)
passes with 1,600 updates per arm, decay after 1,200, 9,600 fresh training paths
per repeat, 51,200 trajectory presentations per arm and 12,288 final outputs.
It validates saved manifests, source identities and checkpoint tensors on CPU;
`cuda_initialized` and `model_or_scorer_executed` are both false. Completed run
artifacts and their computational source remain unchanged. These checks establish
implementation readiness, not a gain from longer training. The
[protocol](experiments/idr_lastblock_1600_2026-11-18.md) governs the pending user run.

## 1,600-update result audit — 18 November 2026

The user completed `runs/idr-lastblock-1600-v1`: two real 1,600-update paired fits
and 12,288 final outputs. The saved-array [audit](verification/idr-lastblock-1600-results-2026-11-18.json)
([script](verification/idr-lastblock-1600-results-2026-11-18.py)) reconstructs the
five parent report, metric, decision and copy-definition artifacts byte-for-byte.
It uses saved child score records and cached V2/V3 arrays, blocks ProGen loading
and neural scorer calls, and confirms all **216 run files** are unchanged.
CUDA remains uninitialized.

An independent [stdlib-only audit](verification/idr-lastblock-1600-independent-2026-11-18.json)
([script](verification/idr-lastblock-1600-independent-2026-11-18.py)) recomputes
charge/disorder/joint acceptance, copy unions, uniqueness, per-prompt counts and
both-repeat thresholds directly from sequences and saved scores. It confirms
350/318 versus 301/301 successes, +4.79/+1.66 percentage points, and both budget
screens failing while quality checks pass. It also records the explicitly post-hoc
late-training windows without treating them as fixed-checkpoint evaluation.

The [protocol audit](verification/idr-lastblock-1600-protocol-2026-11-18.json)
([script](verification/idr-lastblock-1600-protocol-2026-11-18.py)) verifies all
194 root-manifest entries, child manifests, live source/package identities and
original ProGen file hashes. Saved tensors are loaded on CPU only. Final weights
match step 1,600; every rate record follows decay after 1,200. The original
initialization, unchanged head, preserved 800-update controls and historical
400-update training rows match their pinned sources. The first 600 updates and
3,600 fresh trajectories per repeat reproduce the corresponding 800-update fit
exactly. No diagnostic-tail overlap is observed; saved reference-gradient checks
remain false. This validates the completed run's internal provenance, not the
unobserved filesystem handling between the failed and completed `v1` attempts.

These checks execute no real model, scorer, training or GPU work. They cannot
independently validate predicted disorder or prove global target-distribution
accuracy. The [result record](experiments/idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026)
owns interpretation and the decision to retain 800 as the default.
