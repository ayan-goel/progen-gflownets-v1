# Implementation log

This is a chronological record; newer entries are appended below. For the active
stage use [current status](status.md); use the [experiment index](experiments/README.md)
to find each protocol, result and supporting evidence.

## Slice 0 — Workspace and runtime (passed)

Inspected the current spec, v1 instructions, old ProGen adapter and checkpoint
configuration. No existing v1 implementation or running implementation job was
found. Created v1/.venv using the existing Python 3.10 base environment and installed
the two pinned compatibility packages into v1. Verified all four runtime versions
and `torch.cuda.is_initialized() == False` with CUDA hidden. No GPUs queried or used.

The implementation sequence is runtime → data/environment → frozen adapter →
guidance and exact experiment → stability scorer/biological pipeline → evaluation
and end-to-end CLI verification. Each slice records tests before the next begins.

## Slice 1 — Sequence environment and run protection (passed)

Implemented substitution constraints, exact small-space enumeration, finite-support
prefix trees, context features, streamed checksums and no-overwrite run creation.
CPU unittest: **4 tests passed** (2026-10-14). Exhaustively traversed all 256 toy
sequences; verified unique paths, forced residues, invalid candidates, duplicate
rejection, finite-support dead-end removal and refusal to reuse an output directory.
The four-position toy has 85 internal prefixes and 256 leaves.

## Slice 2 — Frozen ProGen adapter (passed)

Vendored unmodified upstream model files, tokenizer and license. Implemented
local checkpoint loading, residue-token validation, literal `1` direction marker,
teacher-forced features/logits, KV-cache decoding, pooled complete-sequence features
and a prefix-feature cache. CPU unittest: **6 tests passed**. The two adapter tests
compare cached/full logits at every position and shared-prefix outputs under different
future suffixes using the actual upstream architecture with tiny random weights.

The first tiny fixture used four heads and failed because upstream attention uses
eight-way packing. Corrected the fixture to 64 hidden dimensions/eight heads; no
upstream source or pretrained checkpoint changed. CPU float32 comparisons now pass.
Real checkpoint and GPU precision behavior remain user-run checks.

## Slice 3 — Guidance, TB and supervised comparison (passed)

Implemented a zero-initialized residual head, locally normalized constrained
probabilities, on-policy sampling, log-space trajectory balance with learned log Z,
Monte Carlo future-reward supervision and exact-distribution diagnostics.
CPU unittest: **10 tests passed**, total test runtime about one second. Added
zero-guidance and beta-zero identities, forced-action probability one, exact-target
TB residual, differentiable/full likelihood agreement, a tiny four-leaf/80-update
optimization smoke test and frozen-backbone tensor equality after training.
This does **not** satisfy the real 256-sequence ProGen2 three-seed convergence gate.

## Slice 4 — Data preparation mechanics (fixture checks passed; release audit pending)

Added schema validation, canonical/length/label filtering, unambiguous reference
mapping, sequence-level replicate averaging and seeded 70/15/15 local splits.
Test identities and test labels are separate artifacts. CPU unittest: **11 tests
passed**, including deterministic partitions, non-finite/missing labels and no
sequence overlap. The real release is still downloading; fixture checks alone do
not establish its coverage or the initial biological reference.

## Slice 5 — Stability scorer (passed)

Implemented train-only feature/label scaling, dual ridge regression, a small MLP,
validation-only selection and a fixed predicted-improvement reward. A required
validation gate compares against the training-mean predictor and checks ranking.
CPU unittest: **13 tests passed**. Synthetic linear data verify the gate, train-only
moments, frozen parameters, reference subtraction and tied-rank correlation.
No real scorer has been fit, and no biological performance is claimed.

## Slice 6 — Exact pipeline and reload evaluation (initial checks passed)

Added JSON configurations, CLI, checkpoint/provenance saving, all four comparison
methods, raw/selected output records and CPU-only saved-run evaluation. **15 CPU
tests passed**, including a 20-update tiny-ProGen exact pipeline and identical
samples/metrics after reload. A smoke run cannot satisfy the real convergence gate.
Further biological integration, diagnostics, cost auditing and documentation checks
remain before the first-GPU handoff is ready.

## Documentation alignment — 2026-10-14 (passed)

Paused implementation at the user's request. Rewrote README and the experiment plan
around the spec's Mega-scale task and A–E milestones. Updated the review's biological
scope, nonlinear-head choice, target notation and baseline priorities. Added the
Mega-scale paper as R29, preserving original search dates and source-reading limits.
Moved standalone results to a historical record with corrected links. Updated the
spec's stale implementation status and execution instructions without changing the
scientific objective. Checked local links and fenced blocks across active docs.

No active plan now proposes GB1, a required linear-only head, or a synthetic scale-up
stage before Mega-scale. GB1 remains in descriptions of external benchmark papers.
Resuming implementation after these documentation checks.

## Slice 7 — Real-data audit, biological integration and diagnostics (passed)

Verified the downloaded CSV against pinned ZIP metadata, CRC and decompressed
size, then recorded SHA-256. Audited all 776,298 rows without fitting a model.
The coverage rule selected 1O6X.pdb, with 950/204/204 train/validation/test sequences;
details are in data.md and the immutable prepared manifest.

Extended CPU checks to the full biological branch using **synthetic labels and
tiny random ProGen**, including scorer-gate failure preventing guidance/test-label
evaluation. Added empirical sampling, beta-zero training, hand-calculated partial
coverage/duplicate metrics and the privileged exact-capacity diagnostic. **18 tests
passed** after formatting. Added progress persistence, backbone token/call counts,
separate score-lookups/new-model-call costs, and predicted/reference improvements.
Real data were prepared only; no real scorer fitting or GPU execution occurred.

## Slice 8 — Documentation and final handoff verification (passed)

Added organized architecture, data, hyperparameter, runbook and verification documents.
Final suite: **19 CPU tests passed**, including the real upstream local-checkpoint
loader with tiny saved weights. Ruff lint/format checks pass. The persistent final
CLI smoke and independent evaluation agree on samples, selection indices, metrics
and distribution diagnostics. Its source snapshot matches the current code.
Checked links/fenced blocks across 17 active documentation files. Saved test, doctor
and audit evidence under docs/verification/. CUDA stayed uninitialized; neither
real experiment directory exists. The user can now launch the first GPU command
from the runbook. Real convergence and stability outcomes remain pending.

After verifying the CSV's CRC/SHA identity, removed two redundant partial transfer
files created during retrieval. Retained the complete compressed member, CSV and
download manifest. Historical test-project files were not changed.

## Slice 9 — Diagnose the user-run exact-task failures (14 October 2026)

The user completed the real exact experiment (three-seed TV 0.185/0.161/0.171)
and exact-conditional diagnostic (TV 0.169). Both miss the criterion. Read-only
CPU analysis independently verified the target and localized about 73% of sequence
KL to length-two prefixes. Distinct but nearby feature rows motivate a conditioning
hypothesis, without proving a capacity limit. See the
[full experiment record](experiments/exact_diagnostics_2026-10-14.md).

Added `inspect-fit`, richer capacity-fit progress/provenance/failure reporting,
and a user-run two-arm CPU diagnostic script. Preserved all old artifacts. No
TB/biological configuration or architecture changed. Updated current README,
runbook and plan status; left the scientific contract intact. **21 synthetic CPU
tests passed**, including original-update parity and forced-edge oracle/KL checks.
The real two-arm fits remain user-run; no agent GPU work or real fitting occurred.

## 2026-10-14 — User-run optimization comparison complete

Inspected both saved curves and verified their source-input hashes. The original
rate reached final TV 0.042336; the lower rate reached 0.055953. No fitting was run
by the agent. Prepared `configs/exact-long.json`, changing only the two update
counts (TB/supervised) to 10,000, and aligned README/runbook/experiment-plan status.
Validated the configuration and its exact diff on CPU without backbone execution.
Training code is unchanged. Next action belongs to the user: run the fresh
three-seed longer TB experiment. Biological work remains gated.

## 2026-10-14 — Longer TB result: one seed passes

Read `exact-long` status/gate/evaluation/costs and all three saved training logs.
CPU evaluation of saved GFN and supervised heads reproduced metrics and decomposed
remaining error by prefix depth. Saved analysis and input hashes under
`docs/verification/exact-long-analysis.json`. The all-seed gate still fails.
Updated current documentation to reflect the completed experiment and proposed
next diagnostic. No training code, configuration, threshold or raw run output
changed; no agent fitting, backbone execution or GPU operation occurred.

## 2026-10-14 — Late-rate schedule and exact monitoring implemented

Implemented optional deterministic two-phase rates for TB and supervised learning,
plus exact no-grad distribution monitoring, applied-rate logging and separate
monitoring costs. New `configs/exact-decay.json` contains the declared schedule
and monitoring options; older configs remain unchanged. Updated progress printing
to handle the new step-0 null loss and show TV/rate. The integrated fixture caught
that formatting issue before the real-run handoff.

All 26 tiny CPU tests pass, including schedule boundaries, on/off monitor parity,
configuration rejection, cost accounting and a persisted end-to-end smoke.
Lint/format checks pass. Separate comparison to the original saved training source
confirms constant-rate CPU weight/loss parity. Documented every new parameter and
aligned active status/run instructions. No pretrained backbone, GPU job or real
fitting experiment was executed by the agent. The user-run scheduled experiment
remains pending; biological work remains gated.

## 2026-10-14 — Scheduled run: two seeds pass

Inspected scheduled status/gate, per-method exact-TV histories and costs. Verified
103 logged points and the applied schedule, with final TB history matching the
gate exactly. Saved read-only summaries and input hashes; no model execution was
needed. Result is two final passes out of three, so biology remains gated.
Prepared/validated `exact-decay-long.json`: only total steps and the fractional
boundary change, preserving the absolute drop at update 8,000. Updated active
documentation; old outputs and configurations are untouched. No real training or
GPU operations were run by the agent.

## 2026-10-15 — Exact gate passes; prepare biological pilot

Verified the real exact gate, final/history agreement, 203 observations per learner,
applied rates and late-window behavior. Stored saved-result analysis with hashes;
no model execution was needed. Prepared `biological-pilot.json` pointing at the
passing run and using the tested guidance schedule; scorer/data/evaluation remain
unchanged. Validated the configuration and passing-run requirements without GPU
work or real fitting. Updated current documentation and the spec's foundation
status. The biological experiment is prepared for the user; no result is claimed.

## 2026-10-15 — Biological pilot completes without measured gain

Read final status/gate, scorer report, raw/selected evaluations and costs.
Evaluated saved heads on CPU to compute exact probability-weighted predicted and
measured stability across all held-out candidates after run completion. Identified
that the target itself does not improve measured stability, and illustrated ranking
errors among dominant candidates. Saved analysis/input hashes and a concise result
record; updated active documentation and spec status. No training code, run output,
model, hyperparameter or data split changed. No GPU or training was run by the agent.

## 2026-10-15 — Target diagnostic implemented and prepared

Added panel preparation, exact development-target analysis, user-run orchestration
and CLI commands `prepare-panel`, `diagnose-target`, `evaluate-target`. Reused the
frozen adapter, scorer fitter, scientific filters and immutable artifact helpers.
Prepared the real panel on CPU without fitting any models. Added synthetic tests
for causal likelihood parity, exact expectations, outcome-independent selection,
reserved-file isolation, immutable outputs and byte-identical saved evaluation.
All 31 CPU tests pass; lint and formatting pass. The CPU preflight verifies the
existing exact gate, checkpoint presence, .venv, configuration and permitted split
checksums without GPU initialization. Updated spec status/diagnostic scope and
aligned docs. GPU execution and real scorer fits remain for the user.
See [handoff](experiments/target_diagnostic_2026-10-15.md),
[test log](verification/target-diagnostic-tests.txt), and
[readiness](verification/target-diagnostic-readiness.json).

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

## 2026-10-16 — Confirmation implementation and GPU handoff

Added identity-only preparation, fixed five-parent orchestration, all-parent fit
sealing and CPU confirmation evaluation/replay. Added `prepare-confirmation`,
`confirm` and `evaluate-confirmation` commands; the existing GPU-selection wrapper
is unchanged. Prepared candidate identities without opening test-label files.
Reused unchanged backbone, scorer, TB and supervised learner implementations.
Added six synthetic tests; fixed a test import caught by lint, then verified all
37 CPU tests and lint/format checks. Checked checkpoint/prerequisite presence and
metadata integrity without GPU initialization. Updated spec, current runbook,
research plan and parameter/verification records. Real experiment not started.
See [protocol](experiments/confirmation_plan_2026-10-16.md),
[readiness](verification/confirmation-readiness.json) and
[test log](verification/confirmation-tests.txt).

## 2026-10-17 — Documentation rules and organization during confirmation

Agent work: documentation-only edits and read-only inspection. Inspected AGENTS,
README/spec metadata, docs/literature indexes, and the saved confirmation status.
The user reported launch and the saved status says running; no independent process
liveness or completion check was performed. No reserved outcomes were read.

Added required documentation rules, a current-status page, grouped experiment and
verification indexes, a record template and a dated launch/warning observation.
Kept all pre-existing record paths stable and updated stale root/runbook/literature
entry points. No new research was conducted; retained existing source-reading dates.
The locally inspected PyTorch source motivates the limited warning interpretation.

Verification procedure: record SHA-256 of 81 source/test/script/config/dependency
files and immutable active-run snapshots/config metadata before editing; compare
afterward. Check relative Markdown file links across docs/literature and root entry
points; skip external URLs and in-page anchors. Save paths/hashes and check results
in [the cleanup audit](verification/documentation_cleanup_2026-10-17.json).
No GPU query, model fit, training restart, code/config change or raw-run modification
was performed. Model tests were not rerun because only documentation changed. The earlier 37-test
pass remains historical implementation evidence, not a new check in this cleanup.

## 2026-10-17 17:37 EDT — Read-only confirmation progress check

At the user's request, inspected `runs/confirmation-v1/status.json`, `config.json`,
per-parent `fit_status.json`/`costs.json`, and the last records and modification times
of `seed-*/*_progress.jsonl`. Used `ps -u agoel320 -o pid,etime,time,stat,args`
to confirm the confirmation process was present (PID 2660559, about 2 h 49 min
elapsed). Read runner/progress-recorder source to interpret phase order. No GPU
query, model execution, reserved outcomes or active artifacts were modified/read.

Parents 03–05 have completed all three GFN/supervised seed pairs. Parent 06 has
completed seed 0 and started seed 1 (GFN step 200 at 17:37:03 EDT); parent 07 has
cached setup but no guidance fits. Thus 10/15 seed pairs are complete. Recent
progress writes and the live process indicate continued execution, superseding the
earlier saved-status-only observation. Completed GFN fits took 918–949 seconds
and supervised fits 41–44 seconds each. Five remaining pairs imply roughly
80–82 minutes of fitting at that rate; allow about **80–95 minutes remaining**
including evaluation/variation, roughly 18:57–19:12 EDT. This is an extrapolation,
not a completion guarantee or evidence of biological success.

System `python3` lacked `zoneinfo`; repeated the read-only summary successfully
with `.venv/bin/python`, using only standard-library modules. No model tests needed.
Decision: leave the progressing run unchanged and await final evaluation. No new
literature or scientific/hyperparameter choice was involved.

## 2026-10-17 — Completed confirmation reviewed

Verified complete non-smoke execution, five passing scorers, every final seed,
57 sealed fit hashes and 27 saved source hashes. All 26 current source modules
match the run snapshot. CPU-only `evaluate-confirmation` replay into
`runs/confirmation-v1-review-replay` reproduced 67 scientific files byte-for-byte,
excluding timing. No real fit or GPU operation was launched; original run unchanged.
Inspected protocol, runner and saved per-parent results/costs/samples; aggregated
seeds within parents, then five parents equally. A post-hoc diagnostic deduplicates
existing selected batches for descriptive unique-quality reporting, without
resampling or new selection. Audit retains calculations and exact replay command.

GFN mean measured gain +0.14461 kcal/mol on all five supports; mean TV 0.000368.
Supervised gain +0.13471 costs 21.82 times less fitting time. Selected performance
is mixed and duplicate counts are high. Preserve the confirmation and design a
fresh development benchmark focused on diversity, constrained-prior position bias,
scorer validity and total cost before any further GPU run. This is a proposed next
step, not changed settings or a new experiment. No new literature or spec changes.
See [results](experiments/confirmation_results_2026-10-17.md) and
[audit](verification/confirmation-result-analysis.json). Updated current status,
root/docs indexes and runbook; checked local links. Existing historical protocols
and raw artifacts remain unchanged. System model tests were not rerun for this
analysis/documentation work; saved CPU replay is the relevant verification.

## 2026-10-17 — Plain-language project summary

At the user's request, expanded current status with accomplishments, limitations
and ordered next steps. Based on the existing confirmed results/protocol, clarified
finite-support generation, within-parent predictor training, no new wet-lab tests,
duplicate concentration and the unresolved comparative advantage. Proposed diagnosis
and protocol design precede further GPU work; no scientific or numerical choices
changed. Read AGENTS/current status/docs index and checked local file links in the
updated pages using standard-library Python. No experiment, model test or new
literature search was run; original results remain the supporting evidence.

## 2026-10-17 — Diversity diagnostics, prior ablation and user GPU handoff

Read AGENTS, original learners/tree/cache/evaluation code, spec and saved development
artifacts. Initial direct import lacked PYTHONPATH; repeated on CPU with source path
set. A guessed validation filename did not exist; used the actual scorer_fits.json.
Computed label-free branching controls and one descendant-count correction using
saved causal logits and frozen MLP-0 predictions. No new model was fitted.

Implemented diversity diagnostics, fixed-budget unique selection, historical
confirmation re-selection, and a two-prior development runner. Reused TB/supervised
learners unchanged; frozen source/checkpoint/tokenizer hashes, cached features,
failed-screen reporting, complete fit seals, costs and CPU replay are explicit.
Added config and five CLI commands; launcher and user GPU selection remain unchanged.
Recorded the explicit scientific-prior extension and every numerical setting in
spec/protocol/hyperparameters; updated current entry points and retained old results.

Executed CPU diagnostics in runs/diversity-diagnostic-v1 (preliminary) and v2
(final CLI/source snapshot); inspect-unique wrote runs/confirmation-unique-review-v1.
Exact commands/procedures and limitations are in the findings record. A smoke test
caught general-validator compatibility requiring filter_pool_count; corrected only
the adapter. Lint caught a lambda and it was replaced with statistics.mean.
After focused tests, all **41 synthetic CPU tests passed** in the full suite; lint
and format checks passed. Smoke fits use four updates on artificial four-residue
proteins and never real data. Readiness verified real files without model load or
CUDA initialization. No new real fit, GPU operation or external literature search.

Documentation links and preserved source/fit artifacts are checked in
[handoff audit](verification/diversity-implementation-audit.json).
[Findings and exact GPU command](experiments/diversity_findings_2026-10-17.md).

Final handoff audit verifies 15 frozen development-source files, 57 sealed
confirmation fit files, 27 original source snapshot files and 25 unchanged original
scientific modules (CLI intentionally extended). Preliminary/final diagnostic
metrics match; all 360 local Markdown file links resolve. Corrected the audit's
displayed fit-file count to count the nested hash entries rather than its two
metadata keys; the underlying hash checks passed. Real `runs/diversity-v1` has
not been created by the agent.

## 2026-10-17 — Initialize local Git repository

At the user's request, initialized Git on branch main in the v1 project root.
Expanded .gitignore to exclude runs, all data, literature, virtual environments,
Python/tool caches, build/coverage outputs, model/tensor artifacts, training logs,
local environment secrets and temporary files. Generated docs/verification evidence
is ignored while its handwritten README stays eligible for version control.
Source, tests, configs, scripts, the spec and handwritten documentation remain
eligible, including vendored tokenizer JSON and license/provenance files.

Verified with git check-ignore and git ls-files --others --exclude-standard;
checked representative retained files and the largest eligible files. No artifacts
were deleted or moved, no model checks or GPU work ran, and no remote or initial
commit was created. Literature and generated-evidence links remain useful locally;
those targets will not be present in a clone unless supplied separately.

## 2026-10-17 — Simplify the README

At the user's request, rewrote the README around purpose, approach, current
findings, the next run command and source/documentation structure. Removed the
version-control/ignored-artifact discussion, duplicated mathematical/reference
detail and obsolete claims that confirmation was pending. Detailed formulation,
results and protocol remain in their authoritative documents. No ignore rules,
code, configuration or experiment artifacts changed. Checked README local links
and the documented command against the CLI/runbook; no model tests were needed.
