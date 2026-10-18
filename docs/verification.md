# Verification records

**14 October 2026.** Verification covers the implementation up to the first user-run
GPU experiment. It does not claim that real ProGen guidance converges, that the
Mega-scale scorer passes validation, or that generated variants improve stability.

## Latest verification — confirmation handoff (16 October 2026)

**37 CPU tests pass**; [saved log](verification/confirmation-tests.txt).
Six new synthetic tests cover candidate reconstruction without reserved-file access,
all-parent outcome isolation, interruption before seal, missing/changed fit
artifacts, matched proposals, failed-screen reporting, duplicate-aware metrics,
CPU-only execution checks and deterministic replay (excluding wall-clock costs).
Tiny smoke fits use artificial four-residue proteins and random ProGen weights;
passing screens are forced only in a marked smoke fixture to exercise training.

The [CPU readiness record](verification/confirmation-readiness.json) checks real
metadata/checksums, fixed configuration, `.venv`, checkpoint presence and completed
prerequisites without parsing reserved outcomes or initializing CUDA. Scientific
core hashes match the development snapshot. Real confirmation fitting/evaluation
remains user-run; these checks establish mechanics, not biological results.

## Earlier verification — development target diagnostic (15 October 2026)

**31 CPU tests pass**; [saved log](verification/target-diagnostic-tests.txt).
The five new tests check teacher-forced/prefix-tree prior parity, exact target
expectations and beta-zero recovery, outcome-independent panel selection,
split/checksum isolation, reserved-file nonconsumption, matched sample counts,
immutable outputs and byte-identical CPU reevaluation. Smoke fits use tiny random
ProGen and artificial four-residue proteins only. CUDA remains uninitialized.

[Readiness](verification/target-diagnostic-readiness.json) records the real prepared
panel, permitted-file hashes, inherited outer split, checkpoint files and passing
exact gate. Real diagnostic numerical behavior and scientific conclusions remain
untested until the user runs it. See the [handoff](experiments/target_diagnostic_2026-10-15.md).

## Earlier verification — late-rate schedule and monitoring

**26 CPU tests pass**; [saved log](verification/schedule-tests.txt). Tests cover
both learners' boundary, actual updates, monitor on/off weight/loss/RNG parity,
metric equality, extra cost counters and persisted progress under a 20-update
random-ProGen smoke. Initialization printing was corrected after an integrated
test exposed the null-loss formatting issue. Lint and formatting pass.
[Compatibility checks](verification/schedule-compatibility.json) verify unchanged
constant-rate behavior against the original saved source and unchanged raw run
hashes. All checks used tiny synthetic CPU fixtures; the scheduled real experiment
is prepared, not executed. See the
[experiment record](experiments/exact_decay_2026-10-14.md).

## Follow-up verification — exact-task diagnostics

The user-run exact and capacity fits failed their distribution-accuracy target;
see [the diagnostic record](experiments/exact_diagnostics_2026-10-14.md).
The updated **21 CPU tests pass**, adding independent marginal/oracle and KL-chain
checks on a forced-edge fixture, original-update parity, saved-fit auditing and
progress persistence. [Saved follow-up test log](verification/diagnostic-tests.txt).
Real-cache analysis loaded saved tensors/heads only; no real fitting or GPU job
was performed by the agent. The user subsequently completed the two-arm optimization experiment: original-rate
TV 0.042336 and lower-rate TV 0.055953. The fresh longer TB configuration passes
validation with exactly two update-count changes and no CUDA initialization;
[configuration check](verification/exact-long-config.json). The longer TB run
has not been launched by the agent.

## Original handoff evidence

- **19 CPU tests pass**, using tiny random ProGen and synthetic labels only.
  [Saved test log](verification/cpu-tests.txt).
- Pinned runtime imports and local checkpoint/tokenizer files pass the CPU doctor;
  CUDA remains uninitialized during development checks.
  [Doctor output](verification/doctor.json).
- Ruff lint and formatting checks pass for our source, tests and scripts.
  Vendored upstream source is excluded and matches the original file hashes.
- The real Mega-scale CSV passed its member CRC/size checks and SHA-256 verification.
  Prepared split hashes match their immutable manifest. This is data setup, not
  model training or a real experiment.
- A persistent [CPU CLI smoke run](../runs/cpu-smoke-20261014-final/status.json)
  exercises training, saving and reporting. Its real-experiment gate is deliberately
  false. [Its report](../runs/cpu-smoke-20261014-final/evaluation/report.md) concerns
  synthetic scores, not folding stability.
- Saved-run evaluation reproduces raw samples, selection indices, distribution
  diagnostics and metrics. Timing is excluded from equality comparisons.
  [Final audit record](verification/audit.json).

## Requirement-by-requirement checks

| Requirement | Authoritative evidence | Scope/limit |
|---|---|---|
| All implementation in v1 | src/, scripts/, configs/, tests/, docs/ | Old project is read-only reference; weights reused by path |
| No agent GPU work or real training | CUDA-hidden test launcher; smoke provenance and doctor | No real checkpoint inference or real scorer fitting executed |
| Atomic implementation slices | implementation_log.md, successive test results | Checks cover stated mechanics, not universal correctness |
| Organized code and comments | Module contracts, docstrings, Ruff checks | Minimal upstream code remains separately attributed |
| Current documentation agrees | README, spec, review, plan, data/architecture/hyperparameter docs | Historical sources/results remain labeled |
| Data release and filtering | Download manifest, prepared manifest, data.md | Assay missingness and replicate noise remain limitations |
| Sequence constraints | Exhaustive small-space traversal and finite-support tests | Initial biological task is enumerable and local |
| Frozen causal backbone | KV/full agreement, prefix causality, unchanged weights after head training | Tiny architecture fixture; real numerical check is in user run |
| Checkpoint loading | Tiny local save/load round trip using actual upstream class | Full pretrained GPU load intentionally untested |
| Correct prior and TB | Normalization, zero head/beta-zero, exact-target residual, likelihood checks | Real three-seed TV threshold still pending |
| Sampler | Fixed-seed empirical frequencies versus exact probabilities | Small four-leaf mechanical fixture |
| Stability scorer | Synthetic fit, train-only moments, frozen parameters, reward transformation | No real validation outcome yet |
| Biological orchestration | Synthetic end-to-end pipeline and forced scorer-gate failure | No test-label evaluation after a failed scorer gate |
| Baselines | Shared-support ProGen, selection, supervised potential and TB | Larger selection pool is explicit; this is not equal compute |
| Evaluation metrics | Hand-calculated duplicates, partial coverage, distances and improvement | Finite-support novelty/coverage are partly built in |
| Checkpoints and repeatability | Saved-head replay matches original outputs | Same CPU software environment; timings differ |
| Capacity diagnostic | Tiny exact-conditional fit smoke and privileged output flag | Not a fair competitor or gate bypass |
| Run artifacts and failure states | Source/model/config identities, progress, status and no-overwrite tests | Hard kills may leave running markers; no automatic resume |
| Compute accounting | Backbone calls/tokens, scorer fit, cached lookups, timing, memory | Finite-cache costs do not prove general generation savings |
| Easy user commands | runbook.md and CLI smoke/evaluation | User supplies the GPU allocation and executes real commands |

## Gates left for the user

1. Real ProGen2 numerical checks and the 256-completion, three-seed TV <0.05 gate.
2. Mega-scale scorer validation before biological guidance starts.
3. Actual measured-outcome comparisons; better proxy rewards alone are insufficient.

Broader mutation neighborhoods, unseen-domain grouped evaluation, RL/SMC comparisons,
and claims about scalable generation or amortization are later research milestones,
not results established by this handoff.

## 17 October 2026 — Diversity development handoff

All 41 synthetic CPU tests pass; lint and formatting pass. Added coverage for
uniform-leaf correction and zero residual behavior, cached causal probability
parity, stable deduplicate-before-rank selection, no refill, label-independent
selection, original/corrected smoke fits, failed scorer visibility, source mutation
rejection and byte-identical saved scientific replay. Only tiny random-model fits
were run. [Test log](verification/diversity-tests.txt).

[CPU readiness](verification/diversity-readiness.json) verifies frozen development
source files, tokenizer and full checkpoint hashes without loading ProGen or
initializing CUDA. GPU availability/convergence has not been tested by the agent.
[Handoff audit](verification/diversity-implementation-audit.json) records protected
artifact checks, diagnostic reproducibility, config/code identities and local links.
The first smoke attempt exposed a reused-validator adapter error; it was corrected
before the final passing suite. Real ablation performance is unestablished.
