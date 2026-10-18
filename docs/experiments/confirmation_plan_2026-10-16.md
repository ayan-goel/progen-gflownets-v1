# Five-parent confirmation: protocol and GPU handoff

16 October 2026. **Implemented and ready for the user to run; confirmation outcomes
remain unevaluated.** The agent performed data preparation and synthetic CPU tests
only. No GPU operation, real scorer fit or real guidance training was launched.

## Decision and purpose

The [development diagnostic](target_diagnostic_results_2026-10-15.md) found positive
exact-target measured gains on three development supports at the original prior.
A single MLP and five-model ensemble had nearly identical average gains; flatter
priors increased coverage but sometimes reduced stability versus original ProGen.
The user authorized confirmation of the simple procedure.

Freeze **MLP seed 0, alpha=1, beta=1**, using the existing inner-validation selection
and frozen ProGen2-medium. This is a development-informed procedure, not a claim of
an optimal scorer or prior. The present experiment asks whether its target improves
measured outcomes on reserved parents and whether trained GFN guidance reproduces
that target. Separately compare quality after selection, diversity and cost. Do
not equate raw stability gains with beating generate-and-select.

## Fixed parents and data

All five parents reserved before development analysis remain in the run:

| Parent | Fit | Inner validation | Reserved test candidates |
|---|---:|---:|---:|
| 2K52.pdb_L42A | 756 | 188 | 203 |
| 2M2J.pdb | 755 | 188 | 202 |
| 2AMI.pdb | 751 | 187 | 201 |
| 2L33.pdb | 745 | 186 | 200 |
| 1WCL.pdb | 733 | 182 | 196 |

The [panel manifest](../../data/processed/stability-panel-v1/manifest.json) retains
its original coverage selection, cluster/sequence-similarity screen and split
hashes. This evaluates within-parent learning across reserved proteins; each scorer
receives that parent's local fitting labels. It is not zero-shot prediction on
unseen families. Confirmation parents' outer development labels are not used.

The original panel combined test identities and labels in one file. To avoid
opening that file during fitting, a new [identity-only support](../../data/processed/confirmation-support-v1/manifest.json)
was prepared from the unchanged raw CSV. The preparation reads sequence/parent
identity and whether a finite dG_ML label exists, then reproduces the original
sorted unique single-mutant list, seed-42 permutation and 85% cut point. It does
not retain, inspect or rank numerical outcomes, and does not open panel test-label
files. Source/panel hashes and identity-file hashes are recorded. This preserves
the existing test split rather than selecting a replacement. Exact identity/label
agreement is checked after outcome release; synthetic tests establish reconstruction
parity without exposing real test values during development.

## Execution order and outcome isolation

1. Verify the configuration, panel/support hashes, allowed fit/inner data, the passed
   exact gate, completed development diagnostic and matching checkpoint hashes.
   Require all five reserved parents; no favorable subset can silently replace them.
2. Load frozen ProGen once. For each parent, extract fit/inner features and select
   an MLP checkpoint by inner RMSE. Fit normalization on fitting data only; do not
   refit after inner selection. Cache predicted test-candidate rewards and the legal
   prefix tree using candidate identities alone. Reference measurements belong to
   fitting data and may supply the known reference baseline.
3. Release the backbone from memory after all setup. For each passing scorer, train
   GFN and supervised guidance for every declared seed using the existing learners.
   Neither reads test outcomes. Only final checkpoints are used; no best-TV or
   measured-outcome checkpoint selection is introduced.
4. Require a finished status for every parent. Seal configuration, panel manifest,
   tasks, scorers, fit reports, costs and final guidance checkpoints with hashes in
   `fits_complete.json`. Missing or changed artifacts block outcome evaluation.
5. Only then verify/copy test labels into the run and evaluate every method on CPU.
   A premature interrupted run has no evaluation labels. Original panel and run
   artifacts are never overwritten; a new run requires a fresh output directory.

**Failed screens:** retain the original MAE/Spearman scorer screen. A failing
scorer skips both learned guidance methods for that parent and is explicitly
reported. Its prior/target/error analysis remains diagnostic, not a validated design
result. All five parents stay visible; aggregates report both the declared count
and contributing count. Do not describe performance on passing parents alone as
unconditional five-parent success. A numerical training error aborts the run before
outcome release rather than silently skipping a seed.

The `evaluate-confirmation` command requires the complete fit seal. It can finish
post-fit evaluation after a failure at the evaluation stage, or reproduce a completed
run using saved CPU artifacts. It cannot resume incomplete training. Replays use
saved label copies once available; they never refit or reload ProGen.

## Frozen numerical choices

Executable configuration: [confirmation.json](../../configs/confirmation.json).
Inherited settings and reasons are in [hyperparameters](../hyperparameters.md);
these are the material confirmation choices:

| Choice | Fixed value / rationale |
|---|---|
| Scorer | One MLP, seed 0, width 128; no ridge competition or ensemble |
| Scorer fitting | 500 updates, batch 64, Adam lr 0.001, weight decay 0.0001, global clip 1 |
| Inner checkpoint selection | RMSE every 25 updates and final update; first best wins a tie |
| Scorer normalization | Fitting-only means/population standard deviations, floor 1e-6 |
| Inner screen | MAE below fitting-mean baseline and Spearman >=0.2; inherited engineering screen |
| Adam defaults | betas (0.9,0.999), epsilon 1e-8, no AMSGrad; guidance weight decay 0 |
| Prior / reward | Original local legal-action normalization, alpha=1; beta=1 scaled predicted improvement |
| Backbone / feature batches | Frozen ProGen2-medium, float32, batch 8; known validated setup |
| Guidance methods | Existing TB residual head and supervised Monte Carlo future-reward head; width 128, zero output initialization |
| Guidance seeds / updates | 0,1,2 / 20,000 per learner per parent; final checkpoints only |
| Guidance batch / clip | 64 / global norm 1 |
| Learning rates | Head 0.001; TB log-normalizer 0.01; multiply both by 0.1 after update 8,000 |
| Supervised rollout budget | 4,096 prior rollouts, unchanged from the development-era comparison |
| Logging | Every 100 updates plus first/final and rate-boundary updates; no exact-distribution monitoring during training |
| Proposal/selection budget | 256 raw draws / 32 score-selected slots for all policies; duplicates retained |
| Sampling seeds | 100000 + guidance seed; deterministic predicted-score ties preserve proposal order |
| CPU threads | 1, as in the existing run configuration |
| Probability checks | Cached prior sum tolerance 1e-5; evaluated policy sum tolerance 1e-4; measured expectations normalize double-precision log probabilities |
| Replication unit | Protein; aggregate seeds within protein, then weight proteins equally |

No new alpha/beta sweep, uncertainty penalty, scorer search or test-adaptive
hyperparameter change is introduced. The known local-prior position bias remains
part of the control. These choices do not promise diverse proposals.

## Comparisons and reporting

- Original constrained ProGen raw samples.
- Generate-and-select: best predicted 32 slots from those same 256 ProGen draws.
- GFN raw samples and its separately selected batch.
- Supervised guidance raw samples and its separately selected batch.
- Exact predicted target, sampled categorically, as a privileged diagnostic reference.

Measure exact expected measured/predicted stability, gain over ProGen, fraction
above the reference, TV and both KL directions to the target, normalization,
concentration, expected unique count, positional concentration and expected pairwise
Hamming distance. Sampled reports retain every raw/selected identity and report
actual unique counts, duplicates, measured/predicted means, coverage and diversity.
Scorer test errors are revealed only after fitting is sealed.

Raw and selected outcomes are separate tables. A selected batch is not a draw from
the trained target distribution. Repeated sequences remain repeated slots; 32
slots must not be described as 32 distinct proteins. No statistical significance
or broad generalization is inferred from a positive mean across five parents.
No automatic biological success threshold is introduced.

Per-parent setup costs distinguish feature extraction, scorer fitting, candidate
scoring and prefix caching. Training costs retain all seed/method records. Evaluation
costs separately record distribution computation, sampling and selection/metrics.
Shared feature/scorer setup is charged explicitly; zero backbone calls during
cached evaluation is not a claim of free online generation. Direct exact sampling
already solves this finite task, so an efficiency claim needs later larger supports.

## Command and outputs

The `.venv`, panel and identity-only support are already prepared. Select your GPU
manually and run from the v1 root:

```bash
source .venv/bin/activate
# Set CUDA_VISIBLE_DEVICES to your chosen GPU first.
scripts/run.sh confirm --output runs/confirmation-v1 --device cuda
```

The launcher preserves your GPU selection and runs training followed by evaluation.
The main report will be `runs/confirmation-v1/evaluation/report.md`; each parent's
`result.json` retains all final-seed metrics, scorer test errors and setup/training
costs. `evaluation/summary.json` reports parent-level effects and coverage/failures.
Raw outputs, source/spec snapshots and the fit seal stay in the run directory.

This is substantially longer than the 62-second diagnostic: five parents, three
seeds, and both guidance learners. The earlier single-parent pilot took about
50 minutes; roughly four hours is a planning estimate if costs are similar, not a
measured runtime guarantee. GPU load and failures can change this. No automatic
scheduler submission, GPU selection, retries or training resume is provided.

For CPU-only evaluation/replay after all fits have finished, choose a new output:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-confirmation \
  --run runs/confirmation-v1 --output runs/confirmation-v1-reevaluation
```

Verification: [CPU test log](../verification/confirmation-tests.txt) and
[readiness record](../verification/confirmation-readiness.json). CPU tests use tiny
random ProGen and artificial four-residue proteins. They exercise both passing
and deliberately failing screens, all-parent outcome isolation, interruption,
sealed-artifact changes, duplicate-aware metrics, matched budgets and replay.
Real confirmation outcomes and GPU execution remain for the user.

## Next decision after the run

Assess whether the predicted target improves measured stability across all reserved
parents; distinguish target failures from GFN approximation errors. Then assess
GFN against supervised guidance and generate-and-select, including distinct-sequence
quality and total cost. Preserve negative results and failed screens. Only after
this review decide whether larger mutation supports or reward/prior changes are
justified; this command does not launch follow-on experiments.
