# First biological pilot: results — 15 October 2026

## Conclusion

The user-run pilot completed in **3,026.42 seconds (50 minutes 26 seconds)**.
Frozen-ProGen GFlowNet guidance accurately learned the predicted-stability target
on the declared finite support. It **did not improve measured stability over the
constrained ProGen baseline**. The exact target itself has slightly lower expected
measured stability, so more accurate GFN fitting cannot resolve that reward mismatch.

This is an end-to-end implementation result and a biological benchmark finding,
not evidence that we designed or experimentally validated new stable proteins.
The 204 candidates are existing measured single substitutions from one domain.
The biological `complete` / `gate.passed` flags certify execution after the scorer
screen; they do not test or certify biological improvement or superiority.

Primary records: [status](../../runs/biological-first/status.json),
[gate/distribution fit](../../runs/biological-first/gate.json),
[evaluation](../../runs/biological-first/evaluation/report.md),
[scorer validation](../../runs/biological-first/scorer_validation.json),
[costs](../../runs/biological-first/costs.json).
The [post-run analysis](../verification/biological-first-analysis.json) records
input hashes, exact finite-support expectations, sample summaries and the most
probable candidates. All originals are preserved. Analysis loaded cached tensors
and saved small heads on CPU only; no backbone, GPU, fitting or new sampling.

## Model fitting and scorer quality

| Seed | GFN TV from predicted target | KL(target‖GFN) |
|---|---:|---:|
| 0 | 0.00131143 | 0.00003602 |
| 1 | 0.00001061 | 0.00000272 |
| 2 | 0.00033946 | 0.00001870 |

These very small whole-support errors substantiate the low sampled TB losses.
The reward distribution is easier to match than the earlier toy and is already
close to the baseline (baseline TV 0.091711). Most sequences have little probability.

The selected MLP (update 450) passed validation: MAE **0.3170 kcal/mol**, versus
0.8535 for the training-mean predictor; Spearman **0.8647**. Across all 204 test
candidates, post-run MAE is **0.3488**, RMSE **0.5012**, and Spearman **0.8285**.
Thus the scorer has useful overall ranking, but global validation/test correlation
does not guarantee reliable ordering among the candidates most likely to be sampled.
Test metrics were computed after completion and did not choose the saved scorer.

## Exact policy expectations, avoiding sample noise

For each policy, sum each candidate's predicted/measured value times its probability
across all 204 candidates. Renormalize saved log probabilities in float64 to remove
small numerical probability-sum error. GFN values below average the three policies;
the ideal target is derived directly from frozen p0 and the predicted reward.
These are raw-policy expectations, not best-of-N selected batches.

| Policy | Expected predicted dG (kcal/mol) | Expected measured dG (kcal/mol) |
|---|---:|---:|
| Constrained frozen ProGen | 3.89403 | 4.28005 |
| Exact predicted-reward target | 3.97291 | 4.26993 |
| GFN, mean over three seed policies | 3.97310 | 4.26975 |
| Supervised, mean over three seed policies | 3.95015 | 4.25379 |

The exact tilt increases predicted stability by **0.07888 kcal/mol** but decreases
measured stability by **0.01011 kcal/mol** relative to p0. The difference is small
and is a descriptive result on this one finite measured set, not a claim of
statistical significance or universal harm. Crucially, it is present in the target,
not just in a lucky/unlucky generated batch or imperfect GFN optimization.
The measured reference is 4.17963: both baseline and guided expected values exceed
that reference, but improvement over the reference is not improvement over ProGen.

## Why the reward tilt does not improve this candidate set

Examples among the five most probable baseline candidates (positions are one-based):

| Variant | Baseline probability | Target probability | Predicted dG | Measured dG |
|---|---:|---:|---:|---:|
| V1G | 0.3852 | 0.3141 | 3.6914 | 4.3287 |
| V1R | 0.3392 | 0.3441 | 3.9488 | 4.4378 |
| G2I | 0.1043 | 0.1386 | 4.2667 | 3.8302 |
| D3G | 0.0438 | 0.0291 | 3.4489 | 4.4194 |
| D3E | 0.0417 | 0.0398 | 3.8769 | 3.8080 |

The predictor overestimates G2I relative to several highly probable alternatives;
the target increases its mass while reducing mass on V1G and D3G. These examples
illustrate errors at influential candidates; all-candidate expectations above
establish the aggregate result. Do not replace predictions with these test labels
or tune rewards to repair these particular held-out variants.

## Diversity and selection

The baseline's five most probable candidates hold **91.4%** of its mass; the target's
five hold **89.9%**. The baseline's five are all mutations in positions 1–3. This
concentration is part of the locally normalized constrained prior/task, rather
than evidence that a previously broad policy collapsed during GFN training.
Target entropy is slightly higher than baseline entropy, but both are narrow.
The finite, single-substitution support also limits accessible sequence diversity.

In the saved 256-slot raw batches, GFN has **10 / 13 / 10 distinct sequences**;
ProGen has 13 / 11 / 12. Mean raw measured stability is 4.26457 for GFN versus
4.27816 for ProGen and 4.23629 for supervised guidance. Sampled GFN pairwise Hamming
is slightly higher (1.2804 vs 1.1694 for ProGen), but this is not broad sequence
exploration. Novelty relative to training is built into the held-out candidate
support and is not a discovery or a statement about PLM pretraining data.

Mean selected-batch measured stability is 4.35769 for generate-and-select,
4.16218 for GFN selection and 4.07813 for supervised selection. These selected
32-slot batches contain only 2 / 3 / 3 distinct sequences per seed, respectively.
Generate-and-select draws 1,024 proposals versus 256 for each guided policy;
this is **not** an equal proposal-budget comparison. Duplicate slots are retained.
Do not claim that this establishes general superiority of either approach.

## Cost and what to do next

Setup took about 86 seconds. Each TB seed took about 15.5 minutes; supervised
training took about 41–44 seconds, with fixed cached rollout targets. The methods
have different training information and costs; equal update counts are not equal
compute. No extra exact training-time monitoring was enabled for biology.

Stop additional GFN optimization on this completed pilot: it already matches its
specified target closely. Next work should examine **reward usefulness and candidate
support**, not simply increase optimizer steps or select a stronger beta from these
test outcomes. In particular:

1. Use training/validation evidence to assess whether ranking is reliable among
   high-prior and high-predicted-stability candidates, including calibration and
   uncertainty. Overall RMSE/Spearman alone is an insufficient reward-utility screen.
2. Audit the concentration induced by the local legal-action prior and limited
   single-substitution support before changing that prior or broadening the task.
   The current p0 definition is intentional and specified; changing it would require
   an explicit documented experimental choice, not a silent probability fix.
3. Predeclare any new scorer/reward/support choices and evaluate them on additional
   references or a fresh held-out set. Treat this pilot as now-inspected evidence;
   do not repeatedly tune against its test labels and present it as fresh confirmation.
4. Match proposal budgets in future selected-batch comparisons and report both raw
   and selected distributions with duplicate counts and compute.

No next training configuration was created and no experiment was launched. The
scientific objective remains frozen-PLM guidance for stability. This first pilot
identifies a distinction between successful distribution learning and useful
biological guidance; it does not yet establish the latter.
