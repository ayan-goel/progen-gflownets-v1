# Development target diagnostic results

15 October 2026. User-run `runs/target-diagnostic-v1` completed in **62.08 seconds**.
This was a real frozen-ProGen2/scorer experiment, not a smoke test. No GFN was
trained in this diagnostic. The agent inspected and reproduced saved evaluation
on CPU only; all output files matched byte-for-byte and source snapshot hashes
verified. [Audit and numerical results](../verification/target-diagnostic-result-analysis.json).
Raw run outputs and the five reserved confirmation parents remain untouched.

## Main finding

Predicted-reward reweighting improves exact expected measured stability on all
three development supports using the original prior (alpha=1, beta=1).
This holds for ridge, single MLP and ensemble: nine positive scorer/parent results.
All 18 individual fitted models pass the existing inner validation screen.

| Parent | Original-prior mean | MLP target mean | Difference, kcal/mol |
|---|---:|---:|---:|
| 1O6X.pdb | 3.85331 | 4.13717 | +0.28386 |
| 1UBQ.pdb_L67S | 3.94959 | 3.99475 | +0.04515 |
| 1UFM.pdb | 2.94203 | 3.09141 | +0.14938 |
| Unweighted mean of within-parent differences | — | — | +0.15947 |

These are exact probability-weighted means over measured development candidates,
not new laboratory measurements, trained-GFN outcomes or statistically established
generalization. Only two parents are new; the anchor was already inspected.
Scorer outer Spearman correlations for the single MLP are 0.8340, 0.8340 and 0.9020.

## Why this does not overturn the first pilot

The previous 1O6X result used its held-out test support: baseline 4.28005, target
4.26993 kcal/mol. This diagnostic uses its **different outer development support**,
whose baseline is 3.85331, and a smaller fitting partition with separate inner
checkpoint selection. Candidate support changes the locally normalized prior too.
Thus this is evidence that useful targets occur across these development tasks;
it is not proof we fixed the original failing test case or that its result was wrong.

## Scorer choice

At alpha=1, mean measured gains are:

| Scorer | Mean gain, kcal/mol |
|---|---:|
| Ridge | +0.14456 |
| Single MLP (seed zero) | +0.15947 |
| Five-MLP ensemble | +0.15999 |

The ensemble improves aggregate gain over the single MLP by only 0.00053 kcal/mol
on this panel and is not uniformly better per parent. It has better MAE on each
outer development set, but that does not establish a meaningful design advantage.
Ensemble-disagreement/error Spearman correlations are only 0.114, 0.113 and 0.171;
these do not justify a confidence-bound interpretation or an uncertainty penalty.
Recommend the single MLP as the simpler primary procedure, not as a statistically
proven best model. Retain inner RMSE checkpoint selection; do not fix each new
parent's checkpoint step to the development-selected steps.

## Prior tempering is a tradeoff, not a universal improvement

All tested targets improve on their corresponding alpha prior. But reducing alpha
changes the baseline itself, so that gain alone cannot establish improvement over
original ProGen. Single-MLP results make the distinction concrete:

| Parent | Alpha=0.5 gain vs matched prior | Alpha=0.5 gain vs original prior |
|---|---:|---:|
| 1O6X.pdb | +0.57346 | +0.33478 |
| 1UBQ.pdb_L67S | +0.15593 | **−0.03617** |
| 1UFM.pdb | +0.13118 | +0.20182 |

At alpha=0.25, UBQ is 0.12108 kcal/mol below the original prior. Flatter priors
increase coverage, but do not improve raw measured stability uniformly. Keep
alpha=1 as primary for confirmation; retain tempering as documented development
evidence rather than silently changing the objective to maximize its headline gain.

Concentration is unresolved. Original-prior mass in the first three positions is
83.25%, 93.18% and 87.84%. Under the single-MLP alpha=1 targets, the expected unique
counts in 256 draws are only 11.94, 12.07 and 13.15. The corresponding baseline
counts are 14.03, 12.42 and 12.99. Reward tilting does not guarantee more diversity.

## Generate-and-select remains a strong baseline

For alpha=1, taking 32 score-selected slots from 256 raw proposals gives the
following measured means, averaged across three sampling seeds. Duplicates remain:

| Parent | Prior then selection | Exact target then selection |
|---|---:|---:|
| 1O6X.pdb | 4.48600 | 4.48489 |
| 1UBQ.pdb_L67S | 4.08039 | 4.10461 |
| 1UFM.pdb | 3.34039 | 3.29597 |

The exact target does not consistently beat generate-and-select after selection.
These small sampled differences are descriptive, not significance tests. Selected
batches contain approximately 2–5 unique sequences despite 32 slots. Neither
policy is a demonstrated solution for producing 32 distinct useful candidates.
Raw expected stability, selected quality, duplication and cost must remain separate
outcomes in confirmation. Exact categorical sampling is a privileged reference;
this experiment establishes no computational advantage for GFN on enumerable sets.

## Recommended next experiment

Freeze the primary development choice before any confirmation labels are inspected:
**single MLP seed zero, alpha=1, beta=1**, with the existing fitting-only
normalization, inner validation and 500-update scorer budget. Keep all five
preselected confirmation parents. Do not select only parents with favorable
measured targets or delete failures.

For each confirmation parent, fit the scorer using its declared fit/inner split,
freeze it, and train the existing GFN and supervised-guidance heads against predicted
rewards on its withheld test candidate identities. Preserve the established
20,000-update schedule and guidance seeds 0–2. Reveal test measurements only after
all choices and fits are fixed. Compare original ProGen, generate-and-select,
supervised guidance, GFN and the exact target, with matched 256-proposal/32-slot
budgets and complete duplicate/cost accounting. Report predictive-screen failures
explicitly rather than conditioning the final report on passing parents.

Primary interpretation: does the predicted target improve measured test stability,
and does GFN reproduce that target? Separate the harder question of beating
selection and supervised guidance at useful diversity/cost. With five proteins,
report per-parent effects and their spread; model seeds are not additional
independent proteins. Do not claim unseen-family prediction: scorers still receive
local training labels for each parent.

The confirmation runner is the next implementation stage; this review introduces
no new training command or configuration. The [diagnostic implementation record](target_diagnostic_2026-10-15.md)
retains the executed choices and commands. Larger supports and alternative scorers
remain conditional follow-ups, not prerequisites for this bounded confirmation.
