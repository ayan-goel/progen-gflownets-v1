# Proposed next steps after the first stability pilot

15 October 2026. **User-authorized plan.** The development diagnostic is complete and the confirmation runner is now
implemented; see the [implementation and GPU handoff](experiments/target_diagnostic_2026-10-15.md).
The text below preserves the rationale and later decision-dependent stages.
The spec now explicitly includes the diagnostic ablation; completed runs and their
configurations are unchanged. All real fits and GPU work remain user-run.
The original investigation used only saved-tensor CPU analysis and literature.

## Decision

Keep the GFlowNet architecture. Next establish a useful stability target and a
meaningful candidate prior before running more guidance training. The central
question remains: can a small GFN head guide frozen ProGen2 toward measurably more
stable, varied protein sequences at a useful cost?

## Evidence

The [pilot record](experiments/biological_first_results_2026-10-15.md) contains the
full results. Over all 204 held-out variants of 1O6X:

| Quantity | Baseline | Ideal predicted-reward target |
|---|---:|---:|
| Expected predicted stability, kcal/mol | 3.89403 | 3.97291 |
| Expected measured stability, kcal/mol | 4.28005 | 4.26993 |
| Probability in five most likely candidates | 91.42% | 89.87% |

GFN's three final target TV distances are 0.001311, 0.0000106 and 0.000339.
The target's measured difference is about −0.0101 kcal/mol: a small descriptive
result on one protein, not evidence of statistically significant deterioration.
The held-out scorer Spearman correlation is 0.829, yet its guidance target does
not improve measured stability. Overall correlation is insufficient for this use.

### New prior diagnostic

The [audit JSON](verification/post-pilot-prior-audit.json) records hashes and
calculation definitions. Positions are one-based; no stability labels or predicted
scores entered these calculations.

| Prior construction on the same 204 candidates | Mass in positions 1–3 | Top-one mass | Effective candidate count* |
|---|---:|---:|---:|
| Current locally normalized ProGen prior | 96.02% | 38.52% | 3.58 |
| Uniform logits with the same local masking | 96.67% | 33.33% | 3.93 |
| Complete-path cached 20-amino-acid likelihood, conditioned on support | 1.43% | 95.74% | 1.09 |

*Effective count is 1/sum(probability squared); it is not the number of candidates
with nonzero probability. Only seven candidates mutate positions 1–3.

Why the uniform control matters: before spending the single mutation, a prefix
with k legal substitutions and a legal reference continuation chooses a mutation
with probability k/(k+1) under uniform logits. Once mutated, its suffix is forced.
Repeated local choices favor early mutations even without any learned preference.
This is a consequence of the defined construction process, not a discrepancy
between the implementation and spec. The control does not quantify how much of
the actual ProGen distribution each cause explains.

The complete-path alternative includes likelihoods of forced residues, but cached
logits cover only 20 amino acids, not the original vocabulary or an EOS term.
It is a diagnostic counterfactual, not a replacement prior or an exact recovery
of original ProGen probabilities. Switching to it alone worsens concentration.

## 1. Establish a development and confirmation protocol

Propose three development parents: the already inspected 1O6X case plus two new
parents selected by valid measurement coverage and distinct sequence clusters.
Reserve five further parents for confirmation using the same outcome-independent
selection rule. Audit sequence similarity between these groups; cluster labels
alone do not prove absence of homology. Save the manifest before running models.
These counts bound the initial cost; they are not a statistical power calculation.

Keep the present single-substitution task while resolving the target. For new
parents retain 70/15/15 train/development/test identities, with the reference in
train. Select scorer checkpoints using an inner split of the training portion,
then assess target utility on the outer development portion. The implemented inner split reserves 20% of outer training variants with seed 43;
the reference remains in fitting and there is no post-selection refit.
Confirmation test labels remain sealed until the procedure is frozen.

The inspected 1O6X test set is now development evidence for future changes. It
cannot become a fresh test by renaming or reshuffling it. Report every preselected
parent, including failed scorer screens and negative results; do not select only
parents where guidance works. This evaluates repeated within-parent learning,
not a scorer's zero-shot generalization to unseen protein families.

## 2. Run an exact-target diagnostic before another GFN fit

For each development support, compute the baseline and ideal predicted target
exactly. Report predicted and measured mean stability, fraction above reference,
prior-weighted prediction error, top-selected ranking, entropy, duplicate rate and
mutation-position distribution. Also include a uniform-over-candidates diagnostic
to distinguish score quality across the support from quality under the current
prior. Uniform candidate sampling differs from uniform legal-action logits.

Compare existing ridge and MLP scorers first. Add an ensemble mean of five MLP
fits (seeds 0–4, existing architecture and optimizer settings) as a bounded
comparison. Five is a practical variance diagnostic, not a proven optimal size.
Measure whether disagreement predicts errors before considering a penalty;
shared model bias can survive averaging. Keep each scorer frozen during guidance.

Separate prior and scorer changes. Retain the current prior as the control;
propose a small sequence-level tempering diagnostic with alpha in {1, 0.5, 0.25}
and beta fixed at 1. These are declared development choices, not values selected
from the inspected test results. They test sensitivity to excessive prior
concentration. The uniform diagnostic is a reference, not the proposed generator.
Do not silently equate sequence tempering with a token-logit temperature.

Proposed diagnostic family:

    p_alpha(x | c) = p0(x | c)^alpha / sum_u p0(u | c)^alpha
    R_alpha(x | c) = p_alpha(x | c) * exp(beta * s_hat(x,c))
    p_star(x | c) = R_alpha(x | c) / sum_u R_alpha(u | c)
    s_hat(x,c) = [dG_hat(x) - dG_hat(x_ref)] / sigma_train

Every symbol: x is a complete candidate protein; c specifies its reference and
candidate support; u ranges over that same support; x_ref is the reference;
p0 is the current locally constrained frozen-ProGen prior; alpha controls how
concentrated its sequence probabilities are; p_alpha is the normalized tempered
prior; dG_hat is the fixed scorer's predicted stability in kcal/mol; sigma_train
is the training-label standard deviation in the same units; s_hat is dimensionless
predicted improvement; beta is guidance strength; exp is the exponential function;
R_alpha is an unnormalized positive reward; p_star is the desired distribution.
Alpha=1 recovers the current prior. Any adopted alpha change requires an explicit
spec update and consistent prior/target/baseline probabilities before GFN training.

**Decision point:** if exact targets offer no repeatable positive measured gain
across development parents, stop guidance training and address scorer/support
quality. If gains are confined to one parent, report that limitation. A positive
development result is a reason to run confirmation, not a biological success claim.
Avoid selecting ever more settings on the same development panel.

## 3. Confirm with matched guidance comparisons

Freeze the selected scorer/prior procedure before opening confirmation labels.
Compare constrained ProGen, generate-and-select, supervised guidance and GFN.
Retain the existing 20,000-update schedule and seeds 0–2 for the first comparison;
current results do not justify a larger head or more optimization.

Use 256 raw proposals per method and 32 selection slots, reporting duplicates
explicitly. The old 1,024-proposal generate-and-select versus 256-proposal guidance
comparison is not a matched proposal-budget result. Report exact expectations
and raw proposals before selected subsets, plus unique sequences and Hamming
diversity. Measure training, feature extraction, scorer calls and sampling cost;
matching proposal count alone does not match compute. Report per-parent results
and their spread, without treating sampled duplicates as independent biological
replicates or training seeds as independent proteins.

Include the enumerated exact target as a diagnostic reference: at this scale we
can sample it directly without GFN. The finite pilot can establish correctness
and reward utility, but cannot establish a computational need for amortized GFN.

## 4. Expand only after target utility is credible

If lightweight reward models remain inadequate, audit ThermoMPNN/SPURS training
splits and label signs before using a checkpoint as a comparator. A model trained
on our test measurements is not an independent oracle. Structure-aware replacement
is a follow-up, not part of the immediate bounded experiment.

If confirmation supports measured improvement, move to larger measured mutation
neighborhoods and then supports where enumeration is impractical. Audit double
mutant coverage and scorer generalization first; unmeasured designs cannot be
reported as experimentally verified. That stage can test the value of amortized
GFN guidance beyond exact reweighting.

The [focused literature update](../literature/POST_PILOT_REVIEW_2026-10-15.md)
provides the scientific rationale and source-reading limits. The [runbook](runbook.md) now provides the implemented development diagnostic
command. The development review supports confirmation; its [fixed runner and protocol](experiments/confirmation_plan_2026-10-16.md) are now ready for the user.
