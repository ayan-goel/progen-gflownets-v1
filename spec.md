# Property-guided IDR generation with frozen ProGen2

**Scope: 12 November 2026. IDR baseline completed; first charge pilot completed; optimization diagnostic completed; direct-context comparison ready.**
The project now investigates IDP/IDR property guidance. Stability experiments remain
completed groundwork. Their [historical specification](docs/stability_spec_2026-11-10.md)
and [results](docs/status.md) preserve their original definitions; they are not
instructions to launch more stability experiments.

## Objective

Guide a frozen pretrained protein language model toward specified properties of
intrinsically disordered sequences using a small GFlowNet head. Study when guidance
learns the desired distribution, how reward quality affects it, and whether it
provides useful diversity or efficiency compared with direct selection/design.

The first proposed object is an isolated, fixed-length IDR sequence. This does not
establish behavior when that region is inserted into a full protein. ProGen2 remains
the primary backbone; IDiom is literature and a candidate comparator, not an
implicitly approved replacement. Flexibility is not the initial property.

The [IDP/IDR review](literature/IDP_IDR_REVIEW_2026-11-12.md) owns source assessment,
property/tool choices and dataset leads. The
[pivot decision](docs/decisions.md#2026-11-12--pivot-from-stability-to-idpidr-property-guidance)
explains why we are leaving stability as the active biological task.

## Proposed first property

Use a local charge profile, calculated directly from the completed sequence under
explicit residue-charge and window conventions. Control charged-residue content;
then test charge ordering at fixed composition. This provides an exact descriptor
benchmark, not an exact prediction of biological function. Disorder predictions
are assessed separately. Hydropathy is a possible later second descriptor.

The [baseline protocol](docs/experiments/idr_baseline_2026-11-12.md) fixes the inherited
length, prompts and support. The authorized [charge-pilot protocol](docs/experiments/idr_charge_2026-11-12.md)
now fixes a coarse two-block profile (+0.125 then −0.125 mean nominal charge),
charged fraction 0.25, score scales, beta 2, tolerances and comparison budgets.
The existing nine-residue sliding profile is a secondary descriptor; finer targets
and exact fixed-composition generation remain later experiments. These numbers
are explicit engineering choices informed by development data, not literature defaults.

## Architecture and distribution

Frozen ProGen produces prefix features and next-residue logits. A trainable head
adds token corrections. A fixed scoring procedure evaluates complete sequences.
Only guidance parameters and a normalization estimate are trained initially.

The proposed target retains our prior-times-property formulation:

\[
R(x\mid c)=p_0(x\mid c)\exp\{\beta S(x,c)\},\qquad
p^*(x\mid c)=R(x\mid c)/Z(c),\qquad
q_\theta(x\mid c)\approx p^*(x\mid c).
\]

| Symbol | Meaning |
|---|---|
| \(x\) | Completed IDR sequence. |
| \(c\) | Fixed design context: prompt, length, allowed sequence space and requested property profile. |
| \(p_0(x\mid c)\) | Frozen-ProGen baseline probability under that explicitly defined generation convention. |
| \(S(x,c)\) | Dimensionless property score; proposed to decrease with profile mismatch. Higher is better. |
| \(\beta\) | Nonnegative strength of property guidance, fixed for each initial run. |
| \(R(x\mid c)\) | Positive, unnormalized combined sequence weight. |
| \(Z(c)\) | Sum of those weights over the allowed completed sequences. |
| \(p^*(x\mid c)\) | Desired normalized sampling distribution. |
| \(q_\theta(x\mid c)\) | Learned sampling distribution from ProGen plus the guidance head. |
| \(\theta\) | Trainable guidance parameters. |

The exponential makes terminal weights positive; the property score itself can be
negative. At zero guidance strength, the target is the baseline. Exact score terms,
scales and constraints need a pre-run protocol. This is not a commitment to a
multiobjective weighting or to fitting arbitrary profiles with one shared head.

Define the prior anew for this task. The previous locally masked mutation tree is
not unrestricted ProGen generation. Fixed-length canonical-residue masking and
termination must be explicit and shared across methods. If support changes, report
it; do not compare gains against mismatched priors or silently condition on a
post-generation filter.

## Training and evaluation contract

Retain trajectory balance as the initial objective. Terminal scores can be computed
from per-residue descriptors without adding intermediate supervision. Subtrajectory
balance or explicit local losses require their own justification and comparison.
A contextual annotator is not generally a sum of independent residue effects.

Compare:

- Frozen ProGen on identical support and prompts.
- ProGen generation followed by selection, including discarded candidates/cost.
- GOOSE or direct descriptor optimization, with compatible constraints.
- Composition-preserving shuffles as an order/control comparison.
- IDiom as an optional domain-specific prior comparator, clearly labeled as such.

Assess property-profile error/success, disorder predictions, duplicates/repetition,
diversity, novelty and total cost. Exact distribution distances require an enumerable
subtask; large-space mean reward and TB loss cannot establish exact sampling.
Report raw samples before filtering or top-score selection. Do not discard natural
IDR repetition by default; distinguish reference-like patterns from trivial collapse.

For calculated descriptors, independent recomputation verifies the stated quantity.
For learned biological scores, evaluate using evidence not optimized during training;
agreement between models is not experimental confirmation. Separate functions of
sequence, predicted attributes and measured outcomes in all reports.

## Data and execution boundaries

Choose a bounded reference corpus with provenance, parent/context identities,
sequence-similarity grouping and explicit outcome-access rules. IDiom-DB is a
predicted-IDR corpus, not a table of experimentally measured residue properties.
DisProt and experimental functional datasets have different evidence granularity.
Do not label absent annotations as negative outcomes. Audit overlap before claims
of independent validation or novelty.

The generation-only baseline is complete. The subsequent authorized charge pilot
implements fresh on-policy trajectories, a shared residual head, one log-normalizer
per prompt, raw generation/selection and composition-preserving controls. Disorder
remains evaluation-only. Real charge training has not been run by the assistant.
The reference snapshot, scorer and environment are prepared. The user runs all
real experiments and every GPU operation; only brief CPU correctness checks are
allowed during development. The authorized [real-state diagnostic](docs/experiments/idr_learning_diagnostic_2026-11-12.md)
has captured fixed-policy prefix states and measured gradients/disposable updates.
Its results motivate the authorized [direct-context comparison](docs/experiments/idr_context_2026-11-12.md):
add a zero-initialized 5-to-20 task-context output path and compare fresh fits
against the unchanged head under the same reward, TB schedule and sampling budget.
This implementation is ready; real outcomes are pending. It does not change the
scientific target or unfreeze ProGen.

Keep existing stability artifacts and code intact. Pause the proposed ESM stability
comparison. Do not restore retired pipelines, introduce generator fine-tuning,
SMC or speculative decoding as part of the baseline experiment.
