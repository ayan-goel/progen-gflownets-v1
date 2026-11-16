# Property-guided IDR generation with frozen ProGen2

**Scope: 16 November 2026. Keep frozen-ProGen GFlowNet guidance. Conditioned fixed-bank TB with a snapshot gradient correction passes the finite learning gate in six sampled fits. Stop local CPU tuning; next assess transfer to new states and fully branching generation. See [variance diagnosis and repair](docs/experiments/idr_variance_repair_2026-11-16.md).**
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
benchmark, not an exact prediction of biological function. Disorder was evaluation-only
in completed runs; the new candidate below explicitly includes predicted disorder.
Hydropathy is a possible later second descriptor.

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

### Revised reward contract: joint_acceptance_v1

The completed joint-acceptance candidate prioritizes satisfying all existing charge tolerances and
the existing predicted-disorder screen, rather than closeness to exact charge
centers. Define `A(x)=1` for a joint pass, zero otherwise, and `V(x)` as the sum of
nonnegative outside-tolerance violations for both block charges, charged fraction,
and predicted-disorder fraction. Constraints inside tolerance contribute zero.
The [reward record](docs/experiments/idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1)
defines the units, thresholds, rationale and saved-pool evaluation in full.

\[
w(x)=\begin{cases}1 & A(x)=1,\\
\{\rho+(1-\rho)e^{-V(x)}\}/M & A(x)=0,
\end{cases}
\qquad R_{\rm new}(x\mid c)=p_0(x\mid c)w(x).
\]

`w` is the property multiplier; `M=20` sets the minimum acceptance advantage and
`rho=0.25` sets the failure floor relative to near-failure credit. Other symbols
retain their definitions above. Every joint pass has multiplier 1; failures get
0.0125–0.05. Equivalently use `S_new=log(w)` and **beta=1**, not historical beta 2.
The target is `R_new` normalized over allowed tails. Its relative probabilities
among passing tails equal those of the frozen prior; perfect fitting does not
guarantee every output passes or biological novelty.

Disorder uses unrounded isolated-tail metapredict 3.0.2 V3 scores: at least 80%
of generated residues must score strictly above 0.5. V3 becomes an optimized proxy,
not independent evaluation. V2 remains sensitivity only; no experimental disorder
claim follows. A later biological claim needs evidence not used in this reward.
Charge remains a nominal sequence calculation under the original conventions.

The separate scorer, saved-data audit and [TB comparison runner](docs/experiments/idr_acceptance_2026-11-13.md)
are implemented. `idr-acceptance` explicitly uses this version; historical commands
still execute their original rewards. It trains three fresh heads under inherited
800-update settings and compares fresh samples from prior/new/old policies, reusing
the old TB checkpoints. Real execution remains user-run. Preserve historical evidence.

## Training and evaluation contract

### Current reward comparison: continuous_acceptance_v1

The user authorized a [bounded reward-only comparison](docs/experiments/idr_continuous_2026-11-15.md)
after the supervised-start TB result. Use `R_lambda(x|c) = p0(x|c) exp(-lambda V(x))`,
where `x` is the complete tail, `c` the prefix, `p0` the frozen prior, `V` the same
outside-tolerance violation sum above, and `lambda=2.5` the audited penalty.
Compute the log property factor directly, with effective beta one. Remove the
failure floor and pass bonus; preserve every pass criterion and optimized V3 proxy.
This changes the target distribution and may increase extreme negative log rewards.

Train one new TB/replay head from the same original supervised checkpoint as the
completed warm-start control. Keep architecture, optimizer, 400-update budget,
training/replay seeds and bank fixed. Reuse the completed current-reward TB and
unchanged starting head; sample all three afresh. No new supervised fit occurs.
The protocol owns calibration, provenance, criteria and stopping; real execution
remains user-run. Historical commands and results retain their original rewards.

### Shared contract

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
remains evaluation-only in those historical runs. Real charge training has not been run by the assistant.
The reference snapshot, scorer and environment are prepared. The user runs all
real experiments and every GPU operation; only brief CPU correctness checks are
allowed during development. The authorized [real-state diagnostic](docs/experiments/idr_learning_diagnostic_2026-11-12.md)
has captured fixed-policy prefix states and measured gradients/disposable updates.
Its results motivate the authorized [direct-context comparison](docs/experiments/idr_context_2026-11-12.md):
add a zero-initialized 5-to-20 task-context output path and compare fresh fits
against the unchanged head under the same reward, TB schedule and sampling budget.
The user completed this comparison: context sensitivity increased but raw charge
and joint disorder success did not improve. The authorized [branch-rate follow-up](docs/experiments/idr_branch_rate_2026-11-12.md)
is implemented: three fresh heads with a tenfold rate only for the added matrix,
using pinned completed controls. Other scientific/training choices remain fixed.
The completed bounded test shows a small observed generation gain. The authorized
[TB/SubTB comparison](docs/experiments/idr_subtb_2026-11-13.md) is implemented with
800 updates per method/seed and fresh evaluation. SubTB adds an intermediate
flow estimator, parameterized after subtracting the known prefix log-prior; its
terminal flow remains fixed by the same reward. Both methods retain the same
token head and rates. The protocol owns segment weighting, auxiliary optimization,
matched budgets and the bounded decision rule. The completed run favors TB on
charge and joint success in all three seeds; SubTB fails the practical screen.
The subsequent [reward audit](docs/experiments/idr_reward_audit_2026-11-13.md)
identified a mismatch between the smooth charge score and operational pass criteria.
The revised joint-acceptance reward makes disorder an optimized proxy. Its
[completed TB comparison](docs/experiments/idr_acceptance_2026-11-13.md#completed-results-and-next-decision--13-november-2026)
achieves 3.13% joint success versus 3.65% for old TB: better target alignment does
not establish successful learning. Saved trajectories show rare successful examples
and poor fit on successful outputs despite falling loss. The subsequent bounded
replay-assisted pilot kept this reward and frozen backbone/head
architecture fixed, with a supervised same-head diagnostic control as motivated
by the [alternative comparison](docs/decisions.md#2026-11-13--compare-alternatives-before-committing-to-replay).
The control tests steering capability, not exact sampling of the GFN target.
The user authorized this follow-up. Its [implemented protocol](docs/experiments/idr_replay_2026-11-13.md)
fixes one seed, three fresh heads, 800 updates and 16 trajectory examples per update.
Replay uses a fixed bank drawn only from earlier training trajectories and fresh
online samples; its comparator is newly fitted on-policy TB. The supervised arm
uses the same head and successful bank rows, with token likelihood instead of TB.
The [completed results](docs/experiments/idr_replay_2026-11-13.md#completed-results--14-november-2026)
are 3.91% joint success for on-policy TB, 4.79% for replay and 8.69% for supervision.
Neither meets the predeclared +5-point gain screen. Supervised steering with this
same head motivated the now-authorized [supervised-start comparison](docs/experiments/idr_warmstart_2026-11-15.md);
it does not establish exact target fitting. Compare the unchanged head with continued
supervision and TB/replay, both starting from that head. The protocol fixes the
400-update budget, common optimizer reset, training-only normalizer initialization
and fresh evaluation. The completed comparison gives 9.38% joint success unchanged,
16.11% with continued supervision and 7.03% with TB/replay. TB improves a limited
selected-bank relative-target diagnostic but fails the practical screen. End this
continuation route; assess reward discrimination before specifying another
experiment. The completed protocol owns interpretation and the next decision.
Stop the previous replay-only intervention; no extension is automatic. All real
execution remains user-run. Preserve historical configurations. No broad guidance success is established.

Keep existing stability artifacts and code intact. Pause the proposed ESM stability
comparison. Do not restore retired pipelines, introduce generator fine-tuning,
SMC or speculative decoding as part of the baseline experiment.
