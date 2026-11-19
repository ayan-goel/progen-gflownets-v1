# Property-guided IDR generation with ProGen2 and GFlowNets

**Scope: 18 November 2026. The goal remains GFlowNet-based IDR property generation.
The completed [1,600-versus-800 comparison](docs/experiments/idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026)
raises mean joint charge/predicted-disorder success from 29.39% to 32.62%, but
both repeats miss the predefined five-point gain. The user nevertheless chose
1,600 as the working budget and deferred further budget/checkpoint tests; the
historical gate outcome is unchanged. The method adapts ProGen's copied final
block and GFN head against a fixed
original prior. This is improved generation on the development task, not exact
target sampling, property/prompt generalization or biological function. The user
selected the matched supervised comparison described below.**
The project now investigates IDP/IDR property guidance. Stability experiments remain
completed groundwork. Their [historical specification](docs/stability_spec_2026-11-10.md)
and [results](docs/status.md) preserve their original definitions; they are not
instructions to launch more stability experiments.

## Objective

Guide a pretrained protein language model toward specified properties of
intrinsically disordered sequences using a small GFlowNet head. Study when guidance
learns the desired distribution, how reward quality affects it, and whether it
provides useful diversity or efficiency compared with direct selection/design.

The current [objective-comparison protocol](docs/experiments/idr_objective_2026-11-18.md)
trains GFN and reward-weighted supervised policies from the same initialization,
with independent final-block/head copies and the same refreshed scored examples.
Both receive 1,600 updates in each of two fresh repeats. Supervised weights use
only the terminal property factor, normalized within prompt; this baseline fits
the weighted shared proposal, not the exact GFN target. Report fresh property
success, diversity, copies and cost without claiming independent-pipeline
efficiency. Current charge/reward conventions remain fixed; other properties and
external-method implementations are deferred. The protocol owns numerical choices.

The first proposed object is an isolated, fixed-length IDR sequence. This does not
establish behavior when that region is inserted into a full protein. ProGen2 remains
the primary backbone; IDiom is literature and a candidate comparator, not an
implicitly approved replacement. Flexibility is not the initial property.

The [IDP/IDR review](literature/IDP_IDR_REVIEW_2026-11-12.md) owns initial source
assessment. The [updated benchmark roadmap](literature/IDR_BENCHMARK_ROADMAP_2026-11-17.md)
owns proposed method comparisons, functional tools and validation requirements.
The
[pivot decision](docs/decisions.md#2026-11-12--pivot-from-stability-to-idpidr-property-guidance)
explains why we are leaving stability as the active biological task.

## Proposed first property

Use a local charge profile, calculated directly from the completed sequence under
explicit residue-charge and window conventions. Control charged-residue content;
then test charge ordering at fixed composition. This provides an exact descriptor
benchmark, not an exact prediction of biological function. Disorder was evaluation-only
in the initial charge experiment; the current continuous reward explicitly includes
predicted disorder.
Hydropathy is a possible later second descriptor.

The [baseline protocol](docs/experiments/idr_baseline_2026-11-12.md) fixes the inherited
length, prompts and support. The completed [charge-pilot protocol](docs/experiments/idr_charge_2026-11-12.md)
introduced a coarse two-block profile (+0.125 then −0.125 mean nominal charge)
and charged fraction 0.25. The current continuous reward retains its acceptance
tolerances, with effective beta one and penalty 2.5 as defined below; the pilot's
beta-two reward is historical.
The existing nine-residue sliding profile is a secondary descriptor; finer targets
and exact fixed-composition generation remain later experiments. These numbers
are explicit engineering choices informed by development data, not literature defaults.

## Architecture and distribution

The policy produces prefix features and next-residue logits, and a trainable head
adds token corrections. A fixed scoring procedure evaluates complete sequences.
Original ProGen stays frozen to define the reference probabilities.
Earlier experiments trained only the guidance head and normalization estimates.
The completed last-block comparison additionally trains a copy of the final
transformer block, retaining the original frozen model as its reference prior.
Its confirmation preserves that architecture; it does not change the IDR objective
or authorize whole-model fine-tuning.

The proposed target retains our prior-times-property formulation:

\[
R(x\mid c)=p_0(x\mid c)\exp\{\beta S(x,c)\},\qquad
p^*(x\mid c)=R(x\mid c)/Z(c),\qquad
q_\theta(x\mid c)\approx p^*(x\mid c).
\]

| Symbol | Meaning |
|---|---|
| \(x\) | Complete generated IDR tail, following the supplied prefix. |
| \(c\) | Fixed design context: prompt, length, allowed sequence space and requested property profile. |
| \(p_0(x\mid c)\) | Frozen-ProGen baseline probability under that explicitly defined generation convention. |
| \(S(x,c)\) | Dimensionless property score; proposed to decrease with profile mismatch. Higher is better. |
| \(\beta\) | Nonnegative strength of property guidance, fixed for each initial run. |
| \(R(x\mid c)\) | Positive, unnormalized combined sequence weight. |
| \(Z(c)\) | Sum of those weights over the allowed completed sequences. |
| \(p^*(x\mid c)\) | Desired normalized sampling distribution. |
| \(q_\theta(x\mid c)\) | Learned sampling distribution from the policy's ProGen plus guidance head; the final block is also trainable in the adapted arm. |
| \(\theta\) | Trainable policy parameters: guidance head and, in the adapted arm, the copied final transformer block. |

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

### Completed experiment: unchanged-design confirmation

The [confirmation protocol](docs/experiments/idr_lastblock_confirmation_2026-11-17.md)
fixes two new training/replay/evaluation seed sets before execution. Each repeats
the same 400-update frozen-versus-adapted GFlowNet comparison from the original
supervised initialization. The saved supervised controls are evaluated afresh;
plain ProGen is a separate final-evaluation arm with no guidance head.

Retain every replicate and every prompt. The original progression criteria remain
unchanged; a separate practical screen checks improvement over plain ProGen.
Both fresh replicates must pass both screens to confirm the result under this
fixed development setting. The historical pilot is reported separately. This
checks training randomness, not unseen-prompt generalization, new properties or
biological function. No additional unfreezing or reward/loss tuning is included.

Both fresh confirmation repeats pass. The subsequent completed
[training-budget comparison](docs/experiments/idr_lastblock_budget_2026-11-17.md#completed-results--18-november-2026)
also passes in both repeats, raising mean joint success from 21.78% to 29.64%.
That result supported 800 updates with the inherited 75%-of-run decay convention. Reward and
architecture stay fixed; the absolute decay point and training duration change
together. This is not exact checkpoint resumption or a pure extra-low-rate-update
test. No outcome-selected checkpoint or automatic extension follows.

The completed [1,600-versus-800 comparison](docs/experiments/idr_lastblock_1600_2026-11-18.md)
keeps reward and architecture fixed, repeats both training streams from the
original initialization, and freshly samples preserved 800-update controls.
Decay moves from after update 600 to after 1,200. Final joint gains are 4.79 and
1.66 percentage points; both miss the five-point raw/noncopy rule while quality
checks pass. The rule initially supported retaining 800, but the user subsequently
chose to keep 1,600 as the working budget and defer further budget/checkpoint tests.
The historical gate remains unchanged; no hard ceiling is established.
The matched supervised baseline with the same adaptable layers and equivalent
scored-data access is now authorized under its separate protocol above.
The protocol owns budgets, copy boundaries, seeds, results and recovery.

A separately trained second charge profile remains
a possible test of property control, not zero-shot target conditioning. The
current head encodes prompt identity, so unseen-prompt testing requires its own
explicit design. The completed protocol owns numerical results and limitations.

### Completed transfer pilot: full-support offline TB

The [implemented pilot](docs/experiments/idr_transfer_2026-11-16.md) retains
`continuous_acceptance_v1` and the original supervised starting head. Compare
unchanged, ordinary TB and variance-reduced TB; both trained arms use identical
full-trajectory conditioning, fixed training-bank sampling, initialization and
400-update schedules. The correction uses a periodically refreshed full-bank
gradient to reduce sampled-gradient noise. All 20 residues remain allowed at
every generated position. No new training rollouts are collected in this pilot.

The primary test is fresh generation: 1,024 outputs per arm, with raw and noncopy
joint charge/disorder success, diversity, conditioning shift and bank coverage.
Finite-bank fit cannot establish full-support target matching; a learned
normalizer can absorb changes in total probability assigned to observed paths.
Do not reuse diagnostic checkpoints as production heads or report bank-normalized
TV as global TV. The protocol owns numerical settings and the bounded progression
screen. The completed run fails that screen despite near-exact training-bank fit;
the result record owns the holdout gap and next design boundary. Real execution
remains user-run.

### Completed reward comparison: continuous_acceptance_v1

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
- Supervised fine-tuning of the same final block and head, with matched initialization
  and data access. The existing head-only control does not establish this comparison.
- One capacity-matched KL-regularized RL method using the same property score and
  reference with corresponding KL strength, not the combined GFN terminal weight
  as its scalar reward. Declare training/tuning cost alongside generation cost.
- GOOSE or direct descriptor optimization, with compatible constraints.
- Composition-preserving shuffles as an order/control comparison.
- IDiom and its post-training workflow as domain-specific comparators, clearly
  distinguished from same-backbone objective ablations.

These additional comparisons are research requirements/proposals, not implemented
experiments or authorization to launch them. The working GFN already performs
partial ProGen fine-tuning; the methodological claim concerns its training objective
and resulting success/diversity/cost tradeoff. Neither property-guided protein
generation nor PLM/GFN combinations are novel by themselves; the linked roadmap
cites GOOSE, IDiom and PG-AbD as relevant precedents.

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
comparison. The completed baseline experiments did not permit generator fine-tuning.
The newly authorized last-block protocol is the specific extension to that boundary;
it does not authorize whole-model training, SMC, speculative decoding or restored
retired pipelines.
