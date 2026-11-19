# Current project status

**18 November 2026 — Matched GFN versus supervised comparison ready for the user to run.**

[Specification](../spec.md) · [Latest results](experiments/idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026)
· [Runbook](runbook.md)

## Current decision

The user selected [matched final-block GFN versus weighted supervised training](experiments/idr_objective_2026-11-18.md).
Both policies start from the same original weights, adapt the same block/head and
share scored training trajectories from both policies, original ProGen and replay.
Two fresh repeats use 1,600 updates per arm and final-only evaluation. This tests
the objective at equal adaptable capacity and data access. It is not an independent
pipeline cost benchmark. Nineteen tiny CPU tests and the real input-only preflight
pass; no new real experiment has run. The
[runbook](runbook.md#matched-gfn-versus-supervised-training) owns the command.
Another profile and external methods remain deferred.

## Latest completed result

The user completed `runs/idr-lastblock-1600-v1` in **3 h 49 min 39 s**.
Both repeats finished training and evaluation after the earlier storage interruption.
The complete artifacts are at `v1`; the suggested `v2` restart directory is absent.
The [experiment record](experiments/idr_lastblock_1600_2026-11-18.md#interrupted-first-attempt--18-november-2026)
preserves that history without assuming how the earlier folder was made reusable.

| Joint charge/predicted-disorder success | Repeat 1 | Repeat 2 | Mean |
|---|---:|---:|---:|
| Preserved 800-update model, fresh evaluation | 29.39% | 29.39% | 29.39% |
| New 1,600-update model | 34.18% | 31.05% | 32.62% |
| Gain | +4.79 points | +1.66 points | +3.22 points |

Each arm has 1,024 raw outputs per repeat. Both improve, but neither reaches the
predeclared five-point raw/noncopy gain. Disorder and uniqueness checks pass.
Mean predicted-disorder pass is 91.06%, uniqueness is 98.63%, and none of the
668 successes exactly copies the common training union. Charge remains limiting
at 36.91% mean pass. Plain ProGen reaches 2.49% mean joint success in this evaluation.

**The user chose to keep the 1,600-update models and budget, deferring further
budget/checkpoint tests.** This supersedes the earlier recommendation to retain
800; it does not change the recorded five-point gate outcome. The additional gain
is modest relative to doubled training exposure. This supports diminishing returns,
not proof of a hard plateau or accurate global target sampling. The
[completed 800-update study](experiments/idr_lastblock_budget_2026-11-17.md#completed-results--18-november-2026)
previously raised mean success from 21.78% to 29.64%; fresh evaluation explains the
small change in the unchanged 800-update controls' percentages here.

The previous supervised control trains only the head, so it cannot isolate the
GFN objective's advantage. This motivates the authorized matched comparison above.
One contrasting charge profile remains a possible subsequent test of whether the
recipe learns another requested pattern. No automatic 3,200-step fit, reward
change or additional layer unfreezing follows.

The [result record](experiments/idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026)
owns full metrics, costs and limitations. [Saved-array verification](verification.md#1600-update-result-audit--18-november-2026)
reproduces all parent results and preserves 216 run files without neural inference.
Current evidence remains limited to eight development prompts, one charge target
and a shared supervised initialization. New-property/prompt generalization,
external-reference novelty, biological function and an advantage over equally
adaptable non-GFN models remain unestablished. The
[benchmark/functional roadmap](../literature/IDR_BENCHMARK_ROADMAP_2026-11-17.md)
remains optional research context. Real experiments and GPUs remain user-run.

## IDR experiment history

The user completed the [frozen-ProGen IDR baseline](experiments/idr_baseline_2026-11-12.md#completed-results--12-november-2026)
in 2 min 58 s. Prompted continuations pass the predicted-disorder screen in
125/192 cases (65.1%), versus 55/192 (28.6%) unprompted. Isolated-tail checks
remain similar (67.2% versus 28.1%), and both arms pass the predeclared per-seed
engineering screen. No GFN was trained.

Prompting provides a useful starting prior, but results depend on context. RPB1's
100% pass rate accompanies motif repetition and only 19/24 unique tails. Excluding
it in a labeled post-hoc sensitivity check leaves 60.1% success and 100% unique
tails. V2 agrees in aggregate but is not independent verification. These are
predicted-disorder results, not experimental validation or proof of global novelty.

**Completed:** the [charge pilot and saved-artifact diagnosis](experiments/idr_charge_2026-11-12.md#completed-results--12-november-2026).
All three heads reached 400 steps in a total 74 min run. Prior and GFN each pass
charge on 20/384 outputs (5.2%); joint charge/disorder success is 10/384 versus
11/384. Selection gives 79 charge / 33 joint passes; swaps give 142 / 66.
Disorder is largely retained, but charge steering is not learned meaningfully.

The policy stays close to ProGen on sampled paths. Lower TB residual largely
reflects learned log-normalizer shifts in a post-hoc algebraic diagnostic; every
training update clips gradients. A finite prior-pool importance estimate indicates
headroom in the fixed target, but is not exact distribution verification. RPB1
repetition worsens, and removing it does not restore a GFN advantage.

**Diagnosis complete:** [checkpoint and tiny CPU controls](experiments/idr_charge_diagnosis_2026-11-12.md).
No missing-gradient/reward-sign/token-alignment bug was found. The trained head
barely responds to explicit block context under fixed synthetic features. The same
head/TB mechanism fits a tiny exact target with low-noise updates; eight-sample
updates are less accurate, and a tenfold rate increase performs worse on that toy.
These support weak/noisy policy learning as the leading explanation, not a proven
single cause or an optimal hyperparameter prescription.

**Completed:** the [real-state learning diagnostic](experiments/idr_learning_diagnostic_2026-11-12.md#completed-results--12-november-2026)
captured 768 continuations in 4 min 46 s. All final heads slightly push charge
negative in both halves instead of learning the requested opposing directions.
Explicit-context response is tiny; frozen-feature contributions to the hidden
preactivation have much larger norms. Gradient averaging and one-step head updates
show mixed results, so a larger batch is not an established fix. CPU replay verifies
the captures and reproduces the conclusions without ProGen or GPU execution.

**Completed:** the [direct-context comparison](experiments/idr_context_2026-11-12.md#completed-results--12-november-2026)
finished all six fits in 2 h 13 min 49 s. The added 100-parameter route learns the
right sign and increases explicit-context sensitivity, but its absolute effect
remains small. Charge passes are 17/384 versus standard 20/384; joint passes are
9/384 versus 11/384. No seed improves either count. Continuous score improves
slightly, without a generation success benefit. These small count differences
do not establish reliable harm. The standard arm exactly reproduces the original
pilot's final heads and metrics; this is matched development evidence, not an
independent replication. CPU replay reproduces all reported metrics exactly.

**Completed:** the [branch-rate follow-up](experiments/idr_branch_rate_2026-11-12.md#completed-results--verified-13-november-2026)
finished in 1 h 5 min 41 s. Higher-rate charge/joint passes are 23/13 out of 384,
versus lower-rate 17/9, original-head 20/11 and ProGen 20/10. Charge improves in
all three trained seeds; joint success improves in two and ties in one. Disorder
and diversity are roughly retained. Mean charge score improves on all eight
prompts, but the gain over ProGen is only three extra passes on either endpoint.
This is encouraging development evidence, not established broad guidance success.
CPU replay exactly reproduces results and all rate schedules/checkpoints verify.

**Completed:** [matched TB/SubTB charge training](experiments/idr_subtb_2026-11-13.md#completed-results-and-next-decision--13-november-2026)
finished in 4 h 52 min. Charge/joint passes are 127/66 for TB, 79/42 for SubTB and
65/28 for raw ProGen, each out of 1,536. TB wins both endpoints in every seed;
SubTB fails its predeclared practical screen. Disorder and diversity remain similar.
Saved hashes, schedules and charge/disorder-threshold calculations verify.

Both policies still steer weakly. A post-hoc importance estimate from the fresh
prior pool suggests approximately 24.6% charge / 11.3% joint success under the
fixed target, with substantial finite-pool limitations; this is not an exact
attainable rate. Generate-and-select reaches 17.6% / 8.4%, at four proposals per
output. These indicate useful examples and possible target-fitting headroom.

**Revised priority after the user's reward question:** defer replay implementation.
The [reward audit](experiments/idr_reward_audit_2026-11-13.md) finds failing sequences
with better smooth charge scores than passing ones in 21/24 prior-pool groups.
Disorder is absent from the objective. The current target estimate is only 24.6%
charge / 11.3% joint success, despite retaining an apparent fitting gap relative
to TB. Better learning alone does not guarantee mostly acceptable IDR outputs.

The [attached RL monograph reading](../literature/RL_THEORY_REWARD_AUDIT_2026-11-13.md)
clarifies this target/learning distinction and the prior-versus-property tradeoff.
A post-hoc beta/acceptance-bonus sensitivity analysis uses saved proposals only;
large tilts concentrate weights on very few samples and are not selected settings.
The user's follow-up authorized a fix. The separate `joint_acceptance_v1` scorer
now gives full credit to all joint passes, bounded partial credit to failures,
and no extra credit within tolerance. V3 disorder becomes part of this candidate
reward; V2 remains sensitivity, not biological verification. Existing commands
and completed run definitions are unchanged.

The saved-pool target estimate is 49.2% charge / 47.9% joint / 84.3% V3 disorder,
versus the old target's 24.6% / 11.3% / 56.4%. These are **post-hoc importance
estimates, not trained GFN results**. Median ESS is only 16.8 of 256, and two
prompt/seed pools have no observed joint pass. The smaller log-factor range is
promising for extreme reward signals, but rare-pass discovery remains unresolved.
The [reward record](experiments/idr_reward_audit_2026-11-13.md#adopted-candidate-joint_acceptance_v1)
owns the formulation, engineering defaults, caveats, evidence and bounded next step.

The user authorized training integration. The [new comparison](experiments/idr_acceptance_2026-11-13.md)
trains three fresh heads for 800 updates, reuses the old TB checkpoints, and samples
all methods afresh. It retains the frozen backbone, direct-context head, rates,
clipping and training budget. Per-sequence reward inputs expose rare-pass encounters;
final scoring reports each policy's residual under its own reward. The practical
screen requires at least a 5-point joint gain over newly sampled old-TB outputs,
two improved seeds and no more than a 5-point unique-tail loss.

The completed comparison fails that screen; see the current decision above and
[full results](experiments/idr_acceptance_2026-11-13.md#completed-results-and-next-decision--13-november-2026).
[Reproduction and recovery](runbook.md#joint-acceptance-tb--completed) remain available.
Replay would be a separate intervention; unfreezing remains out of scope.

## Retained starting point

- Standalone ProGen2 and TF-Bind-8 work remain in the original test workspace.
- Frozen ProGen2 + trajectory-balance guidance passed the 256-sequence synthetic
  test: final total variation distances 0.04090 / 0.04686 / 0.04472.
- The first biological run is preserved at runs/biological-first.
- Its task is 1O6X.pdb, a 72-residue reference: 950 scorer-training sequences,
  204 validation sequences and 204 held-out single-substitution candidates.

| Distribution | Expected predicted stability | Expected measured stability |
|---|---:|---:|
| Constrained ProGen | 3.89403 | 4.28005 |
| Exact predicted-reward target | 3.97291 | 4.26993 |
| GFlowNet, mean over three seeds | 3.97310 | 4.26975 |

Values are kcal/mol. These are exact probability-weighted expectations over the
finite candidate set, not new laboratory measurements. GFlowNet's close target
match does not make the predictor's ranking correct.

## Completed predictor/reward investigation

The user completed runs/predictor-reward-v1: 27 predictor conditions and 216 exact
target evaluations. [Protocol and results](experiments/predictor_reward_study_2026-11-03.md)
own the detailed comparisons and limitations.

- Original reward arithmetic reconstructs correctly; predictor errors matter.
- More training data improves both predictors. At 950 examples, ridge and MLP
  evaluation RMSE are 0.524 and 0.529 (MLP mean over three seeds).
- At beta 1, ridge's exact target raises measured expected stability from 4.28005
  to 4.31965; MLP targets average 4.26144. These are target calculations, not new
  GFN results.
- MLP has better validation RMSE, but worse errors on some high-probability
  variants. Global regression selection alone misses the design issue.
- Capped MLP targets modestly improve mean dG but reduce reference-level success.

## Completed ridge-reward GFlowNet rerun

The user completed runs/ridge-gfn-v1 in 48 min 6 s. All three final policies
closely match the ridge target and improve expected measured stability over
ProGen on the original 204 candidates:

| Distribution | Expected measured stability (kcal/mol) |
|---|---:|
| ProGen | 4.28005 |
| Historical MLP-reward GFN, mean | 4.26975 |
| Exact ridge target | 4.31965 |
| Ridge-reward GFN, mean | **4.31953** |

Final TVs to ridge target: 0.000412 / 0.000227 / 0.001367. Reference-level
success probability rises from 83.95% to 87.06%; expected distinct variants in
256 draws falls from 10.67 to 9.87. See [protocol and results](experiments/ridge_gfn_2026-11-09.md)
for predicted-versus-measured distinctions, training curves and limitations.

## Completed fixed reward-noise study

The user completed all 18 fits in 5 h 2 min. Every GFN closely matches its noisy
target (maximum final TV 0.000504). Noise changes target biological utility:
trained means range 4.29858–4.33340 at sigma 0.25 and 4.25404–4.34721 at sigma
0.50. One higher-noise realization falls below ProGen (4.28005), despite accurate
fitting. Some perturbations help by chance; three noise draws do not establish
noise tolerance or a useful noise-injection method.

See [protocol and completed results](experiments/reward_noise_2026-11-09.md).
Pinned inputs match and full CPU evaluation replay reproduces saved results.
The assistant ran no GPU work or training during analysis.

## Completed local split robustness

All three predefined splits completed in 3 min 21 s. Ridge and MLP targets both
improve expected measured stability over their own ProGen baseline on all three
splits. Mean gains are +0.01512 and +0.02416 kcal/mol respectively. MLP is better
on two splits and essentially tied on the third: the original ridge advantage
is not a general model-ranking result. No new GFlowNet was trained.

Split 44 barely improves (+0.0006 for either model); its prior puts 98.36% on one
variant, and both targets remain similarly concentrated. Absolute baseline values
vary with candidate support and must not be compared to the old 4.28005 baseline.
See the [completed protocol/results](experiments/split_robustness_2026-11-09.md)
for all splits, secondary metrics and limitations. CPU checkpoint/prior/target
reconstruction passed; completed run outputs remain unchanged.

## Completed mutation-position holdout study

All three grouped splits completed in 3 min 14 s. Both predictors improve target
utility on substitutions at positions absent from their training/validation
mutant sets. Ridge target gains are +0.78574 / +0.37963 / +1.36634 kcal/mol;
MLP gains are +0.63510 / +0.29585 / +0.32468. No GFN was trained.

These gains use substantially different candidate supports and weaker baseline
means (2.97034 / 3.94440 / 1.81102); they are not a direct improvement over the
random-split experiment. Prediction errors remain substantial. On split 45, even
the ridge target has only 24.82% probability of reaching reference stability.
See the [completed record](experiments/position_holdout_2026-11-09.md) for all
metrics, selection details and CPU artifact verification.

## Completed GFN follow-up — verified 10 November 2026

All nine fits completed in 2 h 30 min 29 s. Each closely matches its fixed ridge
target and realizes its stability gain. Across splits 43/44/45, expected measured
stability averages 3.75585 / 4.32386 / 3.17742, versus own-support ProGen baselines
2.97034 / 3.94440 / 1.81102. Maximum final TV is 0.001972 and maximum absolute
measured-target gap is 0.002265 kcal/mol. CPU replay reproduces saved evaluation
exactly; the [result record](experiments/position_gfn_2026-11-09.md#completed-results--10-november-2026)
owns full metrics and limitations.

The local sampler check is complete. Diversity follows the target rather than
uniformly increasing, and split 45 still reaches reference stability only about
25% of the time. These are finite, known-candidate results on one parent protein,
not broad generation or new biological measurements.

## Completed: multi-protein stability predictor benchmark

The [fixed protocol](experiments/multiprotein_2026-11-10.md) and
[user command](runbook.md#multi-protein-stability-predictors) define the completed run. CPU data
preparation retained 414 eligible parents, grouped them into 115 components, and
excluded 9 components matching known historical identities. From 106 remaining
components, a fixed identity-only rule selects 64 parents, each with 65 sequences:
44 training groups, 10 validation and 10 evaluation (2,860 / 650 / 650 sequences).
Maximum cross-partition parent-reference identity is 31.94% under the documented
alignment. This heuristic is not a guarantee of unseen biological families.

The user GPU command extracts frozen features and compares constant, ridge and
three MLP seeds. Evaluation reports pooled, per-parent and reference-only errors
and measured top-score selection lift over uniform candidates. There is no GFN
sampling prior, new guidance fitting or generation in this benchmark. All models
share fixed evaluation identities. Test labels remain unopened by fitting code;
CPU preparation used them only for validity, replicate averaging and copying.

All 47 CPU tests pass, including synthetic end-to-end execution and equivalent
primal/dual ridge solving. Real-input preflight passes. No real predictor fitting,
backbone execution or GPU work was run by the assistant. Historical exposure
inventory is incomplete by nature; this remains developmental evidence. Broad
generation still needs an explicit sequence space and independent assessment.

The user completed runs/multiprotein-v1 in **57.875 seconds** (46.231 seconds
loading/extracting frozen features). Test ridge RMSE is 1.202 versus constant
1.442; its within-parent top-10% selection lift averages +0.371 kcal/mol, positive
on 9/10 test parents. MLP test lifts range from -0.008 to +0.227, though seed 2
has lower test RMSE (1.083). These lifts compare against uniform candidates,
not ProGen sampling.

All four learned predictors have worse validation RMSE than constant (1.447)
and fail the reported validation gate. Ridge's validation pooled Spearman is
-0.118 despite positive within-parent ranking (mean 0.285). No model is promoted
to a reliable broad-generation reward. The follow-up diagnosis and regularization
protocol are described below. The test panel is now inspected
development data. [Full results and verification](experiments/multiprotein_2026-11-10.md#completed-results--10-november-2026)
include exact CPU replay and saved-checkpoint verification without retraining.

The [follow-up diagnosis](experiments/multiprotein_2026-11-10.md#follow-up-diagnosis--10-november-2026)
is complete. Ridge's parent-mean errors account for 64.4% of validation MSE;
its entire net disadvantage versus constant is in that component, partly offset
by better within-parent prediction. Removing any single validation parent does
not reverse the failure. Saved normalization uses training-only moments and
checkpoint/identity checks pass. Recommended next user-run experiment: five-fold
grouped validation within the 44 training parents to assess stronger ridge
regularization and constant baselines, with existing validation parents retained
as a separate transfer check. No new fits were performed during this diagnosis.

## Completed: regularization only, fixed protein coverage

The user requested regularization and training-set expansion as separate
experiments. [Regularization protocol](experiments/regularization_2026-11-10.md)
is implemented: compare penalties 1000 through 10000000 in five fixed group folds
within the existing 44 training proteins. Refit the CV-selected ridge and compare
with the original ridge and constant on the same ten validation proteins. Test
labels are not used. No additional proteins or backbone execution. Source caches
and checkpoints are pinned. [Run command](runbook.md#ridge-regularization-only).
The user completed runs/regularization-v1 in 2.003 seconds. CV selects penalty
1000000. Validation RMSE improves from 1.63899 to 1.43303, only slightly better
than constant (1.44731). Within-parent selection lift deteriorates from +0.29065
to -0.30086 kcal/mol; selected CV lift is also negative (-0.1248). Shrinkage reduces
parent-level errors but harms the reward's candidate ranking. Do not promote the
selected ridge to a generation reward on RMSE alone. Saved CV/validation metrics
and checkpoints pass CPU verification. [Results and next-step rationale](experiments/regularization_2026-11-10.md#completed-results--10-november-2026).

## Completed: more training proteins, fixed ridge rule

The user authorized the separate [coverage experiment](experiments/coverage_2026-11-10.md).
CPU preparation adds 42 eligible unused groups, with 65 sequences each, to the
original 44 training parents. Nested stages have 44/64/86 parents and
2,860/4,160/5,590 sequences. The ten validation and ten test parents/identities
remain unchanged. New groups and sequences are disjoint from both held-out sets.

Reuse original frozen features and extract only the 2,730 added training features.
Fit three ridge models using the original regularization per training example;
raw penalties scale as 1000 times stage rows divided by 2,860. No tuning or
stronger-regularization model is carried forward. Report all stages on the same
validation proteins, with both accuracy and measured selection lift; do not
evaluate test labels. A baseline reproduction check protects comparability.
The [user command](runbook.md#training-protein-coverage) completed in 32.321 seconds.
At 44/64/86 parents, validation selection lift is +0.29065/+0.37826/+0.31406
kcal/mol; RMSE is 1.63899/1.67489/1.58750. All three still trail their constant
prediction-error baselines (about 1.447). Baseline reproduction and all saved
predictions/metrics pass CPU checks. No real fitting or backbone execution was
performed by the assistant.

The selection gain is modest and non-monotonic, with much of the aggregate
improvement attributable to one validation protein. Do not select 64 as a general
optimum or claim broad-generation readiness. [Full results](experiments/coverage_2026-11-10.md#completed-results--10-november-2026)
record the parent-level sensitivity and verification. The initial follow-up
question was whether coverage benefits repeat across held-out groups. The subsequent
literature review prioritizes a controlled representation comparison while retaining
that robustness requirement; see the current decision above.

## Evaluation boundaries

The original labels have already been inspected. This study and future resplits
of these variants are development, not fresh confirmation. Full-data ridge seeds
use identical training data and do not provide independent replication. The
gains depend on support, which remains narrow; broad stable-protein generation is
not established. Retired outcomes remain inspected as well.

Verification: original inputs and feature-cache hashes match provenance; CPU
reconstruction reproduces all 216 expected measured dG values within 4.45e-15.
No GPU execution or retraining was performed during analysis. The saved ridge-GFN
checkpoints independently reproduce the final measured expectations on CPU.

## Retired work

The retired project archive was deleted at the user's request on 10 November
2026. Its files are no longer available for recovery from that location.
The [decision record](decisions.md#2026-11-03--return-to-the-first-biological-pilot)
explains the reset. Active experiment artifacts and the frozen identity inventory
remain intact; the prepared multi-protein benchmark does not depend on the archive.
