# Current project status

**12 November 2026 — Direct charge-context comparison ready for the user GPU run.**

[Specification](../spec.md) · [Pilot result](experiments/biological_first_results_2026-10-15.md)
· [Runbook](runbook.md)

## Current decision

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

**Ready:** the user authorized the [direct-context comparison](experiments/idr_context_2026-11-12.md).
A zero-initialized, bias-free 5-to-20 linear map adds the existing task context
directly to residue logits (100 new parameters). Both arms train from scratch
with identical shared initialization, seeds 0/1/2, the original reward/TB loss,
400 updates and eight trajectories per update. No learning-rate, batch-size or
backbone changes are combined with this intervention.

The runner uses one shared prior pool and evaluates 384 raw outputs per head,
including charge/disorder success, diversity and block-specific token corrections.
Full comparison: six fits and 21,504 ProGen continuations including evaluation.
Run `scripts/run.sh idr-context --output runs/idr-context-v1 --device cuda` after
choosing CUDA_VISIBLE_DEVICES and activating `.venv`; evaluation is automatic.
All 76 CPU tests and real-input preflight pass. No real comparison outcomes exist yet.
The assistant has not executed pretrained ProGen or used a GPU. The protocol owns
all settings, interpretation limits, provenance and CPU-only evaluation recovery.

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
