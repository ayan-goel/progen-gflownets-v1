# Matched GFlowNet versus supervised IDR generation — 18 November 2026

**Status: completed by the user at `runs/idr-objective-v1`; analyzed 19 November 2026.**
Weighted supervision has higher joint success in both repeats. Its lead shrinks
after excluding training copies, but remains positive. GFN retains more predicted
disorder and overall distinct tails; this does not establish a GFN advantage in
successful candidate yield. See [completed results](#completed-results--19-november-2026).
The user selected this comparison after retaining the completed 1,600-step recipe.
The [previous result](idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026)
has mean joint success 32.62%, but its supervised control trained only the head.
That comparison cannot isolate the benefit of the GFlowNet objective when our
successful method also adapts ProGen's final transformer block.

## Question and controlled comparison

Does trajectory-balance training produce more successful, diverse IDR candidates
than reward-weighted supervised learning when both adapt the same parameters and
have access to the same scored examples?

Both start from the original supervised initialization and original ProGen block,
not the trained 1,600-step policy. Each has its own copied final transformer block
and identical guidance head. Original ProGen, lower blocks, embeddings, final
normalization and output projection remain fixed. Only GFN learns the per-prompt
normalizers needed by its objective. Evaluate plain ProGen as context, not as the
primary comparator. Existing runs and source modules remain unchanged.

We fit both objectives afresh. The training pool now contains GFN and supervised
discoveries instead of frozen-head and adapted-head GFN discoveries. Reusing an
old GFN checkpoint against a newly trained supervised model would not give the
same data-access comparison. Therefore old 32.62% is historical context, not the
expected or matched GFN endpoint for this study.

This is a **coupled shared-data objective comparison**, not two independently
costed pipelines. Both see discoveries from both policies. It answers the missing
objective/capacity question; it cannot establish how efficiently each method
would discover data alone.

## Objectives and weighting rationale

Keep the existing terminal weight:

\[
\log R(x\mid c)=\log p_0(x\mid c)+\ell(x,c),\qquad
\ell(x,c)=-2.5\sum_j v_j(x,c).
\]

Here `x` is the generated 64-residue tail, `c` the known 16-residue prefix,
`p0` the original frozen ProGen probability under the existing 20-residue
renormalization, `R` an unnormalized terminal weight, `ell` the log property
factor, and `v_j` each nonnegative outside-tolerance charge/disorder violation.
The existing [continuous-reward record](idr_continuous_2026-11-15.md) owns the
violation scales and acceptance criteria. All scores are detached terminal labels.

GFN keeps ordinary trajectory balance:

\[
L_{\mathrm{GFN}}=\operatorname{mean}_i
 [\log Z_\theta(c_i)+\log q_\theta(x_i\mid c_i)
 -\log p_0(x_i\mid c_i)-\ell_i]^2.
\]

`q_theta` is the trainable policy, `theta` its parameters, `Z_theta` the learned
positive normalizer for each prompt, and `i` indexes shared training examples.
The unique autoregressive path has backward probability one. This loss seeks
the prior-times-property distribution; success rate alone does not prove matching.

The supervised comparator uses **reward-weighted mean-token likelihood**:

\[
w_i=\frac{\exp(\ell_i)}{\sum_{j\in B_c}\exp(\ell_j)},\qquad
L_{\mathrm{SFT}}=-\frac{1}{|C|}\sum_{c\in C}\sum_{i\in B_c}
w_i\frac{\log q_\phi(x_i\mid c)}{T}.
\]

`C` is the set of prompts in the batch, `B_c` their shared trajectories,
`T=64` the generated tail length, `w_i` a detached within-prompt weight, and
`phi` the supervised policy's separate parameters. Stable softmax computes the
weights before microbatching. Each prompt gets equal total weight; a prompt with
easier successes cannot consume another prompt's entire update. Higher-scoring
examples receive more weight, and an all-failure batch still supplies signal.

This is a weighted supervised baseline, not unweighted success-only SFT, RL, or
an exact solver for the GFN target. Weighting the shared proposal by the property
factor does not correct its sampling distribution to `p0`. No importance weights
or extra prior-probability factor are silently added. The property factor supplies
a simple soft selection rule using the same scores as GFN, avoiding a tiny or
empty perfect-success-only set. Its normalization and strength are fixed before
execution. Diagnostics record weight concentration/effective sample size.

The [benchmark roadmap](../../literature/IDR_BENCHMARK_ROADMAP_2026-11-17.md#direct-benchmark-design--follow-up-clarification)
motivates matched capacity and data. [Hu et al., ICLR 2024](https://arxiv.org/html/2310.04363v2)
compare GFN language-model adaptation with supervised and reward-based methods.
Our specific weighting/normalization choices are engineering defaults, not that
paper's protein protocol or a claim that this is optimally tuned SFT.

## Fixed settings and data boundaries

Executable settings: [idr-objective.json](../../configs/idr-objective.json).
Inherit the current charge target (+0.125 then −0.125), charged fraction 0.25,
tolerances, metapredict V3 reward, temperature 1, float32 and all 20 residues.

| Setting | Choice and rationale |
|---|---|
| Repeats | Two; bounded first comparison matching the recent budget study |
| Training budget | 1,600 updates per objective, final checkpoint only |
| Initialization | Same original supervised head and original ProGen block in both repeats |
| Shared batch | 32: one trajectory per each of eight prompts from GFN, supervised, prior and original-bank replay |
| Refresh | Every four updates; both methods see the identical batch IDs before another collection |
| Replay | Original training partition only; alternate success/failure strata by prompt and round |
| Microbatch | Four, preserving full-batch loss/weight normalization |
| Optimizer | Adam, inherited betas (0.9,0.999), epsilon 1e-8, no weight decay |
| Initial rates | Head 1e-4, direct task matrix 1e-3, copied block 1e-5; GFN-only normalizers 1e-2 |
| Schedule | All rates ×0.1 after step 1,200; inherited 75% convention |
| Clipping | Separate norm 1 for head/normalizer and copied block; reject nonfinite gradients |
| New tuning allowance | None in this first bounded comparison; matched rates are not evidence either objective is optimally tuned |
| Evaluation | 128 raw tails per prompt, arm and repeat; 1,024 per arm/repeat; 6,144 total |
| Checkpoint storage | Atomic replacement of one latest checkpoint per arm during training; retain final only, no best-step selection |

| Repeat | Training streams | Replay stream | Evaluation stream |
|---|---|---|---|
| 1 | 823000 / 823001 / 823002 (GFN/SFT/prior) | 825000 | 833000 |
| 2 | 923000 / 923001 / 923002 (GFN/SFT/prior) | 925000 | 933000 |

Evaluation restarts the same fresh stream for each arm within a repeat. Both
training repeats remain reported. Shared initialization limits initialization
robustness claims. Each repeat collects 9,600 new training trajectories and makes
51,200 trajectory presentations to each objective. Equal steps/data access do
not imply equal computation; retain proposal generation, scoring, updates and
final generation costs separately. Common historical bank discovery/warm-up
costs are inherited preparation, not free independent-pipeline data.
GFN update timing includes rechecking the original-reference likelihood; the
supervised update uses that checked stored value for diagnostics only. This
additional forward pass is implementation audit overhead, not an unavoidable
cost of the TB objective, and must be labeled in cost comparisons.

No stored diagnostic-bank rows enter updates. Raw generation may independently
produce the same sequence; record that as training exposure. Final evaluation
outputs never enter either learner, replay or checkpoint selection. No previous
adapted-policy training history is treated as new model exposure: both new fits
start from the original initialization.

## Endpoints and stopping rule

Primary comparison: fresh GFN versus supervised generation on identical prompts,
lengths and acceptance rules. Report each repeat and prompt, raw joint
charge/V3-disorder success, charge and disorder separately, unique successful
tails, training copies, and diversity of successes. Noncopy success retains all
raw outputs in its denominator. Use the same conservative copy union of original
training-bank tails plus fresh training tails from **both new repeats** for all
arms. Exact noncopy is not evidence of novelty against external databases.

Report successful-sequence connected-component clusters under at least 80%
positionwise identity within each prompt, using the same fixed-length tails.
This is a declared descriptive diversity threshold; connected components may
contain endpoints below 80% identity through intermediate sequences. Cluster
counts and per-sample cost complement, rather than replace, raw success.

A provisional practical GFN advantage requires **each repeat** to improve raw and
noncopy joint success by at least five percentage points over supervised, while
losing no more than five points of uniqueness or predicted disorder. This is an
engineering screen, not a significance test. Report the full tradeoff even if it
fails; failure does not establish equivalence or universal SFT superiority. Do not
declare a new winning endpoint after seeing results. Final checkpoints only.

Stop after both repeats and final evaluation. No extra steps, new target, weight
sweep, additional unfreezing or external-method implementation follows
automatically. If GFN has a useful advantage, test one contrasting profile next.
If results are similar or favor SFT, determine whether any diversity/cost advantage
remains before making a GFN-superiority claim.

V3 is an optimized prediction; V2 sensitivity is not independent biological
validation. This remains one charge task and eight development prompts, not a
biological-function, unseen-prefix or general property-conditioning result.

## Execution and verification

The [runbook](../runbook.md) owns the user command. Source/input snapshots and
generation are sealed before final scoring so scoring recovery needs no ProGen
load or ancestor directories. Incomplete training is not resumed automatically.
Tiny CPU fixtures and read-only preflight are allowed; all real model/scorer
execution and GPU work remain user-run.

**Verification:** [19 focused CPU tests](../verification/idr-objective-tests-2026-11-18.txt)
pass, including both copied-block/head updates with an unchanged reference,
full-batch/microbatch objective agreement, matched-data and corruption checks,
complete tiny fitting/generation/evaluation, byte-identical scientific scoring
recovery without ancestor runs, and checkpoint preservation on simulated write
failure. Synthetic scorers and a tiny randomly initialized ProGen are the only
models executed. [Real input preflight](../verification/idr-objective-preflight-2026-11-18.json)
passes with CUDA uninitialized and no model/scorer execution. Ruff and
`git diff --check` pass. These are correctness checks, not scientific results.

## Completed results — 19 November 2026

Both paired repeats reached update 1,600 and completed final generation/scoring
in **3 h 41 min 43 s**. The saved [report](../../runs/idr-objective-v1/report.md),
[metrics](../../runs/idr-objective-v1/metrics.json) and
[decision](../../runs/idr-objective-v1/decision.json) preserve the original outcomes.
There are 1,024 raw outputs per arm/repeat, 6,144 total. The table reports the
fraction passing both charge and V3 predicted-disorder requirements:

| Method | Repeat 1 | Repeat 2 | Mean |
|---|---:|---:|---:|
| Plain ProGen | 1.56% (16) | 3.32% (34) | 2.44% |
| GFN | 30.08% (308) | 32.62% (334) | 31.35% |
| Reward-weighted supervised | **38.57% (395)** | **43.26% (443)** | **40.92%** |

GFN minus supervised joint success is −8.50 and −10.64 percentage points.
Both repeats fail the predeclared GFN advantage screen. This is an observed
supervised success advantage under this recipe, not proof of universal method
superiority. Both generators substantially improve on the plain prior.

### What happens after accounting for copies and diversity?

The following aggregates both repeats, with **2,048 raw outputs per method**.
Distinct counts are deduplicated across the two repeats; all rate denominators
remain the full raw output count.

| Outcome | GFN | Weighted supervised |
|---|---:|---:|
| Joint successes | 642 (31.35%) | 838 (40.92%) |
| Successful exact training copies | 0 | 143 |
| Joint successes excluding exact training copies | 642 (31.35%) | 695 (33.94%) |
| Distinct successful tails | 642 | 702 |
| Distinct successful noncopy tails | 642 | 695 |
| Successful noncopy clusters, fixed 80% identity rule | 642 | 687 |
| Predicted-disorder pass, all raw outputs | 91.60% | 84.42% |
| Distinct tails, all raw outputs | 2,021 (98.68%) | 1,912 (93.36%) |

Supervision's 196 extra raw successes include 143 copied-success draws, accounting
for 73.0% of the raw count difference. All its successful training copies occur
under the RPB1 prefix (DP00181). These are copies of the common experimental
training union, not an external novelty assessment. After excluding them,
supervision still produces **53 more distinct successful noncopy tails** and
**45 more successful noncopy clusters**. Per-repeat noncopy success is 31.05%
and 36.82% for supervision versus 30.08% and 32.62% for GFN.

The observed GFN quality/diversity benefit is real but narrower than a better
candidate generator: it preserves more predicted disorder and overall unique
outputs, while supervision yields more distinct passing candidates at this sample
budget. Do not turn the diversity endpoints into a replacement pass rule.
Within-sequence repeated-trimer fraction is actually lower for supervision
(0.0680 versus 0.1103); output duplication and internal motif repetition are
different properties. No claim of uniformly better GFN diversity is supported.

### Prompt dependence, objective interpretation and costs

Pooled supervised raw and noncopy joint counts exceed GFN on seven of eight
prompts; GFN leads on DP01299. RPB1 contributes much of the raw gap, but removing
training copies does not reverse the aggregate result. These prompt breakdowns
are descriptive analysis of the declared development panel; no prompt is removed
from the primary comparison.

The mean own-policy log ratio against original ProGen is 2.36 nats per generated
tail for GFN and 21.72 for supervision. This sampled diagnostic indicates much
larger policy drift under supervision. It is compatible with their different
objectives: GFN seeks a fixed prior-times-property target, while supervision
weights the changing shared proposal. It is not a biological-quality score or
proof that GFN accurately learned its target, and does not establish the causal
reason for the success difference.

The saved training audit finds concentrated supervised weights: effective sample
size averages 1.83 and 1.82 out of four examples per prompt, and replay receives
about 52% of total weight despite supplying 25% of rows. Both GFN parameter groups
hit the clipping threshold at every update; supervision clips no head updates and
about 9–11% of block updates. These are descriptive optimization differences,
not evidence that changing clipping or replay would improve the outcome. The
audit found no material protocol bug or diagnostic-bank holdout exposure.

Measured updates total approximately 1,897 seconds for GFN and 1,811 seconds
for supervision. GFN timing includes the documented original-prior likelihood
audit forwards. Final generation takes approximately 776 seconds per method
across both repeats. There is no demonstrated GFN speed advantage here. Shared
collection/scoring and inherited initialization costs prevent interpreting these
as separately measured end-to-end training pipelines.

The new GFN mean of 31.35% is close to the prior experiment's 32.62%, but this is
not a controlled gain/loss comparison: seeds, proposal mixture, evaluation draws
and the applicable training-copy union differ. The matched comparison is between
the two freshly trained methods in this run.

### Interpretation and next decision

We have a working GFN IDR generator and a stronger capacity-matched supervised
baseline. We **have not demonstrated a GFN advantage in successful noncopy-candidate
yield** on the current task. Keep both outcomes, the 1,600-update budget and the
predeclared failed screen. Do not automatically extend training or change the
score/threshold to rescue the comparison.

Keep weighted supervision in subsequent comparisons. A next bounded question,
if selected, is whether both recipes learn one contrasting charge profile with
the same evaluation of distinct successful noncopy candidates. That would test
transfer of the recipe across requests, not repair or erase this result. No new
fit, property or external baseline is implemented or authorized by this analysis.
Biological function, unseen-prefix control, external novelty and global target
accuracy remain unestablished.

### Saved-artifact verification

[Cached-array reconstruction](../verification/idr-objective-results-2026-11-19.json)
and its [script](../verification/idr-objective-results-2026-11-19.py) reproduce
the scientific output files without ProGen or neural disorder prediction.
The [independent count audit](../verification/idr-objective-independent-2026-11-19.json)
checks success, copy and diversity arithmetic; the
[protocol audit](../verification/idr-objective-protocol-2026-11-19.json) checks
source identity, shared training, schedules and final checkpoints. Audits preserve
completed run files and do not constitute independent biological validation.
