# Matched GFlowNet versus supervised IDR generation — 18 November 2026

**Status: implemented and CPU-verified; ready for the user to run. Real outcomes pending.**
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
