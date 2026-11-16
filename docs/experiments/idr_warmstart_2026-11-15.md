# Supervised initialization before GFlowNet training

**Stage:** completed by the user; results and CPU artifact verification recorded
below on 15 November 2026. The practical progression screen fails.
[Run command](../runbook.md#supervised-start-comparison) · [Configuration](../../configs/idr-warmstart.json)

## Question and rationale

Does TB with replay improve an already useful guided policy more than simply
continuing supervised learning? The [previous comparison](idr_replay_2026-11-13.md#completed-results--14-november-2026)
yielded 3.91% joint success for on-policy TB, 4.79% for replay and 8.69% for
supervision. Both interventions missed their predeclared progression screen.
Supervision nevertheless showed useful steering without exact training copies.
This motivates a bounded initialization test, not a prediction of a large gain.

The [alternatives decision](../decisions.md#2026-11-14--compare-alternatives-to-supervised-initialization)
compares prefix guidance, weighted likelihood, normalizer-free training and backbone
adaptation. The [targeted review](../../literature/IDP_IDR_REVIEW_2026-11-12.md#13-alternatives-after-replay-and-supervised-control--14-november-2026)
owns primary sources and reading limits. The warm-start hypothesis, numerical
budget and progression thresholds are our engineering choices, not prescribed by
those papers. No new literature claims or reward redesign are introduced.

## Arms and fixed settings

All arms start from `runs/idr-replay-v1/supervised/step-800.pt`.
The source generation manifest is SHA-256 pinned in the new configuration; it
captures the checkpoint, feature cache, bank, training logs and source protocol.

| Arm | Additional learning | Examples per update |
|---|---|---:|
| `unchanged` | None; preserve the starting head | 0 |
| `continued_supervised` | Mean token negative log likelihood on stored successes | 16 successes |
| `warm_tb` | Unchanged joint-acceptance TB with fixed replay | 8 fresh + 4 stored successes + 4 stored near-misses |

ProGen2-medium stays frozen. The direct-context head, canonical-residue support,
full-support sampling at temperature 1, eight prompts, 16-residue prefixes,
64-residue tails and `joint_acceptance_v1` reward remain unchanged. No classifier,
new reward, SubTB, buffer refresh, backbone adaptation or sequence selection is
added. The [previous protocol](idr_replay_2026-11-13.md) owns inherited details.

Use the same 408 training bank rows (152 successes, 256 near-misses) and 95
fixed diagnostic rows. Reuse the pinned float32 frozen-feature cache; do not
recompute it or reselect examples. The original tail-grouped split stays fixed.
New online trajectories contribute to their current TB update but are not added
to replay. Diagnostic rows do not enter either optimizer or normalizer initialization.

## Budgets, optimizer and random streams

| Choice | Value | Rationale |
|---|---|---|
| Training seed | Original seed-0 head | One bounded development comparison, not robustness confirmation |
| Updates per continuation | 400 | Half the preceding 800-step budget; enough to assess continuation without another full sweep |
| Examples per update | 16 | Preserve replay composition and match both continuation budgets |
| Optimizer state | Fresh Adam in both branches | Do not carry supervised-loss moments into TB; same restart treatment for the control |
| Original head / direct branch rates | `1e-4` / `1e-3` | Restart the tested source schedule in both arms, including the same branch multiplier |
| TB log-normalizer rate | `1e-2` | Inherit the previous TB rate; only TB has this parameter |
| Decay | All rates ×0.1 after update 300 | Preserve source 75% decay point on the shorter budget |
| Gradient clipping | Norm 1 | Inherited; TB clips head plus normalizer, supervision clips head only |
| Checkpoints / logging | Every 100 / 10 updates | Inherited visibility; use the final step-400 head, no best-checkpoint selection |
| Fresh training RNG | 92000 | Distinct from completed runs |
| Stored-example RNG | 95000 in both branches | Reproducible sampling; strata and exposure still differ |
| Evaluation RNG | 93000, reset for each arm | Fresh common random stream; trajectories can diverge after decisions differ |
| Evaluation count | 128 per prompt per arm | 1,024 raw continuations each, 3,072 total; preserve previous per-arm count |

Both continuations receive 6,400 additional example presentations. TB generates
3,200 fresh training trajectories and sees 1,600 stored-success presentations;
supervision sees 6,400 stored-success presentations. Thus update/example budgets
match, but successful-example exposure, unique data, discovery and wall time do
not. This is not a pure loss-only comparison. The original supervised data
collection/training is a shared sunk cost, retained in `source_costs.json`.

Resetting the optimizer and restarting rates is intentional, not an exact resume
of the source optimizer. Neither arm gets a learning-rate search. Evaluation RNG
is separate from both training streams. No prior-only or generate-and-select arm
is rerun: the question is the added value over the supervised starting policy.

## TB normalizer initialization

The supervised checkpoint has no log normalizer. Initialize it on each prompt's
**training bank only**, giving successes and failures equal total weight:

\[
 z_c^{(0)}=\tfrac12\operatorname{mean}_{x\in B_c^+}
 [\log p_0(x\mid c)+\log w(x)-\log q_{\mathrm{start}}(x\mid c)]
 +\tfrac12\operatorname{mean}_{x\in B_c^-}
 [\log p_0(x\mid c)+\log w(x)-\log q_{\mathrm{start}}(x\mid c)].
\]

Here `c` is a fixed prefix/prompt; `x` is a complete generated tail; `B_c^+` and
`B_c^-` are its successful and failed **training** bank rows; `p_0` is the frozen
canonical ProGen sampling probability; `w` is the unchanged acceptance property
factor; `q_start` is the starting supervised policy; and `z_c^(0)` is the initial
value of the trainable log-normalizer. All logs are natural.

This is the least-squares constant for the equal-stratum replay residual, avoiding
an arbitrary zero offset on a policy already different from the prior. It is
**not** an estimate of the true target partition function, and it does not center
the full fresh-plus-replay mixture. Log normalizers remain trainable. Store the
pass/fail offsets in `warm_tb/initialization.json`; no head updates occur during
this calculation. Supervision never optimizes a normalizer.

TB still minimizes the mean square of `z_c + log q_theta(x|c) - log p_0(x|c) - log w(x)`.
`q_theta` is the current guided policy, `theta` its trainable head parameters, and
`z_c` the current trainable log-normalizer. The desired terminal distribution
remains proportional to `p_0 * w`; the prior enters once. Supervision instead
minimizes the example's negative log policy probability divided by its 64 residues.

## Evaluation and stopping rule

Generate all three arms afresh and use raw, unfiltered outputs. Do not compare a
new TB percentage directly with the old 8.69% sample estimate. Primary outcome is
joint charge/V3 predicted-disorder success. Preserve charge-only, V3, V2 sensitivity,
unique-tail fraction, repeated trimers and per-prompt outcomes.

**Predeclared development screen:** warm TB must improve joint success by at least
**5 percentage points over each of the two controls**, also gain at least 5 points
in joint success excluding exact training copies over each control, and lose no
more than 5 points of uniqueness versus either. This extends the earlier practical
threshold to the new question. It is not a significance test or exact-target claim.
Most importantly, merely recovering the starting policy's performance is not a win.

Exact-copy checks include the original supervised training history for **all** arms
plus each continuation's actually consumed data. Noncopy success uses all outputs
as denominator; copying cannot improve it by removing samples. Common-bank nearest
Hamming identity checks near copies, not novelty against all natural proteins.

Record initial/final positive and negative bank likelihoods and TB residuals.
Also report within-prompt standard deviation of `log q - log p_0 - log w` on the
fixed bank: it measures relative-target mismatch without a learned normalizer.
Exclude holdout tails encountered by either continuation from **all** arms/stages;
report missing/single-row groups honestly. A small discrepancy on this selected
bank cannot establish global coverage, correct total mass or exact target matching.
V3 is optimized and V2 is a related predictor, not biological validation.

If TB passes, consider seed confirmation. If it erases the supervised gain or adds
no practical benefit over continued supervision, stop this warm-start route and
reconsider prefix guidance. Inspect target-fit evidence separately, since pass
rate and matching the soft target are different goals. Do not automatically extend
training, tune the reward or launch another method.

## Artifacts and verification

`initial.pt`, `initial_training.jsonl`, copied bank/cache and source manifest preserve
history. Each trained arm stores checkpoints, initial/final bank fits, initialization,
update logs and fresh scored training tails. `generation.json` captures inputs and
outputs before final scoring. `metrics.json`, `target_shape.json`, `decision.json`
and `report.md` are reproducible from this snapshot without the original source
run or reference-data directory. Failed final scoring can be recovered separately;
interrupted training has no automatic resume. Existing output directories are refused.

CPU-only tests exercise a random tiny backbone and deterministic scorer; no
pretrained model or real scorer runs. The [verification record](../verification.md#supervised-start-comparison--15-november-2026)
links test and preflight evidence. These establish execution and consistency only.

## Completed results — 15 November 2026

`runs/idr-warmstart-v1` completed in **39 min 9 s**. Both continuations reached
400 updates; all three arms produced 1,024 fresh raw outputs (128 per prompt).
The newly evaluated unchanged head is the comparator, not its previous 8.69%
sample estimate. The reward and frozen ProGen were unchanged throughout.

| Arm | Charge passes | Joint charge/V3 passes | V3 disorder passes | Unique tails | Exact own-training copies |
|---|---:|---:|---:|---:|---:|
| Unchanged supervised head | 133 (12.99%) | 96 (9.38%) | 788 (76.95%) | 1,023 (99.90%) | 0 |
| Continue supervision | 228 (22.27%) | 165 (16.11%) | 812 (79.30%) | 1,024 (100%) | 0 |
| Switch to TB with replay | 117 (11.43%) | 72 (7.03%) | 712 (69.53%) | 985 (96.19%) | 40 (3.91%) |

TB loses **2.34 percentage points against the starting head and 9.08 points
against continued supervision**. It fails both required +5-point comparisons.
Uniqueness loss stays within the allowed five points, but this cannot rescue
failed success criteria. All TB training copies fail the joint screen, so removing
copies leaves every arm's joint success unchanged. Zero exact copies does not
establish novelty against all known proteins.

Continued supervision improves joint success on all eight prompts, while TB
decreases it on seven and increases it by one sample on the eighth. Counts below
are out of 128 for each prompt/arm; this is one trained seed, not eight independent
training replications.

| Prompt | Unchanged | Continue supervision | Warm TB |
|---|---:|---:|---:|
| DP01299 | 10 | 21 | 6 |
| DP00187 | 22 | 28 | 16 |
| DP00181 | 7 | 22 | 0 |
| DP00092 | 14 | 17 | 11 |
| DP00510 | 10 | 18 | 8 |
| DP01076 | 8 | 13 | 6 |
| DP01538 | 4 | 10 | 5 |
| DP01109 | 21 | 36 | 20 |

V2 sensitivity preserves the ordering: 9.57%, 15.43%, 6.84% joint success,
respectively. V2 is related to the optimized V3 predictor, not independent
biological verification. Repeated-trimer fractions are 0.116, 0.084 and 0.158.

### TB learns something, but does not improve the generator

The predeclared normalizer-independent diagnostic improves under TB. Mean
within-prompt centered log-ratio error on the fixed held-out bank changes from
**1.973 to 1.370**, improving on seven of eight prompts. Its training-bank value
changes from 2.254 to 1.278. Continued supervision instead increases held-out
error to 2.368 while improving fresh success.

This is evidence of better **relative target fit on these selected examples**,
not merely a fitted normalizer shift. It does not establish global distribution
matching, correct total success mass, or coverage. There are only 95 diagnostic
rows (31 passes, 64 failures), none encountered in fresh continuation training.
The target contains the ProGen prior and nonzero failure weights; maximizing
pass frequency and fitting that target are different objectives. These results
therefore do not support saying that TB learned nothing, or that the reward has
been proved wrong. They do show that this TB continuation is practically worse
for the desired generation criterion.

TB sees 1,600 stored-success presentations plus 225 successes among 3,200 fresh
training trajectories; every update has successes. Continued supervision sees
6,400 stored-success presentations. Both have 6,400 total presentations, but this
is not a pure loss-only or compute-matched comparison. TB clips gradients on all
updates; supervision clips on none. Clipping alone does not establish causality.
Cached supervised continuation takes 2.65 s versus 1,189.47 s for TB, which also
generates and scores new sequences. Cache/data-collection costs are inherited;
these are incremental times, not end-to-end method cost comparisons.

### Post-hoc training reward inspection

Following the [pre-result research proposal](../../literature/RL_REWARD_DESIGN_2026-11-15.md),
recomputed reward values from saved training sequences and V3 arrays, without
running a predictor. This analysis uses training data only, but is post-hoc and
does not add a new predeclared success criterion.

- Among **2,975 fresh failures**, median property weight is **0.01490**, close to
  the theoretical floor of 0.0125. **28.4%** are within 10% above that floor and
  **57.5%** within 25% above it. Passing still gives weight 1.
- Among the **256 stored training near misses**, the median is **0.03524**;
  none lie within 25% of the floor. Replay therefore supplies more differentiated
  failures as well as passes: compressed fresh rewards are not a complete account.
- The thresholds are descriptive engineering choices. These selected and
  changing-policy samples cannot estimate target success mass as if drawn from
  the prior, and the percentages are not a causal test of reward saturation.

The [reproducible analysis](../verification/idr-warmstart-result-analysis-2026-11-15.py)
and [audit output](../verification/idr-warmstart-result-audit-2026-11-15.json)
record exact arithmetic, failure categories and source/manifest hashes. Captured
artifact and implementation hashes verify; six evaluation JSON files reconstruct
exactly on CPU from cached scores. This validates consistency, not scorer biology.
Completed run outputs remain unchanged.

### Next decision

**End this warm-start route; do not extend it or launch a seed confirmation.**
Retain the continued-supervision head as the strongest observed generation
baseline. It is success imitation, not a successful GFlowNet or proof of broad
IDR design. One seed, development prompts and predicted disorder limit inference.

The next reward question is now concrete: does the floor/jump tradeoff impede
useful learning? Before another GPU experiment, assess one continuous
outside-tolerance candidate against the current reward on designated training
proposals, checking implied target mass, per-prompt coverage and weight
concentration. Do not use the selected replay bank or changing-policy fresh
samples as unbiased prior samples. If the candidate retains useful target
preference without concentrating on a few sequences, specify one matched
reward-only training comparison from the same initialization. No penalty strength
or new trainer is selected here. Otherwise prioritize the review's prefix-flow
direction. Keep reward changes separate from local-credit changes so the next
result is interpretable; neither is a guaranteed fix.
