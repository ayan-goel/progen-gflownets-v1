# Replay-assisted TB and supervised control — 13 November 2026

**Stage: completed; results verified 14 November 2026.** All real
training, pretrained execution and GPU work remain user-run. The [alternative
comparison](../decisions.md#2026-11-13--compare-alternatives-before-committing-to-replay)
motivates this bounded test. No reward, backbone or head architecture change.

## Question and design

Does repeatedly learning from informative training examples improve fresh joint
charge/disorder success? Can the same head steer generation under a simpler
supervised objective if replay TB fails? The completed reward comparison found
rare successes and much worse residuals on successful outputs. It does not prove
that replay, sufficient capacity or an optimal reward have been established.

Use **seed 0**, selected before new outcomes, and three freshly initialized heads.
All inherit the direct-context head and the zero-correction initialization. ProGen
and its output projection stay frozen; each head initially recovers the same prior.

| Arm | Each of 800 updates | Purpose |
|---|---|---|
| On-policy TB | 16 fresh trajectories: two per prompt | Fresh control for the new batch budget |
| Replay TB | 8 fresh trajectories plus 4 stored successes and 4 stored near-misses | Test repeated exposure to informative examples |
| Supervised | 16 stored successful trajectories: two per prompt | Same-head steering capability under simpler training |

TB arms have **12,800 trajectory-example presentations each** and the same
optimizer-update budget. Fresh trajectories number 12,800 versus 6,400. Replay
additionally uses a bank derived from 6,400 historical training trajectories;
this source-data discovery cost is disclosed rather than treated as free. Data
exposure and wall time are not matched. The supervised control has 12,800 repeated
successful-example presentations, no fresh training trajectories and no log-Z fit.
It tests an easier objective, not exact sampling of the GFN target. Do not compare
its numerical loss with TB loss or interpret failure alone as a capacity proof.

All learned arms generate **128 new outputs per prompt**, 1,024 per arm. A fresh
ProGen arm produces another 1,024; total **4,096 raw outputs**. No selection,
filtering, edits or deduplication change these primary counts. The previous
four-proposal selection baseline remains historical, not a new matched control.
There is no new prior-pool target estimate or claim of exact distribution matching.

## Training bank and leakage controls

`configs/idr-replay.json` pins the completed acceptance run's generation manifest.
Read only its seed-0 `reward_inputs.jsonl` for training data. Original final
samples and other training seeds never enter the bank. Check source reward flags
and values against saved V3 scores before training; no predictor executes in
preflight. The original source files, checkpoints and configurations stay intact.

1. Deduplicate by prompt and tail, retaining the first trajectory's provenance.
2. Use a SHA-256 hash of split seed 81000 plus the tail to allocate about 20% to a
   diagnostic holdout. Identical tails across prompts get the same partition.
3. If a prompt has no training success, move its lowest-hash success tail globally
   to training. This declared repair preserves a usable supervised control; it
   can leave no positive holdout for a scarce prompt. No repair was needed here.
4. Per prompt/partition, retain up to 64 successes ordered by hash. For training,
   keep the best 32 distinct failures by the fixed property log factor; for
   holdout, keep the best eight. Break ties by the same hash. These are near-miss
   diagnostics, not representative samples of the natural distribution.
5. Training uses **152 successes + 256 failures = 408 bank rows**. Holdout contains
   **31 successes + 64 failures = 95 rows**. Total cache: **503 rows**. Source seed
   0 has 6,400 trajectories and 6,097 distinct prompt/tail pairs. DP00181 has one
   training success and one held-out success, so interpret its results cautiously.

Replay draws one bank row per prompt per update. Alternate success versus failure
by `(prompt index + update) mod 2`; with eight prompts, every update gets exactly
four of each and every prompt receives equal exposure to each stratum over two
updates. Draw uniformly with replacement within a stratum. Supervised training
draws only successes, two per prompt. Neither uses held-out rows for optimization.

**Use a fixed bank in this first pilot.** Fresh replay-arm trajectories contribute
one TB update each and are logged, but do not refresh the stored bank. This keeps
the informative dataset fixed and avoids mixing replay effectiveness with buffer
replacement choices. It tests seeded offline/online mixing, not an adaptive replay
algorithm. A success-aware bank is our design, not ranking by the full prior-times-
property reward. The supervised arm sees the same stored successful examples.

Online draws can coincidentally reproduce a held-out tail. Record these exposures
and exclude affected rows from both initial and final held-out fit summaries for
that arm. This avoids pretending an encountered sequence stayed unseen without
changing the online sampling distribution. Holdout selection and previous outcome
inspection make this development evidence, not a new held-out protein benchmark.

## Objective, architecture and numerical settings

The [acceptance-reward definition](idr_reward_audit_2026-11-13.md#definition-and-every-parameter)
is unchanged, including all charge bounds, V3 cutoff, acceptance ratio 20, failure
floor fraction 0.25 and effective beta 1. Both TB arms use

`mean[(log Z(c) + log q_theta(x|c) - log p0(x|c) - log w(x))^2]`.

`x` is the complete 64-residue generated tail; `c` is its supplied prefix;
`q_theta` is the autoregressive policy with trainable head parameters `theta`;
`p0` is the fixed canonical-residue ProGen prior; `w` is the unchanged property
multiplier; `Z(c)` is a learned prompt-specific normalizer. The mean is over the
16 trajectories in the batch. Replay changes training-example frequency, not this
terminal target. Current head probabilities are recomputed on every replay update;
old behavior probabilities never substitute for them. Positive softmax support
from fresh trajectories is retained, subject to numerical precision.

Supervised training minimizes the **mean negative log probability of the observed
next residue**, averaged over all tail positions and all 16 successful examples.
This is teacher forcing: each prediction receives the actual preceding residues.
There is no TB term, reward weighting, auxiliary mixture or trainable normalizer.
Only the same residual head is updated. Its training examples came from adaptive
historical policies; imitation is not claimed to fit the soft target distribution.

| Setting | Value and rationale |
|---|---|
| Frozen model / support | Inherited ProGen2-medium, float32, temperature 1, 20 canonical residues, no EOS |
| Prompts / lengths | Same eight contexts, 16 supplied residues + 64 generated |
| Head | Inherited 128-hidden-unit residual MLP plus bias-free 5-to-20 task branch |
| Initialization | Seed 0 for every fresh head; all corrections initially zero |
| Updates | 800 per arm; same prior budget, now one seed to screen the mechanism |
| Batch | 16 per arm; provides room for fresh and success/failure examples while preserving prompt balance |
| Adam head / task-branch rates | `1e-4` / `1e-3`, inherited; no new rate search |
| TB log-Z rate / initialization | `1e-2` / zero, inherited; absent from supervised optimizer |
| Rate decay | All active rates ×0.1 after update 600, inherited |
| Gradient clipping | Global active-parameter norm 1; inherited, with pre-clip norms logged |
| Reward scoring | Same metapredict 3.0.2 V3 on isolated tails, CPU, unrounded and detached |
| Cached feature batches | Eight sequences, bounding backbone memory during reconstruction |
| Training split / sampling RNG | Hash split 81000; online draws 82000; bank sampling 85000 |
| Evaluation RNG | 83000 shared across learned arms; prior 84000; independent of training and previous experiments |
| Checkpoints / logs | Every 100 / 10 updates, plus final; evaluate final only, no selected checkpoint |
| CPU threads | One, inherited |

Batch size, mixture, bank limits, split, new RNG streams and the progression rule
are engineering choices, not paper-optimal hyperparameters. Other optimization
settings are inherited to limit simultaneous changes. Different objectives and
batch shapes mean equal seeds do not imply identical training trajectories.

### Frozen feature reuse

Teacher-force saved complete sequences through frozen ProGen, omitting the last
residue from the input. The state at the prefix boundary predicts the first tail
residue; no future token enters its own prediction. Store float32 features, base
logits, actions, explicit context and prior probabilities on CPU. Transfer only
head minibatches to the GPU. The cache is roughly 193 MiB for these 503 sequences,
excluding checkpoints, logs and evaluation artifacts. Real reconstruction is
part of the user's GPU run and its reported cost.

Verify teacher-forced prior log probabilities against original incremental values
with absolute tolerance 0.003 plus relative tolerance `1e-5`, allowing float32
kernel-order differences while rejecting major token/prompt misalignment. This
is an engineering consistency tolerance, not a model-accuracy claim. The cache is
valid only while the backbone stays frozen. Its holdout tensors never enter a
training batch. No pretrained feature extraction ran during implementation.

## Evaluation and bounded decision

Primary: fresh raw joint charge/V3 success against the new on-policy TB control.
Also report charge alone, V3 disorder, V2 sensitivity, unique tails, repeated
trimers, sequence entropy, pairwise Hamming distance, prior likelihood, reference
similarity, every prompt's results, and pass/failure-specific TB residuals. V3 is
an optimized proxy; V2 is related sensitivity, not independent biological evidence.

Measure exact tail copying against each arm's **actually used** fresh/stored
training sequences, across prompts conservatively. Report nearest identity and
exact copying against the common seed training bank for all arms. Training-set
sizes differ, so own-training copying has different exposure opportunities.
No exact copy is not proof of global sequence novelty; inspect near-copy identity.

For each fitted head, save initial/final token NLL on fixed bank rows. TB arms also
save TB residuals, separated into success/failure and bank-train/holdout strata.
The on-policy arm's bank-train partition is reference data, not an optimizer input.
Null strata and fresh-training contamination are explicit. These selected rows
are diagnostics, not exact full-space target-error estimates.

**Predeclare the one-seed progression screen:** at least +5 percentage points raw
joint success over on-policy TB; no more than a 5-point drop in unique-tail fraction;
and at least +5 points in the unconditional rate of joint success **excluding exact
own-training copies**. The last check adds a memorization guard to the earlier
proposal; it is fixed before this run. Apply it separately to replay and supervised
outputs; a supervised pass is not GFN success. These are practical screens, not
significance tests or robust seed-level conclusions.

- Replay passes with acceptable per-prompt/diversity behavior: consider seed
  confirmation, not an automatic rollout to broader biological properties.
- Supervised improves fresh generation while replay does not: useful steering is
  expressible by this head; investigate TB learning or a later warm start. Exact
  target capacity is still unproven.
- Both only improve stored-sequence fit: prioritize generalization/coverage.
- Neither fits stored data: inspect optimizer/head limitations before choosing
  architecture changes. Failure does not establish that unfreezing is necessary.

Stop after the declared 800 updates and final evaluation. No automatic longer
run, rate sweep, reward change, buffer refresh, SubTB or ProGen unfreezing.

## Artifacts, verification and sources

[Commands](../runbook.md#replay-tb-and-supervised-control--completed) reproduce the
complete experiment and support CPU scoring recovery. Outputs include bank and
split provenance, frozen cache, per-arm logs/checkpoints/fresh reward inputs,
initial/final bank fits, raw outputs, `metrics.json`, `decision.json`, `fit_summary.json`,
`training_summary.json`, `costs.json`, and a concise `report.md`.

Hash captured artifacts before scoring. Recovery requires only that snapshot and
the pinned implementation/scorer, not the source run or original data directory.
No overwrite or training resume is implied. [Verification](../verification.md#replay-and-supervised-pilot--13-november-2026)
records random tiny-backbone CPU tests and real-input preflight. Fixture scores
are deterministic substitutes; they establish execution/correctness, not efficacy.

Motivation: [Jain et al., §3.2 and Algorithm 2](https://proceedings.mlr.press/v162/jain22a/jain22a.pdf),
[Shen et al., replay and its task-dependent results](https://proceedings.mlr.press/v202/shen23a/shen23a.pdf),
and [Hu et al., §3.3](https://proceedings.iclr.cc/paper_files/paper/2024/file/bc667ac84ef58f2b5022da97a465cbab-Paper-Conference.pdf).
The [review](../../literature/IDP_IDR_REVIEW_2026-11-12.md#12-alternatives-after-the-joint-acceptance-result--13-november-2026)
owns reading limits. This pilot adapts the mechanism; it is not a reproduction of
those papers or a promise that replay fixes the observed failure.

## Completed results — 14 November 2026

`runs/idr-replay-v1` completed all three fits and evaluation in **1 h 46 min 47 s**.
Replay gives a small improvement; the supervised control gives the clearest useful
steering. **Neither meets the full predeclared progression screen.** Preserve the
threshold rather than retrospectively declaring the supervised near-miss a pass.

Each arm has 1,024 fresh raw outputs, 128 for each of eight prompts.

| Method | Charge passes | Joint charge / V3 passes | V3 disorder | Unique tails | Exact own-training copies |
|---|---:|---:|---:|---:|---:|
| ProGen | 41 (4.00%) | 17 (1.66%) | 65.14% | 986 (96.29%) | N/A: no fitted head |
| On-policy TB | 57 (5.57%) | 40 (3.91%) | 69.24% | 980 (95.70%) | 52 (5.08%) |
| Replay TB | 81 (7.91%) | 49 (4.79%) | 71.78% | 983 (96.00%) | 47 (4.59%) |
| Supervised | 129 (12.60%) | 89 (8.69%) | 78.71% | 1,021 (99.71%) | 0 |

Replay adds **9 joint passes, +0.88 percentage points**, over on-policy TB. The
supervised control adds **49, +4.79 points**, narrowly below the required +5.
With 1,024 outputs, three additional supervised successes would cross that raw
count threshold; this observation does not waive it or justify sampling until it
passes. All successful outputs are distinct from that arm's exact training tails,
so excluding exact training copies leaves the same joint rates. Every arm has
zero exact copies of the common seed training bank.

V2 joint counts are 17 / 37 / 50 / 89 for prior / on-policy / replay / supervised,
respectively; the method ordering persists. V2 is a related-model sensitivity
check. Charge means under supervision move to +0.051 / −0.037 in the two blocks,
versus +0.015 / −0.018 for replay, still short of target centers +0.125 / −0.125.
Most supervised outputs still fail the joint screen. These are proxy-scored IDR
continuations, not experimentally validated proteins.

| Prompt | ProGen joint / 128 | On-policy TB | Replay TB | Supervised |
|---|---:|---:|---:|---:|
| DP01299 | 1 | 4 | 5 | 7 |
| DP00187 | 7 | 7 | 10 | 12 |
| DP00181 | 0 | 1 | 1 | 6 |
| DP00092 | 0 | 9 | 8 | 12 |
| DP00510 | 1 | 10 | 7 | 9 |
| DP01076 | 0 | 0 | 1 | 10 |
| DP01538 | 2 | 1 | 4 | 10 |
| DP01109 | 6 | 8 | 13 | 23 |

Supervised joint counts improve over on-policy TB on seven of eight prompts;
replay improves on five, ties one and declines on two. The effect is not confined
to a single prompt, but there is only one training seed. The old 3.13% acceptance
result used a different batch budget, seeds and evaluation samples; use this
fresh on-policy control for the replay comparison.

### What training tells us

On-policy TB encountered 352 successes in 12,800 fresh trajectories; 505/800
batches contained none. Replay used **3,200 stored success presentations**, plus
203 successes in 6,400 fresh trajectories. Every replay batch contained successes
and all 408 bank training examples were used. Therefore a complete lack of useful
examples no longer explains replay's weak steering. The supervised head consumed
12,800 successful-example presentations drawn from all 152 stored successes.

Replay improves held-out-success TB residual RMSE from on-policy's **3.770 to
2.518**, but the residual remains substantial. On fresh successful outputs the
corresponding RMSE is **3.803 versus 2.346**. Fresh failure RMSE worsens from
0.226 to 1.361, consistent with a changed fitting compromise.

A **post-hoc algebraic decomposition**, not a causal ablation, separates policy
and normalizer terms on the same 31 held-out successes. Mean log-policy/prior
ratio is 0.288 for on-policy TB and 0.376 for replay, a change of only 0.088 nats.
Mean learned log normalizer changes from −4.055 to −2.850. Thus most of the
improvement in their mean signed residual comes from the normalizer difference,
not a large likelihood increase for these successes. This does not imply that
learning the normalizer is wrong or identify a single optimizer fix.

Supervised mean token NLL on stored successes decreases from 2.5252 to 2.5060;
on held-out successes it decreases only from 2.4354 to 2.4338. Fresh generation
shows useful property steering, but this small held-out likelihood change limits
claims about generalizing the whole successful-sequence distribution. No diagnostic
holdout tail was encountered in fresh training for either TB arm.

The supervised head is **not a GFlowNet** and learns a different objective. It
also receives four times as many stored-success presentations as replay. Its
advantage supports the existence of useful steering with the frozen backbone and
this head; it does not isolate the loss as the sole cause, prove exact target
capacity, or establish that supervised imitation is universally better.

The supervised outputs have lower mean nearest-seed-bank identity (23.45%) than
replay (27.41%) and fewer repeated trimers (11.66% versus 15.69%). This does not
support exact-copy collapse. Similarity is assessed against a small known bank,
not all natural proteins, and DP00181 remains an unusual repetition-heavy context.

Fitting times are 40 min 25 s on-policy, 40 min 8 s replay and 5.3 s supervised
after a shared 6.5 s frozen-feature cache. Supervision reuses existing data and
cached features; these timings exclude historical discovery and final generation.
They are not equal-total-cost comparisons or end-to-end speedup claims.

### Next decision

Stop this replay-only intervention; its gain does not justify a replay sweep or
automatic seed confirmation. Retain the supervised model as a useful development
control without changing the failed progression-screen label.

The next candidate is **supervised initialization followed by TB with replay**,
compared against the unchanged supervised checkpoint and, after the
[alternatives reassessment](../decisions.md#2026-11-14--compare-alternatives-to-supervised-initialization),
a matched continued-supervision control. It would test whether TB
can preserve and improve an already useful policy, rather than learning all
steering from the initial prior. This was an anticipated branch of the protocol;
it remains a hypothesis, not a guaranteed improvement. The subsequently authorized
[15 November protocol](idr_warmstart_2026-11-15.md) now fixes the budget,
normalizer initialization and success/diversity rule; implementation is CPU-tested
and real results are pending.
If TB erases the supervised gains or adds no useful steering, reassess the
trajectory-level objective/parameterization rather than extending that run.
The existing head's useful supervised behavior gives no immediate reason to
unfreeze ProGen. Exact target matching and seed robustness remain open.

### Verification

The [saved-artifact audit](../verification/idr-replay-result-2026-11-14.json)
verifies 44 captured hashes, independently reconstructs all 4,096 evaluation and
19,200 fresh-training rewards, and checks 2,400 updates, schedules, stored IDs,
finite checkpoints and logged normalizers. Rebuilding metrics using only the
saved per-residue scores exactly reproduces all metrics, decisions, fit summaries
and report text. Bank partitions are disjoint and every initial policy equals the
prior on cached rows. The [analysis source](../verification/idr-replay-result-analysis-2026-11-14.py)
is retained. No model/scorer execution, fitting or GPU work was performed during
analysis; completed run outputs remain unchanged.
