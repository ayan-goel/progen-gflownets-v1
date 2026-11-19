# Last-block GFlowNet training-budget comparison — 17 November 2026

## Question and rationale

Can a bounded increase in training budget improve the confirmed 20–23% joint
charge/predicted-disorder success, without changing the target or architecture?
The user authorized implementation after the
[saved-output diagnosis](idr_lastblock_confirmation_2026-11-17.md#post-hoc-headroom-analysis--17-november-2026).
Charge success is only 24.27% across the two confirmations, whereas predicted
disorder success is 91.75%. Two of three evolving training curves improved late;
one was mixed. This motivates testing additional learning, not assuming that
more updates will help or that the noisy importance estimates are an attainable
success rate.

**Status:** completed by the user; both repeats pass the predefined budget screen.
The [results below](#completed-results--18-november-2026) and
[saved-score audit](../verification.md#training-budget-result-audit--18-november-2026)
record the outcome. Earlier CPU fixtures/preflight established implementation
readiness; no real model or scorer was executed by the assistant.
The [runbook](../runbook.md#last-block-training-budget-comparison) owns commands.
The [original last-block protocol](idr_lastblock_2026-11-16.md) owns inherited
architecture, reward, data rules and optimizer rationale. No new literature claim
or biological result is introduced by this budget choice.

## Controlled changes and initialization

Run two paired fits of **800 updates per arm**, each starting from the original
supervised head and original ProGen weights. Match each completed confirmation's
training and replay seeds. Evaluate its preserved 400-update adapted checkpoint
afresh alongside the new models; never overwrite or continue that checkpoint.
The old checkpoints omit optimizer/RNG state, so loading their weights would not
be exact resumption.

| Repeat | Training seed | Replay-bank seed | Fresh evaluation seed |
|---|---:|---:|---:|
| `replicate-1` | 223000 | 225000 | 433000 |
| `replicate-2` | 323000 | 325000 | 533000 |

Retain the existing 75%-of-run decay convention: all rates decrease tenfold
**after update 600**, instead of after 300. This tests a larger training budget
with a stretched schedule; it changes both update count and time at the initial
rates. The new update-400 model is therefore **not** the original 400-update
control: schedules diverge at update 301 and subsequently change sampled training
trajectories. Matched seed settings do not make those later data identical.

Unchanged settings are ordinary TB, property penalty 2.5, the frozen original
ProGen reward reference, copied final block plus guidance head, eight development
prompts, 16-residue prefixes, 64-residue tails, all 20 canonical residues,
temperature one, float32 model parameters and existing charge/disorder acceptance.
Keep the head, direct task branch, normalizer and copied-block initial rates at
`1e-4`, `1e-3`, `1e-2` and `1e-5`, respectively. Retain Adam defaults, zero weight
decay, microbatch four and separate norm-one clipping for head/normalizers and
copied block. Other ProGen layers, `ln_f` and the LM output head remain frozen.

This is one predeclared budget change, not a sweep, new reward, optimizer restart,
additional layer unfreezing or change to GFlowNet training. The unchanged paired
trainer is reused through a new wrapper/configuration; completed run artifacts
and source modules remain protected by provenance checks.

## Data and cost

Keep both the frozen-head and adapted arms training together. The frozen arm
supplies part of the shared online data; omitting it would change the method.
Every four updates collect eight frozen-policy, eight adapted-policy and eight
plain-ProGen paths, and eight original-bank replay examples. Both arms consume
the identical 32 examples and targets in each update. Fresh successes are not
added to long-term replay. Original diagnostic rows remain excluded from replay
and normalization, with accidental later exposure tracked as before.

Each repeat has 200 refresh rounds, **4,800 fresh training paths**, 1,600 replay
draws at collection and **25,600 trajectory presentations per trained arm**.
Across both repeats that is 9,600 fresh paths, 3,200 replay draws and 102,400
presentations across the four fitted arms. This doubles training exposure per
arm; it is not a compute-matched comparison with the old models. Report training,
generation and scoring costs separately.

## Fresh final evaluation

Generate 128 unfiltered tails per prompt and arm, resetting the repeat's fresh
evaluation seed for each arm. All eight prompts, including RPB1, stay in the
primary result. Different policies can generate different sequences even with
matched RNG seeds.

| Arm | Role |
|---|---|
| Preserved 400-update last-block TB | Primary baseline from the matching confirmation repeat |
| New 800-update last-block TB | Candidate improved generator |
| New 800-update frozen-head TB | Matched-data architecture control and online data source |
| Original unchanged supervised head | Shared initialization control |
| Saved continued head-only supervision | Existing practical comparator; no new supervised fit |
| Plain ProGen | Original generator, with no guidance head |

This yields **1,024 outputs per arm**, 6,144 per repeat and **12,288 total**.
Use the existing V3 disorder assessment, V2 sensitivity check and charge criteria.
Report each repeat and prompt, raw joint success, charge/disorder rates, uniqueness,
repetition, copy counts and inherited target-fit diagnostics. V2 is related
sensitivity evidence, not independent biological validation.

For the primary copy-aware comparison, use one common set per repeat: the union
of original training tails, the old confirmation's fresh training tails and the
new 800-update run's fresh training tails. Apply it to **all six arms**. Noncopy
joint success counts accepted tails absent from that union, divided by every raw
output, including failures and copies. This prevents the new model's larger
training exposure from receiving a more favorable novelty definition. It measures
exact training-tail reuse, not novelty against all known proteins. Preserve the
inherited fit summaries under their original definitions separately; those are
not substitutes for this common-union comparison.

## Decision fixed before results

In **each repeat**, the final 800-update adapted model must:

1. Improve raw joint success by at least **five percentage points** over its
   freshly evaluated original 400-update adapted model.
2. Improve common-union noncopy joint success by at least **five points** over
   that same model.
3. Lose no more than **five points** in either predicted-disorder pass rate or
   unique-tail fraction against that baseline.

Require both repeats to pass every condition. These are practical engineering
thresholds inherited from earlier progression screens, not a significance test
or power calculation. Report all differences; a favorable average cannot conceal
a failed repeat. Only final update-800 checkpoints determine progression. Interim
checkpoints and training curves remain diagnostic, with no best-checkpoint choice.
The inherited progression screen for the new paired fit is a secondary diagnostic;
it cannot override the primary 800-versus-400 decision.

- If both pass, retain the larger-budget recipe for this task and then plan a
  separate property-control/generalization test.
- If either fails, retain both outcomes and stop this budget comparison at 800.
  Inspect charge, disorder, diversity and copy changes before choosing another
  intervention. A single property-penalty comparison remains a separate possible
  follow-up, requiring its own protocol and authorization.

No automatic extension, extra seeds, reward change, dropped prompt or layer
change follows either outcome. Better charge/disorder proxy success would not
establish exact reward-proportional sampling, experimental disorder, biological
function or superiority to a capacity-matched non-GFN adapted model.

## Reproducibility and recovery

Pin the completed confirmation generation manifest to
`df0320fe54f2053255de8d3076c95c8b632b49feca1d634c45b080e8965d244f`.
Record source/checkpoint hashes, configuration, software versions and actual RNG
settings. Protect existing output directories. Each repeat preserves the inherited
four-arm fit outputs and separate generation from old adapted and plain-ProGen
controls; the outer report owns the common comparison and decision.

Saved-sequence evaluation recovery requires complete generation artifacts and
writes a fresh folder. It does not resume incomplete training or regenerate
missing outputs. Real fitting, generation and disorder scoring remain user-run.
Tiny CPU fixtures and read-only input checks verify implementation only; they do
not supply scientific evidence for improved generation.

## Completed results — 18 November 2026

The user completed `runs/idr-lastblock-budget-v1` in **2 h 41 min 33 s**.
The [run report](../../runs/idr-lastblock-budget-v1/report.md),
[metrics](../../runs/idr-lastblock-budget-v1/metrics.json) and
[decision](../../runs/idr-lastblock-budget-v1/decision.json) retain all six arms,
both repeats and all prompts. Each percentage below has 1,024 raw outputs per arm
and repeat. The controls are freshly generated from the preserved original
400-update models; their differences from the earlier confirmation percentages
reflect new evaluation samples, not changed old checkpoints or acceptance rules.

| Policy | Repeat 1 joint success | Repeat 2 joint success | Equal-weight mean |
|---|---:|---:|---:|
| Plain ProGen | 2.73% (28) | 1.66% (17) | 2.20% |
| Original supervised initialization | 9.86% (101) | 9.08% (93) | 9.47% |
| Saved continued head-only supervision | 13.28% (136) | 12.70% (130) | 12.99% |
| Frozen-backbone TB, 800 updates | 12.50% (128) | 11.82% (121) | 12.16% |
| Preserved last-block TB, 400 updates | 19.92% (204) | 23.63% (242) | 21.78% |
| **Last-block TB, 800 updates** | **29.59% (303)** | **29.69% (304)** | **29.64%** |

### Primary decision and quality checks

| 400 → 800 comparison | Repeat 1 | Repeat 2 |
|---|---:|---:|
| Raw joint gain | +9.67 percentage points | +6.05 percentage points |
| Common-union noncopy joint gain | +9.67 points | +6.05 points |
| Charge pass | 22.36% → 33.50% | 26.66% → 31.74% |
| Predicted-disorder pass | 91.50% → 88.57% | 91.02% → 94.24% |
| Unique-tail fraction | 97.66% → 97.75% | 97.07% → 98.05% |
| Predefined budget screen | **PASS** | **PASS** |

Both repeats exceed the five-point raw/noncopy gain and remain within the
five-point disorder/uniqueness loss limits. The mean gain is **7.86 points**:
607 successes instead of 446 across 2,048 outputs, approximately **36.1% more
successes per raw output**. This is not a total-compute efficiency claim; training
exposure doubles. The mean predicted-disorder rate is 91.41%, and the mean
per-repeat unique fraction is 97.90%.

All 607 new successes are unique within their own repeat and none exactly matches
the common union of original, old-run or new-run training tails. There are still
27 and 23 copied outputs overall, all failures from the RPB1 prompt. The common
training unions contain 5,375 and 5,390 unique tails. These checks exclude exact
training-tail copying as the explanation for success; they do not establish
novelty against external protein databases.

### What improved and what remains limited

The prompt breakdown improves on seven of eight prompts and ties on one in repeat
1; all eight improve in repeat 2. RPB1 remains weak at 5/128 and 7/128 joint
successes. Gains therefore are not confined to one favorable prompt. Mean repeated-
trimer fractions decrease from 11.49% to 11.14%, and 12.43% to 11.44%, respectively.

Post-hoc failure inspection identifies charge as the main remaining limitation:
only 32.62% pass charge versus 91.41% predicted disorder. Insufficiently negative
second-half charge falls from 569 to 399 outputs and 514 to 379; these categories
overlap with other failures. Excessively negative second-half charge increases
from 33 to 71 and 36 to 67, so stronger steering is not uniformly more accurate.
These are explanatory analyses, not new progression criteria or evidence for an
unselected reward change. Related V2 joint success averages 29.44%, a sensitivity
check rather than independent biological validation.

Common diagnostic-holdout mean squared TB residual remains worse for adapted than
frozen TB: 11.77 versus 10.11, and 10.86 versus 9.46. All trained updates still
clip gradients. Better fresh generation therefore does not establish globally
accurate reward-proportional sampling or fully resolved optimization.

Actual logs preserve the first 300 adapted updates of the old fits, then diverge
after the old learning-rate decay. The new schedule retains the initial rates
through update 600 and decays at 601. Both new final checkpoints are step 800;
old controls exactly match the protected step-400 sources. The interpretation is
**a larger training budget with a stretched learning-rate schedule improves this
fixed development task**, not an isolated effect of duration.

Across both repeats, paired training took about 79 min 13 s and final generation
about 79 min 14 s, with remaining time spent on diagnostics, scoring and overhead.
The saved [cost records](../../runs/idr-lastblock-budget-v1/costs.json) and child
records distinguish these phases; this run includes substantially more evaluation
than a single trained policy's deployment.

### Decision

**Retain the 800-update last-block GFlowNet recipe for the current task**, following
the successful branch of the predeclared rule. This is a material improvement over
the confirmed 400-update recipe, while approximately 70% of raw outputs still fail
the joint target. Do not infer that another doubling must help.

The user asked to assess this result before choosing further work. No next
experiment is selected or implemented here. The earlier benchmark/functional
roadmap remains a set of possibilities. This result still concerns one charge
request, eight reused development prefixes and one supervised initialization;
it does not establish new-property control, biological function or superiority
over capacity-matched non-GFN adaptation.

The [audit](../verification/idr-lastblock-budget-results-2026-11-18.json) reproduces
the five parent result files byte-for-byte from saved arrays, with all 182 run
files unchanged and CUDA uninitialized. Independent saved-data arithmetic agrees
with the counts, copy unions and decision. Neural disorder scores and pretrained
likelihoods were not recomputed; this verifies the recorded analysis, not the
biological accuracy of those predictions.

## Post-hoc late-learning analysis — 18 November 2026

The user asked whether improvement continues near update 800 and whether a longer
run is justified. The [saved-log analysis](../verification/idr-lastblock-budget-learning-2026-11-18.json)
and [script](../verification/idr-lastblock-budget-learning-2026-11-18.py) count only
fresh trajectories from the evolving adapted policy. They exclude prior/frozen
paths, replay and repeated optimizer presentations. Input hashes are unchanged;
no model or scorer executes.

| Update window | Repeat 1 joint successes | Repeat 2 joint successes |
|---|---:|---:|
| 1–200 | 35/400 (8.75%) | 44/400 (11.00%) |
| 201–400 | 73/400 (18.25%) | 96/400 (24.00%) |
| 401–600 | 98/400 (24.50%) | 104/400 (26.00%) |
| 601–800 | 113/400 (28.25%) | 116/400 (29.00%) |

The final 200-update windows improve by 3.75 and 3.00 percentage points over the
preceding windows. However, the finer windows are mixed:

| Update window | Repeat 1 joint rate | Repeat 2 joint rate |
|---|---:|---:|
| 401–500 | 23.0% | 28.0% |
| 501–600 | 26.0% | 24.0% |
| 601–700 | 31.5% | 27.0% |
| 701–800 | 25.0% | 31.0% |

Each narrower window contains 200 draws: eight prompts per refresh every four
updates. Rates decrease tenfold at update 601. Samples are collected before their
associated updates (1, 5, ..., 797), so these are changing-policy training
diagnostics, not large independent evaluations of the step-600/700 checkpoints.
They should not replace the 1,024-output final evaluation. Mean violation is nearly
unchanged between the final two windows (1.191→1.197 and 1.096→1.095). Mixed-batch
TB loss falls over the broader windows (7.42→6.15 and 6.89→5.69), but the changing
batches and objective prevent interpreting this as a direct success-rate forecast.

**Interpretation:** gains are slowing and the final local trend is uncertain.
Neither a clear plateau nor steady improvement through step 800 is established.
The material 400-versus-800 gain and modest positive broad late trends justify
considering **one bounded 1,600-versus-800 comparison**, with no promise of another
six-to-ten-point gain. Doubling again is a simple engineering choice consistent
with the preceding budget test, not an optimized or literature-derived duration.

If authorized, propose fresh fits from the original initialization, preserving the
800-update models as fresh-evaluation controls. Retaining the 75% decay convention
would move decay from 600 to 1,200 and again test budget plus schedule. Use two
paired repeats and the same five-point raw/common-noncopy improvement and quality
checks, with a common training-copy union. Do not automatically extend beyond
1,600 if the gain fails those checks. A separate protocol must specify the exact
seeds, budgets and recovery rules before implementation; the current run and
800-update default remain unchanged. This is a recommendation, not an authorized
or implemented next experiment.
