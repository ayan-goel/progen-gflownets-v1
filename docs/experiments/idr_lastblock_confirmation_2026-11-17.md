# Last-block GFlowNet confirmation — 17 November 2026

## Question and evidence

Does the useful generation gain from adapting ProGen's final block repeat under
fresh training randomness, and how does the resulting generator compare with
fresh plain ProGen under exactly the same evaluation rules?

The [completed pilot](idr_lastblock_2026-11-16.md#completed-results--16-november-2026)
reached 20.41% joint charge/predicted-disorder success, versus 12.40% for matched
frozen-head TB and 13.87% for saved continued supervision. This is a material
observed gain worth confirming. It remains one training run on eight development
prompts; common holdout target fit was worse than frozen TB and RPB1 still repeated
heavily. Confirmation reduces uncertainty about reproducibility; it is not intended
to improve the method or remove those limitations.

The pilot's 8.79% unchanged arm already contained a supervised guidance head.
Historical plain-ProGen results suggest a much lower success rate, but those draws
were not a fresh final-evaluation control. Add that missing control now. No new
literature search or new mechanism is required for this question; the local pilot
and its [comparison clarification](idr_lastblock_2026-11-16.md#plain-progen-comparison--post-hoc-clarification)
are the motivating evidence. The original protocol retains literature supporting
GFN policy adaptation.

**Status:** completed by the user; both fresh repeats pass all predefined screens.
The [results below](#completed-results--17-november-2026) preserve both outcomes.
Real fitting, generation and scoring remain user-run. The
[runbook](../runbook.md#last-block-confirmation--completed) owns reproduction and
saved-sequence recovery commands.

## Fixed experiment and independent randomness

Run two new paired fits, sequentially. Every fit starts from the **original
supervised head and original ProGen weights**, with fresh optimizer states and
the same training-only normalizer initialization. Do not continue the adapted
pilot checkpoint. The completed pilot is a historical reference, not one of the
two confirmation repeats and not part of their aggregate.

Pin the completed `runs/idr-lastblock-v1` generation manifest to
`00c7be30752158be58d57320cfe70e770558b5ff7e2ce47ceeb18304174240f2`.
Derive each fit's configuration from its saved `config.json`. Change only these
three RNG settings:

| Repeat | Training seed | Replay-bank seed | Final-evaluation seed |
|---|---:|---:|---:|
| `replicate-1` | 223000 | 225000 | 233000 |
| `replicate-2` | 323000 | 325000 | 333000 |

The three online training sources use offsets 0, 1 and 2 from each training seed,
as in the pilot. Streams are distinct from the pilot and from each other. Reset
the evaluation stream for each arm within a repeat, including plain ProGen;
policies still generate different sequences. Final evaluation draws are generated
after fitting and never consumed by training; accidental exact matches to training
tails are counted as copies. The inherited per-row `seed: 0` is a local legacy label;
repeat identity and the explicit RNG settings above identify the independent runs.
These are different stochastic fits of the same initialization, not different
supervised initializations or different prompts.

The [original protocol](idr_lastblock_2026-11-16.md#fixed-optimization-budget-and-rationale)
owns all inherited numerical settings and their rationale. In particular:

- Ordinary TB and the same continuous reward, with penalty 2.5 against immutable
  original-ProGen sequence probabilities. No reward, support or scorer changes.
- 400 updates per trained arm; refresh every four updates. Each shared batch has
  eight frozen-policy, eight adapted-policy, eight prior and eight replay paths.
  Both trained arms consume identical examples and target values.
- Same head, copied final block, optimizer, learning rates, clipping and decay.
  Only the copied final block and GFN head/normalizers adapt; the reference,
  earlier blocks, `ln_f` and LM output head remain frozen.
- Same eight 16-residue prefixes, 64-residue generated tails, 20-residue canonical
  support, temperature one and float32 model parameters.
- Final update-400 checkpoints only. No best-checkpoint selection, additional
  training, automatic tuning or replacement of an unfavorable repeat.

Reusing the existing trainer preserves this comparison and the source provenance
of completed runs. New orchestration and baseline evaluation live separately;
completed run files and pinned training modules are not edited.

## Five final-evaluation arms

| Arm | What is trained in each repeat? | Purpose |
|---|---|---|
| Plain ProGen (`prior`) | Nothing; no guidance head | Measure improvement over the original generator |
| Unchanged | Nothing; original supervised head | Measure improvement beyond the shared initialization |
| Continued supervision | Nothing new; same saved head-only supervised continuation | Retain the existing practical-performance comparator |
| Frozen-head TB | GFN head and prompt normalizers | Isolate the gain from final-block adaptation under shared data |
| Last-block TB | GFN head, prompt normalizers and copied final transformer block | Candidate generator |

Generate 128 raw outputs per prompt and arm: 1,024 per arm, 5,120 per repeat and
10,240 across both repeats. Plain ProGen is generated afresh for final evaluation;
its training-pool samples cannot substitute for this arm. Both trained arms still
collect 2,400 fresh training paths per repeat, with 800 replay draws at collection
and 12,800 trajectory presentations per arm. Adding the plain baseline changes
final evaluation cost, not training exposure.

Report each repeat separately, every prompt and all five arms. Aggregate summaries
show the two-repeat mean and range, without treating 10,240 outputs as 10,240
independent training runs. Saved controls and the supervised initialization are
shared across repeats; this study tests subsequent training/data-collection
randomness within a fixed development setting.

Primary metrics are raw joint charge/V3 predicted-disorder success and noncopy
joint success. For the latter, every arm uses the same repeat-specific union of
original training tails and new training draws. Copies stay in the denominator
and cannot count as noncopy successes. Also retain separate charge/disorder rates,
V2 sensitivity, uniqueness, repetition, copy counts, near-bank identity, per-prompt
results, common bank/holdout diagnostics and compute costs. No filtering, selection
or removal of RPB1 is allowed in the primary results.

## Decision before observing results

Apply the pilot's original screens **unchanged to each repeat**:

1. Adapted TB must improve both raw and noncopy joint success by at least five
   percentage points over each of frozen-head TB and unchanged initialization.
   Disorder and uniqueness may each decrease by no more than five points against
   either comparator.
2. It must be within two points of saved continued supervision on both raw and
   noncopy joint success, and within five points on disorder and uniqueness.

Add a **separate plain-ProGen screen**: adapted TB must improve raw and noncopy
joint success by at least five points over plain ProGen, with at most five-point
losses in either disorder or uniqueness. This measures useful improvement over
the starting generator; it does not replace or relax the matched architecture
comparison. These thresholds retain the project's existing practical conventions,
not a new power calculation or statistical significance claim.

**Confirmation passes only if both new repeats pass both the original progression
screen and the plain-ProGen screen.** Preserve the individual differences and
failure reasons. A favorable pooled average cannot override a failed repeat.

- If both pass, retain last-block GFN adaptation as the working generation method.
  The next question is generalization to new prompts or property profiles, under
  a separate protocol; do not launch it automatically.
- If results differ across repeats, report sensitivity to training randomness and
  inspect saved per-prompt/data-exposure diagnostics before proposing another run.
- If both fail, do not extend training or unfreeze further layers by default.
  Reassess the explanation for the pilot gain using all recorded outcomes.

This does not establish exact full-space target matching, experimental disorder,
novel biological function, or superiority to a capacity-matched non-GFN adapted
model. The saved supervised comparator trains only the head. V3 is an optimized
proxy, and V2 is related sensitivity evidence, not independent validation.

## Artifacts and verification

The run root contains `report.md`, `metrics.json`, `decision.json`, `costs.json`
and the enclosing generation/provenance records. Each `replicate-N/fit` preserves
the unmodified trainer's four-arm artifacts; `replicate-N/prior` holds the fresh
plain-ProGen generation. Record the true RNG settings and hashes at the root,
retain all source/model/data identities, and protect existing output directories.

Scoring recovery requires a complete outer generation manifest, meaning both
paired fits and both plain-ProGen generation stages finished. Seal this manifest
before final plain-ProGen scoring so a scorer failure cannot strand completed
generation. Recovery re-scores saved
sequences into a fresh directory, without loading ProGen or resuming training.
Current scorer/software compatibility checks still apply. An earlier training
failure needs inspection rather than automatic restart.

All 26 focused CPU tests and the read-only real-input preflight pass; see
[verification](../verification.md#last-block-confirmation--17-november-2026).
The tests use tiny random models and injected synthetic scorers, including recovery
after an injected final-scoring failure. The preflight executes no real model or
scorer. These are correctness checks, not real IDR experiments or confirmation
results; CUDA remains uninitialized.

## Completed results — 17 November 2026

`runs/idr-lastblock-confirmation-v1` completed both independent paired fits and
all five evaluation arms in **1 h 44 min 45 s**. Each arm has 1,024 raw generated
tails per repeat, balanced across the same eight prompts. Both final-block GFN
fits pass the original progression screen and the separate fresh plain-ProGen
screen. No repeat, prompt or checkpoint was selected after observing its outcome.

The [saved-artifact audit](../verification.md#confirmation-result-audit--17-november-2026)
reproduces 23 result files byte-for-byte from stored predictions, verifies
unchanged training and pinned source/model files, and leaves all run artifacts
unchanged. No model/scorer execution or GPU work was needed for this analysis.

| Method | Repeat 1 joint passes | Repeat 2 joint passes | Mean joint success |
|---|---:|---:|---:|
| Fresh plain ProGen | 23/1,024 (2.25%) | 22/1,024 (2.15%) | 2.20% |
| Unchanged supervised starting head | 109/1,024 (10.64%) | 101/1,024 (9.86%) | 10.25% |
| Saved continued head-only supervision | 152/1,024 (14.84%) | 133/1,024 (12.99%) | 13.92% |
| Matched frozen-head TB | 107/1,024 (10.45%) | 113/1,024 (11.04%) | 10.74% |
| **Last-block TB** | **209/1,024 (20.41%)** | **237/1,024 (23.14%)** | **21.78%** |

Joint success means satisfying both charge-profile requirements and the optimized
V3 predicted-disorder screen. These are unfiltered samples, not selected candidates.
The mean weights the two repeats equally; the earlier 20.41% pilot is excluded.

The adapted policy gains **9.96 and 12.11 percentage points over matched frozen
TB**, **5.57 and 10.16 points over saved supervision**, and **18.16 and 21.00 points
over fresh plain ProGen**. Its mean raw success is about **9.9 times** plain
ProGen's. This is a per-generated-output success ratio, not a training-cost or
wall-clock efficiency comparison. Plain ProGen requires no adaptation; the
confirmation's total runtime also includes both trained arms and all controls.

### Diversity, prompt dependence and target fitting

| Adapted-policy endpoint | Repeat 1 | Repeat 2 |
|---|---:|---:|
| Charge success | 22.36% | 26.17% |
| V3 disorder success | 92.29% | 91.21% |
| Unique-tail fraction | 96.97% | 96.88% |
| Exact common-training copies | 33/1,024 | 35/1,024 |
| Mean repeated-trimer fraction | 11.95% | 12.18% |
| Joint success using V2 sensitivity scorer | 20.31% | 23.93% |

Every joint-successful adapted tail is absent from its repeat's common training
tail set. Noncopy joint success therefore equals raw success, retaining all 1,024
outputs in the denominator. This check does not establish novelty against all
known proteins. Most outputs
still fail the combined requirement, primarily because charge acceptance remains
much lower than predicted disorder. V2 remains related sensitivity evidence, not
independent biological validation.

Adaptation beats matched frozen TB on **all eight prompts in both repeats**.
Against saved head-only supervision it wins on seven prompts in repeat 1 and six
in repeat 2. RPB1 (`DP00181`) remains poor: joint success is only 5/128 and 2/128,
unique-tail counts are 97/128 and 96/128, and repeated-trimer fractions are 56.87%
and 58.72%. It accounts for all 33 adapted copies in repeat 1 and 34 of 35 in
repeat 2; the remaining repeat-2 copy is from `DP01538`. RPB1 stays in the primary
results. The run's `report.md` and `metrics.json` retain every per-prompt result.

The common 95-trajectory holdout still does **not** show better target fitting:
adapted versus frozen TB RMSE is 3.312 versus 3.288 in repeat 1, and 3.616 versus
3.200 in repeat 2. Centered log-ratio RMSE is 1.970 versus 1.669, then 2.138 versus
1.703. These saved-path diagnostics are not global target-distribution distances.
Useful generation improves without demonstrating uniformly better reward-target
matching. The copied final blocks change by 1.20% and 1.23% in relative L2 norm;
saved adaptation diagnostics report no gradients on the reference model.

### Interpretation and next decision

The original pilot's generation benefit survives both fresh training streams.
Together with the fresh plain-ProGen comparison, this supports **retaining
last-block GFlowNet adaptation as the working IDR generator for this setting**.
It is stronger evidence than a single favorable run, but remains two new fits
of one supervised initialization, one charge target and eight development prompts.
It does not establish experimental disorder, exact target sampling, arbitrary
property control or superiority to capacity-matched non-GFN adaptation.

The initial next proposal was a **second predeclared charge profile**, keeping
the same eight prompts and the working last-block GFN recipe. Train a separate
fit for the new target, with fresh plain-ProGen and matched frozen-GFN controls.
This asks whether training can steer toward another requested profile; it is not
a claim that the current checkpoint obeys arbitrary new targets without retraining.
A separate bounded protocol must specify the target and acceptance rules, starting
head and replay-bank construction before real execution. None is implemented here.

Test new prompts separately later. The existing head includes eight prompt-identity
inputs and the trainer checks those identities, so merely swapping prefixes would
not constitute a clean held-out-prompt test. That experiment needs explicit handling
of unseen prompt identities. This architectural constraint favors changing the
property target first, without changing prompt support simultaneously. Further
reward/loss/layer sweeps are not motivated by this successful confirmation.
The subsequent user-directed priority and supporting diagnosis follow below.

### Post-hoc headroom analysis — 17 November 2026

The user prioritized improving the current success rate before changing properties.
This analysis reuses the inspected development outputs; it is not a new experiment
or an independent confirmation. The [failure analysis script](../verification/idr-lastblock-headroom-failures-2026-11-17.py)
and [counts](../verification/idr-lastblock-headroom-failures-2026-11-17.json)
retain input provenance and use saved sequences/scores only.

Across the two adapted fits (2,048 outputs), 446 pass both requirements (21.78%),
497 pass charge (24.27%), and 1,879 pass predicted disorder (91.75%).
**1,433 already pass disorder but fail charge**, while only 51 pass charge and
fail disorder. Perfecting the disorder component alone, with charge outcomes fixed,
would therefore add only 2.49 percentage points. Charge control is the immediate
bottleneck, particularly the second half being insufficiently negative (1,093
outputs, 53.37%).

The charge tolerances require net nominal charge +2 to +6 in the first 32 residues,
−6 to −2 in the second 32, and 8–24 charged residues overall. Of outputs already
passing disorder, 456 need one residue-category substitution and 376 need two to
meet all charge constraints. These are exact minimum count changes among positive,
negative and neutral residue categories, not tested sequence repairs. Such edits
could alter disorder, prior probability or diversity; 832 near misses do not imply
an attainable 40.63-point gain. RPB1 is poor (7/256 joint passes), but raising it
to the other seven prompts' 24.50% average would add only 2.72 points overall.
The head already receives charge counts and sequence progress.

The adapted policy's fresh training-path joint success, grouped into quarters of
100 updates (200 paths each), is:

| Fit | Updates 1–100 | 101–200 | 201–300 | 301–400 |
|---|---:|---:|---:|---:|
| Historical pilot | 7.0% | 9.5% | 13.5% | 22.0% |
| Confirmation 1 | 6.5% | 11.0% | 17.5% | 15.0% |
| Confirmation 2 | 8.0% | 14.0% | 22.5% | 23.5% |

These small samples come from evolving training policies, not matched final
checkpoint evaluations. Two fits improve late; one is mixed. This motivates a
bounded budget test, without establishing that more updates will help. The
[curve/importance script](../verification/idr-lastblock-headroom-learning-2026-11-17.py)
and [output](../verification/idr-lastblock-headroom-learning-2026-11-17.json)
retain this calculation. Post-hoc importance weighting of the adapted outputs
toward the current reward estimates 48.0% and 45.0% joint target mass, but prompt
effective sample sizes are only about 8–28 of 128 and individual weights reach
31%. These noisy, potentially biased estimates suggest possible fitting headroom;
they are neither a known target success rate nor a promised attainable result.

**Proposed next comparison:** keep the same last-block/GFN architecture, reward,
prompts and acceptance criteria; compare the preserved 400-update models against
two new 800-update fits from the original initialization. Keep the existing
75%-of-run decay convention: 600 updates at the initial rates, then 200 at one
tenth of them. This deliberately increases both training budget and duration at
the initial learning rates; it is not a pure test of additional low-rate updates.
Freshly evaluate the old and new models under matched sampling settings. The
new run's update-400 checkpoint is not the old control because schedules differ
after update 300. A practical progression screen would require at least five
percentage points of additional joint success in each stream versus its original
400-update control, with inherited diversity, copy and disorder checks. These are
proposed engineering choices, not literature defaults or a finalized implementation
protocol. Stop at 800 if improvement is absent; do not automatically extend fitting.

Preserving the absolute decay at 300 would make 800 updates add only 12.9% to the
sum of learning rates (350 versus 310 full-rate equivalents); the proposed schedule
doubles that sum to 620. This is a schedule heuristic, not a model of Adam progress,
but explains why merely continuing at the reduced rate is a weaker intervention.
The existing checkpoints also omit optimizer/RNG state. The proposal starts fresh;
loading a step-400 weight file would not be exact resumption. No checkpoint is
selected retrospectively by evaluation success.

If longer fitting plateaus, the next distinct candidate is a **single controlled
property-penalty comparison**, 2.5 versus 5, on this working adapted architecture.
The penalty acts on all four charge/disorder violations. For a sequence with one
charge count just outside tolerance and no other violation, its property factor
would fall from exp(−1.25) = 0.287 to exp(−2.5) = 0.082; passing sequences retain
factor one. This could favor accepted outputs more strongly but also make fitting
harder. It changes the reward distribution, so keep it separate from the budget
test. No claim of 100% achievable success follows from GFN theory.

Other options remain secondary: retaining fresh successes in diverse replay could
improve reuse (currently they survive only their four immediate updates), but
replay is not an established bottleneck; unfreezing additional layers has no
demonstrated need yet; imposing hard charge constraints changes the generator's
support and needs its own constrained-ProGen comparator. A second profile tests
transfer of the recipe, rather than improving the current task. These proposals
keep GFlowNet training central. No new training or scorer execution occurred.
