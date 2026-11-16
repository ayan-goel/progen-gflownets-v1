# Charge-context branch rate: one bounded follow-up

**Protocol: 12 November 2026. Results verified 13 November: small generation gain; confirmation sampling proposed.**

## Question and rationale

Does a tenfold learning-rate increase **only for the added task-context matrix**
make its correctly directed but small corrections useful for generation?

The [completed context comparison](idr_context_2026-11-12.md#completed-results--12-november-2026)
found increased internal responsiveness but no generation improvement: 17/384
charge passes and 9/384 joint charge/disorder passes, versus 20 and 11 for the
original head. The matrix's target coefficients have the correct charge sign,
but its final norm is only 0.036–0.044 and its target-column effect is small.
This motivates a step-size test, not a conclusion that rate is the sole bottleneck.

The user authorized one focused follow-up, not an open-ended rate sweep. The
factor ten is an engineering choice large enough to test update scale. It is
not a literature-derived optimum. The original head rate stays unchanged; the
earlier toy result about increasing the entire head's rate is not equivalent to
this branch-only test. No extra literature claim is needed for this local choice.

## Exact intervention

Reuse the identical `DirectContextHead`: original 128-unit residual head plus
the bias-free, zero-initialized 20-by-5 matrix on the existing task inputs. Train
all head weights and log-normalizers from fresh initialization. The added matrix
gets its own Adam parameter group. Parameter groups are disjoint and together
contain precisely the original head parameters, the matrix and the normalizers.

| Parameter group | Original context experiment | This follow-up, steps 1–300 | Steps 301–400 |
|---|---:|---:|---:|
| Original residual head | 0.0001 | 0.0001 | 0.00001 |
| Added 100-parameter task matrix | 0.0001 | **0.001** | **0.0001** |
| Prompt log-normalizers | 0.01 | 0.01 | 0.001 |

The original run also decayed every rate by 0.1 after step 300. No warm start,
new input scaling, extra reward, loss change, longer fit or ProGen fine-tuning is
introduced. Adam defaults and joint gradient norm clipping at one are unchanged;
clipping is still over **all** head parameters and normalizers together, not
separately per group. After the policies diverge, gradients and trajectories can
differ: equal rates on the old path do not imply identical subsequent weights.

## Controls, provenance and computation

Reuse the completed `runs/idr-context-v1` lower-rate direct-head and standard-head
results and its exact prior pool. Those runs already use the intended seeds,
initialization, reward and schedule. Their standard arm reproduced the original
pilot bit-for-bit, and CPU scoring replay reproduced both arms. Rerunning the
same deterministic controls would add cost without providing new replication.

`configs/idr-branch-rate.json` pins the source generation manifest and both
control metrics files by SHA-256. Preflight verifies the manifest's captured
inputs, the completed status, references, package versions and implementation/
scorer hashes. The actual user run verifies ProGen weight/config hashes before
model execution. No source artifact is edited. Copies of control metrics,
behavior, configurations and provenance travel with the new run. CPU replay
does not require the old directory or pretrained weights.

| Setting | Fixed choice and rationale |
|---|---|
| New fits | Seeds 0, 1, 2; identical initialization seeds to the controls, no warm starts |
| New rate | 10 × inherited head rate for `task_weight` only; no sweep or best-rate selection |
| Training | 400 updates × eight prompt-balanced trajectories per fit = 9,600 new training trajectories |
| Evaluation | 16 raw samples × eight prompts × three seeds = 384 new outputs |
| Reused proposals | All 1,536 original prior-pool samples; raw prior, selection and swap controls reconstructed with the original rules |
| RNG | Training 1000 + 10 × seed; final generation 1002 + 10 × seed; identical stream seeds, not guaranteed identical sequences |
| Reward/data | Same eight prefixes, 16 supplied + 64 generated residues, two-block charge/FCR reward, beta 2, thresholds and prompt weighting |
| Model/numerics | Frozen ProGen2 medium, float32, deterministic algorithms, TF32 off, one CPU thread |
| Monitoring | Every step logged, console every 10, checkpoint every 100 and final; record all three rates and branch norm |
| Endpoint | Fixed step 400 for every seed; no early stopping or checkpoint selection by outcomes |

All other settings are copied from the source's `pilot_config.json`; see the
[context protocol](idr_context_2026-11-12.md) for the inherited definitions.
Total new ProGen continuations: **9,984**. This should take roughly **60–70 minutes**
on the hardware/load of the completed comparison, including setup and scoring.
It is an estimate. Prior generation is recorded as zero new work with its original
cost and reused count retained, not billed again as newly generated trajectories.
The saved sample file has 1,920 records: 1,536 reused proposals plus 384 new outputs.

These are the same already-inspected biological contexts and seeds. Reusing
controls and selecting the test after inspecting their outcomes makes this a
developmental follow-up, not independent confirmation. A matched seed comparison
does not imply the two policies share the same sampled trajectories.

## Measures and bounded decision

Primary endpoints remain **raw charge-pass rate** and **joint charge plus isolated
tail predicted-disorder pass rate**. Compare high-rate with low-rate direct heads,
and retain the standard head and raw ProGen references. Beating only a weak
lower-rate control while remaining at ProGen's success level would not establish
useful guidance. Report all seeds, prompts and seed/prompt groups, not just pooled
results. Retain continuous charge score, block means/FCR, disorder, diversity,
repetition and likelihood/TB diagnostics.

The same fresh generated prefixes supply token charge changes and fixed-feature
context contrasts. No extra ProGen trajectories are needed for these diagnostics.
Larger branch weights, stronger context response or lower TB loss alone do not
count as a successful follow-up. Disorder is predicted rather than experimentally
verified; selection/swaps do not represent sampling the desired tilted target.

We will look for charge and joint-success gains that occur across seeds and more
than one prompt, while retaining disorder and diversity. Report tradeoffs and
uncertainty explicitly: small pass counts and three seeds do not establish a
general statistical guarantee. There is no automated promotion threshold.

**Stop branch-specific tuning after this test if gains are absent or inconsistent,
or limited to internal diagnostics.** Preserve the negative result and move to a
separate optimization/credit-assignment investigation, potentially the previously
discussed TB/SubTB comparison. Do not immediately try further rates, multiply
trained logits afterward or extend the budget based on interim outcomes. If
generation does improve, assess robustness before claiming broader usefulness.

## Execution and outputs

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # choose your GPU
scripts/run.sh idr-branch-rate --output runs/idr-branch-rate-v1 --device cuda
```

Preflight only: `scripts/run.sh idr-branch-rate --check-only`.
The main command trains all three heads, generates samples, scores on CPU and
writes `report.md` plus `comparison.json` with the new and reused results.
`high_rate/` contains complete sequence/group metrics and the inherited control
report. `seed-*/` contains architecture-tagged checkpoints and stepwise logs;
each checkpoint records the multiplier, base configuration and optimizer groups.
`source/` preserves copied control evidence. The completion manifest hashes all
evaluation inputs. `samples.jsonl` uses distinct IDs for the new outputs.

Replay scoring after completed generation, without ProGen:

```bash
scripts/run.sh evaluate-idr-branch-rate \
  --run runs/idr-branch-rate-v1 \
  --output runs/idr-branch-rate-v1-evaluation
```

Replay validates inputs and implementation/scorer hashes, then checks that shared
prior/selection/swap counts and pass rates exactly reproduce the control values.
Continuous metrics allow relative tolerance 1e-6 and absolute tolerance 1e-7 for
scorer batch roundoff; a changed binary success rate is never hidden by that tolerance. All runs
require new output directories; interrupted training is not automatically resumed.

## CPU verification

Five new tests check: (1) disjoint optimizer groups and a tenfold branch-only
update under matched gradients; (2) exact schedule boundaries for every group;
(3) multiplier-one equivalence to the original trainer; and (4) complete tiny
execution with unchanged backbone/control artifacts, reused-proposal accounting,
CPU replay without the original control directory, and tamper rejection; and
(5) tolerance for tiny continuous-score roundoff while rejecting changed success
rates or materially changed control scores.
Real-input preflight passes without model execution. All **81 CPU tests pass**
(7.73 seconds), including the five new checks; lint/format checks pass. These tests
use tiny CPU fixtures; all real training and GPU operations remain user-run.


## Completed results — verified 13 November 2026

All three user-run fits reached step 400. The run completed in **3,940.66 seconds
(1 h 5 min 41 s)**, with 9,600 new training trajectories and 384 new evaluation
outputs. The 1,536 prior proposals and lower-rate/standard results were reused.
[Generated report](../../runs/idr-branch-rate-v1/report.md) and
[comparison metrics](../../runs/idr-branch-rate-v1/comparison.json) retain full details.

| Method | Charge passes / 384 | Joint charge/disorder passes / 384 | Isolated disorder passes / 384 | Unique tails / 384 |
|---|---:|---:|---:|---:|
| Frozen ProGen | 20 (5.21%) | 10 (2.60%) | 247 (64.32%) | 373 (97.14%) |
| Original head | 20 (5.21%) | 11 (2.86%) | 254 (66.15%) | 365 (95.05%) |
| Direct branch, original rate | 17 (4.43%) | 9 (2.34%) | 247 (64.32%) | 365 (95.05%) |
| Direct branch, 10× rate | **23 (5.99%)** | **13 (3.39%)** | 249 (64.84%) | 366 (95.31%) |

This is a small actual generation improvement in the observed sample, not only
an internal response change. Relative to the lower-rate branch it adds six charge
passes (+1.56 percentage points) and four joint passes (+1.04 points). Relative to
ProGen it adds only three charge and three joint passes (+0.78 points each).
These low counts do not establish a statistically reliable or broadly useful gain.
Disorder/diversity are roughly retained; this is not new experimental validation.

### Consistency and size of the effect

Charge counts by seed increase from **7/5/5 to 8/7/8**. Joint counts increase from
**4/2/3 to 6/4/3**. Thus charge improves in all three seeds, joint success improves
in two and ties in one. Against the original head, charge ties seed 0 and improves
seeds 1/2; joint success improves seed 1 and ties seeds 0/2. Reused seeds and controls
are not new independent replication.

Compared with the lower-rate branch, prompt-level charge counts increase on five
of eight prompts and tie on the other three. Joint counts increase on four and
tie on four. Mean score improves on all eight prompts and all three seeds: pooled
score changes from −1.8503 to −1.7106. The gain is not solely attributable to one
prompt. RPB1 still has only 30/48 unique tails; the other prompts remain 48/48.

Pooled block means move from (+0.00977, +0.00659) to (+0.01408, −0.00236), in the
requested direction but far from (+0.125, −0.125). FCR changes from 0.16524 to
0.16724 against target 0.25. Charge-only selection still reaches 79 charge / 33
joint passes using four times as many proposals per output; it remains a stronger
practical comparator here, not an exact target-distribution sampler.

### What changed inside the head

The explicit-context contrast grows from 0.000957/0.001088/0.001365 to
0.009625/0.010988/0.012910 nominal charge per residue across seeds. This is roughly
a further tenfold response increase. First-half conditional charge corrections
are now +0.000284 / −0.000279 / +0.000674, versus the lower-rate arm's negative
correction in every seed. Second-half corrections are more consistently negative:
−0.009601 / −0.012441 / −0.013025. The first-half effect remains close to zero,
including the wrong average sign in one seed; the pattern is not fully learned.

The final task-matrix norms are 0.3682 / 0.3790 / 0.4375, versus about
0.0356 / 0.0364 / 0.0441 previously. Fresh-sample TB residual RMSE decreases from
3.1056 to 2.6247, but is not a measure of exact distribution distance. All updates
still clip joint gradients. Nothing here establishes clipping as the cause,
100× as a better rate, or the chosen rate as optimal. Behavior probes compare
policies on their own sampled prefixes, and fixed-feature counterfactuals need
not describe reachable histories.

### Initial next-step proposal: freeze settings and verify the small gain

The bounded test produced improvements across seeds and prompts, so the outcome
is not the predeclared absent/inconsistent-gain case. It justifies **checking
reproducibility of the observed generation gain**, not escalating the rate or
proclaiming the architecture successful. Keep all checkpoints and settings fixed.

Proposed next experiment: generate a larger, fresh evaluation sample from every
saved high-rate and low-rate head, plus frozen ProGen, on the same eight prompts.
For example, 64 samples per prompt and seed would give 1,536 outputs per method,
4,608 total fresh trajectories, with no new training. Fix new sampling seeds and
all counts before execution; exclude the currently inspected draws from the new
primary comparison. Use all three trained seeds, not the best head. Report paired
prompt/seed effects and uncertainty, alongside disorder and diversity.

This addresses sampling variability for the existing checkpoints. It does not
establish training-seed robustness beyond these three fits or generalization to
new prompts/proteins. Exact budgets and seed choices remain to be specified in a
separate implementation protocol. No confirmation command is implemented yet.
Do not launch a 100× run or another tuning sweep. If the fresh-sample gain disappears
or remains too small to be useful, stop branch-specific tuning and revisit the
separate credit-assignment/objective investigation described above.

### Verification after completion

CPU replay to `runs/idr-branch-rate-v1-evaluation` checks captured-input and
source/scorer hashes and **exactly reproduces** comparison, grouped and per-sequence
metrics. All source captures remain unchanged. CPU checkpoint/log inspection
confirms 400 consecutive updates per seed and the declared original-head, branch
and normalizer learning rates at every step, including the step-301 decay.
The assistant performed no GPU execution, pretrained ProGen inference or training.


### Priority revision after discussion — 13 November 2026

The user prioritizes improving guidance. Larger samples only improve measurement,
so defer the standalone verification proposal above. The [current literature-backed
proposal](../../literature/IDP_IDR_REVIEW_2026-11-12.md#8-training-method-priority-after-the-branch-rate-result--13-november-2026)
is a controlled TB/SubTB training comparison, keeping reward and token policy fixed
and using matched training budgets plus fresh evaluation. This does not overturn
the recorded result, establish a unique failure cause, or promise a large gain.
Exact design and implementation remain pending; no new experiment was launched.
