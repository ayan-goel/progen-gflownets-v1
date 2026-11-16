# Direct charge-context path: matched architecture comparison

**12 November 2026 — completed; no generation benefit demonstrated.**

## Question and motivation

Does giving task context a direct path to next-residue logits improve charge
steering under the existing trajectory-balance (TB) training setup?

The [real-state diagnostic](idr_learning_diagnostic_2026-11-12.md#completed-results--12-november-2026)
found tiny explicit-context responses and small negative charge corrections in
both blocks. Feature preactivation contributions were much larger than task-context
contributions. Larger gradient groups were not consistently better. These motivate
an architectural test, not a conclusion that scale imbalance caused the failure.
This is our engineering hypothesis; no paper is claimed to establish this fix.
The [pilot protocol](idr_charge_2026-11-12.md) owns the reward and original TB design;
the [literature review](../../literature/IDP_IDR_REVIEW_2026-11-12.md#7-charge-pilot-optimization-follow-up--12-november-2026)
records TB/SubTB context. The scientific target is unchanged.

## The single intervention

Both arms keep the original LayerNorm → shared hidden layer (128 SiLU units) →
20-logit residual head. The new arm adds a **bias-free linear map from the five
existing task-context values to those same 20 logits**. The map has 100 parameters,
all initially zero, and no new hidden layer, nonlinear activation or input scaling.

The five values are progress through the generated tail, accumulated first-block
charge, accumulated second-block charge, accumulated charged-residue fraction,
and the current block's charge target. Their original denominators are retained.
Prefix residues do not contribute to the reward or these accumulated charges.
Prompt one-hot identity remains available to the original head, but is excluded
from the added path. This tests task conditioning without a new prompt shortcut.

The added correction is `W c`: `c` is this five-value context vector; `W` is the
trainable 20-by-5 matrix. It is added to the original head's output and frozen
ProGen logits before the usual 20-residue softmax. It sees no future sequence,
terminal score, measured label or disorder prediction. The charge reward continues
to be computed only after generating the complete tail.

Both arms start with identical shared weights and zero output corrections, hence
the same ProGen policy. Constructing the added zero matrix consumes no random
numbers. Its parameters use the **same Adam group, learning rate and joint gradient
clipping** as the original head. The original feature path remains trainable;
ProGen remains frozen. This changes routing and capacity together, so improvement
would support the intervention, not uniquely prove the hypothesized cause.

## Fixed settings and budget

`configs/idr-context.json` pins `runs/idr-charge-v1/config.json` by SHA-256
`1cda219dc7e0fd4250be050d01adc92105243319c50a719fcc9f8134ad3f1bad`.
All scientific and optimization settings come from that saved configuration;
the new runner has no overrides for batch size, loss, reward or learning rate.
The prepared references must match the pilot's references, and the actual ProGen
weight/config hashes must match its provenance before model loading.

| Choice | Setting and reason |
|---|---|
| Arms | `standard`, `direct_context`; fresh fits of both, rather than relying solely on an old run |
| Seeds | 0, 1, 2; retain all original seeds, with identical shared initialization within each pair |
| Prompts / length | Same eight 16-residue prefixes, followed by 64 generated residues |
| Charge reward | Same +0.125 / −0.125 block targets, FCR target 0.25, scales/tolerances, beta 2 |
| Training budget | 400 updates per fit, eight on-policy sequences per update, one per prompt |
| Optimizer | Adam; head rate 1e-4, log-normalizer rate 0.01; joint gradient norm clipping at 1 |
| Schedule | Rates multiplied by 0.1 after step 300; fixed final step 400 used for evaluation |
| Added matrix | 20 × 5, zero initialization, no bias or separate learning rate; minimal direct route |
| Random streams | Initialization seed unchanged; training 1000 + 10 × seed; prior sampling +1; guided evaluation +2 |
| Evaluation | 16 raw samples per prompt/seed/head; 384 outputs per learned arm |
| Prior controls | One shared pool of 64 samples per prompt/seed: 1,536 samples; first 16 form raw prior, charge-only best 16 form selection |
| Other controls | Original 128-proposal composition-preserving swaps and GFN shuffles, reported in each arm's evaluation |
| Checkpoints / logs | Every 100 steps and final; every step logged, console progress every 10 |
| Precision / execution | Float32, deterministic algorithms, TF32 off, one CPU thread; user-selected GPU |

Total: **six fits, 19,200 training trajectories, 768 GFN evaluation trajectories,
and 1,536 shared prior proposals = 21,504 ProGen continuations**. Equal training
trajectories/updates do not imply perfectly equal wall time or parameter count.
Each arm's costs include the shared pool for comparison; do not sum that cost
twice when calculating actual total work. ProGen loads once for the experiment.

The earlier 74-minute pilot suggests roughly **2 hours 20 minutes** on comparable
hardware/load, scaling by trajectory count. This is a planning estimate, not a
measured duration. Scoring and setup contribute additional variability.

The two policies use matched random stream seeds but generally follow different
trajectories after they diverge. Historical pilot/diagnostic results and the same
prompt panel have already been inspected: this is development, not independent
confirmation on new proteins. No checkpoint/seed is selected using evaluation.

## What we measure and how we decide

Primary comparisons are **raw charge-pass rate** and **joint charge plus isolated
tail predicted-disorder pass rate**, for direct-context versus standard. Both
use exactly the pilot's thresholds. Show pooled results, every seed, every prompt,
and every seed/prompt group. Mean score and block charges help expose gradual
steering before a binary threshold is crossed.

Also report isolated/contextual predicted disorder, unique tails, repeated trimers,
reference similarity, prior likelihood, and fresh-sample TB residuals using the
unchanged scorer. The original select/swap/shuffle controls remain contextual
comparisons; they do not sample the intended tilted distribution.

On the *same fresh GFN evaluation trajectories*, record conditional next-token
charge changes in the first and second halves, token KL from ProGen, and the
diagnostic's fixed-feature progress/target contrast. These require no additional
ProGen trajectories and do not affect training. Positive-first/negative-second
average corrections are useful evidence of the requested directional response;
they are not sufficient proof of exact target matching. Context counterfactuals
can be inconsistent with actual history, and preactivation norms describe only
the original hidden path, not the new direct branch.

Support for continuing this design would be higher pooled charge and joint success
with consistent direction across seeds and multiple prompts, without a serious
diversity/disorder tradeoff. Report mixed or absent effects as such; no automated
promotion or significance claim follows from three seeds on the same eight prompts.
Do not call lower TB loss alone success. Do not tune a different branch rate or
extend training after inspecting partial outcomes within this experiment.

If the added path responds but generation does not improve, inspect whether its
effect is too weak or incorrectly directed. If it remains ineffective, revisit
optimization/credit assignment separately (including the previously discussed
SubTB comparison). No simultaneous batch-size, reward or backbone change is made.

## Run, artifacts and recovery

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # choose your GPU
scripts/run.sh idr-context --output runs/idr-context-v1 --device cuda
```

`scripts/run.sh idr-context --check-only` validates inputs without model execution.
The main command runs training, raw sampling and CPU disorder evaluation, then
writes a top-level `report.md` and `metrics.json`. `standard/` and `direct_context/`
contain full scorer reports, per-sequence metrics and grouped metrics. Checkpoints
and training logs are in `<arm>-seed-<seed>/`; architecture is explicit in each
checkpoint. Shared samples identify their architecture and prompt/seed, and
`behavior.json` retains per-batch/per-prompt token diagnostics.

`pilot_config.json`, references, provenance (source/model/scorer hashes, package
versions, device), costs and generation-completion checksums support reproduction.
Historical production modules and saved runs remain unchanged. This runner reuses
their rollout, reward, TB update and evaluation directly. New output directories
are required; there is no automatic resume of interrupted training.

If generation completes but scoring fails, replay evaluation without ProGen:

```bash
scripts/run.sh evaluate-idr-context \
  --run runs/idr-context-v1 --output runs/idr-context-v1-evaluation
```

This checks captured inputs and evaluation source/scorer hashes, refuses modified
inputs or incomplete generation, and performs CPU scoring into a new directory.
The raw samples stay untouched. Related metapredict V2/V3 predictions are not
independent experimental verification of disorder or a biological function.

## Verification

Four tiny CPU checks cover identical shared initialization and RNG preservation,
opposing charge responses from the direct path, TB gradients reaching that path
while ProGen stays frozen, and full paired execution/replay. The paired fixture
also reproduces the original trainer's standard-arm weights, normalizers and raw
samples exactly, verifies shared-pool counts and rejects modified replay inputs.
Real-input preflight passes without model execution. All 76 CPU tests pass,
including the four new checks. Formatting/lint checks pass, and historical pilot
and diagnostic implementation hashes remain unchanged. The assistant ran no real
training or GPU operations.


## Completed results — 12 November 2026

The user completed all six fits and evaluations in **8,029.40 seconds
(2 h 13 min 49 s)**. Each fit reached its fixed step 400. The run generated
19,200 training trajectories and 2,304 evaluation proposals (shared prior plus
both guided arms). [Generated comparison](../../runs/idr-context-v1/report.md)
and [detailed metrics](../../runs/idr-context-v1/metrics.json) preserve results.

| Method | Charge passes / 384 | Isolated disorder passes / 384 | Joint passes / 384 | Unique tails | Mean charge score |
|---|---:|---:|---:|---:|---:|
| Frozen prior | 20 (5.21%) | 247 (64.32%) | 10 (2.60%) | 97.14% | -1.935 |
| Standard head | 20 (5.21%) | 254 (66.15%) | 11 (2.86%) | 95.05% | -1.882 |
| Direct-context head | 17 (4.43%) | 247 (64.32%) | 9 (2.34%) | 95.05% | -1.850 |
| Charge-only selection | 79 (20.57%) | 207 (53.91%) | 33 (8.59%) | 98.70% | -0.741 |

Selection uses four times as many prior proposals per output and is not a sampler
from the specified tilted target. Its larger gain cannot be interpreted as proof
of an exact attainable target success rate.

The new head does **not** improve the primary generation metrics. Differences of
three charge passes and two joint passes are small and do not establish reliable
harm. Its charge-pass counts by seed are 7/5/5, versus standard 8/5/7; joint counts
are 4/2/3, versus 6/2/3. No seed improves either pass count. At the prompt level,
charge passes fall by one on DP01299, DP01538 and DP01109 and are unchanged on
five others; joint passes fall by one on DP01538 and DP01109. There is no hidden
broad prompt-level gain. RPB1 still has 29/48 unique tails; all other prompts have
48/48 in both learned arms.

Continuous score improves slightly (−1.882 to −1.850), and pooled fresh-sample
TB residual RMSE decreases from 3.1785 to 3.1056. These are not improvements in the
predeclared pass criteria. Mean block charges change from (+0.00765, +0.00781) to
(+0.00977, +0.00659), still far from (+0.125, −0.125). FCR stays around 0.165–0.166
against target 0.25. Diversity is similar, not rescued by the added path.

### Mechanistic result: responsive, but weak

The fixed-feature context contrast increases from approximately
0.000012 / 0.000051 / 0.000050 to 0.000957 / 0.001088 / 0.001365 nominal charge
per residue for seeds 0/1/2: roughly 20–80 times larger, **but still small in
absolute terms**. This intervention holds ProGen features fixed and may not describe
reachable sequence histories; its relative increase is not a success-rate gain.

Actual conditional charge corrections remain negative in both halves for all
three new heads: (−0.002954, −0.003603), (−0.003172, −0.004496), and
(−0.000645, −0.002029). They weakly distinguish the two blocks but do not produce
the intended positive-first correction on average. These are effects at each
head's own sampled prefixes, not matched-prefix causal differences between arms.

Post-hoc CPU inspection of the saved direct-branch matrices shows the correct
sign on the block-target input in every seed: K/R coefficients positive, D/E
coefficients negative. The final matrix norms are 0.03562 / 0.03645 / 0.04414;
maximum absolute entries are 0.01335 / 0.01477 / 0.01848. The matrix is connected
and learned; it has not remained at zero.

For a switch in target from −0.125 to +0.125, the direct target-column contribution
to the mean K/R-versus-D/E logit contrast changes by only 0.00628 / 0.00664 /
0.00820. This is `0.25 * (mean(K/R coefficients) - mean(D/E coefficients))`,
using `head.task_weight[:, 4]` in each final checkpoint and the fixed canonical
alphabet. It excludes progress and accumulated-charge effects and the original
head, and does not equal an entire-sequence density change. The same contrast
at steps 100/200/300/400 grows in every seed; growth slows after the inherited
rate decay. This is descriptive evidence of a small learned effect, not proof of
an optimal rate, insufficient training duration, or a single causal bottleneck.

Every training update clips the joint gradient in both arms. As before, that alone
does not establish clipping as the cause or convert the multiplier into an Adam
learning rate. The current TB/schedule still produces weak policy steering.

### Next decision (proposed; not implemented)

The added route responds in the right direction but did not improve generation
under the inherited small head rate and fixed budget. The next narrow test should
therefore isolate **the added branch's update scale**: compare its existing 1e-4
rate with a 1e-3 rate, keeping the original head rate, normalizer rate, clipping,
400-step budget, schedule, reward, prompts and frozen ProGen unchanged. The tenfold
factor is an explicit engineering test, not a literature-selected optimum or a
demonstrated remedy. Apply the same final-quarter decay to each branch rate.
Specify the matched execution/reuse plan before running; do not alter these
completed checkpoints or amplify their outputs at evaluation to claim a trained
improvement. No new command or production setting has been selected here.

This follows the predeclared question of whether a responsive path is too weak.
It does not promote the direct-context architecture as better. If a stronger
branch update still fails to improve raw charge and joint success, revisit
credit assignment/optimization with a controlled TB/SubTB comparison.
[Madan et al., ICML 2023](https://proceedings.mlr.press/v202/madan23a.html) motivate
SubTB through a gradient bias/variance tradeoff and partial-trajectory constraints,
with reported convergence benefits on longer action sequences. This does not prove
TB is wrong for our tree or that SubTB will fix this run. Its required intermediate
flow estimator also needs a separate, explicit design. Do not bundle loss changes,
backbone fine-tuning and branch-rate changes into one test.

### Verification after completion

CPU replay to `runs/idr-context-v1-evaluation` validates saved input and source/scorer
hashes and **exactly reproduces** all top-level, grouped and per-sequence metrics
for both arms. No ProGen inference or training is performed during replay.
The standard head's final weights and log-normalizers match the original pilot
bit-for-bit for all three seeds, and its full metrics dictionary is identical.
This verifies preservation of the baseline, not independent experimental replication:
the seeds, inputs and initialization were intentionally reused. All original
artifacts remain untouched. Checkpoint/step-log inspection was CPU-only; the
assistant performed no GPU operation or real training.
