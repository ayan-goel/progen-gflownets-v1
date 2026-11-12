# Direct charge-context path: matched architecture comparison

**12 November 2026 — implemented; awaiting the user's GPU experiment.**

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
