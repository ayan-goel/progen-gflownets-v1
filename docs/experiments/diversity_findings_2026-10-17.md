# Diversity diagnosis and next experiment handoff

17 October 2026. **CPU diagnostic complete; new GPU ablation implemented but not run.**
User authorized the work. No real training or GPU operation was run by the agent.
The [protocol](diversity_plan_2026-10-17.md) fixes the experiment; the
[spec](../../spec.md#17-october-2026--development-diversity-and-prior-construction-ablation)
defines the explicit prior change. These findings use local artifacts, not a new
literature search or a claim of methodological novelty.

## What the diagnostic establishes

Original development data and frozen MLP-0 predictions were reused, with no scorer
refit and no fresh test-label access. Primary executable evidence:
[report](../../runs/diversity-diagnostic-v2/report.md) and
[summary with input/source hashes](../../runs/diversity-diagnostic-v2/summary.json).
Uniform candidates put only 2.5–3.0% of mass in positions 1–3. Uniform legal actions
put 92–93% there. Thus the branching construction alone creates early-position
concentration; this control does not quantify its causal share under real logits.

The count correction is a single derived control: multiply each legal action's
unnormalized ProGen weight by its number of descendant candidates. Uniform logits
then yield uniform candidates exactly. For actual ProGen, concentration can remain.

| Development parent | Original target unique /256 | Corrected target unique /256 | Original target raw stability | Corrected target raw stability |
|---|---:|---:|---:|---:|
| 1O6X.pdb | 11.94 | 105.28 | 4.13717 | 3.99166 |
| 1UBQ.pdb_L67S | 12.07 | 13.97 | 3.99475 | 4.13113 |
| 1UFM.pdb | 13.15 | 49.98 | 3.09141 | 3.28794 |

These are exact expected values; stability is measured kcal/mol. Corrected-target
mass in positions 1–3 falls to 1.3–2.2%. On 1UBQ, removing early-position mass
barely improves distinct count: model/reward concentration remains elsewhere.
On 1O6X, raw expected stability falls while coverage rises. Do not hide this tradeoff.

## Distinct design selection

For 100 categorical batches per policy: draw exactly 256, deduplicate **before**
ranking, select at most 32 by predictions, and never refill. These are exact-law
references, not trained GFN or online generation results.

| Parent | Original target count /32 | Corrected target count /32 | Corrected prior + selection stability | Corrected target + selection stability | Uniform candidates + selection stability |
|---|---:|---:|---:|---:|---:|
| 1O6X.pdb | 11.93 | 32.00 | 4.26702 | 4.29907 | 4.30795 |
| 1UBQ.pdb_L67S | 12.18 | 13.68 | 3.88724 | 4.01471 | 4.09867 |
| 1UFM.pdb | 13.09 | 32.00 | 3.30275 | 3.35689 | 3.43886 |

Read quality together with count. Uniform selection fills all 32 here and is a
strong comparator; these diagnostics do not establish that a GFN is necessary.
The new experiment asks whether the learners approximate the broader target and
whether their distinct-design outputs justify fitting costs. It is a controlled
mechanism/utility ablation, not a promised advantage or fresh confirmation.

A separate [historical confirmation review](../../runs/confirmation-unique-review-v1/report.md)
reuses its **original** proposals with the new rule. No new samples or fits are
made. GFN supplies about 10.7–14.3 distinct designs per parent on average, still
short of 32. Quality differs from the original repeated-slot metric because
lower-ranked distinct proposals are now included. This is post-hoc descriptive
evidence only and was not used to choose the corrected prior. It also differs
from the earlier diagnostic that merely deduplicated already-selected slots.

## Implemented next GPU experiment

Three development parents × two priors × three seeds = **18 seed pairs**, each
with GFN and supervised fitting. Reuse frozen MLP-0 rewards and cache backbone
prefix features once per parent. All original supports and learner hyperparameters
are held fixed. Failed screens remain visible. Every final checkpoint is evaluated;
no best-outcome selection. An exact target and uniform selection remain controls.

Estimated wall time is roughly **five hours** if per-fit costs resemble confirmation;
broader policies and server load may change it. This is an estimate, not measured
new-run timing. Fixed updates permit a controlled prior comparison, not equal-compute
claims. Cached source work, new prefix caching, fitting and decoding costs are separate.

From the project root, after selecting your GPU:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=YOUR_GPU_NUMBER
scripts/run.sh diversity --output runs/diversity-v1 --device cuda
```

The script preserves CUDA_VISIBLE_DEVICES and runs evaluation automatically.
Read `runs/diversity-v1/evaluation/report.md` and `summary.json` afterward. Do not
reuse an existing output directory. No auto-resume or automatic follow-on run.

CPU replay of a completed run, into a fresh output:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-diversity \
  --run runs/diversity-v1 --output runs/diversity-v1-replay
```

## Reproduction and verification

Executed CPU analysis commands:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh diagnose-diversity --output runs/diversity-diagnostic-v2
CUDA_VISIBLE_DEVICES='' scripts/run.sh inspect-unique --run runs/confirmation-v1 --output runs/confirmation-unique-review-v1
```

The diagnostic preserves input hashes and a source snapshot. The historical review
uses only existing confirmation proposals; its new selection rule is post-hoc.
[Readiness checks](../verification/diversity-readiness.json) verify frozen source,
tokenizer and checkpoint identities. All 41 synthetic CPU tests pass:
[test evidence](../verification/diversity-tests.txt). They cover branching correction,
zero-head behavior, causal cache parity, label-independent no-refill selection,
failed screens and replay integrity. They do not establish real-run convergence.

## Next decision after that run

Inspect all parents, including the difficult 1UBQ case. Separate exact-target
coverage/quality from head approximation, then compare each matched baseline,
supervised guidance and uniform selection with costs. If simple selection suffices,
report that. Do not silently switch to a favorable parent, beta or checkpoint.
A later independent confirmation needs genuinely unused evaluation data fixed
before tuning. Larger mutation distances require separate scorer/measurement
validation; they are not implemented or implied by this finite-support ablation.
