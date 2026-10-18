# Exact-task failure and diagnostic follow-up — 14 October 2026

## Current conclusion

**Subsequent result:** the longer TB run is now complete: TV
0.0971 / 0.0471 / 0.0773. Only one seed passes. See the
[longer TB record](exact_long_2026-10-14.md) for the current decision;
the planning sections below preserve the rationale before that run.

The user completed the two 10,000-update exact-supervision fits. The original
learning rate (0.001) reached **TV 0.042336**, below the 0.05 reference criterion;
the smaller rate (0.0001) reached **TV 0.055953**. The existing head can therefore
represent a sufficiently accurate policy on this cache, at least for this seed.
Longer fitting was sufficient here; reducing the learning rate was not necessary.
This does not establish that the original TB failure had the same cause.

The subsequent experiment used fresh three-seed TB training with a longer budget, using
[exact-long.json](../../configs/exact-long.json). The biological experiment remains
gated on TB convergence and scorer validation. No architecture change is justified
by these results alone.

The user ran all real fitting experiments. The agent inspected saved results,
verified input hashes and prepared the new configuration. No agent GPU work or real
fitting experiment was launched. All original outputs remain unchanged.

## Completed experiments

| Run | Training | Result |
|---|---|---|
| [exact-first](../../runs/exact-first/status.json) | Real frozen ProGen2 cache; TB, seeds 0–2, 2,000 updates | TV 0.184645 / 0.161050 / 0.171200; gate failed |
| [exact-capacity](../../runs/exact-capacity/diagnostics.json) | Same head, exact conditional supervision, seed 0, Adam 0.001, 2,000 updates | TV 0.168884; KL(target‖policy) 0.100255 |

The first run took about 56 seconds overall. Its zero-guidance, normalization and
cached/full-backbone checks passed. This establishes execution and those invariants,
not successful distribution learning.

The capacity fit uses full enumerated targets and is privileged. Its `complete`
status means execution finished, not that the original TB gate passed. Its expected
score was −0.551032 versus the target's −0.554751. Agreement of this one mean does
not establish agreement of sequence probabilities.

The original recorded cross-entropy fell from 1.481888 to 1.087507. The lowest
logged value was 1.085066 at step 1,400, with later fluctuations. These historical
losses were measured **before** each numbered update; the final saved head evaluates
to 1.086524 after update 2,000. This timing difference explains the small mismatch
between its last logged loss and the reloaded-head loss; it is not a changed model.

## Read-only audit and findings

Reproduce from the saved cache, using a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh inspect-fit \
  --run runs/exact-first --fit runs/exact-capacity \
  --output runs/my-exact-capacity-analysis
```

The completed [audit report](../../runs/exact-capacity-analysis/report.md),
[full measurements](../../runs/exact-capacity-analysis/audit.json) and
[input hashes/source snapshot](../../runs/exact-capacity-analysis/provenance.json)
are retained. An earlier `exact-capacity-audit` report is also preserved; the final
analysis additionally compares the residual corrections required after ProGen logits.
Neither audit takes optimizer steps, loads ProGen weights, or accesses GPUs.

1. **Target probabilities:** backward subtree-mass calculation reconstructs the
   terminal target with TV about 4.3e-16. Independent leaf-to-edge accumulation
   agrees to 8.9e-16 in log conditional probability. This checks the saved-cache
   target calculation, not the biological validity of a future scorer.
2. **Loss interpretation:** the average conditional entropy is 1.073329. Therefore
   optimal cross-entropy is not zero. The fitted head's excess is 0.013195 under
   uniform-prefix weighting, while whole-sequence KL is 0.100255.
3. **Where error occurs:** prefix lengths 0/1/2/3 contribute respectively
   0.002669 / 0.018968 / 0.073087 / 0.005532 to KL(target‖policy). About 73% is at
   length two. Summing these contributions reproduces terminal KL within 5.6e-8,
   consistent with float32 policy likelihood summation.
4. **Why average loss hides this:** the training objective weights all 85 prefixes
   equally; 64 are at length three. Sequence KL weights a prefix by its probability
   under the target. Both objectives share the exact optimum if it is representable,
   but their finite-fit errors need not track one another closely. This is an
   objective-weighting distinction, not evidence of incorrect target labels.
5. **Feature separation:** all 85 raw feature rows and all 85 initial LayerNorm
   rows are distinct. However, `GA` and `SA` have identical explicit context and
   normalized-feature RMS difference 0.006688, while their required centered
   residual corrections differ by RMS 0.673380. For `GS` and `SS`, the corresponding
   values are 0.008126 and 0.702699. These are small input differences requiring
   substantial output differences, supporting a conditioning hypothesis. They do
   not prove an irrecoverable representation loss or inadequate architecture.

For the neighbor analysis, LayerNorm uses epsilon 1e-5 without learned affine
parameters, so distances describe the initial input representation. Neighbors are
selected by feature RMS distance among prefixes of equal length, for the ten
largest contributors to sequence KL (the root has no peer). Explicit context
separation is reported separately. Residual comparisons subtract each prefix's
mean correction over common legal actions, removing the irrelevant softmax offset;
pairs with fewer than two common actions receive null. No new learned feature
transform was fitted.

## Controlled fitting comparison — completed by the user

The command below was used for this completed comparison. Its default output now
exists; use a new name only if intentionally repeating it. From the project root:

```bash
scripts/diagnose_exact.sh
```

This runs CPU-only on the saved finite cache; no GPU selection is needed. It creates
`runs/exact-optimization` and refuses an existing output directory. To use a new
name: `scripts/diagnose_exact.sh runs/exact-first runs/exact-optimization-2`.
The two arms run sequentially and start from the same seed-0 initialization:

| Arm | Adam learning rate | Updates | Purpose |
|---|---:|---:|---|
| original-lr | 0.001 | 10,000 | Isolate a longer update budget at the original rate |
| lower-lr | 0.0001 | 10,000 | Compare a tenfold smaller rate at the same budget |

Both retain hidden width 128, beta 1, full-batch uniform branching-prefix
cross-entropy, gradient clipping at 1, Adam defaults, cached float32 inputs and
float64 target construction followed by float32 training targets. CPU threads are
fixed to one. The original diagnostic did not explicitly fix its thread count;
reproduction across thread settings may differ slightly numerically.

Ten thousand updates is a bounded fivefold extension, not a claim of sufficient
convergence. The smaller rate tests stability; it may also learn more slowly.
Record step 0, step 1, every 100 updates and the final update. All new loss/TV/KL
measurements are **post-update**; gradient norm is pre-clipping for that update.
Log the entropy floor, excess cross-entropy and both KL directions. Preserve the
final checkpoint, and report the best **logged** TV separately without selecting
or substituting that checkpoint. Logging can miss intermediate minima.

The script writes `comparison.md`, each arm's `progress.jsonl`, final head,
configuration, provenance/source snapshot, diagnostic JSON and read-only audit.
It stops on a failed command; completed/partial arm directories remain for inspection.
Individual fits record failure/interruption. A parent directory alone is not a
completion marker; the comparison appears only after both fits and audits finish.
No architecture, TB hyperparameters, gate threshold or biological configuration
has changed. No rewards, glycine counts or oracle features are added to the head.

## How results will determine the next decision

- If the original-rate longer fit passes, the original optimization budget was
  insufficient under that setup; test the justified change in a fresh TB run.
- If only the lower-rate fit passes, prioritize learning-rate stability. Its
  success supports expressibility on this cache, not biological generalization.
- If both plateau, inspect feature conditioning and an alternative optimizer
  before attributing failure to capacity. A narrowly controlled representation
  change would follow only if warranted; it would need documentation and new runs.
- If TV briefly passes but final TV fails, treat that as instability, not a fixed-final
  success. Examine the recorded curve before choosing a new schedule.

A privileged fit does not satisfy the three-seed **TB** acceptance gate. Biological
work remains gated on a fresh passing exact TB experiment and scorer validation.

## Verification and limits

The updated 21-test CPU suite checks oracle reconstruction on an irregular tree
with forced edges, the KL chain rule, exact agreement with five original optimizer
updates, saved-head metrics, flushed progress records, audit artifacts and overwrite
protection. Existing backbone/data/scorer/pipeline tests still pass. These use
synthetic fixtures only; the user subsequently completed the real-cache comparison described below.

## Results of the controlled comparison

[Saved comparison](../../runs/exact-optimization/comparison.md),
[original-rate audit](../../runs/exact-optimization/original-lr-audit/report.md),
[lower-rate audit](../../runs/exact-optimization/lower-lr-audit/report.md).
Both arms' saved input SHA-256 values were checked against the unchanged source cache
and configuration. Both completed with the declared seed, rates and 10,000 updates.

| Updates | TV, learning rate 0.001 | TV, learning rate 0.0001 |
|---|---:|---:|
| 2,000 | 0.140267 | 0.154215 |
| 4,000 | 0.108570 | 0.101244 |
| 6,000 | 0.057158 | 0.077801 |
| 8,000 | 0.050896 | 0.069660 |
| 10,000 | **0.042336** | 0.055953 |

The final checkpoint is the lowest logged TV in each arm; no oracle checkpoint
substitution was used. The original-rate arm first crossed 0.05 at logged step
7,400, then moved above and below the threshold before the final pass. It is not
uniformly converged throughout the late training window. The lower-rate arm never
crossed at a logged checkpoint. These are single-seed diagnostic results, not a
statistically established ranking of optimizers or learning rates.

The new original-rate arm's step-2,000 TV differs from the historical 0.168884 fit.
The new diagnostic fixes one CPU thread; the historical command did not record
or fix that setting. The exact reason for the difference was not isolated. The
strong budget comparison is **within the same new run**, 0.140267 at 2,000 to
0.042336 at 10,000, rather than an assumption of bitwise historical reproduction.

At the final original-rate checkpoint, KL(target‖policy) is 0.007822; length-two
prefixes contribute 0.007645, down from 0.073087 in the old saved fit. Their errors
remain the dominant residual, but the head has learned much of the necessary
separation. Feature-conditioning concerns remain possible efficiency issues;
an unavoidable capacity barrier at TV 0.05 is contradicted by this fit.

## Next experiment: longer TB and matched supervised training

Prepared configuration: [configs/exact-long.json](../../configs/exact-long.json).
Relative to the original configuration, **only two values change**: `training.steps`
and `supervised.steps`, both from 2,000 to 10,000. Extending both keeps the methods'
update budgets aligned; it does not claim identical wall time or training information.
The Monte Carlo baseline retains 4,096 rollout labels. All learning rates, head
widths, batch sizes, rewards, seeds 0–2, sampling settings, final-checkpoint rule
and TV threshold stay fixed. The original configuration remains unchanged.

This run starts fresh; it does not warm-start from privileged exact-label weights.
The existing GPU pipeline reloads frozen ProGen and rebuilds its cache, then trains
and evaluates both heads across all three seeds. GPU selection remains manual:

```bash
# After activating .venv and exporting your chosen CUDA_VISIBLE_DEVICES:
scripts/run.sh run --config configs/exact-long.json \
  --output runs/exact-long --device cuda
```

Success still requires final **TB TV <0.05 in every seed**. Inspect
`runs/exact-long/gate.json`, `status.json`, `evaluation/report.md` and per-seed
training progress. A passing privileged fit does not predict guaranteed TB success:
TB also learns its normalizer and relies on sampled trajectories instead of full
conditional labels. If longer TB still fails, investigate that objective and its
sampling/optimization before increasing head capacity.

The new configuration was validated without loading a backbone or initializing
CUDA. A structural comparison verified exactly the two documented update-count
changes. No training code changed, so the existing 21-test verification remains
applicable; no additional training test was necessary for this configuration edit.
The original biological configuration still references the failed `exact-first`
gate and is deliberately not made runnable by this diagnostic. A validated passing
TB run will need to be referenced explicitly in a subsequent biological configuration.
