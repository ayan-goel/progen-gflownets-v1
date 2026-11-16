# Conditioned expected-TB control — 16 November 2026

## Pre-run question and protocol

The [paired sampled comparison](idr_sampled_conditioning_2026-11-16.md) improved
mean TV from 0.30887 to 0.25337 with conditioning but still failed the local gate.
Does eliminating finite-sample gradient noise repair the inherited Adam fit?
The successful conditioned L-BFGS result also changed optimizer and proposal
weighting; it cannot answer this question by itself.

Run **one deterministic 400-update CPU fit** from the same original supervised
head and balanced normalizer initialization. Reuse all eight locally masked trees
(408 training leaves, 168 branching states), the frozen input transform, continuous
terminal reward, and unchanged inherited Adam/replay schedule. No new random
initialization, holdout access, generation, pretrained/scorer inference or GPU work.
Do not start from a previously trained diagnostic checkpoint. Reuse completed
sampled results as controls; do not rerun them or select a seed.

At each update, replace each prompt's two sampled squared TB residuals with their
exact weighted average over leaves. Weight each leaf by one-half the current
policy probability (detached and normalized) plus one-half uniform probability
within the pass/fail replay stratum scheduled for that prompt and step. Differentiate
only the residual, exactly as in sampled TB. The alternating strata remain; this
is not the fixed proposal weighting used in the earlier L-BFGS experiment. Exact
target probabilities remain evaluation-only; no teacher supervision is introduced.

This computes the conditional expectation of the **raw sampled gradient at the
current parameters**. Clipping and Adam are nonlinear, so applying them to the
expected gradient is not the expected sampled parameter update. Nor will the two
training paths visit identical parameters. This is the intended noise-removal
control, not an assertion of equivalent optimizer trajectories.

Inherit head/direct-branch/normalizer rates 1e-4/1e-3/1e-2, global clipping 1,
rate multiplier 0.1 after step 300, fresh optimizer state, and float32 parameters
with double-precision tree probabilities/residuals. Conditioning, LayerNorm training
and parameter counts remain as in the linked protocol. A single deterministic fit
needs no repeated sampling seeds. No tuning or extension beyond 400 updates.

Report final step 400: mean and maximum prompt TV, exact joint-pass mass, normalizer
errors, clipping fraction and history. Retain the gate: mean and maximum TV <0.05
and mean joint mass within 0.02 of target. Save source/protocol/input hashes,
checkpoint and ordinary folded head; reconstruct all reported final metrics.

- If it passes, finite-sample effects are a plausible main obstacle at these settings;
  next consider a bounded variance-reduction design, not immediate full generation.
- If it fails, noise removal alone is insufficient; focus on deterministic update
  behavior before increasing batches or changing the reward again. The known
  L-BFGS success provides a feasible target-fit reference, but uses different weights.
- If it improves partially, quantify the gain without calling it a repair.

These selected trees force most residues. Passing would not establish novel IDR
generation, unseen-state generalization or experimental biological validity.
This is our local control derived directly from the existing sampled estimator;
no new literature claim or external search is necessary.

## Reproduction

From the project root, use a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/fit_idr_expected_conditioning_cpu.py \
  --output runs/idr-expected-conditioning-v1
```

This protocol was written before running the fit. Production modules, existing
outputs, and source files pinned by earlier diagnostics remain unchanged.

## Completed results

The single fit completed in 5.34 seconds of CPU fitting. Outputs are in
`runs/idr-expected-conditioning-v1`; no new sampling seeds or extra updates were run.

| Condition | Mean TV | Worst prompt TV | Exact joint-pass probability |
|---|---:|---:|---:|
| Conditioned sampled Adam, mean of three completed seeds | 0.25337 | 0.34286–0.35511 across seeds | 41.03% |
| Conditioned expected-gradient Adam, this control | **0.02974** | **0.04908** | **61.13%** |
| Exact target | 0 | 0 | 62.02% |

**The control passes all predeclared gate conditions.** Worst-prompt TV is close
to the 0.05 boundary, so this is not near-perfect fitting or a large-margin pass.
Mean TV falls by about 88% relative to the sampled-seed mean; joint mass rises by
20.10 percentage points. Every prompt is below 0.05 TV. Clipping occurs in 22.75%
of expected-gradient updates, versus 100% of sampled updates. This is a consequence
of the changed gradient trajectory, not an independent demonstration that clipping
is the underlying cause.

The target, original initialization, transform, support, optimizer settings and
400-step budget match the sampled comparison. Exact enumeration performs more
trajectory summation per update; this is an information-rich diagnostic, not a
claim of equal sampling cost or a deployable enumeration strategy.

Fourteen focused CPU tests pass, including independent enumeration of all
fresh/replay sample pairs to verify the raw expected gradient and a check that
changing exact target probabilities cannot change fitted parameters. The
[test log](../verification/idr-expected-conditioning-tests-2026-11-16.txt) and
[saved-artifact audit](../verification/idr-expected-conditioning-audit-2026-11-16.json)
record verification. Saved per-prompt metrics and normalizer errors reconstruct
exactly; folding preserves the policy within 1e-4 TV. Source hashes and shared
inputs/initialization match the completed sampled control. No ProGen, scorer or
GPU execution occurred; all checkpoints remain diagnostic-only.

## Interpretation and next decision

Unlike the earlier L-BFGS comparison, this successful fit retains Adam and the
same current-policy/replay weighting rule. Removing finite-sample gradient noise
is sufficient to cross the local gate under these tested settings. This strongly
supports sampling noise and its interaction with clipping/Adam as a major remaining
obstacle on this task, after input conditioning. It does not prove a universal
failure mechanism, identify whether fresh or replay draws dominate, or select a
production batch size. It also does not show that conditioning can be removed.

The next proposed bridge is a **bounded sampled gradient-averaging control** on
these same cached trees: average multiple independent fresh/replay trajectories
per prompt before clipping and one Adam step. Keep the reward, conditioning and
optimizer-step budget fixed, and account explicitly for the additional trajectory
budget. This approximates the successful expectation using samples, unlike merely
extending noisy single-draw training. Predeclare one sampling budget and a stop;
do not sweep until a favorable result appears. No such follow-up is implemented
or run here. Full-generation GPU work remains paused until a practical sampled
approximation demonstrates useful fitting. New states and long branching sequences
will still require a separate check afterward.

**Subsequent result:** the [16-pair sampled approximation](idr_averaged_sampling_2026-11-16.md)
is now complete. It improves consistently but fails the local gate in all seeds;
follow that record for the new cost assessment and next analysis.
