# Averaged sampled TB — 16 November 2026

## Pre-run protocol

The [expected-gradient control](idr_expected_conditioning_2026-11-16.md) passed
with mean TV 0.02974 under conditioned Adam; one fresh and one replay draw per
prompt reached only 0.25337 averaged across seeds. Test whether averaging multiple
sampled trajectories approximates the successful raw-gradient expectation.

Use **16 fresh and 16 replay draws per prompt per update**, with replacement.
This is one preselected engineering budget: 16 independent draws reduce variance
of each sample-mean gradient by a factor of 16 at fixed parameters (standard
deviation by 4), without assuming that this determines learning performance.
The choice is a substantial noise reduction, not a tuned optimum. No batch sweep.
Replay uses the same scheduled pass/fail stratum for all 16 draws at a given
prompt/step; the fresh draws use the current policy, detached from differentiation.
Average the two group means equally, then average across eight prompts. Clip the
resulting aggregate gradient once and take one Adam step. Do not clip or step per
trajectory. Repeated leaves count with their sampling multiplicity.

Run three fixed sampling seeds, fresh 92000/92001/92002 and replay
95000/95001/95002, from the same original conditioned supervised head and normalizer
initialization. Larger draw counts consume different RNG streams; these are not
identical trajectory draws to the old controls. Reuse the completed one-pair and
exact-expectation results, verifying common inputs and initialization.

Keep 400 optimizer updates, head/direct/normalizer rates 1e-4/1e-3/1e-2, clipping 1,
and rate multiplier 0.1 after step 300. Keep reward, conditioning, trainable
LayerNorm, support and all other definitions from the linked protocol. No target
probabilities enter fitting. One CPU thread, deterministic operators, CPU only;
no generation, ProGen/scorer inference, holdout access or GPU execution.

**Cost:** 256 sampled trajectory draws per update (128 fresh + 128 replay),
102,400 per fit and 307,200 across three fits. This is **16 times** the earlier
sample count at the same 400 optimizer updates, not an equal-compute comparison.
Draws are from saved finite trees, not new ProGen rollouts; CPU timing does not
predict production generation cost. Exact enumeration remains a reference, not
an affordable full-generation operation.

Report final step 400 (no checkpoint selection): mean and maximum prompt TV,
exact joint-pass probability, clipping fraction, normalizer errors and histories.
Save sampled leaf indices, checkpoints, folded heads and source/input provenance.
Retain the local gate: mean and maximum TV <0.05 and mean joint mass within 0.02
of target. Require all three seeds to pass before calling it a robust sampled
repair; three seeds are a development check, not a significance test.

Stop after these three fits, irrespective of results. If all pass, propose a
separate short fully branching check with an explicit generation-cost budget.
If gains are partial or absent, quantify them and assess whether sampling cost
is justified; do not automatically increase the batch again or launch a GPU run.
The selected 408-leaf, 168-branch-state trees force most residues, so even a pass
would not establish new-IDR generation or biological efficacy.

This protocol is written before execution. Its motivation is the linked local
control and the sample-mean variance identity, not a new literature claim.
Production code and earlier run/source artifacts remain unchanged.

## Reproduction

From the project root, use a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/fit_idr_averaged_sampling_cpu.py \
  --output runs/idr-averaged-sampling-v1
```

## Completed results

All three bounded CPU fits completed (5.5–5.7 seconds of fitting each), saved in
`runs/idr-averaged-sampling-v1`. Final step-400 results:

| Seed | Mean TV | Worst prompt TV | Exact joint-pass probability | Gate |
|---|---:|---:|---:|---|
| 0 | 0.11427 | 0.16459 | 55.40% | Fail |
| 1 | 0.12071 | 0.17262 | 54.70% | Fail |
| 2 | 0.11461 | 0.17038 | 55.32% | Fail |
| Mean | **0.11653** | — | **55.14%** | All fail |

The matched one-pair control averaged TV 0.25337 and joint mass 41.03%; exact
averaging reached TV 0.02974 and joint mass 61.13%, versus target 62.02%.
Sixteen-pair averaging therefore reduces mean error by about **54%** and raises
joint mass by **14.10 percentage points**, improving both endpoints in all three
sampling seeds. It still misses the predeclared gate by a substantial margin.
These are exact probabilities over saved finite leaves, not generated sample
success counts or experimental disorder measurements.

All 400 updates still clip in each fit, compared with 22.75% for exact averaging.
This does not independently identify clipping as a cause. Actual sampled draw
counts are 102,400 per fit and 307,200 total; all scheduled replay strata verify.
The 16-fold sample increase is a real algorithmic cost even though the cached CPU
implementation evaluates each tree once per update and gathers repeated leaves.
Production runtime and memory cannot be inferred from these short CPU timings.

Sixteen focused [CPU tests](../verification/idr-averaged-sampling-tests-2026-11-16.txt)
pass. Tests check averaging and duplicate multiplicity in the gradient, valid
stratum sampling, exact single-draw agreement with the earlier trainer, and unchanged
source weights. The [saved-artifact audit](../verification/idr-averaged-sampling-audit-2026-11-16.json)
reconstructs every metric/normalizer, validates folded heads and all draw indices,
and verifies source hashes and unchanged/matched controls. No ProGen, scorer or
GPU execution occurred. Checkpoints remain diagnostic-only.

## Interpretation and next decision

The averaging direction has a reproducible local benefit; it does not yet provide
a practical sampled repair at this budget. The experiment deliberately spends
more samples per update, so it does not establish better sample efficiency or
justify copying a 16-fold rollout increase into full generation. Stop at this
budget; do not automatically enlarge batches, extend fitting, or launch a GPU run.

The next proposed analysis is to attribute **raw-gradient variance to fresh-policy
versus replay sampling**, at the shared initialization and saved final heads,
with parameters held fixed. Finite support allows checking both distributions
without another training run. This can identify where variance reduction is most
valuable: if replay dominates, averaging cached replay more thoroughly may reduce
noise without proportional new ProGen generation; if fresh sampling dominates,
that cheaper change alone is unlikely to suffice. Any diagnosis must account for
prompt weights and pass/fail strata and distinguish raw variance from nonlinear
clipping/Adam effects. This is a proposed analysis, not a measured attribution or
implemented follow-up. Do not infer the dominant source from the current fits.

Keep the successful exact-expectation result as the reference and the GFlowNet
objective unchanged. Open-generation transfer remains a separate unresolved test.

**Subsequent result:** the [variance diagnosis and repair](idr_variance_repair_2026-11-16.md)
is complete. Both fresh and replay noise matter; a fixed-proposal TB gradient
correction passes the local gate in six sampled fits. The new record owns the
matched control, confirmation, additional cost and transfer limitations.
