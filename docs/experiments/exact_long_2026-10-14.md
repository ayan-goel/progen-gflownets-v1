# Longer exact TB experiment — 14 October 2026

## Result

The user completed `runs/exact-long` in **227.86 seconds** (about 3 minutes
48 seconds). Status is `exact_gate_failed`: one of three final TB seeds satisfies
TV <0.05. All three improve over the original 2,000-update experiment, but the
predeclared all-seed gate remains unmet. No biological experiment is authorized by
this result. The original gate and outputs have not been altered.

| Seed | TB TV, 2,000 updates | TB TV, 10,000 updates | Final criterion |
|---|---:|---:|---|
| 0 | 0.184645 | 0.097115 | Fail |
| 1 | 0.161050 | 0.047138 | Pass |
| 2 | 0.171200 | 0.077344 | Fail |

These are fresh experiments, not resumed checkpoints. The new run changed only
TB and supervised update counts from 2,000 to 10,000. The same architecture,
learning rates, seed list, reward, batch size and fixed-final selection were used.
The previous privileged seed-0 fit reached TV 0.042336 using exact conditional
labels; this TB result demonstrates that one sampled-TB seed can also reach the
reference criterion. Neither establishes reliable convergence across seeds yet.

Primary evidence: [status](../../runs/exact-long/status.json),
[gate](../../runs/exact-long/gate.json),
[evaluation](../../runs/exact-long/evaluation/report.md),
[costs](../../runs/exact-long/costs.json).

## Saved-result checks

A CPU-only reload reproduced the final TV values within 2e-7 of the original GPU
measurements. No backbone weights were loaded and no optimizer step was taken.
The [analysis record](../verification/exact-long-analysis.json) contains input
SHA-256 hashes, complete per-prefix error decomposition, final metrics and late
training-log summaries. Original artifacts remain untouched.

Probability sums remain near one (maximum absolute deviation about 1.5e-7).
The zero-guidance and beta-zero checks passed. Absolute log-normalizer errors are
0.025395 / 0.014626 / 0.010894. There is no observed normalization failure in these
checks; this does not constitute a proof that every implementation detail is correct.

Final supervised-baseline TV from the saved heads is
0.075807 / 0.078932 / 0.108683. Thus the Monte Carlo supervised baseline also has
remaining fitting error; do not interpret this run as a clean statistical
establishment of TB superiority. It differs from the privileged diagnostic:
its conditional information is estimated from 4,096 sampled baseline rollouts.
All methods' selected batches attained the maximum toy score of zero, so selected
score alone cannot distinguish their distribution-learning accuracy.

## Remaining error and optimization evidence

- Two-residue prefixes remain the largest contribution to target-to-policy KL for
  each TB seed. Their contributions are 0.020493 / 0.007397 / 0.011238 out of total
  KL 0.031200 / 0.008741 / 0.021870, respectively. Errors also remain at other depths.
- The last six logged sampled TB losses span 0.0224–0.1084 for seed 0,
  0.0185–0.0914 for seed 1 and 0.0380–0.1103 for seed 2. Those batches differ;
  fluctuating sampled loss is not itself a measurement of changing exact TV.
- At all **21 logged checkpoints** from step 8,000 through 10,000 in each seed,
  the recorded pre-clipping gradient norm exceeded the configured threshold of 1.
  This is a statement about logged checkpoints, not every intervening update.
  Clipping bounds the raw gradient norm; it does not directly bound Adam's
  parameter update or prove the chosen learning rate is wrong.
- Historical TB logging records sampled loss before each update and log Z after
  the update. It has no intermediate exact-TV series or intermediate checkpoints.
  Therefore we cannot establish whether seeds 0 and 2 ever crossed TV 0.05, or
  conclude that their final failure was solely an unlucky last update.

## Next proposed controlled test

Retain the existing head and the exact acceptance criterion. The next hypothesis
is that a smaller learning rate late in training could improve settling under
sampled TB. This is more targeted than another unqualified increase in updates.
A proposed comparison would keep the 10,000-update budget and seeds, and reduce
both head and log-normalizer rates by a factor of ten after update 8,000. Apply
the corresponding head-rate schedule to the supervised baseline, while keeping
its labels and sample budgets fixed. The final 2,000 updates would then be a
controlled settling phase, not oracle-selected early stopping.

The 80% boundary and tenfold reduction are engineering proposals, not fitted
optima or literature-backed guarantees. Log exact TV and both KL directions at
fixed intervals on this fully enumerable toy to distinguish persistent error
from late instability. Such metrics would be observational only and must not
change gradients, sampling, final-checkpoint selection or the passing threshold.
Count their extra evaluation cost separately.

**Implementation update:** this schedule and monitoring are now implemented and
verified on tiny CPU fixtures, but the real run remains user-run. See the
[new experiment record](exact_decay_2026-10-14.md) and `configs/exact-decay.json`.
No additional real experiment was launched by the agent. `exact-long.json` remains
the completed constant-rate configuration. Biology stays gated pending reliable three-seed TB
accuracy and the separate scorer-validation gate.
