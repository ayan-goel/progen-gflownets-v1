# Exact gate passed; first stability pilot — 15 October 2026

**Subsequent result:** the biological pilot is complete. See the
[final result record](biological_first_results_2026-10-15.md); it fits the target
accurately but does not improve measured stability over ProGen. Planning/interim
sections below preserve the state before final evaluation.

## Completed milestone

The user completed `runs/exact-decay-long` in **450.10 seconds**, about 7 minutes
30 seconds. Its status is `complete`, with `smoke=false` and `gate.passed=true`.
All three fixed-final TB checkpoints meet the unchanged **TV <0.05** criterion.

| Seed | Scheduled 10,000-update TV | Scheduled 20,000-update TV | Final criterion |
|---|---:|---:|---|
| 0 | 0.053937 | 0.040901 | Pass |
| 1 | 0.045554 | 0.046860 | Pass |
| 2 | 0.049461 | 0.044724 | Pass |

The longer phase helped the previously failing seed. Improvement is not monotonic
for every seed: seed 1's final TV is slightly higher than in the shorter run, but
still passes. This was a fresh run, not a continuation or a best-checkpoint swap.

Evidence: [gate](../../runs/exact-decay-long/gate.json),
[status](../../runs/exact-decay-long/status.json),
[evaluation](../../runs/exact-decay-long/evaluation/report.md),
[costs](../../runs/exact-decay-long/costs.json),
[read-only audit](../verification/exact-pass-analysis.json).
The audit verifies 203 observations per learner, the declared rates and exact
agreement between final monitored TB values and the gate. It saves input hashes;
no model execution, optimizer steps or GPU operations were performed by the agent.

## What this establishes, and its limits

Frozen ProGen2 plus the existing residual guidance head can learn the specified
whole-sequence tilted distribution under sampled TB to the chosen engineering
accuracy on the 256-sequence toy. The head architecture was unchanged. Additional
optimization and a late rate reduction were sufficient for the three final seeds.
Zero-guidance/beta-zero checks pass, probability sums are within about 1.3e-7 of
one, and log-normalizer absolute errors are 0.00440 / 0.00806 / 0.00455.

The last 21 logged points (updates 18,000–20,000) pass in 18/21, 19/21 and 20/21
cases for seeds 0, 1 and 2. Thus the fixed-final criterion passes, while some late
fluctuations remain. Do not claim every late checkpoint passes or that all future
seeds will pass. These same seeds were used during engineering iteration; this
is not an independent statistical confirmation after hyperparameter tuning.

Final Monte Carlo supervised TVs were 0.075339 / 0.069890 / 0.062804. This single
small benchmark does not establish a general advantage of TB. Its reward counts
glycine and is not folding stability. Neither stable new proteins, novelty, nor
better biological sequences has been demonstrated. Stop tuning the exact task
for this milestone and move to the planned scorer-validation stage.

## Prepared biological pilot

Executable configuration: [biological-pilot.json](../../configs/biological-pilot.json).
The original `biological.json` is retained unchanged as the historical initial
configuration and still references the failed first exact run. Use the new file.

Changes relative to that original biological configuration:

- `required_exact_run` now points to the actual passing `runs/exact-decay-long`.
  Existing pipeline checks require a real completed exact gate and the same
  checkpoint identity; no gate logic was weakened or bypassed.
- TB and supervised `steps` become 20,000, with
  `lr_decay = {"after_fraction": 0.4, "factor": 0.1}`. This transfers the verified
  optimization setting as a first engineering default, not a biological optimum.
  Both start fresh at head rate 0.001 (TB log Z rate 0.01), reducing after update
  8,000 to 0.0001 (log Z 0.001). Preserve all other head and sampling settings.
- Training-time exact distribution monitoring is omitted (disabled by default),
  as required for the biological configuration. Final finite-support distribution
  evaluation remains available against the **predicted** reward; it is not a
  substitute for measured stability.

No scorer candidate, label scaling, gate, prepared data split, beta or evaluation
sample size changes. The original dataset audit and [hyperparameters](../hyperparameters.md)
remain authoritative. This is a transfer of a tested training schedule, not reuse
of toy-trained or privileged head weights.

The pilot uses the prepared **1O6X.pdb** domain, 72 residues, with 950 training,
204 validation and 204 held-out candidate sequences. It compares local measured
single substitutions; it does not demonstrate broad unrestricted protein generation.
The pretrained backbone stays fixed. Test stability values are reserved for final
evaluation; they do not select the scorer or train guidance.

## What the command does

1. Verify the real exact gate/checkpoint identity and extract frozen sequence features.
2. Fit the existing ridge candidates and small MLP on training data; choose by
   validation RMSE. Require validation MAE below the training-mean baseline and
   Spearman correlation at least 0.2.
3. **If the scorer fails, stop** before guidance training and test-value evaluation,
   writing `scorer_validation.json`, costs and `scorer_gate_failed` status.
4. If it passes, freeze the scorer, construct the reference-relative scaled
   stability reward, and train TB plus supervised guidance for seeds 0–2.
5. Evaluate raw and selected sequences, including measured held-out stability,
   predicted scores, diversity and costs. Saved checkpoints support evaluation
   without retraining.

After activating `.venv` and exporting your manually selected GPU:

```bash
scripts/run.sh run --config configs/biological-pilot.json \
  --output runs/biological-first --device cuda
```

The launcher preserves the GPU selection. The output must be new. The biological
prefix space and feature work differ from the tiny exact task; its 7.5-minute
runtime is not a validated duration estimate for this pilot. Inspect status and
`scorer_validation.json` first; after completion, inspect `evaluation/report.md`.

The agent has validated the configuration and checked that only the declared
settings differ. No real scorer or biological training has been launched by the
agent. The training implementation is unchanged, so existing 26-test CPU
verification applies. The scientific specification changes only to record the
newly completed exact milestone; the research objective and acceptance criteria
remain intact.
