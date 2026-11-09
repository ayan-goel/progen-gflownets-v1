# Verification

The preserved source snapshot in runs/biological-first/source is the reference
for this reset. All 13 top-level Python modules match that snapshot after
restoring the original command-line interface. The frozen backbone, scorer,
reward, guidance and evaluation implementations are unchanged.

The retained suite covers backbone/token alignment, finite environments,
zero-head/beta-zero identities, scorer normalization, metrics, trajectory balance,
optimization scheduling and end-to-end synthetic CPU runs with evaluation replay.

## Reset verification — 3 November 2026

- All **26 retained CPU tests pass**, including tiny synthetic end-to-end tests.
- All **13 core Python modules** and **six configurations** match the saved pilot
  source snapshot byte-for-byte.
- Saved biological evaluation reproduces all **10 JSON files exactly**, excluding
  only sampling_seconds. The report is byte-identical.
- The existing .venv has the pinned Torch/Transformers/tokenizers versions;
  original checkpoint/tokenizer files exist and CUDA remained uninitialized.
- Retained Markdown file links resolve, and the CLI exposes only the original
  six commands. No later workflow imports remain in the active core.

Replay uses saved features/heads and inspected outcomes on CPU, with one compute
thread. No ProGen inference, GPU execution or real fitting was performed.
[Machine-readable verification](verification/reset-verification-2026-11-03.json)
and [CPU test output](verification/reset-cpu-tests-2026-11-03.txt) retain evidence.

Historical machine-readable evidence is indexed [here](verification/README.md).
The [exact-test result](experiments/exact_pass_and_biological_pilot_2026-10-15.md)
and [biological result](experiments/biological_first_results_2026-10-15.md)
remain the scientific evidence.


## Predictor/reward study

All 28 CPU tests pass (including the 26 retained tests). New synthetic checks
verify beta-zero recovery, shift/scale identities, capped-reward behavior,
concentration metrics, isolated family fitting, nested subsets and unchanged
predictions after perturbing evaluation labels. Saved pilot scores reconstruct
within 2.17e-7; the [protocol](experiments/predictor_reward_study_2026-11-03.md)
records the numerical findings. No real predictor retraining or GPU run was
performed during implementation.


## Ridge-reward GFN readiness — 9 November

All 30 CPU tests pass. New synthetic checks cover reward substitution with
unchanged prior features/logits, a four-update training run, exact and sampled
evaluation, saved-evaluation replay, immutable inputs, stale-hash rejection and
rejection of accidental real CPU training. Actual saved-input preflight passes:
950 training examples, 204 candidates, beta 1, scale 1.177569, maximum ridge
prediction reconstruction difference 3.34e-6. No GPU or real training was run.


## Fixed reward-noise readiness — 9 November

All 33 CPU tests pass. New synthetic tests verify zero-noise identity, pairing
across sigma, independent noise seeds, isolation from global training RNG, a
crossed panel of four-update fits, exact decomposition of utility change, saved
replay parity, source immutability and changed-input/real-CPU-training rejection.
Clean-run input hashes pass preflight. No actual noisy-target experiment or GPU
execution was performed during implementation.
