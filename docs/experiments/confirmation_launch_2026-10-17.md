# Confirmation launch and initial warning

17 October 2026. Operator: **user**. Real GPU experiment launched; results not yet
reviewed. This is an observation record linked to the
[frozen protocol](confirmation_plan_2026-10-16.md), not a new experimental decision.

## Observed execution

The user reported running:

```bash
scripts/run.sh confirm --output runs/confirmation-v1 --device cuda
```

The actual physical GPU selection is not established by this record. The run's
own provenance records the environment. Its saved `status.json`, read during the
documentation cleanup, reports `running` and creation at 2026-10-17 18:48:22 UTC.
That file is a status observation, not a process-liveness check or completion claim.
No test outcomes were inspected.

The pasted log contained `UserWarning: Can't initialize NVML`, followed by
`2K52.pdb_L42A: frozen features and MLP fitting`.

## Interpretation and motivating source

Read the installed PyTorch CUDA module at
`/usr/scratch/agoel320/miniconda3/envs/abmlm/lib/python3.10/site-packages/torch/cuda/__init__.py`,
lines 725–736, and the project's confirmation execution order. The installed
`_raw_device_count_nvml` emits this warning when NVML initialization returns a
nonzero result. This is local source inspection, not a new literature search or
a diagnosis of the host's driver state.

In the runner, the displayed feature-fitting message follows checkpoint loading
and backbone validation. The log therefore shows that the warning did not prevent
those CUDA operations at that point. It does not guarantee continued health or
successful completion. The exact NVML failure cause remains unknown.

Decision: do not restart solely because of that warning; continue the existing
user-run experiment and investigate any subsequent actual error. No warning
suppression, driver change, GPU query or additional experiment was performed.

## Next observation

After completion, verify the saved status and fit seal, then analyze all declared
parents/seeds and baselines. Keep this partial observation separate from the future
confirmation result record. See [current status](../status.md).
