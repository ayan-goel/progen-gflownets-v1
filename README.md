# Frozen ProGen2 stability guidance

Learn a small GFlowNet guidance head on frozen ProGen2 using whole-sequence
stability rewards. The current focus is the original Mega-scale
single-substitution experiment on **1O6X.pdb**.

The exact 256-sequence mechanism test passed. In the first biological pilot,
GFlowNet learned the predicted target accurately, but measured stability did not
improve over constrained ProGen. We are restarting from that result to understand
the predictor and reward before expanding the task.

- [Current status and next work](docs/status.md)
- [Scientific specification](spec.md)
- [Original biological result](docs/experiments/biological_first_results_2026-10-15.md)
- [Commands](docs/runbook.md)
- [Research documentation](docs/README.md)

## Running

Activate the existing environment and choose the GPU yourself before a real run:

~~~bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
scripts/run.sh --help
~~~

Real experiments and all GPU execution are user initiated. Development checks
are CPU-only:

~~~bash
scripts/test_cpu.sh
~~~

The completed comparison found that a full-data linear predictor gives a modest
exact-target stability gain on the original split. The trained GFlowNet now realizes that modest local gain. Robustness across
local splits remains to be tested. See the
[study protocol](docs/experiments/predictor_reward_study_2026-11-03.md) and
[run command](docs/runbook.md#predictor-and-reward-study).


Completed: [ridge-reward GFlowNet results](docs/experiments/ridge_gfn_2026-11-09.md).
Next: [fixed reward-noise sensitivity](docs/runbook.md#fixed-reward-noise-study)
against this clean baseline; training and evaluation use one command.
