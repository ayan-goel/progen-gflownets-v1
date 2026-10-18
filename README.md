# Stability-guided ProGen2 with GFlowNets

Learn a small guidance model that steers **frozen ProGen2 toward stable protein
variants**, using stability measurements from Mega-scale.

The research question is whether GFlowNet guidance can produce diverse, stable
candidates at a useful computational cost compared with supervised guidance and
sampling followed by selection.

## Approach

1. Fit a stability predictor on frozen ProGen2 sequence representations.
2. Use predicted whole-sequence stability to define a target distribution.
3. Train a GFlowNet guidance head to adjust ProGen2's next-residue choices while
   keeping the pretrained model fixed.
4. Evaluate measured stability, diversity, target accuracy and computational cost
   against matched baselines.

The [project specification](spec.md) defines the formulation, probability
conventions and evaluation rules.

## Current progress

Five-protein confirmation showed that GFlowNet guidance closely matches its target
and improves expected measured stability over constrained ProGen on all five
candidate sets. Diversity remains limited, and cheaper methods are competitive.
These results concern measured single-mutation variants, not unrestricted protein
design or new laboratory validation.

The next development experiment compares the original prior with a correction for
uneven branching in the candidate tree. It evaluates distinct designs at a fixed
proposal budget, alongside supervised guidance and selection baselines.

- [Current status](docs/status.md)
- [Confirmation results](docs/experiments/confirmation_results_2026-10-17.md)
- [Next experiment: findings and protocol](docs/experiments/diversity_findings_2026-10-17.md)

## Run the next experiment

From the project root, activate the environment and select an available GPU:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # replace with your chosen GPU
scripts/run.sh diversity --output runs/diversity-v1 --device cuda
```

The command runs training and evaluation. Use a fresh output directory for each
experiment. See the [runbook](docs/runbook.md) for readiness checks and saved-run
reevaluation.

Run the CPU correctness tests with:

```bash
scripts/test_cpu.sh
```

## Project structure

| Path | Purpose |
|---|---|
| `src/stability_gfn/` | Backbone integration, guidance, training, evaluation and CLI |
| `configs/` | Experiment settings |
| `scripts/` | Run and test entry points |
| `tests/` | CPU correctness and pipeline tests |
| `docs/` | Protocols, findings, decisions and operating instructions |
| `spec.md` | Scientific specification |

Start with the [documentation index](docs/README.md). The
[experiment history](docs/experiments/README.md) records completed work, and
[hyperparameters](docs/hyperparameters.md) explains numerical choices.
