# Frozen ProGen2 stability guidance

Learn a small GFlowNet guidance head on frozen ProGen2 using whole-sequence
stability rewards. Local experiments established accurate reward-target sampling
and measured stability gains on known variants of one protein.

The current step tests whether the stability predictor generalizes across protein
groups. A [64-group benchmark](docs/experiments/multiprotein_2026-11-10.md) compares
constant, ridge and MLP predictors on frozen ProGen features, with automatic
evaluation. It is [ready to run](docs/runbook.md#multi-protein-stability-predictors).

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

The [ridge-reward GFlowNet](docs/experiments/ridge_gfn_2026-11-09.md) realizes a
modest measured-stability gain on the original candidate set. The
[noise study](docs/experiments/reward_noise_2026-11-09.md) shows that accurate
sampling can persist even when corrupted rewards erase that gain.

The [completed three-split comparison](docs/experiments/split_robustness_2026-11-09.md)
finds modest target gains for both ridge and MLP. Ridge is not consistently
better. These are exact target evaluations on known single mutants, not new
GFN training or evidence of stable novel proteins.

The [completed position-held-out study](docs/experiments/position_holdout_2026-11-09.md)
finds useful targets for both predictors, with larger ridge gains on all three
splits. Its candidate sets and baseline probabilities differ from the random-split
experiment. The [completed GFN follow-up](docs/experiments/position_gfn_2026-11-09.md)
closely matches all three ridge targets across nine fits and realizes their
measured stability gains. This establishes the local sampling mechanism, not
generation of new stable proteins. The multi-protein predictor benchmark now tests that next question with related
sequences kept in the same partition.
