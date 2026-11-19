# Property-guided IDR generation with ProGen2 and GFlowNets

Train a GFlowNet to guide ProGen2 toward specified properties of intrinsically
disordered regions (IDRs). The working method jointly adapts ProGen's final
transformer block and a guidance head. The original
ProGen remains fixed as the reward reference.

**Last-block GFlowNet adaptation improves IDR property generation.** The latest
1,600-update models achieve 34.18% and 31.05% joint charge/predicted-disorder success,
versus freshly evaluated 800-update controls at 29.39% in both repeats. These
smaller gains miss the predefined five-point improvement rule. We retain 1,600
updates as the working budget at the user's decision and defer further budget
tests. Diversity and disorder are retained. See the
[completed results](docs/experiments/idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026).
New-prompt/property generalization and experimental biology remain untested.

Completed stability experiments established accurate local reward-target sampling
and showed how reward errors limit biological utility. They remain the foundation
for this pivot; a general cross-protein stability predictor is no longer the
immediate objective.

- [Current status](docs/status.md)
- [Scientific specification](spec.md)
- [Pivot rationale](docs/decisions.md#2026-11-12--pivot-from-stability-to-idpidr-property-guidance)
- [Research documentation](docs/README.md)
- [Completed experiment records](docs/experiments/README.md)
- [Existing commands](docs/runbook.md)

## Next experiment: GFlowNet versus supervised training

Compare GFlowNet training against reward-weighted supervised learning, with the
same adaptable ProGen block/head, initialization and scored training examples.
Two fresh repeats train each method for 1,600 updates, then evaluate fresh IDRs.
See the [protocol](docs/experiments/idr_objective_2026-11-18.md) for the comparison
and its limits.

After choosing a GPU with `CUDA_VISIBLE_DEVICES`:

~~~bash
source .venv/bin/activate
scripts/run.sh idr-objective --output runs/idr-objective-v1 --device cuda
~~~

Training and evaluation run together. The [runbook](docs/runbook.md#matched-gfn-versus-supervised-training)
also provides input checks and saved-generation scoring recovery.

Real experiments and all GPU execution are user initiated. Development checks
are CPU-only:

~~~bash
scripts/test_cpu.sh
~~~
