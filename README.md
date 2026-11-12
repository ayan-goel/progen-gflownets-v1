# Property-guided IDR generation with frozen ProGen2

Learn a small GFlowNet guidance head that steers frozen ProGen2 toward specified
properties of intrinsically disordered regions (IDRs).

**Ready: a matched charge-context experiment.** Compare the original guidance
head with one that gives charge instructions a separate path to its output.
Both use the same frozen ProGen, reward and training budget. The completed
diagnostic motivated this change; whether it improves generation remains to test.
See the [experiment plan](docs/experiments/idr_context_2026-11-12.md).

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

## Running

Activate the environment, choose your GPU, and run both heads and evaluation:

~~~bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
scripts/run.sh idr-context --output runs/idr-context-v1 --device cuda
~~~

Real experiments and all GPU execution are user initiated. Development checks
are CPU-only:

~~~bash
scripts/test_cpu.sh
~~~

The run writes `report.md` with the comparison and detailed results for each head.
See the [protocol](docs/experiments/idr_context_2026-11-12.md) for choices,
interpretation limits, and CPU evaluation recovery.
