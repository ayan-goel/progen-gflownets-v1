# Property-guided IDR generation with frozen ProGen2

Learn a small GFlowNet guidance head that steers frozen ProGen2 toward specified
properties of intrinsically disordered regions (IDRs).

**The continuous-reward comparison is complete.** Joint charge/disorder success
was 9.86% for the starting head, 7.52% after previous-reward TB and 9.57% after
continuous-reward TB. The reward change did not improve on initialization and
failed the practical screen. See the [results and next decision](docs/experiments/idr_continuous_2026-11-15.md#completed-results--15-november-2026).

The [CPU variance-reduction repair](docs/experiments/idr_variance_repair_2026-11-16.md)
passes the small learning check in all six sampled fits: distribution error is
0.019–0.021 and joint-pass probability is about 61.7%, close to the 62.0% target.
A matched ordinary trainer still underfits. This is a fixed-bank GFlowNet training
result; transfer to new states and unrestricted IDR generation remains to be tested.

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

## Reproducing the completed comparison

No rerun is recommended as the next experiment. Use a fresh output directory
when reproducing; existing outputs are protected. Activate the environment and choose your GPU. Training and evaluation run together;
ProGen stays frozen and no new supervised fit occurs.

~~~bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
scripts/run.sh idr-continuous --output runs/idr-continuous-reproduction --device cuda
~~~

Real experiments and all GPU execution are user initiated. Development checks
are CPU-only:

~~~bash
scripts/test_cpu.sh
~~~

The run writes `report.md` and `metrics.json`, with per-method and per-prompt results.
See the [protocol](docs/experiments/idr_continuous_2026-11-15.md) for choices,
interpretation limits, and CPU evaluation recovery.
