# Research documentation

Start with [current status](status.md), then the [runbook](runbook.md) for commands.
The [spec](../spec.md) defines the scientific contract. Documentation requirements
are in [AGENTS.md](../AGENTS.md#research-documentation-required).

## Current work

| Need | Read |
|---|---|
| What is happening and what comes next? | [Current status](status.md) |
| How do I run or reevaluate it? | [Runbook](runbook.md) |
| What is the next GPU experiment? | [Diversity findings / handoff](experiments/diversity_findings_2026-10-17.md) |
| What is fixed for the new ablation? | [Two-prior development protocol](experiments/diversity_plan_2026-10-17.md) |
| What was fixed for confirmation? | [Five-parent protocol](experiments/confirmation_plan_2026-10-16.md) |
| What did confirmation establish? | [Completed results](experiments/confirmation_results_2026-10-17.md) |

## Experiments and evidence

| Need | Read |
|---|---|
| Chronological experiments, protocols and outcomes | [Experiment index](experiments/README.md) |
| Why we chose an approach or changed it | [Decision record](decisions.md) |
| How the implementation supports the research choices | [Implementation decisions](implementation_log.md) |
| Verification findings and their limits | [Verification history](verification.md) |
| Test logs, readiness records and numerical audits | [Evidence index](verification/README.md) |
| A consistent format for a new record | [Experiment/analysis template](experiments/TEMPLATE.md) |

## Design and literature

- [Architecture](architecture.md): model/module responsibilities and probability conventions.
- [Data](data.md): releases, filters, splits and outcome isolation.
- [Hyperparameters](hyperparameters.md): numerical choices and their rationale.
- [Research plan](../literature/EXPERIMENT_PLAN.md): scientific stages and comparisons.
- [Literature index](../literature/README.md): synthesis, bibliography, search record and archived sources.
- [Prior standalone results](prior_results.md): ProGen and TF-Bind-8 history.
- [Post-pilot planning record](next_steps_2026-10-15.md): rationale leading to the diagnostic and confirmation.

Existing records stay at their original paths so saved links and run references
remain valid. Dated records preserve what was known at the time; current state
belongs in [status.md](status.md). A passing CPU smoke test establishes mechanics,
not real-checkpoint convergence or biological improvement. All real fits and GPU
operations remain user initiated.
