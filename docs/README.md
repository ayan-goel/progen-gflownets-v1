# Research documentation

Start with [current status](status.md) and the [scientific specification](../spec.md).

| Question | Record |
|---|---|
| What happened in the original biological pilot? | [Pilot results](experiments/biological_first_results_2026-10-15.md) |
| What was established before it? | [Experiment history](experiments/README.md), [standalone results](prior_results.md) |
| What is the latest result? | [Completed 1,600-versus-800 comparison](experiments/idr_lastblock_1600_2026-11-18.md#completed-results--18-november-2026): modest gains miss the predefined threshold |
| What do we do next? | [Authorized objective comparison](experiments/idr_objective_2026-11-18.md): GFN versus weighted supervision with the same adaptable layers/data, 1,600 updates |
| What must we beat, and when do we target biological function? | [IDR benchmark and functional roadmap](../literature/IDR_BENCHMARK_ROADMAP_2026-11-17.md) |
| Why did the biological task change? | [Pivot rationale](decisions.md#2026-11-12--pivot-from-stability-to-idpidr-property-guidance), [historical stability spec](stability_spec_2026-11-10.md) |
| How does the retained stability implementation work? | [Architecture](architecture.md) |
| What data and split are used? | [Data contract](data.md) |
| Which parameters were chosen and why? | [Hyperparameters](hyperparameters.md), [decisions](decisions.md) |
| What substantive implementation decisions were made? | [Implementation record](implementation_log.md) |
| How do I run or evaluate the retained code? | [Runbook](runbook.md) |
| What has been verified? | [Verification](verification.md) |
| What literature motivates the project? | [Literature index](../literature/README.md) |

Document decisions and experimental evidence, not routine activity. Follow
[project instructions](../AGENTS.md). All GPUs and real experiments remain
user initiated; only small CPU correctness checks are run during development.
