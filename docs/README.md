# Research documentation

Start with [current status](status.md) and the [scientific specification](../spec.md).

| Question | Record |
|---|---|
| What happened in the original biological pilot? | [Pilot results](experiments/biological_first_results_2026-10-15.md) |
| What was established before it? | [Experiment history](experiments/README.md), [standalone results](prior_results.md) |
| What is the latest result? | [Completed success-bonus comparison](experiments/idr_success_bonus_2026-11-19.md#completed-results--19-november-2026): 38.87% GFN versus 44.43% supervised success; noncopy yield nearly ties |
| What do we do next? | [Current decision](status.md#current-decision): use learning/coverage evidence to justify one next intervention; no new GPU run selected |
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
