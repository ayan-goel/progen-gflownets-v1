# Five-parent confirmation: completed results and next decision

17 October 2026 (America/New_York). **Complete, non-smoke, all five parents and
all three final guidance seeds retained.** User ran the GPU experiment; agent
performed saved CPU evaluation and analysis only. No hyperparameters were changed.

## Conclusion

Frozen-ProGen2 GFN guidance reproduces the intended target and improves expected
measured stability versus the constrained ProGen prior on every reserved parent.
This establishes the mechanism on these finite single-mutant supports. It does
not establish an economical or consistently superior alternative to supervised
guidance or generate-and-select, nor diverse unrestricted protein design.

## Integrity and procedure

Protocol: [fixed confirmation plan](confirmation_plan_2026-10-16.md).
Original [report](../../runs/confirmation-v1/evaluation/report.md) and
[summary](../../runs/confirmation-v1/evaluation/summary.json) remain unchanged.
All five inner scorer screens passed. All fits completed before the runner
released outcomes. Saved seal verification passes for 57 artifacts; 27 source
snapshot hashes match provenance and all 26 current source modules match the
snapshot. These verify saved integrity and the implemented execution path, not
an independent forensic proof of every historical access.

Reproduced evaluation from saved artifacts, without ProGen loading or fitting:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-confirmation \
  --run runs/confirmation-v1 --output runs/confirmation-v1-review-replay
```

All 67 scientific output files are byte-identical; evaluation timing is excluded
because elapsed time varies. Calculations and verification counts are preserved
in the [CPU analysis audit](../verification/confirmation-result-analysis.json).
The original run took 14,881.52 seconds (4 h 8 min). No GPU operation was launched
by the agent. No new literature was needed: conclusions below cite this run's
fixed protocol and observed artifacts rather than external claims.

## Raw distributions: does the target help, and does GFN learn it?

Numbers below are exact expectations over each measured support, averaged over
final seeds where relevant. They are not estimates from 256 random draws. Positive
gain means higher measured dG_ML stability than the constrained ProGen prior.

| Parent | Candidates | Exact-target gain | GFN gain | Supervised gain | GFN mean TV |
|---|---:|---:|---:|---:|---:|
| 2K52.pdb_L42A | 203 | +0.26443 | +0.26440 | +0.25396 | 0.000117 |
| 2M2J.pdb | 202 | +0.10408 | +0.10410 | +0.10919 | 0.001010 |
| 2AMI.pdb | 201 | +0.19457 | +0.19433 | +0.19445 | 0.000564 |
| 2L33.pdb | 200 | +0.08386 | +0.08384 | +0.07228 | 0.000055 |
| 1WCL.pdb | 196 | +0.07634 | +0.07636 | +0.04368 | 0.000094 |
| Equal-parent mean | — | **+0.14466** | **+0.14461** | **+0.13471** | **0.000368** |

Gains are kcal/mol. Every one of the 15 final GFN fits has positive gain; worst
individual TV is 0.002234. Supervised mean TV is 0.018313. The mean GFN-versus-target
stability gap is only -0.000050 kcal/mol. Approximation error is no longer the
main limitation in this experiment. More accurate target matching need not yield
higher measured stability on every parent: the target uses an imperfect predictor.

Scorer test Spearman ranges from 0.733 to 0.886; MAE from 0.254 to 0.391 kcal/mol.
These are retrospective checks, not additional test-based selection. Each scorer
was fitted on local labels from its own parent. Reserved parent selection does
not make this a zero-shot or leave-family-out predictor evaluation.

## Selected batches: does it beat choosing good ProGen samples?

Each method supplies 256 proposals; the top 32 predicted-score slots are retained,
including duplicates. These are observed sampled batches, averaged over three
seeds, not exact expected selection performance.

| Parent | ProGen + selection | GFN + selection | Difference | GFN distinct / 32 slots |
|---|---:|---:|---:|---:|
| 2K52.pdb_L42A | 2.39879 | 2.45858 | +0.05980 | 3.67 |
| 2M2J.pdb | 2.71568 | 2.59943 | -0.11625 | 1.00 |
| 2AMI.pdb | 2.34623 | 2.38057 | +0.03434 | 3.33 |
| 2L33.pdb | 2.79085 | 3.10735 | +0.31650 | 7.33 |
| 1WCL.pdb | 3.44472 | 3.44472 | 0.00000 | 1.00 |

GFN wins on three parents, loses on one, ties on one; equal-parent mean difference
is +0.05888 kcal/mol. The largest benefit is on 2L33. As a descriptive sensitivity
check, omitting that parent would give -0.00553, so the positive mean is not a
uniform advantage. This is not a reason to exclude that parent from the result.

A **post-hoc diagnostic** counts each distinct sequence already present in a selected
batch once, using saved sequence identities and labels. The equal-parent gain is
then +0.03497 kcal/mol. This does not select a new batch or demonstrate 32 unique
designs; batch sizes differ. Preserve the predefined duplicate-weighted results
as primary. No significance or broad population claim follows from five parents
or from treating three seeds as independent proteins.

## Diversity and cost remain the limitations

- GFN expects only 10.64–14.89 unique sequences from 256 raw draws; equal-parent
  mean 12.15, versus ProGen 12.82. Selected GFN batches average 3.27 unique sequences
  per 32 slots, versus 3.80 for generate-and-select. Two parents select only one.
- GFN places 68–97% of mass on mutations in the first three sequence positions;
  the prior already places 73–96% there. Concentration is therefore not evidence
  by itself of a GFN optimization failure; the constrained prior and intended
  target are already concentrated. Better approximation does not fix that target.
- Total GFN fitting: 13,863.70 s (3 h 51 min). Supervised fitting: 635.35 s
  (10 min 35 s). GFN is **21.82 times slower in measured fitting time** for only
  +0.00989 kcal/mol additional mean raw gain. This compares the fixed implementations
  and budgets, not optimally tuned methods or equal compute.
- Shared per-parent setup totals 366.51 s and final CPU evaluation 5.50 s; total
  wall time also includes orchestration/loading overhead. Cached backbone calls
  are not free online inference. Direct exact-target sampling already solves this
  finite task, so this run cannot establish a GFN sampling-efficiency advantage.

## Next decision

Treat the finite-support mechanism confirmation as complete and preserve this run.
Do not spend another four-hour run tuning the same confirmation parents: these
outcomes have now been inspected and cannot serve as fresh confirmation again.

Before another GPU experiment, specify the next development benchmark and its
success criteria: distinct-sequence quality/diversity at matched proposal and total
compute budgets, with supervised guidance, generate-and-select and an exact target
where enumeration is possible. Audit the existing local-prior position bias on
**development data** before deciding whether the support construction or prior
needs changing. Any prior change changes the scientific contract and needs an
explicit spec revision and a separate comparator; it must not silently replace
this run's baseline.

Then consider larger supports or mutation distances only with an explicit scorer
validation and measurement-coverage plan. Multi-mutant predictions cannot be called
measured improvements without supporting outcomes. Choose fresh evaluation splits
before tuning that stage. This is a proposed design step, not authorization or a
configuration for a new run. No code or scientific definitions changed in this review.
