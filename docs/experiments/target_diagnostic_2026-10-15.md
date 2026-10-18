# Development target diagnostic: implementation and GPU handoff

15 October 2026. Implemented following the [post-pilot proposal](../next_steps_2026-10-15.md).
**Historical handoff; the user has now completed this run. See [results](target_diagnostic_results_2026-10-15.md).** No GPU operation or
real model fit was executed by the agent. Data preparation and synthetic CPU tests
are complete. Frozen ProGen2 and the existing GFN head remain the project design.

## Why this experiment comes next

The [completed biological pilot](biological_first_results_2026-10-15.md) accurately
matched its predicted target, but measured expectation changed from 4.28005 to
4.26993 kcal/mol. The [label-free audit](../verification/post-pilot-prior-audit.json)
found 96.02% prior mass on positions 1–3; uniform logits on the same tree give
96.67%. Both reward reliability and construction-induced concentration need
attention. The [literature update](../../literature/POST_PILOT_REVIEW_2026-10-15.md)
provides sources and reading limits. This diagnosis does not imply a new algorithm
or a statistically established biological deterioration.

## Prepared panel

The immutable [panel manifest](../../data/processed/stability-panel-v1/manifest.json)
records source checksum, filters, counts, rejection reasons, identities and all
split-file hashes. All 413 eligible parents were considered. The anchor is the
previously inspected 1O6X case. Subsequent choices use descending valid variant
coverage, then name; no outcome-value ranking. Require nonempty disjoint WT_cluster
sets and global reference identity <=0.30 against every previously selected parent.

| Parent | Role | Fit | Inner validation | Outer development | Reserved test |
|---|---|---:|---:|---:|---:|
| 1O6X.pdb | Development, previously inspected | 761 | 189 | 204 | 204 |
| 1UBQ.pdb_L67S | Development | 759 | 189 | 203 | 204 |
| 1UFM.pdb | Development | 759 | 189 | 203 | 203 |
| 2K52.pdb_L42A | Confirmation reserve | 756 | 188 | 202 | 203 |
| 2M2J.pdb | Confirmation reserve | 755 | 188 | 202 | 202 |
| 2AMI.pdb | Confirmation reserve | 751 | 187 | 202 | 201 |
| 2L33.pdb | Confirmation reserve | 745 | 186 | 199 | 200 |
| 1WCL.pdb | Confirmation reserve | 733 | 182 | 196 | 196 |

Alignment uses installed/pinned Biopython 1.86 PairwiseAligner: global mode,
match +1, mismatch −1, gap-open −2, gap-extension −0.5; identity is identical
residues divided by all alignment columns, including gaps. Use the first alignment
on a score tie. These simple fixed settings and the 0.30 cutoff screen similar
references; they do not establish unrelated folds or unseen-family generalization.
The largest selected pairwise identity is 0.27907. Seven earlier candidates were
rejected before filling the panel. Selection favors well-measured long domains;
this is not a representative random sample of all proteins.

Panel size 3 development + 5 confirmation bounds initial cost; it is not a power
calculation. Counts include mutation-background parent names exactly as supplied
by Mega-scale; the manifest stores each actual reference sequence.

## Splits and outcome isolation

Retain the original outer 70/15/15 variant partition, using NumPy default_rng seed
42 and floor cut points. The reference is always in training. Within the outer
training variants, reserve floor(20% * count) for checkpoint selection using seed
43, with the reference retained in fitting. Approximately 56% of all variants
therefore fit the scorer, 14% select it, 15% evaluate development utility, and 15%
remain reserved. This changes the scorer's available fitting data relative to the
old pilot and must be considered when comparing them.

The inner 20% is an explicit new engineering choice to separate model selection
from target assessment. No post-selection refit occurs, preserving the selected
weights and fitting-only normalization. Each MLP member independently chooses its
checkpoint using inner RMSE. Ridge regularization uses the same inner split.
The outer development outcomes do not enter fitting, checkpoint selection, prior
construction, predicted rewards, or proposal ranking. They assess the frozen
alternatives after fitting. They may inform the later procedure decision, so they
are development data, not final test evidence.

Data preparation necessarily reads label availability and aggregates replicates
for all eligible parents. The experiment runner reads only fit, inner-validation
and development files for the three development parents. It neither opens nor
copies test-label files or confirmation-parent row files. Original 1O6X test
results were already inspected historically and cannot supply fresh confirmation.
No failed scorer or unfavorable development parent is omitted from the diagnostic.

## What the command runs

1. Verify configuration, permitted split checksums and the existing real exact gate.
   Check checkpoint hashes against that gate, then load frozen ProGen2-medium.
2. Check frozen weights and causal forward/cache alignment using the existing adapter
   check. Extract features in batches of eight complete, equal-length sequences.
   Mean-pool residue states, excluding the direction marker, as before.
3. For development candidates, cache 20-amino-acid logits at the preceding position:
   the direction marker predicts residue one. Derive the original locally constrained
   prior using the same legal candidate tree. This avoids repeated prefix forwards;
   a synthetic parity test checks it against the existing tree cache.
4. Fit ridge and five MLPs. Compare ridge, MLP seed zero, and the mean of MLP seeds
   0–4. Shared cached features and the seed-zero member are reused. Models remain
   frozen once fitted; no GFN, supervised-guidance head or confirmation fit runs.
5. Evaluate exact targets at alpha 1, 0.5, 0.25 with beta 1. Also evaluate alpha zero
   as a uniform-over-candidates diagnostic. Controls include uniform legal-action
   logits and complete-path 20-amino-acid likelihoods; neither replaces the prior.
6. Save exact expectations and matched categorical sampling references, then stop
   for review. No setting is automatically declared the winner or sent to training.

## Numerical choices

The executable settings are [target-diagnostic.json](../../configs/target-diagnostic.json).

| Setting | Value and reason |
|---|---|
| Backbone / precision | Same frozen ProGen2-medium / float32 as passed exact gate |
| Feature batch / CPU threads | 8 / 1, reuse validated memory and thread budget |
| Scorer features | Existing mean-pooled frozen causal features; isolate target diagnostics |
| Ridge penalties | 0.1, 1, 10, 100, existing screen; select by inner RMSE |
| MLP width / updates / batch | 128 / 500 / 64, existing bounded scorer budget |
| MLP optimizer | Adam, lr 0.001, weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8, no AMSGrad |
| Scorer clipping / validation | Global gradient norm 1; inner validation every 25 updates and final update |
| MLP seeds | 0–4; five fits are a practical disagreement check, not an optimal ensemble-size claim |
| Feature and label normalization | Fitting partition only, population standard deviation, floor 1e-6 |
| Scorer screen | Inner MAE below fitting-mean predictor and Spearman >=0.2; reported for every fit, not used to drop parents |
| Alpha / beta | 1, 0.5, 0.25 / 1; bounded sequence-prior sensitivity, no beta sweep |
| Uniform candidate control | Alpha zero, diagnostic only; differs from uniform token choices |
| Proposal / selection budget | 256 / 32 for each prior and exact target; duplicates retained |
| Categorical sampling seeds | 100000, 100001, 100002; common proposal counts and deterministic predicted-score tie-breaking |
| Unique-candidate ranking | Highest 32 predicted candidates (or all if fewer); explicitly separate from sampled selection |
| Uncertainty | Population standard deviation across five predictions and reference-subtracted predictions; no uncertainty reward penalty |
| Local probability check | Probability sum tolerance 1e-4 for independently batched causal forwards; normalize double-precision log paths before comparison |

The alpha family and every symbol are defined in the [proposal](../next_steps_2026-10-15.md).
Sequence-level tempering is not tokenwise temperature. Alpha=1 retains the scientific
baseline. No guidance implementation has been changed to use another alpha; any
later adoption must keep target, baseline sampling and guidance probabilities
consistent. Uniform and complete-20 controls must not be presented as original
full-vocabulary ProGen or biological design improvements.

## Outputs and interpretation

Each run saves a source/spec snapshot, software/checkpoint/tokenizer/panel hashes,
backbone checks, copied permitted split rows, frozen feature caches, selected scorer
weights, inner validation histories, and evaluation inputs. Per-parent timing and
backbone work counters distinguish feature extraction from scorer fitting. The total
includes setup and evaluation. Failed/interrupted directories are preserved; use a
new output for retry, not an implicit resume.

`evaluation/report.md` summarizes exact measured gains versus both the matched
alpha prior and the original prior. `summary.json` reports mean/range and positive
parent counts for every scorer/alpha, including the two new parents separately.
Per-parent JSON includes predictive errors/correlations, weighted errors,
concentration, positions, expected unique/duplicate counts, Hamming diversity,
probability vectors and all sampled/selected indices. `scorer_fits.json` retains
inner screen failures. Disagreement/error correlation is diagnostic, not calibration.
Top-distinct rankings use all candidates and are not budget-matched generation.

**Next decision:** review gains and concentration across all three parents. If
exact targets still do not give useful repeatable development gains, address the
reward/support before training guidance. Otherwise freeze a specific procedure
and implement the reserved confirmation comparison. Do not silently cherry-pick
parents, increase the search indefinitely, or treat a small positive mean as
significant. At this finite scale exact categorical sampling already solves target
sampling; a GFN efficiency claim needs later non-enumerable tasks.

## Run and reevaluate

From the v1 root, activate `.venv` and select a GPU yourself, then run:

```bash
scripts/run.sh diagnose-target --output runs/target-diagnostic-v1 --device cuda
```

The launcher preserves your `CUDA_VISIBLE_DEVICES`. The panel is already prepared;
no extra setup or automatic GPU selection is needed. After completion, evaluation
can be reproduced without a GPU or another fit, using a new output:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-target \
  --run runs/target-diagnostic-v1 --output runs/target-diagnostic-v1-reevaluation
```

Verification uses the [full CPU test log](../verification/target-diagnostic-tests.txt)
and [readiness record](../verification/target-diagnostic-readiness.json). The user-run diagnostic results and runtime are now recorded separately; confirmation
performance remains untested.
