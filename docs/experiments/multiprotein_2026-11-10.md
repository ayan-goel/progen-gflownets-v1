# Multi-protein stability reward: first grouped benchmark

**Status: data prepared, 47 CPU tests passed, real-input preflight passed; ready for user GPU execution.**

## Question and scope

Can a stability predictor trained on several protein groups predict absolute
folding stability and identify higher-stability sequences in different groups?
The [completed GFN study](position_gfn_2026-11-09.md) established local sampling;
this experiment addresses the reward model's broader usefulness. No GFN training,
generation, ProGen unfreezing or restored archive workflow is included.

This is a bounded feasibility benchmark of short domains from MegaScale, not
validation on arbitrary full-length proteins or novel structures. Candidate sets
are fixed for every predictor. There is **no ProGen sampling prior** in this
experiment; top-score selection is compared to uniform selection over the same
measured candidates. This avoids conflating a changed candidate-tree prior with
predictor improvement. It is not a causal comparison to the old one-protein fits.

## Data, quality and exposure

Use the pinned release/member described in [data.md](../data.md), with aa_seq,
finite dG_ML, canonical residues and lengths 40–72. Absolute dG is kcal/mol, higher
more stable. The release combines natural and designed domains and includes
selected multi-mutants; this panel retains references and single substitutions
only for a comparable, tractable first reward benchmark. The assay measures
folding stability, not all desired biological functions.

The [primary paper](https://www.nature.com/articles/s41586-023-06328-6) describes
quality filtering and residual measurement limitations; the [author's release](https://zenodo.org/records/7992926)
identifies the processed Dataset2/Dataset3 CSV. An additional eligibility rule
requires reported deltaG_95CI to be finite, nonnegative and at most 1 kcal/mol.
This is our conservative engineering check on the supplied uncertainty field,
not a calibrated confidence interval for the ML predictor or a paper-mandated
cutoff. It removes zero additional rows among retained rows in this release.
Non-numeric/censored dG values are excluded, never assigned zero. Identical
sequences within a parent are averaged before splitting; replicate count/range
and quality metadata are retained. Require one unambiguous measured reference
and at least 64 valid distinct single substitutions.

Preparation parsed labels for validity, replicate aggregation and copying, not
for choosing high-stability proteins, stratification, fitting or evaluation.
No evaluation label statistics or selection outcomes are inspected in setup.

Prior exposure inventory: data/metadata/prior-exposure-2026-11-10.json records
16 existing metadata/identity files, including the active local manifest and
archived preparation manifests, scorer-scaling identity lists and scorer-transfer
identities. It contains 19,290 parent identifiers and 19,283 sequence hashes.
No retired label/result files or models are restored. This conservatively records
prepared/used identities; it is not an exhaustive proof of historical outcome
access. Entire components containing a known parent name or exact sequence
identity are excluded from this panel. Unknown past exposure, remote homology
and pretrained-model overlap remain possible: call the result **developmental**,
not a pristine independent confirmation set.

Archive clarification (10 November 2026): archive membership alone does not
invalidate data. The exclusion above is a conservative panel-design choice,
not evidence that every excluded protein was evaluated or that reuse would be
invalid for development. The user subsequently requested deletion of the retired
archive. The frozen inventory, its source hashes and the prepared panel remain
unchanged and usable; rebuilding that inventory from its original sources is no
longer possible locally. Deletion does not undo earlier outcome access.

## Grouping and fixed selection

1. Construct a graph over every eligible parent before selecting the bounded panel.
2. Connect parents sharing any nonempty WT_cluster identifier, any exact retained
   sequence, or reference global-alignment identity at least 40%.
3. Use connected components, so transitive relatives stay together. WT_cluster
   also has coarse design-class identifiers; grouping them together is deliberately
   conservative, not an assertion that every such class is a biological family.
4. Exclude components with known prior-exposure matches. Select one parent per
   remaining component by SHA-256 rank of selection_seed:name, then choose 64
   components by SHA-256 rank of selection_seed:group:component_id. Both rules
   use identities only. The representative component ID is its lexical minimum.
5. NumPy default_rng(42) permutes the selected components; floor(70%) train,
   through floor(85%) validation, remainder evaluation: **44 / 10 / 10 parents**.
6. For each parent, keep its reference plus 64 variants chosen by the fixed hash
   rank of selection_seed:parent:sequence. Equal counts avoid large scans dominating
   fitting. All 65 sequences, including the reference, stay in the parent's split.

Selection seed is **20261110**. Budget 64 × 65 = 4,160 proteins is an engineering
choice for an initial user-run feature extraction, not a power calculation.
Training/validation/evaluation have 2,860 / 650 / 650 sequences. Test reference
stabilities are not available as reward anchors during fitting.

Pairwise alignment uses Biopython 1.86 PairwiseAligner, global BLOSUM62, gap open
−10 and extension −0.5, including terminal gaps. Identity is identical aligned
residues divided by the longer full sequence length, using the first optimum in
sorted parent order. The [official documentation](https://biopython.org/docs/1.86/Tutorial/chapter_pairwise.html)
supports the alignment implementation/settings; the 40% cutoff is our heuristic,
not a guarantee of unseen families. It applies to parent references, not an
exhaustive all-mutant identity or structure search. Exact duplicates cannot cross
partitions. Source clusters and intermediate unselected parents participate in
component formation.

CPU audit: 776,298 input rows; 653,829 valid canonical length-matched ML rows;
414 eligible parents; 115 components; 106 available after exposure exclusions.
The selected panel's maximum cross-partition reference identity is 0.319444.
Group assignments, all parent references, pairwise selected-parent identities,
exclusions, settings and hashes are saved in data/processed/multiprotein-v1.
Neither remaining groups nor rejected rows are evaluated to choose this panel.

## Predictors, budgets and selection

| Setting | Choice and reason |
|---|---|
| Features | Frozen ProGen2-medium, same pinned checkpoint/tokenizer, float32, mean pooled residue representations; no parent ID or stability input. |
| Extraction | Batch 8, group by sequence length without padding, restore original row order. Extract once and reuse across all fits. |
| Constant | Mean of training labels only; no per-parent mean using held-out labels. |
| Ridge | Train-only normalization; penalties 0.1, 1, 10, 100, 1000; minimum validation RMSE. Added 1000 because the earlier grouped study selected the previous boundary 100 and this panel has more training rows. No test-based expansion. |
| Ridge solve | Use primal 1536-dimensional system when rows exceed features, otherwise retain existing dual solve. Same ridge objective, bounded memory. |
| MLP | Original 128-unit SiLU, Adam lr 0.001, weight decay 0.0001, clip 1. Batch 128, 2,000 updates, validate every 50 and retain minimum validation RMSE. |
| MLP budget rationale | Fixed 256,000 sampled training examples (~89.5 passes over 2,860 rows), allowing more updates for the larger dataset while retaining validation-only checkpoint choice. An engineering budget, not a tuned optimum. |
| Seeds | MLP 0/1/2, ridge once (deterministic for fixed data). Report all seeds, do not select a seed by evaluation performance. |
| Normalization | Fit feature and label means/scales on training only, same numerical floors as existing scorer. |
| Selection | Top 10% by predicted score, ceil count. Average over tied candidates at the cutoff, so constant scores do not exploit row order. |

Fit ridge and all MLP seeds before opening evaluation labels. Model-family choice
is not automated from test results. The historical local scorer gate remains a
reported validation diagnostic and does not suppress unfavorable model outcomes.
There is no GFN reward scaling, beta sweep or target distribution in this run.

## Metrics and interpretation

Report pooled MAE/RMSE/Spearman, prediction bias, macro-average per-parent error
and Spearman, and all per-parent results. Report reference-only regression and
selection separately to probe absolute stability across parent proteins rather
than only within-parent variation. Undefined correlations (constant predictions)
are null, not zero. Pooled correlations can reflect between-parent offsets;
per-parent metrics expose failures hidden by pooling.

Report measured mean of predicted top 10% and its lift over uniform candidates,
pooled, per parent and for reference-only sequences. On 65-sequence parent panels
this selects 7; on 10 reference proteins it selects 1, so the reference-only
selection result is very noisy. No arbitrary stable-enough threshold is introduced.
This ranking diagnostic does not measure generative sampling or independent assay
validation. Ten evaluation parents are a small panel; do not treat 650 variants
as 650 independent proteins for significance claims.

Useful held-out prediction/selection would motivate a larger specified benchmark
and a defensible generation reward; it would not automatically justify unrestricted
reward optimization. Poor generalization means repair the reward before connecting
it to broad generation. There is no automatic pass/fail or next run launched from
the metrics. Source lengths, assay bias, grouping heuristic and exposure limitations
remain explicit regardless of outcome.

## Commands and reproducibility

Prepared data is ready; after selecting CUDA_VISIBLE_DEVICES and activating .venv:

```bash
scripts/run.sh multiprotein --output runs/multiprotein-v1 --device cuda
```

This performs frozen feature extraction, all fits and evaluation automatically.
See runs/multiprotein-v1/report.md and evaluation/results.json. Saved artifacts
include feature/identity cache, individual predictor checkpoints, fitting/validation
history, predictions, post-fit evaluation labels, data/config/source/protocol
snapshots and hashes. Output reuse is rejected; failures preserve partial artifacts.
No automatic resume is implemented. Runtime has not been measured.

CPU readiness without model execution or evaluation-label parsing:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh multiprotein --check-only
```

CPU metrics replay, without training:

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh evaluate-multiprotein \
  --run runs/multiprotein-v1 --output runs/multiprotein-v1-recheck
```

CPU re-preparation into a fresh directory (does not alter the pinned data path):

```bash
CUDA_VISIBLE_DEVICES='' scripts/run.sh prepare-multiprotein \
  --output data/processed/multiprotein-recheck
```

Full source/exposure hashes live in configs/multiprotein-data.json; prepared manifest
and model hashes live in configs/multiprotein.json. Biopython 1.86 was already
available and is now declared as a dependency. Prior completed runs remain unchanged.


Verification: [CPU tests](../verification/multiprotein-cpu-tests-2026-11-10.txt)
and [readiness](../verification/multiprotein-readiness-2026-11-10.json). Real
prediction performance was unknown at implementation time; completed results are
below. No real training, backbone inference or GPU work was performed by the assistant.

Source-access note (10 November): the journal full-text fetch was blocked by its
redirect. The assay discussion here uses the indexed primary-paper abstract/methods
excerpt and the author release listing, not a new full-paper review. The Biopython
1.86 alignment documentation was directly accessible.

## Completed results — 10 November 2026

The user completed runs/multiprotein-v1 in **57.875 seconds**. Loading and frozen
feature extraction took 46.231 seconds: 553 forward calls, 4,160 sequences and
248,690 input tokens. This is predictor fitting, not a GFlowNet experiment.

| Predictor | Test RMSE | Test pooled Spearman | Mean within-parent Spearman | Mean within-parent selection lift | Parents with positive lift |
|---|---:|---:|---:|---:|---:|
| Constant | 1.442 | undefined | undefined | 0.000 | 0/10 |
| Ridge | 1.202 | 0.545 | 0.383 | +0.371 | 9/10 |
| MLP seed 0 | 1.342 | 0.339 | 0.195 | -0.008 | 4/10 |
| MLP seed 1 | 1.170 | 0.620 | 0.174 | +0.113 | 6/10 |
| MLP seed 2 | 1.083 | 0.641 | 0.188 | +0.227 | 6/10 |

RMSE and lift are kcal/mol. Within each test parent, rank its 65 measured
candidates by prediction and select seven (ceil of 10%); compare their measured
mean with uniform selection of those same 65. Average the ten lifts equally.
This is not improvement over ProGen and does not establish novel protein design.
Ridge's per-parent lift ranges from -0.0105 to +1.1515. MLP seed 2 has the lowest
test error, but was not selected as a winner by validation.

**Validation prevents promoting any model to a reliable broad-generation reward.**
The constant validation RMSE is 1.447. Selected ridge/MLP 0/1/2 RMSEs are
1.639 / 1.627 / 1.515 / 1.617; pooled Spearman correlations are
-0.118 / -0.065 / 0.158 / 0.159. All four fail the existing reported validation
gate. Ridge selects penalty 1000, the grid's upper boundary. MLP checkpoints
are selected at steps 50 / 50 / 100 out of 2,000; later training does not improve
the selected validation RMSE. These findings suggest generalization/regularization
problems, but do not identify their cause.

A post-hoc CPU diagnostic on the selected checkpoints separates pooled from
within-parent behavior: validation ridge has mean within-parent Spearman 0.285
and selection lift +0.291 despite negative pooled Spearman. Thus useful local
ranking can coexist with poor prediction across parents. The validation/test
discrepancy must not be hidden by reporting test metrics alone. Only ten parents
occupy each partition; hundreds of variants are not independent parent replicates.

Next proposed step: diagnose between-parent bias versus within-parent ranking
using the cached features and training/validation labels; then predefine grouped
cross-validation and a stronger regularization comparison before another user-run
fit. Do not tune to the displayed test table or automatically promote ridge/MLP
to broad generation. The current test panel is now inspected development data.
The immediate bottleneck is reward transfer across proteins, not evidence that
the previously validated GFlowNet optimizer needs changing.

Verification: CPU metrics replay in runs/multiprotein-v1-recheck exactly matches
saved results; feature-cache SHA matches provenance. All four saved checkpoints
reproduce test predictions on CPU within 1.20e-6 kcal/mol. See
[verification evidence](../verification/multiprotein-results-2026-11-10.json).
No training or backbone inference was rerun during analysis; original artifacts
remain unchanged.

## Follow-up diagnosis — 10 November 2026

Question: why do selected models rank variants usefully while failing overall
validation? This is a post-hoc analysis of cached **training and validation**
features and saved checkpoints. No new test analysis, fitting, backbone execution
or GPU job is involved. Full output: runs/multiprotein-diagnostics-v1-verified.

For each protein, split prediction error into its mean error and the remaining
variant errors after subtracting that mean. The two weighted squared-error terms
sum exactly to total MSE. The weights are sequence counts, so unequal group sizes
would also be handled correctly. Centering uses observed labels only to diagnose
errors; it is **not** an available correction for newly generated proteins.

| Model | Training RMSE | Validation RMSE | Validation MSE from parent mean errors | Validation centered-variant RMSE |
|---|---:|---:|---:|---:|
| Constant | 1.366 | 1.447 | 50.6% | 1.017 |
| Ridge | 0.854 | 1.639 | 64.4% | 0.979 |
| MLP seed 0 | 0.903 | 1.627 | 62.1% | 1.002 |
| MLP seed 1 | 0.908 | 1.515 | 55.0% | 1.016 |
| MLP seed 2 | 0.883 | 1.617 | 61.7% | 1.001 |

All errors are in kcal/mol except squared-error shares. For ridge, the parent-mean
error term rises from the constant's 1.0605 to 1.7287 squared units, while the
centered-variant term improves from 1.0342 to 0.9576. Thus **the entire net MSE
disadvantage versus constant comes from parent-mean errors**, partly offset by
better within-parent predictions. This describes the error, not its causal origin.

Ridge's predicted/measured parent means have Spearman 0.994 on the 44 training
parents but -0.261 on the ten validation parents. This is consistent with poor
transfer of learned between-parent patterns; 2,860 training sequences still
represent only 44 distinct parent groups. It does not prove that the backbone
lacks useful features or that more regularization alone will solve the problem.

Examples use the **mean over each parent's 65 candidates**, not the reference
protein's own stability:

| Validation parent | Measured mean | Ridge predicted mean | Prediction bias |
|---|---:|---:|---:|
| 2MC5.pdb | 0.866 | 3.072 | +2.206 |
| 2KVT.pdb | 2.891 | 1.141 | -1.749 |
| 2PFU.pdb | 3.291 | 1.583 | -1.709 |

These biases have opposite signs: a single global offset cannot fix them. The
two largest positive excess-MSE contributions sum to 0.575 of a net 0.592,
but other proteins contribute both positive and negative terms. Removing any
**single** validation parent still leaves ridge worse than constant (remaining
excess MSE ranges 0.290–0.751). Ridge beats constant prediction MSE on only 4/10
validation parents. Do not remove difficult parents to improve the result.

Checks found no feature/identity ordering mismatch or saved-checkpoint mismatch.
All models' normalization buffers match training-only moments within floating-point
tolerance. This rules out those specific bookkeeping failures, not every possible
model or dataset issue. Two synthetic tests verify the decomposition, unequal-size
weighting and leave-one-parent-out calculation. Repeated diagnostics reproduce the
same model metrics exactly. [Verification](../verification/multiprotein-diagnostics-2026-11-10.json).

### Recommended next experiment (not run)

Focus on ridge first. Use five fixed group folds within the existing 44 training
parents to compare stronger regularization (extend beyond the selected boundary
1000) with the constant baseline. Fit normalization separately inside each fold;
keep all variants of a parent together. Select settings by aggregate held-out-fold
prediction error and report parent-mean error and within-parent ranking/selection
alongside it. Retain the existing ten validation parents as a separate transfer
check, rather than selecting settings from the already inspected test table.

This isolates whether the apparent benefit is robust across proteins and whether
shrinking the predictor reduces unreliable between-parent predictions. It is a
proposal, not evidence that shrinkage will work. If it fails, revisit training
protein coverage and representation before training a broad-generation GFlowNet.
Do not introduce measured-parent corrections as a claimed de novo solution.

Reproduce the CPU-only diagnosis into a new directory:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python \
  -m stability_gfn.predictor_diagnostics \
  --run runs/multiprotein-v1 --output runs/multiprotein-diagnostics-recheck
```

The diagnostic saves input hashes and its source alongside the report. Existing
run directories are never overwritten. The initial diagnostic output in
runs/multiprotein-diagnostics-v1 has identical model metrics; the verified output
also includes the source snapshot.
