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
prediction performance is unknown until the user runs the experiment. No real
training, backbone inference or GPU work was performed in implementation.

Source-access note (10 November): the journal full-text fetch was blocked by its
redirect. The assay discussion here uses the indexed primary-paper abstract/methods
excerpt and the author release listing, not a new full-paper review. The Biopython
1.86 alignment documentation was directly accessible.
