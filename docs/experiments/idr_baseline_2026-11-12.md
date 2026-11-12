# Frozen-ProGen IDR baseline

**12 November 2026 — user run complete; results and CPU verification below.**
This is generation and evaluation, with no GFN or ProGen training.

## Question and scope

Does frozen ProGen2-medium produce a useful proportion of plausibly disordered
sequences, either without an amino-acid prompt or continued from a natural IDR
prefix? This precedes charge-profile guidance. We assess predicted disorder and
sequence diversity, not verified biological function or classifier generalization.

## Reference panel and provenance

The [DisProt API](https://disprot.org/api/search?format=json) supplied 3,337 entries.
The local 12 November snapshot is `data/raw/disprot-2026-11-12.json`; its SHA-256 is
`1fdfce26a3ea9b626e63559f36abd86e023b50b612dc2c8756acbb08edead140`.
This is a retrieval-date snapshot, not a claimed versioned database release.

Require a eukaryotic parent with a UniRef50 identifier, a validated, published
disorder annotation (IDPO:0000002, EXP/IDA evidence), and one annotation covering
the entire 80-residue window. Restrict evidence to NMR (ECO:0006165, 0001238,
0006198) or far-UV circular dichroism (ECO:0006204); require the structural-state
consensus to include the window as D. Missing structure density alone is excluded.
Do not splice unrelated annotation intervals to obtain a sufficiently long window.

Pick one annotation per parent by SHA-256 rank with seed 20261112 and use its central
80 residues. Reject noncanonical windows. Rank parents by the same identity-only
rule and greedily retain 16 with different accessions/UniRef50 identifiers and
pairwise window identity below 50%. Identity uses a global alignment (match 2,
mismatch −1, gap-open −3, extension −0.5) with matches divided by the longer length.
These are engineering choices to limit redundancy, not a guarantee of distinct
families or unseen pretraining data. First eight supply prompts; the remaining eight
are reference-only. No predictor scores select windows, parents or roles.

Annotation audit excluded DP00132 before freezing the panel: its cited evidence is
explicitly calcium-free apo calsequestrin, an unsuitable intrinsic-disorder prompt
for this question. There are 109 eligible parents after filtering/exclusion.
Other annotations still depend on their assay context: for example, DP00030 concerns
unphosphorylated AF1. Statements, evidence codes, citations and available condition
fields are retained verbatim in each record. Blank condition metadata does not imply
condition independence. Cropped windows are annotation-supported fragments, not
individually measured synthetic 80-mers.

The [prepared manifest](../../data/processed/idr-reference-v1/manifest.json) and
[reference records](../../data/processed/idr-reference-v1/references.json) freeze
identities, coordinates, sequences, roles and source evidence. The run verifies their
hashes and copies them. Reference-only does not mean unseen by metapredict or ProGen.
No new predictor is fitted and there is no train/test accuracy claim.

## Fixed generation choices

| Choice | Value and rationale |
|---|---|
| Backbone | Existing local ProGen2-medium checkpoint; frozen, evaluation mode, float32. Reuse the tested adapter and avoid a model-change confound. |
| Arms | Unprompted direction marker only; prompted direction marker plus 16 natural residues. This tests whether minimal IDR context helps. |
| Total length | 80 canonical residues in both arms: a bounded first feasibility test. |
| Main evaluation region | Residues 17–80 (64 residues) in both arms. All are generated; natural prompt residues never count toward primary success. |
| Budget | Seeds 0/1/2, 64 samples per arm per seed: 384 generated sequences total. Prompted arm has eight continuations for each of eight prompts per seed. |
| Seed mapping | Dedicated generator seed is twice the run seed plus arm index (unprompted 0, prompted 1). Preserve sample ordering and batch size for exact replay. |
| Decoding | Temperature 1; no top-k, top-p, repetition penalty, reward guidance or rejection filtering. |
| Support | At each step normalize over the 20 canonical residues; suppress stop/special/noncanonical tokens. Stop deterministically at length 80. |
| Batch size | 8, using key/value caching; conservative memory and sufficient throughput for a small baseline. |
| CPU threads | 1 for predictable evaluation and development tests. |

This is a locally canonical-masked fixed-length prior, not unrestricted ProGen
globally conditioned on length or disorder. Save conditional log probability over
generated tokens and mean probability mass assigned to canonical residues before
masking. Prompted log probabilities condition on the supplied prefix and sum over
64 sampled residues; unprompted sums over 80. Do not compare their raw sums as if
length and conditioning were identical.

## Automatic evaluation

Use **metapredict package 3.0.2, V3 network** as primary scorer and its **V2 network**
as a related-model sensitivity check. Both run explicitly on CPU, after user-run
generation. Scores are unrounded and retained per residue. The pinned release
works with the existing NumPy 1.26.4/Torch 2.6 environment; newer releases require a
different NumPy baseline. No core dependency was upgraded. See
[package release](https://pypi.org/project/metapredict/3.0.2/) and
[author API documentation](https://metapredict.readthedocs.io/en/latest/usage/using-in-python.html).

Primary success means **at least 80% of evaluated residues have V3 score >0.5**.
The 80% rule is a declared engineering screen, not an experimentally validated IDR
definition. Report mean scores and fractions as well as this binary summary.
V2 is not an independent verifier; [IUPred3's download](https://iupred3.elte.hu/download_new)
requires an academic request and was not obtained automatically. Independent
disorder confirmation remains future work, not a hidden prerequisite for this run.

For each sequence save full-context residue scores, the final-64-residue summary,
and predictions recomputed on the isolated tail. The latter tests whether success
depends strongly on prompt/context. Full-sequence results are secondary. Evaluate
natural reference windows under the same cropping conventions.

Descriptors include amino-acid composition, signed net charge per residue,
charged-residue fraction, centered 9-residue charge profiles, Shannon composition
entropy in bits, longest identical-residue run and repeated-trimer fraction. Assign
K/R +1, D/E −1 and all other residues zero; ignore termini/PTMs. Edge windows use
only available residues. These exact descriptors are not pH-specific charge or
biological function. The conventions follow the
[SPARROW description](https://idptools-sparrow.readthedocs.io/en/latest/api_guides/protein.html);
simple charge/count arithmetic is implemented locally without introducing its
additional dependency stack. No κ, hydropathy scale or structural predictor is added.

Aggregate raw samples by arm, seed and prompt. Report unique-tail fraction, mean
pairwise normalized Hamming distance, homopolymer flags and nearest reference-tail
global identity. Novelty is only relative to these 16 reference tails; exact matches
and similarities do not establish novelty relative to training data. All raw
sequences count; deduplication is not used to improve pass rates.

## Decision rules and limits

A setup is worth considering for charge-guidance planning if **each seed** has
at least 25% primary tail success, at least 80% unique tails, and at most 10% tails
with an identical-residue run of length eight or longer. These practical thresholds
screen yield/collapse; they do not certify biology. Inspect all eight prompt results
so one easy prefix cannot conceal failures elsewhere. Prompt replicates are not
independent protein families, and natural IDRs can legitimately be repetitive.

If V2 differs sharply or isolated-tail success collapses, review context/scorer
dependence before proceeding. If both arms have poor yield or severe repetition,
investigate the prior/prompt setup and an IDiom comparator before GFN training.
There is no automatic model promotion, reward tuning or selection of favorable
seeds. The natural panel has only positives and cannot estimate false-positive rates.

## Running and artifacts

From the project root, after activating `.venv` and choosing CUDA_VISIBLE_DEVICES:

```bash
scripts/run.sh idr-baseline --output runs/idr-baseline-v1 --device cuda
```

The run saves configuration, reference snapshots, source/model/scorer hashes,
package versions, `samples.jsonl`, `samples.fasta`, generation timing/log probabilities,
`sequence_metrics.json`, `metrics.json`, `report.md` and completion/failure status.
It refuses existing output directories. Samples are flushed batchwise; if generation
completes but scoring fails, recover scoring without generating again:

```bash
scripts/run.sh evaluate-idr --run runs/idr-baseline-v1 --output runs/idr-baseline-v1-evaluation
```

CPU preflight: `scripts/run.sh idr-baseline --check-only`. Reproduce preparation
from the frozen raw snapshot using `prepare-idr --output` with a new directory.
The default user command needs no network access or additional setup here.

## Verification before handoff

Five new CPU tests verify real tiny-ProGen cached probabilities against teacher
forcing, prefix exclusion, reference evidence boundaries, tamper detection and a
tiny end-to-end run with actual V3/V2 scoring plus saved-run reevaluation. Full
CPU suite: 61 tests pass. These are synthetic correctness tests, not biological
results. The actual reference panel was prepared/audited without predictor selection.
The assistant performed no pretrained ProGen inference, real generation or GPU work.


## Completed results — 12 November 2026

The user completed `runs/idr-baseline-v1`: 384 generated sequences plus 16 natural
reference evaluations, in 177.645 seconds (172.015 seconds generation). Status is
complete and `fixture=false`. No GFN training or ProGen fine-tuning occurred.
[Saved report](../../runs/idr-baseline-v1/report.md) and
[metrics](../../runs/idr-baseline-v1/metrics.json) retain the original outputs.

| Setup | V3 context-tail pass | V3 isolated-tail pass | V2 context-tail pass | Unique tails | Homopolymer ≥8 |
|---|---:|---:|---:|---:|---:|
| Unprompted | 55/192 (28.6%) | 54/192 (28.1%) | 50/192 (26.0%) | 192/192 | 1/192 |
| IDR prompted | 125/192 (65.1%) | 129/192 (67.2%) | 124/192 (64.6%) | 187/192 | 4/192 |
| Natural references | 15/16 (93.8%) | 15/16 (93.8%) | 14/16 (87.5%) | 16/16 | 2/16 |

Prompting raises the primary pass rate by 36.5 percentage points, about 2.27 times
the unprompted rate. Both arms satisfy the declared arm-by-seed engineering screen.
Prompted V3 pass rates across seeds are 70.3%, 70.3%, 54.7%; unprompted rates are
29.7%, 28.1%, 28.1%. Prompted per-seed uniqueness is 96.9%, 98.4%, 100%; no arm/seed
exceeds 3.125% homopolymer flags. These are three sampling seeds over the same
panel, not three independent biological datasets.

### Prompt dependence and repetition

Each prompt contributes 24 continuations over three seeds:

| DisProt ID | Reference | V3 passes | Unique tails |
|---|---|---:|---:|
| DP01299 | LEA18 | 11/24 | 24/24 |
| DP00187 | Prion protein N-terminal region | 15/24 | 24/24 |
| DP00181 | RNA polymerase II RPB1 | 24/24 | 19/24 |
| DP00092 | Calcineurin | 6/24 | 24/24 |
| DP00510 | Nuclear protein 1 | 18/24 | 24/24 |
| DP01076 | NUP100 | 18/24 | 24/24 |
| DP01538 | Polyadenylate-binding protein | 16/24 | 24/24 |
| DP01109 | hnRNPA2/B1 | 17/24 | 24/24 |

RPB1's perfect disorder score accompanies repeated multi-residue motifs, 79.2%
unique tails, low composition entropy (2.223 bits) and mean nearest-reference-tail
identity 80.7%. Its motif repetition is missed by the homopolymer flag: repeated
motifs need not contain eight identical consecutive residues. Retain this prompt
as a diagnostic; do not present its 100% yield as an ideal generation result.
DP01538 also has 3/24 homopolymer flags, hidden by the arm-level average.

A **post-hoc sensitivity calculation**, excluding RPB1 only to assess its influence,
still gives 101/168 (60.1%) V3 passes and 100% unique tails across the remaining
seven prompts. The main result remains all 192 samples; no output or inclusion
rule was changed. Prompting's benefit is not solely the repetitive RPB1 case.
The weakest prompt reaches only 25%, so IDR-prefix continuation is not uniformly
reliable across contexts.

### What this establishes and the next decision

Frozen ProGen has a usable predicted-IDR yield under this short, fixed-length
sampling setup. Prompting substantially increases yield without an overall
collapse in uniqueness. Isolated-tail success stays high, so the aggregate result
is not simply the disorder scorer relying on the supplied prefix. V2 rates are
similar, but sample-level V3/V2 pass classifications disagree on 23/192 prompted
and 11/192 unprompted sequences; V2 is not independent biological confirmation.

No generated tail exactly matches any of the 16 reference tails. This limited
comparison does not establish novelty against natural proteins or pretraining.
Natural-reference pass rates are positive controls, not false-positive estimates.
No new biological property or experimental disorder has been demonstrated.

**Recommended next step:** specify a small charge-profile guidance pilot on the
prompted prior. Freeze target profiles, tolerances, reward scaling and budgets
before training; use the saved charge profiles to establish plausible ranges and
label this as development-data-informed design. Compare unguided sampling,
equal-budget generate-and-select and GFN guidance under identical prefixes and
sequence support. Report charge matching alongside isolated-tail disorder,
per-prompt diversity and motif repetition. Preserve all eight prompt results;
do not choose a best-performing prefix from this run as independent confirmation.
A direct-design comparator remains part of the broader plan. Do not interpret
passing this baseline as evidence that GFN guidance already works on open generation.

### Verification

CPU-only inspection checked generation/sample/config/reference integrity hashes,
current source hashes against run provenance, all 384 sample records against the
400 scored records, canonical sequence lengths/prefix placement, descriptor
recalculation, both scorers' saved score-to-threshold arithmetic, and every arm/seed
aggregate. All checks passed. This verifies saved-output consistency, not a rerun
of the disorder networks or ProGen. Original run files remain unchanged; the
assistant performed no generation, training or GPU execution during analysis.
