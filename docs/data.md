# Mega-scale data contract and audit

## Source and verification

Use [Zenodo release 7992926](https://zenodo.org/records/7992926), member
`Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`.
The CSV is **697,658,024 bytes** with SHA-256:

`0c7b2bf1f162122d83d5bb39b901cc446e436b5643684ca268f934be145d8398`

The range downloader verifies the pinned ZIP central directory, member name,
uncompressed length and CRC32 `9ac87977`. It fetches about 210 MB compressed
instead of the entire 1 GB archive. The published whole-archive MD5 is recorded
but **not verified** by a member-only download. The machine-readable record is
[data/raw/download_manifest.json](../data/raw/download_manifest.json).

## Filtering and interpretation

- Use `aa_seq`, excluding assay linkers in `aa_seq_full`.
- Require canonical residues, length 40–72 and finite numeric `dG_ML`.
- Treat missing, censored/non-numeric and non-finite labels as absent; never zero.
- `dG_ML` is the absolute-stability regression target in kcal/mol, with higher
  values more stable. Preserve reported `ddG_ML` separately; do not silently use
  it as the absolute label. Predicted improvement uses our own reference subtraction.
- Map parent identity with `WT_name`, retain `WT_cluster`, and require one unique
  valid wild-type sequence (`mut_type=wt`) per eligible parent.
- Keep same-length sequences with at most one substitution. Average valid
  replicate labels for identical sequences within a parent **before splitting**;
  preserve source names, counts, label ranges and available quality fields.

The [paper](https://www.nature.com/articles/s41586-023-06328-6) describes assay and
filtering limitations. The ML labels do not remove every source of measurement
noise, and the study concerns folding stability rather than melting temperature.

## Audited result — 14 October 2026

| Item | Count |
|---|---:|
| Input rows | 776,298 |
| Missing/non-numeric/non-finite dG_ML | 100,831 |
| Outside the retained length range, after label filtering | 21,638 |
| Retained canonical ML rows | 653,829 |
| Eligible parents with at least 300 valid single substitutions | 413 |

These sequential exclusion counts sum to the input count. No excluded row is
reintroduced into the task.

Choose the parent with the largest number of valid unique single substitutions,
breaking ties by parent name. This favors coverage and can favor longer domains;
it does not select by stability magnitude. The chosen reference is **1O6X.pdb**,
72 residues, cluster `129`, with 1,357 valid single substitutions out of 1,368
possible (99.196% coverage), plus the reference itself.

The measured reference stability is 4.17963218 kcal/mol. The prepared task is
[local-v1/manifest.json](../data/processed/local-v1/manifest.json). The audit is
setup only; no real scorer or guidance model was trained on these measurements.

## Immutable local split and held-out support

Sort unique variants, shuffle with NumPy's default generator seed 42, allocate
floor(70%)/remaining-to-floor(85%)/remainder, and add the reference to training:

| Split | Sequences | Usage |
|---|---:|---|
| Training | 950 | Scorer fitting, scaling, known reference |
| Validation | 204 | Scorer selection and validation gate |
| Test candidate identities | 204 | Legal generation support |
| Test labels | 204 | Evaluation only after model fitting/selection |

Train and validation include 64 replicate observations beyond their unique
sequence counts; the largest replicate range in these two splits is about
0.6724 kcal/mol. Replicate averaging is a declared policy, not noise elimination.
Training-label population mean/std are approximately 3.359656/1.177569 kcal/mol.
The implementation recomputes the scale from training data rather than hardcoding it.

`candidates.json` contains identities only. `test_labels.json` is separate;
the pipeline hashes it for input identity but does not parse its values until
post-training evaluation. Hashes and immutable splits are checked before model work.
The reference is not a test candidate. The initial support therefore contains
exactly one mutation per candidate despite the environment's general at-most-one rule.

Measurement coverage is 100% by construction in this retrospective benchmark.
Absence from training and distance one from the reference are also largely fixed
by this support, so they are descriptive checks, not evidence of open-ended novelty.
The broader neighborhood and unseen-domain studies remain later milestones.

## Confirmation identity isolation — 16 October 2026

The original panel's five reserved parents and all split hashes remain unchanged.
A separate [identity-only support manifest](../data/processed/confirmation-support-v1/manifest.json)
reconstructs their 203/202/201/200/196 test identities from the original source's
sequence/parent identities and finite-label availability, using the same sorted
unique single-mutant list and seed-42 split. This preparation does not open panel
test-label files or retain/rank stability values. It does not create new splits.

The confirmation runner reads only fit/inner rows and candidate identities during
learning, leaving outer development rows unused. All fits must be complete and
hashed before test outcomes are opened. At release, label hashes and exact candidate
identity equality are checked. See [protocol](experiments/confirmation_plan_2026-10-16.md).
