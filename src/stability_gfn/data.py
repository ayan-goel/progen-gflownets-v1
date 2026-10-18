"""Audit Mega-scale ML labels and create leakage-aware local-variant manifests."""

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .artifacts import file_hash, write_json
from .environment import AMINO_ACIDS, hamming

REQUIRED_COLUMNS = {
    "name",
    "aa_seq",
    "WT_name",
    "WT_cluster",
    "mut_type",
    "dG_ML",
    "ddG_ML",
}


def finite_number(value):
    """Missing/censored strings are absent labels, never zero or negative examples."""
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def audit_table(path, min_length=40, max_length=72):
    """Stream the release and retain only valid absolute-stability observations."""
    parents = defaultdict(list)
    counts = Counter()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError(
                f"Missing release columns: {sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))}"
            )
        for row in reader:
            counts["input_rows"] += 1
            sequence = row["aa_seq"].strip()
            label = finite_number(row["dG_ML"])
            if label is None:
                counts["missing_or_censored_dG_ML"] += 1
                continue
            if not sequence or not set(sequence) <= set(AMINO_ACIDS):
                counts["noncanonical_sequence"] += 1
                continue
            if not min_length <= len(sequence) <= max_length:
                counts["length_outside_range"] += 1
                continue
            parent = row["WT_name"].strip()
            if not parent:
                counts["missing_parent"] += 1
                continue
            parents[parent].append(
                {
                    "source_name": row["name"],
                    "sequence": sequence,
                    "stability": label,
                    "reported_ddg": finite_number(row["ddG_ML"]),
                    "is_reference": row["mut_type"].strip().lower() == "wt",
                    "cluster": row["WT_cluster"],
                    "quality": {
                        key: row[key]
                        for key in ("deltaG_95CI", "Stabilizing_mut")
                        if key in row
                    },
                }
            )
            counts["retained_ml_rows"] += 1
    return dict(parents), dict(counts)


def collapse_parent(rows):
    """Aggregate replicate labels before splitting; exclude ambiguous reference mappings."""
    references = {row["sequence"] for row in rows if row["is_reference"]}
    if len(references) != 1:
        return None
    reference = next(iter(references))
    grouped = defaultdict(list)
    for row in rows:
        if (
            len(row["sequence"]) == len(reference)
            and hamming(row["sequence"], reference) <= 1
        ):
            grouped[row["sequence"]].append(row)
    collapsed = []
    for sequence, replicates in sorted(grouped.items()):
        labels = [row["stability"] for row in replicates]
        collapsed.append(
            {
                "sequence": sequence,
                "stability": float(np.mean(labels)),
                "replicate_count": len(replicates),
                "replicate_label_range": max(labels) - min(labels),
                "source_names": sorted(row["source_name"] for row in replicates),
                "reported_ddg": [row["reported_ddg"] for row in replicates],
                "quality": [row["quality"] for row in replicates],
                "mutations": [
                    index + 1
                    for index, (left, right) in enumerate(zip(sequence, reference))
                    if left != right
                ],
            }
        )
    return reference, collapsed


def prepare_local_task(
    source,
    destination,
    parent=None,
    split_seed=42,
    min_variants=300,
    min_length=40,
    max_length=72,
    synthetic=False,
):
    """Choose by coverage, partition unique sequences, and isolate test labels on disk."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared task {destination}")
    parents, counts = audit_table(source, min_length, max_length)
    eligible = {}
    for name, rows in parents.items():
        collapsed = collapse_parent(rows)
        if collapsed and len(collapsed[1]) - 1 >= min_variants:
            eligible[name] = collapsed
    if not eligible:
        raise ValueError("No reference has enough valid single substitutions")
    if parent is None:
        # Coverage then name; no stability-value ranking is used to pick a task.
        parent = min(eligible, key=lambda name: (-len(eligible[name][1]), name))
    if parent not in eligible:
        raise ValueError(f"Requested reference {parent!r} is not eligible")
    reference, rows = eligible[parent]
    reference_row = next(row for row in rows if row["sequence"] == reference)
    variants = [row for row in rows if row["sequence"] != reference]
    order = np.random.default_rng(split_seed).permutation(len(variants))
    train_end = int(0.70 * len(variants))
    validation_end = int(0.85 * len(variants))
    partitions = {
        "train": [reference_row] + [variants[index] for index in order[:train_end]],
        "validation": [variants[index] for index in order[train_end:validation_end]],
        "test": [variants[index] for index in order[validation_end:]],
    }
    if any(not rows for rows in partitions.values()):
        raise ValueError("All three splits must be nonempty")
    destination.mkdir(parents=True)
    write_json(destination / "train.json", partitions["train"])
    write_json(destination / "validation.json", partitions["validation"])
    # Generation may know test sequence identities, but only evaluation reads labels.
    write_json(
        destination / "candidates.json",
        sorted(row["sequence"] for row in partitions["test"]),
    )
    write_json(destination / "test_labels.json", partitions["test"])
    manifest = {
        "synthetic": synthetic,
        "source": str(Path(source).resolve()),
        "source_sha256": file_hash(source),
        "release": "https://zenodo.org/records/7992926",
        "label_column": "dG_ML",
        "label_units": "kcal/mol",
        "higher_is_more_stable": True,
        "sequence_column": "aa_seq",
        "parent": parent,
        "reference": reference,
        "reference_stability": reference_row["stability"],
        "clusters": sorted({row["cluster"] for row in parents[parent]}),
        "mutation_budget": 1,
        "support": "held-out measured single-substitution identities",
        "split_seed": split_seed,
        "split_fractions": [0.70, 0.15, 0.15],
        "split_counts": {name: len(rows) for name, rows in partitions.items()},
        "audit_counts": counts,
        "eligible_parents": len(eligible),
        "unique_local_sequences": len(variants) + 1,
        "replicate_policy": "mean dG_ML per identical sequence within parent before splitting",
        "generalization_claim": "within-parent local mutation design; not unseen-domain generalization",
        "file_sha256": {
            name: file_hash(destination / name)
            for name in [
                "train.json",
                "validation.json",
                "candidates.json",
                "test_labels.json",
            ]
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest
