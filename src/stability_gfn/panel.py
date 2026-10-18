"""Prepare a coverage-selected protein panel without ranking stability outcomes."""

import json
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner

from .artifacts import file_hash, write_json
from .data import audit_table, collapse_parent


def reference_identity(first, second):
    """Global alignment identities / alignment columns, including gap columns.

    This is a reproducible similarity screen, not a guarantee of unrelated folds.
    Equal-score alignment ties use Biopython's first alignment deterministically.
    """
    aligner = PairwiseAligner(
        mode="global",
        match_score=1,
        mismatch_score=-1,
        open_gap_score=-2,
        extend_gap_score=-0.5,
    )
    alignment = aligner.align(first, second)[0]
    return alignment.counts().identities / alignment.length


def split_panel_rows(rows, reference, split_seed, inner_seed, inner_fraction):
    """Reproduce the outer 70/15/15 split, then reserve inner training validation.

    The reference stays in fitting data. No refit on the inner validation labels
    occurs: this keeps the selected checkpoint and its normalization unchanged.
    """
    reference_row = next(row for row in rows if row["sequence"] == reference)
    variants = [row for row in rows if row["sequence"] != reference]
    order = np.random.default_rng(split_seed).permutation(len(variants))
    train_end = int(0.70 * len(variants))
    development_end = int(0.85 * len(variants))
    training = [variants[index] for index in order[:train_end]]
    inner_order = np.random.default_rng(inner_seed).permutation(len(training))
    inner_count = max(1, int(inner_fraction * len(training)))
    partitions = {
        "fit": [reference_row]
        + [training[index] for index in inner_order[inner_count:]],
        "inner_validation": [training[index] for index in inner_order[:inner_count]],
        "development": [variants[index] for index in order[train_end:development_end]],
        "test_labels": [variants[index] for index in order[development_end:]],
    }
    if any(not rows for rows in partitions.values()):
        raise ValueError(
            "Panel requires nonempty fit, inner, development and test splits"
        )
    return partitions


def prepare_panel(source, output, config, synthetic=False):
    """Freeze parent roles and splits before model execution; refuse overwrite."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite panel {output}")
    if not 0 < config["inner_validation_fraction"] < 1:
        raise ValueError("Inner validation fraction must lie between zero and one")
    if config["development_parents"] < 1 or config["confirmation_parents"] < 1:
        raise ValueError("Need development and confirmation parents")
    parents, counts = audit_table(source, config["min_length"], config["max_length"])
    eligible = {}
    for name, rows in parents.items():
        collapsed = collapse_parent(rows)
        clusters = sorted({row["cluster"].strip() for row in rows})
        if (
            collapsed
            and len(collapsed[1]) - 1 >= config["min_variants"]
            and all(clusters)
        ):
            eligible[name] = (collapsed[0], collapsed[1], clusters)
    anchor = config["anchor_parent"]
    if anchor not in eligible:
        raise ValueError(f"Anchor {anchor} is not eligible")
    requested = config["development_parents"] + config["confirmation_parents"]
    selected = [anchor]
    rejected = []
    for name in sorted(eligible, key=lambda key: (-len(eligible[key][1]), key)):
        if len(selected) == requested:
            break
        if name == anchor:
            continue
        reference, _, clusters = eligible[name]
        shared = [
            other for other in selected if set(clusters) & set(eligible[other][2])
        ]
        similar = {
            other: reference_identity(reference, eligible[other][0])
            for other in selected
        }
        if shared or max(similar.values()) > config["maximum_reference_identity"]:
            rejected.append(
                {"parent": name, "shared_cluster_with": shared, "identity": similar}
            )
        else:
            selected.append(name)
    if len(selected) != requested:
        raise ValueError(
            f"Only {len(selected)} parents satisfy panel separation; need {requested}"
        )
    output.mkdir(parents=True)
    entries = []
    for index, name in enumerate(selected):
        reference, rows, clusters = eligible[name]
        role = (
            "development" if index < config["development_parents"] else "confirmation"
        )
        identifier = f"parent-{index:02d}"
        folder = output / identifier
        folder.mkdir()
        partitions = split_panel_rows(
            rows,
            reference,
            config["split_seed"],
            config["inner_seed"],
            config["inner_validation_fraction"],
        )
        for split, values in partitions.items():
            write_json(folder / f"{split}.json", values)
        entry = {
            "id": identifier,
            "parent": name,
            "role": role,
            "reference": reference,
            "clusters": clusters,
            "valid_variants": len(rows) - 1,
            "previously_inspected_parent": name == anchor,
            "split_counts": {key: len(value) for key, value in partitions.items()},
            "file_sha256": {
                f"{key}.json": file_hash(folder / f"{key}.json") for key in partitions
            },
        }
        entries.append(entry)
        write_json(folder / "manifest.json", entry)
    manifest = {
        "synthetic": synthetic,
        "config": config,
        "source": str(Path(source).resolve()),
        "source_sha256": file_hash(source),
        "audit_counts": counts,
        "eligible_parents": len(eligible),
        "parents": entries,
        "selection": "anchor, then descending valid coverage/name; disjoint WT clusters and global identity screen; no outcome-value ranking",
        "rejected_before_panel_filled": rejected,
        "alignment": {
            "library": "Biopython 1.86",
            "mode": "global",
            "match": 1,
            "mismatch": -1,
            "gap_open": -2,
            "gap_extend": -0.5,
            "identity_denominator": "all alignment columns",
        },
        "pairwise_reference_identity": [
            {
                "first": first,
                "second": second,
                "identity": reference_identity(eligible[first][0], eligible[second][0]),
            }
            for index, first in enumerate(selected)
            for second in selected[index + 1 :]
        ],
        "label_policy": "Preparation uses label availability and replicate means, not values for selection. Diagnostic runner reads only development parents' fit/inner_validation/development files. Confirmation and all test labels are reserved.",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def read_development_parent(panel, entry):
    """Verify and load only allowed partitions; never open reserved test labels."""
    if entry["role"] != "development":
        raise ValueError("Confirmation parents are reserved")
    result = {}
    for split in ("fit", "inner_validation", "development"):
        path = Path(panel) / entry["id"] / f"{split}.json"
        if file_hash(path) != entry["file_sha256"][path.name]:
            raise ValueError(f"Panel checksum mismatch: {path}")
        result[split] = json.loads(path.read_text())
    return result
