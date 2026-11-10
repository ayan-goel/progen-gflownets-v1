"""Outcome-independent parent grouping and a bounded multi-protein stability panel."""

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import Bio
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

from .artifacts import file_hash, write_json
from .data import audit_table, collapse_parent, finite_number


def identity(left, right, aligner):
    """Identical aligned residues divided by longer full sequence length.

    Use the first optimal global BLOSUM62 alignment in sorted parent order;
    explicit gap scores and a pinned Biopython version make ties reproducible.
    """
    alignment = aligner.align(left, right)[0]
    return alignment.counts().identities / max(len(left), len(right))


def group_parents(parents, threshold):
    """Connected components of source cluster, shared sequence or reference identity edges."""
    names = sorted(parents)
    roots = {name: name for name in names}

    def find(name):
        while roots[name] != name:
            roots[name] = roots[roots[name]]
            name = roots[name]
        return name

    def union(left, right):
        a, b = sorted([find(left), find(right)])
        roots[b] = a

    clusters, sequences = {}, {}
    for name in names:
        for cluster in parents[name]["clusters"]:
            if cluster.lower() not in {"", "nan", "none"}:
                if cluster in clusters:
                    union(name, clusters[cluster])
                clusters[cluster] = name
        for row in parents[name]["rows"]:
            sequence = row["sequence"]
            if sequence in sequences:
                union(name, sequences[sequence])
            sequences[sequence] = name
    aligner = PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-10,
        extend_gap_score=-0.5,
    )
    similarities = []
    for left, right in combinations(names, 2):
        value = identity(
            parents[left]["reference"], parents[right]["reference"], aligner
        )
        similarities.append((left, right, value))
        if value >= threshold:
            union(left, right)
    groups = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    return groups, similarities


def choose_panel(parents, groups, exposure, config):
    """Exclude known-exposed components, then choose one parent per component by identity hash."""
    exposed_names = set(exposure["known_parent_names"])
    exposed_sequences = set(exposure["known_sequence_sha256"])
    eligible, excluded = [], []
    for group, names in sorted(groups.items()):
        exposed = [
            name
            for name in names
            if name in exposed_names
            or any(
                hashlib.sha256(row["sequence"].encode()).hexdigest()
                in exposed_sequences
                for row in parents[name]["rows"]
            )
        ]
        if exposed:
            excluded.append({"group": group, "matching_parents": exposed})
        else:
            # Parent choice and group budget do not use label magnitudes or coverage ranking.
            selected = min(
                names,
                key=lambda name: hashlib.sha256(
                    f"{config['selection_seed']}:{name}".encode()
                ).hexdigest(),
            )
            eligible.append((group, selected))
    eligible.sort(
        key=lambda pair: hashlib.sha256(
            f"{config['selection_seed']}:group:{pair[0]}".encode()
        ).hexdigest()
    )
    if len(eligible) < config["parent_count"]:
        raise ValueError(
            f"Only {len(eligible)} unexposed components; requested {config['parent_count']}"
        )
    selected = eligible[: config["parent_count"]]
    order = np.random.default_rng(config["split_seed"]).permutation(len(selected))
    train_end = int(0.70 * len(selected))
    validation_end = int(0.85 * len(selected))
    roles = {}
    for rank, index in enumerate(order):
        group, name = selected[index]
        roles[name] = (
            "train"
            if rank < train_end
            else "validation"
            if rank < validation_end
            else "test",
            group,
        )
    partitions = {role: [] for role in ["train", "validation", "test"]}
    for name, (role, group) in sorted(roles.items()):
        parent = parents[name]
        reference = parent["reference"]
        reference_row = next(
            row for row in parent["rows"] if row["sequence"] == reference
        )
        variants = sorted(
            [row for row in parent["rows"] if row["sequence"] != reference],
            key=lambda row: hashlib.sha256(
                f"{config['selection_seed']}:{name}:{row['sequence']}".encode()
            ).hexdigest(),
        )
        for row in [reference_row] + variants[: config["variants_per_parent"]]:
            partitions[role].append(
                {
                    **row,
                    "parent": name,
                    "group": group,
                    "is_reference": row["sequence"] == reference,
                }
            )
    return partitions, roles, excluded, len(eligible)


def prepare_multiprotein(source, output, exposure_path, config, synthetic=False):
    """Audit/filter measurements, freeze group identities and write sealed evaluation labels."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (
        file_hash(source) != config["source_sha256"]
        or file_hash(exposure_path) != config["exposure_sha256"]
    ):
        raise ValueError("Source or exposure inventory changed")
    raw, counts = audit_table(source, config["min_length"], config["max_length"])
    parents, quality_counts = {}, Counter()
    for name, rows in sorted(raw.items()):
        kept = []
        for row in rows:
            width = finite_number(row["quality"].get("deltaG_95CI"))
            if width is None or width < 0 or width > config["maximum_ci_width"]:
                quality_counts["excluded_ci_rows"] += 1
            else:
                kept.append(row)
        collapsed = collapse_parent(kept)
        if (
            collapsed is not None
            and len(collapsed[1]) - 1 >= config["variants_per_parent"]
        ):
            parents[name] = {
                "reference": collapsed[0],
                "rows": collapsed[1],
                "clusters": sorted({row["cluster"] for row in rows}),
            }
    del raw
    groups, similarities = group_parents(parents, config["identity_threshold"])
    exposure = json.loads(Path(exposure_path).read_text())
    partitions, roles, excluded, available = choose_panel(
        parents, groups, exposure, config
    )
    all_sequences = [row["sequence"] for rows in partitions.values() for row in rows]
    if len(set(all_sequences)) != len(all_sequences):
        raise ValueError("Duplicate sequences remain in the selected panel")
    cross = [
        value
        for left, right, value in similarities
        if left in roles and right in roles and roles[left][0] != roles[right][0]
    ]
    if cross and max(cross) >= config["identity_threshold"]:
        raise ValueError("Cross-partition identity exceeds threshold")
    output.mkdir(parents=True)
    for role in ["train", "validation"]:
        write_json(output / f"{role}.json", partitions[role])
    identities = [
        {key: row[key] for key in ["sequence", "parent", "group", "is_reference"]}
        for row in partitions["test"]
    ]
    write_json(output / "test_identities.json", identities)
    write_json(output / "test_labels.json", partitions["test"])
    write_json(output / "groups.json", groups)
    write_json(
        output / "parent_inventory.json",
        {
            name: {
                "reference": parent["reference"],
                "clusters": parent["clusters"],
                "valid_sequences": len(parent["rows"]),
            }
            for name, parent in parents.items()
        },
    )
    write_json(output / "excluded_exposure.json", excluded)
    write_json(
        output / "cross_parent_identity.json",
        [
            {"left": left, "right": right, "identity": value}
            for left, right, value in similarities
            if left in roles and right in roles
        ],
    )
    files = [
        "train.json",
        "validation.json",
        "test_identities.json",
        "test_labels.json",
        "groups.json",
        "parent_inventory.json",
        "excluded_exposure.json",
        "cross_parent_identity.json",
    ]
    manifest = {
        "synthetic": synthetic,
        "settings": config,
        "source": str(Path(source).resolve()),
        "exposure_inventory": str(Path(exposure_path).resolve()),
        "scope": "Development; metadata exclusion is not proof of historically untouched outcomes or pretrained-model independence",
        "audit_counts": counts,
        "quality_counts": dict(quality_counts),
        "eligible_parents": len(parents),
        "components": len(groups),
        "unexposed_components": available,
        "selected_parent_roles": {
            name: {"split": role, "group": group}
            for name, (role, group) in roles.items()
        },
        "split_counts": {role: len(rows) for role, rows in partitions.items()},
        "parent_counts": dict(Counter(role for role, group in roles.values())),
        "maximum_cross_partition_reference_identity": max(cross) if cross else None,
        "biopython": Bio.__version__,
        "file_sha256": {name: file_hash(output / name) for name in files},
    }
    write_json(output / "manifest.json", manifest)
    return manifest
