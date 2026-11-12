"""Extend only training coverage using the frozen multi-protein grouping inventory."""

import hashlib
import json
from collections import Counter

from .artifacts import file_hash, write_json
from .data import audit_table, collapse_parent, finite_number
from .pipeline import project_path


def read_source(config):
    """Verify original run/data artifacts without reading test measurements."""
    source = project_path(config["source_run"])
    required = {
        "config.json",
        "data_manifest.json",
        "features.pt",
        "ridge-seed-0.pt",
        "fitting.json",
    }
    if not required <= set(config["source_sha256"]):
        raise ValueError("Missing source hashes")
    for name, digest in config["source_sha256"].items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Original source changed: {name}")
    original = json.loads((source / "config.json").read_text())
    manifest = json.loads((source / "data_manifest.json").read_text())
    data = project_path(original["data"])
    if file_hash(data / "manifest.json") != original["manifest_sha256"]:
        raise ValueError("Original manifest changed")
    files = {}
    for name in [
        "train.json",
        "validation.json",
        "test_identities.json",
        "groups.json",
        "parent_inventory.json",
        "excluded_exposure.json",
    ]:
        if file_hash(data / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Original prepared data changed: {name}")
        files[name] = json.loads((data / name).read_text())
    return source, original, manifest, files


def addition_order(groups, roles, exclusions, seed):
    """Choose unused components and representatives by identity hash, never labels."""
    forbidden = {role["group"] for role in roles.values()}
    forbidden.update(row["group"] for row in exclusions)
    available = sorted(
        set(groups) - forbidden,
        key=lambda group: hashlib.sha256(f"{seed}:group:{group}".encode()).hexdigest(),
    )
    return [
        {
            "group": group,
            "parent": min(
                groups[group],
                key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
            ),
        }
        for group in available
    ]


def validate_training_sets(base, additional, held, training_sets, variants_per_parent):
    """Require nested parent sets, equal row counts and disjoint evaluation identities."""
    all_rows = base + additional
    sequences = [row["sequence"] for row in all_rows]
    if len(sequences) != len(set(sequences)) or set(sequences) & {
        row["sequence"] for row in held
    }:
        raise ValueError("Training/evaluation sequence overlap or duplicate")
    if {row["group"] for row in all_rows} & {row["group"] for row in held}:
        raise ValueError("Training/evaluation group overlap")
    groups = {}
    for row in all_rows:
        groups.setdefault(row["parent"], set()).add(row["group"])
    if any(len(value) != 1 for value in groups.values()):
        raise ValueError("Parent spans multiple groups")
    if len({next(iter(value)) for value in groups.values()}) != len(groups):
        raise ValueError("Expected one training parent per component")
    counts = Counter(row["parent"] for row in all_rows)
    if set(counts.values()) != {variants_per_parent + 1}:
        raise ValueError("Unequal sequence counts per training parent")
    previous = {row["parent"] for row in base}
    for index, (count, parents) in enumerate(training_sets.items()):
        current = set(parents)
        if len(current) != int(count) or len(parents) != len(current):
            raise ValueError("Training-set parent count mismatch")
        if (index == 0 and current != previous) or not previous <= current:
            raise ValueError("Training sets must extend the unchanged baseline")
        if not current <= set(counts):
            raise ValueError("Missing training parent")
        previous = current
    if previous != set(counts):
        raise ValueError("Unused additional training parents")


def prepare_coverage(config, output):
    """CPU-only preparation: retain splits and collect newly selected training rows."""
    if output.exists():
        raise FileExistsError(output)
    _, _, original, files = read_source(config)
    settings = original["settings"]
    base = files["train.json"]
    base_parents = sorted({row["parent"] for row in base})
    stages = config["training_parent_counts"]
    if (
        stages != sorted(set(stages))
        or len(stages) < 2
        or stages[0] != len(base_parents)
    ):
        raise ValueError(
            "Use increasing stage sizes beginning with the original training set"
        )
    ordered = addition_order(
        files["groups.json"],
        original["selected_parent_roles"],
        files["excluded_exposure.json"],
        settings["selection_seed"],
    )
    needed = stages[-1] - len(base_parents)
    if needed > len(ordered):
        raise ValueError("Insufficient unused protein groups with holdouts fixed")
    ordered = ordered[:needed]
    source = project_path(original["source"])
    if file_hash(source) != settings["source_sha256"]:
        raise ValueError("Raw measurement source changed")
    names = {row["parent"] for row in ordered}
    raw, audit = audit_table(
        source, settings["min_length"], settings["max_length"], parent_names=names
    )
    additional = []
    for entry in ordered:
        name, group = entry["parent"], entry["group"]
        kept = [
            row
            for row in raw[name]
            if (width := finite_number(row["quality"].get("deltaG_95CI"))) is not None
            and 0 <= width <= settings["maximum_ci_width"]
        ]
        collapsed = collapse_parent(kept)
        if collapsed is None:
            raise ValueError("Selected parent no longer has an unambiguous reference")
        reference, rows = collapsed
        inventory = files["parent_inventory.json"][name]
        if (
            reference != inventory["reference"]
            or len(rows) != inventory["valid_sequences"]
        ):
            raise ValueError("Selected parent differs from frozen inventory")
        reference_row = next(row for row in rows if row["sequence"] == reference)
        variants = sorted(
            [row for row in rows if row["sequence"] != reference],
            key=lambda row: hashlib.sha256(
                f"{settings['selection_seed']}:{name}:{row['sequence']}".encode()
            ).hexdigest(),
        )
        for row in [reference_row] + variants[: settings["variants_per_parent"]]:
            additional.append(
                {
                    **row,
                    "parent": name,
                    "group": group,
                    "is_reference": row["sequence"] == reference,
                }
            )
    training_sets = {
        str(count): base_parents
        + [entry["parent"] for entry in ordered[: count - len(base_parents)]]
        for count in stages
    }
    validate_training_sets(
        base,
        additional,
        files["validation.json"] + files["test_identities.json"],
        training_sets,
        settings["variants_per_parent"],
    )
    output.mkdir(parents=True)
    write_json(output / "additional_train.json", additional)
    write_json(output / "training_sets.json", training_sets)
    write_json(output / "addition_order.json", ordered)
    manifest = {
        "synthetic": original["synthetic"],
        "source_config": config,
        "source_manifest_sha256": file_hash(
            project_path(config["source_run"]) / "data_manifest.json"
        ),
        "raw_source_sha256": settings["source_sha256"],
        "training_parent_counts": stages,
        "additional_parents": len(ordered),
        "additional_sequences": len(additional),
        "variants_per_parent": settings["variants_per_parent"],
        "audit_counts": audit,
        "validation_and_test_unchanged": True,
        "file_sha256": {
            name: file_hash(output / name)
            for name in [
                "additional_train.json",
                "training_sets.json",
                "addition_order.json",
            ]
        },
        "scope": "Development; deterministic training expansion with original held-out groups preserved",
    }
    write_json(output / "manifest.json", manifest)
    return manifest
