"""Label-free confirmation identities and checked fitting-only data access."""

import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_hash, write_json
from .data import finite_number
from .environment import AMINO_ACIDS, hamming


def prepare_confirmation_support(panel, output):
    """Reconstruct the frozen test identities from source availability, not outcomes.

    The original panel stores test identities together with their labels. Avoid
    opening those files before fitting. Reproduce the deterministic split using
    only valid-label availability, parent names and canonical sequence identities
    from the original CSV. Never retain or rank numerical stability values here.
    """
    panel, output = Path(panel), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite support {output}")
    manifest = json.loads((panel / "manifest.json").read_text())
    source = Path(manifest["source"])
    if file_hash(source) != manifest["source_sha256"]:
        raise ValueError("Source CSV differs from the frozen panel")
    entries = [
        entry for entry in manifest["parents"] if entry["role"] == "confirmation"
    ]
    references = {entry["parent"]: entry["reference"] for entry in entries}
    identities = {name: set() for name in references}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["WT_name"].strip()
            if name not in identities or finite_number(row["dG_ML"]) is None:
                continue
            sequence = row["aa_seq"].strip()
            reference = references[name]
            if (
                len(sequence) == len(reference)
                and set(sequence) <= set(AMINO_ACIDS)
                and hamming(sequence, reference) <= 1
            ):
                identities[name].add(sequence)
    candidates = {}
    for entry in entries:
        variants = sorted(identities[entry["parent"]] - {entry["reference"]})
        if len(variants) != entry["valid_variants"]:
            raise ValueError(f"Coverage no longer matches panel: {entry['parent']}")
        order = np.random.default_rng(manifest["config"]["split_seed"]).permutation(
            len(variants)
        )
        selected = sorted(
            variants[index] for index in order[int(0.85 * len(variants)) :]
        )
        if len(selected) != entry["split_counts"]["test_labels"]:
            raise ValueError("Test identity count differs from frozen split")
        candidates[entry["id"]] = selected
    output.mkdir(parents=True)
    for identifier, sequences in candidates.items():
        write_json(output / f"{identifier}.json", sequences)
    support_manifest = {
        "synthetic": manifest["synthetic"],
        "panel_manifest_sha256": file_hash(panel / "manifest.json"),
        "source_sha256": manifest["source_sha256"],
        "method": "Original source identity/valid-label availability filter, sorted single mutants, split seed and 85% cut point; no test-label files opened",
        "parents": [
            {
                "id": entry["id"],
                "parent": entry["parent"],
                "count": len(candidates[entry["id"]]),
                "sha256": file_hash(output / f"{entry['id']}.json"),
            }
            for entry in entries
        ],
    }
    write_json(output / "manifest.json", support_manifest)
    return support_manifest


def read_confirmation_training(panel, support, entry, support_entry):
    """Load fit/inner labels and candidate identities, never test/development labels."""
    if entry["role"] != "confirmation" or support_entry["id"] != entry["id"]:
        raise ValueError("Expected a matching reserved confirmation parent")
    rows = {}
    for split in ("fit", "inner_validation"):
        path = Path(panel) / entry["id"] / f"{split}.json"
        if file_hash(path) != entry["file_sha256"][path.name]:
            raise ValueError(f"Changed fitting input: {path}")
        rows[split] = json.loads(path.read_text())
    path = Path(support) / f"{entry['id']}.json"
    if file_hash(path) != support_entry["sha256"]:
        raise ValueError("Changed candidate identities")
    sequences = json.loads(path.read_text())
    groups = [{row["sequence"] for row in values} for values in rows.values()]
    if groups[0] & groups[1] or set(sequences) & (groups[0] | groups[1]):
        raise ValueError("Confirmation split overlap")
    if entry["reference"] not in groups[0]:
        raise ValueError("Reference must remain in fitting data")
    if (
        len(sequences) != len(set(sequences))
        or len(sequences) != entry["split_counts"]["test_labels"]
    ):
        raise ValueError("Candidate identities do not match declared count")
    if any(hamming(sequence, entry["reference"]) != 1 for sequence in sequences):
        raise ValueError("Confirmation requires measured single substitutions")
    return rows, sorted(sequences)
