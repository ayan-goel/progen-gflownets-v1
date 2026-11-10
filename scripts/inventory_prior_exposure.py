"""Historical inventory builder; original archive sources were deleted on 2026-11-10."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
archive = Path("/usr/scratch/agoel320/progen-gflownets-archive/post-pilot-2026-11-03")
paths = [root / "data/processed/local-v1/manifest.json"]
paths += sorted((archive / "data/processed").glob("*/manifest.json"))
paths += sorted((archive / "data/processed/scorer-scaling-v1").glob("*identities.json"))
paths += [archive / "data/processed/scorer-transfer-v1/identities.json"]
destination = root / "data/metadata/prior-exposure-2026-11-10.json"
if destination.exists():
    raise FileExistsError(
        "Exposure inventory is frozen; preserve it rather than overwriting"
    )
if not archive.is_dir():
    raise FileNotFoundError(
        "Original archive sources were deleted; cannot rebuild a complete inventory. "
        "Use the preserved frozen inventory."
    )
parents = set()
sequences = set()
provenance = {}


def inspect(obj):
    if isinstance(obj, list):
        for entry in obj:
            inspect(entry)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key in ["parent", "WT_name"] and isinstance(value, str):
                parents.add(value)
            if (
                key in ["reference", "sequence"]
                and isinstance(value, str)
                and set(value) <= set("ACDEFGHIKLMNPQRSTVWY")
            ):
                sequences.add(hashlib.sha256(value.encode()).hexdigest())
            if isinstance(value, (dict, list)):
                inspect(value)


for path in paths:
    inspect(json.loads(path.read_text()))
    provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "scope": "Conservative prior preparation/exposure inventory from identity-only files and manifests; not proof of all historical outcome access. No retired outcome files read.",
    "known_parent_names": sorted(parents),
    "known_sequence_sha256": sorted(sequences),
    "metadata_source_sha256": provenance,
}
destination.write_text(json.dumps(record, indent=2) + "\n")
print(
    "metadata files",
    len(provenance),
    "parent identifiers",
    len(parents),
    "sequence identities",
    len(sequences),
)
