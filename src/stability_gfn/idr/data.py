"""Select experimental-reference windows without using predictor outcomes."""

import hashlib
import json
from pathlib import Path

from Bio.Align import PairwiseAligner

from ..artifacts import file_hash, write_json

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
REFERENCE_EXCLUSIONS = {
    "DP00132": "Calsequestrin evidence is explicitly calcium-free apo disorder; not an appropriate intrinsic-disorder prompt for this baseline."
}


def rank_key(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def identity(left, right):
    """Global matches / longer length, including indels in the denominator."""
    aligner = PairwiseAligner(
        mode="global",
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-3,
        extend_gap_score=-0.5,
    )
    alignment = aligner.align(left, right)[0]
    return float(alignment.counts().identities / max(len(left), len(right)))


def select_references(entries, length=80, count=16, seed=20261112):
    """One central window per parent; require one published annotation spanning it.

    An annotation applies under its source assay conditions, not all conditions.
    UniRef50 and window-identity exclusions limit redundancy, not pretraining overlap.
    """
    candidates = []
    for entry in entries:
        if entry["disprot_id"] in REFERENCE_EXCLUSIONS:
            continue
        sequence = entry["sequence"]
        if "Eukaryota" not in entry.get("taxonomy", []):
            continue
        if not entry.get("uniref50"):
            continue
        regions = []
        for region in entry.get("regions", []):
            if (
                region.get("term_id") != "IDPO:0000002"
                or region.get("ec_id")
                not in {"ECO:0006165", "ECO:0001238", "ECO:0006198", "ECO:0006204"}
                or region.get("ec_go") not in {"EXP", "IDA"}
                or not region.get("validated")
                or region.get("unpublished")
                or not region.get("reference_id")
            ):
                continue
            start, end = region.get("start", 0), region.get("end", 0)
            if not 1 <= start <= end <= len(sequence) or end - start + 1 < length:
                continue
            offset = start - 1 + (end - start + 1 - length) // 2
            window = sequence[offset : offset + length]
            if set(window) - set(ALPHABET):
                continue
            consensus = entry.get("disprot_consensus", {}).get("Structural state", [])
            if not any(
                z["type"] == "D"
                and z["start"] <= offset + 1
                and z["end"] >= offset + length
                for z in consensus
            ):
                continue
            regions.append((region, offset, window))
        if not regions:
            continue
        region, offset, window = min(
            regions, key=lambda z: rank_key(z[0]["region_id"], seed)
        )
        candidates.append(
            {
                "id": entry["disprot_id"],
                "accession": entry["acc"],
                "uniref50": entry["uniref50"],
                "name": entry["name"],
                "organism": entry["organism"],
                "sequence": window,
                "start_1based": offset + 1,
                "end_1based_inclusive": offset + length,
                "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "annotation": region,
                "source_url": f"https://disprot.org/{entry['disprot_id']}",
            }
        )
    selected = []
    for candidate in sorted(candidates, key=lambda r: rank_key(r["id"], seed)):
        if any(
            candidate["accession"] == old["accession"]
            or candidate["uniref50"] == old["uniref50"]
            or identity(candidate["sequence"], old["sequence"]) >= 0.5
            for old in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible nonredundant references")
    return selected, len(candidates)


def prepare_references(source, output, source_url):
    source, output = Path(source), Path(output)
    references, eligible = select_references(json.loads(source.read_text())["data"])
    output.mkdir(parents=True, exist_ok=False)
    for index, record in enumerate(references):
        record["role"] = "prompt" if index < 8 else "reference_only"
    write_json(output / "references.json", references)
    write_json(
        output / "manifest.json",
        {
            "source_url": source_url,
            "source_sha256": file_hash(source),
            "source_file": str(source.resolve()),
            "snapshot_date": "2026-11-12",
            "release_note": "API snapshot; date is retrieval date, not a named release",
            "eligible_parents": eligible,
            "count": len(references),
            "length": 80,
            "selection_seed": 20261112,
            "manual_annotation_exclusions": REFERENCE_EXCLUSIONS,
            "prompt_count": 8,
            "references_sha256": file_hash(output / "references.json"),
            "interpretation": "Disorder annotation under original assay conditions; no independent pretraining claim",
        },
    )
    return output / "manifest.json"
