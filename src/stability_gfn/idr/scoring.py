"""Sequence descriptors and explicitly labeled disorder-prediction evaluation."""

import math
from collections import Counter
from itertools import combinations

import numpy as np

from .data import ALPHABET, identity


def descriptors(sequence, window=9):
    """Nominal K/R +1, D/E -1; other residues zero, excluding termini and PTMs."""
    if not sequence or set(sequence) - set(ALPHABET) or window < 1 or window % 2 == 0:
        raise ValueError("Expected canonical sequence and positive odd profile window")
    counts = Counter(sequence)
    charge = np.array([1 if r in "KR" else -1 if r in "DE" else 0 for r in sequence])
    half = window // 2
    profile = [
        float(charge[max(0, i - half) : min(len(sequence), i + half + 1)].mean())
        for i in range(len(sequence))
    ]
    longest, current = 1, 1
    for i in range(1, len(sequence)):
        current = current + 1 if sequence[i] == sequence[i - 1] else 1
        longest = max(longest, current)
    kmers = [sequence[i : i + 3] for i in range(max(0, len(sequence) - 2))]
    return {
        "length": len(sequence),
        "ncpr": float(charge.mean()),
        "fcr": float(np.abs(charge).mean()),
        "charge_profile": profile,
        "composition": {r: counts[r] / len(sequence) for r in ALPHABET},
        "entropy_bits": -sum(
            (n / len(sequence)) * math.log2(n / len(sequence)) for n in counts.values()
        ),
        "longest_homopolymer": longest,
        "repeated_trimer_fraction": 1 - len(set(kmers)) / len(kmers) if kmers else 0.0,
    }


def predict_disorder(sequences, version):
    """Pinned metapredict on CPU, even when generation runs on CUDA.

    V2 is a same-package sensitivity check, not an independent biological verifier.
    """
    import metapredict

    if metapredict.__version__ != "3.0.2":
        raise ValueError("This protocol requires metapredict==3.0.2")
    result = metapredict.predict_disorder(
        sequences,
        version=version,
        device="cpu",
        round_values=False,
        return_numpy=True,
        show_progress_bar=False,
    )
    arrays = []
    for sequence, pair in zip(sequences, result):
        returned_sequence, values = pair
        values = np.asarray(values, dtype=float)
        if (
            returned_sequence != sequence
            or values.shape != (len(sequence),)
            or not np.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
        ):
            raise ValueError("Invalid or misaligned metapredict output")
        arrays.append(values)
    if len(arrays) != len(sequences):
        raise ValueError("Missing disorder predictions")
    return arrays


def disorder_summary(scores, cutoff, fraction):
    scores = np.asarray(scores)
    disordered_fraction = float((scores > cutoff).mean())
    return {
        "mean": float(scores.mean()),
        "fraction_above_cutoff": disordered_fraction,
        "passes": disordered_fraction >= fraction,
    }


def score_records(records, references, config, predictor=predict_disorder):
    """Score full contexts and isolated fixed-length tails without prefix inflation."""
    start = config["prefix_length"]
    unique = list(
        dict.fromkeys(
            [r["sequence"] for r in records] + [r["sequence"][start:] for r in records]
        )
    )
    predictions = {}
    for version in ("V3", "V2"):
        print(
            f"Disorder evaluation: {version}, {len(unique)} unique inputs (CPU)",
            flush=True,
        )
        values = predictor(unique, version)
        predictions[version] = dict(zip(unique, values))
    scored = []
    for record in records:
        sequence, tail = record["sequence"], record["sequence"][start:]
        row = dict(record)
        row["full_descriptors"] = descriptors(sequence, config["charge_window"])
        row["tail_descriptors"] = descriptors(tail, config["charge_window"])
        row["disorder"] = {}
        for version in predictions:
            full = np.asarray(predictions[version][sequence])
            isolated = np.asarray(predictions[version][tail])
            row["disorder"][version] = {
                "full_scores": full.tolist(),
                "isolated_tail_scores": isolated.tolist(),
            }
            for name, scores in (
                ("full", full),
                ("context_tail", full[start:]),
                ("isolated_tail", isolated),
            ):
                row["disorder"][version][name] = disorder_summary(
                    scores,
                    config["disorder_cutoff"],
                    config["minimum_disordered_fraction"],
                )
        # Similarity excludes the supplied prefix in both generated arms.
        matches = [
            (identity(tail, ref["sequence"][start:]), ref["id"]) for ref in references
        ]
        best = max(matches)
        row["nearest_reference_tail_identity"] = best[0]
        row["nearest_reference_id"] = best[1]
        row["exact_reference_tail_match"] = any(
            tail == ref["sequence"][start:] for ref in references
        )
        scored.append(row)
    return scored


def summarize(rows):
    """Raw-sample statistics; no filtering or deduplication before success counts."""
    tails = [r["sequence"][r["evaluation_start"] :] for r in rows]
    distances = [
        sum(a != b for a, b in zip(left, right)) / len(left)
        for left, right in combinations(tails, 2)
    ]
    return {
        "count": len(rows),
        "v3_context_tail_success_rate": float(
            np.mean([r["disorder"]["V3"]["context_tail"]["passes"] for r in rows])
        ),
        "v3_isolated_tail_success_rate": float(
            np.mean([r["disorder"]["V3"]["isolated_tail"]["passes"] for r in rows])
        ),
        "v2_context_tail_success_rate": float(
            np.mean([r["disorder"]["V2"]["context_tail"]["passes"] for r in rows])
        ),
        "v3_mean_context_tail_disorder": float(
            np.mean([r["disorder"]["V3"]["context_tail"]["mean"] for r in rows])
        ),
        "unique_tail_fraction": len(set(tails)) / len(tails),
        "mean_pairwise_tail_hamming_fraction": float(np.mean(distances))
        if distances
        else None,
        "mean_tail_entropy_bits": float(
            np.mean([r["tail_descriptors"]["entropy_bits"] for r in rows])
        ),
        "mean_tail_ncpr": float(np.mean([r["tail_descriptors"]["ncpr"] for r in rows])),
        "mean_tail_fcr": float(np.mean([r["tail_descriptors"]["fcr"] for r in rows])),
        "mean_nearest_reference_tail_identity": float(
            np.mean([r["nearest_reference_tail_identity"] for r in rows])
        ),
        "exact_reference_tail_match_fraction": float(
            np.mean([r["exact_reference_tail_match"] for r in rows])
        ),
        "fraction_tail_homopolymer_at_least_8": float(
            np.mean([r["tail_descriptors"]["longest_homopolymer"] >= 8 for r in rows])
        ),
    }
