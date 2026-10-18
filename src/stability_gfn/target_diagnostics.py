"""Exact target utility on development data, separated from any policy fitting."""

import torch

from .environment import AMINO_ACIDS, PrefixTree, SequenceContext
from .scorer import regression_metrics


@torch.no_grad()
def extract_diagnostic_cache(backbone, rows, batch_size, include_logits=False):
    """Teacher-force complete sequences, with next-residue logits shifted correctly.

    A causal state at the direction marker predicts residue zero. The final
    residue's state is used in pooled features but never as its own predictor.
    Unlike caching a whole prefix tree, this needs one forward pass per sequence.
    """
    features = []
    logits = []
    residue_ids = backbone.residue_ids(AMINO_ACIDS)
    for start in range(0, len(rows), batch_size):
        states, outputs = backbone.forward(
            [row["sequence"] for row in rows[start : start + batch_size]]
        )
        features.append(states[:, 1:].mean(1).cpu())
        if include_logits:
            logits.append(outputs[:, :-1, residue_ids].cpu())
    result = {"features": torch.cat(features)}
    if include_logits:
        result["path_logits"] = torch.cat(logits)
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite cached features/logits")
    return result


def candidate_priors(sequences, reference, path_logits):
    """Recover the existing local prior and label-free structural controls."""
    tree = PrefixTree(SequenceContext(reference, 1), sequences)
    if sequences != tree.sequences:
        raise ValueError("Candidate logits must follow sorted unique sequence order")
    legal = torch.tensor(tree.legal)[torch.tensor(tree.paths)]
    actions = torch.tensor(tree.actions)
    if path_logits.shape != legal.shape or not torch.isfinite(path_logits).all():
        raise ValueError("Path logits must have shape candidates x length x 20")
    local = path_logits.double().masked_fill(~legal, -torch.inf).log_softmax(-1)
    local_paths = local.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    if abs(local_paths.exp().sum().item() - 1) > 1e-4:
        raise ValueError(
            "Causal path probabilities do not normalize on the candidate tree"
        )
    uniform = torch.zeros_like(local).masked_fill(~legal, -torch.inf).log_softmax(-1)
    uniform_paths = uniform.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    complete = path_logits.double().log_softmax(-1)
    complete_paths = complete.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    return {
        "local": local_paths.log_softmax(0),
        "uniform_legal_actions": uniform_paths.log_softmax(0),
        "complete_20aa": complete_paths.log_softmax(0),
    }


def support_statistics(sequences, reference):
    """Precompute mutation positions and Hamming distances without consulting labels."""
    encoded = torch.tensor(
        [[AMINO_ACIDS.index(residue) for residue in sequence] for sequence in sequences]
    )
    distances = (encoded[:, None, :] != encoded[None, :, :]).sum(-1).double()
    positions = torch.tensor(
        [
            next(
                index + 1
                for index, (a, b) in enumerate(zip(sequence, reference))
                if a != b
            )
            for sequence in sequences
        ]
    )
    return positions, distances


def probability_metrics(
    probabilities, labels, reference_label, positions, distances, count
):
    """Exact expectations, including expected unique samples from iid proposals."""
    probabilities = probabilities.double()
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
    expected_unique = -torch.expm1(
        count * torch.log1p(-probabilities.clamp(max=1))
    ).sum()
    return {
        "measured_mean": (probabilities @ labels).item(),
        "fraction_above_reference": (
            probabilities @ (labels > reference_label).double()
        ).item(),
        "top1_mass": probabilities.max().item(),
        "top5_mass": probabilities.topk(min(5, len(probabilities))).values.sum().item(),
        "first3_position_mass": probabilities[positions <= 3].sum().item(),
        "entropy_nats": entropy.item(),
        "effective_candidates": probabilities.square().sum().reciprocal().item(),
        "expected_unique_raw": expected_unique.item(),
        "expected_duplicate_fraction_raw": 1 - expected_unique.item() / count,
        "expected_pairwise_hamming": (probabilities @ distances @ probabilities).item(),
    }


def sampled_comparison(
    probabilities, predictions, labels, raw_count, selected_count, seed
):
    """Matched categorical proposals and predicted-score selection; keep duplicates.

    These are exact-distribution references, not trained guidance outputs. Stable
    score sorting breaks ties in proposal order; no measurement enters selection.
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(
        probabilities, raw_count, replacement=True, generator=generator
    )
    chosen = indices[
        torch.argsort(predictions[indices], descending=True, stable=True)[
            :selected_count
        ]
    ]
    return {
        "seed": seed,
        "raw_indices": indices.tolist(),
        "selected_indices": chosen.tolist(),
        "raw_measured_mean": labels[indices].mean().item(),
        "selected_measured_mean": labels[chosen].mean().item(),
        "raw_unique": len(indices.unique()),
        "selected_unique": len(chosen.unique()),
    }


def evaluate_inputs(inputs, config):
    """Evaluate saved predictions; no model execution, fitting or test-label access."""
    sequences = inputs["sequences"]
    labels = inputs["labels"].double()
    reference = inputs["reference"]
    positions, distances = support_statistics(sequences, reference)
    priors = candidate_priors(sequences, reference, inputs["path_logits"])
    original = priors["local"].exp()
    original_measured = (original @ labels).item()
    controls = {}
    for name, log_probabilities in priors.items():
        controls[name] = probability_metrics(
            log_probabilities.exp(),
            labels,
            inputs["reference_label"],
            positions,
            distances,
            config["raw_count"],
        )
    results = []
    scorer_reports = {}
    for name, model in inputs["scorers"].items():
        predictions = model["predictions"].double()
        scores = (predictions - model["reference_prediction"]) / inputs["label_scale"]
        top = torch.argsort(predictions, descending=True, stable=True)[
            : config["selected_count"]
        ]
        scorer_reports[name] = {
            **regression_metrics(predictions.numpy(), labels.numpy()),
            "top_distinct_count": len(top),
            "top_distinct_measured_mean": labels[top].mean().item(),
            "top_distinct_fraction_above_reference": (
                labels[top] > inputs["reference_label"]
            )
            .double()
            .mean()
            .item(),
            "top_distinct_note": "Rank all development candidates; not a sampled proposal-budget comparison",
        }
        for alpha in [*config["alphas"], 0.0]:
            # Alpha zero is uniform over candidates, not uniform local actions.
            prior = (alpha * priors["local"]).softmax(0)
            target = (alpha * priors["local"] + config["beta"] * scores).softmax(0)
            base_metrics = probability_metrics(
                prior,
                labels,
                inputs["reference_label"],
                positions,
                distances,
                config["raw_count"],
            )
            target_metrics = probability_metrics(
                target,
                labels,
                inputs["reference_label"],
                positions,
                distances,
                config["raw_count"],
            )
            gain = target_metrics["measured_mean"] - base_metrics["measured_mean"]
            results.append(
                {
                    "scorer": name,
                    "alpha": alpha,
                    "beta": config["beta"],
                    "uniform_diagnostic_only": alpha == 0,
                    "baseline": base_metrics,
                    "target": target_metrics,
                    "measured_gain_vs_matched_prior": gain,
                    "measured_gain_vs_original_prior": target_metrics["measured_mean"]
                    - original_measured,
                    "predicted_gain_vs_matched_prior": (
                        (target - prior) @ predictions
                    ).item(),
                    "prior_weighted_mae": (prior @ (predictions - labels).abs()).item(),
                    "target_weighted_mae": (
                        target @ (predictions - labels).abs()
                    ).item(),
                    "prior_probabilities": prior.tolist(),
                    "target_probabilities": target.tolist(),
                    "matched_samples": {
                        policy: [
                            sampled_comparison(
                                probabilities,
                                predictions,
                                labels,
                                config["raw_count"],
                                config["selected_count"],
                                seed,
                            )
                            for seed in (100000, 100001, 100002)
                        ]
                        for policy, probabilities in (
                            ("prior", prior),
                            ("exact_target", target),
                        )
                    },
                }
            )
    ensemble = inputs["ensemble_members"].double()
    disagreement = ensemble.std(0, unbiased=False)
    errors = (ensemble.mean(0) - labels).abs()
    uncertainty = {
        "disagreement_error_spearman": regression_metrics(
            disagreement.numpy(), errors.numpy()
        )["spearman"],
        "prediction_std": disagreement.tolist(),
        "delta_prediction_std": (
            ensemble - inputs["ensemble_reference_predictions"].double()[:, None]
        )
        .std(0, unbiased=False)
        .tolist(),
        "note": "Uncalibrated disagreement diagnostic; no confidence-bound claim or reward penalty",
    }
    return {
        "controls": controls,
        "scorers": scorer_reports,
        "uncertainty": uncertainty,
        "comparisons": results,
    }
