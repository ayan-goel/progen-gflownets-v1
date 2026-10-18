"""Explicit distribution, diversity and stability metrics without hidden selection."""

import numpy as np
import torch

from .environment import hamming


def distribution_metrics(log_q, log_p0, scores, beta, log_z=None):
    """Compute TV, KL(q||target), KL(target||q), and normalization errors exactly."""
    log_q = log_q.detach().cpu().double()
    log_weights = (
        log_p0.detach().cpu().double() + beta * torch.as_tensor(scores).cpu().double()
    )
    exact_log_z = torch.logsumexp(log_weights, 0)
    log_target = log_weights - exact_log_z
    probability = log_q.exp()
    target = log_target.exp()
    score_values = torch.as_tensor(scores).detach().cpu().double()
    result = {
        "total_variation": (0.5 * (probability - target).abs().sum()).item(),
        "kl_q_to_target": (probability * (log_q - log_target)).sum().item(),
        "kl_target_to_q": (target * (log_target - log_q)).sum().item(),
        "probability_sum": probability.sum().item(),
        "exact_log_z": exact_log_z.item(),
        "expected_scaled_score": (probability * score_values).sum().item(),
        "target_expected_scaled_score": (target * score_values).sum().item(),
        "entropy_nats": -(probability * log_q).sum().item(),
    }
    if log_z is not None:
        result["absolute_log_z_error"] = abs(float(log_z) - exact_log_z.item())
    return result


def sample_metrics(
    sequences,
    reference,
    training_sequences,
    predictions,
    measured=None,
    reference_label=None,
    reference_prediction=None,
):
    """Metrics describe the provided batch, including multiplicities and coverage."""
    if not sequences:
        raise ValueError("Cannot evaluate an empty sample batch")
    pairs = [
        hamming(left, right)
        for index, left in enumerate(sequences)
        for right in sequences[index + 1 :]
    ]
    mutations = [hamming(sequence, reference) for sequence in sequences]
    novelty_set = list(training_sequences)
    distances = (
        [
            min(hamming(sequence, other) for other in novelty_set)
            for sequence in sequences
        ]
        if novelty_set
        else []
    )
    result = {
        "count": len(sequences),
        "unique_fraction": len(set(sequences)) / len(sequences),
        "duplicate_fraction": 1 - len(set(sequences)) / len(sequences),
        "mean_pairwise_hamming": float(np.mean(pairs)) if pairs else 0.0,
        "mean_pairwise_normalized_hamming": float(np.mean(pairs) / len(reference))
        if pairs
        else 0.0,
        "mean_distance_to_training": float(np.mean(distances)) if distances else None,
        "fraction_absent_from_training": sum(
            sequence not in training_sequences for sequence in sequences
        )
        / len(sequences)
        if training_sequences
        else None,
        "mean_predicted_stability": float(
            np.mean([predictions[sequence] for sequence in sequences])
        ),
        "mean_mutations": float(np.mean(mutations)),
        "mutation_counts": {
            str(count): mutations.count(count) for count in sorted(set(mutations))
        },
    }
    result["mean_predicted_improvement"] = (
        result["mean_predicted_stability"] - reference_prediction
        if reference_prediction is not None
        else None
    )
    result["fraction_predicted_stabilized"] = (
        sum(predictions[sequence] > reference_prediction for sequence in sequences)
        / len(sequences)
        if reference_prediction is not None
        else None
    )
    measured = measured or {}
    covered = [sequence for sequence in sequences if sequence in measured]
    result["measurement_coverage"] = len(covered) / len(sequences)
    result["mean_measured_stability"] = (
        float(np.mean([measured[sequence] for sequence in covered]))
        if covered
        else None
    )
    result["predictor_mae_on_covered"] = (
        float(
            np.mean(
                [
                    abs(predictions[sequence] - measured[sequence])
                    for sequence in covered
                ]
            )
        )
        if covered
        else None
    )
    result["mean_measured_improvement"] = (
        (result["mean_measured_stability"] - reference_label)
        if covered and reference_label is not None
        else None
    )
    result["fraction_measured_stabilized"] = (
        sum(measured[sequence] > reference_label for sequence in covered) / len(covered)
        if covered and reference_label is not None
        else None
    )
    return result
