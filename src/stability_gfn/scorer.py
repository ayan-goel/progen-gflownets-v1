"""Train and validate a small fixed stability predictor on frozen sequence features."""

import copy
import time

import numpy as np
import torch
from torch import nn


def average_ranks(values):
    """Stable average ranks for Spearman correlation, including tied predictions."""
    values = np.asarray(values)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - counts + ends - 1) / 2
    return ranks[inverse]


def regression_metrics(predictions, labels):
    """Absolute-stability errors in kcal/mol; undefined correlations remain null."""
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)
    prediction_ranks = average_ranks(predictions)
    label_ranks = average_ranks(labels)
    valid = len(labels) > 1 and np.std(prediction_ranks) > 0 and np.std(label_ranks) > 0
    return {
        "mae": float(np.mean(np.abs(predictions - labels))),
        "rmse": float(np.sqrt(np.mean((predictions - labels) ** 2))),
        "spearman": float(np.corrcoef(prediction_ranks, label_ranks)[0, 1])
        if valid
        else None,
        "count": len(labels),
    }


class StabilityScorer(nn.Module):
    """Train-only normalization and regression, returning absolute stability in kcal/mol."""

    def __init__(self, feature_size, hidden_size=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("feature_mean", torch.zeros(feature_size))
        self.register_buffer("feature_scale", torch.ones(feature_size))
        self.register_buffer("label_mean", torch.zeros(()))
        self.register_buffer("label_scale", torch.ones(()))
        if hidden_size:
            self.network = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.network = nn.Linear(feature_size, 1)

    def forward(self, features):
        standardized = (features - self.feature_mean) / self.feature_scale
        return (
            self.network(standardized).squeeze(-1) * self.label_scale + self.label_mean
        )


def initialize_scorer(features, labels, hidden_size):
    """Training-set moments only, with explicit floors for constant dimensions."""
    model = StabilityScorer(features.shape[1], hidden_size).to(features.device)
    model.feature_mean.copy_(features.mean(0))
    model.feature_scale.copy_(features.std(0, unbiased=False).clamp_min(1e-6))
    model.label_mean.copy_(labels.mean())
    model.label_scale.copy_(labels.std(unbiased=False).clamp_min(1e-6))
    return model


def fit_scorer(
    train_features, train_labels, validation_features, validation_labels, config, seed=0
):
    """Choose ridge or an MLP using validation RMSE; never accept test labels."""
    torch.manual_seed(seed)
    device = train_features.device
    train_labels = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    validation_labels = torch.as_tensor(
        validation_labels, dtype=torch.float32, device=device
    )
    if train_labels.std(unbiased=False) < 1e-6:
        raise ValueError("Training stability labels have no variation")
    candidates = []
    best_model = None
    best_error = float("inf")
    started = time.perf_counter()

    def consider(model, name, step=None):
        nonlocal best_model, best_error
        with torch.no_grad():
            metrics = regression_metrics(
                model(validation_features).cpu(), validation_labels.cpu()
            )
        candidates.append({"model": name, "step": step, **metrics})
        if metrics["rmse"] < best_error:
            best_error = metrics["rmse"]
            best_model = copy.deepcopy(model)

    for penalty in config["ridge_penalties"]:
        if penalty <= 0:
            raise ValueError("Ridge penalties must be positive")
        model = initialize_scorer(train_features, train_labels, hidden_size=0)
        features = (
            (train_features - model.feature_mean) / model.feature_scale
        ).double()
        labels = ((train_labels - model.label_mean) / model.label_scale).double()
        # Dual ridge avoids a 1536x1536 solve when the pilot has fewer training rows.
        kernel = features @ features.T + penalty * torch.eye(
            len(features), device=device, dtype=torch.float64
        )
        weights = features.T @ torch.linalg.solve(kernel, labels)
        with torch.no_grad():
            model.network.weight.copy_(weights.float()[None])
            model.network.bias.zero_()
        consider(model, f"ridge_{penalty}")

    model = initialize_scorer(train_features, train_labels, config["hidden_size"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    for step in range(1, config["steps"] + 1):
        indices = torch.randint(
            len(train_labels),
            (config["batch_size"],),
            generator=generator,
            device=device,
        )
        loss = (
            (
                (model(train_features[indices]) - train_labels[indices])
                / model.label_scale
            )
            .square()
            .mean()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite scorer loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        if step % config["validate_every"] == 0 or step == config["steps"]:
            consider(model, "mlp", step)
    best_model.eval().requires_grad_(False)
    with torch.no_grad():
        validation = regression_metrics(
            best_model(validation_features).cpu(), validation_labels.cpu()
        )
    mean_baseline = regression_metrics(
        np.full(len(validation_labels), train_labels.mean().item()),
        validation_labels.cpu(),
    )
    passed = (
        validation["mae"] < mean_baseline["mae"]
        and validation["spearman"] is not None
        and validation["spearman"] >= config["minimum_spearman"]
    )
    return best_model, {
        "candidates": candidates,
        "selected_validation": validation,
        "training_mean_baseline_validation": mean_baseline,
        "gate_passed": passed,
        "minimum_spearman": config["minimum_spearman"],
        "reward_scale": best_model.label_scale.item(),
        "training_seconds": time.perf_counter() - started,
    }


def stability_scores(model, features, reference_features):
    """Fixed dimensionless predicted improvement; no clipping in the first pilot."""
    with torch.no_grad():
        predictions = model(features)
        reference_prediction = model(reference_features).squeeze()
        scores = (predictions - reference_prediction) / model.label_scale
    return predictions, scores
