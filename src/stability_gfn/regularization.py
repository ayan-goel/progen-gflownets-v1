"""Grouped ridge regularization study on cached frozen features; no backbone calls."""

import json
import shutil
import time

import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .multiprotein import panel_metrics
from .pipeline import ROOT, project_path
from .predictor_diagnostics import decompose_errors
from .scorer import StabilityScorer, fit_scorer


def grouped_folds(rows, count, seed):
    """Assign whole protein groups to deterministic, nearly equal-sized folds."""
    groups = sorted({row["group"] for row in rows})
    if not 2 <= count <= len(groups):
        raise ValueError("Fold count must be between two and the training group count")
    shuffled = np.random.default_rng(seed).permutation(groups)
    assignments = {
        str(group): fold
        for fold, part in enumerate(np.array_split(shuffled, count))
        for group in part
    }
    return np.array([assignments[row["group"]] for row in rows]), assignments


def fixed_ridge(features, labels, held_features, held_labels, penalty):
    """Use the existing ridge solver with exactly one candidate, not an inner search."""
    model, _ = fit_scorer(
        features,
        labels,
        held_features,
        held_labels,
        {"ridge_penalties": [penalty], "minimum_spearman": 0.2},
        seed=0,
        family="ridge",
    )
    return model


def cross_validate(features, rows, config, progress=None):
    """Select strength using training-parent out-of-fold predictions only."""
    fold_ids, assignments = grouped_folds(rows, config["folds"], config["fold_seed"])
    labels = np.array([row["stability"] for row in rows])
    penalties = config["ridge_penalties"]
    predictions = {"constant": np.empty(len(rows))}
    predictions.update({str(penalty): np.empty(len(rows)) for penalty in penalties})
    for fold in range(config["folds"]):
        training = np.flatnonzero(fold_ids != fold)
        held = np.flatnonzero(fold_ids == fold)
        train_indices = torch.tensor(training, device=features.device)
        held_indices = torch.tensor(held, device=features.device)
        predictions["constant"][held] = labels[training].mean()
        for penalty in penalties:
            # fit_scorer computes both feature and label moments from this fold's
            # training rows; held labels only calculate metrics for one fixed fit.
            model = fixed_ridge(
                features[train_indices],
                labels[training],
                features[held_indices],
                labels[held],
                penalty,
            )
            with torch.no_grad():
                predictions[str(penalty)][held] = (
                    model(features[held_indices]).cpu().numpy()
                )
        if progress is not None:
            progress(fold + 1)
    metrics = {
        name: panel_metrics(values, rows, config["selection_fraction"])
        for name, values in predictions.items()
    }
    # Equal parent weighting, rather than averaging unequal-sized fold RMSEs.
    scores = {
        name: float(
            np.sqrt(np.mean([parent["rmse"] ** 2 for parent in value["per_parent"]]))
        )
        for name, value in metrics.items()
    }
    selected = min(penalties, key=lambda penalty: (scores[str(penalty)], -penalty))
    return {
        "group_folds": assignments,
        "row_folds": fold_ids.tolist(),
        "predictions": {name: values.tolist() for name, values in predictions.items()},
        "metrics": metrics,
        "parent_balanced_rmse": scores,
        "selected_penalty": selected,
        "constant_matches_or_beats_selected": scores["constant"]
        <= scores[str(selected)],
        "selected_at_upper_boundary": selected == max(penalties),
    }


def check_regularization(config):
    """CPU preflight: verify source snapshots and row identities without fitting."""
    run = project_path(config["source_run"])
    required = {
        "features.pt",
        "config.json",
        "data_manifest.json",
        "ridge-seed-0.pt",
        "fitting.json",
    }
    if not required <= set(config["source_sha256"]):
        raise ValueError("Missing required source hashes")
    for name, expected in config["source_sha256"].items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Source changed: {name}")
    penalties = config["ridge_penalties"]
    if (
        not penalties
        or len(set(penalties)) != len(penalties)
        or any(not np.isfinite(value) or value <= 0 for value in penalties)
        or config["reference_penalty"] not in penalties
    ):
        raise ValueError("Use unique positive ridge penalties including the reference")
    source_config = json.loads((run / "config.json").read_text())
    fitting = json.loads((run / "fitting.json").read_text())["ridge-seed-0"]
    original_choice = min(
        fitting["candidates"], key=lambda candidate: candidate["rmse"]
    )
    if (
        float(original_choice["model"].removeprefix("ridge_"))
        != config["reference_penalty"]
    ):
        raise ValueError(
            "Reference checkpoint's selected penalty differs from configuration"
        )
    manifest = json.loads((run / "data_manifest.json").read_text())
    data = project_path(source_config["data"])
    if file_hash(data / "manifest.json") != source_config["manifest_sha256"]:
        raise ValueError("Data manifest changed")
    cache = torch.load(run / "features.pt", map_location="cpu", weights_only=False)
    rows = {}
    for split in ["train", "validation"]:
        path = data / f"{split}.json"
        if file_hash(path) != manifest["file_sha256"][path.name]:
            raise ValueError(f"Data changed: {split}")
        rows[split] = json.loads(path.read_text())
        identities = cache["identities"][split]
        if len(rows[split]) != len(cache["features"][split]):
            raise ValueError("Feature row count mismatch")
        for row, identity in zip(rows[split], identities, strict=True):
            if any(row[key] != value for key, value in identity.items()):
                raise ValueError("Feature identity ordering mismatch")
        if not torch.isfinite(cache["features"][split]).all():
            raise ValueError("Non-finite cached features")
    groups = [{row["group"] for row in rows[split]} for split in rows]
    if groups[0] & groups[1]:
        raise ValueError("Training and validation groups overlap")
    grouped_folds(rows["train"], config["folds"], config["fold_seed"])
    return run, manifest, cache, rows


def write_report(output, cv, validation):
    """Keep cross-validation selection separate from the fixed validation check."""
    lines = [
        "# Ridge regularization study",
        "",
        "Same training/validation proteins and frozen features; no test-label evaluation.",
        f"Selected ridge penalty from training-group CV: **{cv['selected_penalty']}**.",
        "",
        "| CV candidate | Parent-balanced RMSE | Mean within-parent selection lift |",
        "|---|---:|---:|",
    ]
    for name, score in cv["parent_balanced_rmse"].items():
        lift = cv["metrics"][name]["macro_parent"]["selection_lift"]
        lines.append(f"| {name} | {score:.4f} | {lift:+.4f} |")
    lines += [
        "",
        "## Fixed validation proteins",
        "",
        "| Predictor | RMSE | Mean-error share of MSE | Within-parent Spearman | Selection lift |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in validation.items():
        metrics, error = result["metrics"], result["decomposition"]
        correlation = metrics["macro_parent"]["spearman"]
        corr_text = "undefined" if correlation is None else f"{correlation:.4f}"
        lines.append(
            f"| {name} | {metrics['pooled']['rmse']:.4f} | "
            f"{error['parent_mean_error_fraction']:.1%} | {corr_text} | "
            f"{metrics['macro_parent']['selection_lift']:+.4f} |"
        )
    lines += [
        "",
        "Errors and lift are kcal/mol; lift compares predicted top 10% against uniform candidates within each protein, not ProGen.",
    ]
    if cv["constant_matches_or_beats_selected"]:
        lines += [
            "",
            "**Constant matches or beats the selected ridge in CV. Do not promote ridge based on favorable validation alone.**",
        ]
    if cv["selected_at_upper_boundary"]:
        lines += [
            "",
            "Selected strength is at the upper grid boundary; an optimum beyond it has not been ruled out.",
        ]
    lines += [
        "",
        "Validation has already informed development; this is not independent confirmation. No GFlowNet or generation was run.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_regularization(config, output, device, smoke=False):
    """User-run grouped CV and fixed validation; synthetic CPU fits only in tests."""
    if (smoke and device != "cpu") or (not smoke and not device.startswith("cuda")):
        raise ValueError(
            "Real runs require user CUDA execution; CPU fitting is smoke-only"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source, manifest, cache, rows = check_regularization(config)
    if smoke != manifest["synthetic"]:
        raise ValueError("Smoke requires synthetic inputs")
    output = new_run(output, {**config, "device": device, "smoke": smoke})
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(
            ROOT / "docs/experiments/regularization_2026-11-10.md",
            output / "protocol.md",
        )
        for split, partition in rows.items():
            write_json(output / f"{split}.json", partition)
        write_json(output / "data_manifest.json", manifest)
        features = cache["features"]["train"].to(device)
        labels = np.array([row["stability"] for row in rows["train"]])

        def progress(completed):
            print(
                f"Completed grouped CV fold {completed}/{config['folds']}", flush=True
            )
            write_json(output / "progress.json", {"completed_folds": completed})

        cv = cross_validate(features, rows["train"], config, progress)
        write_json(output / "cross_validation.json", cv)
        write_json(
            output / "selection.json",
            {
                key: cv[key]
                for key in [
                    "selected_penalty",
                    "constant_matches_or_beats_selected",
                    "selected_at_upper_boundary",
                ]
            },
        )
        # Selection is already saved. This is a single-strength refit on all
        # training rows; training rows are also the helper's metric-only argument.
        selected = fixed_ridge(
            features, labels, features, labels, cv["selected_penalty"]
        )
        torch.save(
            {
                "state_dict": {
                    key: value.cpu() for key, value in selected.state_dict().items()
                },
                "hidden_size": 0,
                "feature_size": features.shape[1],
                "penalty": cv["selected_penalty"],
            },
            output / "selected-ridge.pt",
        )
        saved = torch.load(
            source / "ridge-seed-0.pt", map_location="cpu", weights_only=False
        )
        reference = StabilityScorer(saved["feature_size"], saved["hidden_size"])
        reference.load_state_dict(saved["state_dict"])
        reference = reference.to(device).eval()
        shutil.copy2(source / "ridge-seed-0.pt", output / "reference-ridge.pt")
        val_features = cache["features"]["validation"].to(device)
        with torch.no_grad():
            predictions = {
                "constant": np.full(len(rows["validation"]), labels.mean()),
                "reference_ridge": reference(val_features).cpu().numpy(),
                "selected_ridge": selected(val_features).cpu().numpy(),
            }
        validation = {
            name: {
                "metrics": panel_metrics(
                    values, rows["validation"], config["selection_fraction"]
                ),
                "decomposition": decompose_errors(
                    values, rows["validation"], labels.mean()
                ),
            }
            for name, values in predictions.items()
        }
        write_json(
            output / "validation_predictions.json",
            {
                "sequences": [row["sequence"] for row in rows["validation"]],
                "models": {
                    name: values.tolist() for name, values in predictions.items()
                },
            },
        )
        write_json(output / "validation_results.json", validation)
        write_report(output, cv, validation)
        write_json(
            output / "status.json",
            {
                "status": "smoke_complete" if smoke else "complete",
                "total_seconds": time.perf_counter() - started,
                "scope": "development",
                "test_evaluated": False,
            },
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    print(output / "report.md")
