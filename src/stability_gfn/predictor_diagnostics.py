"""CPU-only error decomposition of saved predictors; no fitting or backbone execution."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import file_hash, write_json
from .multiprotein import panel_metrics
from .pipeline import project_path
from .scorer import StabilityScorer, regression_metrics


def decompose_errors(predictions, rows, baseline):
    """Split MSE into parent mean error and centered variant error exactly.

    Centering uses measured parent means solely for diagnosis. It is not an
    available correction for unseen proteins, and must not be reported as a model.
    """
    predictions = np.asarray(predictions, dtype=float)
    labels = np.array([row["stability"] for row in rows])
    parents = np.array([row["parent"] for row in rows])
    errors = predictions - labels
    centered_errors = np.empty_like(errors)
    per_parent = []
    mean_error_mse = 0.0
    for parent in sorted(set(parents)):
        mask = parents == parent
        bias = float(errors[mask].mean())
        centered_errors[mask] = errors[mask] - bias
        mean_error_mse += mask.mean() * bias**2
        per_parent.append(
            {
                "parent": parent,
                "count": int(mask.sum()),
                "measured_mean": float(labels[mask].mean()),
                "predicted_mean": float(predictions[mask].mean()),
                "bias": bias,
                "mse": float(np.mean(errors[mask] ** 2)),
                "constant_mse": float(np.mean((baseline - labels[mask]) ** 2)),
                "centered_mse": float(np.mean(centered_errors[mask] ** 2)),
            }
        )
    mse = float(np.mean(errors**2))
    centered_mse = float(np.mean(centered_errors**2))
    if not np.isclose(mse, mean_error_mse + centered_mse, atol=1e-10):
        raise ValueError("MSE decomposition failed")
    baseline_mse = float(np.mean((baseline - labels) ** 2))
    for row in per_parent:
        row["excess_mse_contribution"] = (
            row["count"] / len(rows) * (row["mse"] - row["constant_mse"])
        )
        retained = parents != row["parent"]
        row["leave_parent_out_excess_mse"] = (
            float(np.mean(errors[retained] ** 2 - (baseline - labels[retained]) ** 2))
            if retained.any()
            else None
        )
    return {
        "mse": mse,
        "constant_mse": baseline_mse,
        "parent_mean_error_mse": mean_error_mse,
        "parent_mean_error_fraction": mean_error_mse / mse if mse else 0.0,
        "centered_variant_mse": centered_mse,
        "parent_mean_spearman": regression_metrics(
            [row["predicted_mean"] for row in per_parent],
            [row["measured_mean"] for row in per_parent],
        )["spearman"],
        "parents_beating_constant_mse": sum(
            row["mse"] < row["constant_mse"] for row in per_parent
        ),
        "per_parent": per_parent,
    }


def diagnose(run, output):
    """Audit train/validation data only; preserve the inspected test panel unchanged."""
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)

    def read(path):
        return json.loads(path.read_text())

    config = read(run / "config.json")
    manifest = read(run / "data_manifest.json")
    provenance = read(run / "provenance.json")
    if file_hash(run / "features.pt") != provenance["feature_sha256"]:
        raise ValueError("Feature cache changed")
    cache = torch.load(run / "features.pt", map_location="cpu", weights_only=False)
    data = project_path(config["data"])
    rows = {}
    inputs = {
        "features.pt": file_hash(run / "features.pt"),
        "config.json": file_hash(run / "config.json"),
    }
    for split in ["train", "validation"]:
        path = data / f"{split}.json"
        if file_hash(path) != manifest["file_sha256"][path.name]:
            raise ValueError("Data changed")
        rows[split] = read(path)
        inputs[str(path)] = file_hash(path)
        for row, identity in zip(rows[split], cache["identities"][split], strict=True):
            if any(row[key] != value for key, value in identity.items()):
                raise ValueError("Feature identity ordering changed")
    baseline = float(np.mean([row["stability"] for row in rows["train"]]))
    results = {
        "scope": "Post-hoc training/validation diagnosis; no refitting or test selection",
        "inputs_sha256": inputs,
        "models": {},
    }
    fitting = read(run / "fitting.json")
    for name in ["constant", *fitting]:
        model = None
        if name != "constant":
            path = run / f"{name}.pt"
            inputs[path.name] = file_hash(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model = StabilityScorer(saved["feature_size"], saved["hidden_size"])
            model.load_state_dict(saved["state_dict"])
            model.eval()
        partitions = {}
        for split, partition in rows.items():
            with torch.no_grad():
                predictions = (
                    model(cache["features"][split]).numpy()
                    if model
                    else np.full(len(partition), baseline)
                )
            partitions[split] = {
                "metrics": panel_metrics(
                    predictions, partition, config["selection_fraction"]
                ),
                "decomposition": decompose_errors(predictions, partition, baseline),
            }
        if model is not None:
            actual = partitions["validation"]["metrics"]["pooled"]["rmse"]
            if not np.isclose(
                actual, fitting[name]["selected_validation"]["rmse"], atol=1e-5
            ):
                raise ValueError("Saved checkpoint does not reproduce validation RMSE")
        results["models"][name] = partitions
    output.mkdir(parents=True)
    inputs["diagnostic_source"] = file_hash(Path(__file__))
    (output / "diagnostic_source.py").write_text(Path(__file__).read_text())
    write_json(output / "results.json", results)
    lines = [
        "# Saved-predictor error diagnosis",
        "",
        "Training and validation only; no training or backbone execution.",
        "",
        "| Model | Split | RMSE | Parent mean error share | Centered variant RMSE | Parent mean Spearman |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, partitions in results["models"].items():
        for split, result in partitions.items():
            dec = result["decomposition"]
            correlation = dec["parent_mean_spearman"]
            correlation_text = (
                "undefined" if correlation is None else f"{correlation:.3f}"
            )
            lines.append(
                f"| {name} | {split} | {dec['mse'] ** 0.5:.3f} | "
                f"{dec['parent_mean_error_fraction']:.1%} | "
                f"{dec['centered_variant_mse'] ** 0.5:.3f} | {correlation_text} |"
            )
    lines += [
        "",
        "Centered errors use observed parent means for diagnosis; they are not corrected predictions available at generation time.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(output / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run, args.output)
