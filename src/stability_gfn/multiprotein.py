"""Frozen-feature stability regression across protein groups, without GFN fitting."""

import copy
import json
import math
import shutil
import time
from collections import defaultdict

import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import TOKENIZER_PATH, FrozenProGen
from .pipeline import ROOT, project_path
from .scorer import fit_scorer, regression_metrics


def check_multiprotein(config):
    """Verify frozen data/model identities without opening evaluation measurements."""
    data = project_path(config["data"])
    if file_hash(data / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if file_hash(data / name) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    if not manifest["synthetic"]:
        for name, expected in config["checkpoint_sha256"].items():
            if file_hash(project_path(config["checkpoint"]) / name) != expected:
                raise ValueError("ProGen checkpoint changed")
        if file_hash(TOKENIZER_PATH) != config["tokenizer_sha256"]:
            raise ValueError("Tokenizer changed")
    rows = {
        split: json.loads((data / name).read_text())
        for split, name in [
            ("train", "train.json"),
            ("validation", "validation.json"),
            ("test", "test_identities.json"),
        ]
    }
    groups = [{row["group"] for row in rows[split]} for split in rows]
    if any(groups[left] & groups[right] for left, right in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Protein groups cross partitions")
    sequences = [row["sequence"] for partition in rows.values() for row in partition]
    if len(sequences) != len(set(sequences)):
        raise ValueError("Sequences cross partitions or contain duplicates")
    return manifest, rows


def pooled_features(backbone, rows, batch_size):
    """Preserve row order while batching equal-length proteins; no padding in ProGen."""
    buckets = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[len(row["sequence"])].append(index)
    features = torch.empty(len(rows), backbone.hidden_size)
    for length, indices in sorted(buckets.items()):
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            features[selected] = backbone.sequence_features(
                [rows[index]["sequence"] for index in selected]
            ).cpu()
        print(f"  extracted {len(indices)} proteins of length {length}", flush=True)
    return features


def selection_metrics(predictions, labels, fraction):
    """Top-fraction selection versus uniform candidates; not a ProGen sampling baseline."""
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    count = max(1, math.ceil(fraction * len(labels)))
    # Average ties at the cutoff instead of letting row ordering favor one parent.
    cutoff = np.sort(predictions)[-count]
    above, tied = predictions > cutoff, predictions == cutoff
    selected_mean = float(
        (labels[above].sum() + (count - above.sum()) * labels[tied].mean()) / count
    )
    baseline = float(labels.mean())
    return {
        "selected_count": count,
        "uniform_measured_mean": baseline,
        "selected_measured_mean": selected_mean,
        "lift_over_uniform": selected_mean - baseline,
    }


def panel_metrics(predictions, rows, fraction):
    """Report pooled, parent-balanced and reference-only prediction/selection quality."""
    predictions = np.asarray(predictions, dtype=float)
    labels = np.array([row["stability"] for row in rows])
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["parent"]].append(index)
    per_parent = []
    for parent, indices in sorted(groups.items()):
        per_parent.append(
            {
                "parent": parent,
                **regression_metrics(predictions[indices], labels[indices]),
                **selection_metrics(predictions[indices], labels[indices], fraction),
            }
        )
    correlations = [
        row["spearman"] for row in per_parent if row["spearman"] is not None
    ]
    reference_indices = [index for index, row in enumerate(rows) if row["is_reference"]]
    return {
        "pooled": {
            **regression_metrics(predictions, labels),
            **selection_metrics(predictions, labels, fraction),
            "prediction_bias": float((predictions - labels).mean()),
        },
        "macro_parent": {
            "rmse": float(np.mean([row["rmse"] for row in per_parent])),
            "spearman": float(np.mean(correlations)) if correlations else None,
            "defined_spearman_parents": len(correlations),
            "selection_lift": float(
                np.mean([row["lift_over_uniform"] for row in per_parent])
            ),
        },
        "reference_only": {
            **regression_metrics(
                predictions[reference_indices], labels[reference_indices]
            ),
            **selection_metrics(
                predictions[reference_indices], labels[reference_indices], fraction
            ),
        },
        "per_parent": per_parent,
    }


def evaluate_multiprotein(run, output):
    """CPU-only replay of metrics from saved predictions and post-fit labels."""
    if output.exists():
        raise FileExistsError(output)
    for name, expected in json.loads(
        (run / "evaluation_inputs.json").read_text()
    ).items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Evaluation input changed: {name}")
    config = json.loads((run / "config.json").read_text())
    rows = json.loads((run / "evaluation_labels.json").read_text())
    predictions = json.loads((run / "predictions.json").read_text())
    if predictions["sequences"] != [row["sequence"] for row in rows]:
        raise ValueError("Prediction and measurement ordering differ")
    metrics = {
        name: panel_metrics(values, rows, config["selection_fraction"])
        for name, values in predictions["models"].items()
    }
    output.mkdir(parents=True)
    write_json(output / "results.json", metrics)
    lines = [
        "# Multi-protein stability predictor benchmark",
        "",
        "Held-out protein groups; developmental evidence, not unrestricted generation.",
        "Stability and errors are in kcal/mol. Selection lift compares the predicted top 10% with uniform selection of the same measured candidates; it is not a ProGen/GFN gain.",
        "",
        "| Predictor | Pooled RMSE | Pooled Spearman | Bias | Macro parent RMSE | Macro parent Spearman | Macro selection lift | Reference-only RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics.items():
        pooled, macro, reference = (
            result["pooled"],
            result["macro_parent"],
            result["reference_only"],
        )

        def corr(value):
            return "undefined" if value is None else f"{value:.3f}"

        lines.append(
            f"| {name} | {pooled['rmse']:.3f} | {corr(pooled['spearman'])} | {pooled['prediction_bias']:+.3f} | {macro['rmse']:.3f} | {corr(macro['spearman'])} | {macro['selection_lift']:+.3f} | {reference['rmse']:.3f} |"
        )
    lines += [
        "",
        "results.json contains every parent, reference-only selection and pooled selection. Ten evaluation parents are a small feasibility panel, not ten thousand independent sequences.",
        "Validation alone selected penalties and MLP checkpoints; all test models are shown. No model is automatically selected using this report.",
        "Mean-pooled frozen ProGen features, fixed candidate measurements, no GFlowNet training, generation or assay validation.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return metrics


def run_multiprotein(config, output, device, smoke=False):
    """Extract frozen features, fit matched predictors, then open evaluation labels."""
    if (smoke and device != "cpu") or (not smoke and not device.startswith("cuda")):
        raise ValueError(
            "Real runs require user CUDA execution; CPU fitting is synthetic-smoke only"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest, rows = check_multiprotein(config)
    if smoke != manifest["synthetic"]:
        raise ValueError(
            "Smoke requires synthetic data; real run requires measured data"
        )
    effective = copy.deepcopy(config)
    effective.update(device=device, smoke=smoke)
    if smoke:
        effective["scorer"].update(
            hidden_size=8, steps=4, batch_size=4, validate_every=2
        )
        effective["mlp_seeds"] = [0, 1]
    output = new_run(output, effective)
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(
            ROOT / "docs/experiments/multiprotein_2026-11-10.md", output / "protocol.md"
        )
        write_json(output / "data_manifest.json", manifest)
        # Sealed evaluation label file is hashed in preflight but is not parsed during fitting.
        backbone = (
            FrozenProGen.tiny()
            if smoke
            else FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, "float32"
            )
        )
        features = {
            split: pooled_features(backbone, partition, config["feature_batch_size"])
            for split, partition in rows.items()
        }
        if any(parameter.requires_grad for parameter in backbone.model.parameters()):
            raise ValueError("ProGen must remain frozen")
        write_json(
            output / "feature_cost.json",
            {"backbone_work": backbone.work, "seconds": time.perf_counter() - started},
        )
        del backbone
        torch.save(
            {
                "features": features,
                "identities": {
                    split: [
                        {
                            key: row[key]
                            for key in ["sequence", "parent", "group", "is_reference"]
                        }
                        for row in partition
                    ]
                    for split, partition in rows.items()
                },
            },
            output / "features.pt",
        )
        labels = {
            split: [row["stability"] for row in rows[split]]
            for split in ["train", "validation"]
        }
        train_features = features["train"].to(device)
        validation_features = features["validation"].to(device)
        test_features = features["test"].to(device)
        predictions = {
            "sequences": [row["sequence"] for row in rows["test"]],
            "models": {
                "constant": [float(np.mean(labels["train"]))] * len(rows["test"])
            },
        }
        fitting = {}
        conditions = [("ridge", 0)] + [("mlp", seed) for seed in effective["mlp_seeds"]]
        for family, seed in conditions:
            name = f"{family}-seed-{seed}"
            print(f"Fitting {name} on {len(labels['train'])} sequences", flush=True)
            model, fit = fit_scorer(
                train_features,
                labels["train"],
                validation_features,
                labels["validation"],
                effective["scorer"],
                seed,
                family,
            )
            with torch.no_grad():
                predictions["models"][name] = model(test_features).cpu().tolist()
            torch.save(
                {
                    "state_dict": {
                        key: value.cpu() for key, value in model.state_dict().items()
                    },
                    "hidden_size": model.hidden_size,
                    "feature_size": train_features.shape[1],
                },
                output / f"{name}.pt",
            )
            fitting[name] = fit
            write_json(output / "fitting.json", fitting)
        write_json(output / "predictions.json", predictions)
        data = project_path(config["data"])
        if (
            file_hash(data / "test_labels.json")
            != manifest["file_sha256"]["test_labels.json"]
        ):
            raise ValueError("Held-out labels changed during fitting")
        shutil.copy2(data / "test_labels.json", output / "evaluation_labels.json")
        names = ["config.json", "predictions.json", "evaluation_labels.json"]
        write_json(
            output / "evaluation_inputs.json",
            {name: file_hash(output / name) for name in names},
        )
        write_json(
            output / "provenance.json",
            {
                "source_config": config,
                "torch": torch.__version__,
                "frozen_backbone": True,
                "feature_sha256": file_hash(output / "features.pt"),
                "scope": manifest["scope"],
            },
        )
        evaluate_multiprotein(output, output / "evaluation")
        shutil.copy2(output / "evaluation/report.md", output / "report.md")
        write_json(
            output / "status.json",
            {
                "status": "smoke_complete" if smoke else "complete",
                "scope": "development",
                "total_seconds": time.perf_counter() - started,
            },
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    print(output / "report.md", flush=True)
