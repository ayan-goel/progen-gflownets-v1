"""User-run training-coverage comparison with fixed held-out proteins and ridge rule."""

import json
import shutil
import time

import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import TOKENIZER_PATH, FrozenProGen
from .coverage_data import read_source, validate_training_sets
from .multiprotein import panel_metrics, pooled_features
from .pipeline import ROOT, project_path
from .predictor_diagnostics import decompose_errors
from .regularization import fixed_ridge
from .scorer import StabilityScorer


def scaled_penalty(reference_penalty, training_rows, reference_rows):
    """Keep the coefficient on squared weights fixed relative to mean data loss."""
    if reference_penalty <= 0 or training_rows <= 0 or reference_rows <= 0:
        raise ValueError("Penalty and training sizes must be positive")
    return reference_penalty * training_rows / reference_rows


def check_coverage(config):
    """Verify source/features, additions and model files on CPU without inference."""
    data = project_path(config["data"])
    if file_hash(data / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("Coverage manifest changed")
    manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in manifest["file_sha256"].items():
        if file_hash(data / name) != digest:
            raise ValueError(f"Coverage input changed: {name}")
    source, original, source_manifest, files = read_source(manifest["source_config"])
    if manifest["synthetic"] != source_manifest["synthetic"]:
        raise ValueError("Synthetic/real source mismatch")
    fitting = json.loads((source / "fitting.json").read_text())["ridge-seed-0"]
    choice = min(fitting["candidates"], key=lambda row: row["rmse"])
    if float(choice["model"].removeprefix("ridge_")) != config["reference_penalty"]:
        raise ValueError("Reference penalty differs from original saved ridge")
    additional = json.loads((data / "additional_train.json").read_text())
    stages = json.loads((data / "training_sets.json").read_text())
    validate_training_sets(
        files["train.json"],
        additional,
        files["validation.json"] + files["test_identities.json"],
        stages,
        manifest["variants_per_parent"],
    )
    cache = torch.load(source / "features.pt", map_location="cpu", weights_only=False)
    for split in ["train", "validation"]:
        rows = files[f"{split}.json"]
        if (
            len(cache["features"][split]) != len(rows)
            or not torch.isfinite(cache["features"][split]).all()
        ):
            raise ValueError("Cached feature size or values invalid")
        for row, identity in zip(rows, cache["identities"][split], strict=True):
            if any(row[key] != value for key, value in identity.items()):
                raise ValueError("Cached feature ordering changed")
    if not manifest["synthetic"]:
        for name, digest in original["checkpoint_sha256"].items():
            if file_hash(project_path(original["checkpoint"]) / name) != digest:
                raise ValueError("ProGen checkpoint changed")
        if file_hash(TOKENIZER_PATH) != original["tokenizer_sha256"]:
            raise ValueError("Tokenizer changed")
    return source, original, manifest, files, additional, stages, cache


def write_report(output, results):
    """Show all fixed coverage stages and their own training-mean controls."""
    lines = [
        "# Training-protein coverage",
        "",
        "Same validation proteins and candidate sequences at every stage. Test labels are not evaluated.",
        "Original ridge regularization per training example is fixed; raw penalty scales with training rows.",
        "",
        "| Training parents | Raw penalty | Ridge RMSE | Constant RMSE | Within-parent Spearman | Selection lift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for count, stage in results.items():
        metrics = stage["ridge"]["metrics"]
        correlation = metrics["macro_parent"]["spearman"]
        corr_text = "undefined" if correlation is None else f"{correlation:.4f}"
        lines.append(
            f"| {count} | {stage['penalty']:.3f} | {metrics['pooled']['rmse']:.4f} | "
            f"{stage['constant']['metrics']['pooled']['rmse']:.4f} | {corr_text} | "
            f"{metrics['macro_parent']['selection_lift']:+.4f} |"
        )
    lines += [
        "",
        "RMSE and lift are kcal/mol. Lift compares predicted top 10% against uniform candidates within each validation protein, not ProGen.",
        "",
        "No validation-based model or stage selection. One deterministic nested training expansion; changes reflect these added groups, not a universal data-size effect.",
        "",
        "Validation was previously inspected. These are developmental results, not proof of broad generation or experimental validation of new proteins.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_coverage(config, output, device, smoke=False):
    """Extract only added training features, fit all stages, then report validation."""
    if (smoke and device != "cpu") or (not smoke and not device.startswith("cuda")):
        raise ValueError(
            "Real runs require user CUDA execution; CPU fitting is smoke-only"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source, original, manifest, files, additional, stages, cache = check_coverage(
        config
    )
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
            ROOT / "docs/experiments/coverage_2026-11-10.md", output / "protocol.md"
        )
        write_json(output / "data_manifest.json", manifest)
        write_json(output / "training_sets.json", stages)
        write_json(output / "additional_train.json", additional)
        write_json(output / "base_train.json", files["train.json"])
        backbone = (
            FrozenProGen.tiny()
            if smoke
            else FrozenProGen.from_checkpoint(
                project_path(original["checkpoint"]), device, "float32"
            )
        )
        extra_features = pooled_features(
            backbone, additional, config["feature_batch_size"]
        )
        if any(parameter.requires_grad for parameter in backbone.model.parameters()):
            raise ValueError("ProGen must remain frozen")
        write_json(
            output / "feature_cost.json",
            {"seconds": time.perf_counter() - started, "backbone_work": backbone.work},
        )
        del backbone
        torch.save(
            {
                "features": extra_features,
                "sequences": [row["sequence"] for row in additional],
            },
            output / "additional_features.pt",
        )
        all_rows = files["train.json"] + additional
        all_features = torch.cat([cache["features"]["train"], extra_features]).to(
            device
        )
        labels = np.array([row["stability"] for row in all_rows])
        models, fits = {}, {}
        for count, parents in stages.items():
            parent_set = set(parents)
            indices = [
                index
                for index, row in enumerate(all_rows)
                if row["parent"] in parent_set
            ]
            stage_features = all_features[torch.tensor(indices, device=device)]
            stage_labels = labels[indices]
            penalty = scaled_penalty(
                config["reference_penalty"], len(indices), len(files["train.json"])
            )
            print(
                f"Fitting {count} parents, {len(indices)} sequences, ridge penalty {penalty:.6g}",
                flush=True,
            )
            model = fixed_ridge(
                stage_features, stage_labels, stage_features, stage_labels, penalty
            )
            models[count] = model
            fits[count] = {
                "penalty": penalty,
                "training_rows": len(indices),
                "training_mean": float(stage_labels.mean()),
            }
            torch.save(
                {
                    "state_dict": {
                        key: value.cpu() for key, value in model.state_dict().items()
                    },
                    "feature_size": all_features.shape[1],
                    "hidden_size": 0,
                    **fits[count],
                },
                output / f"ridge-{count}.pt",
            )
        write_json(output / "fitting_complete.json", fits)
        # No validation information is passed to any fit or stage selection.
        validation = files["validation.json"]
        write_json(output / "validation.json", validation)
        val_features = cache["features"]["validation"].to(device)
        results, predictions = {}, {}
        for count, model in models.items():
            with torch.no_grad():
                predicted = model(val_features).cpu().numpy()
            constant = np.full(len(validation), fits[count]["training_mean"])
            predictions[count] = {
                "ridge": predicted.tolist(),
                "constant": constant.tolist(),
            }
            results[count] = {
                **fits[count],
                **{
                    name: {
                        "metrics": panel_metrics(
                            values, validation, config["selection_fraction"]
                        ),
                        "decomposition": decompose_errors(
                            values, validation, fits[count]["training_mean"]
                        ),
                    }
                    for name, values in [("ridge", predicted), ("constant", constant)]
                },
            }
        baseline = next(iter(stages))
        saved = torch.load(
            source / "ridge-seed-0.pt", map_location="cpu", weights_only=False
        )
        reference = StabilityScorer(saved["feature_size"], saved["hidden_size"])
        reference.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            replay = reference(cache["features"]["validation"]).numpy()
        difference = float(
            np.max(np.abs(replay - np.array(predictions[baseline]["ridge"])))
        )
        if difference > 1e-4:
            raise ValueError("Baseline refit does not reproduce the original ridge")
        write_json(
            output / "baseline_check.json", {"max_prediction_difference": difference}
        )
        write_json(output / "validation_predictions.json", predictions)
        write_json(output / "validation_results.json", results)
        write_json(
            output / "provenance.json",
            {
                "torch": torch.__version__,
                "additional_features_sha256": file_hash(
                    output / "additional_features.pt"
                ),
                "source_sha256": manifest["source_config"]["source_sha256"],
            },
        )
        write_report(output, results)
        write_json(
            output / "status.json",
            {
                "status": "smoke_complete" if smoke else "complete",
                "total_seconds": time.perf_counter() - started,
                "test_evaluated": False,
                "scope": "development",
            },
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    print(output / "report.md")
