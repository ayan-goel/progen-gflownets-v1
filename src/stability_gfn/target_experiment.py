"""User-run development diagnostic: frozen features, scorers, then exact evaluation."""

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import Bio
import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import TOKENIZER_PATH, FrozenProGen
from .panel import read_development_parent
from .pipeline import ROOT, project_path, validate_backbone
from .scorer import fit_scorer
from .target_diagnostics import evaluate_inputs, extract_diagnostic_cache


def validate_target_config(config):
    """Reject unsupported settings before any model load or output creation."""
    if config["experiment"] != "target-diagnostic":
        raise ValueError("Expected target-diagnostic experiment")
    if config["precision"] != "float32":
        raise ValueError("The diagnostic uses the validated float32 backbone")
    seeds = config["ensemble_seeds"]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError("Diagnostic ensemble is fixed to seeds 0 through 4")
    alphas = config["alphas"]
    if (
        not alphas
        or len(set(alphas)) != len(alphas)
        or any(not np.isfinite(a) or not 0 < a <= 1 for a in alphas)
    ):
        raise ValueError("Need distinct sequence exponents in (0, 1]")
    if not np.isfinite(config["beta"]) or config["beta"] < 0:
        raise ValueError("Beta must be finite and nonnegative")
    if not 0 < config["selected_count"] <= config["raw_count"]:
        raise ValueError("Need 0 < selected_count <= raw_count")
    for key in ("feature_batch_size", "cpu_threads"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    scorer = config["scorer"]
    for key in (
        "hidden_size",
        "steps",
        "batch_size",
        "learning_rate",
        "gradient_clip",
        "validate_every",
    ):
        if scorer[key] <= 0:
            raise ValueError(f"scorer.{key} must be positive")
    if not scorer["ridge_penalties"] or any(p <= 0 for p in scorer["ridge_penalties"]):
        raise ValueError("Need positive ridge penalties")


def fit_diagnostic_scorers(cache, rows, reference, config, destination, device):
    """Fit only on inner-fit labels; select checkpoints only on inner validation."""
    features = {key: value["features"].to(device) for key, value in cache.items()}
    labels = {
        key: [row["stability"] for row in rows[key]]
        for key in ("fit", "inner_validation")
    }
    reference_index = [row["sequence"] for row in rows["fit"]].index(reference)
    reference_features = features["fit"][reference_index : reference_index + 1]
    reports = {}
    predictions = {}
    reference_predictions = {}
    started = time.perf_counter()
    specifications = [("ridge", 0)] + [
        (f"mlp-{seed}", seed) for seed in config["ensemble_seeds"]
    ]
    label_scale = None
    for name, seed in specifications:
        print(f"  Fitting {name} (inner validation only)", flush=True)
        settings = copy.deepcopy(config["scorer"])
        if name == "ridge":
            settings["steps"] = 0
        else:
            settings["ridge_penalties"] = []
        model, report = fit_scorer(
            features["fit"],
            labels["fit"],
            features["inner_validation"],
            labels["inner_validation"],
            settings,
            seed,
        )
        report["selected_candidate"] = min(
            report["candidates"], key=lambda row: row["rmse"]
        )
        report["fit_config"] = settings
        reports[name] = report
        label_scale = model.label_scale.item()
        with torch.no_grad():
            predictions[name] = model(features["development"]).cpu()
            reference_predictions[name] = model(reference_features).item()
        torch.save(
            {
                "state_dict": {
                    key: value.cpu() for key, value in model.state_dict().items()
                },
                "hidden_size": model.hidden_size,
                "seed": seed,
            },
            destination / f"{name}.pt",
        )
    members = torch.stack(
        [predictions[f"mlp-{seed}"] for seed in config["ensemble_seeds"]]
    )
    member_references = torch.tensor(
        [reference_predictions[f"mlp-{seed}"] for seed in config["ensemble_seeds"]]
    )
    scorers = {
        "ridge": {
            "predictions": predictions["ridge"],
            "reference_prediction": reference_predictions["ridge"],
        },
        "mlp": {
            "predictions": predictions["mlp-0"],
            "reference_prediction": reference_predictions["mlp-0"],
        },
        "ensemble": {
            "predictions": members.mean(0),
            "reference_prediction": member_references.mean().item(),
        },
    }
    write_json(destination / "scorer_fits.json", reports)
    return (
        scorers,
        members,
        member_references,
        label_scale,
        time.perf_counter() - started,
    )


def evaluate_target_run(run, output):
    """Reproduce exact metrics and categorical references using CPU artifacts only."""
    run, output = Path(run), Path(output)
    config = json.loads((run / "config.json").read_text())
    completed = json.loads((run / "diagnostic_manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    all_results = []
    lines = [
        "# Development target diagnostic",
        "",
        "Synthetic smoke test; not scientific evidence."
        if config["smoke"]
        else "Development analysis only; confirmation proteins and test labels remain reserved.",
        "",
        "No GFN was trained. Exact-target samples are privileged categorical references.",
        "All measured values and gains are kcal/mol. All parents and scorer failures are retained.",
        "",
        "| Parent | Scorer | Alpha | Prior measured | Target measured | Gain vs matched prior | Gain vs original prior | Target effective candidates |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in completed["parents"]:
        path = run / entry["id"] / "inputs.pt"
        if file_hash(path) != entry["inputs_sha256"]:
            raise ValueError(f"Changed diagnostic inputs: {path}")
        inputs = torch.load(path, map_location="cpu", weights_only=True)
        result = evaluate_inputs(inputs, config)
        result.update(
            parent=entry["parent"],
            id=entry["id"],
            previously_inspected_parent=entry["previously_inspected_parent"],
        )
        all_results.append(result)
        write_json(output / f"{entry['id']}.json", result)
        for row in result["comparisons"]:
            lines.append(
                f"| {entry['parent']} | {row['scorer']} | {row['alpha']:g} | "
                f"{row['baseline']['measured_mean']:.5f} | {row['target']['measured_mean']:.5f} | "
                f"{row['measured_gain_vs_matched_prior']:+.5f} | {row['measured_gain_vs_original_prior']:+.5f} | "
                f"{row['target']['effective_candidates']:.2f} |"
            )
    aggregates = []
    for scorer in ("ridge", "mlp", "ensemble"):
        for alpha in [*config["alphas"], 0.0]:
            rows = [
                next(
                    row
                    for row in result["comparisons"]
                    if row["scorer"] == scorer and row["alpha"] == alpha
                )
                for result in all_results
            ]
            gains = [row["measured_gain_vs_matched_prior"] for row in rows]
            aggregates.append(
                {
                    "scorer": scorer,
                    "alpha": alpha,
                    "parent_count": len(rows),
                    "mean_gain": float(np.mean(gains)),
                    "minimum_gain": min(gains),
                    "maximum_gain": max(gains),
                    "positive_parent_count": sum(value > 0 for value in gains),
                    "new_parent_count": sum(
                        not item["previously_inspected_parent"] for item in all_results
                    ),
                    "positive_new_parent_count": sum(
                        gain > 0 and not item["previously_inspected_parent"]
                        for gain, item in zip(gains, all_results)
                    ),
                }
            )
    lines.extend(
        [
            "",
            "Alpha=0 is uniform-over-candidates diagnostic only. Alpha=1 is the original prior.",
            "Review stability gains alongside prior concentration, weighted error, inner-fit screens and duplicate counts in the JSON files.",
            "Selection metrics use 256 proposals / 32 slots in real runs, with duplicates retained for both policies.",
            "Do not treat positive development means as confirmation or silently select a winning protein.",
            "Next: review all development parents, freeze a procedure if supported, then implement fresh confirmation. No automatic next training job is launched.",
        ]
    )
    write_json(
        output / "summary.json",
        {
            "smoke": config["smoke"],
            "aggregates": aggregates,
            "decision": "Review required; no automatic model or prior selection",
            "parents": [
                {"id": item["id"], "parent": item["parent"]} for item in all_results
            ],
        },
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return aggregates


def run_target_diagnostic(config, output, device="cpu", smoke=False):
    """Execute the fixed development experiment; never launch a confirmation fit."""
    config = copy.deepcopy(config)
    if smoke and device != "cpu":
        raise ValueError("Smoke diagnostics must use CPU")
    if not smoke and not device.startswith("cuda"):
        raise ValueError("Real diagnostics require explicit --device cuda")
    if smoke:
        config["scorer"].update(hidden_size=8, steps=4, batch_size=8, validate_every=2)
        config.update(feature_batch_size=8, raw_count=16, selected_count=4)
    validate_target_config(config)
    panel = project_path(config["panel"])
    manifest = json.loads((panel / "manifest.json").read_text())
    if bool(manifest["synthetic"]) != bool(smoke):
        raise ValueError(
            "CPU smoke requires synthetic panel; real runs require real data"
        )
    if manifest["config"] != config["panel_preparation"]:
        raise ValueError("Prepared panel settings differ from run configuration")
    development = [
        entry for entry in manifest["parents"] if entry["role"] == "development"
    ]
    if len(development) != config["panel_preparation"]["development_parents"]:
        raise ValueError("Incorrect development parent count")
    config.update(smoke=smoke, device=device)
    # Check data hashes before loading weights; confirmation/test files are not opened.
    partitions = {
        entry["id"]: read_development_parent(panel, entry) for entry in development
    }
    exact_provenance = None
    if not smoke:
        gate_dir = project_path(config["required_exact_run"])
        gate = json.loads((gate_dir / "gate.json").read_text())
        status = json.loads((gate_dir / "status.json").read_text())
        if (
            not gate.get("passed")
            or gate.get("smoke")
            or gate.get("experiment") != "exact"
            or status["status"] != "complete"
        ):
            raise ValueError("Require the completed real exact gate")
        exact_provenance = json.loads((gate_dir / "provenance.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    output = new_run(output, config)
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source/src",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(ROOT / "spec.md", output / "source/spec.md")
        write_json(output / "panel_manifest.json", manifest)
        provenance = {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "biopython": Bio.__version__,
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tokenizer_sha256": file_hash(TOKENIZER_PATH),
            "panel_manifest_sha256": file_hash(panel / "manifest.json"),
            "source_sha256": {
                str(path.relative_to(output / "source")): file_hash(path)
                for path in sorted((output / "source").rglob("*"))
                if path.is_file()
            },
            "model": "tiny_random_fixture" if smoke else "ProGen2-medium",
        }
        if not smoke:
            checkpoint = project_path(config["checkpoint"])
            hashes = {
                name: file_hash(checkpoint / name)
                for name in ("config.json", "pytorch_model.bin")
            }
            if hashes != exact_provenance["checkpoint_sha256"]:
                raise ValueError("Checkpoint differs from the passed exact experiment")
            provenance.update(checkpoint=str(checkpoint), checkpoint_sha256=hashes)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
            provenance["gpu_name"] = torch.cuda.get_device_name(device)
        write_json(output / "provenance.json", provenance)
        print("Loading frozen backbone for development diagnostics", flush=True)
        backbone = (
            FrozenProGen.tiny()
            if smoke
            else FrozenProGen.from_checkpoint(checkpoint, device, config["precision"])
        )
        write_json(
            output / "backbone_checks.json",
            validate_backbone(backbone, config["precision"]),
        )
        completed = []
        costs = {}
        for entry in development:
            print(f"Development parent {entry['parent']} ({entry['id']})", flush=True)
            destination = output / entry["id"]
            destination.mkdir()
            rows = partitions[entry["id"]]
            rows["development"] = sorted(
                rows["development"], key=lambda row: row["sequence"]
            )
            for split, values in rows.items():
                write_json(destination / f"{split}.json", values)
            cache_start = time.perf_counter()
            cache = {
                split: extract_diagnostic_cache(
                    backbone,
                    values,
                    config["feature_batch_size"],
                    include_logits=split == "development",
                )
                for split, values in rows.items()
            }
            torch.save(cache, destination / "features.pt")
            cache_seconds = time.perf_counter() - cache_start
            scorers, members, member_references, scale, fit_seconds = (
                fit_diagnostic_scorers(
                    cache, rows, entry["reference"], config, destination, device
                )
            )
            reference_row = next(
                row for row in rows["fit"] if row["sequence"] == entry["reference"]
            )
            inputs = {
                "sequences": [row["sequence"] for row in rows["development"]],
                "labels": torch.tensor(
                    [row["stability"] for row in rows["development"]],
                    dtype=torch.float64,
                ),
                "reference": entry["reference"],
                "reference_label": reference_row["stability"],
                "path_logits": cache["development"]["path_logits"],
                "label_scale": scale,
                "scorers": scorers,
                "ensemble_members": members,
                "ensemble_reference_predictions": member_references,
            }
            torch.save(inputs, destination / "inputs.pt")
            costs[entry["id"]] = {
                "cache_seconds": cache_seconds,
                "scorer_seconds": fit_seconds,
            }
            completed.append(
                {**entry, "inputs_sha256": file_hash(destination / "inputs.pt")}
            )
            write_json(
                output / "progress.json",
                {
                    "completed_development_parents": [
                        item["parent"] for item in completed
                    ]
                },
            )
        if any(parameter.requires_grad for parameter in backbone.model.parameters()):
            raise RuntimeError("Backbone unexpectedly became trainable")
        costs["backbone_work"] = backbone.work
        if not smoke:
            costs["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        del backbone
        write_json(output / "diagnostic_manifest.json", {"parents": completed})
        evaluate_target_run(output, output / "evaluation")
        costs["total_seconds"] = time.perf_counter() - started
        write_json(output / "costs.json", costs)
        write_json(
            output / "status.json",
            {
                "status": "smoke_complete" if smoke else "complete",
                "smoke": smoke,
                "confirmation_run": False,
                "gfn_trained": False,
                "total_seconds": costs["total_seconds"],
            },
        )
        print(f"Diagnostic complete: {output / 'evaluation/report.md'}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            output / "status.json",
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "smoke": smoke,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
