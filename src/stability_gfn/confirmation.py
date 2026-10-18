"""Fixed five-parent confirmation, with outcomes revealed only after all fits."""

import copy
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import TOKENIZER_PATH, FrozenProGen, cache_tree
from .confirmation_data import read_confirmation_training
from .confirmation_evaluation import evaluate_confirmation
from .environment import PrefixTree, SequenceContext
from .evaluation import load_task
from .pipeline import (
    ROOT,
    extract_features,
    progress_recorder,
    project_path,
    save_head,
    validate_backbone,
    validate_config,
)
from .scorer import fit_scorer, stability_scores
from .training import train_supervised, train_tb


def validate_confirmation_config(config):
    """Keep the primary scorer/prior and matched proposal budgets explicit."""
    if config["experiment"] != "confirmation":
        raise ValueError("Expected confirmation configuration")
    equivalent = copy.deepcopy(config)
    equivalent["experiment"] = "biological"
    validate_config(equivalent)
    if config["alpha"] != 1 or config["training"]["beta"] != 1:
        raise ValueError("Confirmation primary fixes alpha=1 and beta=1")
    if config["scorer_seed"] != 0 or config["scorer"]["ridge_penalties"]:
        raise ValueError("Confirmation primary uses MLP seed zero only")
    if config["precision"] != "float32":
        raise ValueError("Retain float32 for the validated backbone")
    evaluation = config["evaluation"]
    if evaluation["raw_count"] != evaluation["filter_pool_count"]:
        raise ValueError("All methods must use the same proposal budget")
    for key in (
        "hidden_size",
        "steps",
        "batch_size",
        "learning_rate",
        "validate_every",
        "gradient_clip",
    ):
        if config["scorer"][key] <= 0:
            raise ValueError(f"Invalid scorer {key}")


def check_confirmation_inputs(config, smoke=False):
    """Check metadata/checksums without reading any reserved stability values."""
    panel = project_path(config["panel"])
    support = project_path(config["candidate_support"])
    manifest = json.loads((panel / "manifest.json").read_text())
    supports = json.loads((support / "manifest.json").read_text())
    if bool(manifest["synthetic"]) != smoke or bool(supports["synthetic"]) != smoke:
        raise ValueError(
            "Smoke requires synthetic inputs; real confirmation requires measured panel"
        )
    if supports["panel_manifest_sha256"] != file_hash(panel / "manifest.json"):
        raise ValueError("Support belongs to a different panel")
    if supports["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Support source differs from panel source")
    entries = [
        entry for entry in manifest["parents"] if entry["role"] == "confirmation"
    ]
    if not entries or (not smoke and len(entries) != 5):
        raise ValueError("Real confirmation must retain all five reserved parents")
    if [entry["id"] for entry in entries] != [
        entry["id"] for entry in supports["parents"]
    ]:
        raise ValueError(
            "Candidate support must retain every confirmation parent in order"
        )
    for entry, support_entry in zip(entries, supports["parents"]):
        for split in ("fit", "inner_validation"):
            path = panel / entry["id"] / f"{split}.json"
            if file_hash(path) != entry["file_sha256"][path.name]:
                raise ValueError(f"Changed input {path}")
        if file_hash(support / f"{entry['id']}.json") != support_entry["sha256"]:
            raise ValueError("Changed candidate identity file")
    if not smoke:
        gate_dir = project_path(config["required_exact_run"])
        gate = json.loads((gate_dir / "gate.json").read_text())
        gate_status = json.loads((gate_dir / "status.json").read_text())
        if (
            not gate.get("passed")
            or gate.get("smoke")
            or gate.get("experiment") != "exact"
            or gate_status["status"] != "complete"
        ):
            raise ValueError("Require a completed real exact gate")
        development = project_path(config["required_development_run"])
        status = json.loads((development / "status.json").read_text())
        if status["status"] != "complete" or status["smoke"]:
            raise ValueError("Require the completed real development diagnostic")
        saved_panel = json.loads((development / "panel_manifest.json").read_text())
        if saved_panel != manifest:
            raise ValueError("Confirmation panel differs from development reservation")
    return manifest, supports, entries


def cache_confirmation_parent(
    backbone, config, entry, support_entry, destination, device
):
    """Select one MLP with inner labels, cache predicted rewards and legal prefixes."""
    started = time.perf_counter()
    rows, sequences = read_confirmation_training(
        project_path(config["panel"]),
        project_path(config["candidate_support"]),
        entry,
        support_entry,
    )
    destination.mkdir()
    write_json(destination / "config.json", config)
    for split, values in rows.items():
        write_json(destination / f"{split}.json", values)
    write_json(destination / "candidates.json", sequences)
    print(f"{entry['parent']}: frozen features and MLP fitting", flush=True)
    feature_started = time.perf_counter()
    train_sequences = [row["sequence"] for row in rows["fit"]]
    inner_sequences = [row["sequence"] for row in rows["inner_validation"]]
    train_features = extract_features(
        backbone, train_sequences, config["feature_batch_size"]
    )
    inner_features = extract_features(
        backbone, inner_sequences, config["feature_batch_size"]
    )
    feature_seconds = time.perf_counter() - feature_started
    scorer, report = fit_scorer(
        train_features,
        [row["stability"] for row in rows["fit"]],
        inner_features,
        [row["stability"] for row in rows["inner_validation"]],
        config["scorer"],
        config["scorer_seed"],
    )
    report["selected_candidate"] = min(
        report["candidates"], key=lambda row: row["rmse"]
    )
    write_json(destination / "scorer_validation.json", report)
    torch.save(
        {
            "state_dict": {
                key: value.cpu() for key, value in scorer.state_dict().items()
            },
            "hidden_size": scorer.hidden_size,
        },
        destination / "scorer.pt",
    )
    reference_index = train_sequences.index(entry["reference"])
    reference_features = train_features[reference_index : reference_index + 1]
    candidate_started = time.perf_counter()
    candidate_features = extract_features(
        backbone, sequences, config["feature_batch_size"]
    )
    predictions, scores = stability_scores(
        scorer, candidate_features, reference_features
    )
    with torch.no_grad():
        reference_prediction = scorer(reference_features).item()
    candidate_seconds = time.perf_counter() - candidate_started
    tree = PrefixTree(SequenceContext(entry["reference"], 1), sequences)
    print(f"{entry['parent']}: caching {len(tree.prefixes)} prefixes", flush=True)
    prefix_started = time.perf_counter()
    features, logits = cache_tree(backbone, tree, config["feature_batch_size"])
    payload = {
        "context": asdict(tree.context),
        "sequences": tree.sequences,
        "features": features,
        "logits": logits,
        "scores": scores.cpu(),
        "predictions": predictions.cpu(),
        "reference_prediction": reference_prediction,
        "reference_label": rows["fit"][reference_index]["stability"],
        "training_sequences": train_sequences + inner_sequences,
        "evaluation_labels_sha256": entry["file_sha256"]["test_labels.json"],
    }
    torch.save(payload, destination / "task.pt")
    task, _ = load_task(destination, device="cpu")
    if abs(task.base_log_probabilities.double().exp().sum().item() - 1) > 1e-5:
        raise ValueError("Cached baseline probabilities do not normalize")
    costs = {
        "feature_seconds": feature_seconds,
        "scorer_seconds": report["training_seconds"],
        "candidate_scoring_seconds": candidate_seconds,
        "prefix_cache_seconds": time.perf_counter() - prefix_started,
        "prefix_count": len(tree.prefixes),
        "candidate_count": len(sequences),
        "setup_seconds": time.perf_counter() - started,
    }
    write_json(destination / "costs.json", {"setup": costs, "seeds": {}})
    return report["gate_passed"]


def fit_confirmation_heads(config, entry, destination, device, scorer_passed):
    """Retain failed screens, and otherwise use the unchanged fixed-step learners."""
    if not scorer_passed:
        write_json(
            destination / "fit_status.json",
            {"status": "scorer_gate_failed", "guidance_trained": False},
        )
        print(
            f"{entry['parent']}: scorer screen failed; retained in report, guidance skipped",
            flush=True,
        )
        return
    task, _ = load_task(destination, device)
    costs = json.loads((destination / "costs.json").read_text())
    for seed in config["seeds"]:
        folder = destination / f"seed-{seed}"
        folder.mkdir()
        print(f"{entry['parent']}: guidance seed {seed}", flush=True)
        head, log_z, history, cost = train_tb(
            task, config["training"], seed, progress_recorder(folder, "gfn")
        )
        save_head(folder / "gfn.pt", head, config["training"]["hidden_size"], log_z)
        write_json(folder / "gfn_history.json", history)
        head, history, supervised_cost = train_supervised(
            task, config["supervised"], seed, progress_recorder(folder, "supervised")
        )
        save_head(folder / "supervised.pt", head, config["supervised"]["hidden_size"])
        write_json(folder / "supervised_history.json", history)
        costs["seeds"][str(seed)] = {"gfn": cost, "supervised": supervised_cost}
        write_json(destination / "costs.json", costs)
    write_json(
        destination / "fit_status.json",
        {"status": "complete", "guidance_trained": True},
    )


def seal_confirmation_fits(output, entries, config):
    """Record all required fit artifacts before outcomes can be opened."""
    output = Path(output)
    hashes = {
        "config.json": file_hash(output / "config.json"),
        "panel_manifest.json": file_hash(output / "panel_manifest.json"),
    }
    for entry in entries:
        folder = output / entry["id"]
        status = json.loads((folder / "fit_status.json").read_text())
        if status["status"] not in ("complete", "scorer_gate_failed"):
            raise ValueError("All parents must finish fitting before evaluation")
        required = [
            "task.pt",
            "scorer.pt",
            "scorer_validation.json",
            "fit_status.json",
            "costs.json",
        ]
        if status["status"] == "complete":
            required += [
                f"seed-{seed}/{method}.pt"
                for seed in config["seeds"]
                for method in ("gfn", "supervised")
            ]
        for name in required:
            path = folder / name
            hashes[str(path.relative_to(output))] = file_hash(path)
    write_json(
        output / "fits_complete.json",
        {"parents": [entry["id"] for entry in entries], "file_sha256": hashes},
    )


def run_confirmation(config, output, device="cpu", smoke=False):
    """Fit every parent first, seal artifacts, and only then evaluate measurements."""
    config = copy.deepcopy(config)
    validate_confirmation_config(config)
    if smoke and device != "cpu":
        raise ValueError("Smoke uses CPU only")
    if not smoke and not device.startswith("cuda"):
        raise ValueError("Real experiments require explicit --device cuda")
    if smoke:
        config["seeds"] = [0]
        config["scorer"].update(hidden_size=8, steps=4, batch_size=8, validate_every=2)
        for method in ("training", "supervised"):
            config[method].update(hidden_size=16, steps=10, batch_size=8, log_every=5)
        config["supervised"]["rollouts"] = 16
        config["evaluation"].update(
            raw_count=16, filter_pool_count=16, selected_count=4
        )
    manifest, supports, entries = check_confirmation_inputs(config, smoke)
    config.update(smoke=smoke, device=device)
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
        write_json(output / "candidate_manifest.json", supports)
        provenance = {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tokenizer_sha256": file_hash(TOKENIZER_PATH),
            "source_sha256": {
                str(path.relative_to(output / "source")): file_hash(path)
                for path in sorted((output / "source").rglob("*"))
                if path.is_file()
            },
        }
        if not smoke:
            checkpoint = project_path(config["checkpoint"])
            hashes = {
                name: file_hash(checkpoint / name)
                for name in ("config.json", "pytorch_model.bin")
            }
            for previous in ("required_exact_run", "required_development_run"):
                previous_provenance = json.loads(
                    (project_path(config[previous]) / "provenance.json").read_text()
                )
                if hashes != previous_provenance["checkpoint_sha256"]:
                    raise ValueError("Checkpoint differs from the validated runs")
            provenance.update(checkpoint=str(checkpoint), checkpoint_sha256=hashes)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
            provenance["gpu_name"] = torch.cuda.get_device_name(device)
        write_json(output / "provenance.json", provenance)
        backbone = (
            FrozenProGen.tiny()
            if smoke
            else FrozenProGen.from_checkpoint(checkpoint, device, config["precision"])
        )
        write_json(
            output / "backbone_checks.json",
            validate_backbone(backbone, config["precision"]),
        )
        write_json(output / "backbone_config.json", backbone.model.config.to_dict())
        screens = {}
        for entry, support_entry in zip(entries, supports["parents"]):
            screens[entry["id"]] = cache_confirmation_parent(
                backbone, config, entry, support_entry, output / entry["id"], device
            )
        backbone_work = dict(backbone.work)
        if any(parameter.requires_grad for parameter in backbone.model.parameters()):
            raise RuntimeError("Backbone became trainable")
        del backbone
        for entry in entries:
            fit_confirmation_heads(
                config, entry, output / entry["id"], device, screens[entry["id"]]
            )
        seal_confirmation_fits(output, entries, config)
        write_json(
            output / "status.json",
            {"status": "fits_complete", "smoke": smoke, "labels_revealed": False},
        )
        evaluate_confirmation(output, output / "evaluation")
        costs = {
            "total_seconds": time.perf_counter() - started,
            "backbone_work": backbone_work,
        }
        if not smoke:
            costs["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        write_json(output / "costs.json", costs)
        write_json(
            output / "status.json",
            {
                "status": "smoke_complete" if smoke else "complete",
                "smoke": smoke,
                "labels_revealed": True,
                "parents": len(entries),
                "scorer_passed_parents": sum(screens.values()),
                "total_seconds": costs["total_seconds"],
            },
        )
        print(f"Confirmation complete: {output / 'evaluation/report.md'}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            output / "status.json",
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "smoke": smoke,
                "fits_sealed": (output / "fits_complete.json").exists(),
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        raise
