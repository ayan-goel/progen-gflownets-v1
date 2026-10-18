"""Reproducible experiment orchestration; only the user launches real runs."""

import copy
import json
import os
import resource
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tokenizers
import torch
import transformers

from .artifacts import file_hash, new_run, write_json
from .backbone import TOKENIZER_PATH, FrozenProGen, cache_tree
from .environment import PrefixTree, SequenceContext
from .evaluation import evaluate_run
from .guidance import CachedTask
from .metrics import distribution_metrics
from .scorer import fit_scorer, stability_scores
from .training import train_supervised, train_tb, validate_training_options

ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    """Interpret configured relative paths from the v1 project root, not shell cwd."""
    return (ROOT / path).resolve()


def validate_config(config):
    """Reject unsupported or contradictory experiments before loading model weights."""
    if config["experiment"] not in ("exact", "biological"):
        raise ValueError("Experiment must be exact or biological")
    if len(set(config["seeds"])) != len(config["seeds"]) or not config["seeds"]:
        raise ValueError("Need distinct nonempty seeds")
    for section in ["training", "supervised"]:
        settings = config[section]
        validate_training_options(settings)
        if (
            settings.get("monitor_exact_distribution", False)
            and config["experiment"] != "exact"
        ):
            raise ValueError(
                "Training-time exact monitoring is restricted to the exact toy"
            )
        for key in [
            "hidden_size",
            "steps",
            "batch_size",
            "learning_rate",
            "gradient_clip",
            "log_every",
        ]:
            if settings[key] <= 0:
                raise ValueError(f"{section}.{key} must be positive")
        if settings["beta"] < 0:
            raise ValueError("Guidance strength cannot be negative")
    if config["training"]["beta"] != config["supervised"]["beta"]:
        raise ValueError("Methods must use the same reward strength")
    evaluation = config["evaluation"]
    if (
        not 0
        < evaluation["selected_count"]
        <= evaluation["raw_count"]
        <= evaluation["filter_pool_count"]
    ):
        raise ValueError("Require 0 < selected_count <= raw_count <= filter_pool_count")
    if config["feature_batch_size"] <= 0 or config["cpu_threads"] <= 0:
        raise ValueError("Batch size and thread count must be positive")


def smoke_config(config):
    """A bounded fixture overrides costs, never pretends to satisfy scientific gates."""
    config = copy.deepcopy(config)
    config["smoke"] = True
    config["precision"] = "float32"
    config["seeds"] = [0]
    config["feature_batch_size"] = 8
    for section in ["training", "supervised"]:
        config[section].update(hidden_size=32, steps=20, batch_size=16, log_every=10)
    config["supervised"]["rollouts"] = 32
    config["evaluation"].update(raw_count=16, selected_count=4, filter_pool_count=32)
    if "scorer" in config:
        config["scorer"].update(hidden_size=8, steps=10, batch_size=8, validate_every=5)
    return config


def validate_backbone(backbone, precision):
    """Check causal cached/full alignment and frozen parameters on a short prefix."""
    states, logits = backbone.forward(["AGL"])
    tokens = backbone.encode(["AGL"])
    cache = None
    difference = 0.0
    tolerance = 1e-4 if precision == "float32" else 0.08
    for position in range(tokens.shape[1]):
        hidden, incremental, cache = backbone.step(
            tokens[:, position : position + 1], cache
        )
        difference = max(
            difference, (incremental - logits[:, position]).abs().max().item()
        )
        torch.testing.assert_close(
            incremental, logits[:, position], atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            hidden, states[:, position], atol=tolerance, rtol=tolerance
        )
    if backbone.model.training or any(
        parameter.requires_grad for parameter in backbone.model.parameters()
    ):
        raise RuntimeError("Backbone must remain frozen and in evaluation mode")
    return {
        "cache_max_logit_difference": difference,
        "tolerance": tolerance,
        "direction_marker_id": backbone.boundary_id[0],
        "hidden_size": backbone.hidden_size,
        "parameters": sum(
            parameter.numel() for parameter in backbone.model.parameters()
        ),
    }


def extract_features(backbone, sequences, batch_size):
    """Extract fixed complete-sequence features in equal-length batches."""
    rows = []
    for start in range(0, len(sequences), batch_size):
        rows.append(backbone.sequence_features(sequences[start : start + batch_size]))
    return torch.cat(rows).to(backbone.device)


def save_head(path, head, hidden_size, log_z=None):
    """Only learned head weights are saved; no duplicate pretrained checkpoints."""
    state = {
        "state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "hidden_size": hidden_size,
    }
    if log_z is not None:
        state["log_z"] = float(log_z)
    torch.save(state, path)


def progress_recorder(directory, method):
    """Flush observed training points even if a later update is interrupted."""

    def record(point):
        with (directory / f"{method}_progress.jsonl").open("a") as handle:
            handle.write(json.dumps(point, allow_nan=False) + "\n")
        loss_text = "n/a" if point["loss"] is None else f"{point['loss']:.5g}"
        message = f"{directory.name} {method}: step {point['step']}, loss {loss_text}"
        if "learning_rate" in point:
            message += f", lr {point['learning_rate']:.3g}"
        if "distribution" in point:
            message += f", TV {point['distribution']['total_variation']:.5f}"
        print(message, flush=True)

    return record


def run_experiment(config, output, device="cpu", smoke=False):
    """Run setup, fixed-step learners and evaluation, preserving failures as artifacts."""
    config = smoke_config(config) if smoke else copy.deepcopy(config)
    config["smoke"] = bool(smoke)
    config["device"] = device
    validate_config(config)
    if smoke and device != "cpu":
        raise ValueError("Smoke tests must use CPU")
    if not smoke and not device.startswith("cuda"):
        raise ValueError(
            "Real experiments require explicit --device cuda; use --smoke for CPU checks"
        )
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    if config["experiment"] == "biological" and not smoke:
        gate = json.loads(
            (project_path(config["required_exact_run"]) / "gate.json").read_text()
        )
        prior_status = json.loads(
            (project_path(config["required_exact_run"]) / "status.json").read_text()
        )
        if (
            not gate.get("passed")
            or gate.get("smoke")
            or gate.get("experiment") != "exact"
            or prior_status["status"] != "complete"
        ):
            raise ValueError(
                "The real exact ProGen experiment must pass before the biological run"
            )
    output = new_run(output, config)
    try:
        return _run(config, output, device, smoke)
    except KeyboardInterrupt:
        write_json(output / "status.json", {"status": "interrupted", "smoke": smoke})
        raise
    except Exception as error:
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "smoke": smoke,
            },
        )
        raise


def _run(config, output, device, smoke):
    """Inner run body; exceptions are recorded by the public wrapper."""
    print(
        f"Preparing {config['experiment']} {'CPU smoke test' if smoke else 'experiment'}",
        flush=True,
    )
    started = time.perf_counter()
    shutil.copytree(
        ROOT / "src",
        output / "source" / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(ROOT / "configs", output / "source" / "configs")
    shutil.copy2(ROOT / "spec.md", output / "source" / "spec.md")
    provenance = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tokenizer_sha256": file_hash(TOKENIZER_PATH),
        "model": "tiny_random_fixture" if smoke else "ProGen2-medium",
        "source_sha256": {
            str(path.relative_to(output / "source")): file_hash(path)
            for path in sorted((output / "source").rglob("*"))
            if path.is_file()
        },
    }
    if not smoke:
        provenance["gpu_name"] = torch.cuda.get_device_name(device)
        checkpoint = project_path(config["checkpoint"])
        provenance["checkpoint"] = str(checkpoint)
        provenance["checkpoint_sha256"] = {
            name: file_hash(checkpoint / name)
            for name in ["config.json", "pytorch_model.bin"]
        }
        if config["experiment"] == "biological":
            exact_provenance = json.loads(
                (
                    project_path(config["required_exact_run"]) / "provenance.json"
                ).read_text()
            )
            if provenance["checkpoint_sha256"] != exact_provenance["checkpoint_sha256"]:
                raise ValueError(
                    "Biological checkpoint differs from the one that passed the exact gate"
                )
    write_json(output / "provenance.json", provenance)
    backbone = (
        FrozenProGen.tiny()
        if smoke
        else FrozenProGen.from_checkpoint(
            project_path(config["checkpoint"]), device, config["precision"]
        )
    )
    write_json(
        output / "backbone_checks.json",
        validate_backbone(backbone, config["precision"]),
    )
    write_json(output / "backbone_config.json", backbone.model.config.to_dict())
    setup_costs = {}
    if config["experiment"] == "exact":
        context = SequenceContext(
            config["reference"], config["mutation_budget"], config["alphabet"]
        )
        sequences = context.enumerate_sequences(config["maximum_sequences"])
        scores = torch.tensor(
            [-((sequence.count("G") - 2) ** 2) for sequence in sequences],
            dtype=torch.float32,
        )
        predictions = scores.clone()
        training_sequences = []
        reference_label = None
        reference_prediction = None
    else:
        task_dir = project_path(config["data_task"])
        manifest = json.loads((task_dir / "manifest.json").read_text())
        if smoke != bool(manifest.get("synthetic", False)):
            raise ValueError(
                "CPU biological smoke tests require synthetic data; real runs require measured data"
            )
        for name, expected in manifest["file_sha256"].items():
            if file_hash(task_dir / name) != expected:
                raise ValueError(f"Prepared input changed: {name}")
        (output / "data_manifest").mkdir()
        for name in [
            "manifest.json",
            "train.json",
            "validation.json",
            "candidates.json",
        ]:
            shutil.copy2(task_dir / name, output / "data_manifest" / name)
        context = SequenceContext(manifest["reference"], manifest["mutation_budget"])
        sequences = sorted(json.loads((task_dir / "candidates.json").read_text()))
        if len(sequences) > config["maximum_sequences"]:
            raise ValueError(
                "Too many candidate sequences for the initial finite benchmark"
            )
        training = json.loads((task_dir / "train.json").read_text())
        validation = json.loads((task_dir / "validation.json").read_text())
        training_sequences = [row["sequence"] for row in training]
        validation_sequences = [row["sequence"] for row in validation]
        if set(training_sequences) & set(validation_sequences) or set(sequences) & (
            set(training_sequences) | set(validation_sequences)
        ):
            raise ValueError("Sequence leakage across declared partitions")
        print("Extracting frozen features and validating stability scorer", flush=True)
        feature_started = time.perf_counter()
        train_features = extract_features(
            backbone, training_sequences, config["feature_batch_size"]
        )
        validation_features = extract_features(
            backbone, validation_sequences, config["feature_batch_size"]
        )
        setup_costs["scorer_feature_seconds"] = time.perf_counter() - feature_started
        scorer, scorer_report = fit_scorer(
            train_features,
            [row["stability"] for row in training],
            validation_features,
            [row["stability"] for row in validation],
            config["scorer"],
            config["scorer_seed"],
        )
        torch.save(
            {
                "state_dict": {
                    name: value.cpu() for name, value in scorer.state_dict().items()
                },
                "hidden_size": scorer.hidden_size,
                "feature_size": backbone.hidden_size,
            },
            output / "scorer.pt",
        )
        write_json(output / "scorer_validation.json", scorer_report)
        if not scorer_report["gate_passed"]:
            setup_costs["scorer_training"] = scorer_report
            setup_costs["total_setup_seconds"] = time.perf_counter() - started
            setup_costs["backbone_work"] = backbone.work
            write_json(output / "costs.json", {"setup": setup_costs, "seeds": {}})
            write_json(
                output / "status.json",
                {
                    "status": "scorer_gate_failed",
                    "smoke": smoke,
                    "message": "Guidance training was not started; inspect scorer_validation.json",
                },
            )
            return False
        candidate_started = time.perf_counter()
        candidate_features = extract_features(
            backbone, sequences, config["feature_batch_size"]
        )
        reference_features = train_features[
            training_sequences.index(context.reference) : training_sequences.index(
                context.reference
            )
            + 1
        ]
        predictions, scores = stability_scores(
            scorer, candidate_features, reference_features
        )
        with torch.no_grad():
            reference_prediction = scorer(reference_features).item()
        predictions = predictions.cpu()
        scores = scores.cpu()
        setup_costs["scorer_training"] = scorer_report
        setup_costs["candidate_reward_evaluations"] = len(sequences)
        setup_costs["candidate_scoring_seconds"] = (
            time.perf_counter() - candidate_started
        )
        reference_label = manifest["reference_stability"]
        shutil.copy2(task_dir / "test_labels.json", output / "evaluation_labels.json")
    tree = PrefixTree(context, sequences)
    print(
        f"Caching {len(tree.prefixes)} prefixes for {len(sequences)} candidates",
        flush=True,
    )
    cache_started = time.perf_counter()
    features, logits = cache_tree(backbone, tree, config["feature_batch_size"])
    setup_costs["prefix_cache_seconds"] = time.perf_counter() - cache_started
    setup_costs["prefix_count"] = len(tree.prefixes)
    setup_costs["prefix_input_tokens"] = sum(
        len(prefix) + 1 for prefix in tree.prefixes
    )
    setup_costs["backbone_work_including_checks_and_scorer_features"] = backbone.work
    if any(parameter.requires_grad for parameter in backbone.model.parameters()):
        raise RuntimeError("Unexpected trainable backbone parameter")
    del backbone
    task = CachedTask(tree, features, logits, scores, device)
    payload = {
        "context": asdict(context),
        "sequences": tree.sequences,
        "features": features,
        "logits": logits,
        "scores": scores,
        "predictions": predictions,
        "training_sequences": training_sequences,
        "reference_label": reference_label,
        "reference_prediction": reference_prediction,
    }
    if config["experiment"] == "biological":
        payload["evaluation_labels_sha256"] = file_hash(
            output / "evaluation_labels.json"
        )
    torch.save(payload, output / "task.pt")
    baseline = task.all_log_probabilities(task.head(config["training"]["hidden_size"]))
    torch.testing.assert_close(baseline, task.base_log_probabilities, atol=0, rtol=0)
    if abs(baseline.double().exp().sum().item() - 1) > 1e-5:
        raise ValueError("Constrained sequence probabilities do not normalize")
    setup_costs["total_setup_seconds"] = time.perf_counter() - started
    costs = {"setup": setup_costs, "seeds": {}}
    gates = []
    for seed in config["seeds"]:
        print(f"Training seed {seed}", flush=True)
        destination = output / f"seed-{seed}"
        destination.mkdir()
        head, log_z, history, cost = train_tb(
            task, config["training"], seed, progress_recorder(destination, "gfn")
        )
        save_head(
            destination / "gfn.pt", head, config["training"]["hidden_size"], log_z
        )
        write_json(destination / "gfn_history.json", history)
        diagnostics = distribution_metrics(
            task.all_log_probabilities(head),
            task.base_log_probabilities,
            task.scores,
            config["training"]["beta"],
            log_z,
        )
        write_json(destination / "gfn_distribution.json", diagnostics)
        gates.append({"seed": seed, **diagnostics})
        supervised, history, supervised_cost = train_supervised(
            task,
            config["supervised"],
            seed,
            progress_recorder(destination, "supervised"),
        )
        save_head(
            destination / "supervised.pt",
            supervised,
            config["supervised"]["hidden_size"],
        )
        write_json(destination / "supervised_history.json", history)
        costs["seeds"][str(seed)] = {"gfn": cost, "supervised": supervised_cost}
    if device.startswith("cuda"):
        costs["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    costs["process_peak_host_memory_kib"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    write_json(output / "costs.json", costs)
    passed = not smoke and (
        config["experiment"] != "exact"
        or all(row["total_variation"] < config["tv_threshold"] for row in gates)
    )
    write_json(
        output / "gate.json",
        {
            "passed": passed,
            "smoke": smoke,
            "experiment": config["experiment"],
            "seeds": gates,
            "zero_guidance_matches_prior": True,
            "beta_zero_normalizer": baseline.double().exp().sum().item(),
        },
    )
    evaluate_run(output, output / "evaluation", device="cpu")
    status = (
        "smoke_complete" if smoke else ("complete" if passed else "exact_gate_failed")
    )
    write_json(
        output / "status.json",
        {
            "status": status,
            "smoke": smoke,
            "total_seconds": time.perf_counter() - started,
        },
    )
    print(f"{status}: {output / 'evaluation/report.md'}", flush=True)
    return passed or smoke
