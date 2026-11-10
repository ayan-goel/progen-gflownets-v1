"""Matched local resplits: predictor generalization and exact reward-target utility."""

import copy
import json
import shutil
import time

import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import TOKENIZER_PATH, FrozenProGen, cache_tree
from .environment import PrefixTree, SequenceContext
from .guidance import CachedTask
from .pipeline import ROOT, project_path
from .reward_study import compare_predictors


def mutation_positions(sequences, reference):
    """Return one-based changed positions; the unmutated reference has no group.

    Reject indels and multi-mutants rather than assigning them to an arbitrary
    group. Group assignment never accepts stability measurements.
    """
    positions = []
    for sequence in sequences:
        if len(sequence) != len(reference):
            raise ValueError("Position holdouts require equal-length single mutants")
        changes = [
            index + 1
            for index, (original, residue) in enumerate(zip(reference, sequence))
            if original != residue
        ]
        if len(changes) > 1:
            raise ValueError("Position holdouts require single mutants")
        positions.append(changes[0] if changes else None)
    return positions


def position_split_indices(sequences, reference, seed):
    """Shuffle observed mutation positions, keeping every substitution group intact."""
    if len(set(sequences)) != len(sequences) or sequences.count(reference) != 1:
        raise ValueError("Need unique sequence identities including one reference")
    positions = mutation_positions(sequences, reference)
    groups = sorted({position for position in positions if position is not None})
    shuffled = np.random.default_rng(seed).permutation(groups).tolist()
    train_end = int(0.70 * len(groups))
    validation_end = int(0.85 * len(groups))
    group_partitions = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    if any(not group for group in group_partitions.values()):
        raise ValueError("Need enough mutation positions for three nonempty partitions")
    partitions = {
        split: sorted(
            [index for index, position in enumerate(positions) if position in group],
            key=lambda index: sequences[index],
        )
        for split, group in group_partitions.items()
    }
    # The reference provides the shared reward anchor, not a substituted-position label.
    partitions["train"].insert(0, sequences.index(reference))
    return partitions


def split_indices(sequences, reference, seed):
    """Original 70/15/15 rule on sorted mutant identities, with reference in training.

    Sequence identities alone determine the split; labels cannot influence it.
    Sorting evaluation identities aligns rows with PrefixTree's canonical ordering.
    """
    if len(set(sequences)) != len(sequences) or sequences.count(reference) != 1:
        raise ValueError("Need unique sequence identities including one reference")
    variants = sorted(
        (i for i, s in enumerate(sequences) if s != reference),
        key=lambda i: sequences[i],
    )
    order = np.random.default_rng(seed).permutation(len(variants))
    shuffled = [variants[i] for i in order]
    train_end = int(0.70 * len(variants))
    validation_end = int(0.85 * len(variants))
    partitions = {
        "train": [sequences.index(reference)] + shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": sorted(shuffled[validation_end:], key=lambda i: sequences[i]),
    }
    if any(not indices for indices in partitions.values()):
        raise ValueError("All partitions must be nonempty")
    return partitions


def prepare_split_inputs(config):
    """Verify reusable feature/data provenance on CPU; no model execution or fitting."""
    split_unit = config.get("split_unit", "sequence")
    if split_unit not in {"sequence", "position"}:
        raise ValueError("split_unit must be sequence or position")
    seeds = config["split_seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Need distinct nonnegative split seeds")
    if type(config["predictor_seed"]) is not int or config["predictor_seed"] < 0:
        raise ValueError("Need a nonnegative predictor seed")
    for path, expected in config["input_sha256"].items():
        if file_hash(project_path(path)) != expected:
            raise ValueError(f"Input changed: {path}")
    pilot = project_path(config["pilot_run"])
    study = project_path(config["predictor_run"])
    required = {
        pilot / "config.json",
        pilot / "data_manifest/manifest.json",
        study / "features.pt",
        study / "provenance.json",
    }
    if not required <= {project_path(p) for p in config["input_sha256"]}:
        raise ValueError("Missing required data/source hashes")
    original = json.loads((pilot / "config.json").read_text())
    manifest = json.loads((pilot / "data_manifest/manifest.json").read_text())
    provenance = json.loads((study / "provenance.json").read_text())
    if manifest["split_seed"] in seeds:
        raise ValueError("New split seeds must exclude the original split seed")
    if file_hash(study / "features.pt") != provenance["feature_cache_sha256"]:
        raise ValueError("Pooled feature cache differs from original study")
    if not original.get("smoke", False):
        checkpoint = project_path(original["checkpoint"])
        for name, expected in provenance["checkpoint_sha256"].items():
            if file_hash(checkpoint / name) != expected:
                raise ValueError("ProGen checkpoint changed")
        if file_hash(TOKENIZER_PATH) != config["tokenizer_sha256"]:
            raise ValueError("Tokenizer changed")
    cache = torch.load(study / "features.pt", map_location="cpu", weights_only=True)
    rows = [
        r for split in ["train", "validation", "test"] for r in cache["rows"][split]
    ]
    features = torch.cat(
        [cache["features"][split] for split in ["train", "validation", "test"]]
    )
    sequences = [r["sequence"] for r in rows]
    if len(features) != len(rows) or not torch.isfinite(features).all():
        raise ValueError("Invalid pooled features")
    if not all(np.isfinite(r["stability"]) for r in rows):
        raise ValueError("Non-finite stability labels")
    original_indices = split_indices(
        sequences, manifest["reference"], manifest["split_seed"]
    )
    for split, indices in original_indices.items():
        if {sequences[i] for i in indices} != {
            r["sequence"] for r in cache["rows"][split]
        }:
            raise ValueError(
                "Cannot reconstruct the original split from cached identities"
            )
        if len(indices) != manifest["split_counts"][split]:
            raise ValueError("Original partition sizes differ from manifest")
    splitter = position_split_indices if split_unit == "position" else split_indices
    partitions = {
        seed: splitter(sequences, manifest["reference"], seed) for seed in seeds
    }
    original_test = set(original_indices["test"])
    readiness = {
        "parent": manifest["parent"],
        "sequence_count": len(rows),
        "original_split_sizes": {s: len(v) for s, v in original_indices.items()},
        "split_seeds": seeds,
        "predictor_seed": config["predictor_seed"],
        "beta": 1,
        "new_gfn_fits": 0,
        "evaluation_overlap_with_original": {
            str(seed): len(set(p["test"]) & original_test)
            for seed, p in partitions.items()
        },
        "smoke_source": original.get("smoke", False),
        "split_unit": split_unit,
        "partition_sizes": {
            str(seed): {split: len(indices) for split, indices in partition.items()}
            for seed, partition in partitions.items()
        },
    }
    if split_unit == "position":
        positions = mutation_positions(sequences, manifest["reference"])
        readiness["mutation_positions"] = {
            str(seed): {
                split: sorted({positions[index] for index in indices} - {None})
                for split, indices in partition.items()
            }
            for seed, partition in partitions.items()
        }
    return original, manifest, rows, features, partitions, readiness


def summarize_split_results(output, seeds):
    """Compare each target against the prior for its own support, not a global baseline."""
    combined = []
    config = json.loads((output / "config.json").read_text())
    split_unit = config.get("split_unit", "sequence")
    for seed in seeds:
        directory = output / f"split-{seed}"
        prior = torch.load(
            directory / "prior.pt", map_location="cpu", weights_only=True
        )
        probabilities = prior["log_prior"].double().softmax(0)
        for result in json.loads((directory / "results.json").read_text()):
            name = f"seed-{result['seed']}-n-{result['train_size']}-{result['family']}"
            predictions = json.loads(
                (directory / f"{name}-predictions.json").read_text()
            )
            if predictions["sequences"] != prior["sequences"]:
                raise ValueError("Prediction/prior sequence order mismatch")
            error = torch.tensor(
                predictions["predictions"], dtype=torch.float64
            ) - torch.tensor(predictions["measurements"], dtype=torch.float64)
            target = next(
                t
                for t in result["targets"]
                if t["reward"] == "linear" and t["beta"] == 1
            )
            scores = (
                torch.tensor(predictions["predictions"], dtype=torch.float64)
                - result["reference_prediction"]
            ) / result["scale"]
            target_probabilities = (prior["log_prior"].double() + scores).softmax(0)
            combined.append(
                {
                    "split_seed": seed,
                    "family": result["family"],
                    "predictor_seed": result["seed"],
                    "evaluation": result["evaluation"],
                    "prior_weighted_mae": float(probabilities @ error.abs()),
                    "prior_weighted_rmse": float(
                        (probabilities @ error.square()).sqrt()
                    ),
                    "reward_scale": result["scale"],
                    "prior_top_one_mass": float(probabilities.max()),
                    "target_top_one_mass": float(target_probabilities.max()),
                    "baseline": result["baseline"],
                    "target": target,
                    "measured_gain_over_own_progen": target["expected_measured_dg"]
                    - result["baseline"]["expected_measured_dg"],
                }
            )
    write_json(output / "comparison.json", combined)
    lines = [
        "# Mutation-position holdout study"
        if split_unit == "position"
        else "# Local split-robustness study",
        "",
        "New partitions of previously inspected variants; development, not independent confirmation.",
        "Exact targets at beta 1. No GFlowNet was trained in this study.",
        f"Partition unit: {split_unit}. Reference is retained in training.",
        "",
        "| Split seed | Predictor | RMSE | ProGen-weighted RMSE | ProGen expected measured stability | Target expected measured stability | Gain |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in combined:
        lines.append(
            f"| {r['split_seed']} | {r['family']} | {r['evaluation']['rmse']:.4f} | {r['prior_weighted_rmse']:.4f} | {r['baseline']['expected_measured_dg']:.5f} | {r['target']['expected_measured_dg']:.5f} | {r['measured_gain_over_own_progen']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive summary",
            "",
            "| Predictor | Positive-gain splits | Mean gain | Gain range |",
            "|---|---|---:|---|",
        ]
    )
    summary = []
    for family in ["constant", "ridge", "mlp"]:
        gains = [
            r["measured_gain_over_own_progen"]
            for r in combined
            if r["family"] == family
        ]
        item = {
            "family": family,
            "mean_gain": float(np.mean(gains)),
            "gain_range": [min(gains), max(gains)],
            "positive_gain_splits": sum(g > 1e-10 for g in gains),
            "split_count": len(gains),
        }
        summary.append(item)
        lines.append(
            f"| {family} | {item['positive_gain_splits']} / {len(gains)} | {item['mean_gain']:+.5f} | {min(gains):+.5f} to {max(gains):+.5f} |"
        )
    write_json(output / "summary.json", summary)
    if split_unit == "position":
        readiness = json.loads((output / "readiness.json").read_text())
        lines.extend(
            [
                "",
                "## Grouped partitions",
                "",
                "Position numbers are one-based. All variants at each position stay together; the reference is training-only.",
                "",
                "| Split | Train / validation / evaluation sequences | Evaluation mutation positions |",
                "|---|---|---|",
            ]
        )
        for seed in seeds:
            sizes = readiness["partition_sizes"][str(seed)]
            groups = readiness["mutation_positions"][str(seed)]
            lines.append(
                f"| {seed} | {sizes['train']} / {sizes['validation']} / {sizes['test']} | "
                + ", ".join(str(position) for position in groups["test"])
                + " |"
            )
    lines.extend(
        [
            "",
            "## Concentration, reference-level success and diversity",
            "",
            "Success is probability of measured dG at least the reference; unique is expected distinct sequences in 256 draws. Probabilities below are percentages.",
            "",
            "| Split | Predictor | Prior top-one % | Target top-one % | Prior success % | Target success % | Prior unique | Target unique |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in combined:
        baseline = result["baseline"]
        target = result["target"]
        lines.append(
            f"| {result['split_seed']} | {result['family']} | "
            f"{100 * result['prior_top_one_mass']:.3f} | "
            f"{100 * result['target_top_one_mass']:.3f} | "
            f"{100 * baseline['probability_at_least_reference']:.6f} | "
            f"{100 * target['probability_at_least_reference']:.6f} | "
            f"{baseline['expected_unique_in_256']:.3f} | "
            f"{target['expected_unique_in_256']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All stability values are kcal/mol. Predicted stability, success probabilities, diversity and top-20 selection metrics are retained in per-split results.json.",
            "The prior is locally normalized on each split's evaluation tree, so absolute baselines can change with support.",
            "Each partition gets a newly fitted predictor with its own training-only normalization. Hyperparameters/checkpoints use validation labels only.",
            "Splits overlap and share one parent. One fixed MLP seed per split does not characterize optimization variability.",
            "No reward or model is selected using these evaluation outcomes; inspect all three splits, not just mean gain.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return combined


def run_split_study(config, output, device, smoke=False):
    """Fit matched predictors, recompute each support's frozen prior, evaluate targets."""
    if smoke and device != "cpu":
        raise ValueError("Smoke tests require CPU")
    if not smoke and not device.startswith("cuda"):
        raise ValueError(
            "Real experiments are user-run on CUDA; use check-only for readiness"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original, manifest, rows, features, partitions, readiness = prepare_split_inputs(
        config
    )
    if bool(readiness["smoke_source"]) != smoke:
        raise ValueError(
            "Smoke requires synthetic inputs; real runs require measured inputs"
        )
    effective = copy.deepcopy(config)
    effective.update(
        scorer=copy.deepcopy(original["scorer"]), beta=1, device=device, smoke=smoke
    )
    split_unit = config.get("split_unit", "sequence")
    if smoke:
        effective["scorer"].update(
            hidden_size=8, steps=4, batch_size=4, validate_every=2
        )
    output = new_run(output, effective)
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        protocol = (
            "position_holdout_2026-11-09.md"
            if split_unit == "position"
            else "split_robustness_2026-11-09.md"
        )
        shutil.copy2(ROOT / "docs/experiments" / protocol, output / "protocol.md")
        write_json(output / "readiness.json", readiness)
        # Save every partition before fitting. Labels are inherited, never used to choose splits.
        for seed, partition in partitions.items():
            directory = output / f"split-{seed}"
            directory.mkdir()
            write_json(
                directory / "partitions.json",
                {s: [rows[i] for i in indices] for s, indices in partition.items()},
            )
            if split_unit == "position":
                write_json(
                    directory / "mutation_positions.json",
                    readiness["mutation_positions"][str(seed)],
                )
        print(
            "Loading frozen ProGen to compute each new candidate-set prior", flush=True
        )
        backbone = (
            FrozenProGen.tiny()
            if smoke
            else FrozenProGen.from_checkpoint(
                project_path(original["checkpoint"]), device, original["precision"]
            )
        )
        for seed, partition in partitions.items():
            directory = output / f"split-{seed}"
            split_rows = {
                s: [rows[i] for i in indices] for s, indices in partition.items()
            }
            split_features = {s: features[indices] for s, indices in partition.items()}
            context = SequenceContext(
                manifest["reference"], manifest["mutation_budget"]
            )
            tree = PrefixTree(context, [r["sequence"] for r in split_rows["test"]])
            print(
                f"Split {seed}: frozen prior for {len(tree.sequences)} candidates, {len(tree.prefixes)} prefixes",
                flush=True,
            )
            cached_features, logits = cache_tree(
                backbone, tree, original["feature_batch_size"]
            )
            task = CachedTask(
                tree, cached_features, logits, torch.zeros(len(tree.sequences))
            )
            log_prior = task.base_log_probabilities.cpu()
            if abs(log_prior.double().exp().sum().item() - 1) > 1e-5:
                raise ValueError("New constrained prior does not normalize")
            torch.save(
                {
                    "sequences": tree.sequences,
                    "log_prior": log_prior,
                    "logits": logits,
                    "context": {
                        "reference": context.reference,
                        "mutation_budget": context.mutation_budget,
                        "alphabet": context.alphabet,
                    },
                },
                directory / "prior.pt",
            )
            del task, cached_features
            settings = {
                "seeds": [config["predictor_seed"]],
                "train_sizes": [len(split_rows["train"])],
                "betas": [0.0, 1.0],
                "rewards": ["linear"],
                "top_k": min(20, len(tree.sequences)),
            }
            compare_predictors(
                split_features,
                split_rows,
                0,
                log_prior,
                settings,
                effective["scorer"],
                directory,
                device,
            )
            write_json(directory / "status.json", {"status": "complete"})
        write_json(
            output / "provenance.json",
            {
                "input_sha256": config["input_sha256"],
                "torch": torch.__version__,
                "checkpoint_sha256": json.loads(
                    (
                        project_path(config["predictor_run"]) / "provenance.json"
                    ).read_text()
                )["checkpoint_sha256"],
                "tokenizer_sha256": file_hash(TOKENIZER_PATH),
                "backbone_work": backbone.work,
                "backbone_frozen": not any(
                    p.requires_grad for p in backbone.model.parameters()
                ),
            },
        )
        del backbone
        summarize_split_results(output, config["split_seeds"])
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
