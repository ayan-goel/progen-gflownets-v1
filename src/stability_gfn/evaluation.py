"""Reloadable evaluation with sealed labels opened only after training finishes."""

import json
import time
from pathlib import Path

import torch

from .artifacts import file_hash, write_json
from .environment import PrefixTree, SequenceContext, hamming
from .guidance import CachedTask
from .metrics import distribution_metrics, sample_metrics


def load_task(run, device="cpu"):
    """Restore frozen numerical inputs; evaluation needs no ProGen weights or GPU."""
    payload = torch.load(Path(run) / "task.pt", map_location=device, weights_only=True)
    tree = PrefixTree(SequenceContext(**payload["context"]), payload["sequences"])
    task = CachedTask(
        tree, payload["features"], payload["logits"], payload["scores"], device
    )
    return task, payload


def load_head(task, path):
    """Reconstruct a saved residual head without retraining."""
    checkpoint = torch.load(path, map_location=task.device, weights_only=True)
    head = task.head(checkpoint["hidden_size"])
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    return head, checkpoint


def evaluate_run(run, output, device="cpu"):
    """Save raw proposals and selection indices for every method and seed."""
    run = Path(run)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    config = json.loads((run / "config.json").read_text())
    task, payload = load_task(run, device)
    labels = {}
    if (run / "evaluation_labels.json").exists():
        expected_hash = payload["evaluation_labels_sha256"]
        if file_hash(run / "evaluation_labels.json") != expected_hash:
            raise ValueError("Evaluation labels changed since experiment setup")
        labels = {
            row["sequence"]: row["stability"]
            for row in json.loads((run / "evaluation_labels.json").read_text())
        }
    output.mkdir(parents=True)
    settings = config["evaluation"]
    predictions = dict(zip(task.tree.sequences, payload["predictions"].tolist()))
    training_sequences = set(payload["training_sequences"])
    metrics = []
    for seed in config["seeds"]:
        seed_path = run / f"seed-{seed}"
        heads = {"progen": None}
        checkpoint_data = {}
        for method in ["gfn", "supervised"]:
            heads[method], checkpoint_data[method] = load_head(
                task, seed_path / f"{method}.pt"
            )
        for method, head in heads.items():
            generator = torch.Generator(device=task.device).manual_seed(
                settings["seed_offset"] + seed
            )
            started = time.perf_counter()
            count = (
                settings["filter_pool_count"]
                if method == "progen"
                else settings["raw_count"]
            )
            leaves = task.sample(head, count, generator).cpu().tolist()
            elapsed = time.perf_counter() - started
            raw = [task.tree.sequences[index] for index in leaves]
            # Keep multiplicities and ties deterministic; no post-hoc deduplication.
            ranking = sorted(
                range(len(leaves)), key=lambda index: (-predictions[raw[index]], index)
            )
            selected = ranking[: settings["selected_count"]]
            batches = [
                ("raw", list(range(min(settings["raw_count"], count)))),
                ("selected", selected),
            ]
            if method == "progen":
                batches.append(("unranked", list(range(settings["selected_count"]))))
            rows = []
            for batch_name, indices in batches:
                sequences = [raw[index] for index in indices]
                row = {
                    "seed": seed,
                    "method": "generate_select"
                    if method == "progen" and batch_name == "selected"
                    else method,
                    "batch": batch_name,
                    **sample_metrics(
                        sequences,
                        task.tree.context.reference,
                        training_sequences,
                        predictions,
                        labels,
                        payload["reference_label"],
                        payload["reference_prediction"],
                    ),
                }
                row["constraint_violations"] = sum(
                    hamming(sequence, task.tree.context.reference)
                    > task.tree.context.mutation_budget
                    for sequence in sequences
                )
                rows.append(row)
                metrics.append(row)
            exact = distribution_metrics(
                task.all_log_probabilities(head),
                task.base_log_probabilities,
                task.scores,
                config["training"]["beta"],
                checkpoint_data.get(method, {}).get("log_z"),
            )
            write_json(
                output / f"seed-{seed}-{method}.json",
                {
                    "raw_sequences": raw,
                    "leaf_ids": leaves,
                    "selected_indices": selected,
                    "selected_sequences": [raw[index] for index in selected],
                    "batches": rows,
                    "distribution_before_selection": exact,
                    "sampling_seconds": elapsed,
                    "sample_count": count,
                    "backbone_calls": 0,
                    "terminal_score_lookups": count,
                },
            )
    summary = {
        "smoke": config.get("smoke", False),
        "metrics": metrics,
        "costs": json.loads((run / "costs.json").read_text()),
        "notes": [
            "Post-selection batches do not follow the raw policy distribution.",
            "GFN/supervised use raw_count proposals; generate-select uses the larger filter_pool_count.",
            "Measurement metrics describe covered candidates only; toy scores are not stability.",
        ],
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# Experiment evaluation",
        "",
        f"Smoke test: **{summary['smoke']}**",
        "",
        "| Seed | Method | Batch | Predicted score | Measured stability | Unique fraction | Pairwise Hamming | Coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        measured = (
            "n/a"
            if row["mean_measured_stability"] is None
            else f"{row['mean_measured_stability']:.4f}"
        )
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['batch']} | {row['mean_predicted_stability']:.4f} | {measured} | {row['unique_fraction']:.3f} | {row['mean_pairwise_hamming']:.3f} | {row['measurement_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "See summary.json for locality, novelty, prediction errors and setup/training costs.",
            "Per-method JSON files retain every raw proposal and selection index. Exact distribution diagnostics apply before selection.",
            "For the exact toy, the predicted-score column is a synthetic log reward, not kcal/mol.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    return summary
