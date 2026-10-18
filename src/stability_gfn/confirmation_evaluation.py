"""Post-fit confirmation evaluation and CPU replay, including the exact target."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import file_hash, write_json
from .evaluation import load_head, load_task
from .metrics import distribution_metrics
from .pipeline import project_path
from .scorer import regression_metrics
from .target_diagnostics import probability_metrics, support_statistics


def verify_fit_seal(run):
    """Require every declared parent's final artifacts before reading test labels."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "panel_manifest.json").read_text())
    entries = [
        entry for entry in manifest["parents"] if entry["role"] == "confirmation"
    ]
    if not (run / "fits_complete.json").exists():
        raise ValueError(
            "All confirmation fits must finish before outcomes are revealed"
        )
    seal = json.loads((run / "fits_complete.json").read_text())
    if seal["parents"] != [entry["id"] for entry in entries]:
        raise ValueError("Incomplete confirmation parent set")
    required = {"config.json", "panel_manifest.json"}
    for entry in entries:
        folder = run / entry["id"]
        status = json.loads((folder / "fit_status.json").read_text())
        if status["status"] not in ("complete", "scorer_gate_failed"):
            raise ValueError("Unfinished confirmation fit")
        names = [
            "task.pt",
            "scorer.pt",
            "scorer_validation.json",
            "fit_status.json",
            "costs.json",
        ]
        if status["status"] == "complete":
            names += [
                f"seed-{seed}/{method}.pt"
                for seed in config["seeds"]
                for method in ("gfn", "supervised")
            ]
        required.update(f"{entry['id']}/{name}" for name in names)
    if required != set(seal["file_sha256"]):
        raise ValueError("Fit seal does not contain every required artifact")
    for name, digest in seal["file_sha256"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Changed fit artifact: {name}")
    return config, entries


def reveal_labels(run, entry, sequences, config):
    """Called only after all-parent seal verification; preserve original label bytes."""
    destination = Path(run) / entry["id"] / "evaluation_labels.json"
    expected = entry["file_sha256"]["test_labels.json"]
    if not destination.exists():
        source = project_path(config["panel"]) / entry["id"] / "test_labels.json"
        if file_hash(source) != expected:
            raise ValueError("Reserved labels changed since panel preparation")
        temporary = destination.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    if file_hash(destination) != expected:
        raise ValueError("Saved evaluation labels changed")
    rows = json.loads(destination.read_text())
    labels = {row["sequence"]: row["stability"] for row in rows}
    if len(labels) != len(rows) or set(labels) != set(sequences):
        raise ValueError(
            "Revealed labels do not exactly match reserved candidate identities"
        )
    values = torch.tensor(
        [labels[sequence] for sequence in sequences], dtype=torch.float64
    )
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite confirmation measurements")
    return values


def batch_metrics(indices, predictions, labels, reference_label, distances):
    """Measured/predicted quality and duplicates with multiplicities retained."""
    count = len(indices)
    selected_distances = distances[indices][:, indices]
    return {
        "count": count,
        "unique_count": len(indices.unique()),
        "duplicate_fraction": 1 - len(indices.unique()) / count,
        "mean_measured_stability": labels[indices].mean().item(),
        "mean_predicted_stability": predictions[indices].mean().item(),
        "fraction_measured_above_reference": (labels[indices] > reference_label)
        .double()
        .mean()
        .item(),
        "predictor_mae": (predictions[indices] - labels[indices]).abs().mean().item(),
        "mean_pairwise_hamming": selected_distances.sum().item() / (count * (count - 1))
        if count > 1
        else 0.0,
        "measurement_coverage": 1.0,
    }


@torch.no_grad()
def evaluate_confirmation_parent(run, entry, config, output):
    """Evaluate all raw policies and score-selected batches with equal budgets."""
    folder = Path(run) / entry["id"]
    task, payload = load_task(folder, device="cpu")
    labels = reveal_labels(run, entry, task.tree.sequences, config)
    predictions = payload["predictions"].double()
    positions, distances = support_statistics(
        task.tree.sequences, task.tree.context.reference
    )
    baseline = task.base_log_probabilities.double().softmax(0)
    baseline_mean = (baseline @ labels).item()
    log_target = (
        task.base_log_probabilities.double()
        + config["training"]["beta"] * task.scores.double()
    ).log_softmax(0)
    fitting = json.loads((folder / "fit_status.json").read_text())
    screen = json.loads((folder / "scorer_validation.json").read_text())
    methods = ["progen", "exact_target"]
    if fitting["guidance_trained"]:
        methods += ["gfn", "supervised"]
    output.mkdir()
    exact_rows, sampled_rows, timings = [], [], []
    settings = config["evaluation"]
    for seed in config["seeds"]:
        for method in methods:
            started = time.perf_counter()
            head, checkpoint = None, {}
            if method in ("gfn", "supervised"):
                head, checkpoint = load_head(task, folder / f"seed-{seed}/{method}.pt")
            if method == "exact_target":
                log_probabilities = log_target
            elif method == "progen":
                log_probabilities = task.base_log_probabilities.double()
            else:
                log_probabilities = task.all_log_probabilities(head).double()
            mass = log_probabilities.exp().sum().item()
            if not abs(mass - 1) < 1e-4:
                raise ValueError("Policy probabilities fail normalization")
            probabilities = log_probabilities.softmax(0)
            distribution = distribution_metrics(
                log_probabilities,
                task.base_log_probabilities,
                task.scores,
                config["training"]["beta"],
                checkpoint.get("log_z"),
            )
            exact = {
                "seed": seed,
                "method": method,
                **probability_metrics(
                    probabilities,
                    labels,
                    payload["reference_label"],
                    positions,
                    distances,
                    settings["raw_count"],
                ),
                "predicted_mean": (probabilities @ predictions).item(),
                "measured_gain_vs_progen": (probabilities @ labels).item()
                - baseline_mean,
                "distribution": distribution,
            }
            exact_rows.append(exact)
            distribution_seconds = time.perf_counter() - started
            generator = torch.Generator().manual_seed(settings["seed_offset"] + seed)
            started = time.perf_counter()
            if method == "exact_target":
                leaves = torch.multinomial(
                    probabilities,
                    settings["raw_count"],
                    replacement=True,
                    generator=generator,
                )
            else:
                leaves = task.sample(head, settings["raw_count"], generator)
            sample_seconds = time.perf_counter() - started
            started = time.perf_counter()
            ranking = torch.argsort(predictions[leaves], descending=True, stable=True)
            selected = leaves[ranking[: settings["selected_count"]]]
            rows = [
                {
                    "seed": seed,
                    "method": "generate_select"
                    if method == "progen" and batch == "selected"
                    else method,
                    "batch": batch,
                    **batch_metrics(
                        indices,
                        predictions,
                        labels,
                        payload["reference_label"],
                        distances,
                    ),
                }
                for batch, indices in (("raw", leaves), ("selected", selected))
            ]
            selection_seconds = time.perf_counter() - started
            sampled_rows.extend(rows)
            write_json(
                output / f"seed-{seed}-{method}.json",
                {
                    "raw_leaf_ids": leaves.tolist(),
                    "selected_leaf_ids": selected.tolist(),
                    "raw_sequences": [
                        task.tree.sequences[index] for index in leaves.tolist()
                    ],
                    "selected_sequences": [
                        task.tree.sequences[index] for index in selected.tolist()
                    ],
                    "exact": exact,
                    "batches": rows,
                    "probabilities": probabilities.tolist(),
                },
            )
            timings.append(
                {
                    "seed": seed,
                    "method": method,
                    "distribution_seconds": distribution_seconds,
                    "sampling_seconds": sample_seconds,
                    "selection_and_metrics_seconds": selection_seconds,
                    "terminal_score_lookups_for_selection": settings["raw_count"],
                    "backbone_calls": 0,
                }
            )
    result = {
        "id": entry["id"],
        "parent": entry["parent"],
        "scorer_gate_passed": screen["gate_passed"],
        "fit_status": fitting["status"],
        "test_candidate_count": len(labels),
        "scorer_test_metrics": regression_metrics(predictions.numpy(), labels.numpy()),
        "exact": exact_rows,
        "sampled": sampled_rows,
        "training_costs": json.loads((folder / "costs.json").read_text()),
        "failed_screen_note": None
        if screen["gate_passed"]
        else "Guidance not trained; prior/target/scorer metrics are diagnostic only, not a validated design result",
    }
    write_json(output / "result.json", result)
    return result, timings


def evaluate_confirmation(run, output):
    """CPU-only reveal/evaluation after complete fitting; also supports saved replay."""
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    config, entries = verify_fit_seal(run)
    torch.set_num_threads(config["cpu_threads"])
    output.mkdir(parents=True)
    started = time.perf_counter()
    results, timing = [], {}
    for entry in entries:
        print(f"Evaluating sealed confirmation parent {entry['parent']}", flush=True)
        result, elapsed = evaluate_confirmation_parent(
            run, entry, config, output / entry["id"]
        )
        results.append(result)
        timing[entry["id"]] = elapsed
    aggregates = []
    for method in ("progen", "exact_target", "gfn", "supervised"):
        per_parent = []
        for result in results:
            rows = [row for row in result["exact"] if row["method"] == method]
            if rows:
                per_parent.append(
                    {
                        "parent": result["parent"],
                        "gain": float(
                            np.mean([row["measured_gain_vs_progen"] for row in rows])
                        ),
                        "mean_tv": float(
                            np.mean(
                                [row["distribution"]["total_variation"] for row in rows]
                            )
                        ),
                    }
                )
        gains = [row["gain"] for row in per_parent]
        aggregates.append(
            {
                "method": method,
                "reported_parents": len(gains),
                "declared_parents": len(entries),
                "mean_gain": float(np.mean(gains)) if gains else None,
                "min_gain": min(gains) if gains else None,
                "max_gain": max(gains) if gains else None,
                "positive_parents": sum(gain > 0 for gain in gains),
                "per_parent": per_parent,
            }
        )
    summary = {
        "smoke": config["smoke"],
        "declared_parent_count": len(entries),
        "scorer_passed_parent_count": sum(
            result["scorer_gate_passed"] for result in results
        ),
        "aggregates": aggregates,
        "parents": [
            {
                "parent": result["parent"],
                "id": result["id"],
                "fit_status": result["fit_status"],
            }
            for result in results
        ],
        "notes": [
            "All parent fits were sealed before any test outcomes were revealed.",
            "Exact target is a privileged categorical reference, not a trained competitor.",
            "Raw policy and selected-batch outcomes are separate; duplicate proposals are retained.",
            "Guidance aggregates exclude explicitly reported failed scorer screens; failures are not imputed or silently removed.",
            "Mean gains weight parents equally after averaging training seeds. Seeds are not independent proteins.",
            "Complete finite-support feature/scoring work is setup cost, not free online generation.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "evaluation_costs.json",
        {"seconds": time.perf_counter() - started, "methods": timing},
    )
    lines = [
        "# Five-parent stability confirmation",
        "",
        f"Smoke: {config['smoke']}. Scorer screens passed: {summary['scorer_passed_parent_count']}/{len(entries)}.",
        "",
        "Exact raw-policy expectations (kcal/mol); averages across final training seeds.",
        "",
        "| Parent | Scorer screen | Method | Measured mean | Gain vs ProGen | TV to target | Expected unique / raw budget |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        for method in ("progen", "exact_target", "gfn", "supervised"):
            rows = [row for row in result["exact"] if row["method"] == method]
            if not rows:
                lines.append(
                    f"| {result['parent']} | FAILED | {method} | skipped | — | — | — |"
                )
                continue
            lines.append(
                f"| {result['parent']} | {'pass' if result['scorer_gate_passed'] else 'FAILED'} | {method} | {np.mean([row['measured_mean'] for row in rows]):.5f} | {np.mean([row['measured_gain_vs_progen'] for row in rows]):+.5f} | {np.mean([row['distribution']['total_variation'] for row in rows]):.6f} | {np.mean([row['expected_unique_raw'] for row in rows]):.2f} / {config['evaluation']['raw_count']} |"
            )
    lines += [
        "",
        f"Score-selected batches: {config['evaluation']['selected_count']} slots from {config['evaluation']['raw_count']} proposals for every method. Duplicates retained.",
        "",
        "| Parent | Method | Selected measured mean | Selected unique count |",
        "|---|---|---:|---:|",
    ]
    for result in results:
        for method in ("generate_select", "exact_target", "gfn", "supervised"):
            rows = [
                row
                for row in result["sampled"]
                if row["method"] == method and row["batch"] == "selected"
            ]
            if rows:
                lines.append(
                    f"| {result['parent']} | {method} | {np.mean([row['mean_measured_stability'] for row in rows]):.5f} | {np.mean([row['unique_count'] for row in rows]):.2f} |"
                )
    lines += [
        "",
        "No automatic biological success claim: inspect per-parent effects, failed screens, duplicates and compute.",
        "Parent result.json files retain test scorer errors, all seeds, all budgets and training/setup costs. evaluation_costs.json records CPU evaluation cost.",
        "At this finite scale the exact target can already be sampled directly; GFN efficiency requires later non-enumerable tests.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return summary
