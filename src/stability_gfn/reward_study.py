"""Local predictor comparison and exact reward analysis; no new GFN training."""

import json
import shutil

import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import FrozenProGen
from .evaluation import load_task
from .pipeline import ROOT, extract_features, project_path
from .scorer import fit_scorer, regression_metrics


def reward_distribution(
    log_prior, predictions, reference_prediction, scale, beta, reward="linear"
):
    """Positive exponential reward; capped utility stops improving at reference dG."""
    if scale <= 0 or beta < 0 or reward not in {"linear", "capped"}:
        raise ValueError("Invalid reward settings")
    scores = (torch.as_tensor(predictions).double() - reference_prediction) / scale
    if reward == "capped":
        scores = scores.clamp_max(0)
    return torch.softmax(torch.as_tensor(log_prior).double() + beta * scores, 0)


def target_metrics(probabilities, predictions, labels, threshold):
    """Exact finite-support utility and concentration, without sampled selection."""
    probabilities = torch.as_tensor(probabilities).double()
    predictions = torch.as_tensor(predictions).double()
    labels = torch.as_tensor(labels).double()
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
    return {
        "expected_predicted_dg": float(probabilities @ predictions),
        "expected_measured_dg": float(probabilities @ labels),
        "probability_at_least_reference": float(
            probabilities @ (labels >= threshold).double()
        ),
        "effective_support": float(entropy.exp()),
        "top_five_mass": float(probabilities.topk(min(5, len(labels))).values.sum()),
        "expected_unique_in_256": float(
            (-torch.expm1(256 * torch.log1p(-probabilities))).sum()
        ),
    }


def audit_pilot(run, output):
    """Check the saved reward arithmetic and identify influential ranking errors on CPU."""
    torch.set_num_threads(1)
    config = json.loads((run / "config.json").read_text())
    task, payload = load_task(run)
    checkpoint = torch.load(run / "scorer.pt", map_location="cpu", weights_only=True)
    scale = float(checkpoint["state_dict"]["label_scale"])
    predictions = payload["predictions"].double()
    reference = payload["reference_prediction"]
    recomputed = (predictions - reference) / scale
    torch.testing.assert_close(recomputed, task.scores.double(), atol=1e-6, rtol=1e-5)
    if file_hash(run / "evaluation_labels.json") != payload["evaluation_labels_sha256"]:
        raise ValueError("Saved evaluation labels changed")
    label_map = {
        r["sequence"]: r["stability"]
        for r in json.loads((run / "evaluation_labels.json").read_text())
    }
    labels = torch.tensor(
        [label_map[s] for s in task.tree.sequences], dtype=torch.float64
    )
    prior = torch.softmax(task.base_log_probabilities.double(), 0)
    target = reward_distribution(
        task.base_log_probabilities,
        predictions,
        reference,
        scale,
        config["training"]["beta"],
    )
    # Centering removes the arbitrary absolute dG origin from attribution.
    contributions = (target - prior) * (labels - prior @ labels)
    indices = torch.argsort(contributions)[:10].tolist()
    result = {
        "scope": "post-hoc development audit of previously inspected outcomes",
        "reward_scale": scale,
        "reference_prediction": reference,
        "beta": config["training"]["beta"],
        "maximum_score_reconstruction_error": float(
            (recomputed - task.scores.double()).abs().max()
        ),
        "predictor": regression_metrics(predictions, labels),
        "baseline": target_metrics(
            prior, predictions, labels, payload["reference_label"]
        ),
        "target": target_metrics(
            target, predictions, labels, payload["reference_label"]
        ),
        "largest_negative_contributions": [
            {
                "sequence": task.tree.sequences[i],
                "prediction": float(predictions[i]),
                "measurement": float(labels[i]),
                "prior_probability": float(prior[i]),
                "target_probability": float(target[i]),
                "measured_expectation_contribution": float(contributions[i]),
            }
            for i in indices
        ],
        "input_sha256": {
            name: file_hash(run / name)
            for name in [
                "task.pt",
                "scorer.pt",
                "config.json",
                "evaluation_labels.json",
            ]
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", result)
    return result


def compare_predictors(
    features, rows, reference_index, log_prior, settings, scorer_config, output, device
):
    """Matched nested training subsets; validation selects models, evaluation only reports."""
    train_labels = torch.tensor([r["stability"] for r in rows["train"]])
    validation_labels = [r["stability"] for r in rows["validation"]]
    labels = torch.tensor([r["stability"] for r in rows["test"]], dtype=torch.float64)
    threshold = float(train_labels[reference_index])
    prior = torch.softmax(log_prior.double(), 0)
    results = []
    subset_records = []
    for seed in settings["seeds"]:
        generator = np.random.default_rng(seed)
        pool = np.delete(np.arange(len(train_labels)), reference_index)
        permutation = generator.permutation(pool)
        for size in settings["train_sizes"]:
            if not 2 <= size <= len(train_labels):
                raise ValueError("Training size outside available pool")
            indices = np.concatenate(
                ([reference_index], permutation[: size - 1])
            ).tolist()
            subset_records.append(
                {
                    "seed": seed,
                    "size": size,
                    "sequences": [rows["train"][i]["sequence"] for i in indices],
                }
            )
            scale = float(train_labels[indices].std(unbiased=False).clamp_min(1e-6))
            for family in ["constant", "ridge", "mlp"]:
                print(f"seed={seed}, training={size}, predictor={family}", flush=True)
                name = f"seed-{seed}-n-{size}-{family}"
                if family == "constant":
                    prediction = torch.full(
                        (len(labels),), float(train_labels[indices].mean())
                    )
                    reference_prediction = float(prediction[0])
                    validation_prediction = np.full(
                        len(validation_labels), reference_prediction
                    )
                else:
                    model, fit_report = fit_scorer(
                        features["train"][indices].to(device),
                        train_labels[indices],
                        features["validation"].to(device),
                        validation_labels,
                        scorer_config,
                        seed=seed,
                        family=family,
                    )
                    with torch.no_grad():
                        prediction = model(features["test"].to(device)).cpu()
                        validation_prediction = model(
                            features["validation"].to(device)
                        ).cpu()
                        reference_prediction = float(
                            model(
                                features["train"][
                                    reference_index : reference_index + 1
                                ].to(device)
                            )
                        )
                    torch.save(
                        {
                            "state_dict": {
                                k: v.cpu() for k, v in model.state_dict().items()
                            },
                            "hidden_size": model.hidden_size,
                            "feature_size": features["train"].shape[1],
                        },
                        output / f"{name}.pt",
                    )
                    write_json(output / f"{name}-fit.json", fit_report)
                # Stable ordering gives deterministic ties; the constant control is not an oracle.
                top = np.argsort(-prediction.numpy(), kind="stable")[
                    : settings["top_k"]
                ]
                row = {
                    "seed": seed,
                    "train_size": size,
                    "family": family,
                    "validation": regression_metrics(
                        validation_prediction, validation_labels
                    ),
                    "evaluation": regression_metrics(prediction, labels),
                    "top_k_measured_mean": float(labels[top].mean()),
                    "top_k_fraction_at_least_reference": float(
                        (labels[top] >= threshold).double().mean()
                    ),
                    "scale": scale,
                    "reference_prediction": reference_prediction,
                    "baseline": target_metrics(prior, prediction, labels, threshold),
                    "targets": [],
                }
                for reward in settings["rewards"]:
                    for beta in settings["betas"]:
                        probabilities = reward_distribution(
                            log_prior,
                            prediction,
                            reference_prediction,
                            scale,
                            beta,
                            reward,
                        )
                        row["targets"].append(
                            {
                                "reward": reward,
                                "beta": beta,
                                **target_metrics(
                                    probabilities, prediction, labels, threshold
                                ),
                            }
                        )
                results.append(row)
                write_json(
                    output / f"{name}-predictions.json",
                    {
                        "sequences": [r["sequence"] for r in rows["test"]],
                        "predictions": prediction.tolist(),
                        "measurements": labels.tolist(),
                    },
                )
                write_json(output / "results.json", results)
    write_json(output / "subsets.json", subset_records)
    lines = [
        "# Predictor/reward development study",
        "",
        "Previously inspected within-parent evaluation data; no fresh confirmation or new GFN training.",
        "",
        "| Seed | Train n | Predictor | RMSE | Spearman | Top-k measured dG |",
        "|---|---|---|---|---|---|",
    ]
    for row in results:
        metrics = row["evaluation"]
        correlation = (
            "undefined" if metrics["spearman"] is None else f"{metrics['spearman']:.3f}"
        )
        lines.append(
            f"| {row['seed']} | {row['train_size']} | {row['family']} | {metrics['rmse']:.3f} | {correlation} | {row['top_k_measured_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Exact targets",
            "",
            "Delta measured dG is relative to constrained ProGen on the same 204 candidates.",
            "",
            "| Seed | Train n | Predictor | Reward | Beta | Delta measured dG | P(at least reference) | Effective support |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in results:
        for target in row["targets"]:
            delta = (
                target["expected_measured_dg"] - row["baseline"]["expected_measured_dg"]
            )
            lines.append(
                f"| {row['seed']} | {row['train_size']} | {row['family']} | {target['reward']} | {target['beta']} | {delta:+.4f} | {target['probability_at_least_reference']:.3f} | {target['effective_support']:.2f} |"
            )
    lines.extend(
        [
            "",
            "All numerical metrics are in results.json.",
            "No model or reward is selected using evaluation labels. Compare paired seeds; these are not independent test cohorts.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return results


def run_reward_study(config, output, device):
    """Extract pooled features once, then run the user-initiated small-model study."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run = project_path(config["pilot_run"])
    original = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "data_manifest/manifest.json").read_text())
    rows = {
        split: json.loads((run / "data_manifest" / f"{split}.json").read_text())
        for split in ["train", "validation"]
    }
    task, payload = load_task(run)
    labels_path = run / "evaluation_labels.json"
    if file_hash(labels_path) != payload["evaluation_labels_sha256"]:
        raise ValueError("Evaluation labels changed")
    lookup = {r["sequence"]: r for r in json.loads(labels_path.read_text())}
    rows["test"] = [lookup[s] for s in task.tree.sequences]
    for split in ["train", "validation"]:
        if (
            file_hash(run / "data_manifest" / f"{split}.json")
            != manifest["file_sha256"][f"{split}.json"]
        ):
            raise ValueError("Pilot split changed")
    sequence_sets = [set(r["sequence"] for r in rows[s]) for s in rows]
    if any(sequence_sets[i] & sequence_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Partitions overlap")
    reference_index = [r["sequence"] for r in rows["train"]].index(
        manifest["reference"]
    )
    provenance = json.loads((run / "provenance.json").read_text())
    checkpoint_hashes = {
        name: file_hash(project_path(original["checkpoint"]) / name)
        for name in provenance["checkpoint_sha256"]
    }
    if checkpoint_hashes != provenance["checkpoint_sha256"]:
        raise ValueError("ProGen checkpoint differs from original pilot")
    output = new_run(output, config)
    shutil.copytree(
        ROOT / "src", output / "source", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(
        ROOT / "docs/experiments/predictor_reward_study_2026-11-03.md",
        output / "protocol.md",
    )
    write_json(output / "scorer_config.json", original["scorer"])
    try:
        print(
            "Extracting frozen pooled ProGen features (no ProGen training)", flush=True
        )
        backbone = FrozenProGen.from_checkpoint(
            project_path(original["checkpoint"]), device, original["precision"]
        )
        features = {
            split: extract_features(
                backbone,
                [r["sequence"] for r in records],
                original["feature_batch_size"],
            ).cpu()
            for split, records in rows.items()
        }
        del backbone
        torch.save(
            {
                "features": features,
                "rows": rows,
                "reference_index": reference_index,
                "log_prior": task.base_log_probabilities.cpu(),
            },
            output / "features.pt",
        )
        write_json(
            output / "provenance.json",
            {
                "pilot_run": str(run),
                "device": device,
                "torch": torch.__version__,
                "checkpoint_sha256": checkpoint_hashes,
                "input_sha256": {
                    str(p.relative_to(run)): file_hash(p)
                    for p in [
                        run / "task.pt",
                        run / "config.json",
                        labels_path,
                        run / "data_manifest/train.json",
                        run / "data_manifest/validation.json",
                        run / "data_manifest/manifest.json",
                    ]
                },
                "feature_cache_sha256": file_hash(output / "features.pt"),
            },
        )
        compare_predictors(
            features,
            rows,
            reference_index,
            task.base_log_probabilities.cpu(),
            config,
            original["scorer"],
            output,
            device,
        )
        write_json(
            output / "status.json", {"status": "complete", "scope": "development"}
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    print(output / "report.md", flush=True)
