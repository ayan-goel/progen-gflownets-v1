"""Fixed ridge rewards on position-held-out supports, using the existing TB learner."""

import copy
import json
import shutil
import time

import torch

from .artifacts import file_hash, new_run, write_json
from .backbone import FrozenProGen, cache_tree
from .environment import PrefixTree, SequenceContext
from .evaluation import load_head, load_task
from .metrics import distribution_metrics
from .pipeline import ROOT, progress_recorder, project_path, save_head
from .reward_study import target_metrics
from .scorer import StabilityScorer
from .split_study import prepare_split_inputs
from .training import train_tb


def prepare_position_guidance(config):
    """Verify saved folds, priors and ridge predictions on CPU; never fit a model."""
    source = project_path(config["source_run"])
    pinned = {project_path(path) for path in config["input_sha256"]}
    for path, expected in config["input_sha256"].items():
        if file_hash(project_path(path)) != expected:
            raise ValueError(f"Input changed: {path}")
    required = {
        source / name for name in ["config.json", "status.json", "provenance.json"]
    }
    if not required <= pinned:
        raise ValueError("Missing source-run hashes")
    source_config = json.loads((source / "config.json").read_text())
    if source_config.get("split_unit") != "position":
        raise ValueError("Need a position-held-out source")
    expected_status = "smoke_complete" if source_config["smoke"] else "complete"
    if json.loads((source / "status.json").read_text())["status"] != expected_status:
        raise ValueError("Source study is not complete")
    original, manifest, rows, features, partitions, readiness = prepare_split_inputs(
        source_config
    )
    if original["training"]["beta"] != 1 or source_config["beta"] != 1:
        raise ValueError("This follow-up requires the fixed beta-one target")
    payloads = {}
    details = []
    for split_seed, partition in partitions.items():
        directory = source / f"split-{split_seed}"
        count = len(partition["train"])
        prefix = f"seed-{source_config['predictor_seed']}-n-{count}"
        needed = [
            "partitions.json",
            "mutation_positions.json",
            "prior.pt",
            "results.json",
            f"{prefix}-ridge.pt",
            f"{prefix}-ridge-predictions.json",
            f"{prefix}-mlp-predictions.json",
        ]
        if not {directory / name for name in needed} <= pinned:
            raise ValueError("Missing per-split reward/data hashes")
        saved_rows = json.loads((directory / "partitions.json").read_text())
        if saved_rows != {
            split: [rows[i] for i in indices] for split, indices in partition.items()
        }:
            raise ValueError(
                "Saved partition differs from the predefined position groups"
            )
        if (
            json.loads((directory / "mutation_positions.json").read_text())
            != readiness["mutation_positions"][str(split_seed)]
        ):
            raise ValueError("Saved mutation groups differ")
        prior = torch.load(
            directory / "prior.pt", map_location="cpu", weights_only=True
        )
        if (
            prior["context"]["reference"] != manifest["reference"]
            or prior["context"]["mutation_budget"] != manifest["mutation_budget"]
        ):
            raise ValueError("Prior context differs from source manifest")
        sequences = [row["sequence"] for row in saved_rows["test"]]
        if prior["sequences"] != sequences:
            raise ValueError("Prior sequence order differs")
        tree = PrefixTree(SequenceContext(**prior["context"]), sequences)
        conditionals = (
            prior["logits"]
            .masked_fill(~torch.tensor(tree.legal), -torch.inf)
            .log_softmax(-1)
        )
        log_prior = conditionals[
            torch.tensor(tree.paths), torch.tensor(tree.actions)
        ].sum(-1)
        torch.testing.assert_close(log_prior, prior["log_prior"], atol=1e-6, rtol=1e-6)
        if abs(float(log_prior.double().exp().sum()) - 1) > 1e-5:
            raise ValueError("Saved prior is not normalized")
        results = json.loads((directory / "results.json").read_text())
        values = {}
        selected = {}
        for family in ["ridge", "mlp"]:
            matches = [
                result
                for result in results
                if result["family"] == family
                and result["seed"] == source_config["predictor_seed"]
                and result["train_size"] == count
            ]
            if len(matches) != 1:
                raise ValueError("Need a unique saved full-data predictor result")
            selected[family] = matches[0]
            prediction_file = json.loads(
                (directory / f"{prefix}-{family}-predictions.json").read_text()
            )
            if prediction_file["sequences"] != sequences:
                raise ValueError("Prediction sequence order differs")
            values[family] = torch.tensor(
                prediction_file["predictions"], dtype=torch.float32
            )
            if not torch.isfinite(values[family]).all():
                raise ValueError("Non-finite saved predictions")
        ridge = selected["ridge"]
        checkpoint = torch.load(
            directory / f"{prefix}-ridge.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["hidden_size"] != 0:
            raise ValueError("Expected a linear ridge scorer")
        model = StabilityScorer(checkpoint["feature_size"], 0)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().requires_grad_(False)
        with torch.no_grad():
            reconstructed = model(features[partition["test"]])
            reference = float(model(features[partition["train"][:1]]))
        torch.testing.assert_close(reconstructed, values["ridge"], atol=5e-5, rtol=1e-5)
        scale = ridge["scale"]
        if (
            scale <= 0
            or abs(float(model.label_scale) - scale) > 1e-6
            or abs(reference - ridge["reference_prediction"]) > 5e-5
        ):
            raise ValueError("Ridge reward normalization differs from checkpoint")
        if selected["mlp"]["scale"] != scale:
            raise ValueError("Predictor scales differ within the same split")
        # Keep saved predictions and logits: newly computed prefix features must not change the prior.
        scores = (
            (values["ridge"].double() - ridge["reference_prediction"]) / scale
        ).float()
        source_target = (
            log_prior.double()
            + (values["ridge"].double() - ridge["reference_prediction"]) / scale
        ).softmax(0)
        training_target = (log_prior.double() + scores.double()).softmax(0)
        rounding_tv = float(0.5 * (training_target - source_target).abs().sum())
        if rounding_tv > 1e-6:
            raise ValueError("Float32 training scores changed the saved target")
        payloads[split_seed] = {
            "context": prior["context"],
            "sequences": sequences,
            "logits": prior["logits"],
            "scores": scores,
            "predictions": values["ridge"],
            "reference_prediction": ridge["reference_prediction"],
            "reward_scale": scale,
            "mlp_predictions": values["mlp"],
            "mlp_reference_prediction": selected["mlp"]["reference_prediction"],
        }
        details.append(
            {
                "split_seed": split_seed,
                "candidate_count": len(sequences),
                "training_count": count,
                "maximum_prediction_reconstruction_error": float(
                    (reconstructed - values["ridge"]).abs().max()
                ),
                "score_rounding_target_tv": rounding_tv,
                "ridge_checkpoint": str(directory / f"{prefix}-ridge.pt"),
            }
        )
    readiness.update(
        splits=details,
        training_seeds=original["seeds"],
        new_gfn_fits=len(partitions) * len(original["seeds"]),
    )
    return original, payloads, readiness


@torch.no_grad()
def evaluate_position_guidance(run, output):
    """CPU exact evaluation from saved tasks and heads, without source-run dependencies."""
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    config = json.loads((run / "config.json").read_text())
    hashes = json.loads((run / "evaluation_inputs.json").read_text())
    for path, expected in hashes.items():
        if file_hash(run / path) != expected:
            raise ValueError(f"Evaluation input changed: {path}")
    rows = []
    for split_seed in config["split_seeds"]:
        directory = run / f"split-{split_seed}"
        task, payload = load_task(directory)
        measured = json.loads((directory / "evaluation_labels.json").read_text())
        if [row["sequence"] for row in measured["rows"]] != payload["sequences"]:
            raise ValueError("Evaluation label order differs")
        labels = torch.tensor(
            [row["stability"] for row in measured["rows"]], dtype=torch.float64
        )
        log_target = (
            task.base_log_probabilities.double() + task.scores.double()
        ).log_softmax(0)
        mlp_scores = (
            payload["mlp_predictions"].double() - payload["mlp_reference_prediction"]
        ) / payload["reward_scale"]
        distributions = [
            ("ProGen", None, task.base_log_probabilities, None),
            ("Exact ridge target", None, log_target, None),
            (
                "Exact MLP target",
                None,
                (task.base_log_probabilities.double() + mlp_scores).log_softmax(0),
                None,
            ),
        ]
        for seed in config["seeds"]:
            head, checkpoint = load_head(task, directory / f"seed-{seed}/gfn.pt")
            distributions.append(
                (
                    "Ridge GFN",
                    seed,
                    task.all_log_probabilities(head),
                    checkpoint["log_z"],
                )
            )
        baseline = float(task.base_log_probabilities.double().softmax(0) @ labels)
        target_mean = float(log_target.exp() @ labels)
        for method, seed, log_probability, log_z in distributions:
            probabilities = log_probability.double().softmax(0)
            metrics = target_metrics(
                probabilities,
                payload["predictions"],
                labels,
                measured["reference_label"],
            )
            rows.append(
                {
                    "split_seed": split_seed,
                    "method": method,
                    "seed": seed,
                    **metrics,
                    **distribution_metrics(
                        log_probability,
                        task.base_log_probabilities,
                        task.scores,
                        1,
                        log_z,
                    ),
                    "top_one_mass": float(probabilities.max()),
                    "expected_mlp_predicted_dg": float(
                        probabilities @ payload["mlp_predictions"].double()
                    ),
                    "measured_gain_over_progen": metrics["expected_measured_dg"]
                    - baseline,
                    "measured_gap_from_ridge_target": metrics["expected_measured_dg"]
                    - target_mean,
                }
            )
    output.mkdir(parents=True)
    write_json(output / "results.json", rows)
    summaries = []
    for split_seed in config["split_seeds"]:
        fitted = [
            row
            for row in rows
            if row["split_seed"] == split_seed and row["method"] == "Ridge GFN"
        ]
        summaries.append(
            {
                "split_seed": split_seed,
                "training_seed_count": len(fitted),
                "mean_measured_dg": sum(row["expected_measured_dg"] for row in fitted)
                / len(fitted),
                "mean_gain_over_progen": sum(
                    row["measured_gain_over_progen"] for row in fitted
                )
                / len(fitted),
                "maximum_final_tv": max(row["total_variation"] for row in fitted),
                "maximum_absolute_measured_target_gap": max(
                    abs(row["measured_gap_from_ridge_target"]) for row in fitted
                ),
            }
        )
    write_json(output / "summary.json", summaries)
    lines = [
        "# GFlowNet on position-held-out ridge rewards",
        "",
        "Development evidence on known single mutants; final checkpoints, all splits and seeds.",
        "Expected measured stability and gains are in kcal/mol. TV is always to that split's ridge target.",
        "",
        "| Split | Distribution | Seed | Measured stability | Gain over ProGen | Gap from ridge target | TV | Success % | Top-one % | Unique in 256 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split_seed']} | {row['method']} | {row['seed'] if row['seed'] is not None else '—'} | {row['expected_measured_dg']:.5f} | {row['measured_gain_over_progen']:+.5f} | {row['measured_gap_from_ridge_target']:+.5f} | {row['total_variation']:.6f} | {100 * row['probability_at_least_reference']:.3f} | {100 * row['top_one_mass']:.3f} | {row['expected_unique_in_256']:.3f} |"
        )
    lines += [
        "",
        "results.json separates ridge-predicted, MLP-predicted and measured expectations, both KL directions and normalization error.",
        "Exact MLP target is a distribution comparator, not a trained MLP-reward GFN. Its TV to ridge does not measure MLP fitting failure.",
        "No sampled best-of selection. These are exact policy expectations, not new stability assays or combination generalization.",
        "Completion does not certify target matching; inspect each final TV, target gap, success and diversity.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def run_position_guidance(config, output, device, smoke=False):
    """Cache frozen prefixes once, then train the unchanged TB head for every split/seed."""
    if (smoke and device != "cpu") or (not smoke and not device.startswith("cuda")):
        raise ValueError(
            "Real runs require user CUDA execution; only synthetic smoke may train on CPU"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original, payloads, readiness = prepare_position_guidance(config)
    if smoke != readiness["smoke_source"]:
        raise ValueError(
            "Smoke mode requires synthetic source data; real runs require measured data"
        )
    effective = copy.deepcopy(config)
    effective.update(
        training=copy.deepcopy(original["training"]),
        seeds=original["seeds"],
        split_seeds=list(payloads),
        smoke=smoke,
        device=device,
    )
    effective["training"]["monitor_exact_distribution"] = True
    if smoke:
        effective["training"].update(steps=4, batch_size=4, hidden_size=8, log_every=2)
        effective["seeds"] = [0, 1]
    readiness["training_seeds"] = effective["seeds"]
    readiness["new_gfn_fits"] = len(payloads) * len(effective["seeds"])
    output = new_run(output, effective)
    source = project_path(config["source_run"])
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(
            ROOT / "docs/experiments/position_gfn_2026-11-09.md", output / "protocol.md"
        )
        write_json(output / "readiness.json", readiness)
        costs = {"setup": {}, "fits": {}}
        print(
            "Caching frozen prefix representations for the three saved supports",
            flush=True,
        )
        backbone = (
            FrozenProGen.tiny()
            if smoke
            else FrozenProGen.from_checkpoint(
                project_path(original["checkpoint"]), device, original["precision"]
            )
        )
        for detail in readiness["splits"]:
            split_seed = detail["split_seed"]
            destination = output / f"split-{split_seed}"
            destination.mkdir()
            payload = payloads[split_seed]
            tree = PrefixTree(
                SequenceContext(**payload["context"]), payload["sequences"]
            )
            features, new_logits = cache_tree(
                backbone, tree, original["feature_batch_size"]
            )
            torch.testing.assert_close(
                new_logits, payload["logits"], atol=5e-5, rtol=1e-5
            )
            payload["features"] = features
            torch.save(payload, destination / "task.pt")
            shutil.copy2(
                project_path(detail["ridge_checkpoint"]), destination / "scorer.pt"
            )
            shutil.copy2(
                source / f"split-{split_seed}/mutation_positions.json",
                destination / "mutation_positions.json",
            )
        if any(parameter.requires_grad for parameter in backbone.model.parameters()):
            raise ValueError("Backbone must remain frozen")
        costs["setup"] = {
            "backbone_work": backbone.work,
            "seconds": time.perf_counter() - started,
            "predictor_fitting": False,
        }
        del backbone, payloads, features, new_logits, payload
        write_json(output / "costs.json", costs)
        for split_seed in effective["split_seeds"]:
            directory = output / f"split-{split_seed}"
            task, _ = load_task(directory, device)
            torch.testing.assert_close(
                task.all_log_probabilities(
                    task.head(effective["training"]["hidden_size"])
                ),
                task.base_log_probabilities,
                atol=0,
                rtol=0,
            )
            for seed in effective["seeds"]:
                print(f"Position split {split_seed}, training seed {seed}", flush=True)
                destination = directory / f"seed-{seed}"
                destination.mkdir()
                head, log_z, history, cost = train_tb(
                    task,
                    effective["training"],
                    seed,
                    progress_recorder(destination, "gfn"),
                )
                save_head(
                    destination / "gfn.pt",
                    head,
                    effective["training"]["hidden_size"],
                    log_z,
                )
                write_json(destination / "gfn_history.json", history)
                costs["fits"][f"split-{split_seed}/seed-{seed}"] = cost
                write_json(output / "costs.json", costs)
                del head
            del task
        # Held-out measurements are not placed in task.pt or passed to train_tb.
        for split_seed in effective["split_seeds"]:
            path = source / f"split-{split_seed}/partitions.json"
            expected = next(
                value
                for name, value in config["input_sha256"].items()
                if project_path(name) == path
            )
            if file_hash(path) != expected:
                raise ValueError("Source labels changed during training")
            partitions = json.loads(path.read_text())
            write_json(
                output / f"split-{split_seed}/evaluation_labels.json",
                {
                    "rows": partitions["test"],
                    "reference_label": float(
                        torch.tensor(partitions["train"][0]["stability"])
                    ),
                },
            )
        inputs = [output / "config.json"]
        for split_seed in effective["split_seeds"]:
            directory = output / f"split-{split_seed}"
            inputs += [directory / "task.pt", directory / "evaluation_labels.json"]
            inputs += [directory / f"seed-{seed}/gfn.pt" for seed in effective["seeds"]]
        write_json(
            output / "evaluation_inputs.json",
            {str(path.relative_to(output)): file_hash(path) for path in inputs},
        )
        write_json(
            output / "provenance.json",
            {
                "input_sha256": config["input_sha256"],
                "torch": torch.__version__,
                "backbone_frozen": True,
                "saved_prior_logits_reused": True,
                "predictor_refitted": False,
            },
        )
        evaluate_position_guidance(output, output / "evaluation")
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
