"""Train the original guidance head against the saved full-data ridge reward."""

import copy
import json
import shutil
import time

import torch

from .artifacts import file_hash, new_run, write_json
from .evaluation import evaluate_run, load_head, load_task
from .metrics import distribution_metrics
from .pipeline import ROOT, progress_recorder, project_path, save_head
from .reward_study import target_metrics
from .scorer import StabilityScorer
from .training import train_tb


def prepare_ridge_task(config):
    """Verify immutable inputs and reconstruct predictions on CPU, without fitting."""
    for path, expected in config["input_sha256"].items():
        if file_hash(project_path(path)) != expected:
            raise ValueError(f"Input changed: {path}")
    pilot = project_path(config["pilot_run"])
    study = project_path(config["predictor_run"])
    original = json.loads((pilot / "config.json").read_text())
    task, payload = load_task(pilot)
    cache = torch.load(study / "features.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(
        study / config["scorer_file"], map_location="cpu", weights_only=True
    )
    if checkpoint["hidden_size"] != 0:
        raise ValueError("Expected a linear ridge checkpoint")
    if [r["sequence"] for r in cache["rows"]["test"]] != payload["sequences"]:
        raise ValueError("Predictor and prefix cache candidate order differs")
    if set(r["sequence"] for r in cache["rows"]["train"]) != set(
        payload["training_sequences"]
    ):
        raise ValueError("Predictor training sequences differ from original pilot")
    reference_index = cache["reference_index"]
    if (
        cache["rows"]["train"][reference_index]["sequence"]
        != payload["context"]["reference"]
    ):
        raise ValueError("Reference identity differs")
    torch.testing.assert_close(
        cache["log_prior"], task.base_log_probabilities, atol=1e-6, rtol=1e-6
    )
    model = StabilityScorer(checkpoint["feature_size"], 0)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    scores = json.loads((study / config["predictions_file"]).read_text())
    if scores["sequences"] != payload["sequences"]:
        raise ValueError("Saved prediction ordering differs")
    candidates = [
        r
        for r in json.loads((study / "results.json").read_text())
        if r["family"] == "ridge"
        and r["train_size"] == len(payload["training_sequences"])
        and r["seed"] == config["predictor_seed"]
    ]
    if len(candidates) != 1:
        raise ValueError("Missing unique full-data ridge result")
    selected = candidates[0]
    predictions = torch.tensor(scores["predictions"], dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(cache["features"]["test"])
        reference = model(
            cache["features"]["train"][reference_index : reference_index + 1]
        ).item()
    torch.testing.assert_close(predictions, reconstructed, atol=5e-5, rtol=1e-5)
    if abs(reference - selected["reference_prediction"]) > 5e-5:
        raise ValueError("Saved reference prediction differs from checkpoint")
    scale = model.label_scale.item()
    if scale <= 0 or abs(scale - selected["scale"]) > 1e-6:
        raise ValueError("Reward scale differs from saved study")
    # Use the saved values after verification to preserve the original study target.
    payload["predictions"] = predictions
    payload["reference_prediction"] = selected["reference_prediction"]
    payload["scores"] = (predictions - payload["reference_prediction"]) / scale
    metadata = {
        "family": "ridge",
        "predictor_seed": config["predictor_seed"],
        "training_count": len(payload["training_sequences"]),
        "scale": scale,
        "reference_prediction": payload["reference_prediction"],
        "maximum_prediction_reconstruction_error": float(
            (reconstructed - predictions).abs().max()
        ),
        "beta": original["training"]["beta"],
        "candidate_count": len(predictions),
        "smoke_source": original.get("smoke", False),
    }
    if original["training"]["beta"] != 1:
        raise ValueError("This comparison requires the original beta=1")
    return payload, original, metadata


def evaluate_ridge_guidance(run):
    """Exact predicted/measured expectations after training; never select a checkpoint."""
    config = json.loads((run / "config.json").read_text())
    task, payload = load_task(run)
    if file_hash(run / "evaluation_labels.json") != payload["evaluation_labels_sha256"]:
        raise ValueError("Evaluation labels changed")
    lookup = {
        r["sequence"]: r["stability"]
        for r in json.loads((run / "evaluation_labels.json").read_text())
    }
    labels = torch.tensor(
        [lookup[s] for s in payload["sequences"]], dtype=torch.float64
    )
    predictions = payload["predictions"].double()
    beta = config["training"]["beta"]
    distributions = [
        ("ProGen", None, task.base_log_probabilities, None),
        (
            "Exact ridge target",
            None,
            torch.log_softmax(
                task.base_log_probabilities.double() + beta * task.scores.double(), 0
            ),
            None,
        ),
    ]
    for seed in config["seeds"]:
        head, state = load_head(task, run / f"seed-{seed}/gfn.pt")
        distributions.append(
            ("Ridge GFN", seed, task.all_log_probabilities(head), state["log_z"])
        )
    rows = []
    for method, seed, log_q, log_z in distributions:
        row = {
            "method": method,
            "seed": seed,
            **target_metrics(
                log_q.double().softmax(0),
                predictions,
                labels,
                payload["reference_label"],
            ),
            **distribution_metrics(
                log_q, task.base_log_probabilities, task.scores, beta, log_z
            ),
        }
        rows.append(row)
    baseline = rows[0]["expected_measured_dg"]
    target = rows[1]["expected_measured_dg"]
    for row in rows:
        row["measured_gain_over_progen"] = row["expected_measured_dg"] - baseline
        row["measured_gap_from_ridge_target"] = row["expected_measured_dg"] - target
    write_json(run / "exact_comparison.json", rows)
    lines = [
        "# Ridge-reward GFlowNet",
        "",
        f"Synthetic smoke: {config['smoke']}",
        "",
        "Development on the original candidate set; final checkpoints only.",
        "",
        "All stability values below are probability-weighted expectations in kcal/mol.",
        "",
        "| Distribution | Seed | Expected predicted stability (ridge) | Expected measured stability | TV to ridge target |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        seed = "—" if row["seed"] is None else str(row["seed"])
        lines.append(
            f"| {row['method']} | {seed} | {row['expected_predicted_dg']:.5f} | {row['expected_measured_dg']:.5f} | {row['total_variation']:.6f} |"
        )
    # This is the actual original MLP reward, not the newer three MLP predictor fits.
    pilot = project_path(config["pilot_run"])
    if not config["smoke"]:
        historical_task, historical_payload = load_task(pilot)
        historical = []
        for seed in config["seeds"]:
            head, _ = load_head(historical_task, pilot / f"seed-{seed}/gfn.pt")
            probability = (
                historical_task.all_log_probabilities(head).double().softmax(0)
            )
            historical.append(
                {
                    "seed": seed,
                    **target_metrics(
                        probability,
                        historical_payload["predictions"],
                        labels,
                        payload["reference_label"],
                    ),
                }
            )
        write_json(run / "historical_mlp_gfn.json", historical)
        mean = sum(r["expected_measured_dg"] for r in historical) / len(historical)
        lines.extend(
            [
                "",
                f"Original MLP-reward GFN expected measured stability (seed mean): {mean:.5f}.",
                "This is a historical paired-seed comparator, not a fresh retraining or the newer MLP target study.",
            ]
        )
    lines.extend(
        [
            "",
            "exact_comparison.json also records success probability, diversity, KL and normalization error.",
            "evaluation/report.md contains sampled raw/selected batches. Selection changes the distribution.",
            "Training curves track predicted-target fit only; measured labels are used after fitting.",
            "Completion does not certify convergence or biological improvement. No best-seed selection.",
        ]
    )
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def run_ridge_guidance(config, output, device, smoke=False):
    """Keep the original TB optimizer and cached prior; change only terminal scores."""
    if smoke and device != "cpu":
        raise ValueError("Smoke tests require CPU")
    if not smoke and not device.startswith("cuda"):
        raise ValueError(
            "Real training is user-run on CUDA; use check-only for CPU readiness"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload, original, metadata = prepare_ridge_task(config)
    if smoke and not metadata["smoke_source"]:
        raise ValueError("CPU smoke may only train on a synthetic pilot fixture")
    if not smoke and metadata["smoke_source"]:
        raise ValueError("Synthetic source cannot be used for a real run")
    effective = copy.deepcopy(config)
    effective.update(
        training=copy.deepcopy(original["training"]),
        evaluation=copy.deepcopy(original["evaluation"]),
        seeds=original["seeds"],
        smoke=smoke,
        device=device,
    )
    # Exact monitoring reads only fixed predicted scores, not measured outcomes.
    effective["training"]["monitor_exact_distribution"] = True
    effective["evaluation"]["methods"] = ["gfn"]
    if smoke:
        effective["training"].update(steps=4, batch_size=4, hidden_size=8, log_every=2)
        effective["seeds"] = [0]
        effective["evaluation"].update(
            raw_count=8, selected_count=2, filter_pool_count=8
        )
    output = new_run(output, effective)
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(
            ROOT / "docs/experiments/ridge_gfn_2026-11-09.md", output / "protocol.md"
        )
        shutil.copy2(
            project_path(config["predictor_run"]) / config["scorer_file"],
            output / "scorer.pt",
        )
        write_json(output / "reward.json", metadata)
        write_json(
            output / "provenance.json",
            {
                "input_sha256": config["input_sha256"],
                "torch": torch.__version__,
                "device": device,
                "backbone_loaded": False,
            },
        )
        torch.save(payload, output / "task.pt")
        task, _ = load_task(output, device)
        torch.testing.assert_close(
            task.all_log_probabilities(task.head(effective["training"]["hidden_size"])),
            task.base_log_probabilities,
            atol=0,
            rtol=0,
        )
        costs = {
            "seeds": {},
            "setup": {"backbone_calls": 0, "predictor_fitting": False},
        }
        for seed in effective["seeds"]:
            destination = output / f"seed-{seed}"
            destination.mkdir()
            head, log_z, history, cost = train_tb(
                task, effective["training"], seed, progress_recorder(destination, "gfn")
            )
            save_head(
                destination / "gfn.pt",
                head,
                effective["training"]["hidden_size"],
                log_z,
            )
            write_json(destination / "gfn_history.json", history)
            costs["seeds"][str(seed)] = {"gfn": cost}
            write_json(output / "costs.json", costs)
        del task
        # Labels do not enter the training task, objective, monitoring or stopping.
        shutil.copy2(
            project_path(config["pilot_run"]) / "evaluation_labels.json",
            output / "evaluation_labels.json",
        )
        evaluate_run(output, output / "evaluation", device="cpu")
        evaluate_ridge_guidance(output)
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
