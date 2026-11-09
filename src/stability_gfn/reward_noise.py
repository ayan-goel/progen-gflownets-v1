"""Fixed per-sequence score corruption, separating target damage from GFN fit."""

import copy
import json
import math
import shutil
import time

import torch

from .artifacts import file_hash, new_run, write_json
from .evaluation import load_head, load_task
from .guidance import CachedTask
from .metrics import distribution_metrics
from .pipeline import ROOT, progress_recorder, project_path, save_head
from .reward_study import target_metrics
from .training import train_tb


def fixed_score_noise(scores, sigma, noise_seed):
    """Local CPU RNG: one standard-normal draw per identity, scaled across levels.

    Do not center, normalize, clip or redraw the realized noise. Reusing the local
    seed pairs the same perturbation direction across sigma values, independently
    of training RNG. Scores are dimensionless; sigma is not an assay-error estimate.
    """
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Noise sigma must be finite and nonnegative")
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    standard = torch.randn(len(scores), generator=generator, dtype=torch.float32)
    clean = torch.as_tensor(scores).detach().cpu().float()
    return clean + sigma * standard, standard


def check_noise_inputs(config):
    """Check pinned clean-run artifacts without generating noise or accessing a GPU."""
    levels = config["noise_sigmas"]
    seeds = config["noise_seeds"]
    if (
        not levels
        or len(set(levels)) != len(levels)
        or any(
            isinstance(s, bool)
            or not isinstance(s, (int, float))
            or not math.isfinite(s)
            or s <= 0
            for s in levels
        )
    ):
        raise ValueError("Need distinct positive finite noise levels")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Need distinct nonnegative integer noise seeds")
    clean_run = project_path(config["clean_run"])
    for relative, expected in config["input_sha256"].items():
        if file_hash(clean_run / relative) != expected:
            raise ValueError(f"Clean input changed: {relative}")
    original = json.loads((clean_run / "config.json").read_text())
    required = {
        "task.pt",
        "config.json",
        "reward.json",
        "status.json",
        "evaluation_labels.json",
    }
    required.update(f"seed-{seed}/gfn.pt" for seed in original["seeds"])
    if not required <= set(config["input_sha256"]):
        raise ValueError("Missing required input hashes")
    status = json.loads((clean_run / "status.json").read_text())
    if status["status"] not in {"complete", "smoke_complete"}:
        raise ValueError("Clean baseline is not complete")
    if original["training"]["beta"] != 1:
        raise ValueError("Noise study requires beta=1")
    reward = json.loads((clean_run / "reward.json").read_text())
    if (
        reward["family"] != "ridge"
        or not math.isfinite(reward["scale"])
        or reward["scale"] <= 0
    ):
        raise ValueError("Need the clean ridge baseline and positive reward scale")
    return original, {
        "candidate_count": reward["candidate_count"],
        "noise_sigmas": levels,
        "equivalent_dg_noise_std": [s * reward["scale"] for s in levels],
        "noise_seeds": seeds,
        "training_seeds": original["seeds"],
        "new_fits": len(levels) * len(seeds) * len(original["seeds"]),
        "steps_per_fit": original["training"]["steps"],
        "smoke_source": original.get("smoke", False),
    }


def noise_condition_path(root, noise_seed, sigma):
    """Stable, readable directory names for crossed experimental factors."""
    return root / f"noise-{noise_seed}" / f"sigma-{sigma:g}"


def evaluate_noise_study(run, output):
    """Evaluate saved policies on CPU; no new sampling, fitting or ProGen loading."""
    torch.set_num_threads(1)
    config = json.loads((run / "config.json").read_text())
    clean_task, payload = load_task(run)
    if file_hash(run / "evaluation_labels.json") != payload["evaluation_labels_sha256"]:
        raise ValueError("Evaluation labels changed")
    labels_by_sequence = {
        r["sequence"]: r["stability"]
        for r in json.loads((run / "evaluation_labels.json").read_text())
    }
    labels = torch.tensor(
        [labels_by_sequence[s] for s in payload["sequences"]], dtype=torch.float64
    )
    clean_log_target = torch.log_softmax(
        clean_task.base_log_probabilities.double() + clean_task.scores.double(), 0
    )

    def describe(log_q):
        values = target_metrics(
            log_q.double().softmax(0),
            payload["predictions"],
            labels,
            payload["reference_label"],
        )
        # Predictions remain the clean ridge estimates, never noisy scores relabeled as dG.
        values["expected_clean_ridge_predicted_dg"] = values.pop(
            "expected_predicted_dg"
        )
        return values

    baseline = {
        "progen": describe(clean_task.base_log_probabilities),
        "clean_target": describe(clean_log_target),
        "clean_gfn": [],
    }
    for seed in config["training_seeds"]:
        head, state = load_head(clean_task, run / f"clean_heads/seed-{seed}.pt")
        log_q = clean_task.all_log_probabilities(head)
        baseline["clean_gfn"].append(
            {
                "training_seed": seed,
                **describe(log_q),
                "fit_to_clean": distribution_metrics(
                    log_q,
                    clean_task.base_log_probabilities,
                    clean_task.scores,
                    1,
                    state["log_z"],
                ),
            }
        )
    rows = []
    for noise_seed in config["noise_seeds"]:
        for sigma in config["noise_sigmas"]:
            directory = noise_condition_path(run, noise_seed, sigma)
            saved = torch.load(
                directory / "noise.pt", map_location="cpu", weights_only=True
            )
            scores, standard = fixed_score_noise(clean_task.scores, sigma, noise_seed)
            torch.testing.assert_close(saved["scores"], scores, atol=0, rtol=0)
            torch.testing.assert_close(
                saved["standard_noise"], standard, atol=0, rtol=0
            )
            noisy_log_target = torch.log_softmax(
                clean_task.base_log_probabilities.double() + scores.double(), 0
            )
            target = describe(noisy_log_target)
            shift = (
                target["expected_measured_dg"]
                - baseline["clean_target"]["expected_measured_dg"]
            )
            target_tv = float(
                0.5 * (noisy_log_target.exp() - clean_log_target.exp()).abs().sum()
            )
            for seed in config["training_seeds"]:
                head, state = load_head(clean_task, directory / f"seed-{seed}/gfn.pt")
                log_q = clean_task.all_log_probabilities(head)
                policy = describe(log_q)
                rows.append(
                    {
                        "noise_sigma": sigma,
                        "noise_seed": noise_seed,
                        "training_seed": seed,
                        "noisy_target": target,
                        "policy": policy,
                        "target_tv_from_clean": target_tv,
                        "fit_to_noisy": distribution_metrics(
                            log_q,
                            clean_task.base_log_probabilities,
                            scores,
                            1,
                            state["log_z"],
                        ),
                        "fit_to_clean": distribution_metrics(
                            log_q,
                            clean_task.base_log_probabilities,
                            clean_task.scores,
                            1,
                        ),
                        "target_measured_shift_from_clean": shift,
                        "policy_measured_gap_from_noisy_target": policy[
                            "expected_measured_dg"
                        ]
                        - target["expected_measured_dg"],
                        "policy_measured_gain_over_progen": policy[
                            "expected_measured_dg"
                        ]
                        - baseline["progen"]["expected_measured_dg"],
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", {"baseline": baseline, "conditions": rows})
    lines = [
        "# Fixed reward-noise sensitivity",
        "",
        f"Synthetic smoke: {config['smoke']}. All stability values are expected measured dG in kcal/mol.",
        "",
        f"ProGen: {baseline['progen']['expected_measured_dg']:.5f}; clean ridge target: {baseline['clean_target']['expected_measured_dg']:.5f}.",
        "Clean GFN policies are reused from the completed baseline, not retrained.",
        "",
        "Target shift measures reward corruption. GFN−target measures imperfect learning of the corrupted target.",
        "",
        "| Sigma | Noise seed | Training seed | Noisy target stability | GFN stability | Target shift | GFN−target | TV to noisy target |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['noise_sigma']} | {r['noise_seed']} | {r['training_seed']} | {r['noisy_target']['expected_measured_dg']:.5f} | {r['policy']['expected_measured_dg']:.5f} | {r['target_measured_shift_from_clean']:+.5f} | {r['policy_measured_gap_from_noisy_target']:+.5f} | {r['fit_to_noisy']['total_variation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Summary by noise level",
            "",
            "| Sigma | Mean target shift | Range across noise realizations | Mean GFN−target | Maximum final TV to noisy target |",
            "|---|---:|---|---:|---:|",
        ]
    )
    aggregates = []
    for sigma in config["noise_sigmas"]:
        group = [r for r in rows if r["noise_sigma"] == sigma]
        # Each target is counted once, not once per training replicate.
        shifts = [
            r["target_measured_shift_from_clean"]
            for r in group
            if r["training_seed"] == config["training_seeds"][0]
        ]
        aggregate = {
            "sigma": sigma,
            "mean_target_shift": sum(shifts) / len(shifts),
            "target_shift_range": [min(shifts), max(shifts)],
            "mean_policy_gap": sum(
                r["policy_measured_gap_from_noisy_target"] for r in group
            )
            / len(group),
            "maximum_fit_tv": max(r["fit_to_noisy"]["total_variation"] for r in group),
        }
        aggregates.append(aggregate)
        lines.append(
            f"| {sigma} | {aggregate['mean_target_shift']:+.5f} | {min(shifts):+.5f} to {max(shifts):+.5f} | {aggregate['mean_policy_gap']:+.5f} | {aggregate['maximum_fit_tv']:.6f} |"
        )
    write_json(output / "aggregates.json", aggregates)
    lines.extend(
        [
            "",
            "results.json includes clean-predictor expectations, reference-level success, diversity, KL and all clean controls.",
            "Three training seeds share each noisy target; they are not independent noise realizations.",
            "Fixed Gaussian score corruption is a sensitivity model, not an assay-noise model or fresh confirmation.",
            "Use measured utility and target-fit curves separately. No best-seed or best-noise selection.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return {"baseline": baseline, "conditions": rows}


def run_noise_study(config, output, device, smoke=False):
    """Train the fully crossed fixed-noise panel using the original TB schedule."""
    if smoke and device != "cpu":
        raise ValueError("Smoke requires CPU")
    if not smoke and not device.startswith("cuda"):
        raise ValueError("Real training is user-run on CUDA; use check-only on CPU")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original, readiness = check_noise_inputs(config)
    if bool(readiness["smoke_source"]) != smoke:
        raise ValueError(
            "Smoke may only use synthetic source runs; real runs require real sources"
        )
    clean_run = project_path(config["clean_run"])
    effective = copy.deepcopy(config)
    effective.update(
        training=copy.deepcopy(original["training"]),
        training_seeds=original["seeds"],
        smoke=smoke,
        device=device,
    )
    effective["training"]["monitor_exact_distribution"] = True
    if smoke:
        effective["training"].update(steps=4, batch_size=4, log_every=2)
    output = new_run(output, effective)
    started = time.perf_counter()
    try:
        shutil.copytree(
            ROOT / "src",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(
            ROOT / "docs/experiments/reward_noise_2026-11-09.md", output / "protocol.md"
        )
        shutil.copy2(clean_run / "task.pt", output / "task.pt")
        shutil.copy2(clean_run / "reward.json", output / "reward.json")
        (output / "clean_heads").mkdir()
        for seed in effective["training_seeds"]:
            shutil.copy2(
                clean_run / f"seed-{seed}/gfn.pt",
                output / f"clean_heads/seed-{seed}.pt",
            )
        write_json(
            output / "provenance.json",
            {
                "input_sha256": config["input_sha256"],
                "clean_run": str(clean_run),
                "torch": torch.__version__,
                "device": device,
                "backbone_loaded": False,
            },
        )
        clean_task, payload = load_task(output)
        costs = []
        for noise_seed in config["noise_seeds"]:
            for sigma in config["noise_sigmas"]:
                directory = noise_condition_path(output, noise_seed, sigma)
                directory.mkdir(parents=True)
                scores, standard = fixed_score_noise(
                    clean_task.scores, sigma, noise_seed
                )
                torch.save(
                    {"scores": scores, "standard_noise": standard},
                    directory / "noise.pt",
                )
                write_json(
                    directory / "noise.json",
                    {
                        "noise_seed": noise_seed,
                        "sigma": sigma,
                        "realized_score_noise_mean": float(
                            (scores - clean_task.scores).mean()
                        ),
                        "realized_score_noise_std": float(
                            (scores - clean_task.scores).std(unbiased=False)
                        ),
                    },
                )
                task = CachedTask(
                    clean_task.tree,
                    payload["features"],
                    payload["logits"],
                    scores,
                    device,
                )
                for seed in effective["training_seeds"]:
                    print(
                        f"Noise seed {noise_seed}, sigma {sigma}, training seed {seed}",
                        flush=True,
                    )
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
                    write_json(
                        destination / "status.json",
                        {
                            "status": "complete",
                            "final_step": effective["training"]["steps"],
                        },
                    )
                    costs.append(
                        {
                            "noise_seed": noise_seed,
                            "sigma": sigma,
                            "training_seed": seed,
                            **cost,
                        }
                    )
                    write_json(output / "costs.json", costs)
                del task
        shutil.copy2(
            clean_run / "evaluation_labels.json", output / "evaluation_labels.json"
        )
        evaluate_noise_study(output, output / "evaluation")
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
    print(output / "evaluation/report.md", flush=True)
