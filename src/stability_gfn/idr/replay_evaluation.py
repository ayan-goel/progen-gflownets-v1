"""Raw generation, memorization and fit diagnostics for the replay pilot."""

import json
import time
from collections import Counter

import numpy as np

from ..artifacts import write_json
from .acceptance_evaluation import summarize_acceptance
from .acceptance_reward import AcceptanceRewardConfig, joint_acceptance_reward
from .charge import charge_reward
from .scoring import predict_disorder, score_records

ARMS = ("on_policy_tb", "replay_tb", "supervised", "prior")


def summarize_fit(rows):
    """Exclude holdout tails encountered by online training; disclose zero counts."""
    result = {}
    for arm in ARMS[:-1]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        exposed = {
            r["id"]
            for r in arm_rows
            if r["stage"] == "final" and r["fresh_training_overlap"]
        }
        for stage in ("initial", "final"):
            for split in ("train", "holdout"):
                for passed in (False, True):
                    group = [
                        r
                        for r in arm_rows
                        if r["stage"] == stage
                        and r["split"] == split
                        and r["joint_pass"] == passed
                        and not (split == "holdout" and r["id"] in exposed)
                    ]
                    key = f"{arm}/{stage}/bank_{split}/{'pass' if passed else 'fail'}"
                    result[key] = {
                        "count": len(group),
                        "token_nll": float(np.mean([r["token_nll"] for r in group]))
                        if group
                        else None,
                    }
                    if arm != "supervised":
                        result[key]["tb_rmse"] = (
                            float(
                                np.sqrt(np.mean([r["tb_residual"] ** 2 for r in group]))
                            )
                            if group
                            else None
                        )
    return result


def summarize_outputs(rows):
    """Raw endpoints plus exact-copy controls; no filtering of primary rates."""
    result = summarize_acceptance(rows)
    result.update(
        mean_nearest_seed_bank_identity=float(
            np.mean([r["nearest_seed_bank_identity"] for r in rows])
        ),
        exact_seed_bank_copy_fraction=float(
            np.mean([r["seed_bank_copy"] for r in rows])
        ),
        exact_own_training_copy_fraction=float(
            np.mean([r["own_training_copy"] for r in rows])
        ),
        joint_success_without_exact_training_copy=float(
            np.mean(
                [
                    r["acceptance"]["joint_pass"] and not r["own_training_copy"]
                    for r in rows
                ]
            )
        ),
    )
    if all("own_reward_tb_residual" in r for r in rows):
        for passed in (False, True):
            residuals = [
                r["own_reward_tb_residual"]
                for r in rows
                if r["acceptance"]["joint_pass"] == passed
            ]
            result[f"tb_rmse_joint_{'pass' if passed else 'fail'}"] = (
                float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None
            )
    return result


def evaluate_replay(run, output, predictor=predict_disorder):
    """Reconstruct all metrics from the self-contained generation snapshot."""
    started = time.monotonic()
    config = json.loads((run / "pilot_config.json").read_text())
    settings = json.loads((run / "config.json").read_text())
    references = json.loads((run / "references.json").read_text())
    bank = json.loads((run / "bank.json").read_text())
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    prompts = {r["id"]: r for r in references if r["role"] == "prompt"}
    expected = {
        (arm, settings["seed"], prompt): config["samples_per_prompt"]
        for arm in ARMS
        for prompt in prompts
    }
    if Counter(
        (r["arm"], r["seed"], r["prompt_id"]) for r in records
    ) != expected or len({r["id"] for r in records}) != len(records):
        raise ValueError("Generation counts or identities differ from protocol")
    for row in records:
        prefix = prompts[row["prompt_id"]]["sequence"][: config["prefix_length"]]
        if (
            row["prefix"] != prefix
            or not row["sequence"].startswith(prefix)
            or len(row["sequence"]) != config["length"]
            or row["evaluation_start"] != config["prefix_length"]
        ):
            raise ValueError("Invalid evaluation sequence support")
    bank_by_id = {r["id"]: r for r in bank}
    training_tails = {arm: set() for arm in ARMS}
    training_summary = {}
    fits = []
    for arm in ARMS[:-1]:
        logs = [
            json.loads(line)
            for line in (run / arm / "training.jsonl").read_text().splitlines()
        ]
        fresh = [
            json.loads(line)
            for line in (run / arm / "fresh_training.jsonl").read_text().splitlines()
        ]
        if [r["step"] for r in logs] != list(range(1, config["steps"] + 1)) or any(
            r["examples"] != 2 * len(prompts) for r in logs
        ):
            raise ValueError("Training budget differs from protocol")
        if len(fresh) != sum(r["fresh_examples"] for r in logs):
            raise ValueError("Missing fresh training trajectories")
        stored_ids = {identity for r in logs for identity in r["replay_ids"]}
        if any(
            identity not in bank_by_id or bank_by_id[identity]["split"] != "train"
            for identity in stored_ids
        ):
            raise ValueError("Unknown or held-out replay example was used")
        if arm == "supervised" and any(
            not bank_by_id[identity]["joint_pass"] for identity in stored_ids
        ):
            raise ValueError("Supervised control used failed examples")
        training_tails[arm].update(r["tail"] for r in fresh)
        training_tails[arm].update(
            bank_by_id[identity]["tail"] for identity in stored_ids
        )
        training_summary[arm] = {
            "updates": len(logs),
            "trajectory_examples": sum(r["examples"] for r in logs),
            "fresh_trajectories": len(fresh),
            "fresh_successes": sum(r["joint_pass"] for r in fresh),
            "stored_example_draws": sum(len(r["replay_ids"]) for r in logs),
            "distinct_stored_examples_used": len(stored_ids),
            "stored_success_draws": sum(r["stored_successes"] for r in logs),
            "updates_with_no_success": sum(
                r["fresh_joint_passes"] + r["stored_successes"] == 0 for r in logs
            ),
            "clipped_update_fraction": float(
                np.mean(
                    [
                        r["gradient_norm_before_clip"] > config["gradient_clip"]
                        for r in logs
                    ]
                )
            ),
        }
        fits.extend(json.loads((run / arm / "bank_fit.json").read_text()))
    scored = score_records(records, references, config, predictor)
    tails = [r["sequence"][config["prefix_length"] :] for r in scored]
    values = joint_acceptance_reward(
        tails, [r["disorder"]["V3"]["isolated_tail_scores"] for r in scored], reward
    )
    old_scores = charge_reward(tails, config["reward"])[0]
    # Common seed-bank distance is comparable across arms. Exact copying also
    # checks each arm's actually consumed fresh/stored data, with unequal sizes.
    seed_tails = sorted({r["tail"] for r in bank if r["split"] == "train"})
    seed_codes = np.frombuffer(
        "".join(seed_tails).encode("ascii"), dtype=np.uint8
    ).reshape(len(seed_tails), reward.tail_length)
    for index, row in enumerate(scored):
        row["acceptance"] = {
            key: value[index].tolist() for key, value in values.items()
        }
        row["old_charge_score"] = float(old_scores[index])
        tail = tails[index]
        row["seed_bank_copy"] = tail in seed_tails
        row["own_training_copy"] = tail in training_tails[row["arm"]]
        codes = np.frombuffer(tail.encode("ascii"), dtype=np.uint8)
        row["nearest_seed_bank_identity"] = float((seed_codes == codes).mean(1).max())
        if row["arm"] in ("on_policy_tb", "replay_tb"):
            row["own_reward_tb_residual"] = (
                row["log_z"]
                + row["log_policy"]
                - row["log_prior"]
                - row["acceptance"]["log_factor"]
            )
    metrics = {
        "arms": {
            arm: summarize_outputs([r for r in scored if r["arm"] == arm])
            for arm in ARMS
        },
        "by_prompt": {
            f"{arm}/{prompt}": summarize_outputs(
                [r for r in scored if r["arm"] == arm and r["prompt_id"] == prompt]
            )
            for arm in ARMS
            for prompt in prompts
        },
    }
    control = metrics["arms"]["on_policy_tb"]
    decisions = {}
    for arm in ("replay_tb", "supervised"):
        candidate = metrics["arms"][arm]
        gain = candidate["joint_success_rate"] - control["joint_success_rate"]
        unique_change = (
            candidate["unique_tail_fraction"] - control["unique_tail_fraction"]
        )
        noncopy_gain = (
            candidate["joint_success_without_exact_training_copy"]
            - control["joint_success_without_exact_training_copy"]
        )
        decisions[arm] = {
            "joint_gain_over_on_policy_tb": gain,
            "unique_fraction_change": unique_change,
            "noncopy_joint_gain": noncopy_gain,
            "passes_practical_screen": gain >= 0.05
            and unique_change >= -0.05
            and noncopy_gain >= 0.05,
        }
    decision = {
        "arms": decisions,
        "required_joint_gain": 0.05,
        "maximum_unique_fraction_drop": 0.05,
        "required_noncopy_joint_gain": 0.05,
        "interpretation": "One-seed development screen, not significance or exact target matching. Supervised imitation is not a GFlowNet. No automatic extension.",
    }
    fit_summary = summarize_fit(fits)
    for name, value in (
        ("sequence_metrics.json", scored),
        ("metrics.json", metrics),
        ("decision.json", decision),
        ("training_summary.json", training_summary),
        ("fit_summary.json", fit_summary),
    ):
        write_json(output / name, value)
    report = [
        "# Replay TB and supervised same-head pilot",
        "",
        "One training seed; frozen ProGen; raw outputs without filtering. V3 disorder is an optimized proxy, not biological validation.",
        "",
        "| Method | Outputs | Charge | Joint charge/V3 | V3 disorder | Unique tails | Exact own-training copy | Joint without exact copy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in metrics["arms"].items():
        report.append(
            f"| {arm} | {values['count']} | {values['charge_success_rate']:.2%} | {values['joint_success_rate']:.2%} | {values['v3_disorder_success_rate']:.2%} | {values['unique_tail_fraction']:.2%} | {values['exact_own_training_copy_fraction']:.2%} | {values['joint_success_without_exact_training_copy']:.2%} |"
        )
    report += [
        "",
        "## Predeclared progression screen",
        "",
        "Require +5 percentage points joint success and +5 points joint success excluding exact training copies, with no more than 5 points lost uniqueness, relative to the fresh on-policy TB control.",
        "",
    ]
    for arm, result in decisions.items():
        report.append(
            f"- {arm}: {'PASS' if result['passes_practical_screen'] else 'FAIL'}; joint gain {result['joint_gain_over_on_policy_tb']:+.2%}, noncopy gain {result['noncopy_joint_gain']:+.2%}."
        )
    report += [
        "",
        "## Interpretation and diagnostics",
        "",
        "Both TB arms use the same updates and trajectory-example count. Replay uses fewer new rollouts plus historical training examples; discovery cost, unique data exposure and wall time are not matched. The supervised arm is an easier imitation control with the same architecture, not a target-distribution estimator.",
        "",
        "See fit_summary.json for initial/final positive and negative fit on bank training/holdout rows; missing strata are null. Holdout tails encountered during fresh training are excluded from both initial/final holdout summaries. For on-policy TB, bank_train is a reference partition, not data supplied to its optimizer.",
        "",
        "See metrics.json for every prompt, V2 sensitivity, repetition, seed-bank similarity and success/failure TB residuals. No exact copying does not establish global novelty. Check nearest-seed-bank identity before interpreting success as generalization.",
        "",
        "A replay gain motivates seed confirmation. Supervised-only gains motivate revisiting TB credit assignment. Failure of both requires inspecting fit/coverage/architecture; it does not prove that unfreezing is necessary. A pass never launches another run.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    write_json(
        output / "evaluation_cost.json",
        {"seconds": time.monotonic() - started, "raw_records": len(records)},
    )
    return output / "report.md"
