"""Fixed acceptance evaluation and cross-reward diagnostics for three policies."""

import json
from collections import Counter

import numpy as np

from ..artifacts import write_json
from .acceptance_reward import AcceptanceRewardConfig
from .charge import charge_reward
from .continuous_reward import continuous_reward
from .replay_evaluation import summarize_outputs
from .scoring import predict_disorder, score_records

ARMS = ("unchanged", "current_tb", "continuous_tb")


def progression_decision(arms):
    """Require practical raw/noncopy gains over current TB AND its starting head."""
    candidate = arms["continuous_tb"]
    comparisons = {}
    for control in ARMS[:2]:
        reference = arms[control]
        gain = candidate["joint_success_rate"] - reference["joint_success_rate"]
        noncopy = (
            candidate["joint_success_without_exact_training_copy"]
            - reference["joint_success_without_exact_training_copy"]
        )
        unique = candidate["unique_tail_fraction"] - reference["unique_tail_fraction"]
        comparisons[control] = {
            "joint_gain": gain,
            "noncopy_joint_gain": noncopy,
            "unique_fraction_change": unique,
            "passes": gain >= 0.05 and noncopy >= 0.05 and unique >= -0.05,
        }
    return {
        "comparisons": comparisons,
        "passes_practical_screen": all(row["passes"] for row in comparisons.values()),
        "required_joint_gain": 0.05,
        "required_noncopy_joint_gain": 0.05,
        "maximum_unique_fraction_drop": 0.05,
        "interpretation": "One-seed development screen, not statistical significance or exact target matching. No automatic extension.",
    }


def target_shape(fits, bank, penalty):
    """Compare every policy to BOTH rewards on the same unexposed finite bank."""
    by_id = {row["id"]: row for row in bank}
    exposed = {row["id"] for row in fits if row["fresh_training_overlap"]}
    result = {}
    for arm in ARMS:
        for stage in ("initial", "final"):
            for target in ("current", "continuous"):
                for split in ("train", "holdout"):
                    groups = {}
                    for row in fits:
                        if (
                            row["arm"] != arm
                            or row["stage"] != stage
                            or row["split"] != split
                            or (split == "holdout" and row["id"] in exposed)
                        ):
                            continue
                        source = by_id[row["id"]]
                        factor = (
                            source["log_factor"]
                            if target == "current"
                            else -penalty * sum(source["violations"])
                        )
                        delta = row["log_policy"] - row["log_prior"] - factor
                        groups.setdefault(row["prompt_id"], []).append(delta)
                    spreads = {
                        prompt: float(np.std(values)) if len(values) >= 2 else None
                        for prompt, values in groups.items()
                    }
                    valid = [value for value in spreads.values() if value is not None]
                    result[f"{arm}/{stage}/{target}/{split}"] = {
                        "by_prompt_centered_rmse": spreads,
                        "mean_prompt_centered_rmse": float(np.mean(valid))
                        if valid
                        else None,
                        "excluded_exposed_holdout_rows": sum(
                            row["split"] == "holdout" and row["id"] in exposed
                            for row in bank
                        ),
                    }
    return result


def evaluate_continuous(run, output, predictor=predict_disorder):
    """Score snapshots independently of the source runs and pretrained backbone."""
    config = json.loads((run / "pilot_config.json").read_text())
    settings = json.loads((run / "config.json").read_text())
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    references = json.loads((run / "references.json").read_text())
    prompts = {r["id"]: r for r in references if r["role"] == "prompt"}
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    expected = {
        (arm, prompt): config["samples_per_prompt"]
        for arm in ARMS
        for prompt in prompts
    }
    if Counter((row["arm"], row["prompt_id"]) for row in records) != expected:
        raise ValueError("Evaluation counts differ from protocol")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate evaluation identities")
    for row in records:
        prefix = prompts[row["prompt_id"]]["sequence"][: config["prefix_length"]]
        if (
            row["seed"] != 0
            or row["prefix"] != prefix
            or not row["sequence"].startswith(prefix)
            or len(row["sequence"]) != config["length"]
            or row["evaluation_start"] != config["prefix_length"]
        ):
            raise ValueError("Invalid evaluation support")
    bank = json.loads((run / "bank.json").read_text())
    bank_values = continuous_reward(
        [row["tail"] for row in bank],
        [row["isolated_tail_v3_scores"] for row in bank],
        reward,
        settings["penalty"],
    )
    bank = [
        dict(row, violations=bank_values["violations"][i].tolist())
        for i, row in enumerate(bank)
    ]
    by_id = {row["id"]: row for row in bank}
    inherited = [
        json.loads(line)
        for line in (run / "initial_training.jsonl").read_text().splitlines()
    ]
    inherited_ids = {identity for row in inherited for identity in row["replay_ids"]}
    if any(
        by_id[i]["split"] != "train" or not by_id[i]["joint_pass"]
        for i in inherited_ids
    ):
        raise ValueError("Invalid inherited supervised data")
    training_tails = {arm: {by_id[i]["tail"] for i in inherited_ids} for arm in ARMS}
    training, fits, logs_by_arm = {}, [], {}
    for arm in ARMS:
        fits.extend(json.loads((run / arm / "bank_fit.json").read_text()))
        if arm == "unchanged":
            continue
        logs = [
            json.loads(line)
            for line in (run / arm / "training.jsonl").read_text().splitlines()
        ]
        fresh = [
            json.loads(line)
            for line in (run / arm / "fresh_training.jsonl").read_text().splitlines()
        ]
        if (
            [row["step"] for row in logs] != list(range(1, config["steps"] + 1))
            or len(fresh) != config["steps"] * len(prompts)
            or any(row["examples"] != 2 * len(prompts) for row in logs)
        ):
            raise ValueError("Training budget differs from the matched protocol")
        ids = {identity for row in logs for identity in row["replay_ids"]}
        if any(by_id[i]["split"] != "train" for i in ids):
            raise ValueError("Holdout used for training")
        training_tails[arm].update(by_id[i]["tail"] for i in ids)
        training_tails[arm].update(row["tail"] for row in fresh)
        logs_by_arm[arm] = logs
        training[arm] = {
            "updates": len(logs),
            "fresh_trajectories": len(fresh),
            "fresh_successes": sum(row["joint_pass"] for row in fresh),
            "stored_success_presentations": sum(
                row["stored_successes"] for row in logs
            ),
            "clipped_update_fraction": float(
                np.mean(
                    [
                        row["gradient_norm_before_clip"] > config["gradient_clip"]
                        for row in logs
                    ]
                )
            ),
        }
    for current, candidate in zip(
        logs_by_arm["current_tb"], logs_by_arm["continuous_tb"]
    ):
        if (
            current["replay_ids"] != candidate["replay_ids"]
            or current["rates"] != candidate["rates"]
        ):
            raise ValueError("Stored-example order or optimizer schedule differs")
    scored = score_records(records, references, config, predictor)
    tails = [row["sequence"][config["prefix_length"] :] for row in scored]
    values = continuous_reward(
        tails,
        [row["disorder"]["V3"]["isolated_tail_scores"] for row in scored],
        reward,
        settings["penalty"],
    )
    old_scores = charge_reward(tails, config["reward"])[0]
    seed_tails = sorted({row["tail"] for row in bank if row["split"] == "train"})
    seed_codes = np.frombuffer(
        "".join(seed_tails).encode("ascii"), dtype=np.uint8
    ).reshape(len(seed_tails), reward.tail_length)
    for index, row in enumerate(scored):
        row["acceptance"] = {
            key: value[index].tolist() for key, value in values.items()
        }
        row["old_charge_score"] = float(old_scores[index])
        row["own_training_copy"] = tails[index] in training_tails[row["arm"]]
        row["seed_bank_copy"] = tails[index] in seed_tails
        codes = np.frombuffer(tails[index].encode("ascii"), dtype=np.uint8)
        row["nearest_seed_bank_identity"] = float((seed_codes == codes).mean(1).max())
        if row["arm"] != "unchanged":
            factor = values[
                "log_factor" if row["arm"] == "continuous_tb" else "current_log_factor"
            ][index]
            row["own_reward_tb_residual"] = float(
                row["log_z"] + row["log_policy"] - row["log_prior"] - factor
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
    decision = progression_decision(metrics["arms"])
    for name, value in (
        ("sequence_metrics.json", scored),
        ("metrics.json", metrics),
        ("decision.json", decision),
        ("training_summary.json", training),
        ("target_shape.json", target_shape(fits, bank, settings["penalty"])),
    ):
        write_json(output / name, value)
    report = [
        "# Continuous-reward GFlowNet comparison",
        "",
        "Frozen ProGen; unchanged acceptance criteria; one training seed. Current TB is reused; all outputs are freshly sampled.",
        "",
        "| Method | Outputs | Charge | Joint charge/V3 | Unique tails | Training copies | Noncopy joint |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in metrics["arms"].items():
        report.append(
            f"| {arm} | {row['count']} | {row['charge_success_rate']:.2%} | {row['joint_success_rate']:.2%} | {row['unique_tail_fraction']:.2%} | {row['exact_own_training_copy_fraction']:.2%} | {row['joint_success_without_exact_training_copy']:.2%} |"
        )
    report += [
        "",
        f"Practical screen: **{'PASS' if decision['passes_practical_screen'] else 'FAIL'}**.",
        "",
        "Require +5 percentage points raw and noncopy joint success over both controls, with no more than 5 points lost uniqueness. This is not a significance test.",
        "The terminal distribution changes. Cross-reward centered log-ratio diagnostics on the selected bank do not establish global target matching. V3 is optimized; V2 is sensitivity, not biological validation.",
        "The mean_new_log_factor metric is the continuous factor for ALL arms. Own-reward TB residuals use each trained arm's actual reward. See target_shape.json for both target definitions.",
        "No new supervised training occurred. The shared starting head was previously supervised; this experiment trains only a GFlowNet using TB. No automatic extension or penalty sweep.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output / "report.md"
