"""Fresh full-support generation evaluation; never infer global TV from bank fit."""

import json
from collections import Counter

import numpy as np

from ..artifacts import write_json
from .acceptance_reward import AcceptanceRewardConfig
from .charge import charge_reward
from .continuous_reward import continuous_reward
from .replay_evaluation import summarize_outputs
from .scoring import predict_disorder, score_records

ARMS = ("unchanged", "plain_tb", "svrg_tb")


def progression_decision(arms):
    candidate = arms["svrg_tb"]
    comparisons = {}
    for name in ARMS[:2]:
        reference = arms[name]
        gain = candidate["joint_success_rate"] - reference["joint_success_rate"]
        noncopy = (
            candidate["joint_success_without_seed_bank_copy"]
            - reference["joint_success_without_seed_bank_copy"]
        )
        unique = candidate["unique_tail_fraction"] - reference["unique_tail_fraction"]
        comparisons[name] = dict(
            joint_gain=gain,
            noncopy_joint_gain=noncopy,
            unique_fraction_change=unique,
            passes=gain >= 0.05 and noncopy >= 0.05 and unique >= -0.05,
        )
    return dict(
        comparisons=comparisons,
        passes_practical_screen=all(r["passes"] for r in comparisons.values()),
        interpretation="One-seed development screen, not statistical significance or exact global target matching.",
    )


def summarize(rows):
    result = summarize_outputs(rows)
    result["joint_success_without_seed_bank_copy"] = float(
        np.mean(
            [r["acceptance"]["joint_pass"] and not r["seed_bank_copy"] for r in rows]
        )
    )
    result["mean_unseen_decision_prefix_fraction"] = float(
        np.mean([r["unseen_decision_prefix_fraction"] for r in rows])
    )
    return result


def evaluate_transfer(run, output, predictor=predict_disorder):
    config = json.loads((run / "pilot_config.json").read_text())
    settings = json.loads((run / "config.json").read_text())
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    references = json.loads((run / "references.json").read_text())
    prompts = {r["id"]: r for r in references if r["role"] == "prompt"}
    bank = json.loads((run / "bank.json").read_text())
    if any(r["split"] != "train" for r in bank):
        raise ValueError("Transfer snapshot must contain training rows only")
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    expected = {
        (arm, prompt): config["samples_per_prompt"]
        for arm in ARMS
        for prompt in prompts
    }
    if Counter((r["arm"], r["prompt_id"]) for r in records) != expected or len(
        {r["id"] for r in records}
    ) != len(records):
        raise ValueError("Fresh generation counts or identities differ")
    for row in records:
        prefix = prompts[row["prompt_id"]]["sequence"][: config["prefix_length"]]
        if (
            row["seed"] != 0
            or row["prefix"] != prefix
            or not row["sequence"].startswith(prefix)
            or len(row["sequence"]) != config["length"]
            or row["evaluation_start"] != config["prefix_length"]
        ):
            raise ValueError("Invalid generation support")
    by_id = {r["id"]: r for r in bank}
    inherited = [
        json.loads(line)
        for line in (run / "initial_training.jsonl").read_text().splitlines()
    ]
    initial_ids = {identity for row in inherited for identity in row["replay_ids"]}
    if not initial_ids.issubset(by_id):
        raise ValueError("Starting training identities outside training bank")
    own_tails = {
        "unchanged": {by_id[i]["tail"] for i in initial_ids},
        "plain_tb": {r["tail"] for r in bank},
        "svrg_tb": {r["tail"] for r in bank},
    }
    logs = {}
    for arm in ARMS[1:]:
        logs[arm] = [
            json.loads(line)
            for line in (run / arm / "training.jsonl").read_text().splitlines()
        ]
        if [r["step"] for r in logs[arm]] != list(range(1, config["steps"] + 1)):
            raise ValueError("Incomplete training updates")
        for row in logs[arm]:
            if len(row["replay_ids"]) != len(prompts) * settings[
                "draws_per_prompt"
            ] or any(i not in by_id for i in row["replay_ids"]):
                raise ValueError("Invalid sampled training identities/budget")
    for first, second in zip(logs["plain_tb"], logs["svrg_tb"]):
        if (
            first["replay_ids"] != second["replay_ids"]
            or first["rates"] != second["rates"]
        ):
            raise ValueError("Matched draws or optimizer schedule differ")
    scored = score_records(records, references, config, predictor)
    tails = [r["sequence"][config["prefix_length"] :] for r in scored]
    values = continuous_reward(
        tails,
        [r["disorder"]["V3"]["isolated_tail_scores"] for r in scored],
        reward,
        settings["penalty"],
    )
    old_scores = charge_reward(tails, config["reward"])[0]
    seed_tails = sorted({r["tail"] for r in bank})
    seed_codes = np.frombuffer(
        "".join(seed_tails).encode("ascii"), dtype=np.uint8
    ).reshape(len(seed_tails), reward.tail_length)
    known_prefixes = {
        prompt: {
            r["tail"][:step]
            for r in bank
            if r["prompt_id"] == prompt
            for step in range(reward.tail_length)
        }
        for prompt in prompts
    }
    for index, row in enumerate(scored):
        tail = tails[index]
        row["acceptance"] = {k: v[index].tolist() for k, v in values.items()}
        row["old_charge_score"] = float(old_scores[index])
        row["own_training_copy"] = tail in own_tails[row["arm"]]
        row["seed_bank_copy"] = tail in seed_tails
        codes = np.frombuffer(tail.encode("ascii"), dtype=np.uint8)
        row["nearest_seed_bank_identity"] = float((seed_codes == codes).mean(1).max())
        row["unseen_decision_prefix_fraction"] = (
            sum(
                tail[:step] not in known_prefixes[row["prompt_id"]]
                for step in range(reward.tail_length)
            )
            / reward.tail_length
        )
        row["own_reward_tb_residual"] = (
            row["log_z"]
            + row["log_policy"]
            - row["log_prior"]
            - float(values["log_factor"][index])
        )
    metrics = dict(
        arms={arm: summarize([r for r in scored if r["arm"] == arm]) for arm in ARMS},
        by_prompt={
            f"{arm}/{prompt}": summarize(
                [r for r in scored if r["arm"] == arm and r["prompt_id"] == prompt]
            )
            for arm in ARMS
            for prompt in prompts
        },
    )
    decision = progression_decision(metrics["arms"])
    bank_diagnostics = {}
    for arm in ARMS:
        fits = json.loads((run / arm / "bank_fit.json").read_text())
        for stage in ("initial", "final"):
            for prompt in prompts:
                rows = [
                    r for r in fits if r["stage"] == stage and r["prompt_id"] == prompt
                ]
                residuals = np.array([r["tb_residual"] for r in rows])
                log_policy = np.array([r["log_policy"] for r in rows])
                maximum = log_policy.max()
                bank_diagnostics[f"{arm}/{stage}/{prompt}"] = dict(
                    count=len(rows),
                    tb_rmse=float(np.sqrt(np.mean(residuals**2))),
                    centered_log_ratio_rmse=float(residuals.std()),
                    log_policy_mass_on_bank=float(
                        maximum + np.log(np.exp(log_policy - maximum).sum())
                    ),
                )
    for name, value in (
        ("sequence_metrics.json", scored),
        ("metrics.json", metrics),
        ("decision.json", decision),
        ("bank_diagnostics.json", bank_diagnostics),
    ):
        write_json(output / name, value)
    report = [
        "# Full-support IDR transfer pilot",
        "",
        "Frozen ProGen; all 20 residues at every position; one training seed. All evaluation outputs are fresh.",
        "",
        "| Head | Outputs | Charge | Joint charge/V3 | Unique tails | Bank copies | Noncopy joint |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in metrics["arms"].items():
        report.append(
            f"| {arm} | {row['count']} | {row['charge_success_rate']:.2%} | {row['joint_success_rate']:.2%} | {row['unique_tail_fraction']:.2%} | {row['exact_seed_bank_copy_fraction']:.2%} | {row['joint_success_without_seed_bank_copy']:.2%} |"
        )
    report += [
        "",
        f"Practical screen: **{'PASS' if decision['passes_practical_screen'] else 'FAIL'}**.",
        "",
        "Require +5 points raw and common-bank-noncopy joint success over both controls, with no more than 5 points lost uniqueness. No automatic extension.",
        "Bank log-mass and centered residuals are fitting/coverage diagnostics, not global target TV. Fixed-bank training does not establish unrestricted target matching.",
        "V3 disorder is optimized; V2 is sensitivity, not independent biological validation. Copy controls do not prove novelty outside the bank.",
        "See generation_diagnostics.json for input amplification on fresh states and metrics.json for unseen-prefix fractions. Costs include snapshot work.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output / "report.md"
