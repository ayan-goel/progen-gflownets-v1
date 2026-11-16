"""Fresh-output and inherited-memorization checks for supervised-start guidance."""

import json
import time
from collections import Counter

import numpy as np

from ..artifacts import write_json
from .acceptance_reward import AcceptanceRewardConfig, joint_acceptance_reward
from .charge import charge_reward
from .replay_evaluation import summarize_outputs
from .scoring import predict_disorder, score_records

ARMS = ("unchanged", "continued_supervised", "warm_tb")


def summarize_fit(rows):
    """Exclude holdout tails encountered by online training; disclose zero counts."""
    result = {}
    exposed = {r["id"] for r in rows if r["fresh_training_overlap"]}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
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
                    if arm == "warm_tb":
                        result[key]["tb_rmse"] = (
                            float(
                                np.sqrt(np.mean([r["tb_residual"] ** 2 for r in group]))
                            )
                            if group
                            else None
                        )
    return result


def evaluate_warmstart(run, output, predictor=predict_disorder):
    """Reconstruct all metrics from the self-contained generation snapshot."""
    started = time.monotonic()
    config = json.loads((run / "pilot_config.json").read_text())
    references = json.loads((run / "references.json").read_text())
    bank = json.loads((run / "bank.json").read_text())
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    prompts = {r["id"]: r for r in references if r["role"] == "prompt"}
    expected = {
        (arm, 0, prompt): config["samples_per_prompt"]
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
    initial_logs = [
        json.loads(line)
        for line in (run / "initial_training.jsonl").read_text().splitlines()
    ]
    inherited_ids = {identity for row in initial_logs for identity in row["replay_ids"]}
    if any(
        identity not in bank_by_id
        or bank_by_id[identity]["split"] != "train"
        or not bank_by_id[identity]["joint_pass"]
        for identity in inherited_ids
    ):
        raise ValueError("Invalid inherited supervised training data")
    inherited_tails = {bank_by_id[identity]["tail"] for identity in inherited_ids}
    # Every branch has already learned from the initial supervised data. Count
    # copies against that history as well as newly consumed examples.
    training_tails = {arm: set(inherited_tails) for arm in ARMS}
    training_summary = {
        "unchanged": {
            "updates": 0,
            "trajectory_examples": 0,
            "inherited_distinct_training_tails": len(inherited_tails),
        }
    }
    fits = json.loads((run / "unchanged/bank_fit.json").read_text())
    for arm in ARMS[1:]:
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
        if arm == "continued_supervised" and any(
            not bank_by_id[identity]["joint_pass"] for identity in stored_ids
        ):
            raise ValueError("Supervised control used failed examples")
        training_tails[arm].update(r["tail"] for r in fresh)
        training_tails[arm].update(
            bank_by_id[identity]["tail"] for identity in stored_ids
        )
        training_summary[arm] = {
            "updates": len(logs),
            "inherited_distinct_training_tails": len(inherited_tails),
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
        if row["arm"] == "warm_tb":
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
    decision = progression_decision(metrics["arms"])
    fit_summary = summarize_fit(fits)
    shape_summary = summarize_target_shape(fits, bank)
    for name, value in (
        ("sequence_metrics.json", scored),
        ("metrics.json", metrics),
        ("decision.json", decision),
        ("training_summary.json", training_summary),
        ("fit_summary.json", fit_summary),
        ("target_shape.json", shape_summary),
    ):
        write_json(output / name, value)
    report = [
        "# Supervised-start GFlowNet comparison",
        "",
        "One development seed; frozen ProGen; fresh raw outputs, no filtering. V3 disorder is an optimized proxy, not biological validation.",
        "",
        "| Method | Outputs | Charge | Joint charge/V3 | V3 disorder | Unique tails | Exact training copy | Joint without exact copy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in metrics["arms"].items():
        report.append(
            f"| {arm} | {values['count']} | {values['charge_success_rate']:.2%} | {values['joint_success_rate']:.2%} | {values['v3_disorder_success_rate']:.2%} | {values['unique_tail_fraction']:.2%} | {values['exact_own_training_copy_fraction']:.2%} | {values['joint_success_without_exact_training_copy']:.2%} |"
        )
    report += [
        "",
        "## Predeclared progression screen",
        "",
        "Warm TB must gain at least 5 percentage points in raw AND noncopy joint success over EACH control, losing no more than 5 points in uniqueness. A one-seed screen is not a significance test.",
        "",
    ]
    for control, result in decision["comparisons"].items():
        report.append(
            f"- Against {control}: {'PASS' if result['passes_practical_screen'] else 'FAIL'}; joint gain {result['joint_gain']:+.2%}, noncopy gain {result['noncopy_joint_gain']:+.2%}."
        )
    report += [
        "",
        f"Overall: **{'PASS' if decision['passes_practical_screen'] else 'FAIL'}**.",
        "",
        "## Interpretation",
        "",
        "Both continuations start from the identical supervised head, reset Adam and use the same head-rate schedule and update/example budget. Their success exposure, fresh sampling and wall time differ. The unchanged head is evaluated afresh, not compared with its old 8.69% estimate.",
        "",
        "Copies include inherited supervised examples as well as new training. The common seed bank supports comparable identity checks; this does not establish global novelty. See metrics.json for prompt-level outcomes, V2 sensitivity, repetition and success/failure TB residuals.",
        "",
        "fit_summary.json separates positive/negative bank fitting. target_shape.json reports within-prompt centered log-policy/target discrepancies on the fixed bank, independent of the learned normalizer. It excludes holdout tails exposed by either continuation from all arms. This selected finite bank is not an exact global distribution test.",
        "",
        "The TB normalizer starts at the equal-pass/fail training-bank least-squares offset, not an estimate of the true partition function. See warm_tb/initialization.json. No held-out outcomes initialize it.",
        "",
        "A practical gain may justify seed confirmation. No gain over continued supervision ends this bounded warm-start route; do not automatically extend or tune it. Property success and matching the soft target are different claims, so inspect target-fit diagnostics before interpretation.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    write_json(
        output / "evaluation_cost.json",
        {"seconds": time.monotonic() - started, "raw_records": len(records)},
    )
    return output / "report.md"


def progression_decision(arms):
    """Require meaningful improvement over BOTH freshly evaluated controls."""
    candidate = arms["warm_tb"]
    comparisons = {}
    for control in ("unchanged", "continued_supervised"):
        reference = arms[control]
        gain = candidate["joint_success_rate"] - reference["joint_success_rate"]
        noncopy_gain = (
            candidate["joint_success_without_exact_training_copy"]
            - reference["joint_success_without_exact_training_copy"]
        )
        unique_change = (
            candidate["unique_tail_fraction"] - reference["unique_tail_fraction"]
        )
        comparisons[control] = {
            "joint_gain": gain,
            "noncopy_joint_gain": noncopy_gain,
            "unique_fraction_change": unique_change,
            "passes_practical_screen": gain >= 0.05
            and noncopy_gain >= 0.05
            and unique_change >= -0.05,
        }
    return {
        "comparisons": comparisons,
        "passes_practical_screen": all(
            r["passes_practical_screen"] for r in comparisons.values()
        ),
        "required_joint_gain": 0.05,
        "required_noncopy_joint_gain": 0.05,
        "maximum_unique_fraction_drop": 0.05,
        "interpretation": "One-seed development screen; not significance or exact target matching. No automatic extension.",
    }


def summarize_target_shape(fits, bank):
    """Within-prompt log-ratio spread removes any fitted normalizer offset.

    This selected-bank diagnostic measures relative probabilities only on observed
    sequences. It cannot establish total mass, global coverage, or exact sampling.
    Exclude the union of online-exposed holdout tails for comparable arms/stages.
    """
    by_id = {row["id"]: row for row in bank}
    exposed = {r["id"] for r in fits if r["fresh_training_overlap"]}
    summaries = {}
    for arm in ARMS:
        for stage in ("initial", "final"):
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
                    delta = (
                        row["log_policy"]
                        - row["log_prior"]
                        - by_id[row["id"]]["log_factor"]
                    )
                    groups.setdefault(row["prompt_id"], []).append(delta)
                per_prompt = {
                    prompt: {
                        "count": len(values),
                        "centered_rmse": float(np.std(values))
                        if len(values) >= 2
                        else None,
                    }
                    for prompt, values in groups.items()
                }
                spreads = [
                    r["centered_rmse"]
                    for r in per_prompt.values()
                    if r["centered_rmse"] is not None
                ]
                summaries[f"{arm}/{stage}/{split}"] = {
                    "by_prompt": per_prompt,
                    "mean_prompt_centered_rmse": float(np.mean(spreads))
                    if spreads
                    else None,
                    "excluded_exposed_holdout_rows": sum(
                        r["split"] == "holdout" and r["id"] in exposed for r in bank
                    ),
                }
    return summaries
