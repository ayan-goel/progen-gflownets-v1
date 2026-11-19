"""Compare 1600-update IDR fits with freshly sampled preserved 800-step models.

The primary comparison changes both the update budget and the absolute decay
point. Copy controls retain the 400-step training history as well as both current
budgets, using one common union for all six policies in each replicate.
"""

import json
from pathlib import Path

import numpy as np

from ..artifacts import file_hash, write_json
from .acceptance_experiment import verify_implementation, verify_manifest
from .acceptance_reward import AcceptanceRewardConfig
from .budget_evaluation import _read_scored, _summary, _validate_records
from .charge import charge_reward
from .confirmation_evaluation import SUMMARY_METRICS
from .continuous_reward import continuous_reward
from .lastblock_evaluation import (
    ARMS,
    progression_decision,
    read_jsonl,
    validate_support,
    validate_training,
)
from .scoring import predict_disorder, score_records

BUDGET1600_ARMS = ("prior", *ARMS, "lastblock_800")


def budget1600_decision(replicates):
    """Require meaningful gains over the paired old model in both fresh samples."""
    if len(replicates) != 2:
        raise ValueError("Budget comparison requires exactly two paired replicates")
    decisions = {}
    for name, metrics in replicates.items():
        arms = metrics["arms"]
        candidate, control = arms["lastblock_tb"], arms["lastblock_800"]
        comparison = {
            "joint_gain": candidate["joint_success_rate"]
            - control["joint_success_rate"],
            "common_noncopy_joint_gain": (
                candidate["joint_success_without_common_training_copy"]
                - control["joint_success_without_common_training_copy"]
            ),
            "unique_fraction_change": (
                candidate["unique_tail_fraction"] - control["unique_tail_fraction"]
            ),
            "disorder_success_change": (
                candidate["v3_disorder_success_rate"]
                - control["v3_disorder_success_rate"]
            ),
        }
        comparison["passes"] = (
            comparison["joint_gain"] >= 0.05 - 1e-12
            and comparison["common_noncopy_joint_gain"] >= 0.05 - 1e-12
            and comparison["unique_fraction_change"] >= -0.05 - 1e-12
            and comparison["disorder_success_change"] >= -0.05 - 1e-12
        )
        decisions[name] = {
            "paired_800_control": comparison,
            "inherited_1600_screen_secondary": progression_decision(arms),
        }
    return {
        "replicates": decisions,
        "passes_budget_screen": all(
            row["paired_800_control"]["passes"] for row in decisions.values()
        ),
        "interpretation": (
            "Both paired fits must pass; seed means do not determine passing. "
            "Practical development thresholds, not significance tests or biological "
            "validation. The inherited 1600-step screen is secondary, not an extra gate."
        ),
    }


def evaluate_budget1600(run, output, predictor=predict_disorder):
    """Score preserved-model outputs and combine already evaluated child policies.

    Recovery callers first reconstruct fit/prior scores under ``output`` using
    their existing evaluators. No model checkpoint is loaded by this function.
    """
    run, output = Path(run), Path(output)
    manifest = verify_manifest(run)
    if not {
        "config.json",
        "provenance.json",
        "budget_config.json",
        "budget_generation.json",
    }.issubset(manifest["files"]):
        raise ValueError("Budget settings or provenance are not protected")
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    settings = json.loads((run / "config.json").read_text())
    if (
        file_hash(run / "budget_generation.json")
        != settings["budget_generation_sha256"]
    ):
        raise ValueError("Pinned budget manifest differs")
    budget_manifest = json.loads((run / "budget_generation.json").read_text())
    if not budget_manifest["complete"] or file_hash(
        run / "budget_config.json"
    ) != budget_manifest["files"].get("config.json"):
        raise ValueError("Copied source-budget settings differ")
    planned = settings["replicates"]
    names = [row["name"] for row in planned]
    if (
        len(names) != 2
        or len(set(names)) != 2
        or any(Path(name).name != name or name in (".", "..") for name in names)
        or len({row["evaluation_seed"] for row in planned}) != 2
    ):
        raise ValueError("Need two distinct replicate names and evaluation streams")
    metrics, all_records, copy_definitions = {}, [], {}
    expected_root_records = []
    common_prompts = None
    common_config = None
    for replicate in planned:
        name = replicate["name"]
        fit, control = run / name / "fit", run / name / "control800"
        prior = run / name / "prior"
        verify_manifest(fit)
        verify_manifest(prior)
        required = {
            f"{name}/control800/{filename}"
            for filename in (
                "config.json",
                "checkpoint.pt",
                "source_settings.json",
                "bank.json",
                "training_rows.jsonl",
                "source_generation.json",
                "samples.jsonl",
            )
        }
        history_path = run / name / "history400_training_rows.jsonl"
        required.add(str(history_path.relative_to(run)))
        if not required.issubset(manifest["files"]):
            raise ValueError("Preserved control is not protected by the root manifest")
        if file_hash(history_path) != budget_manifest["files"].get(
            f"{name}/control400/training_rows.jsonl"
        ):
            raise ValueError("Preserved 400-step training history differs")
        current = json.loads((fit / "config.json").read_text())
        original = json.loads((control / "source_settings.json").read_text())
        control_settings = json.loads((control / "config.json").read_text())
        if (
            control_settings["arm"] != "lastblock_800"
            or control_settings["evaluation_seed"] != replicate["evaluation_seed"]
            or control_settings["source_checkpoint_sha256"]
            != file_hash(control / "checkpoint.pt")
        ):
            raise ValueError("Preserved-control evaluation settings differ")
        if (
            current["training_steps"] != settings["training_steps"]
            or current["training_steps"] != 2 * original["training_steps"]
            or current["evaluation_seed"] != replicate["evaluation_seed"]
            or {
                key: value
                for key, value in current.items()
                if key not in ("training_steps", "evaluation_seed")
            }
            != {
                key: value
                for key, value in original.items()
                if key not in ("training_steps", "evaluation_seed")
            }
        ):
            raise ValueError(
                "Budget/control settings differ beyond steps and fresh evaluation"
            )
        source_manifest = json.loads((control / "source_generation.json").read_text())
        if not source_manifest["complete"] or file_hash(
            control / "source_generation.json"
        ) != budget_manifest["files"].get(f"{name}/fit/generation.json"):
            raise ValueError("Preserved control does not match the pinned budget")
        for destination, source_name in (
            (control / "checkpoint.pt", "lastblock_tb/checkpoint.pt"),
            (control / "source_settings.json", "config.json"),
            (control / "bank.json", "bank.json"),
            (control / "training_rows.jsonl", "training_rows.jsonl"),
            (fit / "reward.json", "reward.json"),
            (fit / "references.json", "references.json"),
        ):
            if file_hash(destination) != source_manifest["files"].get(source_name):
                raise ValueError(f"Preserved 800-step input differs: {source_name}")
        config = json.loads((fit / "pilot_config.json").read_text())
        references = json.loads((fit / "references.json").read_text())
        prompts = {row["id"]: row for row in references if row["role"] == "prompt"}
        if common_config is not None and (
            config != common_config or prompts != common_prompts
        ):
            raise ValueError("Budget replicate support or scoring settings differ")
        common_config, common_prompts = config, prompts
        reward = AcceptanceRewardConfig(**json.loads((fit / "reward.json").read_text()))
        bank = json.loads((fit / "bank.json").read_text())
        if bank != json.loads((control / "bank.json").read_text()):
            raise ValueError("Training banks differ between budgets")
        fresh = read_jsonl(fit / "training_rows.jsonl")
        old_fresh = read_jsonl(control / "training_rows.jsonl")
        history400 = read_jsonl(history_path)
        _, current_tails, _, _ = validate_training(
            fit, current, config, prompts, bank, fresh
        )
        for row in old_fresh + history400:
            validate_support(row, prompts, config)
        common_tails = current_tails | {row["tail"] for row in old_fresh + history400}
        seed_tails = sorted({row["tail"] for row in bank if row["split"] == "train"})
        seed_codes = np.frombuffer(
            "".join(seed_tails).encode("ascii"), dtype=np.uint8
        ).reshape(len(seed_tails), reward.tail_length)
        records = _read_scored(output / name / "fit", fit, ARMS, prompts, config)
        records += _read_scored(
            output / name / "prior", prior, ("prior",), prompts, config
        )
        raw_control = read_jsonl(control / "samples.jsonl")
        _validate_records(raw_control, ("lastblock_800",), prompts, config)
        records += score_records(raw_control, references, config, predictor)
        _validate_records(records, BUDGET1600_ARMS, prompts, config)
        for part in (fit, control, prior):
            expected_root_records.extend(
                dict(row, id=f"{name}/{row['id']}", replicate=name)
                for row in read_jsonl(part / "samples.jsonl")
            )
        for row in records:
            for version in ("V3", "V2"):
                for field, length in (
                    ("full_scores", config["length"]),
                    ("isolated_tail_scores", reward.tail_length),
                ):
                    scores = np.asarray(row["disorder"][version][field])
                    if (
                        scores.shape != (length,)
                        or not np.isfinite(scores).all()
                        or (scores < 0).any()
                        or (scores > 1).any()
                    ):
                        raise ValueError("Invalid saved disorder scores")
        tails = [row["sequence"][config["prefix_length"] :] for row in records]
        disorder = [row["disorder"]["V3"]["isolated_tail_scores"] for row in records]
        values = continuous_reward(tails, disorder, reward, current["penalty"])
        old_scores = charge_reward(tails, config["reward"])[0]
        for index, row in enumerate(records):
            tail = tails[index]
            row["acceptance"] = {
                key: value[index].tolist() for key, value in values.items()
            }
            row["old_charge_score"] = float(old_scores[index])
            row["common_training_copy"] = tail in common_tails
            row["seed_bank_copy"] = tail in seed_tails
            row.pop("own_training_copy", None)
            codes = np.frombuffer(tail.encode("ascii"), dtype=np.uint8)
            row["nearest_seed_bank_identity"] = float(
                (seed_codes == codes).mean(1).max()
            )
            if row.get("log_z") is not None:
                row["own_reward_tb_residual"] = (
                    row["log_z"]
                    + row["log_policy"]
                    - row["log_prior"]
                    - row["acceptance"]["log_factor"]
                )
            row["replicate"] = name
            row["id"] = f"{name}/{row['id']}"
        metrics[name] = {
            "arms": {
                arm: _summary([row for row in records if row["arm"] == arm])
                for arm in BUDGET1600_ARMS
            },
            "by_prompt": {
                f"{arm}/{prompt}": _summary(
                    [
                        row
                        for row in records
                        if row["arm"] == arm and row["prompt_id"] == prompt
                    ]
                )
                for arm in BUDGET1600_ARMS
                for prompt in prompts
            },
            "evaluation_seed": replicate["evaluation_seed"],
        }
        copy_definitions[name] = {
            "common_training_unique_tails": len(common_tails),
            "history400_fresh_trajectories": len(history400),
            "old_fresh_trajectories": len(old_fresh),
            "new_fresh_trajectories": len(fresh),
        }
        all_records.extend(records)
    root_records = read_jsonl(run / "samples.jsonl")
    if (
        root_records != expected_root_records
        or len(root_records) != manifest["count"]
        or len({row["id"] for row in root_records}) != len(root_records)
    ):
        raise ValueError("Root generation does not match concatenated child outputs")
    decision = budget1600_decision(metrics)
    summary = {
        arm: {
            field: {
                "mean": float(np.mean(values)),
                "minimum": min(values),
                "maximum": max(values),
            }
            for field in SUMMARY_METRICS
            for values in [[metrics[name]["arms"][arm][field] for name in names]]
        }
        for arm in BUDGET1600_ARMS
    }
    write_json(output / "sequence_metrics.json", all_records)
    write_json(
        output / "metrics.json",
        {
            "replicates": metrics,
            "equal_weight_seed_summary": summary,
        },
    )
    write_json(output / "decision.json", decision)
    write_json(
        output / "copy_definition.json",
        {
            "replicates": copy_definitions,
            "common_copy": "Union of original train bank, historical 400-step fresh data, preserved 800-step fresh data and new 1600-step fresh data, shared by all six arms within a replicate",
            "denominator": "All raw outputs; copies and failures are never removed from the denominator",
            "own_copy": "Omitted from parent outputs; child fit/prior files retain their original own-training definitions",
        },
    )
    lines = [
        "# IDR larger-training-budget comparison",
        "",
        "Two seed-matched reruns from the original initialization, comparing final 1600-update models with freshly sampled preserved 800-update models. The learning-rate decay is stretched from update 600 to 1200: this tests a larger-budget schedule, not duration alone or continuation of a saved optimizer.",
        "",
        "| Replicate | Policy | Outputs | Joint | Noncopy joint | Charge | V3 disorder | Unique | Common copies |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        for arm in BUDGET1600_ARMS:
            row = metrics[name]["arms"][arm]
            lines.append(
                f"| {name} | {arm} | {row['count']} | {row['joint_success_rate']:.2%} | "
                f"{row['joint_success_without_common_training_copy']:.2%} | "
                f"{row['charge_success_rate']:.2%} | {row['v3_disorder_success_rate']:.2%} | "
                f"{row['unique_tail_fraction']:.2%} | {row['exact_common_training_copy_fraction']:.2%} |"
            )
    lines += [
        "",
        "## Primary paired comparison",
        "",
        "| Replicate | Joint gain (pp) | Noncopy gain (pp) | Unique change (pp) | Disorder change (pp) | Screen |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, result in decision["replicates"].items():
        row = result["paired_800_control"]
        lines.append(
            f"| {name} | {100 * row['joint_gain']:+.2f} | "
            f"{100 * row['common_noncopy_joint_gain']:+.2f} | "
            f"{100 * row['unique_fraction_change']:+.2f} | "
            f"{100 * row['disorder_success_change']:+.2f} | {'PASS' if row['passes'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"Overall budget screen: **{'PASS' if decision['passes_budget_screen'] else 'FAIL'}**.",
        "",
        "Both replicates must gain at least 5 percentage points in raw and common-training-noncopy joint success over their freshly sampled 800-step control, losing at most 5 points in uniqueness and V3 disorder. Inherited comparisons against unchanged, frozen TB and continued supervised are reported in decision.json as secondary diagnostics; they are not additional gates.",
        "",
        "## Joint success by prompt",
        "",
        "| Replicate | Prompt | Plain ProGen | Unchanged | Continued supervised | Frozen TB 1600 | Last-block TB 1600 | Last-block TB 800 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        for prompt in common_prompts:
            rates = " | ".join(
                f"{metrics[name]['by_prompt'][f'{arm}/{prompt}']['joint_success_rate']:.2%}"
                for arm in BUDGET1600_ARMS
            )
            lines.append(f"| {name} | {prompt} | {rates} |")
    lines += [
        "",
        "All arms use the union of original training tails and the 400-, 800- and 1600-update fresh training tails for common-copy controls; every raw output stays in the denominator. The 400-update history is retained for consistency with the accepted 800-update comparison; no 400-update policy is evaluated again. Own-copy metrics are omitted from this combined report because they refer to different exposures. Child fit/prior outputs preserve their narrower original definitions.",
        "",
        "The same prompts, target and initialization remain development settings. Fresh evaluation uses the same declared seed across arms, not necessarily matched generated sequences. Predicted disorder is not biological validation, noncopy is not proof of novelty, and bank residuals do not prove global target matching. No automatic extension, checkpoint selection or reward tuning follows this bounded test.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return output / "report.md"
