"""Fresh plain-ProGen control and two-fit confirmation of final-block guidance.

Training and the inherited progression screen remain unchanged. The original
pilot is excluded: both new fits must pass, without pooling away a failed fit.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..artifacts import write_json
from .acceptance_experiment import verify_manifest
from .acceptance_reward import AcceptanceRewardConfig
from .charge import charge_reward
from .continuous_reward import continuous_reward
from .lastblock_evaluation import (
    ARMS,
    progression_decision,
    read_jsonl,
    summarize,
    validate_support,
    validate_training,
)
from .scoring import predict_disorder, score_records

CONFIRMATION_ARMS = ("prior", *ARMS)
SEED_FIELDS = ("training_seed", "evaluation_seed", "bank_sampling_seed")
SUMMARY_METRICS = (
    "joint_success_rate",
    "joint_success_without_common_training_copy",
    "charge_success_rate",
    "v3_disorder_success_rate",
    "v2_joint_success_rate",
    "unique_tail_fraction",
    "exact_common_training_copy_fraction",
    "mean_repeated_trimer_fraction",
)


def evaluate_prior(run, prior_run, output=None, predictor=predict_disorder):
    """Score a separately generated prior against its paired fit's copy union.

    The prior run already contains samples.jsonl and generation.json. This
    uses saved child training data and never loads ProGen or trains a model. Plain
    ProGen has no project-trained parameters, but shares the common-copy control.
    """
    run, prior_run = Path(run), Path(prior_run)
    output = prior_run if output is None else Path(output)
    verify_manifest(run)
    verify_manifest(prior_run)
    config = json.loads((run / "pilot_config.json").read_text())
    settings = json.loads((run / "config.json").read_text())
    prior_settings = json.loads((prior_run / "config.json").read_text())
    if (
        prior_settings["arm"] != "prior"
        or prior_settings["evaluation_seed"] != settings["evaluation_seed"]
        or prior_settings["fit_settings"] != settings
    ):
        raise ValueError("Plain-ProGen configuration differs from the paired fit")
    references = json.loads((run / "references.json").read_text())
    prompts = {row["id"]: row for row in references if row["role"] == "prompt"}
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    bank = json.loads((run / "bank.json").read_text())
    fresh = read_jsonl(run / "training_rows.jsonl")
    _, common_tails, _, training_summary = validate_training(
        run, settings, config, prompts, bank, fresh
    )
    records = read_jsonl(prior_run / "samples.jsonl")
    expected = {("prior", prompt): config["samples_per_prompt"] for prompt in prompts}
    if Counter((row["arm"], row["prompt_id"]) for row in records) != expected or len(
        {row["id"] for row in records}
    ) != len(records):
        raise ValueError("Plain-ProGen generation counts or identities differ")
    for row in records:
        validate_support(row, prompts, config, generated=True)
        if not np.isfinite(row["log_policy"]) or not np.isclose(
            row["log_policy"], row["log_prior"], atol=1e-6, rtol=0
        ):
            raise ValueError("Plain-ProGen policy must equal the original reference")
        if "log_z" in row or "own_reward_tb_residual" in row:
            raise ValueError("Plain-ProGen control has no fitted TB normalizer")
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
        tail = tails[index]
        row["acceptance"] = {
            key: value[index].tolist() for key, value in values.items()
        }
        row["old_charge_score"] = float(old_scores[index])
        row["seed_bank_copy"] = tail in seed_tails
        row["common_training_copy"] = tail in common_tails
        row["own_training_copy"] = False
        codes = np.frombuffer(tail.encode("ascii"), dtype=np.uint8)
        row["nearest_seed_bank_identity"] = float((seed_codes == codes).mean(1).max())
    metrics = dict(
        arms={"prior": summarize(scored)},
        by_prompt={
            f"prior/{prompt}": summarize(
                [row for row in scored if row["prompt_id"] == prompt]
            )
            for prompt in prompts
        },
    )
    write_json(output / "sequence_metrics.json", scored)
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "copy_definition.json",
        dict(
            common_training_unique_tails=training_summary[
                "common_training_unique_tails"
            ],
            common_copy="Same union of original train-bank and fresh training tails as the paired fit",
            own_copy="None: plain ProGen receives no project training",
            denominator="All generated outputs, including copies and failures",
        ),
    )
    row = metrics["arms"]["prior"]
    (output / "report.md").write_text(
        "# Fresh plain-ProGen control\n\n"
        f"{row['count']} raw outputs; joint charge/V3 success "
        f"{row['joint_success_rate']:.2%}; common-training-noncopy joint "
        f"{row['joint_success_without_common_training_copy']:.2%}.\n\n"
        "No guidance head, filtering, fitted normalizer or TB accuracy claim. "
        "Common-copy rates use the paired fit's training union, keeping all "
        "outputs in the denominator. V3 disorder is a prediction.\n"
    )
    return output / "report.md"


def confirmation_decision(replicates):
    """Require both independent fits to pass inherited and fresh-prior screens."""
    if len(replicates) != 2:
        raise ValueError("Confirmation requires exactly two fresh replicates")
    decisions = {}
    for name, metrics in replicates.items():
        arms = metrics["arms"]
        inherited = progression_decision(arms)
        candidate, prior = arms["lastblock_tb"], arms["prior"]
        plain = dict(
            joint_gain=candidate["joint_success_rate"] - prior["joint_success_rate"],
            common_noncopy_joint_gain=(
                candidate["joint_success_without_common_training_copy"]
                - prior["joint_success_without_common_training_copy"]
            ),
            unique_fraction_change=(
                candidate["unique_tail_fraction"] - prior["unique_tail_fraction"]
            ),
            disorder_success_change=(
                candidate["v3_disorder_success_rate"]
                - prior["v3_disorder_success_rate"]
            ),
        )
        plain["passes"] = (
            plain["joint_gain"] >= 0.05 - 1e-12
            and plain["common_noncopy_joint_gain"] >= 0.05 - 1e-12
            and plain["unique_fraction_change"] >= -0.05 - 1e-12
            and plain["disorder_success_change"] >= -0.05 - 1e-12
        )
        decisions[name] = dict(
            inherited=inherited,
            plain_control=plain,
            passes_confirmation_screen=(
                inherited["passes_progression_screen"] and plain["passes"]
            ),
        )
    return dict(
        replicates=decisions,
        passes_inherited_screen_all_replicates=all(
            row["inherited"]["passes_progression_screen"] for row in decisions.values()
        ),
        passes_plain_screen_all_replicates=all(
            row["plain_control"]["passes"] for row in decisions.values()
        ),
        passes_confirmation_screen=all(
            row["passes_confirmation_screen"] for row in decisions.values()
        ),
        interpretation=(
            "Both fresh fits must pass; the historical pilot is excluded. Practical "
            "development screens, not significance tests or biological validation. "
            "No automatic extra seeds, checkpoint selection or training extension."
        ),
    )


def summarize_confirmation(run, output, evaluations=None):
    """Combine all five policies per fit without pooling away seed variation."""
    run, output = Path(run), Path(output)
    evaluations = run if evaluations is None else Path(evaluations)
    settings = json.loads((run / "config.json").read_text())
    planned = settings["replicates"]
    names = [row["name"] for row in planned]
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError("Need two uniquely named fresh confirmation replicates")
    for field in SEED_FIELDS:
        if len({row[field] for row in planned}) != 2:
            raise ValueError(f"Confirmation replicates reuse {field}")
    metrics = {}
    prompt_ids = None
    for replicate in planned:
        name = replicate["name"]
        # These are direct child names, never arbitrary relative paths.
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid confirmation replicate name")
        fit, prior = run / name / "fit", run / name / "prior"
        verify_manifest(fit)
        verify_manifest(prior)
        child_settings = json.loads((fit / "config.json").read_text())
        if any(child_settings[field] != replicate[field] for field in SEED_FIELDS):
            raise ValueError("Child random streams differ from planned confirmation")
        child_config = json.loads((fit / "pilot_config.json").read_text())
        references = json.loads((fit / "references.json").read_text())
        prompts = [row["id"] for row in references if row["role"] == "prompt"]
        if prompt_ids is not None and prompts != prompt_ids:
            raise ValueError("Confirmation prompt panels differ")
        prompt_ids = prompts
        combined = dict(arms={}, by_prompt={})
        for folder, arms in (
            (evaluations / name / "fit", ARMS),
            (evaluations / name / "prior", ("prior",)),
        ):
            saved = json.loads((folder / "metrics.json").read_text())
            if set(saved["arms"]) != set(arms) or set(saved["by_prompt"]) != {
                f"{arm}/{prompt}" for arm in arms for prompt in prompts
            }:
                raise ValueError(
                    "Missing or extra confirmation evaluation arms/prompts"
                )
            for arm in arms:
                if saved["arms"][arm]["count"] != (
                    len(prompts) * child_config["samples_per_prompt"]
                ) or any(
                    saved["by_prompt"][f"{arm}/{prompt}"]["count"]
                    != child_config["samples_per_prompt"]
                    for prompt in prompts
                ):
                    raise ValueError("Confirmation evaluation budget differs")
            combined["arms"].update(saved["arms"])
            combined["by_prompt"].update(saved["by_prompt"])
        combined["random_streams"] = {field: replicate[field] for field in SEED_FIELDS}
        metrics[name] = combined
    decision = confirmation_decision(metrics)
    seed_summary = {}
    for arm in CONFIRMATION_ARMS:
        seed_summary[arm] = {}
        for field in SUMMARY_METRICS:
            values = [metrics[name]["arms"][arm][field] for name in names]
            seed_summary[arm][field] = dict(
                mean=float(np.mean(values)), minimum=min(values), maximum=max(values)
            )
    write_json(
        output / "metrics.json",
        dict(replicates=metrics, equal_weight_seed_summary=seed_summary),
    )
    write_json(output / "decision.json", decision)
    lines = [
        "# Final-block GFlowNet confirmation",
        "",
        "Two fresh paired fits, with fresh plain-ProGen and saved-head controls. "
        "All endpoints use final checkpoints and raw generated outputs. "
        "The historical pilot is excluded from the confirmation decision.",
        "",
        "| Replicate | Policy | Outputs | Joint | Noncopy joint | Charge | V3 disorder | Unique tails | Common copies | Repeated trimers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        for arm in CONFIRMATION_ARMS:
            row = metrics[name]["arms"][arm]
            lines.append(
                f"| {name} | {arm} | {row['count']} | "
                f"{row['joint_success_rate']:.2%} | "
                f"{row['joint_success_without_common_training_copy']:.2%} | "
                f"{row['charge_success_rate']:.2%} | "
                f"{row['v3_disorder_success_rate']:.2%} | "
                f"{row['unique_tail_fraction']:.2%} | "
                f"{row['exact_common_training_copy_fraction']:.2%} | "
                f"{row['mean_repeated_trimer_fraction']:.2%} |"
            )
    lines += [
        "",
        "## Paired differences and predefined screens",
        "",
        "| Replicate | Adapted minus control | Joint gain (pp) | Noncopy gain (pp) | Unique change (pp) | Disorder change (pp) | Screen |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name, result in decision["replicates"].items():
        comparisons = {
            "prior": result["plain_control"],
            **result["inherited"]["comparisons"],
            "continued_supervised": result["inherited"]["supervised_noninferiority"],
        }
        for arm, row in comparisons.items():
            lines.append(
                f"| {name} | {arm} | {100 * row['joint_gain']:+.2f} | "
                f"{100 * row['common_noncopy_joint_gain']:+.2f} | "
                f"{100 * row['unique_fraction_change']:+.2f} | "
                f"{100 * row['disorder_success_change']:+.2f} | "
                f"{'PASS' if row['passes'] else 'FAIL'} |"
            )
    lines += [
        "",
        f"Overall confirmation screen: **{'PASS' if decision['passes_confirmation_screen'] else 'FAIL'}**.",
        "",
        "Every fresh fit must pass the inherited screen: at least +5 percentage "
        "points in raw and common-training-noncopy joint success over unchanged "
        "and frozen TB, with at most 5-point losses in uniqueness/disorder. "
        "Against continued supervised, joint/noncopy losses must be at most "
        "2 points, uniqueness/disorder losses at most 5 points. The separate "
        "fresh-prior screen requires the same +5-point gains and 5-point quality "
        "limits. Neither seed means nor pooled output counts determine passing.",
        "",
        "## Joint success by prompt",
        "",
        "| Replicate | Prompt | Plain ProGen | Unchanged head | Continued supervised | Frozen TB | Last-block TB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        for prompt in prompt_ids:
            rates = " | ".join(
                f"{metrics[name]['by_prompt'][f'{arm}/{prompt}']['joint_success_rate']:.2%}"
                for arm in CONFIRMATION_ARMS
            )
            lines.append(f"| {name} | {prompt} | {rates} |")
    lines += [
        "",
        "metrics.json preserves every per-prompt endpoint, copy rate, diversity "
        "measure, V2 sensitivity result and each fit's actual random streams. "
        "Sequence records retain legacy local seed=0; replicate names and the "
        "three stream seeds identify the independent fits.",
        "",
        "Common-copy checks use each fit's own union of original bank training "
        "and fresh training tails. Noncopy joint success retains all outputs in "
        "the denominator. No project training is applied to plain ProGen; its "
        "own-training-copy flag is false while common-copy checks still apply.",
        "",
        "Within each fit, the frozen and adapted policies share refreshed "
        "training examples. The fits start from the same original saved head "
        "and original final block, with independent random streams. The prompt "
        "panel and charge target remain development settings, so this does not "
        "test unseen prompts or new properties. V3 disorder is an optimized "
        "prediction; V2 is sensitivity, not independent biological validation. "
        "Copies/uniqueness do not establish biological novelty. Common-bank "
        "residuals remain in each fit's bank_diagnostics.json and do not prove "
        "global target matching. A same-capacity non-GFN fine-tuning comparison "
        "has not been performed.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return output / "report.md"
