"""Three-repeat success-bonus comparison with unchanged acceptance endpoints.

Historical objective modules are pinned by completed runs. Reuse their pure
summaries and training checks, while keeping this version's reward, repeat count
and decision screen explicit here. All copy-adjusted rates retain raw denominators.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..artifacts import write_json
from .acceptance_reward import AcceptanceRewardConfig
from .charge import charge_reward
from .lastblock_evaluation import read_jsonl, validate_support
from .objective_evaluation import (
    ARMS,
    EFFECT_METRICS,
    TRAINED_ARMS,
    _validate_training,
    objective_effects,
    summarize_objective,
)
from .scoring import predict_disorder, score_records
from .success_bonus_reward import REWARD_VERSION, success_bonus_reward


def success_bonus_decision(replicates):
    """Require substantial success gain and useful sequence yield in every repeat.

    The count comparison uses equal raw output budgets. A favorable mean cannot
    rescue a failed repeat; neither can noncopy rate or a post-hoc cluster metric.
    """
    if len(replicates) != 3:
        raise ValueError(
            "Success-bonus comparison requires exactly three paired repeats"
        )
    decisions = {}
    for name, arms in replicates.items():
        gfn, supervised = arms["gfn"], arms["supervised"]
        if gfn["count"] <= 0 or gfn["count"] != supervised["count"]:
            raise ValueError(
                "Practical screen requires equal positive raw output budgets"
            )
        effects = objective_effects(arms)
        extra_distinct = (
            gfn["distinct_noncopy_successful_tail_count"]
            - supervised["distinct_noncopy_successful_tail_count"]
        )
        decisions[name] = dict(
            gfn_minus_supervised=effects,
            distinct_noncopy_successful_tail_count_gain=extra_distinct,
            passes=(
                effects["joint_success_rate"] >= 0.10 - 1e-12
                and extra_distinct > 0
                and effects["unique_tail_fraction"] >= -0.05 - 1e-12
                and effects["v3_disorder_success_rate"] >= -0.05 - 1e-12
            ),
        )
    return dict(
        replicates=decisions,
        passes_practical_screen=all(row["passes"] for row in decisions.values()),
        required_joint_gain=0.10,
        requires_strictly_more_distinct_successful_noncopy_tails=True,
        maximum_unique_fraction_loss=0.05,
        maximum_disorder_success_loss=0.05,
        interpretation=(
            "All three fresh repeats must pass this practical development screen. "
            "Three seeds on development prompts do not establish formal superiority. "
            "Failure does not establish supervised superiority or equivalence. "
            "Noncopy rates, cluster yields and costs remain descriptive; they do "
            "not replace the declared success, distinct-yield and quality criteria."
        ),
    )


def _validate_reward_definition(run, fit):
    """Reject unversioned rewards or an accidental return to the old weighting."""
    expected = dict(version=REWARD_VERSION, penalty=2.5, pass_multiplier=9.0)
    definition = json.loads((run / "reward_definition.json").read_text())
    if definition != expected or any(
        fit.get(key) != expected[key] for key in ("penalty", "pass_multiplier")
    ):
        raise ValueError(
            "Success-bonus reward definition differs from the fixed protocol"
        )
    return expected


def _validate_saved_reward(rows, reward, fit, label):
    """Recompute detached rewards and acceptance labels from saved scorer arrays."""
    if not rows:
        raise ValueError(f"Missing {label} rewards")
    required = (
        "base_log_factor",
        "success_log_bonus",
        "log_factor",
        "joint_pass",
        "isolated_tail_v3_scores",
        "reward_version",
    )
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError(f"Missing versioned {label} reward fields")
    if any(row["reward_version"] != REWARD_VERSION for row in rows):
        raise ValueError(f"Incorrect {label} reward version")
    values = success_bonus_reward(
        [row["tail"] for row in rows],
        [row["isolated_tail_v3_scores"] for row in rows],
        reward,
        fit["penalty"],
        fit["pass_multiplier"],
    )
    for key in ("base_log_factor", "success_log_bonus", "log_factor"):
        if not np.allclose(
            [row[key] for row in rows], values[key].numpy(), atol=1e-6, rtol=1e-6
        ):
            raise ValueError(f"Saved {label} {key} differs from the fixed reward")
    for index, row in enumerate(rows):
        if type(row["joint_pass"]) is not bool or row["joint_pass"] != bool(
            values["joint_pass"][index]
        ):
            raise ValueError(f"Saved {label} acceptance labels differ")
        if "violations" in row and not np.allclose(
            row["violations"], values["violations"][index].numpy(), atol=1e-6
        ):
            raise ValueError(f"Saved {label} violations differ")


def _validate_checkpoints(folder, prompts, fit, definition):
    """Inspect final saved CPU tensors and metadata without constructing a model."""
    gfn_normalizers = None
    for arm in TRAINED_ARMS:
        checkpoint = torch.load(
            folder / arm / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        expected = dict(
            architecture="lastblock_direct_context",
            arm=arm,
            step=fit["training_steps"],
            prompt_ids=list(prompts),
            objective="trajectory_balance" if arm == "gfn" else "weighted_token_nll",
            reward_version=REWARD_VERSION,
            reward_definition=definition,
        )
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Final checkpoint metadata differs from the reward/protocol"
            )
        for group in ("head", "last_block"):
            weights = checkpoint.get(group)
            if (
                not isinstance(weights, dict)
                or not weights
                or any(
                    not isinstance(value, torch.Tensor)
                    or not torch.isfinite(value).all()
                    for value in weights.values()
                )
            ):
                raise ValueError("Missing or nonfinite final checkpoint weights")
        normalizer = checkpoint.get("log_z")
        if arm == "supervised":
            if normalizer is not None:
                raise ValueError("Only the GFN may have a fitted normalizer")
        elif (
            not isinstance(normalizer, torch.Tensor)
            or normalizer.shape != (len(prompts),)
            or not torch.isfinite(normalizer).all()
        ):
            raise ValueError("Invalid final GFN normalizer")
        else:
            gfn_normalizers = dict(zip(prompts, normalizer.tolist()))
    return gfn_normalizers


def evaluate_success_bonus(run, output, predictor=predict_disorder):
    """Evaluate saved final samples with an injectable CPU/cache-only predictor.

    The caller owns provenance/manifest verification and recovery. This function
    inspects saved checkpoint tensors on CPU without constructing a model and
    does not mutate the source run when output differs.
    """
    run, output = Path(run), Path(output)
    settings = json.loads((run / "config.json").read_text())
    fit = settings["fit"]
    definition = _validate_reward_definition(run, fit)
    config = json.loads((run / "pilot_config.json").read_text())
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    references = json.loads((run / "references.json").read_text())
    prompt_rows = [row for row in references if row["role"] == "prompt"]
    prompts = {row["id"]: row for row in prompt_rows}
    if not prompts or len(prompts) != len(prompt_rows):
        raise ValueError("Missing or duplicated prompt identities")
    if fit["sft_weighting"] != "property_factor_per_prompt":
        raise ValueError("Supervised objective differs from the declared weighting")
    if reward.tail_length != config["length"] - config["prefix_length"]:
        raise ValueError("Reward and generation tail lengths differ")
    if any(
        getattr(reward, field) != config[field]
        for field in ("disorder_cutoff", "minimum_disordered_fraction")
    ):
        raise ValueError("Reward and evaluation disorder thresholds differ")
    if any(
        not np.allclose(getattr(reward, field), config["reward"][field])
        for field in ("block_targets", "block_tolerance", "fcr_target", "fcr_tolerance")
    ):
        raise ValueError("Reward and evaluation charge thresholds differ")
    if (
        fit.get("evaluation_samples_per_prompt", config["samples_per_prompt"])
        != config["samples_per_prompt"]
    ):
        raise ValueError("Fit and pilot evaluation budgets differ")
    planned = settings["replicates"]
    names = [row["name"] for row in planned]
    if (
        len(names) != 3
        or len(set(names)) != 3
        or any(Path(name).name != name or name in (".", "..") for name in names)
    ):
        raise ValueError("Need exactly three distinct, safe replicate names")
    for field in ("training_seed", "evaluation_seed", "bank_sampling_seed"):
        if len({row[field] for row in planned}) != len(planned):
            raise ValueError(f"Replicates must use distinct {field} streams")
    bank = json.loads((run / "bank.json").read_text())
    bank_by_id = {row["id"]: row for row in bank}
    if not bank or len(bank_by_id) != len(bank):
        raise ValueError("Missing or duplicated bank identities")
    for row in bank:
        validate_support(row, prompts, config)
        if row["split"] not in ("train", "holdout"):
            raise ValueError("Unknown bank split")
        if not np.isfinite(row["log_factor"]):
            raise ValueError("Nonfinite bank property factor")
    _validate_saved_reward(bank, reward, fit, "bank")
    initial_ids = {
        identity
        for row in read_jsonl(run / "initial_training.jsonl")
        for identity in row["replay_ids"]
    }
    if not initial_ids or any(
        identity not in bank_by_id or bank_by_id[identity]["split"] != "train"
        for identity in initial_ids
    ):
        raise ValueError("Initial shared supervised fit contains invalid replay")
    seed_tails = {row["tail"] for row in bank if row["split"] == "train"}
    common_tails = set(seed_tails)
    training_summary = dict(replicates={})
    own_tails = {}
    records = []
    seen_training_ids = set(bank_by_id)
    costs = {}
    for replicate in planned:
        name = replicate["name"]
        folder = run / name
        current = json.loads((folder / "config.json").read_text())
        expected_fit = {
            **fit,
            **{key: value for key, value in replicate.items() if key != "name"},
        }
        if any(current.get(key) != value for key, value in expected_fit.items()):
            raise ValueError(
                "Replicate fit settings differ from the declared comparison"
            )
        gfn_normalizers = _validate_checkpoints(folder, prompts, fit, definition)
        fresh = read_jsonl(folder / "training_rows.jsonl")
        for row in fresh:
            validate_support(row, prompts, config)
            if row["id"] in seen_training_ids or row.get("replicate", name) != name:
                raise ValueError("Fresh training identities overlap across repeats")
            seen_training_ids.add(row["id"])
            if row["generator_arm"] not in ARMS or row.get("split", "train") != "train":
                raise ValueError("Unknown fresh training source or split")
        own_tails[name], summary = _validate_training(
            folder, current, config, prompts, bank, fresh, initial_ids
        )
        _validate_saved_reward(fresh, reward, fit, "fresh training")
        training_summary["replicates"][name] = summary
        common_tails.update(row["tail"] for row in fresh)
        samples = read_jsonl(folder / "samples.jsonl")
        expected_counts = {
            (arm, prompt): config["samples_per_prompt"]
            for arm in ARMS
            for prompt in prompts
        }
        if (
            Counter((row["arm"], row["prompt_id"]) for row in samples)
            != expected_counts
        ):
            raise ValueError("Fresh final generation counts differ from the protocol")
        for row in samples:
            validate_support(row, prompts, config, generated=True)
            if row.get("replicate", name) != name or not np.isfinite(row["log_policy"]):
                raise ValueError("Invalid evaluation replicate or policy likelihood")
            if row["arm"] == "prior" and not np.isclose(
                row["log_policy"], row["log_prior"], atol=1e-6, rtol=0
            ):
                raise ValueError("Plain ProGen policy must equal the frozen reference")
            if row["arm"] != "gfn" and row.get("log_z") is not None:
                raise ValueError("Only the GFN has a fitted TB normalizer")
            if row.get("log_z") is not None and not np.isfinite(row["log_z"]):
                raise ValueError("Nonfinite fitted TB normalizer")
            if row["arm"] == "gfn" and (
                row.get("log_z") is None
                or not np.isclose(
                    row["log_z"],
                    gfn_normalizers[row["prompt_id"]],
                    atol=1e-6,
                    rtol=1e-6,
                )
            ):
                raise ValueError(
                    "Evaluation GFN normalizer differs from final checkpoint"
                )
            row["replicate"] = name
            row["tail"] = row["sequence"][config["prefix_length"] :]
        records.extend(samples)
        for filename, destination in (
            ("costs.json", costs),
            ("adaptation.json", summary),
        ):
            path = folder / filename
            if path.exists():
                destination[name if filename == "costs.json" else "adaptation"] = (
                    json.loads(path.read_text())
                )
        if "adaptation" in summary:
            adaptation = summary["adaptation"]
            if adaptation["reference_has_gradients"]:
                raise ValueError("Original frozen reference acquired gradients")
            for arm in TRAINED_ARMS:
                if any(
                    not np.isfinite(value) or value < 0
                    for value in adaptation[arm].values()
                ):
                    raise ValueError("Invalid final-block adaptation diagnostics")
        if name in costs:
            measured = costs[name]
            for key in ("updates_per_arm", "examples_per_arm", "fresh_trajectories"):
                if measured["training"][key] != summary[key]:
                    raise ValueError(
                        "Recorded training costs disagree with saved draws"
                    )
            for arm in ARMS:
                if (
                    measured["generation"][arm]["trajectories"]
                    != len(prompts) * config["samples_per_prompt"]
                ):
                    raise ValueError("Recorded generation costs disagree with outputs")
    if len({row["id"] for row in records}) != len(records) or any(
        row["id"] in seen_training_ids for row in records
    ):
        raise ValueError(
            "Evaluation identities are duplicated or reuse training records"
        )
    if (run / "samples.jsonl").exists() and read_jsonl(
        run / "samples.jsonl"
    ) != records:
        raise ValueError("Combined evaluation records differ from the three repeats")
    scoring_cost = dict(seconds=0.0, calls=0, sequence_inputs=0, by_version={})

    def measured_predictor(sequences, version):
        started = time.monotonic()
        values = predictor(sequences, version)
        elapsed = time.monotonic() - started
        scoring_cost["seconds"] += elapsed
        scoring_cost["calls"] += 1
        scoring_cost["sequence_inputs"] += len(sequences)
        scoring_cost["by_version"][version] = dict(
            seconds=elapsed, sequence_inputs=len(sequences)
        )
        return values

    scored = score_records(records, references, config, measured_predictor)
    tails = [row["tail"] for row in scored]
    values = success_bonus_reward(
        tails,
        [row["disorder"]["V3"]["isolated_tail_scores"] for row in scored],
        reward,
        fit["penalty"],
        fit["pass_multiplier"],
    )
    old_scores = charge_reward(tails, config["reward"])[0]
    seed_codes = np.frombuffer(
        "".join(sorted(seed_tails)).encode("ascii"), dtype=np.uint8
    ).reshape(len(seed_tails), reward.tail_length)
    for index, row in enumerate(scored):
        tail = row["tail"]
        row["acceptance"] = {
            key: value[index].tolist() for key, value in values.items()
        }
        row["old_charge_score"] = float(old_scores[index])
        row["seed_bank_copy"] = tail in seed_tails
        row["common_training_copy"] = tail in common_tails
        row["own_training_copy"] = (
            row["arm"] in TRAINED_ARMS and tail in own_tails[row["replicate"]]
        )
        codes = np.frombuffer(tail.encode("ascii"), dtype=np.uint8)
        row["nearest_seed_bank_identity"] = float((seed_codes == codes).mean(1).max())
        if row["arm"] == "gfn" and row.get("log_z") is not None:
            row["own_reward_tb_residual"] = (
                row["log_z"]
                + row["log_policy"]
                - row["log_prior"]
                - row["acceptance"]["log_factor"]
            )
    metrics = dict(
        arms={
            arm: summarize_objective([row for row in scored if row["arm"] == arm])
            for arm in ARMS
        },
        by_replicate={
            name: {
                arm: summarize_objective(
                    [
                        row
                        for row in scored
                        if row["replicate"] == name and row["arm"] == arm
                    ]
                )
                for arm in ARMS
            }
            for name in names
        },
        by_prompt={
            f"{arm}/{prompt}": summarize_objective(
                [
                    row
                    for row in scored
                    if row["arm"] == arm and row["prompt_id"] == prompt
                ]
            )
            for arm in ARMS
            for prompt in prompts
        },
        by_replicate_prompt={
            f"{name}/{arm}/{prompt}": summarize_objective(
                [
                    row
                    for row in scored
                    if row["replicate"] == name
                    and row["arm"] == arm
                    and row["prompt_id"] == prompt
                ]
            )
            for name in names
            for arm in ARMS
            for prompt in prompts
        },
    )
    effects = {
        name: objective_effects(arms) for name, arms in metrics["by_replicate"].items()
    }
    prompt_effects = {
        f"{name}/{prompt}": objective_effects(
            {
                arm: metrics["by_replicate_prompt"][f"{name}/{arm}/{prompt}"]
                for arm in ARMS
            }
        )
        for name in names
        for prompt in prompts
    }
    metrics["paired_effects"] = dict(
        per_replicate=effects,
        per_replicate_prompt=prompt_effects,
        per_prompt={
            prompt: {
                metric: float(
                    np.mean(
                        [prompt_effects[f"{name}/{prompt}"][metric] for name in names]
                    )
                )
                for metric in EFFECT_METRICS
            }
            for prompt in prompts
        },
        mean_paired_gfn_minus_supervised={
            metric: float(np.mean([row[metric] for row in effects.values()]))
            for metric in EFFECT_METRICS
        },
    )
    decision = success_bonus_decision(metrics["by_replicate"])
    training_summary.update(
        reward_definition=definition,
        common_training_unique_tails=len(common_tails),
        initial_supervised_unique_used_ids=len(initial_ids),
        copy_definition=(
            "Every arm and repeat uses one conservative union of the initial "
            "training-bank tails and all fresh training tails from ALL THREE repeats. "
            "Own-training copies use actual warmup plus same-repeat consumed "
            "examples; prior has none. Historical adapted 400/800/1600 fits are "
            "not exposures of these fresh fits and are not included."
        ),
        denominator="All raw outputs, including copies and failures",
        cluster_definition=(
            "Connected components within each prompt at >=80% equal-length "
            "positionwise tail identity; pooled metrics combine repeats within "
            "prompt. Success/noncopy clusters are recomputed on their respective "
            "subsets but divided by ALL raw outputs. Components may contain "
            "dissimilar pairs through transitive links."
        ),
        costs=dict(
            parent=json.loads((run / "costs.json").read_text())
            if (run / "costs.json").exists()
            else None,
            replicates=costs,
            evaluation_predictor_counts=dict(
                calls=scoring_cost["calls"],
                sequence_inputs=scoring_cost["sequence_inputs"],
                by_version={
                    version: row["sequence_inputs"]
                    for version, row in scoring_cost["by_version"].items()
                },
            ),
            interpretation=(
                "Reported measurements retain setup, shared refresh generation/"
                "scoring, paired training updates, and final generation/evaluation "
                "stages. Shared collection/scoring costs are not charged twice or "
                "arbitrarily allocated. Per-arm update times, when recorded, do "
                "not represent independent end-to-end pipeline costs. Inherited "
                "initialization/search costs may not be measured by this run."
                " GFN update timings include the original-reference likelihood "
                "audit forwards used to verify shared labels for both arms; "
                "this is implementation overhead, not intrinsic objective cost."
            ),
        ),
        interpretation=(
            "Both adaptable final-block-plus-head models receive identical "
            "ordered batches and the same fixed scorer/reward. The GFN alone "
            "has a TB normalizer. Data come from both current models, the frozen "
            "prior and replay, so this is a coupled objective comparison."
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (
        ("sequence_metrics.json", scored),
        ("metrics.json", metrics),
        ("decision.json", decision),
        ("training_summary.json", training_summary),
        ("scoring_cost.json", scoring_cost),
    ):
        write_json(output / filename, value)
    report = [
        "# Ninefold success bonus: matched GFN versus supervised adaptation",
        "",
        "Three fresh paired repeats adapt the same final block and guidance head from the original shared supervised initialization, using identical refreshed examples and the same ninefold joint-success reward bonus. Charge/disorder pass thresholds are unchanged.",
        "",
        "The earlier finite-pool target estimate motivated this intervention; it is not a forecast of trained success. Historical no-bonus results are context, not a paired causal reward comparison.",
        "",
        "| Repeat | Policy | Raw outputs | Joint | Noncopy joint | Unique tails | V3 disorder | Distinct successful noncopy tails | Noncopy successful clusters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arms in metrics["by_replicate"].items():
        for arm, row in arms.items():
            report.append(
                f"| {name} | {arm} | {row['count']} | {row['joint_success_rate']:.2%} | "
                f"{row['joint_success_without_common_training_copy']:.2%} | "
                f"{row['unique_tail_fraction']:.2%} | {row['v3_disorder_success_rate']:.2%} | "
                f"{row['distinct_noncopy_successful_tail_count']} | "
                f"{row['near_identity_clusters']['noncopy_successful']['count']} |"
            )
    if costs:
        report.extend(
            [
                "",
                "Measured stage times in seconds (shared refresh includes generation and scoring):",
                "",
                "| Repeat | Shared refresh | Paired updates | GFN updates | Supervised updates | GFN final generation | Supervised final generation | Prior final generation |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, measured in costs.items():
            updates = training_summary["replicates"][name][
                "measured_update_seconds_by_arm"
            ]
            values = [
                measured["training"].get("refresh_seconds"),
                measured["training"].get("paired_update_seconds"),
                updates.get("gfn"),
                updates.get("supervised"),
                *[measured["generation"][arm].get("seconds") for arm in ARMS],
            ]
            cells = [
                "unrecorded" if value is None else f"{value:.2f}" for value in values
            ]
            report.append(f"| {name} | " + " | ".join(cells) + " |")
        parent_cost = training_summary["costs"]["parent"] or {}
        if "setup_seconds" in parent_cost:
            report.extend(
                [
                    "",
                    f"Common setup: {parent_cost['setup_seconds']:.2f} seconds. Current evaluation scorer timing is recorded separately in scoring_cost.json.",
                ]
            )
    mean_effect = metrics["paired_effects"]["mean_paired_gfn_minus_supervised"]
    report.extend(
        [
            "",
            f"Mean paired GFN minus supervised: **{100 * mean_effect['joint_success_rate']:+.2f} percentage points** raw joint and **{100 * mean_effect['joint_success_without_common_training_copy']:+.2f} points** common-training-noncopy joint.",
            "",
            f"Practical screen: **{'PASS' if decision['passes_practical_screen'] else 'FAIL'}**. All three repeats must gain at least ten points in raw joint success and produce strictly more distinct successful noncopy tails, while losing no more than five points in uniqueness or V3 disorder. Means cannot rescue a failed repeat.",
            decision["interpretation"],
            "",
            training_summary["copy_definition"],
            "All rates and successful-tail/cluster yields retain raw outputs in their denominator; exact noncopy does not establish biological novelty.",
            training_summary["cluster_definition"],
            "",
            "Distinct successful noncopy tail count is part of the declared screen. Noncopy success rate and cluster yields remain descriptive; they cannot replace a failed raw-success or quality criterion. Per-repeat and per-prompt effects are retained in metrics.json.",
            training_summary["interpretation"],
            training_summary["costs"]["interpretation"],
            "Detailed measured stages, weighting concentration and gradient diagnostics are retained in training_summary.json and the original costs.json files.",
            "",
            "V3 disorder is the optimized prediction; V2 is a sensitivity check, not independent biological validation. This development comparison does not establish generalization to new prompts/properties, biological function, or global target-distribution accuracy. No checkpoint selection or automatic training extension follows.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output / "report.md"
