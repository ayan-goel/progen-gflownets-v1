"""CPU evaluation for the joint-acceptance reward comparison.

Raw generation is the primary endpoint. Selection and self-normalized prior-pool
target estimates are separate diagnostics, not samples from the trained policy.
"""

import json
import time
from collections import Counter

import numpy as np

from ..artifacts import write_json
from .acceptance_reward import AcceptanceRewardConfig, joint_acceptance_reward
from .charge import charge_reward
from .scoring import predict_disorder, score_records


def summarize_acceptance(rows):
    """No filtering; exact mean pairwise Hamming via residue counts, not pair loops."""
    tails = [r["sequence"][r["evaluation_start"] :] for r in rows]
    count, length = len(tails), len(tails[0])
    codes = np.frombuffer("".join(tails).encode("ascii"), dtype=np.uint8).reshape(
        count, length
    )
    matching_pairs = 0
    for column in codes.T:
        counts = np.bincount(column)
        matching_pairs += np.sum(counts * (counts - 1))
    result = {
        "count": count,
        "charge_success_rate": float(
            np.mean([r["acceptance"]["charge_pass"] for r in rows])
        ),
        "joint_success_rate": float(
            np.mean([r["acceptance"]["joint_pass"] for r in rows])
        ),
        "v3_disorder_success_rate": float(
            np.mean([r["acceptance"]["disorder_pass"] for r in rows])
        ),
        "v2_disorder_success_rate": float(
            np.mean([r["disorder"]["V2"]["isolated_tail"]["passes"] for r in rows])
        ),
        "v2_joint_success_rate": float(
            np.mean(
                [
                    r["acceptance"]["charge_pass"]
                    and r["disorder"]["V2"]["isolated_tail"]["passes"]
                    for r in rows
                ]
            )
        ),
        "unique_tail_fraction": len(set(tails)) / count,
        "mean_pairwise_tail_hamming_fraction": float(
            1 - matching_pairs / (count * (count - 1) * length)
        )
        if count > 1
        else None,
        "mean_repeated_trimer_fraction": float(
            np.mean([r["tail_descriptors"]["repeated_trimer_fraction"] for r in rows])
        ),
        "mean_tail_entropy_bits": float(
            np.mean([r["tail_descriptors"]["entropy_bits"] for r in rows])
        ),
        "mean_block_charge": np.mean(
            [r["acceptance"]["blocks"] for r in rows], axis=0
        ).tolist(),
        "mean_fcr": float(np.mean([r["acceptance"]["fcr"] for r in rows])),
        "mean_new_log_factor": float(
            np.mean([r["acceptance"]["log_factor"] for r in rows])
        ),
        "mean_old_charge_score": float(np.mean([r["old_charge_score"] for r in rows])),
        "mean_nearest_reference_tail_identity": float(
            np.mean([r["nearest_reference_tail_identity"] for r in rows])
        ),
        "exact_reference_tail_match_fraction": float(
            np.mean([r["exact_reference_tail_match"] for r in rows])
        ),
    }
    if all("log_prior" in r for r in rows):
        result["mean_log_prior_per_residue"] = float(
            np.mean([r["log_prior"] for r in rows]) / length
        )
    if all("own_reward_tb_residual" in r for r in rows):
        residuals = np.array([r["own_reward_tb_residual"] for r in rows])
        result["own_reward_tb_residual_rmse"] = float(np.sqrt(np.mean(residuals**2)))
        result["own_reward_tb_residual_mean"] = float(residuals.mean())
    return result


def prior_target_estimates(rows, beta):
    """Within-prompt SNIS; proposals already include p0, so do not multiply twice."""
    groups = sorted({(r["seed"], r["prompt_id"]) for r in rows})
    estimates = {"old_target": [], "new_target": []}
    for seed, prompt in groups:
        group = [r for r in rows if (r["seed"], r["prompt_id"]) == (seed, prompt)]
        for name in estimates:
            factors = np.array(
                [
                    r["acceptance"]["log_factor"]
                    if name == "new_target"
                    else beta * r["old_charge_score"]
                    for r in group
                ]
            )
            unnormalized = np.exp(factors - factors.max())
            weights = unnormalized / unnormalized.sum()
            estimates[name].append(
                {
                    "seed": seed,
                    "prompt_id": prompt,
                    "proposals": len(group),
                    "observed_joint_passes": sum(
                        r["acceptance"]["joint_pass"] for r in group
                    ),
                    "ess": float(1 / np.square(weights).sum()),
                    "max_weight": float(weights.max()),
                    "estimated_log_z": float(
                        factors.max() + np.log(unnormalized.mean())
                    ),
                    **{
                        key: float(
                            weights @ np.array([r["acceptance"][key] for r in group])
                        )
                        for key in ("joint_pass", "charge_pass", "disorder_pass")
                    },
                }
            )
    return {
        name: {
            "by_seed_prompt": group_rows,
            "summary": {
                **{
                    key: float(np.mean([r[key] for r in group_rows]))
                    for key in ("joint_pass", "charge_pass", "disorder_pass")
                },
                "median_ess": float(np.median([r["ess"] for r in group_rows])),
                "minimum_ess": min(r["ess"] for r in group_rows),
                "maximum_single_weight": max(r["max_weight"] for r in group_rows),
                "groups_without_observed_joint_pass": sum(
                    r["observed_joint_passes"] == 0 for r in group_rows
                ),
            },
        }
        for name, group_rows in estimates.items()
    }


def training_diagnostics(run, config):
    """Summarize encounter frequency and fixed windows, never select checkpoints."""
    summaries = {}
    for seed in config["seeds"]:
        folder = run / f"seed-{seed}"
        logs = [
            json.loads(line)
            for line in (folder / "training.jsonl").read_text().splitlines()
        ]
        inputs = [
            json.loads(line)
            for line in (folder / "reward_inputs.jsonl").read_text().splitlines()
        ]
        accepted = [r for r in inputs if r["joint_pass"]]
        windows = []
        for start in range(0, len(logs), 100):
            window = logs[start : start + 100]
            windows.append(
                {
                    "first_step": window[0]["step"],
                    "last_step": window[-1]["step"],
                    **{
                        key: float(np.mean([r[key] for r in window]))
                        for key in (
                            "tb_loss",
                            "joint_success",
                            "charge_success",
                            "gradient_norm_before_clip",
                        )
                    },
                }
            )
        summaries[str(seed)] = {
            "trajectories": len(inputs),
            "joint_encounters": len(accepted),
            "unique_joint_tails": len({r["tail"] for r in accepted}),
            "joint_encounters_by_prompt": {
                prompt: sum(r["prompt_id"] == prompt for r in accepted)
                for prompt in sorted({r["prompt_id"] for r in inputs})
            },
            "fraction_batches_without_joint_pass": float(
                np.mean([r["joint_count"] == 0 for r in logs])
            ),
            "fraction_updates_clipped": float(
                np.mean(
                    [
                        r["gradient_norm_before_clip"] > config["gradient_clip"]
                        for r in logs
                    ]
                )
            ),
            "windows": windows,
        }
    return summaries


def evaluate_acceptance(run, output, predictor=predict_disorder):
    """Evaluate raw methods identically and separate V3 optimization from V2 checks."""
    started = time.monotonic()
    config = json.loads((run / "pilot_config.json").read_text())
    source_config = json.loads((run / "source_config.json").read_text())
    settings = json.loads((run / "config.json").read_text())
    reward = AcceptanceRewardConfig(**settings["reward"])
    references = json.loads((run / "references.json").read_text())
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    prompts = {r["id"]: r for r in references if r["role"] == "prompt"}
    counts = Counter((r["arm"], r["seed"], r["prompt_id"]) for r in records)
    expected = {
        (arm, seed, prompt): config["samples_per_prompt"]
        * (config["selection_factor"] if arm == "prior_pool" else 1)
        for arm in ("new_tb", "old_tb", "prior_pool")
        for seed in config["seeds"]
        for prompt in prompts
    }
    if counts != expected or len({r["id"] for r in records}) != len(records):
        raise ValueError("Invalid sample identities/counts")
    for row in records:
        prefix = prompts[row["prompt_id"]]["sequence"][: config["prefix_length"]]
        if (
            row["prefix"] != prefix
            or not row["sequence"].startswith(prefix)
            or len(row["sequence"]) != config["length"]
            or row["evaluation_start"] != config["prefix_length"]
        ):
            raise ValueError("Sample support/prompt differs from protocol")
    scored = score_records(records, references, config, predictor)
    tails = [r["sequence"][config["prefix_length"] :] for r in scored]
    predictions = np.array(
        [r["disorder"]["V3"]["isolated_tail_scores"] for r in scored]
    )
    values = joint_acceptance_reward(tails, predictions, reward)
    old_scores, _, _, old_pass = charge_reward(tails, source_config["reward"])
    if not np.array_equal(old_pass.numpy(), values["charge_pass"].numpy()):
        raise ValueError("Acceptance and historical charge criteria differ")
    for index, row in enumerate(scored):
        row["acceptance"] = {
            key: value[index].tolist() for key, value in values.items()
        }
        row["old_charge_score"] = float(old_scores[index])
        if row["arm"] in ("old_tb", "new_tb"):
            log_factor = (
                source_config["beta"] * row["old_charge_score"]
                if row["arm"] == "old_tb"
                else row["acceptance"]["log_factor"]
            )
            row["own_reward_tb_residual"] = (
                row["log_z"] + row["log_policy"] - row["log_prior"] - log_factor
            )
    pool = [r for r in scored if r["arm"] == "prior_pool"]
    targets = prior_target_estimates(pool, source_config["beta"])
    for seed in config["seeds"]:
        for prompt in prompts:
            group = [r for r in pool if r["seed"] == seed and r["prompt_id"] == prompt]
            # Ties preserve generation order; no extra center preference or deduplication.
            selected = sorted(group, key=lambda r: -r["acceptance"]["log_factor"])[
                : config["samples_per_prompt"]
            ]
            for arm, subset in (
                ("prior", group[: config["samples_per_prompt"]]),
                ("select", selected),
            ):
                for row in subset:
                    derived = dict(
                        row, arm=arm, source_id=row["id"], id=f"{arm}-{row['id']}"
                    )
                    if arm == "select":
                        derived.pop(
                            "log_policy"
                        )  # Proposal density is not selection density.
                    scored.append(derived)
    grouped = {"arms": {}, "by_seed": {}, "by_prompt": {}, "by_seed_prompt": {}}
    for row in scored:
        keys = (
            row["arm"],
            f"{row['arm']}/seed-{row['seed']}",
            f"{row['arm']}/{row['prompt_id']}",
            f"{row['arm']}/seed-{row['seed']}/{row['prompt_id']}",
        )
        for category, key in zip(grouped.values(), keys):
            category.setdefault(key, []).append(row)
    metrics = {
        name: {key: summarize_acceptance(group) for key, group in groups.items()}
        for name, groups in grouped.items()
    }
    old, new = [metrics["arms"][arm] for arm in ("old_tb", "new_tb")]
    delta = new["joint_success_rate"] - old["joint_success_rate"]
    unique_delta = new["unique_tail_fraction"] - old["unique_tail_fraction"]
    improved_seeds = sum(
        metrics["by_seed"][f"new_tb/seed-{seed}"]["joint_success_rate"]
        > metrics["by_seed"][f"old_tb/seed-{seed}"]["joint_success_rate"]
        for seed in config["seeds"]
    )
    decision = {
        "joint_gain": delta,
        "seeds_improved": improved_seeds,
        "unique_fraction_change": unique_delta,
        "required_joint_gain": 0.05,
        "required_seeds_improved": 2,
        "maximum_unique_fraction_drop": 0.05,
        "passes_practical_screen": bool(
            delta >= 0.05 and improved_seeds >= 2 and unique_delta >= -0.05
        ),
        "interpretation": "Engineering screen, not a significance test or biological validation; no automatic extension or parameter sweep.",
    }
    write_json(output / "sequence_metrics.json", scored)
    write_json(output / "metrics.json", metrics)
    write_json(output / "target_estimates.json", targets)
    write_json(output / "decision.json", decision)
    write_json(output / "training_diagnostics.json", training_diagnostics(run, config))
    write_json(
        output / "evaluation_cost.json",
        {
            "seconds": time.monotonic() - started,
            "raw_records": len(records),
            "derived_records": len(scored) - len(records),
            "unique_scorer_inputs_per_version": len(
                {
                    s
                    for r in records
                    for s in (r["sequence"], r["sequence"][config["prefix_length"] :])
                }
            ),
        },
    )
    lines = [
        "# Joint-acceptance reward comparison",
        "",
        "Three new TB fits; old TB checkpoints reused with fresh samples. Same frozen ProGen, prompts, support and training budget. Final checkpoints only.",
        "",
        "| Method | n | Charge | Joint charge/V3 | V3 disorder | V2 joint sensitivity | Unique tails | Repeated trimers |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("prior", "old_tb", "new_tb", "select", "prior_pool"):
        row = metrics["arms"][arm]
        lines.append(
            f"| {arm} | {row['count']} | {row['charge_success_rate']:.1%} | {row['joint_success_rate']:.1%} | {row['v3_disorder_success_rate']:.1%} | {row['v2_joint_success_rate']:.1%} | {row['unique_tail_fraction']:.1%} | {row['mean_repeated_trimer_fraction']:.1%} |"
        )
    lines += [
        "",
        "Selection keeps the best quarter of each prior pool by the NEW property factor, with stable ties. It costs four proposals per output. Raw policy metrics have no filtering.",
        "",
        "## Joint success by seed",
        "",
        "| Seed | ProGen | Old TB | New TB |",
        "|---|---:|---:|---:|",
    ]
    for seed in config["seeds"]:
        rates = " | ".join(
            f"{metrics['by_seed'][f'{arm}/seed-{seed}']['joint_success_rate']:.1%}"
            for arm in ("prior", "old_tb", "new_tb")
        )
        lines.append(f"| {seed} | {rates} |")
    lines += [
        "",
        "## Joint success by prompt",
        "",
        "| Prompt | ProGen | Old TB | New TB |",
        "|---|---:|---:|---:|",
    ]
    for prompt in prompts:
        rates = " | ".join(
            f"{metrics['by_prompt'][f'{arm}/{prompt}']['joint_success_rate']:.1%}"
            for arm in ("prior", "old_tb", "new_tb")
        )
        lines.append(f"| {prompt} | {rates} |")
    lines += [
        "",
        "## Prior-pool target estimates",
        "",
        "| Target | Estimated joint | Median ESS | Minimum ESS | Groups without observed passes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, target in targets.items():
        row = target["summary"]
        lines.append(
            f"| {name} | {row['joint_pass']:.1%} | {row['median_ess']:.1f} | {row['minimum_ess']:.1f} | {row['groups_without_observed_joint_pass']} |"
        )
    lines += [
        "",
        "Importance estimates use fresh prior proposals within each prompt/seed. They are not exact target rates, trained-policy forecasts or evidence for unseen modes.",
        "",
        "## Decision and interpretation",
        "",
        f"New minus old raw joint success: {delta:+.1%}; improved seeds: {improved_seeds}/{len(config['seeds'])}; unique-tail change: {unique_delta:+.1%}.",
        f"Practical screen passed: **{decision['passes_practical_screen']}** (>=5 percentage points joint gain, >=2 improved seeds, <=5-point unique-tail loss). This is not a significance test.",
        "",
        "V3 is optimized by the new reward; V2 is related-model sensitivity, not independent biological validation. Reused training seeds/prompts form a development panel. Small differences should not trigger an automatic rate/reward sweep.",
        "",
        "metrics.json includes each policy's TB residual under its OWN reward. Different residual scales do not rank methods or establish exact distribution matching. training_diagnostics.json records rare-pass encounters, clipping and fixed training windows.",
        "",
        "costs.json separates new training (including CPU V3 scoring) and fresh generation. source_costs.json records historical costs; old training was reused, not repeated. evaluation_cost.json accounts for final CPU scoring.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return output / "report.md"
