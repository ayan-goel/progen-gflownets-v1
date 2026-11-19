"""Fresh IDR generation and common-data diagnostics for final-block adaptation.

The primary endpoint is the fraction of raw samples meeting the charge and V3
disorder criteria. Finite-bank residuals cannot establish global target matching.
"""

import json
from collections import Counter

import numpy as np

from ..artifacts import write_json
from .acceptance_reward import AcceptanceRewardConfig
from .charge import charge_reward
from .continuous_reward import continuous_reward
from .data import ALPHABET
from .replay_evaluation import summarize_outputs
from .scoring import predict_disorder, score_records

ARMS = ("unchanged", "continued_supervised", "frozen_tb", "lastblock_tb")
TRAINED_ARMS = ARMS[2:]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def progression_decision(arms):
    """A bounded development screen, with the strong supervised control explicit."""
    candidate = arms["lastblock_tb"]
    comparisons = {}
    for name in ("unchanged", "frozen_tb"):
        reference = arms[name]
        gain = candidate["joint_success_rate"] - reference["joint_success_rate"]
        noncopy_gain = (
            candidate["joint_success_without_common_training_copy"]
            - reference["joint_success_without_common_training_copy"]
        )
        uniqueness = (
            candidate["unique_tail_fraction"] - reference["unique_tail_fraction"]
        )
        disorder = (
            candidate["v3_disorder_success_rate"]
            - reference["v3_disorder_success_rate"]
        )
        comparisons[name] = dict(
            joint_gain=gain,
            common_noncopy_joint_gain=noncopy_gain,
            unique_fraction_change=uniqueness,
            disorder_success_change=disorder,
            passes=gain >= 0.05 - 1e-12
            and noncopy_gain >= 0.05 - 1e-12
            and uniqueness >= -0.05 - 1e-12
            and disorder >= -0.05 - 1e-12,
        )
    supervised = arms["continued_supervised"]
    supervised_gain = candidate["joint_success_rate"] - supervised["joint_success_rate"]
    supervised_noncopy_gain = (
        candidate["joint_success_without_common_training_copy"]
        - supervised["joint_success_without_common_training_copy"]
    )
    supervised_unique_change = (
        candidate["unique_tail_fraction"] - supervised["unique_tail_fraction"]
    )
    supervised_disorder_change = (
        candidate["v3_disorder_success_rate"] - supervised["v3_disorder_success_rate"]
    )
    noninferiority = dict(
        joint_gain=supervised_gain,
        common_noncopy_joint_gain=supervised_noncopy_gain,
        unique_fraction_change=supervised_unique_change,
        disorder_success_change=supervised_disorder_change,
        passes=supervised_gain >= -0.02 - 1e-12
        and supervised_noncopy_gain >= -0.02 - 1e-12
        and supervised_unique_change >= -0.05 - 1e-12
        and supervised_disorder_change >= -0.05 - 1e-12,
    )
    primary = all(row["passes"] for row in comparisons.values())
    return dict(
        comparisons=comparisons,
        supervised_noninferiority=noninferiority,
        passes_primary_screen=primary,
        passes_progression_screen=primary and noninferiority["passes"],
        interpretation=(
            "One-seed development screen, not a significance test or formal "
            "noninferiority analysis. No automatic training extension."
        ),
    )


def validate_support(row, prompts, config, generated=False):
    """Keep the original prompt, canonical support and fixed tail length."""
    prompt_id = row["prompt_id"]
    if prompt_id not in prompts:
        raise ValueError("Unknown prompt identity")
    prefix = prompts[prompt_id]["sequence"][: config["prefix_length"]]
    sequence = row["sequence"]
    tail = sequence[config["prefix_length"] :]
    if (
        not sequence.startswith(prefix)
        or len(sequence) != config["length"]
        or set(sequence) - set(ALPHABET)
        or row.get("tail", tail) != tail
    ):
        raise ValueError("Invalid sequence support or tail")
    if generated and (
        row["prefix"] != prefix
        or row["evaluation_start"] != config["prefix_length"]
        or row["seed"] != 0
    ):
        raise ValueError("Invalid evaluation support or seed")
    if "prompt_index" in row and row["prompt_index"] != list(prompts).index(prompt_id):
        raise ValueError("Prompt index and identity disagree")
    if not np.isfinite(row["log_prior"]):
        raise ValueError("Nonfinite frozen-reference likelihood")


def validate_training(run, settings, config, prompts, bank, fresh):
    """Audit actual common draws, including accidental holdout-tail encounters."""
    bank_by_id = {row["id"]: row for row in bank}
    fresh_by_id = {row["id"]: row for row in fresh}
    if (
        len(bank_by_id) != len(bank)
        or len(fresh_by_id) != len(fresh)
        or bank_by_id.keys() & fresh_by_id.keys()
    ):
        raise ValueError("Duplicated training identities")
    if any(row["split"] not in ("train", "holdout") for row in bank):
        raise ValueError("Unknown bank split")
    for row in bank + fresh:
        validate_support(row, prompts, config)
    inherited_ids = {}
    for arm, filename in (
        ("unchanged", "initial_training.jsonl"),
        ("continued_supervised", "supervised_training.jsonl"),
    ):
        inherited_ids[arm] = {
            identity
            for row in read_jsonl(run / filename)
            for identity in row["replay_ids"]
        }
        if any(
            identity not in bank_by_id or bank_by_id[identity]["split"] != "train"
            for identity in inherited_ids[arm]
        ):
            raise ValueError("Inherited fit contains unknown or held-out replay")
    inherited_ids["continued_supervised"] |= inherited_ids["unchanged"]
    shared = read_jsonl(run / "shared_training.jsonl")
    steps = list(range(1, config["steps"] + 1))
    if [row["step"] for row in shared] != steps:
        raise ValueError("Shared training update budget differs")
    if config["steps"] != settings["training_steps"]:
        raise ValueError("Training budget differs between settings and pilot config")
    logs = {arm: read_jsonl(run / arm / "training.jsonl") for arm in TRAINED_ARMS}
    all_rows = {**bank_by_id, **fresh_by_id}
    used_ids = set()
    counts = Counter()
    refresh_ids = {}
    for arm, rows in logs.items():
        if [row["step"] for row in rows] != steps:
            raise ValueError(f"Incomplete training budget: {arm}")
    for step_index, row in enumerate(shared):
        ids = row["example_ids"]
        refresh_round = step_index // settings["refresh_every"]
        if row["refresh_round"] != refresh_round:
            raise ValueError("Refresh schedule differs")
        if refresh_round in refresh_ids and ids != refresh_ids[refresh_round]:
            raise ValueError("Pool changed between scheduled refreshes")
        refresh_ids[refresh_round] = ids
        if not ids or any(identity not in all_rows for identity in ids):
            raise ValueError("Unknown or empty shared training draw")
        if any(
            identity in bank_by_id and bank_by_id[identity]["split"] != "train"
            for identity in ids
        ):
            raise ValueError("Explicit held-out replay example was used")
        actual = Counter(
            fresh_by_id[identity]["generator_arm"]
            if identity in fresh_by_id
            else "replay"
            for identity in ids
        )
        expected = settings["shared_batch"]
        if actual != Counter(expected) or actual != Counter(row["counts"]):
            raise ValueError("Shared source contribution budget differs")
        for arm in TRAINED_ARMS:
            if logs[arm][step_index]["example_ids"] != ids:
                raise ValueError("Arms did not consume identical ordered examples")
        frozen_rates = logs["frozen_tb"][step_index]["rates"]
        adapted_rates = logs["lastblock_tb"][step_index]["rates"]
        if any(adapted_rates.get(key) != rate for key, rate in frozen_rates.items()):
            raise ValueError("Matched head/normalizer learning-rate schedules differ")
        for identity in ids:
            if (
                identity in fresh_by_id
                and fresh_by_id[identity]["refresh_round"] != refresh_round
            ):
                raise ValueError("Fresh trajectory used outside its refresh pool")
        used_ids.update(ids)
        counts.update(actual)
    expected_fresh = len(refresh_ids) * sum(
        count
        for source, count in settings["shared_batch"].items()
        if source != "replay"
    )
    if len(fresh) != expected_fresh:
        raise ValueError("Fresh trajectory budget differs")
    for refresh_round, ids in refresh_ids.items():
        for source, count in settings["shared_batch"].items():
            prompt_counts = Counter(
                all_rows[identity]["prompt_id"]
                for identity in ids
                if (
                    fresh_by_id[identity]["generator_arm"]
                    if identity in fresh_by_id
                    else "replay"
                )
                == source
            )
            if count % len(prompts) or prompt_counts != Counter(
                {prompt: count // len(prompts) for prompt in prompts}
            ):
                raise ValueError("Shared source budget is not prompt-balanced")
    if fresh_by_id.keys() - used_ids:
        raise ValueError("Saved fresh training rows were not consumed")
    own_tails = {
        arm: {bank_by_id[identity]["tail"] for identity in inherited_ids[arm]}
        for arm in ARMS[:2]
    }
    for arm in TRAINED_ARMS:
        own_tails[arm] = own_tails["unchanged"] | {
            all_rows[identity]["tail"] for identity in used_ids
        }
    common_tails = {row["tail"] for row in bank if row["split"] == "train"} | {
        row["tail"] for row in fresh
    }
    exposed = {
        row["id"]
        for row in bank
        if row["split"] == "holdout" and row["tail"] in {item["tail"] for item in fresh}
    }
    summary = dict(
        updates_per_trained_arm=len(shared),
        examples_per_trained_arm=sum(counts.values()),
        source_draws=dict(counts),
        fresh_trajectories=len(fresh),
        common_training_unique_tails=len(common_tails),
        accidental_holdout_overlap_count=len(exposed),
        accidental_holdout_overlap_ids=sorted(exposed),
        matched_example_ids=True,
        interpretation=(
            "The two learned policies share the same refreshed data. This is a "
            "coupled representation comparison, not independent training pipelines."
        ),
    )
    return own_tails, common_tails, exposed, summary


def summarize_bank(run, bank, prompts, exposed):
    """Use the same unexposed holdout for every arm; never compute global TV."""
    by_id = {row["id"]: row for row in bank}
    output = {}
    for arm in ARMS:
        fits = json.loads((run / arm / "bank_fit.json").read_text())
        final = [row for row in fits if row["stage"] == "final"]
        if Counter(row["id"] for row in final) != Counter(by_id.keys()):
            raise ValueError("Final bank-fit identities differ")
        for row in final:
            original = by_id[row["id"]]
            if (
                row["prompt_id"] != original["prompt_id"]
                or row["split"] != original["split"]
                or row["joint_pass"] != original["joint_pass"]
                or (
                    bool(row["fresh_training_overlap"]) != (row["id"] in exposed)
                    and row["split"] == "holdout"
                )
                or not np.isclose(
                    row["log_prior"], original["log_prior"], atol=0.003, rtol=1e-5
                )
                or not np.isfinite(row["log_policy"])
            ):
                raise ValueError(
                    "Bank-fit metadata or frozen-reference likelihood differs"
                )
            if row.get("log_z") is not None:
                residual = (
                    row["log_z"]
                    + row["log_policy"]
                    - row["log_prior"]
                    - original["log_factor"]
                )
                if not np.isclose(row["tb_residual"], residual, atol=1e-3):
                    raise ValueError(
                        "Stored bank TB residual differs from fixed reward"
                    )
        for split in ("train", "holdout"):
            for prompt in (None, *prompts):
                rows = [
                    row
                    for row in final
                    if row["split"] == split
                    and (prompt is None or row["prompt_id"] == prompt)
                    and not (split == "holdout" and row["id"] in exposed)
                ]
                result = dict(count=len(rows))
                if rows:
                    ratios = np.array(
                        [
                            row["log_policy"]
                            - row["log_prior"]
                            - by_id[row["id"]]["log_factor"]
                            for row in rows
                        ]
                    )
                    # Center within each prompt because normalizers are conditional.
                    centered = np.array(
                        [
                            ratio
                            - np.mean(
                                [
                                    other_ratio
                                    for other_ratio, other in zip(ratios, rows)
                                    if other["prompt_id"] == row["prompt_id"]
                                ]
                            )
                            for ratio, row in zip(ratios, rows)
                        ]
                    )
                    result["centered_log_ratio_rmse"] = float(
                        np.sqrt(np.mean(centered**2))
                    )
                    if all(row.get("log_z") is not None for row in rows):
                        result["tb_rmse"] = float(
                            np.sqrt(np.mean([row["tb_residual"] ** 2 for row in rows]))
                        )
                    if prompt is not None:
                        log_q = np.array([row["log_policy"] for row in rows])
                        maximum = log_q.max()
                        result["log_policy_mass_on_bank"] = float(
                            maximum + np.log(np.exp(log_q - maximum).sum())
                        )
                output[f"{arm}/{split}/{prompt or 'all'}"] = result
    return output


def summarize(rows):
    result = summarize_outputs(rows)
    result.update(
        exact_common_training_copy_fraction=float(
            np.mean([row["common_training_copy"] for row in rows])
        ),
        joint_success_without_common_training_copy=float(
            np.mean(
                [
                    row["acceptance"]["joint_pass"] and not row["common_training_copy"]
                    for row in rows
                ]
            )
        ),
        mean_log_policy_minus_frozen_log_prior=float(
            np.mean([row["log_policy"] - row["log_prior"] for row in rows])
        ),
        log_ratio_interpretation=(
            "Own-policy Monte Carlo estimate of forward KL(q || frozen p0), "
            "conditional on the fixed prompts; finite estimates can be negative."
        ),
    )
    return result


def evaluate_lastblock(run, output, predictor=predict_disorder):
    """Reconstruct endpoints from saved sequences without loading ProGen."""
    config = json.loads((run / "pilot_config.json").read_text())
    settings = json.loads((run / "config.json").read_text())
    reward = AcceptanceRewardConfig(**json.loads((run / "reward.json").read_text()))
    references = json.loads((run / "references.json").read_text())
    prompts = {row["id"]: row for row in references if row["role"] == "prompt"}
    bank = json.loads((run / "bank.json").read_text())
    fresh = read_jsonl(run / "training_rows.jsonl")
    records = read_jsonl(run / "samples.jsonl")
    expected = {
        (arm, prompt): config["samples_per_prompt"]
        for arm in ARMS
        for prompt in prompts
    }
    if Counter((row["arm"], row["prompt_id"]) for row in records) != expected or len(
        {row["id"] for row in records}
    ) != len(records):
        raise ValueError("Fresh generation counts or identities differ")
    for row in records:
        validate_support(row, prompts, config, generated=True)
        if not np.isfinite(row["log_policy"]):
            raise ValueError("Nonfinite generated policy likelihood")
    own_tails, common_tails, exposed, training_summary = validate_training(
        run, settings, config, prompts, bank, fresh
    )
    training_summary["gradients"] = {}
    for arm in TRAINED_ARMS:
        logs = read_jsonl(run / arm / "training.jsonl")
        gradients = {}
        for group, field, threshold in (
            (
                "head_and_normalizer",
                "head_gradient_norm_before_clip",
                config["gradient_clip"],
            ),
            (
                "last_block",
                "block_gradient_norm_before_clip",
                settings["block_gradient_clip"],
            ),
        ):
            norms = [row[field] for row in logs if row.get(field) is not None]
            if norms:
                gradients[group] = dict(
                    mean_norm_before_clip=float(np.mean(norms)),
                    maximum_norm_before_clip=max(norms),
                    clipped_update_fraction=float(np.mean(np.array(norms) > threshold)),
                )
        training_summary["gradients"][arm] = gradients
    fresh_values = continuous_reward(
        [row["tail"] for row in fresh],
        [row["isolated_tail_v3_scores"] for row in fresh],
        reward,
        settings["penalty"],
    )
    if not np.allclose(
        [row["log_factor"] for row in fresh],
        fresh_values["log_factor"].numpy(),
        atol=1e-5,
    ):
        raise ValueError("Saved fresh reward factors differ from the fixed reward")
    diagnostics = summarize_bank(run, bank, prompts, exposed)
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
        row["own_training_copy"] = tail in own_tails[row["arm"]]
        codes = np.frombuffer(tail.encode("ascii"), dtype=np.uint8)
        row["nearest_seed_bank_identity"] = float((seed_codes == codes).mean(1).max())
        if row.get("log_z") is not None:
            row["own_reward_tb_residual"] = (
                row["log_z"]
                + row["log_policy"]
                - row["log_prior"]
                - float(values["log_factor"][index])
            )
    metrics = dict(
        arms={
            arm: summarize([row for row in scored if row["arm"] == arm]) for arm in ARMS
        },
        by_prompt={
            f"{arm}/{prompt}": summarize(
                [
                    row
                    for row in scored
                    if row["arm"] == arm and row["prompt_id"] == prompt
                ]
            )
            for arm in ARMS
            for prompt in prompts
        },
    )
    decision = progression_decision(metrics["arms"])
    for name, value in (
        ("sequence_metrics.json", scored),
        ("metrics.json", metrics),
        ("decision.json", decision),
        ("bank_diagnostics.json", diagnostics),
        ("training_summary.json", training_summary),
    ):
        write_json(output / name, value)
    report = [
        "# Final-block GFlowNet adaptation pilot",
        "",
        "Original frozen ProGen defines every terminal reward; only the final-block arm adapts its policy representation.",
        "All endpoints use fresh, unrestricted 20-residue generation, with one training seed.",
        "",
        "| Policy | Outputs | Joint charge/V3 | V3 disorder | Unique tails | Common-training copies | Noncopy joint |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in metrics["arms"].items():
        report.append(
            f"| {arm} | {row['count']} | {row['joint_success_rate']:.2%} | "
            f"{row['v3_disorder_success_rate']:.2%} | {row['unique_tail_fraction']:.2%} | "
            f"{row['exact_common_training_copy_fraction']:.2%} | "
            f"{row['joint_success_without_common_training_copy']:.2%} |"
        )
    report += [
        "",
        f"Primary screen: **{'PASS' if decision['passes_primary_screen'] else 'FAIL'}**. "
        f"Supervised-control tolerance: **{'PASS' if decision['supervised_noninferiority']['passes'] else 'FAIL'}**. "
        f"Combined progression screen: **{'PASS' if decision['passes_progression_screen'] else 'FAIL'}**.",
        "",
        "Require +5 percentage points in raw and common-training-noncopy joint success over both unchanged and frozen TB; lose no more than 5 points in uniqueness or V3 disorder against either. Also remain within 2 points of continued supervised joint and noncopy joint success, and within 5 points of its uniqueness and V3 disorder. These are practical development thresholds, not formal statistical tests.",
        f"Common holdout excludes {len(exposed)} accidentally encountered training tails for every arm. No held-out bank rows are explicitly replayed.",
        "The training arms share refreshed examples, so this isolates adaptation under common data, not independent end-to-end pipelines. Common-copy controls use the union of original training-bank and fresh training tails; their noncopy rates retain all outputs in the denominator.",
        "V3 disorder is an optimized prediction; V2 is sensitivity, not independent biological validation. Exact noncopy does not establish biological novelty. Fixed-bank residuals are diagnostics, not global TV or proof of target matching.",
        "Controls use train-only normalizer offsets for diagnostics; these are not fitted partition-function estimates. See bank_diagnostics.json for common held-out residuals, metrics.json for own-policy log-ratio drift, and training_summary.json plus costs.json for data budgets and computational cost.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output / "report.md"
