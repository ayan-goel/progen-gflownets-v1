"""Audit the paired final-block GFN and reward-weighted supervised comparison.

Every endpoint retains the raw generation denominator. Both fits consume common
refreshed examples; this compares objectives in a coupled data-collection system,
not the cost or performance of two independently operated training pipelines.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..artifacts import write_json
from .acceptance_reward import AcceptanceRewardConfig
from .charge import charge_reward
from .continuous_reward import continuous_reward
from .lastblock_evaluation import read_jsonl, summarize, validate_support
from .scoring import predict_disorder, score_records

ARMS = ("gfn", "supervised", "prior")
TRAINED_ARMS = ARMS[:2]
CLUSTER_IDENTITY = 0.8
EFFECT_METRICS = (
    "joint_success_rate",
    "joint_success_without_common_training_copy",
    "unique_tail_fraction",
    "v3_disorder_success_rate",
    "distinct_successful_tail_fraction",
    "distinct_noncopy_successful_tail_fraction",
    "near_identity_cluster_fraction",
    "successful_cluster_fraction",
    "noncopy_successful_cluster_fraction",
    "exact_common_training_copy_fraction",
    "charge_success_rate",
)


def cluster_sizes(tails):
    """Connected components at >=80% equal-length, positionwise tail identity.

    Exact duplicates contribute their raw multiplicity to component size. The
    transitive rule is deliberate: all pairs in a component need not be similar.
    This describes sequence diversity, not evolutionary or functional families.
    """
    multiplicities = Counter(tails)
    unique = sorted(multiplicities)
    if not unique:
        return []
    length = len(unique[0])
    if not length or any(len(tail) != length for tail in unique):
        raise ValueError("Clustering requires nonempty equal-length tails")
    codes = np.frombuffer("".join(unique).encode("ascii"), dtype=np.uint8).reshape(
        len(unique), length
    )
    parents = list(range(len(unique)))

    def component(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    minimum_matches = int(np.ceil(CLUSTER_IDENTITY * length - 1e-12))
    for index in range(len(unique) - 1):
        matches = (codes[index + 1 :] == codes[index]).sum(axis=1)
        for other in np.flatnonzero(matches >= minimum_matches) + index + 1:
            parents[component(int(other))] = component(index)
    sizes = Counter()
    for index, tail in enumerate(unique):
        sizes[component(index)] += multiplicities[tail]
    return sorted(sizes.values(), reverse=True)


def summarize_objective(rows):
    """Add successful-tail yield and within-prompt clusters to raw endpoints."""
    result = summarize(rows)
    count = len(rows)
    successful = [row for row in rows if row["acceptance"]["joint_pass"]]
    noncopy_successful = [row for row in successful if not row["common_training_copy"]]
    result.update(
        joint_success_count=len(successful),
        noncopy_joint_success_count=len(noncopy_successful),
        unique_tail_count=len({row["tail"] for row in rows}),
        distinct_successful_tail_count=len({row["tail"] for row in successful}),
        distinct_noncopy_successful_tail_count=len(
            {row["tail"] for row in noncopy_successful}
        ),
    )
    for label in ("distinct_successful_tail", "distinct_noncopy_successful_tail"):
        result[f"{label}_fraction"] = result[f"{label}_count"] / count
    cluster_groups = {}
    for label, selected in (
        ("all", rows),
        ("successful", successful),
        ("noncopy_successful", noncopy_successful),
    ):
        sizes = [
            size
            for prompt in sorted({row["prompt_id"] for row in rows})
            for size in cluster_sizes(
                [row["tail"] for row in selected if row["prompt_id"] == prompt]
            )
        ]
        cluster_groups[label] = dict(
            count=len(sizes),
            fraction_of_raw_outputs=len(sizes) / count,
            largest_cluster_raw_count=max(sizes, default=0),
        )
    result.update(
        near_identity_clusters=cluster_groups,
        near_identity_cluster_fraction=cluster_groups["all"]["fraction_of_raw_outputs"],
        successful_cluster_fraction=cluster_groups["successful"][
            "fraction_of_raw_outputs"
        ],
        noncopy_successful_cluster_fraction=cluster_groups["noncopy_successful"][
            "fraction_of_raw_outputs"
        ],
    )
    return result


def objective_effects(arms):
    """GFN minus matched supervised effects; positive rates favor the GFN."""
    return {
        metric: arms["gfn"][metric] - arms["supervised"][metric]
        for metric in EFFECT_METRICS
        if metric in arms["gfn"] and metric in arms["supervised"]
    }


def objective_decision(replicates):
    """Apply the declared practical screen to each repeat without selecting one."""
    if len(replicates) != 2:
        raise ValueError("Objective comparison requires exactly two paired repeats")
    decisions = {}
    for name, arms in replicates.items():
        effects = objective_effects(arms)
        decisions[name] = dict(
            gfn_minus_supervised=effects,
            passes=(
                effects["joint_success_rate"] >= 0.05 - 1e-12
                and effects["joint_success_without_common_training_copy"]
                >= 0.05 - 1e-12
                and effects["unique_tail_fraction"] >= -0.05 - 1e-12
                and effects["v3_disorder_success_rate"] >= -0.05 - 1e-12
            ),
        )
    return dict(
        replicates=decisions,
        passes_practical_screen=all(row["passes"] for row in decisions.values()),
        required_joint_gain=0.05,
        required_common_noncopy_joint_gain=0.05,
        maximum_unique_fraction_loss=0.05,
        maximum_disorder_success_loss=0.05,
        interpretation=(
            "Both repeats must pass this practical development screen. Two seeds "
            "on development prompts do not establish formal superiority. Failure "
            "does not establish supervised superiority or equivalence. Diversity "
            "and cost remain descriptive tradeoffs, not alternative post-hoc gates."
        ),
    )


def _validate_fresh_reward(fresh, reward, penalty):
    """Recompute the detached terminal factor from saved training scorer arrays."""
    values = continuous_reward(
        [row["tail"] for row in fresh],
        [row["isolated_tail_v3_scores"] for row in fresh],
        reward,
        penalty,
    )
    if not np.allclose(
        [row["log_factor"] for row in fresh],
        values["log_factor"].numpy(),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("Saved fresh property factors differ from the fixed reward")
    for index, row in enumerate(fresh):
        if row.get("joint_pass", bool(values["joint_pass"][index])) != bool(
            values["joint_pass"][index]
        ):
            raise ValueError("Saved fresh acceptance labels differ")
        if "violations" in row and not np.allclose(
            row["violations"], values["violations"][index].numpy(), atol=1e-6
        ):
            raise ValueError("Saved fresh violations differ")


def _validate_weighting(logged, diagnostics):
    """Check optional trainer diagnostics against full-batch property weights."""
    if "weighted_nll" in logged and not np.isclose(
        logged["weighted_nll"], logged["loss"], atol=1e-8, rtol=1e-6
    ):
        raise ValueError("Supervised loss and weighted NLL differ")
    if "weighting" not in logged:
        return False
    weighting = logged["weighting"]
    if (
        weighting["method"] != "per_prompt_softmax_log_factor_equal_prompt_mean"
        or weighting["normalized_before_microbatch"] is not True
        or weighting["prior_in_weights"] is not False
        or weighting["represented_prompts"] != len(diagnostics)
        or not np.isclose(weighting["loss_weight_sum"], 1.0)
        or len(weighting["prompts"]) != len(diagnostics)
    ):
        raise ValueError("Supervised weighting convention differs")
    for saved, recomputed in zip(weighting["prompts"], diagnostics):
        for key in (
            "prompt_index",
            "examples",
            "weight_sum",
            "minimum_weight",
            "maximum_weight",
            "effective_sample_size",
            "minimum_log_factor",
            "maximum_log_factor",
        ):
            if not np.isclose(saved[key], recomputed[key], atol=1e-8, rtol=1e-6):
                raise ValueError(f"Supervised weight diagnostic differs: {key}")
    total_ess = len(diagnostics) ** 2 / sum(
        1 / row["effective_sample_size"] for row in diagnostics
    )
    if "effective_sample_size" in logged and not np.isclose(
        logged["effective_sample_size"], total_ess, atol=1e-8, rtol=1e-6
    ):
        raise ValueError("Supervised total effective sample size differs")
    return True


def _validate_training(folder, settings, config, prompts, bank, fresh, initial_ids):
    """Verify every ordered batch, its replay boundary and its refresh lifetime."""
    bank_by_id = {row["id"]: row for row in bank}
    fresh_by_id = {row["id"]: row for row in fresh}
    if len(fresh_by_id) != len(fresh) or fresh_by_id.keys() & bank_by_id.keys():
        raise ValueError("Duplicated training identities")
    if config["steps"] != settings["training_steps"]:
        raise ValueError("Training budgets differ between fit and pilot settings")
    shared = read_jsonl(folder / "shared_training.jsonl")
    steps = list(range(1, settings["training_steps"] + 1))
    logs = {arm: read_jsonl(folder / arm / "training.jsonl") for arm in TRAINED_ARMS}
    for name, rows in {"shared": shared, **logs}.items():
        if [row["step"] for row in rows] != steps:
            raise ValueError(f"Incomplete or unordered training budget: {name}")
    expected_sources = settings["shared_batch"]
    if set(expected_sources) != {*ARMS, "replay"} or any(
        type(count) is not int or count <= 0 or count % len(prompts)
        for count in expected_sources.values()
    ):
        raise ValueError("Shared contributions must be positive and prompt-balanced")
    if settings["refresh_every"] <= 0:
        raise ValueError("Refresh interval must be positive")
    all_rows = {**bank_by_id, **fresh_by_id}
    used_ids = set()
    source_draws = Counter()
    refresh_ids = {}
    weight_diagnostics = []
    checked_weighting_updates = 0
    for index, row in enumerate(shared):
        refresh_round = index // settings["refresh_every"]
        identities = row["example_ids"]
        if row["refresh_round"] != refresh_round:
            raise ValueError("Shared refresh schedule differs")
        if refresh_round in refresh_ids and refresh_ids[refresh_round] != identities:
            raise ValueError("Shared pool changed between scheduled refreshes")
        refresh_ids[refresh_round] = identities
        if not identities or any(identity not in all_rows for identity in identities):
            raise ValueError("Unknown or empty shared training batch")
        if any(
            identity in bank_by_id and bank_by_id[identity]["split"] != "train"
            for identity in identities
        ):
            raise ValueError("Held-out replay row was explicitly consumed")
        sources = Counter(
            fresh_by_id[identity]["generator_arm"]
            if identity in fresh_by_id
            else "replay"
            for identity in identities
        )
        if sources != Counter(expected_sources) or sources != Counter(row["counts"]):
            raise ValueError("Shared source contributions differ")
        for source, count in expected_sources.items():
            prompt_counts = Counter(
                all_rows[identity]["prompt_id"]
                for identity in identities
                if (
                    fresh_by_id[identity]["generator_arm"]
                    if identity in fresh_by_id
                    else "replay"
                )
                == source
            )
            if prompt_counts != Counter(
                {prompt: count // len(prompts) for prompt in prompts}
            ):
                raise ValueError("Shared batch is not balanced within each source")
        for identity in identities:
            if identity in fresh_by_id and (
                fresh_by_id[identity]["refresh_round"] != refresh_round
            ):
                raise ValueError("Fresh trajectory consumed outside its refresh pool")
        for arm in TRAINED_ARMS:
            logged = logs[arm][index]
            if logged["example_ids"] != identities:
                raise ValueError("Arms consumed different ordered shared examples")
            if not np.isfinite(logged["loss"]):
                raise ValueError("Nonfinite training loss")
        gfn_rates = {
            key: value
            for key, value in logs["gfn"][index]["rates"].items()
            if key != "normalizer"
        }
        sft_rates = logs["supervised"][index]["rates"]
        if gfn_rates != sft_rates or any(
            not np.isfinite(value) or value < 0 for value in gfn_rates.values()
        ):
            raise ValueError("Matched head/block learning-rate schedules differ")
        used_ids.update(identities)
        source_draws.update(sources)
        # Calculate the weighting diagnostics from the actual examples rather
        # than a separate data pool. A prompt gets equal total objective weight.
        prompt_diagnostics = []
        for prompt_index, prompt in enumerate(prompts):
            selected = [
                all_rows[identity]
                for identity in identities
                if all_rows[identity]["prompt_id"] == prompt
            ]
            factors = np.array([item["log_factor"] for item in selected])
            if not np.isfinite(factors).all():
                raise ValueError("Nonfinite training property factor")
            weights = np.exp(factors - factors.max())
            weights /= weights.sum()
            prompt_diagnostics.append(
                dict(
                    step=row["step"],
                    prompt_id=prompt,
                    prompt_index=prompt_index,
                    examples=len(selected),
                    effective_sample_size=float(1 / (weights @ weights)),
                    weight_sum=float(weights.sum()),
                    minimum_weight=float(weights.min()),
                    maximum_weight=float(weights.max()),
                    minimum_log_factor=float(factors.min()),
                    maximum_log_factor=float(factors.max()),
                )
            )
        checked_weighting_updates += _validate_weighting(
            logs["supervised"][index], prompt_diagnostics
        )
        weight_diagnostics.extend(prompt_diagnostics)
    expected_fresh = len(refresh_ids) * sum(expected_sources[arm] for arm in ARMS)
    if len(fresh) != expected_fresh or fresh_by_id.keys() - used_ids:
        raise ValueError("Fresh trajectory budget or consumption differs")
    own_tails = {all_rows[identity]["tail"] for identity in used_ids | initial_ids}
    exposed = sorted(
        row["id"]
        for row in bank
        if row["split"] == "holdout" and row["tail"] in {item["tail"] for item in fresh}
    )
    gradients = {}
    update_seconds = {}
    for arm, rows in logs.items():
        times = [row["update_seconds"] for row in rows if "update_seconds" in row]
        if times:
            if (
                len(times) != len(rows)
                or not np.isfinite(times).all()
                or min(times) < 0
            ):
                raise ValueError("Invalid per-arm update timings")
            update_seconds[arm] = sum(times)
        gradients[arm] = {}
        for group, field, threshold in (
            ("head", "head_gradient_norm_before_clip", config["gradient_clip"]),
            (
                "block",
                "block_gradient_norm_before_clip",
                settings["block_gradient_clip"],
            ),
        ):
            norms = [row[field] for row in rows if row.get(field) is not None]
            if norms:
                if not np.isfinite(norms).all():
                    raise ValueError("Nonfinite gradient diagnostics")
                gradients[arm][group] = dict(
                    mean_norm_before_clip=float(np.mean(norms)),
                    maximum_norm_before_clip=max(norms),
                    clipped_update_fraction=float(np.mean(np.array(norms) > threshold)),
                )
    return own_tails, dict(
        updates_per_arm=len(shared),
        examples_per_arm=sum(source_draws.values()),
        fresh_trajectories=len(fresh),
        refresh_rounds=len(refresh_ids),
        source_draws=dict(source_draws),
        matched_ordered_example_ids=True,
        explicit_holdout_replay_count=0,
        accidental_holdout_overlap_count=len(exposed),
        accidental_holdout_overlap_ids=exposed,
        gradients=gradients,
        measured_update_seconds_by_arm=update_seconds,
        supervised_weighting=dict(
            rule="property_factor_per_prompt",
            diagnostics_source="Recomputed from the actual shared examples",
            logged_weighting_updates_checked=checked_weighting_updates,
            by_step_prompt=weight_diagnostics,
            mean_effective_sample_size=float(
                np.mean([row["effective_sample_size"] for row in weight_diagnostics])
            ),
            maximum_weight=max(row["maximum_weight"] for row in weight_diagnostics),
        ),
    )


def evaluate_objective(run, output, predictor=predict_disorder):
    """Evaluate saved final samples with an injectable CPU/cache-only predictor.

    The caller owns provenance/manifest verification and recovery. This function
    does not load policy checkpoints or mutate the source run when output differs.
    """
    run, output = Path(run), Path(output)
    settings = json.loads((run / "config.json").read_text())
    fit = settings["fit"]
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
    if (
        fit.get("evaluation_samples_per_prompt", config["samples_per_prompt"])
        != config["samples_per_prompt"]
    ):
        raise ValueError("Fit and pilot evaluation budgets differ")
    planned = settings["replicates"]
    names = [row["name"] for row in planned]
    if (
        len(names) != 2
        or len(set(names)) != 2
        or any(Path(name).name != name or name in (".", "..") for name in names)
    ):
        raise ValueError("Need exactly two distinct, safe replicate names")
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
        _validate_fresh_reward(fresh, reward, fit["penalty"])
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
        raise ValueError("Combined evaluation records differ from the two repeats")
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
    values = continuous_reward(
        tails,
        [row["disorder"]["V3"]["isolated_tail_scores"] for row in scored],
        reward,
        fit["penalty"],
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
    decision = objective_decision(metrics["by_replicate"])
    training_summary.update(
        common_training_unique_tails=len(common_tails),
        initial_supervised_unique_used_ids=len(initial_ids),
        copy_definition=(
            "Every arm and repeat uses one conservative union of the initial "
            "training-bank tails and all fresh training tails from BOTH repeats. "
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
        "# Matched GFN versus reward-weighted supervised adaptation",
        "",
        "Both models adapt the same final block and guidance head from the original shared supervised initialization, using identical refreshed examples.",
        "",
        "| Repeat | Policy | Raw outputs | Joint | Noncopy joint | Unique tails | V3 disorder | Distinct successful tails / raw | Successful clusters / raw |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arms in metrics["by_replicate"].items():
        for arm, row in arms.items():
            report.append(
                f"| {name} | {arm} | {row['count']} | {row['joint_success_rate']:.2%} | "
                f"{row['joint_success_without_common_training_copy']:.2%} | "
                f"{row['unique_tail_fraction']:.2%} | {row['v3_disorder_success_rate']:.2%} | "
                f"{row['distinct_successful_tail_fraction']:.2%} | {row['successful_cluster_fraction']:.2%} |"
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
            f"Practical screen: **{'PASS' if decision['passes_practical_screen'] else 'FAIL'}**. Both repeats must gain at least five points in raw and noncopy joint success, while losing no more than five points in uniqueness or V3 disorder. Means cannot rescue a failed repeat.",
            decision["interpretation"],
            "",
            training_summary["copy_definition"],
            "All rates and successful-tail/cluster yields retain raw outputs in their denominator; exact noncopy does not establish biological novelty.",
            training_summary["cluster_definition"],
            "",
            "When success is similar, examine distinct successful tails, cluster yields, repetition and measured cost together; these are descriptive tradeoffs, not a post-hoc alternative winner rule. Per-repeat and per-prompt effects are retained in metrics.json.",
            training_summary["interpretation"],
            training_summary["costs"]["interpretation"],
            "Detailed measured stages, weighting concentration and gradient diagnostics are retained in training_summary.json and the original costs.json files.",
            "",
            "V3 disorder is the optimized prediction; V2 is a sensitivity check, not independent biological validation. This development comparison does not establish generalization to new prompts/properties, biological function, or global target-distribution accuracy. No checkpoint selection or automatic training extension follows.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n")
    return output / "report.md"
