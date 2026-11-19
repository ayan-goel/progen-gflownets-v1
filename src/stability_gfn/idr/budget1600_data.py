"""Pin the completed 800-update comparison for one bounded 1,600-update test."""

import copy
import json
import re

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .budget_data import (
    _check_final_policy,
    _completed_snapshot,
    check_budget_inputs,
)
from .confirmation_data import MAX_SEED, _stream_seeds


def check_budget1600_inputs(settings, check_checkpoint=True):
    """Derive two new fits without loading a model or changing saved artifacts.

    The original head and ProGen weights remain the training initialization.
    Saved 800-update checkpoints are evaluation controls only. Historical
    400-update training rows are pinned because the copy screen retains them.
    """
    if set(settings) != {
        "budget_run",
        "budget_generation_sha256",
        "training_steps",
        "replicates",
    }:
        raise ValueError("1600 comparison accepts only source, doubled steps and seeds")
    if type(settings["training_steps"]) is not int or settings["training_steps"] < 1:
        raise ValueError("Invalid training_steps")
    replicas = settings["replicates"]
    if not isinstance(replicas, list) or len(replicas) != 2:
        raise ValueError("1600 comparison requires both budget replicates")
    names = set()
    for replica in replicas:
        if not isinstance(replica, dict) or set(replica) != {"name", "evaluation_seed"}:
            raise ValueError("Each replicate accepts only name and evaluation_seed")
        name, seed = replica["name"], replica["evaluation_seed"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is None
            or name in names
        ):
            raise ValueError("Replicate names must be safe and distinct")
        names.add(name)
        if type(seed) is not int or not 0 <= seed <= MAX_SEED:
            raise ValueError("Invalid evaluation_seed")

    source = project_path(settings["budget_run"])
    if file_hash(source / "generation.json") != settings["budget_generation_sha256"]:
        raise ValueError("Pinned 800-update generation manifest differs")
    manifest, provenance = _completed_snapshot(source, check_checkpoint)
    required = {
        "config.json",
        "provenance.json",
        "confirmation_config.json",
        "confirmation_generation.json",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("800-update source snapshots are not protected")
    source_settings = json.loads((source / "config.json").read_text())
    confirmation, old_plans, old_config, references, inherited_provenance = (
        check_budget_inputs(source_settings, check_checkpoint=check_checkpoint)
    )
    if [plan["name"] for plan in old_plans] != [
        replica["name"] for replica in replicas
    ]:
        raise ValueError("Use both original budget replicate identities in order")
    expected_steps = 8 if provenance["fixture"] else 800
    if (
        old_config["steps"] != expected_steps
        or settings["training_steps"] != 2 * expected_steps
    ):
        raise ValueError("Require the fixed 800-to-1600 budget (8-to-16 for fixtures)")
    for filename in ("config.json", "generation.json"):
        if file_hash(confirmation / filename) != file_hash(
            source / f"confirmation_{filename}"
        ):
            raise ValueError("800-update confirmation snapshot differs")
    if provenance["packages"] != inherited_provenance["packages"] or any(
        provenance["files"].get(filename) != digest
        for filename, digest in inherited_provenance["files"].items()
    ):
        raise ValueError("800-update and inherited provenance differ")

    # Both fits reuse their paired training/replay seeds. Only evaluation needs
    # new streams, disjoint from all older evaluations and adjacent samplers.
    historical = [plan["fit_settings"] for plan in old_plans]
    historical.extend(
        json.loads((plan["source_fit"] / "config.json").read_text())
        for plan in old_plans
    )
    confirmation_settings = json.loads((confirmation / "config.json").read_text())
    pilot = project_path(confirmation_settings["pilot_run"])
    historical.append(json.loads((pilot / "config.json").read_text()))
    for key in ("source_run", "control_run"):
        folder = project_path(historical[0][key])
        historical.append(json.loads((folder / "config.json").read_text()))
    used_seeds = {seed for previous in historical for seed in _stream_seeds(previous)}
    for replica in replicas:
        if replica["evaluation_seed"] in used_seeds:
            raise ValueError("1600 evaluation streams must be fresh and disjoint")
        used_seeds.add(replica["evaluation_seed"])

    plans = []
    for replica, previous in zip(replicas, old_plans):
        name = replica["name"]
        fit = source / name / "fit"
        fit_manifest, fit_provenance = _completed_snapshot(fit, check_checkpoint)
        child_required = {
            "config.json",
            "pilot_config.json",
            "references.json",
            "provenance.json",
            "initial.pt",
            "initial_training.jsonl",
            "bank.json",
            "reward.json",
            "reward_definition.json",
            "adaptation.json",
            "lastblock_tb/checkpoint.pt",
            "training_rows.jsonl",
        }
        history_required = {
            "config.json",
            "checkpoint.pt",
            "source_settings.json",
            "bank.json",
            "training_rows.jsonl",
            "source_generation.json",
        }
        if (
            not child_required.issubset(fit_manifest["files"])
            or any(
                f"{name}/fit/{filename}" not in manifest["files"]
                for filename in child_required | {"generation.json"}
            )
            or any(
                f"{name}/control400/{filename}" not in manifest["files"]
                for filename in history_required
            )
        ):
            raise ValueError(
                "800-update child or historical copy inputs are not protected"
            )
        if json.loads((fit / "config.json").read_text()) != previous["fit_settings"]:
            raise ValueError("800-update child settings differ from its derivation")
        if json.loads((fit / "pilot_config.json").read_text()) != old_config:
            raise ValueError("800-update child support/configuration differs")
        if json.loads((fit / "references.json").read_text()) != references:
            raise ValueError("800-update child reference panel differs")
        if fit_provenance["packages"] != provenance["packages"] or any(
            provenance["files"].get(filename) != digest
            for filename, digest in fit_provenance["files"].items()
        ):
            raise ValueError("800-update parent/child provenance differs")
        for filename in (
            "initial.pt",
            "initial_training.jsonl",
            "bank.json",
            "reward.json",
            "reward_definition.json",
            "supervised.pt",
            "supervised_training.jsonl",
            "source_generation.json",
            "control_generation.json",
        ):
            if file_hash(fit / filename) != file_hash(
                previous["source_fit"] / filename
            ):
                raise ValueError(f"Original initialization or bank changed: {filename}")
        _check_final_policy(fit, old_config, references, check_checkpoint)

        history = source / name / "control400"
        for filename, original in (
            ("checkpoint.pt", "lastblock_tb/checkpoint.pt"),
            ("source_settings.json", "config.json"),
            ("bank.json", "bank.json"),
            ("training_rows.jsonl", "training_rows.jsonl"),
            ("source_generation.json", "generation.json"),
        ):
            if file_hash(history / filename) != file_hash(
                previous["source_fit"] / original
            ):
                raise ValueError(
                    f"Historical 400-update copy input differs: {filename}"
                )
        historical_control = json.loads((history / "config.json").read_text())
        if (
            historical_control["arm"] != "lastblock_400"
            or historical_control["evaluation_seed"]
            != previous["fit_settings"]["evaluation_seed"]
            or historical_control["source_checkpoint_sha256"]
            != file_hash(history / "checkpoint.pt")
        ):
            raise ValueError("Historical 400-update control identity differs")
        fit_settings = copy.deepcopy(previous["fit_settings"])
        fit_settings.update(
            training_steps=settings["training_steps"],
            evaluation_seed=replica["evaluation_seed"],
        )
        plans.append(dict(name=name, source_fit=fit, fit_settings=fit_settings))

    prompt_count = sum(row["role"] == "prompt" for row in references)
    if (
        manifest["count"]
        != len(plans) * 6 * prompt_count * old_config["samples_per_prompt"]
    ):
        raise ValueError("800-update generation count differs from its protocol")
    config = copy.deepcopy(old_config)
    config["steps"] = settings["training_steps"]
    return source, plans, config, references, copy.deepcopy(provenance)


def budget1600_preflight(settings):
    """Summarize the bounded user-run workload after CPU-only artifact checks."""
    source, plans, config, references, _ = check_budget1600_inputs(settings)
    fit = plans[0]["fit_settings"]
    prompt_count = sum(row["role"] == "prompt" for row in references)
    per_arm = prompt_count * config["samples_per_prompt"]
    fresh_per_refresh = sum(
        count for name, count in fit["shared_batch"].items() if name != "replay"
    )
    return dict(
        ready=True,
        budget_run=str(source),
        replicates=[
            dict(
                name=plan["name"],
                evaluation_seed=plan["fit_settings"]["evaluation_seed"],
            )
            for plan in plans
        ],
        training_arms_per_replicate=2,
        updates_per_training_arm=fit["training_steps"],
        reference_updates_per_training_arm=fit["training_steps"] // 2,
        learning_rate_decay_after_update=int(
            config["decay_fraction"] * fit["training_steps"]
        ),
        fresh_training_trajectories_per_replicate=(
            fit["training_steps"] // fit["refresh_every"] * fresh_per_refresh
        ),
        trajectory_presentations_per_training_arm=(
            fit["training_steps"] * sum(fit["shared_batch"].values())
        ),
        evaluation_arms_per_replicate=6,
        evaluation_samples_per_arm=per_arm,
        total_final_evaluation_samples=len(plans) * 6 * per_arm,
        cuda_initialized=torch.cuda.is_initialized(),
        model_or_scorer_executed=False,
    )
