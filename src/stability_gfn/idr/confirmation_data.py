"""Pin the successful last-block protocol while changing only random streams."""

import copy
import json
import re
from dataclasses import asdict

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .lastblock_data import check_lastblock_inputs

SEED_KEYS = ("training_seed", "evaluation_seed", "bank_sampling_seed")
MAX_SEED = 2**63 - 3  # Reserve training_seed + 1 and + 2 for the paired sampler.


def _stream_seeds(settings):
    """Enumerate independent streams; arm evaluations deliberately share one seed."""
    return [
        settings["training_seed"],
        settings["training_seed"] + 1,
        settings["training_seed"] + 2,
        settings["evaluation_seed"],
        settings["bank_sampling_seed"],
    ]


def check_confirmation_inputs(settings, check_checkpoint=True):
    """Validate two fresh fits without loading ProGen or executing disorder models.

    Every non-seed setting comes from the completed pilot, including its original
    supervised initialization. The adapted pilot checkpoint is never a new start.
    """
    if set(settings) != {"pilot_run", "pilot_generation_sha256", "replicates"}:
        raise ValueError("Confirmation accepts only the pinned pilot and two seed sets")
    replicas = settings["replicates"]
    if not isinstance(replicas, list) or len(replicas) != 2:
        raise ValueError("Confirmation requires exactly two fresh replicates")
    names = set()
    for replica in replicas:
        if not isinstance(replica, dict) or set(replica) != {"name", *SEED_KEYS}:
            raise ValueError(
                "Each replicate must specify only its name and three seeds"
            )
        name = replica["name"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is None
            or name in names
        ):
            raise ValueError("Replicate names must be safe, distinct folder names")
        names.add(name)
        for key in SEED_KEYS:
            if type(replica[key]) is not int or not 0 <= replica[key] <= MAX_SEED:
                raise ValueError(f"Invalid replicate {key}")

    pilot = project_path(settings["pilot_run"])
    if file_hash(pilot / "generation.json") != settings["pilot_generation_sha256"]:
        raise ValueError("Pinned last-block pilot manifest differs")
    status = json.loads((pilot / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Last-block pilot is not complete")
    manifest = verify_manifest(pilot)
    required = {
        "config.json",
        "pilot_config.json",
        "references.json",
        "provenance.json",
        "reward.json",
        "reward_definition.json",
        "bank.json",
        "initial.pt",
        "initial_training.jsonl",
        "supervised.pt",
        "supervised_training.jsonl",
        "source_generation.json",
        "control_generation.json",
        "samples.jsonl",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Pilot snapshots are not protected by its manifest")
    provenance = json.loads((pilot / "provenance.json").read_text())
    verify_implementation(provenance)
    if check_checkpoint and (provenance["fixture"] or status.get("fixture", False)):
        raise ValueError("Real confirmation cannot use a fixture pilot")
    fit_template = json.loads((pilot / "config.json").read_text())
    config = json.loads((pilot / "pilot_config.json").read_text())
    references = json.loads((pilot / "references.json").read_text())

    # Keep all new streams disjoint from the pilot and both inherited controls.
    # The last-block trainer uses three sequential seeds for its refreshed pool.
    used_seeds = set(_stream_seeds(fit_template))
    for folder_key in ("source_run", "control_run"):
        old_settings = json.loads(
            (project_path(fit_template[folder_key]) / "config.json").read_text()
        )
        used_seeds.update(_stream_seeds(old_settings))
    derived = []
    for replica in replicas:
        seeds = _stream_seeds(replica)
        if len(set(seeds)) != len(seeds) or used_seeds.intersection(seeds):
            raise ValueError("Confirmation random streams must be fresh and disjoint")
        used_seeds.update(seeds)
        fit_settings = copy.deepcopy(fit_template)
        fit_settings.update({key: replica[key] for key in SEED_KEYS})
        derived.append((replica["name"], fit_settings))

    for index, (_, fit_settings) in enumerate(derived):
        # Both fits use identical model files. Hash the large checkpoint once;
        # the second call still checks all saved control inputs and software.
        inputs = check_lastblock_inputs(
            fit_settings, check_checkpoint=check_checkpoint and index == 0
        )
        source, control, derived_config, derived_references, source_provenance = inputs[
            :5
        ]
        reward, bank = inputs[5:7]
        if derived_config != config or derived_references != references:
            raise ValueError(
                "Pilot support/configuration differs from its source inputs"
            )
        if json.loads((pilot / "reward.json").read_text()) != json.loads(
            json.dumps(asdict(reward))
        ):
            raise ValueError("Pilot reward differs from the inherited reward")
        if json.loads((pilot / "bank.json").read_text()) != bank:
            raise ValueError("Pilot training bank differs from its original source")
        reward_definition = json.loads((pilot / "reward_definition.json").read_text())
        if (
            reward_definition.get("version") != "continuous_acceptance_v1"
            or reward_definition.get("penalty") != fit_settings["penalty"]
        ):
            raise ValueError("Pilot reward definition differs from the fixed target")
        for folder, filename, snapshot in (
            (source, "initial.pt", "initial.pt"),
            (source, "initial_training.jsonl", "initial_training.jsonl"),
            (control, fit_settings["control_checkpoint"], "supervised.pt"),
            (
                control,
                "continued_supervised/training.jsonl",
                "supervised_training.jsonl",
            ),
            (source, "generation.json", "source_generation.json"),
            (control, "generation.json", "control_generation.json"),
        ):
            if file_hash(folder / filename) != file_hash(pilot / snapshot):
                raise ValueError(
                    f"Pilot initialization/source snapshot differs: {snapshot}"
                )
        for filename, digest in source_provenance["files"].items():
            if provenance["files"].get(filename) != digest:
                raise ValueError(
                    f"Pilot and inherited source provenance differ: {filename}"
                )
        if provenance["packages"] != source_provenance["packages"]:
            raise ValueError("Pilot and inherited source package versions differ")

    prompt_count = sum(row["role"] == "prompt" for row in references)
    expected_count = 4 * prompt_count * fit_template["evaluation_samples_per_prompt"]
    if manifest["count"] != expected_count:
        raise ValueError("Pilot final evaluation count differs from its protocol")
    return pilot, derived, config, references, provenance


def confirmation_preflight(settings):
    """Summarize the fixed workload after read-only artifact/model-file checks."""
    pilot, replicas, config, references, _ = check_confirmation_inputs(settings)
    fit_settings = replicas[0][1]
    prompt_count = sum(row["role"] == "prompt" for row in references)
    per_arm = prompt_count * config["samples_per_prompt"]
    refresh_rounds = fit_settings["training_steps"] // fit_settings["refresh_every"]
    new_training_rows = refresh_rounds * sum(
        count
        for source, count in fit_settings["shared_batch"].items()
        if source != "replay"
    )
    return {
        "pilot_run": str(pilot),
        "replicates": [
            dict(name=name, **{key: fit[key] for key in SEED_KEYS})
            for name, fit in replicas
        ],
        "training_arms_per_replicate": 2,
        "updates_per_training_arm": fit_settings["training_steps"],
        "fresh_training_trajectories_per_replicate": new_training_rows,
        "trajectory_presentations_per_training_arm": (
            fit_settings["training_steps"] * sum(fit_settings["shared_batch"].values())
        ),
        "evaluation_arms_per_replicate": 5,
        "evaluation_samples_per_arm": per_arm,
        "total_final_evaluation_samples": len(replicas) * 5 * per_arm,
        "cuda_initialized": torch.cuda.is_initialized(),
        "model_or_scorer_executed": False,
    }
