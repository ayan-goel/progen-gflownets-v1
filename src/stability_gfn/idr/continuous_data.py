"""Validate the matched completed TB control without changing historical code."""

import json

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .continuous_reward import continuous_reward
from .warmstart_data import check_warmstart_inputs


def relabel_bank(bank, reward, penalty):
    """Keep identities/splits/acceptance labels; replace only terminal log weight."""
    values = continuous_reward(
        [row["tail"] for row in bank],
        [row["isolated_tail_v3_scores"] for row in bank],
        reward,
        penalty,
    )
    if values["joint_pass"].tolist() != [row["joint_pass"] for row in bank]:
        raise ValueError("Acceptance labels changed")
    return [
        dict(row, log_factor=float(values["log_factor"][index]))
        for index, row in enumerate(bank)
    ]


def check_continuous_inputs(settings, check_checkpoint=True):
    """Pin the original start and completed comparator; no model execution."""
    inputs = check_warmstart_inputs(settings, check_checkpoint)
    source, config, references, provenance, reward, bank, cache, initial = inputs
    if settings["penalty"] != 2.5:
        raise ValueError("This bounded comparison fixes the audited penalty at 2.5")
    if check_checkpoint:
        audit = project_path(settings["reward_audit"])
        if file_hash(audit) != settings["reward_audit_sha256"]:
            raise ValueError("Reward calibration record changed")
        if json.loads(audit.read_text())["selected_penalty"] != settings["penalty"]:
            raise ValueError("Penalty differs from the calibration record")
    control = project_path(settings["control_run"])
    if file_hash(control / "generation.json") != settings["control_generation_sha256"]:
        raise ValueError("Pinned warm-start control changed")
    if json.loads((control / "status.json").read_text())["status"] != "complete":
        raise ValueError("Warm-start control is incomplete")
    verify_manifest(control)
    control_provenance = json.loads((control / "provenance.json").read_text())
    verify_implementation(control_provenance)
    if check_checkpoint and control_provenance["fixture"]:
        raise ValueError("Real runs cannot use fixture controls")
    control_settings = json.loads((control / "config.json").read_text())
    for key in (
        "source_generation_sha256",
        "training_steps",
        "training_seed",
        "bank_sampling_seed",
    ):
        if settings[key] != control_settings[key]:
            raise ValueError(f"Matched training setting changed: {key}")
    if settings["evaluation_seed"] == control_settings["evaluation_seed"]:
        raise ValueError("Use a fresh evaluation stream")
    control_config = json.loads((control / "pilot_config.json").read_text())
    if {k: v for k, v in config.items() if k != "samples_per_prompt"} != {
        k: v for k, v in control_config.items() if k != "samples_per_prompt"
    }:
        raise ValueError("Training configuration differs from the matched control")
    for filename in ("bank.json", "bank_cache.pt", "references.json", "reward.json"):
        if file_hash(source / filename) != file_hash(control / filename):
            raise ValueError(f"Source and control disagree: {filename}")
    if file_hash(source / f"supervised/step-{initial['step']}.pt") != file_hash(
        control / "initial.pt"
    ):
        raise ValueError("Starting head differs from the completed TB control")
    checkpoint = torch.load(
        control / f"warm_tb/step-{config['steps']}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint["arm"] != "warm_tb"
        or checkpoint["step"] != config["steps"]
        or checkpoint["config"] != control_config
        or checkpoint["settings"] != control_settings
        or checkpoint["prompt_ids"] != initial["prompt_ids"]
        or checkpoint["architecture"] != "direct_context"
        or checkpoint["reward_version"] != "joint_acceptance_v1"
        or not torch.isfinite(checkpoint["log_z"]).all()
        or any(not torch.isfinite(value).all() for value in checkpoint["head"].values())
    ):
        raise ValueError("Completed TB checkpoint metadata/weights differ")
    return inputs, control, checkpoint
