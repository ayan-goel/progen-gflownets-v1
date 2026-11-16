"""Pinned supervised checkpoint and unchanged cached replay bank for continuation."""

import json
from dataclasses import asdict

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .acceptance_experiment import (
    REWARD_VERSION,
    verify_implementation,
    verify_manifest,
)
from .acceptance_reward import AcceptanceRewardConfig
from .charge_experiment import check_charge_inputs
from .replay_data import CACHE_KEYS


def check_warmstart_inputs(settings, check_checkpoint=True):
    """Read and validate artifacts only; never instantiate a pretrained model."""
    for key in ("training_steps", "evaluation_samples_per_prompt"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in ("training_seed", "evaluation_seed", "bank_sampling_seed"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"Invalid {key}")
    source = project_path(settings["source_run"])
    if file_hash(source / "generation.json") != settings["source_generation_sha256"]:
        raise ValueError("Pinned replay manifest changed")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("Replay source is not complete")
    manifest = verify_manifest(source)
    provenance = json.loads((source / "provenance.json").read_text())
    verify_implementation(provenance)
    if check_checkpoint and provenance["fixture"]:
        raise ValueError("Real continuation cannot use fixture inputs")
    config = json.loads((source / "pilot_config.json").read_text())
    source_settings = json.loads((source / "config.json").read_text())
    references, _ = check_charge_inputs(config, check_checkpoint)
    if references != json.loads((source / "references.json").read_text()):
        raise ValueError("Reference panel differs from source")
    prompts = [r for r in references if r["role"] == "prompt"]
    if len(prompts) % 2 or config["seeds"] != [0] or source_settings["seed"] != 0:
        raise ValueError("Require the one-seed replay pilot and even prompt count")
    reward = AcceptanceRewardConfig(**json.loads((source / "reward.json").read_text()))
    if config["beta"] != 1 or config["reward_version"] != REWARD_VERSION:
        raise ValueError("Require the unchanged acceptance reward")
    if check_checkpoint and json.dumps(asdict(reward), sort_keys=True) != json.dumps(
        asdict(AcceptanceRewardConfig()), sort_keys=True
    ):
        raise ValueError("Real continuation must keep the original reward")
    checkpoint_name = f"supervised/step-{config['steps']}.pt"
    required = {
        checkpoint_name,
        "bank.json",
        "bank_cache.pt",
        "supervised/training.jsonl",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Source checkpoint/cache not captured by manifest")
    checkpoint = torch.load(
        source / checkpoint_name, map_location="cpu", weights_only=True
    )
    if (
        checkpoint["arm"] != "supervised"
        or checkpoint["architecture"] != "direct_context"
        or checkpoint["log_z"] is not None
        or checkpoint["reward_version"] != REWARD_VERSION
        or checkpoint["config"] != config
        or checkpoint["settings"] != source_settings
        or checkpoint["step"] != config["steps"]
        or checkpoint["prompt_ids"] != [r["id"] for r in prompts]
    ):
        raise ValueError("Supervised checkpoint metadata differs from source")
    if any(not torch.isfinite(value).all() for value in checkpoint["head"].values()):
        raise ValueError("Nonfinite starting head")
    bank = json.loads((source / "bank.json").read_text())
    cache = torch.load(source / "bank_cache.pt", map_location="cpu", weights_only=True)
    if set(cache) != set(CACHE_KEYS) or any(
        len(value) != len(bank) or not torch.isfinite(value).all()
        for value in cache.values()
    ):
        raise ValueError("Invalid source feature cache")
    train_tails = {r["tail"] for r in bank if r["split"] == "train"}
    if train_tails.intersection(r["tail"] for r in bank if r["split"] == "holdout"):
        raise ValueError("Bank partitions share a tail")
    for index in range(len(prompts)):
        for passed in (True, False):
            if not any(
                r["prompt_index"] == index
                and r["split"] == "train"
                and r["joint_pass"] == passed
                for r in bank
            ):
                raise ValueError("Missing training stratum for normalizer/replay")
    if check_checkpoint:
        for name in ("config.json", "pytorch_model.bin"):
            path = project_path(config["checkpoint"]) / name
            if file_hash(path) != provenance["files"].get(str(path.resolve())):
                raise ValueError("Frozen ProGen checkpoint changed")
    effective = dict(
        config,
        steps=settings["training_steps"],
        samples_per_prompt=settings["evaluation_samples_per_prompt"],
    )
    return source, effective, references, provenance, reward, bank, cache, checkpoint


def initialize_normalizer(fit_rows, bank, prompt_ids):
    """Least-squares offset with equal pass/fail weight per prompt, training only.

    This centers the initial TB residual on the replay distribution. It is NOT
    log E_prior[w] or an unbiased partition-function estimate. Holdout rows cannot
    influence it. The normalizer remains trainable during continuation.
    """
    by_id = {r["id"]: r for r in bank}
    offsets, diagnostics = [], []
    for prompt in prompt_ids:
        means = []
        for passed in (True, False):
            values = [
                r["log_prior"] + by_id[r["id"]]["log_factor"] - r["log_policy"]
                for r in fit_rows
                if r["prompt_id"] == prompt
                and r["split"] == "train"
                and r["joint_pass"] == passed
            ]
            if not values:
                raise ValueError("No training examples for normalizer initialization")
            means.append(sum(values) / len(values))
        offset = sum(means) / 2
        offsets.append(offset)
        diagnostics.append(
            {
                "prompt_id": prompt,
                "log_z": offset,
                "pass_offset": means[0],
                "fail_offset": means[1],
            }
        )
    return torch.tensor(offsets, dtype=torch.float32), diagnostics
