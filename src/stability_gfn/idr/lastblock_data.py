"""Pinned starting policies and training-only replay for last-block adaptation."""

import json
import math
from dataclasses import asdict

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .acceptance_reward import AcceptanceRewardConfig
from .charge_experiment import check_charge_inputs
from .continuous_data import relabel_bank
from .replay_data import CACHE_KEYS


def check_lastblock_inputs(settings, check_checkpoint=True):
    """Read artifacts and hashes only; never load ProGen or execute a scorer."""
    for key in (
        "training_steps",
        "refresh_every",
        "microbatch_size",
        "evaluation_samples_per_prompt",
    ):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    if settings["training_steps"] % settings["refresh_every"]:
        raise ValueError("Training must contain complete refresh rounds")
    for key in ("training_seed", "evaluation_seed", "bank_sampling_seed"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("last_block_learning_rate", "block_gradient_clip"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if settings["penalty"] != 2.5:
        raise ValueError("Keep the fixed continuous reward penalty")
    source = project_path(settings["source_run"])
    control = project_path(settings["control_run"])
    manifests, provenances = [], []
    for folder, digest in (
        (source, settings["source_generation_sha256"]),
        (control, settings["control_generation_sha256"]),
    ):
        if file_hash(folder / "generation.json") != digest:
            raise ValueError("Pinned source/control manifest differs")
        if json.loads((folder / "status.json").read_text())["status"] != "complete":
            raise ValueError("Source/control incomplete")
        manifests.append(verify_manifest(folder))
        provenance = json.loads((folder / "provenance.json").read_text())
        verify_implementation(provenance)
        if check_checkpoint and provenance["fixture"]:
            raise ValueError("Real runs cannot use fixture inputs")
        provenances.append(provenance)
    required = {
        "bank.json",
        "bank_cache.pt",
        "initial.pt",
        "initial_training.jsonl",
        "reward.json",
        "reward_definition.json",
        "pilot_config.json",
        "references.json",
        "provenance.json",
        "config.json",
    }
    if not required.issubset(manifests[0]["files"]):
        raise ValueError("Source inputs not protected by manifest")
    checkpoint_name = settings["control_checkpoint"]
    control_required = {
        checkpoint_name,
        "continued_supervised/training.jsonl",
        "initial.pt",
        "bank.json",
        "references.json",
        "reward.json",
        "pilot_config.json",
    }
    if not control_required.issubset(manifests[1]["files"]):
        raise ValueError("Supervised control inputs not protected by manifest")
    if file_hash(control / checkpoint_name) != settings["control_checkpoint_sha256"]:
        raise ValueError("Pinned continued-supervised checkpoint differs")
    for name in ("bank.json", "references.json", "reward.json", "initial.pt"):
        if file_hash(source / name) != file_hash(control / name):
            raise ValueError(f"Source/control disagree: {name}")
    config = json.loads((source / "pilot_config.json").read_text())
    references, _ = check_charge_inputs(config, check_checkpoint)
    if references != json.loads((source / "references.json").read_text()):
        raise ValueError("Reference panel changed")
    prompts = [r for r in references if r["role"] == "prompt"]
    expected_mix = {
        name: len(prompts) for name in ("frozen_tb", "lastblock_tb", "prior", "replay")
    }
    if settings["shared_batch"] != expected_mix:
        raise ValueError("Require one example per prompt from each of four sources")
    if settings["microbatch_size"] > sum(expected_mix.values()):
        raise ValueError("Microbatch cannot exceed the paired batch")
    if (
        config["precision"] != "float32"
        or config["seeds"] != [0]
        or config["beta"] != 1
    ):
        raise ValueError("Keep the one-seed float32 probability convention")
    for folder in (source, control):
        if (
            settings["evaluation_seed"]
            == json.loads((folder / "config.json").read_text())["evaluation_seed"]
        ):
            raise ValueError("Use a fresh evaluation stream")
    reward = AcceptanceRewardConfig(**json.loads((source / "reward.json").read_text()))
    definition = json.loads((source / "reward_definition.json").read_text())
    if (
        definition["version"] != "continuous_acceptance_v1"
        or definition["penalty"] != settings["penalty"]
    ):
        raise ValueError("Source reward differs")
    if check_checkpoint and json.dumps(asdict(reward), sort_keys=True) != json.dumps(
        asdict(AcceptanceRewardConfig()), sort_keys=True
    ):
        raise ValueError("Real runs must retain the charge/disorder reward")
    initial = torch.load(source / "initial.pt", map_location="cpu", weights_only=True)
    supervised = torch.load(
        control / checkpoint_name, map_location="cpu", weights_only=True
    )
    for checkpoint in (initial, supervised):
        if (
            checkpoint["prompt_ids"] != [p["id"] for p in prompts]
            or checkpoint["architecture"] != "direct_context"
        ):
            raise ValueError("Starting policy prompt/architecture mismatch")
        if checkpoint["log_z"] is not None or any(
            not torch.isfinite(v).all() for v in checkpoint["head"].values()
        ):
            raise ValueError("Invalid supervised checkpoint")
    if supervised["arm"] != "continued_supervised":
        raise ValueError("Wrong saved comparator")
    bank = json.loads((source / "bank.json").read_text())
    cache = torch.load(source / "bank_cache.pt", map_location="cpu", weights_only=True)
    if set(cache) != set(CACHE_KEYS) or any(
        len(v) != len(bank) or not torch.isfinite(v).all() for v in cache.values()
    ):
        raise ValueError("Invalid source cache")
    train_ids = {r["id"] for r in bank if r["split"] == "train"}
    train_tails = {r["tail"] for r in bank if r["split"] == "train"}
    if len({r["id"] for r in bank}) != len(bank) or train_tails.intersection(
        r["tail"] for r in bank if r["split"] == "holdout"
    ):
        raise ValueError("Duplicate identities or overlapping bank partitions")
    for folder, name in (
        (source, "initial_training.jsonl"),
        (control, "continued_supervised/training.jsonl"),
    ):
        ids = {
            i
            for line in (folder / name).read_text().splitlines()
            for i in json.loads(line)["replay_ids"]
        }
        if not ids.issubset(train_ids):
            raise ValueError("Saved supervised policy used nontraining bank rows")
    for index in range(len(prompts)):
        for passed in (True, False):
            if not any(
                r["split"] == "train"
                and r["prompt_index"] == index
                and r["joint_pass"] == passed
                for r in bank
            ):
                raise ValueError("Missing prompt-specific training replay stratum")
    reconstructed = (
        cache["logits"]
        .log_softmax(-1)
        .gather(-1, cache["actions"][..., None])
        .squeeze(-1)
        .sum(-1)
    )
    if not torch.allclose(reconstructed, cache["log_prior"], atol=0.003, rtol=1e-5):
        raise ValueError("Cached original prior does not reconstruct")
    if check_checkpoint:
        for name in ("config.json", "pytorch_model.bin"):
            path = project_path(config["checkpoint"]) / name
            if file_hash(path) != provenances[0]["files"].get(str(path.resolve())):
                raise ValueError("Original frozen ProGen checkpoint changed")
    config = dict(
        config,
        steps=settings["training_steps"],
        samples_per_prompt=settings["evaluation_samples_per_prompt"],
    )
    return (
        source,
        control,
        config,
        references,
        provenances[0],
        reward,
        relabel_bank(bank, reward, settings["penalty"]),
        cache,
        initial,
        supervised,
    )
