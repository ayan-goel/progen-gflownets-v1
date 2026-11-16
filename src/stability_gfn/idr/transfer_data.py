"""Pinned training-only full-trajectory inputs for the IDR transfer pilot."""

import json

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .acceptance_reward import AcceptanceRewardConfig
from .charge_experiment import check_charge_inputs
from .continuous_data import relabel_bank
from .replay_data import CACHE_KEYS


def check_transfer_inputs(settings, check_checkpoint=True):
    for key in (
        "training_steps",
        "evaluation_samples_per_prompt",
        "draws_per_prompt",
        "snapshot_interval",
        "microbatch_size",
    ):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in ("bank_sampling_seed", "evaluation_seed"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"Invalid {key}")
    if settings["penalty"] != 2.5 or settings["conditioning_maximum_gain"] != 32:
        raise ValueError("Keep the predeclared reward and conditioning cap")
    source = project_path(settings["source_run"])
    if file_hash(source / "generation.json") != settings["source_generation_sha256"]:
        raise ValueError("Pinned source generation manifest differs")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("Source incomplete")
    manifest = verify_manifest(source)
    required = {
        "bank.json",
        "bank_cache.pt",
        "initial.pt",
        "reward.json",
        "pilot_config.json",
        "references.json",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Source inputs not protected by manifest")
    provenance = json.loads((source / "provenance.json").read_text())
    verify_implementation(provenance)
    if check_checkpoint and provenance["fixture"]:
        raise ValueError("Real runs cannot use synthetic fixtures")
    config = json.loads((source / "pilot_config.json").read_text())
    references, _ = check_charge_inputs(config, check_checkpoint)
    if references != json.loads((source / "references.json").read_text()):
        raise ValueError("Reference panel changed")
    if (
        settings["evaluation_seed"]
        == json.loads((source / "config.json").read_text())["evaluation_seed"]
    ):
        raise ValueError("Evaluation seed must be fresh")
    reward = AcceptanceRewardConfig(**json.loads((source / "reward.json").read_text()))
    definition = json.loads((source / "reward_definition.json").read_text())
    if (
        definition["version"] != "continuous_acceptance_v1"
        or definition["penalty"] != settings["penalty"]
    ):
        raise ValueError("Source reward differs")
    initial = torch.load(source / "initial.pt", map_location="cpu", weights_only=True)
    prompts = [r for r in references if r["role"] == "prompt"]
    if (
        initial["prompt_ids"] != [p["id"] for p in prompts]
        or initial["architecture"] != "direct_context"
    ):
        raise ValueError("Initial head prompt/architecture mismatch")
    if initial["log_z"] is not None or any(
        not torch.isfinite(v).all() for v in initial["head"].values()
    ):
        raise ValueError("Invalid supervised starting checkpoint")
    bank = json.loads((source / "bank.json").read_text())
    cache = torch.load(source / "bank_cache.pt", map_location="cpu", weights_only=True)
    if set(cache) != set(CACHE_KEYS) or any(
        len(v) != len(bank) for v in cache.values()
    ):
        raise ValueError("Invalid cache keys/counts")
    selected = [i for i, row in enumerate(bank) if row["split"] == "train"]
    bank = [bank[i] for i in selected]
    cache = {key: value[selected] for key, value in cache.items()}
    if not bank or any(not torch.isfinite(v).all() for v in cache.values()):
        raise ValueError("Empty or nonfinite training cache")
    if (
        cache["logits"].shape[-1] != 20
        or cache["actions"].shape[1] != reward.tail_length
    ):
        raise ValueError("Need complete 20-residue full trajectories")
    prior = (
        cache["logits"]
        .log_softmax(-1)
        .gather(-1, cache["actions"][..., None])
        .squeeze(-1)
        .sum(-1)
    )
    if not torch.allclose(prior, cache["log_prior"], atol=0.003, rtol=1e-5):
        raise ValueError("Full-support cached prior does not reconstruct")
    for prompt in range(len(prompts)):
        for passed in (True, False):
            if not any(
                r["prompt_index"] == prompt and r["joint_pass"] == passed for r in bank
            ):
                raise ValueError("Missing training proposal stratum")
    if check_checkpoint:
        for name in ("config.json", "pytorch_model.bin"):
            path = project_path(config["checkpoint"]) / name
            if file_hash(path) != provenance["files"].get(str(path.resolve())):
                raise ValueError("Frozen ProGen checkpoint changed")
    config = dict(
        config,
        steps=settings["training_steps"],
        samples_per_prompt=settings["evaluation_samples_per_prompt"],
    )
    return (
        source,
        config,
        references,
        provenance,
        reward,
        relabel_bank(bank, reward, settings["penalty"]),
        cache,
        initial,
    )
