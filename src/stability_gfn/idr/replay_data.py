"""Training-only replay bank and exact teacher-forced caches for frozen ProGen."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .acceptance_experiment import (
    REWARD_VERSION,
    verify_implementation,
    verify_manifest,
)
from .acceptance_reward import AcceptanceRewardConfig, joint_acceptance_reward
from .charge import context_features
from .charge_experiment import check_charge_inputs
from .data import ALPHABET

CACHE_KEYS = ("features", "context", "logits", "actions", "log_prior")


def make_bank(rows, prompts, settings):
    """Deduplicate per prompt; split identical tails together before selecting near-misses.

    Hash splitting is outcome-independent. A deterministic repair reserves at least
    one training success per prompt, moving that tail globally if necessary. The
    diagnostic holdout may consequently have no positives for a scarce prompt.
    """
    unique = {}
    for row in rows:
        unique.setdefault((row["prompt_id"], row["tail"]), row)
    split_seed = settings["split_seed"]

    def ordering(row):
        return hashlib.sha256(f"{split_seed}:{row['tail']}".encode()).hexdigest()

    held_out = {
        row["tail"]
        for row in unique.values()
        if int(ordering(row)[:16], 16) / 2**64 < settings["holdout_fraction"]
    }
    repaired = []
    for prompt in prompts:
        positives = sorted(
            [
                r
                for r in unique.values()
                if r["prompt_id"] == prompt["id"] and r["joint_pass"]
            ],
            key=ordering,
        )
        if not positives:
            raise ValueError(f"No source training successes for {prompt['id']}")
        if all(r["tail"] in held_out for r in positives):
            tail = positives[0]["tail"]
            held_out.remove(tail)
            repaired.append({"prompt_id": prompt["id"], "tail": tail})
    bank = []
    for prompt_index, prompt in enumerate(prompts):
        group = [r for r in unique.values() if r["prompt_id"] == prompt["id"]]
        for split in ("train", "holdout"):
            candidates = [
                r for r in group if (r["tail"] in held_out) == (split == "holdout")
            ]
            positives = sorted([r for r in candidates if r["joint_pass"]], key=ordering)
            negatives = sorted(
                [r for r in candidates if not r["joint_pass"]],
                key=lambda r: (-r["log_factor"], ordering(r)),
            )
            failure_limit = (
                settings["near_misses_per_prompt"]
                if split == "train"
                else settings["holdout_failures_per_prompt"]
            )
            for row in (
                positives[: settings["success_cap_per_prompt"]]
                + negatives[:failure_limit]
            ):
                bank.append(
                    dict(
                        row,
                        id=f"source-{row['seed']}-{row['step']}-{prompt_index}",
                        prompt_index=prompt_index,
                        split=split,
                        sequence=prompt["sequence"][: settings["prefix_length"]]
                        + row["tail"],
                    )
                )
        if not any(
            r["prompt_id"] == prompt["id"]
            and r["split"] == "train"
            and not r["joint_pass"]
            for r in bank
        ):
            raise ValueError(f"No training near-misses for {prompt['id']}")
    counts = Counter(
        f"{r['prompt_id']}/{r['split']}/{'pass' if r['joint_pass'] else 'fail'}"
        for r in bank
    )
    train_tails = {r["tail"] for r in bank if r["split"] == "train"}
    assert not train_tails.intersection(
        r["tail"] for r in bank if r["split"] == "holdout"
    )
    return bank, {
        "counts": dict(counts),
        "split_repairs": repaired,
        "source_rows": len(rows),
        "source_unique_prompt_tails": len(unique),
    }


def check_replay_inputs(settings, check_checkpoint=True):
    """Verify pinned completed inputs; no pretrained execution or GPU queries."""
    source = project_path(settings["source_run"])
    if file_hash(source / "generation.json") != settings["source_generation_sha256"]:
        raise ValueError("Acceptance source manifest changed")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("Acceptance source is incomplete")
    verify_manifest(source)
    provenance = json.loads((source / "provenance.json").read_text())
    verify_implementation(provenance)
    if check_checkpoint and provenance["fixture"]:
        raise ValueError("Real experiments cannot use fixture inputs")
    config = json.loads((source / "pilot_config.json").read_text())
    references, _ = check_charge_inputs(config, check_checkpoint)
    if references != json.loads((source / "references.json").read_text()):
        raise ValueError("Reference panel changed")
    reward_settings = json.loads((source / "config.json").read_text())["reward"]
    reward = AcceptanceRewardConfig(**reward_settings)
    if config.get("reward_version") != REWARD_VERSION or config["beta"] != 1:
        raise ValueError("Require the completed joint-acceptance experiment")
    if check_checkpoint and json.dumps(asdict(reward), sort_keys=True) != json.dumps(
        asdict(AcceptanceRewardConfig()), sort_keys=True
    ):
        raise ValueError("Keep the completed acceptance reward unchanged")
    for key in ("block_targets", "block_tolerance", "fcr_target", "fcr_tolerance"):
        if json.dumps(getattr(reward, key)) != json.dumps(config["reward"][key]):
            raise ValueError("Charge settings differ from the source reward")
    for key in ("disorder_cutoff", "minimum_disordered_fraction"):
        if getattr(reward, key) != config[key]:
            raise ValueError("Disorder settings differ from the source reward")
    if settings["seed"] != 0 or settings["seed"] not in config["seeds"]:
        raise ValueError("This pilot fixes seed zero before seeing new outcomes")
    if not 0 < settings["holdout_fraction"] < 0.5:
        raise ValueError("Invalid diagnostic holdout fraction")
    for key in (
        "training_steps",
        "evaluation_samples_per_prompt",
        "cache_batch_size",
        "near_misses_per_prompt",
        "holdout_failures_per_prompt",
        "success_cap_per_prompt",
    ):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    for key in (
        "split_seed",
        "training_seed",
        "evaluation_seed",
        "prior_seed",
        "bank_sampling_seed",
    ):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"Invalid {key}")
    if check_checkpoint:
        for name in ("config.json", "pytorch_model.bin"):
            path = project_path(config["checkpoint"]) / name
            if file_hash(path) != provenance["files"].get(str(path.resolve())):
                raise ValueError("Frozen ProGen checkpoint changed")
    prompts = [r for r in references if r["role"] == "prompt"]
    if len(prompts) % 2:
        raise ValueError("Replay requires an even prompt count for balanced strata")
    rows = [
        json.loads(line)
        for line in (source / f"seed-{settings['seed']}/reward_inputs.jsonl")
        .read_text()
        .splitlines()
    ]
    expected_prompts = {p["id"] for p in prompts}
    if len(rows) != config["steps"] * len(prompts) or Counter(
        (r["step"], r["prompt_id"]) for r in rows
    ) != Counter(
        {
            (step, prompt): 1
            for step in range(1, config["steps"] + 1)
            for prompt in expected_prompts
        }
    ):
        raise ValueError("Incomplete source training trajectories")
    values = joint_acceptance_reward(
        [r["tail"] for r in rows], [r["isolated_tail_v3_scores"] for r in rows], reward
    )
    for index, row in enumerate(rows):
        if row["seed"] != settings["seed"] or row["joint_pass"] != bool(
            values["joint_pass"][index]
        ):
            raise ValueError("Invalid source training identity or label")
        if abs(row["log_factor"] - float(values["log_factor"][index])) > 1e-10:
            raise ValueError("Source reward cannot be reconstructed")
    bank, summary = make_bank(
        rows, prompts, dict(settings, prefix_length=config["prefix_length"])
    )
    effective = dict(
        config,
        seeds=[settings["seed"]],
        steps=settings["training_steps"],
        samples_per_prompt=settings["evaluation_samples_per_prompt"],
    )
    return source, effective, references, provenance, reward, bank, summary


@torch.no_grad()
def cache_sequences(backbone, bank, prompts, config, batch_size):
    """Teacher-force fixed sequences with causal alignment identical to rollout.

    State at index prefix_length (including the direction marker) predicts the
    first generated residue. Full inputs omit the last residue; no future token
    enters its own prediction. Keep float32 CPU caches for detached head fitting.
    """
    parts = {key: [] for key in CACHE_KEYS}
    prefix_length = config["prefix_length"]
    length = config["length"] - prefix_length
    token_ids = backbone.residue_ids(ALPHABET)
    for start in range(0, len(bank), batch_size):
        rows = bank[start : start + batch_size]
        states, logits = backbone.forward([r["sequence"][:-1] for r in rows])
        features = states[:, prefix_length:]
        base_logits = logits[:, prefix_length:, token_ids]
        actions = torch.tensor(
            [[ALPHABET.index(a) for a in r["tail"]] for r in rows],
            device=backbone.device,
        )
        context = torch.stack(
            [
                context_features(
                    [r["tail"][:step] for r in rows],
                    [r["prompt_index"] for r in rows],
                    len(prompts),
                    length,
                    config["reward"]["block_targets"],
                    backbone.device,
                )
                for step in range(length)
            ],
            dim=1,
        )
        log_prior = (
            base_logits.log_softmax(-1)
            .gather(-1, actions[..., None])
            .squeeze(-1)
            .sum(-1)
        )
        # Incremental and batched kernels may differ slightly in float32.
        expected = torch.tensor([r["log_prior"] for r in rows], device=backbone.device)
        if not torch.allclose(log_prior, expected, atol=0.003, rtol=1e-5):
            raise ValueError("Teacher-forced prior disagrees with source trajectory")
        for key, value in zip(
            CACHE_KEYS, (features, context, base_logits, actions, log_prior)
        ):
            parts[key].append(value.detach().cpu())
    return {key: torch.cat(values) for key, values in parts.items()}


def cached_batch(cache, indices, device):
    """Transfer only a head minibatch; frozen states never receive gradients."""
    return {key: value[indices].to(device) for key, value in cache.items()}


def bank_indices(bank, prompt_count, generator, step, supervised=False):
    """Prompt-balanced draws: replay alternates success/near-miss by prompt and step."""
    chosen = []
    for repeat in range(2 if supervised else 1):
        for prompt in range(prompt_count):
            positive = supervised or (prompt + step) % 2 == 0
            candidates = [
                i
                for i, row in enumerate(bank)
                if row["split"] == "train"
                and row["prompt_index"] == prompt
                and row["joint_pass"] == positive
            ]
            chosen.append(
                candidates[
                    int(torch.randint(len(candidates), (1,), generator=generator))
                ]
            )
    return chosen
