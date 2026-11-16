"""One matched TB/replay continuation using the continuous terminal factor."""

import json
import time

import numpy as np
import torch

from ..artifacts import write_json
from .acceptance_experiment import BRANCH_MULTIPLIER, update_acceptance_head
from .branch_rate import branch_optimizer, set_rates
from .charge import rollout
from .charge_experiment import work_delta
from .context_experiment import make_context_head
from .continuous_data import relabel_bank
from .continuous_reward import continuous_reward
from .replay_data import CACHE_KEYS, bank_indices, cached_batch
from .replay_experiment import bank_fit
from .warmstart_data import initialize_normalizer


def train_continuous(
    backbone, prompts, config, settings, reward, bank, cache, initial, output, predictor
):
    """Match the historical control's RNG, optimizer, replay and update budget."""
    bank = relabel_bank(bank, reward, settings["penalty"])
    head = make_context_head(backbone, len(prompts), config, 0, "direct_context")
    head.load_state_dict(initial["head"])
    initial_fit = bank_fit(
        head, None, bank, cache, config, "continuous_tb", "initial", set()
    )
    offsets, diagnostics = initialize_normalizer(
        initial_fit, bank, [p["id"] for p in prompts]
    )
    normalizer = torch.nn.Parameter(offsets.to(backbone.device))
    optimizer = branch_optimizer(head, normalizer, config, BRANCH_MULTIPLIER)
    write_json(
        output / "initialization.json",
        {
            "optimizer": "fresh_adam",
            "normalizer": diagnostics,
            "normalizer_rule": "equal_pass_fail_training_bank_least_squares",
            "reward_version": "continuous_acceptance_v1",
            "penalty": settings["penalty"],
        },
    )
    fits = bank_fit(
        head, normalizer, bank, cache, config, "continuous_tb", "initial", set()
    )
    generator = torch.Generator(device=backbone.device).manual_seed(
        settings["training_seed"]
    )
    bank_generator = torch.Generator().manual_seed(settings["bank_sampling_seed"])
    started, before = time.monotonic(), dict(backbone.work)
    seen = set()
    with (
        (output / "training.jsonl").open("w") as log,
        (output / "fresh_training.jsonl").open("w") as fresh_log,
    ):
        for step in range(1, config["steps"] + 1):
            set_rates(optimizer, config, BRANCH_MULTIPLIER, step)
            indices = list(range(len(prompts)))
            fresh = rollout(
                backbone, head, prompts, indices, config, generator, retain=True
            )
            tails = [
                sequence[config["prefix_length"] :] for sequence in fresh["sequences"]
            ]
            scores = np.asarray(predictor(tails, "V3"), dtype=float)
            values = continuous_reward(tails, scores, reward, settings["penalty"])
            seen.update(tails)
            for index, tail in enumerate(tails):
                fresh_log.write(
                    json.dumps(
                        {
                            "step": step,
                            "prompt_id": prompts[index]["id"],
                            "tail": tail,
                            "isolated_tail_v3_scores": scores[index].tolist(),
                            "log_prior": float(fresh["log_prior"][index]),
                            "behavior_log_policy": float(fresh["log_policy"][index]),
                            "joint_pass": bool(values["joint_pass"][index]),
                            "log_factor": float(values["log_factor"][index]),
                            "current_log_factor": float(
                                values["current_log_factor"][index]
                            ),
                            "violations": values["violations"][index].tolist(),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            fresh_log.flush()
            selected = bank_indices(bank, len(prompts), bank_generator, step)
            stored = cached_batch(cache, selected, backbone.device)
            batch = {key: torch.cat([fresh[key], stored[key]]) for key in CACHE_KEYS}
            factors = torch.cat(
                [
                    values["log_factor"].to(backbone.device),
                    torch.tensor(
                        [bank[i]["log_factor"] for i in selected],
                        device=backbone.device,
                    ),
                ]
            )
            indices += [bank[i]["prompt_index"] for i in selected]
            stats = update_acceptance_head(
                head, normalizer, optimizer, batch, indices, config, factors
            )
            stats.update(
                step=step,
                examples=len(batch["actions"]),
                fresh_examples=len(prompts),
                fresh_joint_passes=int(values["joint_pass"].sum()),
                replay_ids=[bank[i]["id"] for i in selected],
                stored_successes=sum(bank[i]["joint_pass"] for i in selected),
                rates={group["name"]: group["lr"] for group in optimizer.param_groups},
                log_z=normalizer.detach().cpu().tolist(),
                seconds=time.monotonic() - started,
            )
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                print(
                    f"continuous_tb step {step}/{config['steps']}: TB {stats['tb_loss']:.4g}, fresh passes {stats['fresh_joint_passes']}",
                    flush=True,
                )
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                torch.save(
                    {
                        "arm": "continuous_tb",
                        "architecture": "direct_context",
                        "reward_version": "continuous_acceptance_v1",
                        "penalty": settings["penalty"],
                        "head": head.state_dict(),
                        "log_z": normalizer.detach().cpu(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": config,
                        "settings": settings,
                        "prompt_ids": [p["id"] for p in prompts],
                        "generator_state": generator.get_state(),
                        "bank_generator_state": bank_generator.get_state(),
                    },
                    output / f"step-{step}.pt",
                )
            del fresh, stored, batch
    fits += bank_fit(
        head, normalizer, bank, cache, config, "continuous_tb", "final", seen
    )
    write_json(output / "bank_fit.json", fits)
    return (
        head,
        normalizer.detach(),
        {
            "seconds": time.monotonic() - started,
            "updates": config["steps"],
            "trajectory_examples": config["steps"] * 2 * len(prompts),
            "fresh_trajectories": config["steps"] * len(prompts),
            "backbone_work": work_delta(backbone, before),
        },
    )
