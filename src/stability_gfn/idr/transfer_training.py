"""Matched offline full-support TB fits with optional snapshot gradient correction."""

import copy
import json
import time

import torch

from .branch_rate import branch_optimizer, set_rates
from .charge import path_log_probabilities
from .replay_data import cached_batch
from .transfer_head import amplification_summary


@torch.no_grad()
def make_proposal(bank, initial_fits, prompt_count):
    """Bank normalization selects training rows only; policy support stays unrestricted."""
    by_id = {r["id"]: r for r in initial_fits}
    weights = torch.zeros(len(bank), dtype=torch.float64)
    groups = []
    for prompt in range(prompt_count):
        indices = [i for i, r in enumerate(bank) if r["prompt_index"] == prompt]
        groups.append(torch.tensor(indices))
        log_policy = torch.tensor(
            [by_id[bank[i]["id"]]["log_policy"] for i in indices], dtype=torch.float64
        )
        passed = torch.tensor([bank[i]["joint_pass"] for i in indices])
        proposal = (
            0.5 * log_policy.softmax(0)
            + 0.25 * passed / passed.sum()
            + 0.25 * (~passed) / (~passed).sum()
        )
        weights[indices] = proposal / proposal.sum() / prompt_count
    return weights, groups


def batch_gradient(head, normalizer, bank, cache, indices, weights, microbatch):
    """Accumulate globally weighted microbatches, without clipping individual chunks."""
    parameters = [*head.parameters(), normalizer]
    for parameter in parameters:
        parameter.grad = None
    device = normalizer.device
    total_loss = 0.0
    for start in range(0, len(indices), microbatch):
        selected = indices[start : start + microbatch]
        batch = cached_batch(cache, selected, device)
        prompts = torch.tensor(
            [bank[i]["prompt_index"] for i in selected], device=device
        )
        factor = torch.tensor(
            [bank[i]["log_factor"] for i in selected],
            dtype=torch.float64,
            device=device,
        )
        residual = (
            normalizer[prompts].double()
            + path_log_probabilities(head, batch).double()
            - batch["log_prior"].double()
            - factor
        )
        loss = (
            weights[start : start + microbatch].to(device) * residual.square()
        ).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite full-path TB objective")
        loss.backward()
        total_loss += float(loss.detach())
    return total_loss, [p.grad.detach().clone() for p in parameters]


@torch.no_grad()
def fold_check(head, bank, cache, microbatch):
    folded = head.fold()
    max_logit, max_path = 0.0, 0.0
    gains = []
    for start in range(0, len(bank), microbatch):
        batch = cached_batch(
            cache,
            list(range(start, min(start + microbatch, len(bank)))),
            next(head.parameters()).device,
        )
        first, second = (
            head(batch["features"], batch["context"]),
            folded(batch["features"], batch["context"]),
        )
        max_logit = max(max_logit, float((first - second).abs().max()))
        max_path = max(
            max_path,
            float(
                (
                    path_log_probabilities(head, batch)
                    - path_log_probabilities(folded, batch)
                )
                .abs()
                .max()
            ),
        )
        gains.extend(
            head.amplification(batch["features"], batch["context"])
            .flatten()
            .cpu()
            .tolist()
        )
    if max_logit > 0.001 or max_path > 0.005:
        raise ValueError("Folding materially changes full-support likelihoods")
    return folded, dict(
        maximum_logit_difference=max_logit,
        maximum_path_difference=max_path,
        training_input_amplification=amplification_summary(gains),
    )


def train_transfer(
    initial,
    offsets,
    bank,
    cache,
    weights,
    groups,
    config,
    settings,
    arm,
    output,
    device,
):
    head = copy.deepcopy(initial).to(device)
    normalizer = torch.nn.Parameter(offsets.clone().to(device))
    parameters = [*head.parameters(), normalizer]
    optimizer = branch_optimizer(head, normalizer, config, 10)
    generator = torch.Generator().manual_seed(settings["bank_sampling_seed"])
    corrected = arm == "svrg_tb"
    snapshots, clipped = 0, 0
    started = time.monotonic()
    with (output / "training.jsonl").open("w") as log:
        for step in range(1, config["steps"] + 1):
            set_rates(optimizer, config, 10, step)
            if corrected and (step - 1) % settings["snapshot_interval"] == 0:
                snapshot = copy.deepcopy(head)
                snapshot_z = torch.nn.Parameter(normalizer.detach().clone())
                _, snapshot_mean = batch_gradient(
                    snapshot,
                    snapshot_z,
                    bank,
                    cache,
                    list(range(len(bank))),
                    weights,
                    settings["microbatch_size"],
                )
                snapshots += 1
            indices = []
            for group in groups:
                local = torch.multinomial(
                    weights[group],
                    settings["draws_per_prompt"],
                    replacement=True,
                    generator=generator,
                )
                indices.extend(group[local].tolist())
            sampled_weights = torch.full(
                (len(indices),), 1 / len(indices), dtype=torch.float64
            )
            loss, gradients = batch_gradient(
                head,
                normalizer,
                bank,
                cache,
                indices,
                sampled_weights,
                settings["microbatch_size"],
            )
            if corrected:
                _, old = batch_gradient(
                    snapshot,
                    snapshot_z,
                    bank,
                    cache,
                    indices,
                    sampled_weights,
                    settings["microbatch_size"],
                )
                gradients = [
                    g - previous + mean
                    for g, previous, mean in zip(gradients, old, snapshot_mean)
                ]
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, config["gradient_clip"], error_if_nonfinite=True
            )
            clipped += float(norm) > config["gradient_clip"]
            optimizer.step()
            row = dict(
                step=step,
                sampled_tb_loss=loss,
                gradient_norm_before_clip=float(norm),
                replay_ids=[bank[i]["id"] for i in indices],
                log_z=normalizer.detach().cpu().tolist(),
                rates={g["name"]: g["lr"] for g in optimizer.param_groups},
            )
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                print(
                    f"{arm} {step}/{config['steps']}: sampled TB {loss:.4g}, gradient {float(norm):.4g}",
                    flush=True,
                )
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                torch.save(
                    dict(
                        head=head.state_dict(),
                        log_z=normalizer.detach(),
                        step=step,
                        architecture="full_path_conditioned",
                        arm=arm,
                    ),
                    output / f"step-{step}.pt",
                )
    occurrences = config["steps"] * len(groups) * settings["draws_per_prompt"]
    costs = dict(
        seconds=time.monotonic() - started,
        updates=config["steps"],
        sampled_occurrences=occurrences,
        snapshot_sample_occurrences=occurrences if corrected else 0,
        full_snapshot_occurrences=snapshots * len(bank),
        snapshot_refreshes=snapshots,
        fresh_training_rollouts=0,
        clipped_fraction=clipped / config["steps"],
    )
    return head, normalizer.detach(), costs
