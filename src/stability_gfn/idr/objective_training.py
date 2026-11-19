"""Capacity-matched GFN and reward-weighted supervised final-block updates.

Both policies consume the same scored trajectories, with independent copies of
the final ProGen block and guidance head. The supervised objective fits the
reward-weighted shared proposal distribution. It is not an exact fit to the GFN
target p0(x | prompt) * exp(log_factor(x)); no prior or proposal correction is
included in its weights.
"""

import math
import time

import torch

from .data import ALPHABET
from .lastblock_backbone import LastBlockProGen
from .lastblock_training import _check_reference, trajectory_terms

ARMS = ("gfn", "supervised")


def make_objective_optimizer(head, config, settings, block, log_z=None):
    """Use the inherited Adam rates, with a normalizer only for the GFN arm."""
    groups = [
        {
            "params": [
                parameter
                for name, parameter in head.named_parameters()
                if name != "task_weight"
            ],
            "lr": config["learning_rate"],
            "name": "base",
        },
        {
            "params": [head.task_weight],
            "lr": 10 * config["learning_rate"],
            "name": "task",
        },
        {
            "params": list(block.parameters()),
            "lr": settings["last_block_learning_rate"],
            "name": "last_block",
        },
    ]
    if log_z is not None:
        groups.append(
            {
                "params": [log_z],
                "lr": config["normalizer_learning_rate"],
                "name": "normalizer",
            }
        )
    for group in groups:
        group["starting_rate"] = group["lr"]
    return torch.optim.Adam(groups, weight_decay=0)


def _check_models(policies, heads, normalizer, optimizers):
    """Reject shared trainable parameters or changes to the original reference."""
    if any(set(mapping) != set(ARMS) for mapping in (policies, heads, optimizers)):
        raise ValueError(
            "Policies, heads and optimizers must contain both objective arms"
        )
    if any(not isinstance(policies[arm], LastBlockProGen) for arm in ARMS):
        raise ValueError("Both objective policies must adapt a ProGen final block")
    reference = policies["gfn"].reference
    if policies["supervised"].reference is not reference:
        raise ValueError("Both policies must share the original frozen reference")
    if any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in reference.model.parameters()
    ) or any(module.training for module in reference.model.modules()):
        raise ValueError(
            "Original reference must be frozen, in eval mode and gradient-free"
        )
    if normalizer.ndim != 1 or not normalizer.numel() or not normalizer.requires_grad:
        raise ValueError("GFN needs one trainable normalizer vector")
    if not torch.isfinite(normalizer).all():
        raise ValueError("GFN normalizer must be finite")
    owned = set()
    for arm in ARMS:
        expected = [*heads[arm].parameters(), *policies[arm].parameters()]
        if arm == "gfn":
            expected.append(normalizer)
        expected_ids = {id(parameter) for parameter in expected}
        actual = [
            parameter
            for group in optimizers[arm].param_groups
            for parameter in group["params"]
        ]
        actual_ids = {id(parameter) for parameter in actual}
        if (
            actual_ids != expected_ids
            or len(actual_ids) != len(actual)
            or owned & actual_ids
            or any(not parameter.requires_grad for parameter in expected)
        ):
            raise ValueError(
                "Objective optimizers must own disjoint head/block parameters"
            )
        owned.update(actual_ids)


def _supervised_weights(rows, prompt_count, config):
    """Normalize detached rewards within each full prompt batch before slicing.

    Each represented prompt receives total loss weight 1 / represented_prompts.
    The per-prompt ESS describes reward concentration; the total ESS also includes
    equal prompt weighting when the numbers of supplied trajectories differ.
    """
    prefix_length = config["prefix_length"]
    sequence_length = config["length"]
    if not 0 <= prefix_length < sequence_length:
        raise ValueError("Need a nonempty generated tail")
    prompt_rows = {}
    prompt_identity = {}
    for index, row in enumerate(rows):
        if row.get("split") == "holdout":
            raise ValueError("Held-out bank records cannot enter training updates")
        prompt = row["prompt_index"]
        if (
            not isinstance(prompt, int)
            or not 0 <= prompt < prompt_count
            or len(row["sequence"]) != sequence_length
            or row["tail"] != row["sequence"][prefix_length:]
            or set(row["sequence"]) - set(ALPHABET)
        ):
            raise ValueError("Trajectory sequence, tail or prompt identity is invalid")
        identity = (row.get("prompt_id"), row["sequence"][:prefix_length])
        if prompt in prompt_identity and prompt_identity[prompt] != identity:
            raise ValueError("Rows disagree about their prompt identity or prefix")
        prompt_identity[prompt] = identity
        prompt_rows.setdefault(prompt, []).append(index)
        if any(
            not math.isfinite(float(row[key])) for key in ("log_factor", "log_prior")
        ):
            raise ValueError(
                "Terminal property factors and reference likelihoods must be finite"
            )
    log_factors = torch.tensor(
        [row["log_factor"] for row in rows], dtype=torch.float64
    ).detach()
    weights = torch.empty_like(log_factors)
    summaries = []
    for prompt, indices in sorted(prompt_rows.items()):
        prompt_factors = log_factors[indices]
        prompt_weights = prompt_factors.softmax(0)
        weights[indices] = prompt_weights / len(prompt_rows)
        summaries.append(
            dict(
                prompt_index=prompt,
                examples=len(indices),
                weight_sum=float(prompt_weights.sum()),
                minimum_weight=float(prompt_weights.min()),
                maximum_weight=float(prompt_weights.max()),
                effective_sample_size=float(prompt_weights.square().sum().reciprocal()),
                minimum_log_factor=float(prompt_factors.min()),
                maximum_log_factor=float(prompt_factors.max()),
            )
        )
    summary = dict(
        method="per_prompt_softmax_log_factor_equal_prompt_mean",
        normalized_before_microbatch=True,
        prior_in_weights=False,
        represented_prompts=len(prompt_rows),
        loss_weight_sum=float(weights.sum()),
        prompts=summaries,
    )
    return weights, summary


def objective_update(
    policies, heads, normalizer, optimizers, rows, config, settings, step
):
    """Update both adapted policies on one shared scored batch.

    GFN uses mean squared trajectory-balance residual with the immutable original
    reference. Supervised learning minimizes reward-weighted mean-token NLL,
    averaging represented prompts equally. All rewards participate, including
    failed examples; weights are detached and never renormalized per microbatch.
    Each policy graph is backpropagated separately. Both gradient sets are checked
    and clipped before either optimizer steps, so neither policy changes the
    other arm's examples or loss during this update.
    """
    if not rows or settings["microbatch_size"] <= 0 or step < 1:
        raise ValueError("Need nonempty rows, positive microbatch size and step")
    _check_models(policies, heads, normalizer, optimizers)
    prompt_count = normalizer.numel()
    weights, weighting = _supervised_weights(rows, prompt_count, config)
    tail_length = config["length"] - config["prefix_length"]
    decay = (
        config["final_lr_factor"]
        if step > int(config["steps"] * config["decay_fraction"])
        else 1.0
    )
    totals = {
        arm: dict(loss=0.0, mean_log_q_minus_log_p0=0.0, update_seconds=0.0)
        for arm in ARMS
    }
    for arm in ARMS:
        optimizers[arm].zero_grad(set_to_none=True)
        for group in optimizers[arm].param_groups:
            group["lr"] = group["starting_rate"] * decay

    for arm in ARMS:
        started = time.monotonic()
        policy = policies[arm]
        for start in range(0, len(rows), settings["microbatch_size"]):
            batch = rows[start : start + settings["microbatch_size"]]
            sequences = [row["sequence"][:-1] for row in batch]
            if arm == "gfn":
                features, logits, reference_features, reference_logits = (
                    policy.forward_with_reference(sequences)
                )
                _, log_prior, _ = trajectory_terms(
                    reference_features,
                    reference_logits,
                    None,
                    batch,
                    config,
                    prompt_count,
                )
                _check_reference(log_prior, batch)
            else:
                features, logits = policy.forward(sequences)
                # GFN has already checked every stored p0 against the shared
                # reference. This value is a diagnostic only for supervised fitting.
                log_prior = torch.tensor(
                    [row["log_prior"] for row in batch], device=policy.device
                )
            log_q, _, _ = trajectory_terms(
                features, logits, heads[arm], batch, config, prompt_count
            )
            if arm == "gfn":
                indices = torch.tensor(
                    [row["prompt_index"] for row in batch], device=policy.device
                )
                factors = torch.tensor(
                    [row["log_factor"] for row in batch],
                    dtype=torch.float64,
                    device=policy.device,
                ).detach()
                residual = (
                    normalizer[indices].double()
                    + log_q.double()
                    - log_prior.double()
                    - factors
                )
                loss = residual.square().sum() / len(rows)
            else:
                batch_weights = weights[start : start + len(batch)].to(policy.device)
                loss = -(batch_weights * log_q.double() / tail_length).sum()
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {arm} objective loss")
            totals[arm]["loss"] += float(loss.detach())
            totals[arm]["mean_log_q_minus_log_p0"] += float(
                (log_q.detach() - log_prior).sum() / len(rows)
            )
            loss.backward()

        head_parameters = list(heads[arm].parameters())
        if arm == "gfn":
            head_parameters.append(normalizer)
        head_norm = torch.nn.utils.clip_grad_norm_(
            head_parameters, config["gradient_clip"], error_if_nonfinite=True
        )
        block_norm = torch.nn.utils.clip_grad_norm_(
            list(policies[arm].parameters()),
            settings["block_gradient_clip"],
            error_if_nonfinite=True,
        )
        totals[arm].update(
            step=step,
            examples=len(rows),
            example_ids=[row["id"] for row in rows],
            gradient_norm_before_clip=float(head_norm),
            head_gradient_norm_before_clip=float(head_norm),
            block_gradient_norm_before_clip=float(block_norm),
            rates={
                group["name"]: group["lr"] for group in optimizers[arm].param_groups
            },
        )
        # Reading both norms above completes this arm's accumulated gradients
        # before its elapsed time is recorded, including on an accelerator.
        totals[arm]["update_seconds"] = time.monotonic() - started
    _check_models(policies, heads, normalizer, optimizers)
    for arm in ARMS:
        started = time.monotonic()
        optimizers[arm].step()
        parameters = [*heads[arm].parameters(), *policies[arm].parameters()]
        if arm == "gfn":
            parameters.append(normalizer)
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise ValueError("Nonfinite parameter after the objective update")
        totals[arm]["update_seconds"] += time.monotonic() - started
    totals["gfn"].update(
        tb_loss=totals["gfn"]["loss"], log_z=normalizer.detach().cpu().tolist()
    )
    totals["supervised"].update(
        weighted_nll=totals["supervised"]["loss"],
        effective_sample_size=float(weights.square().sum().reciprocal()),
        weighting=weighting,
    )
    return totals
