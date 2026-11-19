"""Paired GFlowNet updates with a fixed reference and one adapted ProGen block."""

from functools import lru_cache

import torch
from tokenizers import Tokenizer

from ..backbone import TOKENIZER_PATH
from .charge import context_features
from .data import ALPHABET
from .lastblock_backbone import LastBlockProGen

ARMS = ("frozen_tb", "lastblock_tb")


@lru_cache(maxsize=1)
def _residue_ids():
    """Use the same vendored one-residue token mapping as FrozenProGen."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    encoded = [tokenizer.encode(residue).ids for residue in ALPHABET]
    if any(len(ids) != 1 for ids in encoded):
        raise ValueError("Canonical residues must each occupy one token")
    return tuple(ids[0] for ids in encoded)


def trajectory_terms(features, logits, head, rows, config, prompt_count):
    """Return full-support log q, base log probability and canonical token mass.

    Inputs contain the forward-direction token at index zero. State at
    ``prefix_length`` predicts the first generated residue. ``features`` and
    ``logits`` may include the final input residue, which is never used to predict
    itself. The base is the supplied backbone: only reference logits define p0.
    Returned vectors have one entry per trajectory; canonical mass is averaged
    across its generated positions before renormalizing over the 20 residues.
    """
    prefix_length = config["prefix_length"]
    sequence_length = config["length"]
    tail_length = sequence_length - prefix_length
    if not rows or tail_length <= 0:
        raise ValueError("Need trajectories with a nonempty generated tail")
    if (
        features.ndim != 3
        or logits.ndim != 3
        or features.shape[:2] != logits.shape[:2]
        or features.shape[0] != len(rows)
        or features.shape[1] < sequence_length
    ):
        raise ValueError("Full-sequence features/logits do not match the trajectories")
    if any(
        len(row["sequence"]) != sequence_length
        or row["tail"] != row["sequence"][prefix_length:]
        or set(row["sequence"]) - set(ALPHABET)
        or not 0 <= row["prompt_index"] < prompt_count
        for row in rows
    ):
        raise ValueError("Trajectory sequence, tail or prompt identity is invalid")
    features = features[:, prefix_length:sequence_length]
    logits = logits[:, prefix_length:sequence_length]
    canonical_logits = logits[..., list(_residue_ids())]
    correction = 0
    if head is not None:
        # Build these small deterministic descriptors on CPU, then transfer once
        # instead of launching a transfer for every sequence position.
        context = torch.stack(
            [
                context_features(
                    [row["tail"][:position] for row in rows],
                    [row["prompt_index"] for row in rows],
                    prompt_count,
                    tail_length,
                    config["reward"]["block_targets"],
                    "cpu",
                )
                for position in range(tail_length)
            ],
            dim=1,
        ).to(features.device)
        correction = head(features, context)
    actions = torch.tensor(
        [[ALPHABET.index(residue) for residue in row["tail"]] for row in rows],
        device=features.device,
    )
    log_policy = (
        (canonical_logits + correction)
        .log_softmax(-1)
        .gather(-1, actions[..., None])
        .squeeze(-1)
        .sum(-1)
    )
    log_base = (
        canonical_logits.log_softmax(-1)
        .gather(-1, actions[..., None])
        .squeeze(-1)
        .sum(-1)
    )
    canonical_mass = (
        (canonical_logits.logsumexp(-1) - logits.logsumexp(-1)).exp().mean(-1)
    )
    if not all(
        torch.isfinite(value).all() for value in (log_policy, log_base, canonical_mass)
    ):
        raise ValueError("Nonfinite trajectory likelihood or canonical mass")
    return log_policy, log_base, canonical_mass


def make_optimizer(head, log_z, config, settings, block=None):
    """Use inherited head rates and a separately declared copied-block rate."""
    groups = [
        {
            "params": [
                p for name, p in head.named_parameters() if name != "task_weight"
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
            "params": [log_z],
            "lr": config["normalizer_learning_rate"],
            "name": "normalizer",
        },
    ]
    if block is not None:
        groups.append(
            {
                "params": list(block.parameters()),
                "lr": settings["last_block_learning_rate"],
                "name": "last_block",
            }
        )
    for group in groups:
        group["starting_rate"] = group["lr"]
    return torch.optim.Adam(groups, weight_decay=0)


def _check_reference(log_prior, rows):
    expected = log_prior.new_tensor([row["log_prior"] for row in rows])
    # Batched teacher forcing and incremental decoding differ slightly in fp32.
    if not torch.allclose(log_prior.detach(), expected, atol=0.003, rtol=1e-5):
        raise ValueError(
            "Original-reference likelihood disagrees with stored trajectory"
        )


def paired_update(
    adapted, heads, normalizers, optimizers, rows, config, settings, step
):
    """One matched update, averaging each arm's TB loss over all supplied rows.

    The shared frozen lower pass creates no gradient graph. Reference-head and
    adapted-head/block losses can therefore be summed for one backward call per
    microbatch while remaining distinct objectives with distinct optimizers.
    Clip head/normalizer and copied-block gradients separately after accumulation.
    """
    if not rows or settings["microbatch_size"] <= 0 or step < 1:
        raise ValueError("Need nonempty rows, positive microbatch size and step")
    if any(row.get("split") == "holdout" for row in rows):
        raise ValueError("Held-out bank records cannot enter training updates")
    prompt_count = normalizers[ARMS[0]].numel()
    if normalizers[ARMS[1]].numel() != prompt_count:
        raise ValueError("Both arms must use the same prompt normalizers")
    decay = (
        config["final_lr_factor"]
        if step > int(config["steps"] * config["decay_fraction"])
        else 1.0
    )
    totals = {arm: {"tb_loss": 0.0, "mean_log_q_minus_log_p0": 0.0} for arm in ARMS}
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = group["starting_rate"] * decay

    for start in range(0, len(rows), settings["microbatch_size"]):
        batch = rows[start : start + settings["microbatch_size"]]
        policy_features, policy_logits, reference_features, reference_logits = (
            adapted.forward_with_reference([row["sequence"][:-1] for row in batch])
        )
        frozen_q, log_prior, _ = trajectory_terms(
            reference_features,
            reference_logits,
            heads["frozen_tb"],
            batch,
            config,
            prompt_count,
        )
        adapted_q, _, _ = trajectory_terms(
            policy_features,
            policy_logits,
            heads["lastblock_tb"],
            batch,
            config,
            prompt_count,
        )
        _check_reference(log_prior, batch)
        indices = torch.tensor(
            [row["prompt_index"] for row in batch], device=adapted.device
        )
        log_factor = torch.tensor(
            [row["log_factor"] for row in batch],
            dtype=torch.float64,
            device=adapted.device,
        )
        if not torch.isfinite(log_factor).all():
            raise ValueError("Terminal property factors must be finite")
        losses = []
        for arm, log_q in zip(ARMS, (frozen_q, adapted_q)):
            residual = (
                normalizers[arm][indices].double()
                + log_q.double()
                - log_prior.double()
                - log_factor
            )
            loss = residual.square().sum() / len(rows)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite trajectory-balance loss")
            losses.append(loss)
            totals[arm]["tb_loss"] += float(loss.detach())
            totals[arm]["mean_log_q_minus_log_p0"] += float(
                (log_q.detach() - log_prior).sum() / len(rows)
            )
        sum(losses).backward()

    block_norm = torch.nn.utils.clip_grad_norm_(
        list(adapted.parameters()),
        settings["block_gradient_clip"],
        error_if_nonfinite=True,
    )
    for arm in ARMS:
        head_norm = torch.nn.utils.clip_grad_norm_(
            [*heads[arm].parameters(), normalizers[arm]],
            config["gradient_clip"],
            error_if_nonfinite=True,
        )
        totals[arm].update(
            step=step,
            examples=len(rows),
            example_ids=[row["id"] for row in rows],
            gradient_norm_before_clip=float(head_norm),
            head_gradient_norm_before_clip=float(head_norm),
            block_gradient_norm_before_clip=(
                float(block_norm) if arm == "lastblock_tb" else None
            ),
            rates={
                group["name"]: group["lr"] for group in optimizers[arm].param_groups
            },
        )
    # Check both gradient sets before either update, then step the disjoint models.
    for arm in ARMS:
        optimizers[arm].step()
        parameters = [*heads[arm].parameters(), normalizers[arm]]
        if arm == "lastblock_tb":
            parameters += list(adapted.parameters())
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise ValueError("Nonfinite parameter after the paired update")
        totals[arm]["log_z"] = normalizers[arm].detach().cpu().tolist()
    return totals


@torch.no_grad()
def bank_fits(
    backbone, head, log_z, bank, config, prompts, arm, fresh_seen, microbatch_size
):
    """Recompute common-bank diagnostics; no cached adapted states or fitting.

    A stored held-out path remains diagnostic data even when raw online generation
    happens to visit it. Record that overlap explicitly without training on a
    held-out bank record. These finite-bank residuals are not global target TV.
    """
    if microbatch_size <= 0:
        raise ValueError("Microbatch size must be positive")
    results = []
    for start in range(0, len(bank), microbatch_size):
        rows = bank[start : start + microbatch_size]
        sequences = [row["sequence"][:-1] for row in rows]
        if isinstance(backbone, LastBlockProGen):
            features, logits, reference_features, reference_logits = (
                backbone.forward_with_reference(sequences)
            )
            _, log_prior, _ = trajectory_terms(
                reference_features, reference_logits, None, rows, config, len(prompts)
            )
        else:
            features, logits = backbone.forward(sequences)
            _, log_prior, _ = trajectory_terms(
                features, logits, None, rows, config, len(prompts)
            )
        log_policy, _, _ = trajectory_terms(
            features, logits, head, rows, config, len(prompts)
        )
        _check_reference(log_prior, rows)
        for index, row in enumerate(rows):
            normalizer = float(log_z[row["prompt_index"]])
            policy_value, reference_value = (
                float(log_policy[index]),
                float(log_prior[index]),
            )
            results.append(
                dict(
                    id=row["id"],
                    arm=arm,
                    stage="final",
                    prompt_id=row["prompt_id"],
                    split=row["split"],
                    joint_pass=row["joint_pass"],
                    fresh_training_overlap=row["tail"] in fresh_seen,
                    log_policy=policy_value,
                    log_prior=reference_value,
                    log_z=normalizer,
                    tb_residual=normalizer
                    + policy_value
                    - reference_value
                    - row["log_factor"],
                    token_nll=-policy_value / len(row["tail"]),
                )
            )
    return results
