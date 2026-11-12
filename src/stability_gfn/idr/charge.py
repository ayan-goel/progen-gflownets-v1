"""Two-block charge rewards and on-policy guidance over open ProGen continuations."""

import torch
from torch.nn import functional as F

from ..guidance import GuidanceHead, trajectory_balance
from .data import ALPHABET


def charge_values(tails, device="cpu"):
    """Nominal residue charges; no termini, histidine ionization or PTMs."""
    if not tails or len({len(t) for t in tails}) != 1:
        raise ValueError("Need equal-length nonempty tails")
    if not tails[0] or any(set(t) - set(ALPHABET) for t in tails):
        raise ValueError("Need canonical residues")
    return torch.tensor(
        [[1 if a in "KR" else -1 if a in "DE" else 0 for a in t] for t in tails],
        dtype=torch.float32,
        device=device,
    )


def charge_reward(tails, reward, device="cpu"):
    """Continuous negative mismatch, plus separately declared binary criteria.

    Each block has its own signed mean charge. FCR discourages achieving these
    means with extremely charged or almost entirely neutral sequences.
    """
    charges = charge_values(tails, device)
    if charges.shape[1] % 2:
        raise ValueError("The two-block task requires an even tail length")
    midpoint = charges.shape[1] // 2
    blocks = torch.stack(
        [charges[:, :midpoint].mean(1), charges[:, midpoint:].mean(1)], dim=1
    )
    targets = blocks.new_tensor(reward["block_targets"])
    fcr = charges.abs().mean(1)
    error = blocks - targets
    score = -(error / reward["block_scale"]).square().mean(1)
    score -= ((fcr - reward["fcr_target"]) / reward["fcr_scale"]).square()
    success = (error.abs() <= reward["block_tolerance"]).all(1)
    success &= (fcr - reward["fcr_target"]).abs() <= reward["fcr_tolerance"]
    return score, blocks, fcr, success


def make_head(backbone, prompt_count, hidden_size):
    """Reuse the tested residual head; zero output recovers the same masked prior."""
    return GuidanceHead(backbone.hidden_size, prompt_count + 5, 20, hidden_size).to(
        backbone.device
    )


def context_features(tails, prompt_indices, prompt_count, tail_length, targets, device):
    """Expose progress and accumulated charge, not future reward labels.

    Cumulative sums use final block lengths as denominators. Prefix residues are
    excluded. Frozen ProGen states still encode the complete supplied context.
    """
    position = len(tails[0])
    midpoint = tail_length // 2
    values = []
    for tail in tails:
        charges = [1 if a in "KR" else -1 if a in "DE" else 0 for a in tail]
        values.append(
            [
                position / tail_length,
                sum(charges[:midpoint]) / midpoint,
                sum(charges[midpoint:]) / midpoint,
                sum(abs(q) for q in charges) / tail_length,
                targets[int(position >= midpoint)],
            ]
        )
    scalar = torch.tensor(values, dtype=torch.float32, device=device)
    identities = F.one_hot(
        torch.as_tensor(prompt_indices, device=device), prompt_count
    ).float()
    return torch.cat([scalar, identities], dim=-1)


@torch.no_grad()
def rollout(backbone, head, prompts, prompt_indices, config, generator, retain=False):
    """Sample fresh full trajectories, caching detached states only for one update.

    ProGen never has a gradient graph. Reusing the sampled states for the head's
    likelihood computation is exact because the backbone is fixed. The head does
    not change until the entire rollout is finished.
    """
    prefix_length = config["prefix_length"]
    tail_length = config["length"] - prefix_length
    prefixes = [prompts[i]["sequence"][:prefix_length] for i in prompt_indices]
    sequences = prefixes.copy()
    token_ids = torch.tensor(backbone.residue_ids(ALPHABET), device=backbone.device)
    next_input = backbone.encode(prefixes)
    cache = None
    log_prior = torch.zeros(len(sequences), device=backbone.device)
    log_policy = torch.zeros_like(log_prior)
    canonical_mass = torch.zeros_like(log_prior)
    stored = {name: [] for name in ("features", "context", "logits", "actions")}
    for _ in range(tail_length):
        features, logits, cache = backbone.step(next_input, cache)
        base_logits = logits[:, token_ids]
        context = context_features(
            [s[prefix_length:] for s in sequences],
            prompt_indices,
            len(prompts),
            tail_length,
            config["reward"]["block_targets"],
            backbone.device,
        )
        correction = 0 if head is None else head(features, context)
        base_logps = base_logits.log_softmax(-1)
        policy_logps = (base_logits + correction).log_softmax(-1)
        if not torch.isfinite(policy_logps).all():
            raise ValueError("Nonfinite policy probabilities")
        actions = torch.multinomial(policy_logps.exp(), 1, generator=generator)
        log_prior += base_logps.gather(1, actions).squeeze(1)
        log_policy += policy_logps.gather(1, actions).squeeze(1)
        canonical_mass += (base_logits.logsumexp(-1) - logits.logsumexp(-1)).exp()
        if retain:
            for name, value in zip(
                stored, (features, context, base_logits, actions.squeeze(1))
            ):
                stored[name].append(value.detach())
        for index, action in enumerate(actions.squeeze(1).tolist()):
            sequences[index] += ALPHABET[action]
        next_input = token_ids[actions]
    result = {
        "sequences": sequences,
        "log_prior": log_prior,
        "log_policy": log_policy,
        "canonical_mass": canonical_mass / tail_length,
    }
    if retain:
        result.update(
            {name: torch.stack(values, dim=1) for name, values in stored.items()}
        )
    return result


def path_log_probabilities(head, batch):
    """Differentiable current-policy path likelihood on detached sampled states."""
    logits = batch["logits"] + head(batch["features"], batch["context"])
    return (
        logits.log_softmax(-1)
        .gather(-1, batch["actions"][..., None])
        .squeeze(-1)
        .sum(-1)
    )


def update_head(head, log_z, optimizer, batch, prompt_indices, config):
    """One unique-path trajectory-balance update; no backprop through sampling."""
    tails = [s[config["prefix_length"] :] for s in batch["sequences"]]
    scores, _, _, success = charge_reward(tails, config["reward"], log_z.device)
    log_q = path_log_probabilities(head, batch)
    indices = torch.tensor(prompt_indices, device=log_z.device)
    loss = trajectory_balance(
        log_q, batch["log_prior"], scores, log_z[indices], config["beta"]
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite trajectory-balance loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [*head.parameters(), log_z], config["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "tb_loss": float(loss.detach()),
        "mean_score": float(scores.mean()),
        "charge_success": float(success.float().mean()),
        "gradient_norm_before_clip": float(norm),
        "mean_log_q_minus_log_p0": float((log_q.detach() - batch["log_prior"]).mean()),
    }
