"""Prior-factored subtrajectory balance for unique-parent sequence trees.

The learned scalar is log F(prefix) - log p0(prefix), not a new reward.
Root flow is log Z and terminal flow is fixed to beta times the charge score.
See docs/experiments/idr_subtb_2026-11-13.md for the cancellation of the prior.
"""

import math

import torch

from ..guidance import GuidanceHead
from .charge import charge_reward


def segment_residuals(log_ratios, flows):
    """Return every i<j residual, ordered by torch.triu_indices, and its length.

    log_ratios has shape [batch, T]; flows has shape [batch, T+1]. Prefix
    cumulative sums let all T(T+1)/2 constraints reuse one sampled trajectory.
    """
    if log_ratios.ndim != 2 or flows.shape != (
        log_ratios.shape[0],
        log_ratios.shape[1] + 1,
    ):
        raise ValueError("Need B-by-T log ratios and B-by-(T+1) flow values")
    length = log_ratios.shape[1]
    if length < 1:
        raise ValueError("Need at least one action")
    starts, ends = torch.triu_indices(
        length + 1, length + 1, offset=1, device=log_ratios.device
    )
    cumulative = torch.cat(
        [log_ratios.new_zeros((len(log_ratios), 1)), log_ratios.cumsum(-1)], dim=1
    )
    residuals = flows[:, starts] - flows[:, ends]
    residuals = residuals + cumulative[:, ends] - cumulative[:, starts]
    return residuals, ends - starts


def subtrajectory_balance(log_ratios, flows, weighting=1.0):
    """Normalized lambda**length weighting per segment, then average trajectories."""
    if not math.isfinite(weighting) or weighting <= 0:
        raise ValueError("SubTB lambda must be finite and positive")
    residuals, lengths = segment_residuals(log_ratios, flows)
    weights = (lengths.to(log_ratios.dtype) * math.log(weighting)).softmax(0)
    return (residuals.square() * weights).sum(-1).mean()


def make_flow_head(backbone, prompt_count, config, seed):
    """Separate scalar MLP; independent initialization leaves the token head intact."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20000 + seed)
        flow = GuidanceHead(
            backbone.hidden_size, prompt_count + 5, 1, config["hidden_size"]
        )
    return flow.to(backbone.device)


def update_subtb(head, flow, log_z, optimizer, batch, indices, config, weighting):
    """Backpropagate segment constraints through policy and interior flows only."""
    tails = [s[config["prefix_length"] :] for s in batch["sequences"]]
    scores, _, _, success = charge_reward(tails, config["reward"], log_z.device)
    actions = batch["actions"][..., None]
    base_logps = batch["logits"].log_softmax(-1).gather(-1, actions).squeeze(-1)
    policy_logits = batch["logits"] + head(batch["features"], batch["context"])
    policy_logps = policy_logits.log_softmax(-1).gather(-1, actions).squeeze(-1)
    log_ratios = policy_logps - base_logps
    # Stored features at t describe the prefix BEFORE action t. The root and
    # terminal boundaries are never supplied by the auxiliary network.
    interior = flow(batch["features"][:, 1:], batch["context"][:, 1:]).squeeze(-1)
    root = log_z[torch.as_tensor(indices, device=log_z.device)]
    terminal = config["beta"] * scores
    flows = torch.cat([root[:, None], interior, terminal[:, None]], dim=1)
    loss = subtrajectory_balance(log_ratios, flows, weighting)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite subtrajectory-balance loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Keep the historical policy+normalizer clipping rule. Clip the new flow
    # network separately so its norm cannot consume the policy's clipping budget.
    policy_norm = torch.nn.utils.clip_grad_norm_(
        [*head.parameters(), log_z], config["gradient_clip"], error_if_nonfinite=True
    )
    flow_norm = torch.nn.utils.clip_grad_norm_(
        flow.parameters(), config["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    with torch.no_grad():
        full_residual = root + log_ratios.sum(-1) - terminal
    return {
        "loss": float(loss.detach()),
        "tb_loss": float(full_residual.square().mean()),
        "mean_score": float(scores.mean()),
        "charge_success": float(success.float().mean()),
        "gradient_norm_before_clip": float(policy_norm),
        "flow_gradient_norm_before_clip": float(flow_norm),
        "mean_log_q_minus_log_p0": float(log_ratios.detach().sum(-1).mean()),
    }
