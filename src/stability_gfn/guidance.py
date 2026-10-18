"""Residual token guidance over a frozen, locally constrained ProGen baseline."""

import torch
from torch import nn

from .environment import PrefixTree


class GuidanceHead(nn.Module):
    """Small shared correction head; zero output exactly recovers the prior."""

    def __init__(self, feature_size, context_size, actions, hidden_size=128):
        super().__init__()
        self.normalization = nn.LayerNorm(feature_size)
        self.hidden = nn.Linear(feature_size + context_size, hidden_size)
        self.output = nn.Linear(hidden_size, actions)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, context):
        combined = torch.cat([self.normalization(features), context], dim=-1)
        return self.output(torch.nn.functional.silu(self.hidden(combined)))


class CachedTask:
    """Frozen features and the same legal-action normalization for every method."""

    def __init__(self, tree: PrefixTree, features, logits, scores, device="cpu"):
        self.tree = tree
        self.device = torch.device(device)
        self.features = features.detach().to(device)
        self.logits = logits.detach().to(device)
        self.context = torch.tensor(tree.context_features(), device=device)
        self.legal = torch.tensor(tree.legal, device=device)
        self.paths = torch.tensor(tree.paths, device=device)
        self.actions = torch.tensor(tree.actions, device=device)
        self.children = torch.tensor(tree.children, device=device)
        self.leaves = torch.tensor(tree.leaves, device=device)
        self.scores = torch.as_tensor(
            scores, dtype=torch.float32, device=device
        ).detach()
        if (
            len(self.scores) != len(tree.sequences)
            or not torch.isfinite(self.scores).all()
        ):
            raise ValueError("Need one finite fixed score per sequence")
        if features.shape[0] != len(tree.prefixes) or logits.shape != self.legal.shape:
            raise ValueError("Cached features/logits do not match the task")
        if (
            not torch.isfinite(self.features).all()
            or not torch.isfinite(self.logits).all()
        ):
            raise ValueError("Non-finite cached model outputs")
        self.base_conditionals = self.logits.masked_fill(
            ~self.legal, -torch.inf
        ).log_softmax(-1)
        self.base_log_probabilities = self.base_conditionals[
            self.paths, self.actions
        ].sum(-1)

    def head(self, hidden_size=128):
        return GuidanceHead(
            self.features.shape[1],
            self.context.shape[1],
            self.logits.shape[1],
            hidden_size,
        ).to(self.device)

    def conditionals(self, head, nodes):
        """Use float32 log-softmax after masking, including probability-one forced edges."""
        correction = (
            0 if head is None else head(self.features[nodes], self.context[nodes])
        )
        return (
            (self.logits[nodes] + correction)
            .masked_fill(~self.legal[nodes], -torch.inf)
            .log_softmax(-1)
        )

    @torch.no_grad()
    def sample(self, head, count, generator):
        """Sample complete trajectories; no gradients through discrete choices."""
        nodes = torch.zeros(count, dtype=torch.long, device=self.device)
        for position in range(len(self.tree.context.reference)):
            probabilities = self.conditionals(head, nodes).exp()
            actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(
                -1
            )
            if position + 1 == len(self.tree.context.reference):
                return self.leaves[nodes, actions]
            nodes = self.children[nodes, actions]
            if (nodes < 0).any():
                raise RuntimeError("Tree traversal reached an invalid edge")

    def log_probabilities(self, head, leaves):
        """Recompute differentiable likelihoods of sampled paths under current policy."""
        nodes = self.paths[leaves]
        actions = self.actions[leaves]
        log_conditionals = self.conditionals(head, nodes)
        return log_conditionals.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)

    @torch.no_grad()
    def all_log_probabilities(self, head, batch_size=512):
        """Evaluate each prefix once rather than repeatedly evaluating shared paths."""
        conditionals = []
        for start in range(0, len(self.features), batch_size):
            nodes = torch.arange(
                start, min(start + batch_size, len(self.features)), device=self.device
            )
            conditionals.append(self.conditionals(head, nodes))
        all_conditionals = torch.cat(conditionals)
        return all_conditionals[self.paths, self.actions].sum(-1)


def trajectory_balance(log_q, log_p0, scores, log_normalizer, beta):
    """Unique-path TB; backward probability is one and reward stays in log space."""
    residual = log_normalizer + log_q - log_p0 - beta * scores
    return residual.square().mean()
