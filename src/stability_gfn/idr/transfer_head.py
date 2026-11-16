"""Function-preserving, bounded conditioning for full cached trajectories."""

import copy

import torch
from torch.nn import functional as F

from .context_experiment import DirectContextHead


def amplification_summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not len(values) or not torch.isfinite(values).all():
        raise ValueError("Invalid conditioning amplification measurements")
    return dict(
        count=len(values),
        mean=float(values.mean()),
        median=float(values.quantile(0.5)),
        p99=float(values.quantile(0.99)),
        maximum=float(values.max()),
    )


class TransferHead(DirectContextHead):
    """Same head function class, with a fixed invertible hidden-input transform."""

    def __init__(self, initial, transform, inverse):
        feature_size = initial.normalization.normalized_shape[0]
        context_size = initial.hidden.in_features - feature_size
        super().__init__(feature_size, context_size, 20, initial.hidden.out_features)
        self.load_state_dict(initial.state_dict())
        augmented = torch.cat([initial.hidden.weight, initial.hidden.bias[:, None]], 1)
        self.hidden = torch.nn.Linear(
            augmented.shape[1], augmented.shape[0], bias=False
        )
        with torch.no_grad():
            self.hidden.weight.copy_((augmented.double() @ inverse.double()).float())
        self.register_buffer("transform", transform.float())

    def inputs(self, features, context):
        combined = torch.cat([self.normalization(features), context], -1)
        return torch.cat([combined, combined.new_ones(*combined.shape[:-1], 1)], -1)

    def forward(self, features, context):
        # Associativity avoids transforming every cached token by a large matrix.
        effective = self.hidden.weight @ self.transform
        hidden = F.linear(self.inputs(features, context), effective)
        return self.output(F.silu(hidden)) + F.linear(
            context[..., :5], self.task_weight
        )

    def fold(self):
        feature_size = self.normalization.normalized_shape[0]
        context_size = self.hidden.in_features - feature_size - 1
        head = DirectContextHead(
            feature_size, context_size, 20, self.hidden.out_features
        )
        head = head.to(self.hidden.weight.device)
        head.normalization.load_state_dict(self.normalization.state_dict())
        head.output.load_state_dict(self.output.state_dict())
        with torch.no_grad():
            effective = self.hidden.weight @ self.transform
            head.hidden.weight.copy_(effective[:, :-1])
            head.hidden.bias.copy_(effective[:, -1])
            head.task_weight.copy_(self.task_weight)
        return head

    @torch.no_grad()
    def amplification(self, features, context):
        inputs = self.inputs(features, context)
        return (inputs @ self.transform).norm(dim=-1) / inputs.norm(dim=-1).clamp_min(
            1e-12
        )


@torch.no_grad()
def condition_head(initial, cache, maximum_gain=32.0):
    """Training-only uncentered input covariance; cap weak-direction amplification."""
    initial = copy.deepcopy(initial).cpu()
    dimension = initial.hidden.in_features + 1
    moment = torch.zeros(dimension, dimension, dtype=torch.float64)
    count = 0
    for start in range(0, len(cache["features"]), 8):
        features = cache["features"][start : start + 8]
        context = cache["context"][start : start + 8]
        combined = torch.cat([initial.normalization(features), context], -1).reshape(
            -1, dimension - 1
        )
        augmented = torch.cat([combined, torch.ones(len(combined), 1)], -1).double()
        moment += augmented.T @ augmented
        count += len(augmented)
    eigenvalues, basis = torch.linalg.eigh(moment / count)
    active = eigenvalues > eigenvalues.max() * 1e-10
    scales = torch.ones_like(eigenvalues)
    scales[active] = eigenvalues[active].clamp_min(1 / maximum_gain**2).rsqrt()
    transform = (basis * scales) @ basis.T
    inverse = (basis / scales) @ basis.T
    head = TransferHead(initial, transform, inverse)
    max_logit_error, max_path_error = 0.0, 0.0
    gains = []
    for start in range(0, len(cache["features"]), 8):
        features = cache["features"][start : start + 8]
        context = cache["context"][start : start + 8]
        logits = cache["logits"][start : start + 8]
        actions = cache["actions"][start : start + 8, :, None]
        old, new = initial(features, context), head(features, context)
        max_logit_error = max(max_logit_error, float((old - new).abs().max()))
        old_path = (logits + old).log_softmax(-1).gather(-1, actions).sum((1, 2))
        new_path = (logits + new).log_softmax(-1).gather(-1, actions).sum((1, 2))
        max_path_error = max(max_path_error, float((old_path - new_path).abs().max()))
        gains.extend(head.amplification(features, context).flatten().tolist())
    if max_logit_error > 0.001 or max_path_error > 0.005:
        raise ValueError("Conditioning changed the initial function materially")
    return head, dict(
        states=count,
        dimensions=dimension,
        active_directions=int(active.sum()),
        minimum_eigenvalue=float(eigenvalues.min()),
        maximum_eigenvalue=float(eigenvalues.max()),
        maximum_gain=maximum_gain,
        actual_maximum_scale=float(scales.max()),
        minimum_scale=float(scales.min()),
        null_relative_threshold=1e-10,
        maximum_initial_logit_difference=max_logit_error,
        maximum_initial_path_difference=max_path_error,
        training_input_amplification=amplification_summary(gains),
    )


class GenerationMonitor(torch.nn.Module):
    """Observe conditioning on fresh states without changing the folded policy."""

    def __init__(self, head, conditioned):
        super().__init__()
        self.head = head
        self.conditioned = conditioned
        self.gains = []

    @torch.no_grad()
    def forward(self, features, context):
        self.gains.extend(
            self.conditioned.amplification(features, context).flatten().cpu().tolist()
        )
        return self.head(features, context)
