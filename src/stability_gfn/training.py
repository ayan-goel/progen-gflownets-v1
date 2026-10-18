"""On-policy trajectory balance and Monte Carlo future-reward supervision."""

import time

import torch

from .guidance import trajectory_balance
from .metrics import distribution_metrics


def validate_training_options(config):
    """Validate optional diagnostics and a single, predeclared rate reduction."""
    if not isinstance(config.get("monitor_exact_distribution", False), bool):
        raise ValueError("monitor_exact_distribution must be a boolean")
    schedule = config.get("lr_decay")
    if schedule is None:
        return
    if not isinstance(schedule, dict) or set(schedule) != {"after_fraction", "factor"}:
        raise ValueError("lr_decay requires after_fraction and factor")
    for name in ("after_fraction", "factor"):
        value = schedule[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 < value < 1
        ):
            raise ValueError(
                f"lr_decay.{name} must be finite and strictly between 0 and 1"
            )
    boundary = int(config["steps"] * schedule["after_fraction"])
    if not 1 <= boundary < config["steps"]:
        raise ValueError("lr_decay must leave at least one update in each phase")


def _set_learning_rates(optimizer, config, step):
    """Updates through floor(steps*fraction) use the original rates; later ones decay.

    Set from configured rates, rather than repeatedly multiplying, and preserve
    Adam moments across the transition. TB's second group is the learned log Z.
    """
    schedule = config.get("lr_decay")
    multiplier = 1.0
    if schedule is not None and step > int(
        config["steps"] * schedule["after_fraction"]
    ):
        multiplier = schedule["factor"]
    optimizer.param_groups[0]["lr"] = config["learning_rate"] * multiplier
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = (
            config["normalizer_learning_rate"] * multiplier
        )


def _should_log(step, config):
    """Always observe both sides of a rate change, as well as normal checkpoints."""
    schedule = config.get("lr_decay")
    at_transition = False
    if schedule is not None:
        boundary = int(config["steps"] * schedule["after_fraction"])
        at_transition = step in (boundary, boundary + 1)
    return (
        step == 1
        or step % config["log_every"] == 0
        or step == config["steps"]
        or at_transition
    )


@torch.no_grad()
def _monitor_distribution(task, head, beta, log_z=None):
    """Observe the finite target without labels/metrics feeding training or RNG.

    Synchronize outstanding GPU training before timing this separate evaluation.
    CPU copies inside distribution_metrics finish the observed work before return.
    """
    if task.device.type == "cuda":
        torch.cuda.synchronize(task.device)
    started = time.perf_counter()
    metrics = distribution_metrics(
        task.all_log_probabilities(head),
        task.base_log_probabilities,
        task.scores,
        beta,
        log_z,
    )
    return metrics, time.perf_counter() - started


def train_tb(task, config, seed, progress=None):
    """Train only residual guidance and one scalar log Z; no replay in this pilot."""
    validate_training_options(config)
    torch.manual_seed(seed)
    generator = torch.Generator(device=task.device).manual_seed(seed)
    head = task.head(config["hidden_size"])
    log_normalizer = torch.nn.Parameter(torch.zeros((), device=task.device))
    optimizer = torch.optim.Adam(
        [
            {"params": head.parameters(), "lr": config["learning_rate"]},
            {"params": [log_normalizer], "lr": config["normalizer_learning_rate"]},
        ]
    )
    history = []
    monitoring_seconds = 0.0
    monitoring_calls = 0
    started = time.perf_counter()
    if config.get("monitor_exact_distribution", False):
        metrics, elapsed = _monitor_distribution(
            task, head, config["beta"], log_normalizer
        )
        monitoring_seconds += elapsed
        monitoring_calls += 1
        history.append(
            {
                "step": 0,
                "loss": None,
                "log_z": log_normalizer.item(),
                "gradient_norm": None,
                "learning_rate": config["learning_rate"],
                "normalizer_learning_rate": config["normalizer_learning_rate"],
                "distribution": metrics,
            }
        )
        if progress is not None:
            progress(history[-1])
    for step in range(1, config["steps"] + 1):
        _set_learning_rates(optimizer, config, step)
        leaves = task.sample(head, config["batch_size"], generator)
        loss = trajectory_balance(
            task.log_probabilities(head, leaves),
            task.base_log_probabilities[leaves],
            task.scores[leaves],
            log_normalizer,
            config["beta"],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite TB loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(head.parameters()) + [log_normalizer],
            config["gradient_clip"],
            error_if_nonfinite=True,
        )
        optimizer.step()
        if _should_log(step, config):
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "log_z": log_normalizer.item(),
                    "gradient_norm": gradient_norm.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "normalizer_learning_rate": optimizer.param_groups[1]["lr"],
                }
            )
            if config.get("monitor_exact_distribution", False):
                metrics, elapsed = _monitor_distribution(
                    task, head, config["beta"], log_normalizer
                )
                monitoring_seconds += elapsed
                monitoring_calls += 1
                history[-1]["distribution"] = metrics
            if progress is not None:
                progress(history[-1])
    total_seconds = time.perf_counter() - started
    return (
        head,
        log_normalizer.detach(),
        history,
        {
            "training_seconds": total_seconds,
            "training_seconds_excluding_exact_monitoring": total_seconds
            - monitoring_seconds,
            "exact_monitoring_seconds": monitoring_seconds,
            "exact_monitoring_calls": monitoring_calls,
            "exact_monitoring_cached_score_reads": monitoring_calls * len(task.scores),
            "cached_terminal_reward_lookups": config["steps"] * config["batch_size"],
            "new_scorer_forward_calls": 0,
            "backbone_calls_during_training": 0,
        },
    )


def train_supervised(task, config, seed, progress=None):
    """Estimate log E[exp(beta*s(X)) | prefix, action] from baseline rollouts.

    This is a FUDGE-style reward-potential adaptation, not the original binary
    classifier. Unvisited edges have no supervised loss. Monte Carlo labels use
    predicted terminal reward only, never withheld experimental measurements.
    """
    validate_training_options(config)
    torch.manual_seed(seed)
    generator = torch.Generator(device=task.device).manual_seed(seed)
    head = task.head(config["hidden_size"])
    started = time.perf_counter()
    leaves = task.sample(None, config["rollouts"], generator)
    action_count = task.logits.shape[1]
    edge_ids = (task.paths[leaves] * action_count + task.actions[leaves]).flatten()
    log_rewards = (config["beta"] * task.scores[leaves]).double()
    # Common offset avoids overflow without changing conditional log means.
    offset = log_rewards.max()
    weights = (log_rewards - offset).exp().repeat_interleave(task.paths.shape[1])
    sums = torch.zeros(task.legal.numel(), dtype=torch.float64, device=task.device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, edge_ids, weights)
    counts.index_add_(0, edge_ids, torch.ones_like(weights))
    visited = counts > 0
    targets = torch.zeros_like(sums)
    targets[visited] = (sums[visited] / counts[visited]).log() + offset
    if not torch.isfinite(targets[visited]).all():
        raise FloatingPointError(
            "Future-reward targets underflowed; inspect reward scale"
        )
    targets = targets.float()
    optimizer = torch.optim.Adam(head.parameters(), lr=config["learning_rate"])
    history = []
    monitoring_seconds = 0.0
    monitoring_calls = 0
    if config.get("monitor_exact_distribution", False):
        metrics, elapsed = _monitor_distribution(task, head, config["beta"])
        monitoring_seconds += elapsed
        monitoring_calls += 1
        history.append(
            {
                "step": 0,
                "loss": None,
                "learning_rate": config["learning_rate"],
                "distribution": metrics,
            }
        )
        if progress is not None:
            progress(history[-1])
    for step in range(1, config["steps"] + 1):
        _set_learning_rates(optimizer, config, step)
        chosen = torch.randint(
            len(edge_ids),
            (config["batch_size"],),
            generator=generator,
            device=task.device,
        )
        edges = edge_ids[chosen]
        nodes = edges // action_count
        actions = edges % action_count
        corrections = head(task.features[nodes], task.context[nodes])
        predictions = corrections.gather(-1, actions[:, None]).squeeze(-1)
        loss = (predictions - targets[edges]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            head.parameters(), config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        if _should_log(step, config):
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if config.get("monitor_exact_distribution", False):
                metrics, elapsed = _monitor_distribution(task, head, config["beta"])
                monitoring_seconds += elapsed
                monitoring_calls += 1
                history[-1]["distribution"] = metrics
            if progress is not None:
                progress(history[-1])
    total_seconds = time.perf_counter() - started
    return (
        head,
        history,
        {
            "training_seconds": total_seconds,
            "training_seconds_excluding_exact_monitoring": total_seconds
            - monitoring_seconds,
            "exact_monitoring_seconds": monitoring_seconds,
            "exact_monitoring_calls": monitoring_calls,
            "exact_monitoring_cached_score_reads": monitoring_calls * len(task.scores),
            "cached_terminal_reward_lookups": config["rollouts"],
            "new_scorer_forward_calls": 0,
            "observed_edge_fraction": visited.sum().item() / task.legal.sum().item(),
            "backbone_calls_during_training": 0,
        },
    )
