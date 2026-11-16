"""Versioned joint charge/disorder reward, separate from historical charge scores.

The caller supplies isolated-tail metapredict V3 scores. This module does not run
a predictor or train it. Its log factor replaces beta * S in trajectory balance;
it must not be multiplied by the historical beta again.
"""

import math
from dataclasses import dataclass

import torch

from .charge import charge_values


@dataclass(frozen=True)
class AcceptanceRewardConfig:
    """Engineering defaults for joint_acceptance_v1, not biological constants."""

    tail_length: int = 64
    block_targets: tuple[float, float] = (0.125, -0.125)
    block_tolerance: float = 0.0625
    fcr_target: float = 0.25
    fcr_tolerance: float = 0.125
    disorder_cutoff: float = 0.5
    minimum_disordered_fraction: float = 0.8
    disorder_deficit_scale: float = 0.2
    pass_to_near_miss_ratio: float = 20.0
    failure_floor_fraction: float = 0.25

    def __post_init__(self):
        if (
            type(self.tail_length) is not int
            or self.tail_length < 2
            or self.tail_length % 2
        ):
            raise ValueError("Need a positive even tail length")
        if len(self.block_targets) != 2 or any(
            not math.isfinite(x) or not -1 <= x <= 1 for x in self.block_targets
        ):
            raise ValueError("Need two finite block targets in [-1, 1]")
        for name in ("block_tolerance", "fcr_tolerance", "disorder_deficit_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        for name in ("fcr_target", "disorder_cutoff", "minimum_disordered_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name}")
        if (
            not math.isfinite(self.pass_to_near_miss_ratio)
            or self.pass_to_near_miss_ratio <= 1
        ):
            raise ValueError("Pass/near-miss ratio must exceed one")
        if (
            not math.isfinite(self.failure_floor_fraction)
            or not 0 < self.failure_floor_fraction < 1
        ):
            raise ValueError(
                "Failure floor fraction must lie strictly between zero and one"
            )


@torch.no_grad()
def joint_acceptance_reward(tails, disorder_scores, settings=None, device="cpu"):
    """Return detached log weights and diagnostics from complete generated tails.

    A pass receives factor 1. Failures receive between 1/80 and 1/20 under
    defaults, improving with proximity to the acceptable region. No constraint
    earns extra credit inside its acceptable range. Thus every joint pass has a
    larger PROPERTY factor than every failure, regardless of the other terms.
    The full reward still includes the prior probability.
    """
    settings = settings or AcceptanceRewardConfig()
    charges = charge_values(tails, device).to(torch.float64)
    if charges.shape[1] != settings.tail_length:
        raise ValueError("Tail length differs from the reward contract")
    disorder = torch.as_tensor(disorder_scores, dtype=torch.float64, device=device)
    if disorder.shape != charges.shape:
        raise ValueError("Need one isolated-tail disorder score per generated residue")
    if not torch.isfinite(disorder).all() or ((disorder < 0) | (disorder > 1)).any():
        raise ValueError("Disorder scores must be finite and in [0, 1]")

    midpoint = settings.tail_length // 2
    blocks = torch.stack(
        [charges[:, :midpoint].mean(1), charges[:, midpoint:].mean(1)], dim=1
    )
    block_error = (blocks - blocks.new_tensor(settings.block_targets)).abs()
    fcr = charges.abs().mean(1)
    fcr_error = (fcr - settings.fcr_target).abs()
    # Preserve the historical strict per-residue > cutoff and inclusive fraction.
    disorder_fraction = (disorder > settings.disorder_cutoff).double().mean(1)
    charge_pass = (block_error <= settings.block_tolerance).all(1)
    charge_pass &= fcr_error <= settings.fcr_tolerance
    disorder_pass = disorder_fraction >= settings.minimum_disordered_fraction
    joint_pass = charge_pass & disorder_pass

    block_violation = (block_error / settings.block_tolerance - 1).clamp_min(0)
    fcr_violation = (fcr_error / settings.fcr_tolerance - 1).clamp_min(0)
    disorder_violation = (
        (settings.minimum_disordered_fraction - disorder_fraction)
        / settings.disorder_deficit_scale
    ).clamp_min(0)
    violations = torch.cat(
        [block_violation, fcr_violation[:, None], disorder_violation[:, None]], dim=1
    )
    total_violation = violations.sum(1)
    # Positive floor bounds log reward and prevents zero-reward/log(0) failures.
    proximity_factor = settings.failure_floor_fraction + (
        1 - settings.failure_floor_fraction
    ) * torch.exp(-total_violation)
    log_factor = torch.where(
        joint_pass,
        torch.zeros_like(total_violation),
        proximity_factor.log() - math.log(settings.pass_to_near_miss_ratio),
    )
    return {
        "log_factor": log_factor,
        "violations": violations,
        "proximity_factor": proximity_factor,
        "blocks": blocks,
        "fcr": fcr,
        "disorder_fraction": disorder_fraction,
        "charge_pass": charge_pass,
        "disorder_pass": disorder_pass,
        "joint_pass": joint_pass,
    }
