"""Acceptance-aware terminal weight without a failure floor or pass bonus."""

import math

from .acceptance_reward import joint_acceptance_reward


def continuous_reward(tails, disorder_scores, reward, penalty=2.5, device="cpu"):
    """Keep every pass criterion; penalize only distance outside its tolerance.

    This changes the target, not the TB algorithm. Log weights avoid numerical
    underflow and replace the old factor directly, with no additional beta.
    The scorer is detached just like the historical reward.
    """
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Continuous reward penalty must be finite and positive")
    values = joint_acceptance_reward(tails, disorder_scores, reward, device)
    values["current_log_factor"] = values["log_factor"]
    values["log_factor"] = -penalty * values["violations"].sum(1)
    return values
