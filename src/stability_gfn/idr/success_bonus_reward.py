"""Continuous charge/disorder penalties with a joint-success weight bonus.

The multiplier changes the terminal target, not the acceptance thresholds. The
original ProGen likelihood remains a separate factor supplied by the TB trainer.
"""

import copy
import math

import torch

from .continuous_reward import continuous_reward

REWARD_VERSION = "success_bonus_v1"


def _log_bonus(pass_multiplier):
    """Reject invalid or non-strengthening property multipliers."""
    if (
        isinstance(pass_multiplier, bool)
        or not isinstance(pass_multiplier, (int, float))
        or not math.isfinite(pass_multiplier)
        or pass_multiplier <= 1
    ):
        raise ValueError("Success multiplier must be finite and greater than one")
    return math.log(pass_multiplier)


@torch.no_grad()
def success_bonus_reward(
    tails,
    disorder_scores,
    reward=None,
    penalty=2.5,
    pass_multiplier=9.0,
    device="cpu",
):
    """Multiply passing property weights while preserving all failure ratios.

    The log bonus is added once after computing the historical continuous log
    factor. Every diagnostic is detached; this function never runs a scorer.
    """
    log_bonus = _log_bonus(pass_multiplier)
    values = continuous_reward(tails, disorder_scores, reward, penalty, device)
    values["base_log_factor"] = values["log_factor"].detach().clone()
    values["success_log_bonus"] = values["joint_pass"].to(torch.float64) * log_bonus
    values["log_factor"] = values["base_log_factor"] + values["success_log_bonus"]
    return values


def _check_base_row(row, penalty, factor_key):
    """Check the continuous reward and acceptance labels before transforming it."""
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Continuous reward penalty must be finite and positive")
    violations = row.get("violations")
    if (
        not isinstance(violations, list)
        or len(violations) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in violations
        )
    ):
        raise ValueError("Expected four finite nonnegative reward violations")
    if type(row.get("joint_pass")) is not bool or row["joint_pass"] != (
        sum(violations) == 0
    ):
        raise ValueError("Joint-success label disagrees with reward violations")
    value = row.get(factor_key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not math.isclose(value, -penalty * sum(violations), abs_tol=1e-9)
    ):
        raise ValueError("Stored continuous log factor differs from violations")


def apply_success_bonus_rows(rows, pass_multiplier=9.0, penalty=2.5):
    """Clone already continuous-scored rows and apply the success bonus once.

    Fresh paths and the original replay bank use this identical conversion. The
    original rows, including their score arrays and partition labels, stay intact.
    """
    log_bonus = _log_bonus(pass_multiplier)
    result = []
    for row in rows:
        if any(
            key in row
            for key in ("base_log_factor", "success_log_bonus", "reward_version")
        ):
            raise ValueError("Cannot apply success bonus to an already versioned row")
        _check_base_row(row, penalty, "log_factor")
        transformed = copy.deepcopy(row)
        transformed["base_log_factor"] = row["log_factor"]
        transformed["success_log_bonus"] = log_bonus if row["joint_pass"] else 0.0
        transformed["log_factor"] = (
            transformed["base_log_factor"] + transformed["success_log_bonus"]
        )
        transformed["reward_version"] = REWARD_VERSION
        result.append(transformed)
    return result


def validate_success_bonus_rows(rows, pass_multiplier=9.0, penalty=2.5):
    """Verify replay rows have the same bonus as fresh paths, without adding it."""
    log_bonus = _log_bonus(pass_multiplier)
    for row in rows:
        _check_base_row(row, penalty, "base_log_factor")
        expected_bonus = log_bonus if row["joint_pass"] else 0.0
        expected_factor = row["base_log_factor"] + expected_bonus
        if row.get("reward_version") != REWARD_VERSION:
            raise ValueError("Replay row has the wrong success-bonus reward version")
        for key, expected in (
            ("success_log_bonus", expected_bonus),
            ("log_factor", expected_factor),
        ):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(value, expected, abs_tol=1e-9)
            ):
                raise ValueError(
                    "Replay reward does not contain exactly one success bonus"
                )
