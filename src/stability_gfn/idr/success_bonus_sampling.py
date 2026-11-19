"""Apply the same versioned success bonus to fresh and replayed shared paths."""

import time

from .objective_sampling import refresh_objective_pool
from .success_bonus_reward import apply_success_bonus_rows, validate_success_bonus_rows


def refresh_success_bonus_pool(
    reference,
    policies,
    heads,
    prompts,
    config,
    settings,
    reward,
    train_bank,
    generators,
    bank_generator,
    refresh_round,
    replicate,
    predictor,
):
    """Reuse collection unchanged; transform fresh paths, never replay them twice.

    The inherited sampler scores fresh paths with the continuous reward and
    selects replay rows from the already transformed original training bank.
    """
    pool, fresh, costs = refresh_objective_pool(
        reference,
        policies,
        heads,
        prompts,
        config,
        settings,
        reward,
        train_bank,
        generators,
        bank_generator,
        refresh_round,
        replicate,
        predictor,
    )
    started = time.monotonic()
    refreshed = apply_success_bonus_rows(
        fresh, settings["pass_multiplier"], settings["penalty"]
    )
    replay = pool[len(fresh) :]
    validate_success_bonus_rows(
        replay, settings["pass_multiplier"], settings["penalty"]
    )
    costs["scoring_seconds"] += time.monotonic() - started
    return refreshed + replay, refreshed, costs
