"""Common scored examples for equally adaptable GFN and supervised policies."""

import time

import torch

from .continuous_reward import continuous_reward
from .lastblock_sampling import sample_records


def refresh_objective_pool(
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
    """Share every discovery; rewards and proposal identities are saved once.

    Replay stays in the original training partition. As in the preceding study,
    its success/failure stratum alternates by prompt and refresh. Neither method
    can keep discoveries private or replace the shared data with evaluation rows.
    """
    fresh, generation_costs = [], {}
    for arm in ("gfn", "supervised", "prior"):
        started = time.monotonic()
        records = sample_records(
            reference if arm == "prior" else policies[arm],
            None if arm == "prior" else heads[arm],
            prompts,
            config,
            arm,
            generators[arm],
            refresh_round,
        )
        generation_costs[arm] = time.monotonic() - started
        for row in records:
            row.update(
                id=f"{replicate}-fresh-{row['id']}",
                replicate=replicate,
                generator_arm=arm,
                refresh_round=refresh_round,
                split="train",
                behavior_log_policy=row["log_policy"],
            )
        fresh.extend(records)
    started = time.monotonic()
    scores = predictor([row["tail"] for row in fresh], "V3")
    values = continuous_reward(
        [row["tail"] for row in fresh], scores, reward, settings["penalty"]
    )
    scoring_seconds = time.monotonic() - started
    for index, row in enumerate(fresh):
        row.update(
            isolated_tail_v3_scores=[float(value) for value in scores[index]],
            log_factor=float(values["log_factor"][index]),
            violations=values["violations"][index].tolist(),
            joint_pass=bool(values["joint_pass"][index]),
        )
    replay = []
    for index in range(len(prompts)):
        passed = (refresh_round + index) % 2 == 0
        candidates = [
            row
            for row in train_bank
            if row["split"] == "train"
            and row["prompt_index"] == index
            and row["joint_pass"] == passed
        ]
        if not candidates:
            raise ValueError("Missing prompt-specific training replay stratum")
        choice = int(torch.randint(len(candidates), (), generator=bank_generator))
        replay.append(candidates[choice])
    costs = dict(
        generation_seconds=generation_costs,
        scoring_seconds=scoring_seconds,
        scorer_calls=1,
        scored_trajectories=len(fresh),
    )
    return fresh + replay, fresh, costs
