"""Fresh paths with an immutable original-ProGen reward reference."""

import torch

from .charge import rollout
from .continuous_reward import continuous_reward
from .lastblock_backbone import LastBlockProGen
from .lastblock_training import trajectory_terms


@torch.no_grad()
def sample_records(
    backbone, head, prompts, config, arm, generator, repeat=0, log_z=None
):
    """One independent continuation per prompt; never label adapted logits p0.

    The historical rollout returns its supplied backbone's base likelihood.
    For the adapted policy we explicitly replace that field with teacher-forced
    likelihood from the untouched original model on the sampled sequence.
    """
    batch = rollout(
        backbone, head, prompts, list(range(len(prompts))), config, generator
    )
    rows = []
    for index, sequence in enumerate(batch["sequences"]):
        prefix = prompts[index]["sequence"][: config["prefix_length"]]
        row = dict(
            id=f"{arm}-0-{repeat}-{index}",
            arm=arm,
            seed=0,
            prompt_id=prompts[index]["id"],
            prompt_index=index,
            prefix=prefix,
            sequence=sequence,
            tail=sequence[config["prefix_length"] :],
            evaluation_start=config["prefix_length"],
            log_prior=float(batch["log_prior"][index]),
            log_policy=float(batch["log_policy"][index]),
            mean_canonical_probability_mass=float(batch["canonical_mass"][index]),
        )
        if log_z is not None:
            row["log_z"] = float(log_z[index])
        rows.append(row)
    if isinstance(backbone, LastBlockProGen):
        features, logits = backbone.reference.forward(
            [r["sequence"][:-1] for r in rows]
        )
        _, reference_logps, mass = trajectory_terms(
            features, logits, None, rows, config, len(prompts)
        )
        for index, row in enumerate(rows):
            row["adapted_base_log_probability"] = row["log_prior"]
            row["log_prior"] = float(reference_logps[index])
            row["mean_canonical_probability_mass"] = float(mass[index])
    return rows


def refresh_pool(
    reference,
    adapted,
    heads,
    prompts,
    config,
    settings,
    reward,
    bank,
    generators,
    bank_generator,
    refresh_round,
    predictor,
):
    """Equal current-policy/prior/replay contributions, shared by both trainers.

    No ranking or rejection is applied to fresh paths. Replay alternates success
    and failure per prompt across rounds; sampling is from training rows only.
    Scores are detached terminal labels and are never differentiated.
    """
    fresh = []
    for arm, backbone, head in (
        ("frozen_tb", reference, heads["frozen_tb"]),
        ("lastblock_tb", adapted, heads["lastblock_tb"]),
        ("prior", reference, None),
    ):
        records = sample_records(
            backbone, head, prompts, config, arm, generators[arm], refresh_round
        )
        for row in records:
            row.update(
                id=f"fresh-{row['id']}",
                generator_arm=arm,
                refresh_round=refresh_round,
                split="train",
                behavior_log_policy=row["log_policy"],
            )
        fresh.extend(records)
    scores = predictor([r["tail"] for r in fresh], "V3")
    values = continuous_reward(
        [r["tail"] for r in fresh], scores, reward, settings["penalty"]
    )
    for index, row in enumerate(fresh):
        row.update(
            isolated_tail_v3_scores=[float(value) for value in scores[index]],
            log_factor=float(values["log_factor"][index]),
            violations=values["violations"][index].tolist(),
            joint_pass=bool(values["joint_pass"][index]),
        )
    replay = []
    for index in range(len(prompts)):
        # Stagger strata across prompts, then swap on the next refresh.
        passed = (refresh_round + index) % 2 == 0
        candidates = [
            r
            for r in bank
            if r["split"] == "train"
            and r["prompt_index"] == index
            and r["joint_pass"] == passed
        ]
        choice = int(torch.randint(len(candidates), (), generator=bank_generator))
        replay.append(candidates[choice])
    return fresh + replay, fresh
