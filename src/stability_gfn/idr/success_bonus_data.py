"""Read-only inputs for three matched GFN/SFT success-bonus repeats."""

import copy
import json
import re

import torch

from .confirmation_data import MAX_SEED, SEED_KEYS, _stream_seeds
from .objective_data import (
    EVALUATION_ARMS,
    HISTORICAL_STREAMS,
    SFT_WEIGHTING,
    TRAINING_ARMS,
    check_objective_inputs,
)
from .objective_data import (
    REAL_FIT_SETTINGS as OBJECTIVE_FIT_SETTINGS,
)
from .success_bonus_reward import REWARD_VERSION, apply_success_bonus_rows

REAL_FIT_SETTINGS = dict(OBJECTIVE_FIT_SETTINGS, pass_multiplier=9.0)
OBJECTIVE_STREAMS = ((823000, 833000, 825000), (923000, 933000, 925000))


def check_success_bonus_inputs(settings, check_checkpoint=True):
    """Preserve the original initialization and data checks; relabel a bank copy.

    One inherited validation checks all shared artifacts and model-file hashes.
    Its two-repeat adapter does not modify historical modules or globals. The
    third repeat and all three streams are validated here against protected
    source configurations and the completed objective-comparison streams.
    """
    if not isinstance(settings, dict) or set(settings) != {"fit", "replicates"}:
        raise ValueError(
            "Success-bonus comparison accepts only fit and three replicates"
        )
    fit = settings["fit"]
    if not isinstance(fit, dict) or set(fit) != {*REAL_FIT_SETTINGS, *SEED_KEYS}:
        raise ValueError("Specify the complete fixed success-bonus fit settings")
    if isinstance(fit["pass_multiplier"], bool) or fit["pass_multiplier"] != 9.0:
        raise ValueError("This comparison fixes the success multiplier at nine")
    replicas = settings["replicates"]
    if not isinstance(replicas, list) or len(replicas) != 3:
        raise ValueError(
            "Success-bonus comparison requires exactly three fresh replicates"
        )
    names = set()
    used_seeds = set()
    for training, evaluation, replay in (*HISTORICAL_STREAMS, *OBJECTIVE_STREAMS):
        used_seeds.update((training, training + 1, training + 2, evaluation, replay))
    for replica in replicas:
        if not isinstance(replica, dict) or set(replica) != {"name", *SEED_KEYS}:
            raise ValueError("Each replicate specifies only its name and three seeds")
        name = replica["name"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is None
            or name in names
        ):
            raise ValueError("Replicate names must be safe, distinct folder names")
        names.add(name)
        for key in SEED_KEYS:
            if type(replica[key]) is not int or not 0 <= replica[key] <= MAX_SEED:
                raise ValueError(f"Invalid replicate {key}")
        seeds = _stream_seeds(replica)
        if len(set(seeds)) != len(seeds) or used_seeds.intersection(seeds):
            raise ValueError("Success-bonus random streams must be fresh and disjoint")
        used_seeds.update(seeds)

    adapter = copy.deepcopy(settings)
    adapter["fit"].pop("pass_multiplier")
    adapter["replicates"] = adapter["replicates"][:2]
    inputs, _ = check_objective_inputs(adapter, check_checkpoint)
    historical_source_seeds = set()
    for folder in inputs[:2]:
        historical_source_seeds.update(
            _stream_seeds(json.loads((folder / "config.json").read_text()))
        )
    if any(
        historical_source_seeds.intersection(_stream_seeds(replica))
        for replica in replicas
    ):
        raise ValueError("Success-bonus random streams must be fresh and disjoint")
    plans = []
    for replica in replicas:
        fit_settings = copy.deepcopy(fit)
        fit_settings.update({key: replica[key] for key in SEED_KEYS})
        plans.append(dict(name=replica["name"], fit_settings=fit_settings))
    bank = apply_success_bonus_rows(inputs[6], fit["pass_multiplier"], fit["penalty"])
    return (*inputs[:6], bank, *inputs[7:]), plans


def success_bonus_preflight(settings, check_checkpoint=True):
    """Describe the complete fixed workload without model or scorer execution."""
    inputs, plans = check_success_bonus_inputs(settings, check_checkpoint)
    source, control, config, references = inputs[:4]
    fit = plans[0]["fit_settings"]
    prompt_count = sum(row["role"] == "prompt" for row in references)
    per_arm = prompt_count * config["samples_per_prompt"]
    rounds = fit["training_steps"] // fit["refresh_every"]
    fresh_per_refresh = sum(
        count for name, count in fit["shared_batch"].items() if name != "replay"
    )
    return dict(
        ready=True,
        source_run=str(source),
        control_run=str(control),
        replicates=[
            dict(
                name=plan["name"],
                **{key: plan["fit_settings"][key] for key in SEED_KEYS},
            )
            for plan in plans
        ],
        training_arms=list(TRAINING_ARMS),
        training_arms_per_replicate=len(TRAINING_ARMS),
        updates_per_training_arm=fit["training_steps"],
        learning_rate_decay_after_update=int(
            config["decay_fraction"] * fit["training_steps"]
        ),
        shared_batch=fit["shared_batch"],
        refresh_rounds_per_replicate=rounds,
        fresh_training_trajectories_per_replicate=rounds * fresh_per_refresh,
        replay_draws_per_replicate=rounds * fit["shared_batch"]["replay"],
        trajectory_presentations_per_training_arm=fit["training_steps"]
        * sum(fit["shared_batch"].values()),
        sft_weighting=SFT_WEIGHTING,
        sft_weight_formula="exp(log_factor) normalized within each prompt",
        reward_version=REWARD_VERSION,
        reward_penalty=fit["penalty"],
        pass_multiplier=fit["pass_multiplier"],
        evaluation_arms=list(EVALUATION_ARMS),
        evaluation_arms_per_replicate=len(EVALUATION_ARMS),
        evaluation_samples_per_arm=per_arm,
        total_final_evaluation_samples=len(plans) * len(EVALUATION_ARMS) * per_arm,
        cuda_initialized=torch.cuda.is_initialized(),
        model_or_scorer_executed=False,
    )
