"""Read-only inputs for matched final-block GFN and reward-weighted SFT fits."""

import copy
import json
import re

import torch

from ..pipeline import project_path
from .confirmation_data import MAX_SEED, SEED_KEYS, _stream_seeds
from .lastblock_data import check_lastblock_inputs

SFT_WEIGHTING = "property_factor_per_prompt"
TRAINING_ARMS = ("gfn", "supervised")
EVALUATION_ARMS = (*TRAINING_ARMS, "prior")

# These are the completed last-block protocol's settings. Keep them literal:
# editing an unrelated configuration file must not authorize a different fit.
REAL_FIT_SETTINGS = {
    "source_run": "runs/idr-continuous-v1",
    "source_generation_sha256": "2f4dfc3296b4d72b8eba7f8bb57a5f1a9832677d488bcbdbe6923d1ff85c0c11",
    "control_run": "runs/idr-warmstart-v1",
    "control_generation_sha256": "eab5e8066d85ebe8244aa26b5b3d87ac0def8fd3dd4aa3d3990cf9f9473b2312",
    "control_checkpoint": "continued_supervised/step-400.pt",
    "control_checkpoint_sha256": "e0db233fa73caaee03a9b7f7076d87ce595236ec0a69b2bd58350cc4f9a7a999",
    "training_steps": 1600,
    "refresh_every": 4,
    "shared_batch": {"gfn": 8, "supervised": 8, "prior": 8, "replay": 8},
    "microbatch_size": 4,
    "evaluation_samples_per_prompt": 128,
    "penalty": 2.5,
    "last_block_learning_rate": 1e-5,
    "block_gradient_clip": 1.0,
    "sft_weighting": SFT_WEIGHTING,
}

# Include the adjacent online-sampling streams, not only the named base seeds.
# These declarations are historical; the original run folders need not be loaded.
HISTORICAL_STREAMS = (
    (123000, 133000, 125000),
    (223000, 233000, 225000),
    (323000, 333000, 325000),
    (223000, 433000, 225000),
    (323000, 533000, 325000),
    (223000, 633000, 225000),
    (323000, 733000, 325000),
)


def check_objective_inputs(settings, check_checkpoint=True):
    """Validate two matched fits using saved files and CPU tensors only.

    The inherited checker retains the original p0, reward, bank and initialization
    safeguards. Its historical arm names exist only inside the validation adapter.
    Neither disabling model-file checks nor changing status.json permits a smaller
    real budget: only protected fixture provenance enables tiny test budgets.
    """
    if not isinstance(settings, dict) or set(settings) != {"fit", "replicates"}:
        raise ValueError("Objective comparison accepts only fit and two replicates")
    fit = settings["fit"]
    if not isinstance(fit, dict) or set(fit) != {*REAL_FIT_SETTINGS, *SEED_KEYS}:
        raise ValueError("Specify the complete fixed objective fit settings")
    if fit["sft_weighting"] != SFT_WEIGHTING:
        raise ValueError("Keep property_factor_per_prompt SFT weighting")
    if not isinstance(fit["shared_batch"], dict) or set(fit["shared_batch"]) != {
        "gfn",
        "supervised",
        "prior",
        "replay",
    }:
        raise ValueError("Use the shared gfn/supervised/prior/replay batch")
    if any(type(count) is not int for count in fit["shared_batch"].values()):
        raise ValueError("Shared batch counts must be integers")
    for key in SEED_KEYS:
        if type(fit[key]) is not int or not 0 <= fit[key] <= MAX_SEED:
            raise ValueError(f"Invalid fit {key}")

    replicas = settings["replicates"]
    if not isinstance(replicas, list) or len(replicas) != 2:
        raise ValueError("Objective comparison requires exactly two fresh replicates")
    names = set()
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

    used_seeds = set()
    for training, evaluation, replay in HISTORICAL_STREAMS:
        used_seeds.update((training, training + 1, training + 2, evaluation, replay))
    fixture_flags = []
    for folder_key in ("source_run", "control_run"):
        folder = project_path(fit[folder_key])
        manifest = json.loads((folder / "generation.json").read_text())
        if not {"provenance.json", "config.json"}.issubset(manifest["files"]):
            raise ValueError("Source/control provenance and config must be protected")
        provenance = json.loads((folder / "provenance.json").read_text())
        if type(provenance.get("fixture")) is not bool:
            raise ValueError("Source/control fixture provenance must be explicit")
        fixture_flags.append(provenance["fixture"])
        historical = json.loads((folder / "config.json").read_text())
        used_seeds.update(_stream_seeds(historical))
    if len(set(fixture_flags)) != 1:
        raise ValueError("Source/control fixture provenance differs")
    fixture = all(fixture_flags)
    if fixture:
        if (
            type(fit["training_steps"]) is not int
            or not 1 <= fit["training_steps"] <= 16
        ):
            raise ValueError("Tiny fixtures require at most 16 updates")
    elif any(fit[key] != expected for key, expected in REAL_FIT_SETTINGS.items()):
        raise ValueError("Real objective fits require the fixed 1600-update protocol")

    plans = []
    for replica in replicas:
        seeds = _stream_seeds(replica)
        if len(set(seeds)) != len(seeds) or used_seeds.intersection(seeds):
            raise ValueError("Objective random streams must be fresh and disjoint")
        used_seeds.update(seeds)
        fit_settings = copy.deepcopy(fit)
        fit_settings.update({key: replica[key] for key in SEED_KEYS})
        plans.append(dict(name=replica["name"], fit_settings=fit_settings))

    # Every non-seed input is shared. One inherited validation hashes the model
    # once, checks both source manifests, and loads only small saved CPU tensors.
    adapted = copy.deepcopy(plans[0]["fit_settings"])
    adapted.pop("sft_weighting")
    adapted["shared_batch"] = dict(
        frozen_tb=fit["shared_batch"]["gfn"],
        lastblock_tb=fit["shared_batch"]["supervised"],
        prior=fit["shared_batch"]["prior"],
        replay=fit["shared_batch"]["replay"],
    )
    inputs = check_lastblock_inputs(adapted, check_checkpoint=check_checkpoint)
    return inputs, plans


def objective_preflight(settings, check_checkpoint=True):
    """Report the complete workload without constructing a model or scorer."""
    inputs, plans = check_objective_inputs(settings, check_checkpoint)
    source, control, config, references = inputs[:4]
    fit = plans[0]["fit_settings"]
    prompt_count = sum(row["role"] == "prompt" for row in references)
    per_arm = prompt_count * config["samples_per_prompt"]
    refresh_rounds = fit["training_steps"] // fit["refresh_every"]
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
        refresh_rounds_per_replicate=refresh_rounds,
        fresh_training_trajectories_per_replicate=refresh_rounds * fresh_per_refresh,
        replay_draws_per_replicate=refresh_rounds * fit["shared_batch"]["replay"],
        trajectory_presentations_per_training_arm=(
            fit["training_steps"] * sum(fit["shared_batch"].values())
        ),
        sft_weighting=SFT_WEIGHTING,
        sft_weight_formula="exp(log_factor) normalized within each prompt",
        reward_penalty=fit["penalty"],
        evaluation_arms=list(EVALUATION_ARMS),
        evaluation_arms_per_replicate=len(EVALUATION_ARMS),
        evaluation_samples_per_arm=per_arm,
        total_final_evaluation_samples=len(plans) * len(EVALUATION_ARMS) * per_arm,
        cuda_initialized=torch.cuda.is_initialized(),
        model_or_scorer_executed=False,
    )
