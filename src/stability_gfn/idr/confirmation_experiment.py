"""Two fresh last-block fits and matched plain-ProGen evaluation.

The original trainer is reused without changing its code, initialization, reward
or budgets. Each child owns its original-format artifacts; the parent adds the
replicate identity, a fifth evaluation policy and a decision across both fits.
"""

import json
import os
import shutil
import time
from pathlib import Path

import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .confirmation_data import check_confirmation_inputs
from .confirmation_evaluation import evaluate_prior, summarize_confirmation
from .lastblock_evaluation import evaluate_lastblock, read_jsonl
from .lastblock_experiment import run_lastblock
from .lastblock_sampling import sample_records
from .scoring import predict_disorder


def _capture_generation(output, count):
    """Pin completed sampling and all available inputs before making the report."""
    captured = {
        str(path.relative_to(output)): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name != "status.json"
        and path != output / "generation.json"
    }
    write_json(
        output / "generation.json",
        dict(complete=True, count=count, files=captured),
    )


def _generate_prior(fit, output, device, backbone):
    """Unfiltered fresh p0 samples; training-time prior rows are never reused."""
    settings = json.loads((fit / "config.json").read_text())
    config = json.loads((fit / "pilot_config.json").read_text())
    references = json.loads((fit / "references.json").read_text())
    prompts = [row for row in references if row["role"] == "prompt"]
    output = new_run(
        output,
        dict(
            arm="prior",
            evaluation_seed=settings["evaluation_seed"],
            fit_settings=settings,
            reference="original frozen ProGen; no guidance head",
        ),
    )
    started = time.monotonic()
    try:
        shutil.copy2(fit / "provenance.json", output / "provenance.json")
        if backbone is None:
            print("Loading original ProGen for the fresh plain baseline", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        # Reset the same evaluation stream for all five policies, as in the
        # pilot's four arms. It is disjoint from every training/replay stream.
        generator = torch.Generator(device=backbone.device).manual_seed(
            settings["evaluation_seed"]
        )
        tick = time.monotonic()
        with (output / "samples.jsonl").open("w") as handle:
            for repeat in range(config["samples_per_prompt"]):
                records = sample_records(
                    backbone, None, prompts, config, "prior", generator, repeat
                )
                for row in records:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                if (repeat + 1) % 16 == 0 or repeat + 1 == config["samples_per_prompt"]:
                    print(
                        f"prior: {repeat + 1}/{config['samples_per_prompt']} fresh outputs per prompt",
                        flush=True,
                    )
        count = len(prompts) * config["samples_per_prompt"]
        write_json(
            output / "costs.json",
            dict(generation_seconds=time.monotonic() - tick, trajectories=count),
        )
        _capture_generation(output, count)
        write_json(
            output / "status.json",
            dict(status="generated", seconds=time.monotonic() - started),
        )
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise


def run_confirmation(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """Run both predeclared replications sequentially, with no best-seed selection.

    Injected backbones/scorers are for tiny CPU fixtures only. Real calls leave
    backbone=None so each fit loads the immutable original checkpoint afresh.
    """
    pilot, replicas, _, _, provenance = check_confirmation_inputs(
        settings, check_checkpoint=backbone is None
    )
    output = new_run(output, settings)
    started = time.monotonic()
    fixture = backbone is not None
    try:
        snapshot = output / "source"
        snapshot.mkdir()
        provenance = dict(
            provenance,
            files=dict(provenance["files"]),
            fixture=fixture,
            device=str(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        for path in sorted(Path(__file__).parent.glob("confirmation_*.py")):
            provenance["files"][str(path.resolve())] = file_hash(path)
            shutil.copy2(path, snapshot / path.name)
        cli = Path(__file__).parents[1] / "__main__.py"
        # Preserve invocation code, but unrelated future CLI additions need not
        # invalidate re-scoring. Computational modules stay live-hash checked.
        shutil.copy2(cli, snapshot / "__main__.py")
        shutil.copy2(
            project_path("docs/experiments/idr_lastblock_confirmation_2026-11-17.md"),
            snapshot / "protocol.md",
        )
        for name in ("generation.json", "config.json"):
            shutil.copy2(pilot / name, output / f"pilot_{name}")
        write_json(output / "provenance.json", provenance)
        costs = {}
        for index, (name, fit_settings) in enumerate(replicas, 1):
            folder = output / name
            folder.mkdir()
            print(
                f"Confirmation {index}/{len(replicas)}: {name} fresh paired fit",
                flush=True,
            )
            write_json(
                output / "status.json",
                dict(
                    status="running",
                    replicate=name,
                    stage="paired_fit",
                    fixture=fixture,
                ),
            )
            # The copied final block and both trainable heads are newly created
            # inside each call, never continued from the positive pilot/first fit.
            run_lastblock(fit_settings, folder / "fit", device, backbone, predictor)
            write_json(
                output / "status.json",
                dict(
                    status="running",
                    replicate=name,
                    stage="plain_baseline",
                    fixture=fixture,
                ),
            )
            _generate_prior(folder / "fit", folder / "prior", device, backbone)
            costs[name] = {
                part: json.loads((folder / part / "status.json").read_text())["seconds"]
                for part in ("fit", "prior")
            }
            write_json(output / "costs.json", dict(replicate_seconds=costs))

        # Keep child artifacts intact. Parent IDs identify the independent fit;
        # legacy child seed=0 remains a local row label, not the RNG seed.
        count = 0
        with (output / "samples.jsonl").open("w") as handle:
            for name, _ in replicas:
                for part in ("fit", "prior"):
                    for row in read_jsonl(output / name / part / "samples.jsonl"):
                        row = dict(row, id=f"{name}/{row['id']}", replicate=name)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        count += 1
        write_json(
            output / "costs.json",
            dict(
                replicate_seconds=costs,
                seconds_before_plain_scoring=time.monotonic() - started,
            ),
        )
        _capture_generation(output, count)
        # Seal all generation before the remaining scorer calls. A failure here
        # can be recovered from saved sequences, without repeating either fit.
        scoring_costs = {}
        for name, _ in replicas:
            folder = output / name
            tick = time.monotonic()
            evaluate_prior(folder / "fit", folder / "prior", predictor=predictor)
            scoring_costs[name] = time.monotonic() - tick
            write_json(
                folder / "prior/status.json",
                dict(
                    status="complete",
                    seconds=costs[name]["prior"] + scoring_costs[name],
                ),
            )
        write_json(
            output / "evaluation_cost.json", dict(prior_scoring_seconds=scoring_costs)
        )
        report = summarize_confirmation(output, output)
        write_json(
            output / "status.json",
            dict(
                status="complete", fixture=fixture, seconds=time.monotonic() - started
            ),
        )
        return report
    except Exception as error:
        write_json(
            output / "status.json",
            dict(status="failed", fixture=fixture, error=repr(error)),
        )
        raise


def reevaluate_confirmation(run, output, predictor=predict_disorder):
    """Re-score complete saved generation without loading any ProGen checkpoint."""
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    settings = json.loads((run / "config.json").read_text())
    output = new_run(output, dict(source_run=str(run.resolve())))
    try:
        for replicate in settings["replicates"]:
            name = replicate["name"]
            source = run / name
            destination = output / name
            (destination / "fit").mkdir(parents=True)
            (destination / "prior").mkdir()
            config = json.loads((source / "fit/pilot_config.json").read_text())
            torch.set_num_threads(config["cpu_threads"])
            evaluate_lastblock(source / "fit", destination / "fit", predictor)
            evaluate_prior(
                source / "fit", source / "prior", destination / "prior", predictor
            )
        report = summarize_confirmation(run, output, evaluations=output)
        write_json(
            output / "status.json",
            dict(status="complete", source_fixture=provenance["fixture"]),
        )
        return report
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise
