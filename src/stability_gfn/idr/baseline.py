"""User-run frozen-ProGen sampling with automatic IDR baseline evaluation."""

import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import TOKENIZER_PATH, FrozenProGen
from ..pipeline import project_path
from .data import ALPHABET
from .scoring import predict_disorder, score_records, summarize


def scorer_files():
    """Hash packaged scoring code and weights without running a predictor."""
    package = Path(
        importlib.metadata.distribution("metapredict").locate_file("metapredict")
    )
    return (
        list(package.glob("backend/networks/*"))
        + list(package.glob("backend/*.py"))
        + [package / "meta.py", package / "parameters.py"]
    )


def check_inputs(config, check_checkpoint=True):
    """Read-only preparation; no model inference and no CUDA readiness queries."""
    if not 0 < config["prefix_length"] < config["length"]:
        raise ValueError("Prefix must be shorter than output length")
    if config["charge_window"] < 1 or config["charge_window"] % 2 != 1:
        raise ValueError("Charge window must be positive and odd")
    if config["temperature"] != 1.0 or config["precision"] != "float32":
        raise ValueError("This baseline protocol fixes temperature 1 and float32")
    for field in ("batch_size", "samples_per_prompt", "cpu_threads"):
        if not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f"Invalid {field}")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Require distinct nonempty seeds")
    if (
        not 0 <= config["disorder_cutoff"] <= 1
        or not 0 < config["minimum_disordered_fraction"] <= 1
    ):
        raise ValueError("Invalid disorder criteria")
    data = project_path(config["data"])
    manifest_path = data / "manifest.json"
    if file_hash(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Reference manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if file_hash(data / "references.json") != manifest["references_sha256"]:
        raise ValueError("Reference sequences changed")
    references = json.loads((data / "references.json").read_text())
    if (
        len(references) != manifest["count"]
        or len({r["id"] for r in references}) != len(references)
        or any(
            len(r["sequence"]) != config["length"] or set(r["sequence"]) - set(ALPHABET)
            for r in references
        )
    ):
        raise ValueError("Invalid reference panel")
    if sum(r["role"] == "prompt" for r in references) != manifest["prompt_count"]:
        raise ValueError("Invalid prompt count")
    if check_checkpoint:
        checkpoint = project_path(config["checkpoint"])
        for name in ("config.json", "pytorch_model.bin"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        model_config = json.loads((checkpoint / "config.json").read_text())
        if config["length"] > model_config["n_positions"]:
            raise ValueError("Output length exceeds the ProGen context limit")
        if importlib.metadata.version("metapredict") != "3.0.2":
            raise ValueError("Install the pinned IDR dependencies")
    return references, manifest


@torch.no_grad()
def generate_batch(backbone, prefixes, length, generator):
    """Ancestral sampling over 20 canonical residues; fixed-length stop, no EOS.

    Renormalizing at each step defines a locally masked prior. It is not globally
    conditioning unrestricted ProGen on length, canonical residues or disorder.
    """
    if len({len(p) for p in prefixes}) != 1 or len(prefixes[0]) >= length:
        raise ValueError("Batch prefixes must share a length shorter than completion")
    residue_ids = torch.tensor(backbone.residue_ids(ALPHABET), device=backbone.device)
    sequences = list(prefixes)
    next_input = backbone.encode(prefixes)
    cache = None
    log_prob = torch.zeros(len(prefixes), device=backbone.device)
    mass = torch.zeros(len(prefixes), device=backbone.device)
    for _ in range(length - len(prefixes[0])):
        _, logits, cache = backbone.step(next_input, cache)
        selected_logits = logits[:, residue_ids]
        log_probs = selected_logits.log_softmax(-1)
        if not torch.isfinite(log_probs).all():
            raise ValueError("Nonfinite generation probabilities")
        actions = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(1)
        log_prob += log_probs.gather(1, actions[:, None]).squeeze(1)
        mass += (selected_logits.logsumexp(-1) - logits.logsumexp(-1)).exp()
        for index, action in enumerate(actions.tolist()):
            sequences[index] += ALPHABET[action]
        next_input = residue_ids[actions, None]
    steps = length - len(prefixes[0])
    return sequences, log_prob.cpu().tolist(), (mass / steps).cpu().tolist()


def evaluate_records(records, references, config, output, predictor=predict_disorder):
    output = Path(output)
    scored = score_records(records, references, config, predictor)
    write_json(output / "sequence_metrics.json", scored)
    groups = {}
    for arm in sorted({r["arm"] for r in scored}):
        group = [r for r in scored if r["arm"] == arm]
        groups[arm] = summarize(group)
    by_seed = {}
    by_prompt = {}
    for row in scored:
        if row["arm"] == "reference":
            continue
        by_seed.setdefault(f"{row['arm']}/seed-{row['seed']}", []).append(row)
        if row["prompt_id"]:
            by_prompt.setdefault(row["prompt_id"], []).append(row)
    metrics = {
        "arms": groups,
        "by_seed": {k: summarize(v) for k, v in by_seed.items()},
        "by_prompt": {k: summarize(v) for k, v in by_prompt.items()},
        "notes": [
            "V2 is a related-model sensitivity check, not independent validation.",
            "Reference similarity is against this small panel, not the ProGen training corpus.",
            "Disorder scores and threshold success are predictions, not biological measurements.",
        ],
    }
    write_json(output / "metrics.json", metrics)
    lines = [
        "# Frozen-ProGen IDR baseline",
        "",
        "Generation and evaluation only; no guidance training.",
        "",
        "Primary outcome: at least 80% of the final 64 residues have V3 score >0.5 (default protocol).",
        "The supplied prefix is excluded; isolated-tail scoring checks context dependence.",
        "",
        "| Arm | n | V3 tail pass | Isolated tail pass | V2 tail pass | Unique tails | Mean tail entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    # Render criteria from config, including tiny fixtures instead of mislabeling smoke outputs.
    lines[4] = (
        f"Primary outcome: at least {config['minimum_disordered_fraction']:.0%} of the final "
        f"{config['length'] - config['prefix_length']} residues have V3 score >{config['disorder_cutoff']}."
    )
    for arm, stats in groups.items():
        lines.append(
            f"| {arm} | {stats['count']} | {stats['v3_context_tail_success_rate']:.1%} | "
            f"{stats['v3_isolated_tail_success_rate']:.1%} | {stats['v2_context_tail_success_rate']:.1%} | "
            f"{stats['unique_tail_fraction']:.1%} | {stats['mean_tail_entropy_bits']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Inspect seed and prompt breakdowns in metrics.json before choosing a setup. "
            "Reference scores are positive-reference checks, not held-out classifier accuracy; "
            "annotation conditions and possible predictor/pretraining overlap remain limitations.",
            "",
            "A candidate setup warrants charge-guidance planning if every seed has >=25% V3 tail success, "
            ">=80% unique tails and <=10% tails with homopolymers of length >=8. These are engineering "
            "screening thresholds, not biological validation. V2 disagreement or strong context-only "
            "success requires review before proceeding. No automatic winner or promotion is made.",
            "",
            "No samples were filtered. Repetition can be natural in IDRs; the homopolymer flag is diagnostic. "
            "Similarity is limited to the saved reference panel. Check sequence_metrics.json for charge "
            "profiles, composition, residue scores and nearest-reference matches.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    return metrics


def run_baseline(
    config, output, device="cuda", backbone=None, predictor=predict_disorder
):
    references, manifest = check_inputs(config, check_checkpoint=backbone is None)
    output = new_run(output, config)
    started = time.monotonic()
    fixture = backbone is not None
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        write_json(output / "references.json", references)
        write_json(output / "reference_manifest.json", manifest)
        files = [
            TOKENIZER_PATH,
            *Path(__file__).parent.glob("*.py"),
            Path(__file__).parent.parent / "backbone.py",
            Path(__file__).parent.parent / "vendor/progen/modeling_progen.py",
            Path(__file__).parent.parent / "vendor/progen/configuration_progen.py",
        ]
        files.extend(path for path in scorer_files() if path.is_file())
        if not fixture:
            checkpoint = project_path(config["checkpoint"])
            files += [checkpoint / "config.json", checkpoint / "pytorch_model.bin"]
        write_json(
            output / "provenance.json",
            {
                "files": {str(p.resolve()): file_hash(p) for p in files},
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "numpy",
                        "transformers",
                        "tokenizers",
                        "biopython",
                        "metapredict",
                    )
                },
                "device": device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "fixture": fixture,
                "scorer_device": "cpu",
                "scorer": "metapredict 3.0.2 V3 and V2",
            },
        )
        if backbone is None:
            print("Loading frozen ProGen; no training", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        records = []
        prompts = [r for r in references if r["role"] == "prompt"]
        with (output / "samples.jsonl").open("w") as handle:
            for seed in config["seeds"]:
                for arm_index, arm in enumerate(("unprompted", "prompted")):
                    generator = torch.Generator(device=backbone.device).manual_seed(
                        seed * 2 + arm_index
                    )
                    scheduled = [
                        ref
                        for ref in prompts
                        for _ in range(config["samples_per_prompt"])
                    ]
                    for offset in range(0, len(scheduled), config["batch_size"]):
                        selected = scheduled[offset : offset + config["batch_size"]]
                        prefixes = [
                            r["sequence"][: config["prefix_length"]]
                            if arm == "prompted"
                            else ""
                            for r in selected
                        ]
                        seqs, logs, masses = generate_batch(
                            backbone, prefixes, config["length"], generator
                        )
                        for index, sequence in enumerate(seqs):
                            row = {
                                "id": f"{arm}-{seed}-{offset + index}",
                                "arm": arm,
                                "seed": seed,
                                "prompt_id": selected[index]["id"]
                                if arm == "prompted"
                                else None,
                                "prefix": prefixes[index],
                                "sequence": sequence,
                                "evaluation_start": config["prefix_length"],
                                "sampled_residues": config["length"]
                                - len(prefixes[index]),
                                "conditional_log_probability": logs[index],
                                "mean_canonical_probability_mass": masses[index],
                            }
                            handle.write(json.dumps(row) + "\n")
                            records.append(row)
                        handle.flush()
                        print(
                            f"seed {seed} {arm}: {offset + len(selected)}/{len(scheduled)}",
                            flush=True,
                        )
        write_json(
            output / "generation.json",
            {
                "complete": True,
                "count": len(records),
                "samples_sha256": file_hash(output / "samples.jsonl"),
                "config_sha256": file_hash(output / "config.json"),
                "seconds": time.monotonic() - started,
                "backbone_work": backbone.work,
            },
        )
        (output / "samples.fasta").write_text(
            "".join(f">{r['id']}\n{r['sequence']}\n" for r in records)
        )
        for ref in references:
            records.append(
                {
                    "id": ref["id"],
                    "arm": "reference",
                    "sequence": ref["sequence"],
                    "reference_role": ref["role"],
                    "evaluation_start": config["prefix_length"],
                }
            )
        evaluate_records(records, references, config, output, predictor)
        write_json(
            output / "status.json",
            {
                "status": "complete",
                "fixture": fixture,
                "seconds": time.monotonic() - started,
            },
        )
    except Exception as error:
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "fixture": fixture,
                "error": str(error),
                "seconds": time.monotonic() - started,
            },
        )
        raise
    return output / "report.md"


def reevaluate(run, output):
    """Recover evaluation without resampling; use saved reference/config snapshots."""
    run, output = Path(run), Path(output)
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    for path in scorer_files():
        if path.is_file() and provenance["files"].get(str(path.resolve())) != file_hash(
            path
        ):
            raise ValueError("Saved scorer provenance differs from installed scorer")
    generation = json.loads((run / "generation.json").read_text())
    if file_hash(run / "config.json") != generation["config_sha256"]:
        raise ValueError("Saved generation configuration changed")
    if (
        not generation["complete"]
        or file_hash(run / "samples.jsonl") != generation["samples_sha256"]
    ):
        raise ValueError("Incomplete or modified generation")
    records = [
        json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()
    ]
    references = json.loads((run / "references.json").read_text())
    manifest = json.loads((run / "reference_manifest.json").read_text())
    if (
        file_hash(run / "references.json") != manifest["references_sha256"]
        or len(records) != generation["count"]
    ):
        raise ValueError("Saved reference or sample count changed")
    output = new_run(output, config)
    torch.set_num_threads(config["cpu_threads"])
    for ref in references:
        records.append(
            {
                "id": ref["id"],
                "arm": "reference",
                "sequence": ref["sequence"],
                "reference_role": ref["role"],
                "evaluation_start": config["prefix_length"],
            }
        )
    try:
        evaluate_records(records, references, config, output)
        write_json(
            output / "status.json",
            {"status": "complete", "source_run": str(run.resolve())},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
    return output / "report.md"
