"""One CPU control averaging the sampled TB gradient before inherited Adam."""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import torch
from condition_idr_exact_cpu import ConditionedHead, equivalence
from diagnose_idr_updates_cpu import load_task, metrics

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.idr.acceptance_experiment import verify_manifest
from stability_gfn.idr.branch_rate import branch_optimizer, set_rates


def expected_loss(log_q, log_reward, normalizer, stratum):
    """Average raw residual gradients; never differentiate sampling weights."""
    probabilities = log_q.detach().exp()
    probabilities = probabilities / probabilities.sum()
    replay = stratum.to(log_q.dtype)
    if not bool(replay.sum()):
        raise ValueError("Replay stratum must contain a leaf")
    weights = (probabilities + replay / replay.sum()) / 2
    residual = normalizer + log_q - log_reward
    return (weights * residual.square()).sum()


def fit_expected(trees, initial, offsets, config):
    """Exactly average leaf draws, keeping the alternating replay schedule."""
    head = copy.deepcopy(initial)
    normalizer = torch.nn.Parameter(offsets.detach().clone())
    parameters = [*head.parameters(), normalizer]
    optimizer = branch_optimizer(head, normalizer, config, 10)
    history, updates = [], []
    clipped = 0
    started = time.monotonic()
    for step in range(1, config["steps"] + 1):
        set_rates(optimizer, config, 10, step)
        losses = [
            expected_loss(
                tree.log_probabilities(head).double(),
                tree.log_reward,
                normalizer[prompt],
                tree.passed == ((prompt + step) % 2 == 0),
            )
            for prompt, tree in enumerate(trees)
        ]
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config["gradient_clip"], error_if_nonfinite=True
        )
        clipped += float(norm) > config["gradient_clip"]
        optimizer.step()
        updates.append(
            dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm))
        )
        if step == 1 or step % 100 == 0:
            history.append(dict(step=step, **metrics(trees, head)))
    result = metrics(trees, head)
    target_joint = sum(float(t.target[t.passed].sum()) for t in trees) / len(trees)
    result.update(
        steps=config["steps"],
        max_prompt_tv=max(r["tv"] for r in result["by_prompt"].values()),
        target_joint_success=target_joint,
        clipped_fraction=clipped / config["steps"],
        elapsed_seconds=time.monotonic() - started,
        history=history,
    )
    result["passes_gate"] = (
        result["mean_tv"] < 0.05
        and result["max_prompt_tv"] < 0.05
        and abs(result["mean_joint_success"] - target_joint) < 0.02
    )
    for prompt, tree in enumerate(trees):
        result["by_prompt"][tree.rows[0]["prompt_id"]]["normalizer_error"] = float(
            normalizer[prompt].detach() - tree.log_reward.logsumexp(0)
        )
    return result, head, normalizer, updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-continuous-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    trees, initial, offsets, config = load_task(args.run)
    if config["steps"] != 400:
        raise ValueError("The protocol fixes 400 updates")
    candidate = ConditionedHead(initial, trees)
    initial_check = equivalence(trees, initial, candidate)
    if initial_check["maximum_prompt_tv_between_policies"] > 1e-4:
        raise ValueError("Initial policy changed")
    output = new_run(
        args.output,
        dict(
            source=str(args.run.resolve()),
            steps=400,
            diagnostic_only=True,
            support="locally_masked_training_bank",
            optimizer="inherited Adam",
            estimator="exact conditional mean of raw sampled TB gradient",
        ),
    )
    root = Path(__file__).resolve().parents[1]
    source = output / "source"
    source.mkdir()
    for name in (
        Path(__file__).name,
        "condition_idr_exact_cpu.py",
        "fit_idr_exact_optimizer_cpu.py",
        "diagnose_idr_updates_cpu.py",
        "diagnose_idr_exact_cpu.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    shutil.copy2(
        root / "docs/experiments/idr_expected_conditioning_2026-11-16.md",
        source / "protocol.md",
    )
    inputs = {
        name: file_hash(args.run / name)
        for name in (
            "generation.json",
            "bank.json",
            "bank_cache.pt",
            "initial.pt",
            "pilot_config.json",
            "reward.json",
            "provenance.json",
        )
    }
    write_json(
        output / "initialization.json",
        dict(
            equivalence=initial_check,
            conditioning=candidate.conditioning,
            config=config,
            initial_metrics=metrics(trees, candidate),
            leaf_ids=[[r["id"] for r in t.rows] for t in trees],
        ),
    )
    result, head, normalizer, updates = fit_expected(trees, candidate, offsets, config)
    write_json(output / "results.json", result)
    write_json(output / "updates.json", updates)
    torch.save(
        dict(head=head.state_dict(), log_z=normalizer.detach(), diagnostic_only=True),
        output / "conditioned.pt",
    )
    saved = torch.load(output / "conditioned.pt", map_location="cpu", weights_only=True)
    restored = copy.deepcopy(candidate)
    restored.load_state_dict(saved["head"])
    reproduced = metrics(trees, restored)
    for key in ("mean_tv", "mean_joint_success"):
        if reproduced[key] != result[key]:
            raise ValueError("Saved aggregate metric differs")
    for prompt, tree in enumerate(trees):
        name = tree.rows[0]["prompt_id"]
        for key, value in reproduced["by_prompt"][name].items():
            if result["by_prompt"][name][key] != value:
                raise ValueError("Saved per-prompt metric differs")
        error = float(saved["log_z"][prompt] - tree.log_reward.logsumexp(0))
        if error != result["by_prompt"][name]["normalizer_error"]:
            raise ValueError("Saved normalizer differs")
    folded = restored.fold()
    torch.save(
        dict(head=folded.state_dict(), log_z=saved["log_z"], diagnostic_only=True),
        output / "folded.pt",
    )
    fold_check = equivalence(trees, restored, folded)
    if fold_check["maximum_prompt_tv_between_policies"] > 1e-4:
        raise ValueError("Folding changes policy")
    verify_manifest(args.run)
    if inputs != {name: file_hash(args.run / name) for name in inputs}:
        raise ValueError("Source inputs changed")
    assert not torch.cuda.is_initialized()
    write_json(
        output / "verification.json",
        dict(
            all_saved_metrics_match=True,
            fold_equivalence=fold_check,
            source_inputs_unchanged=True,
            cuda_initialized=False,
        ),
    )
    write_json(
        output / "provenance.json",
        dict(
            input_hashes=inputs,
            source_files={p.name: file_hash(p) for p in source.iterdir()},
            torch=torch.__version__,
            cuda_initialized=False,
        ),
    )
    write_json(output / "status.json", dict(status="complete", diagnostic_only=True))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("history", "by_prompt")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
