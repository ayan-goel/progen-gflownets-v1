"""Bounded deterministic TB optimizer comparison on the cached finite IDR task."""

import argparse
import copy
import inspect
import shutil
import time
from pathlib import Path

import torch
from diagnose_idr_updates_cpu import load_task, metrics, objective

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.idr.acceptance_experiment import verify_manifest
from stability_gfn.idr.branch_rate import branch_optimizer, set_rates


class GradientBudgetExhausted(Exception):
    """Stop before evaluating another trial; caller restores the accepted state."""


def restore_parameters(parameters, values):
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            parameter.copy_(value)


def fit_fixed_tb(trees, initial, initial_normalizer, config, arm, budget=400):
    """Same fixed TB objective; count every backward evaluation, including trials."""
    if arm not in ("adam_clipped", "adam_unclipped", "lbfgs") or budget < 1:
        raise ValueError("Invalid optimizer condition or budget")
    head = copy.deepcopy(initial)
    normalizer = torch.nn.Parameter(initial_normalizer.detach().clone())
    parameters = list(head.parameters()) + [normalizer]
    if arm == "lbfgs":
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=1,
            max_eval=20,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = branch_optimizer(head, normalizer, config, 10)
    evaluations, accepted_updates, over_clip = 0, 0, 0
    history, raw_norms = [], []
    stopped = "budget"
    start_time = time.monotonic()

    def closure():
        nonlocal evaluations, over_clip
        if evaluations >= budget:
            raise GradientBudgetExhausted
        optimizer.zero_grad(set_to_none=True)
        loss = objective(trees, head, normalizer)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite deterministic TB objective")
        loss.backward()
        evaluations += 1
        maximum = config["gradient_clip"] if arm == "adam_clipped" else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, maximum, error_if_nonfinite=True
        )
        raw_norms.append(float(norm))
        over_clip += float(norm) > config["gradient_clip"]
        return loss

    while evaluations < budget:
        before = [p.detach().clone() for p in parameters]
        if arm == "lbfgs":
            try:
                optimizer.step(closure)
            except GradientBudgetExhausted:
                # A line-search trial may have changed parameters. It is not an
                # accepted update, and must never become the reported final model.
                restore_parameters(parameters, before)
                stopped = "budget_during_trial_restored_last_accepted"
                break
        else:
            set_rates(optimizer, config, 10, accepted_updates + 1)
            closure()
            optimizer.step()
        if all(torch.equal(p.detach(), old) for p, old in zip(parameters, before)):
            stopped = "optimizer_no_parameter_change"
            break
        accepted_updates += 1
        if accepted_updates == 1 or accepted_updates % 25 == 0:
            with torch.no_grad():
                record = metrics(trees, head)
                history.append(
                    {
                        "accepted_updates": accepted_updates,
                        "gradient_evaluations": evaluations,
                        "fixed_tb_loss": float(objective(trees, head, normalizer)),
                        "mean_tv": record["mean_tv"],
                    }
                )
    with torch.no_grad():
        result = metrics(trees, head)
        result["fixed_tb_loss"] = float(objective(trees, head, normalizer))
        result["normalizer_errors"] = [
            float(normalizer[i] - t.log_reward.logsumexp(0))
            for i, t in enumerate(trees)
        ]
    target_success = sum(float(t.target[t.passed].sum()) for t in trees) / len(trees)
    maximum_tv = max(row["tv"] for row in result["by_prompt"].values())
    result.update(
        gradient_evaluations=evaluations,
        accepted_updates=accepted_updates,
        stop_reason=stopped,
        seconds=time.monotonic() - start_time,
        fraction_raw_norm_above_one=over_clip / evaluations,
        raw_gradient_norm_quantiles=torch.tensor(raw_norms)
        .quantile(torch.tensor([0.0, 0.5, 1.0]))
        .tolist(),
        actual_clipping=arm == "adam_clipped",
        maximum_prompt_tv=maximum_tv,
        exact_target_joint_success=target_success,
        passes_local_gate=result["mean_tv"] < 0.05
        and maximum_tv < 0.05
        and abs(result["mean_joint_success"] - target_success) < 0.02,
        history=history,
    )
    return result, head, normalizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-continuous-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    trees, initial, offsets, config = load_task(args.run)
    arms = ["adam_clipped", "adam_unclipped", "lbfgs"]
    output = new_run(
        args.output,
        {
            "source": str(args.run.resolve()),
            "arms": arms,
            "gradient_budget": 400,
            "weighting": "0.5 initial policy + 0.25 uniform pass + 0.25 uniform fail per prompt",
            "optimizer_controls": "Adam inherited rates/schedule; L-BFGS lr1 history20 strong_wolfe tol_grad1e-7 tol_change1e-9",
            "support": "same 408 selected training leaves, locally masked; no open-generation claim",
        },
    )
    source_folder = output / "source"
    source_folder.mkdir()
    for name in (
        "fit_idr_exact_optimizer_cpu.py",
        "diagnose_idr_updates_cpu.py",
        "diagnose_idr_exact_cpu.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), source_folder / name)
    results = {}
    for arm in arms:
        results[arm], head, normalizer = fit_fixed_tb(
            trees, initial, offsets, config, arm
        )
        torch.save(
            {
                "head": head.state_dict(),
                "log_z": normalizer.detach(),
                "diagnostic_only": True,
            },
            output / f"{arm}.pt",
        )
        write_json(output / "results.json", results)
        print(
            arm,
            "TV",
            results[arm]["mean_tv"],
            "gate",
            results[arm]["passes_local_gate"],
            flush=True,
        )
    verify_manifest(args.run)
    assert not torch.cuda.is_initialized()
    write_json(
        output / "provenance.json",
        {
            "source_manifest_sha256": file_hash(args.run / "generation.json"),
            "source_files": {
                p.name: file_hash(p) for p in sorted(source_folder.iterdir())
            },
            "production_sources": {
                str(p): file_hash(p)
                for p in sorted(Path("src/stability_gfn").rglob("*.py"))
            },
            "torch": torch.__version__,
            "lbfgs_source_sha256": file_hash(Path(inspect.getfile(torch.optim.LBFGS))),
            "cuda_initialized": False,
        },
    )
    write_json(output / "status.json", {"status": "complete", "diagnostic_only": True})


if __name__ == "__main__":
    main()
