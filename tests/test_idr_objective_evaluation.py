"""Synthetic CPU checks of paired-objective endpoints and shared-data auditing."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from test_idr_charge import configuration
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import write_json
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.continuous_reward import continuous_reward
from stability_gfn.idr.objective_evaluation import (
    ARMS,
    cluster_sizes,
    evaluate_objective,
    objective_decision,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def evaluation_fixture(root):
    """Use explicit synthetic log probabilities and a deterministic fake scorer."""
    config = configuration()
    config.update(length=10, steps=2, samples_per_prompt=4)
    config["reward"] = dict(config["reward"], block_targets=[0, 0], fcr_target=0)
    fit = dict(
        penalty=2.5,
        block_gradient_clip=1.0,
        training_steps=2,
        refresh_every=2,
        shared_batch=dict(gfn=2, supervised=2, prior=2, replay=2),
        sft_weighting="property_factor_per_prompt",
        evaluation_samples_per_prompt=4,
    )
    replicates = [
        dict(
            name=f"replicate-{index}",
            training_seed=index,
            evaluation_seed=index + 10,
            bank_sampling_seed=index + 20,
        )
        for index in (1, 2)
    ]
    reward = replace(
        AcceptanceRewardConfig(), tail_length=6, block_targets=[0, 0], fcr_target=0
    )
    prompts = [
        dict(id="a", role="prompt", sequence="GSKQAAAAAA"),
        dict(id="b", role="prompt", sequence="AQPSGGGGGG"),
    ]
    for name, value in (
        ("config.json", dict(fit=fit, replicates=replicates)),
        ("pilot_config.json", config),
        ("reward.json", asdict(reward)),
        ("references.json", prompts),
        ("costs.json", dict(setup_seconds=1.0)),
    ):
        write_json(root / name, value)
    bank = []
    for prompt_index, prompt in enumerate(prompts):
        for tail, split in (("AAAAAA", "train"), ("KKKKKK", "holdout")):
            values = continuous_reward([tail], [[0.9] * 6], reward, fit["penalty"])
            bank.append(
                dict(
                    id=f"{prompt['id']}-{tail}",
                    prompt_id=prompt["id"],
                    prompt_index=prompt_index,
                    tail=tail,
                    split=split,
                    sequence=prompt["sequence"][:4] + tail,
                    joint_pass=bool(values["joint_pass"][0]),
                    log_factor=float(values["log_factor"][0]),
                    log_prior=-10.0,
                )
            )
    write_json(root / "bank.json", bank)
    write_jsonl(
        root / "initial_training.jsonl",
        [dict(step=1, replay_ids=["a-AAAAAA", "b-AAAAAA"])],
    )
    all_samples = []
    for replicate in replicates:
        name = replicate["name"]
        folder = root / name
        folder.mkdir()
        write_json(
            folder / "config.json",
            {
                **fit,
                **{key: value for key, value in replicate.items() if key != "name"},
            },
        )
        fresh = []
        for source, tail in (
            ("gfn", "GGGGGG" if name == "replicate-1" else "SSSSSS"),
            ("supervised", "QQQQQQ"),
            ("prior", "NNNNNN"),
        ):
            for prompt_index, prompt in enumerate(prompts):
                values = continuous_reward([tail], [[0.9] * 6], reward, fit["penalty"])
                fresh.append(
                    dict(
                        id=f"{name}-fresh-{source}-{prompt['id']}",
                        replicate=name,
                        generator_arm=source,
                        refresh_round=0,
                        prompt_id=prompt["id"],
                        prompt_index=prompt_index,
                        tail=tail,
                        sequence=prompt["sequence"][:4] + tail,
                        isolated_tail_v3_scores=[0.9] * 6,
                        log_factor=float(values["log_factor"][0]),
                        log_prior=-10.0,
                        joint_pass=bool(values["joint_pass"][0]),
                    )
                )
        write_jsonl(folder / "training_rows.jsonl", fresh)
        shared = [
            dict(
                step=step,
                refresh_round=0,
                example_ids=[row["id"] for row in fresh] + ["a-AAAAAA", "b-AAAAAA"],
                counts=fit["shared_batch"],
            )
            for step in (1, 2)
        ]
        write_jsonl(folder / "shared_training.jsonl", shared)
        for arm in ("gfn", "supervised"):
            (folder / arm).mkdir()
            logs = []
            for shared_row in shared:
                logged = dict(
                    shared_row,
                    rates=dict(base=0.001, task=0.01, last_block=0.0001),
                    loss=0.1,
                    update_seconds=0.25,
                )
                if arm == "gfn":
                    logged["rates"]["normalizer"] = 0.01
                else:
                    logged.update(
                        weighted_nll=0.1,
                        effective_sample_size=8.0,
                        weighting=dict(
                            method="per_prompt_softmax_log_factor_equal_prompt_mean",
                            normalized_before_microbatch=True,
                            prior_in_weights=False,
                            represented_prompts=2,
                            loss_weight_sum=1.0,
                            prompts=[
                                dict(
                                    prompt_index=index,
                                    examples=4,
                                    weight_sum=1.0,
                                    minimum_weight=0.25,
                                    maximum_weight=0.25,
                                    effective_sample_size=4.0,
                                    minimum_log_factor=0.0,
                                    maximum_log_factor=0.0,
                                )
                                for index in range(2)
                            ],
                        ),
                    )
                logs.append(logged)
            write_jsonl(folder / arm / "training.jsonl", logs)
        samples = []
        for arm in ARMS:
            tails = ("GGGGGG", "SSSSSS", "TTTTTT", "KKKKKK")
            if arm == "supervised":
                tails = ("GGGGGG", "TTTTTT", "TTTTTT", "KKKKKK")
            for prompt_index, prompt in enumerate(prompts):
                for index, tail in enumerate(tails):
                    prefix = prompt["sequence"][:4]
                    row = dict(
                        id=f"{name}-{arm}-{prompt['id']}-{index}",
                        arm=arm,
                        replicate=name,
                        seed=0,
                        prompt_id=prompt["id"],
                        prompt_index=prompt_index,
                        prefix=prefix,
                        tail=tail,
                        sequence=prefix + tail,
                        evaluation_start=4,
                        log_prior=-10.0,
                        log_policy=-10.0 if arm == "prior" else -9.0,
                    )
                    if arm == "gfn":
                        row["log_z"] = 0.0
                    samples.append(row)
        write_jsonl(folder / "samples.jsonl", samples)
        all_samples.extend(samples)
        write_json(
            folder / "costs.json",
            dict(
                training=dict(
                    updates_per_arm=2,
                    examples_per_arm=16,
                    fresh_trajectories=6,
                    seconds=3.0,
                    refresh_seconds=1.0,
                    paired_update_seconds=2.0,
                ),
                generation={arm: dict(seconds=1.0, trajectories=8) for arm in ARMS},
            ),
        )
        write_json(
            folder / "adaptation.json",
            dict(
                gfn=dict(last_block_delta_l2=0.2, last_block_relative_delta_l2=0.01),
                supervised=dict(
                    last_block_delta_l2=0.3, last_block_relative_delta_l2=0.02
                ),
                reference_has_gradients=False,
            ),
        )
    write_jsonl(root / "samples.jsonl", all_samples)
    return root / "evaluation"


class ObjectiveEvaluationTests(unittest.TestCase):
    def test_raw_denominators_and_conservative_cross_repeat_copy_union(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = evaluation_fixture(root)
            source_bytes = {
                path: path.read_bytes() for path in root.rglob("*") if path.is_file()
            }
            calls = []

            def predictor(sequences, version):
                calls.append((list(sequences), version))
                return fixture_predictor(sequences, version)

            report = evaluate_objective(root, output, predictor)
            self.assertTrue(report.exists())
            self.assertEqual([version for _, version in calls], ["V3", "V2"])
            metrics = json.loads((output / "metrics.json").read_text())
            row = metrics["by_replicate"]["replicate-1"]["gfn"]
            self.assertEqual(row["count"], 8)
            self.assertEqual(row["joint_success_rate"], 0.75)
            self.assertEqual(row["joint_success_without_common_training_copy"], 0.25)
            self.assertEqual(row["exact_common_training_copy_fraction"], 0.5)
            self.assertEqual(row["exact_own_training_copy_fraction"], 0.25)
            self.assertEqual(row["distinct_successful_tail_fraction"], 3 / 8)
            self.assertEqual(row["distinct_noncopy_successful_tail_fraction"], 1 / 8)
            self.assertEqual(row["successful_cluster_fraction"], 6 / 8)
            prior = metrics["by_replicate"]["replicate-1"]["prior"]
            self.assertEqual(prior["exact_own_training_copy_fraction"], 0.0)
            self.assertEqual(prior["exact_common_training_copy_fraction"], 0.5)
            self.assertNotIn("own_reward_tb_residual_rmse", prior)
            effects = metrics["paired_effects"]
            self.assertEqual(
                effects["mean_paired_gfn_minus_supervised"][
                    "joint_success_without_common_training_copy"
                ],
                -0.25,
            )
            self.assertEqual(len(effects["per_replicate_prompt"]), 4)
            training = json.loads((output / "training_summary.json").read_text())
            summary = training["replicates"]["replicate-1"]
            self.assertEqual(summary["examples_per_arm"], 16)
            self.assertEqual(
                summary["supervised_weighting"]["logged_weighting_updates_checked"], 2
            )
            self.assertEqual(
                summary["measured_update_seconds_by_arm"], dict(gfn=0.5, supervised=0.5)
            )
            self.assertEqual(
                training["costs"]["evaluation_predictor_counts"]["calls"], 2
            )
            saved = json.loads((output / "sequence_metrics.json").read_text())
            cache = {"V3": {}, "V2": {}}
            for record in saved:
                for version, disorder in record["disorder"].items():
                    cache[version][record["sequence"]] = disorder["full_scores"]
                    cache[version][record["tail"]] = disorder["isolated_tail_scores"]

            def cached_predictor(sequences, version):
                return [np.asarray(cache[version][sequence]) for sequence in sequences]

            recovered = root / "recovered"
            evaluate_objective(root, recovered, cached_predictor)
            for filename in (
                "sequence_metrics.json",
                "metrics.json",
                "decision.json",
                "training_summary.json",
                "report.md",
            ):
                self.assertEqual(
                    (output / filename).read_bytes(),
                    (recovered / filename).read_bytes(),
                )
            self.assertTrue(
                all(path.read_bytes() == value for path, value in source_bytes.items())
            )

    def test_screen_needs_both_repeats_and_all_quality_checks(self):
        arms = {
            arm: dict(
                joint_success_rate=rate,
                joint_success_without_common_training_copy=rate,
                unique_tail_fraction=0.9,
                v3_disorder_success_rate=0.9,
            )
            for arm, rate in (("gfn", 0.4), ("supervised", 0.3), ("prior", 0.1))
        }
        repeats = {"first": arms, "second": copy.deepcopy(arms)}
        self.assertTrue(objective_decision(repeats)["passes_practical_screen"])
        for metric, value in (
            ("joint_success_rate", 0.34),
            ("joint_success_without_common_training_copy", 0.34),
            ("unique_tail_fraction", 0.84),
            ("v3_disorder_success_rate", 0.84),
        ):
            with self.subTest(metric=metric):
                changed = copy.deepcopy(repeats)
                changed["second"]["gfn"][metric] = value
                decision = objective_decision(changed)
                self.assertFalse(decision["passes_practical_screen"])
                self.assertTrue(decision["replicates"]["first"]["passes"])
                self.assertIn(
                    "does not establish supervised superiority",
                    decision["interpretation"],
                )

    def test_clustering_uses_inclusive_identity_and_transitive_components(self):
        # First/second and second/third share 80%; first/third share only 60%.
        tails = ["AAAAAAAAAA", "CCAAAAAAAA", "CCCCAAAAAA", "GGGGGGGGGG"]
        self.assertEqual(cluster_sizes(tails + [tails[0]]), [4, 1])
        self.assertEqual(cluster_sizes([]), [])
        self.assertEqual(cluster_sizes(["AAAAAA", "CAAAAA"]), [2])
        with self.assertRaises(ValueError):
            cluster_sizes(["AAAA", "AAAAA"])

    def test_rejects_corrupted_or_missing_shared_training(self):
        for corruption in (
            "missing",
            "step",
            "counts",
            "unpaired",
            "holdout",
            "refresh",
            "weighting",
            "nan_reward",
        ):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = evaluation_fixture(root)
                folder = root / "replicate-1"
                path = folder / "shared_training.jsonl"
                if corruption == "missing":
                    path.unlink()
                elif corruption == "unpaired":
                    path = folder / "supervised/training.jsonl"
                    rows = read_jsonl(path)
                    rows[0]["example_ids"].reverse()
                    write_jsonl(path, rows)
                elif corruption == "weighting":
                    path = folder / "supervised/training.jsonl"
                    rows = read_jsonl(path)
                    rows[0]["weighting"]["prompts"][0]["maximum_weight"] = 1.0
                    write_jsonl(path, rows)
                elif corruption == "nan_reward":
                    path = folder / "training_rows.jsonl"
                    rows = read_jsonl(path)
                    rows[0]["log_prior"] = np.nan
                    write_jsonl(path, rows)
                else:
                    rows = read_jsonl(path)
                    if corruption == "step":
                        rows.pop()
                    elif corruption == "counts":
                        rows[0]["counts"]["prior"] += 1
                    elif corruption == "holdout":
                        rows[0]["example_ids"][-1] = "b-KKKKKK"
                    elif corruption == "refresh":
                        rows[1]["example_ids"] = list(reversed(rows[1]["example_ids"]))
                    write_jsonl(path, rows)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    evaluate_objective(root, output, fixture_predictor)

    def test_rejects_evaluation_counts_identity_support_and_reference_drift(self):
        for corruption in (
            "count",
            "duplicate",
            "prefix",
            "reference",
            "root_copy",
            "reference_gradient",
        ):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = evaluation_fixture(root)
                folder = root / "replicate-1"
                if corruption == "reference_gradient":
                    path = folder / "adaptation.json"
                    value = json.loads(path.read_text())
                    value["reference_has_gradients"] = True
                    write_json(path, value)
                else:
                    path = (
                        root / "samples.jsonl"
                        if corruption == "root_copy"
                        else folder / "samples.jsonl"
                    )
                    rows = read_jsonl(path)
                    if corruption in ("count", "root_copy"):
                        rows.pop()
                    elif corruption == "duplicate":
                        rows[1]["id"] = rows[0]["id"]
                    elif corruption == "prefix":
                        rows[0]["sequence"] = "AAAA" + rows[0]["tail"]
                    elif corruption == "reference":
                        next(row for row in rows if row["arm"] == "prior")[
                            "log_prior"
                        ] -= 1.0
                    write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    evaluate_objective(root, output, fixture_predictor)


if __name__ == "__main__":
    unittest.main()
