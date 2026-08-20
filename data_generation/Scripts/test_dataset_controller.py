#!/usr/bin/env python3
"""Fast controller contract tests; does not launch Unreal."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_controller as controller
import dataset_worker as worker


def plan_args(
    root: Path,
    *,
    budget: int = 1200000,
    workers: int = 1,
    seed_start: int = 1000,
    allow_infeasible_diagnostic: bool = False,
) -> Namespace:
    return Namespace(
        collection=root,
        frame_budget=budget,
        workers=workers,
        recipes_per_assignment=32,
        tail_single_recipes=64,
        episode_seconds=10,
        observation_rate=20,
        width=64,
        height=64,
        storage_format="webp_parquet",
        webp_effort=0,
        seed_start=seed_start,
        duration_calibration=controller.DEFAULT_DURATION_CALIBRATION,
        split="train",
        plan_id=None,
        allow_infeasible_diagnostic=allow_infeasible_diagnostic,
    )


def create(
    root: Path, *, budget: int = 1200000, workers: int = 1, seed_start: int = 1000
) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        controller.create_plan(
            plan_args(root, budget=budget, workers=workers, seed_start=seed_start)
        )


class ControllerContractTests(unittest.TestCase):
    def test_campaign_budget_rounds_to_exact_v1_targets(self) -> None:
        self.assertEqual(
            controller.rounded_frame_targets(1_111_111),
            {
                "semi_markov": 777_778,
                "object_view": 146_666,
                "contact_recovery": 110_000,
                "ramp_traverse": 36_667,
                "hoop_pass": 36_667,
                "static_no_input": 3_333,
            },
        )

    def test_complete_catalog_and_single_worker_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            one = base / "one"
            create(one, workers=1)
            recipes_one = controller.read_jsonl(one / "plan" / "recipes.jsonl")
            active_one = [recipe for recipe in recipes_one if recipe["active"]]
            self.assertGreater(len(active_one), sum(controller.MISSION_COUNTS.values()))
            plan = controller.read_json(one / "plan" / "collection-plan.json")
            self.assertEqual(plan["worker_count"], 1)
            self.assertTrue(all(
                controller.read_json(path)["logical_worker_id"] == 0
                for path in (one / "assignments").glob("*.json")
            ))
            self.assertEqual(
                len({(recipe["mission"], recipe["scenario_index"]) for recipe in active_one[: sum(controller.MISSION_COUNTS.values())]}),
                sum(controller.MISSION_COUNTS.values()),
            )

    def test_multiple_workers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "exactly one worker"):
                create(Path(temporary) / "four", workers=4)

    def test_larger_budget_extends_same_canonical_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            small = base / "small"
            large = base / "large"
            create(small, budget=controller.minimum_feasible_frame_budget() + 1000)
            create(large, budget=1250000)
            fields = (
                "recipe_index",
                "episode_index",
                "mission",
                "scenario_index",
                "continuous_sample_ordinal",
                "refinement_level",
                "repetition_index",
            )
            small_recipes = [recipe for recipe in controller.read_jsonl(small / "plan" / "recipes.jsonl") if recipe["active"]]
            large_recipes = [recipe for recipe in controller.read_jsonl(large / "plan" / "recipes.jsonl") if recipe["active"]]
            self.assertGreater(len(large_recipes), len(small_recipes))
            self.assertEqual(
                [tuple(recipe[field] for field in fields) for recipe in small_recipes],
                [tuple(recipe[field] for field in fields) for recipe in large_recipes[: len(small_recipes)]],
            )

    def test_infeasible_budget_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            minimum = controller.minimum_feasible_frame_budget()
            with self.assertRaisesRegex(ValueError, "infeasible"):
                create(Path(temporary) / "bad", budget=minimum - 1)

    def test_infeasible_diagnostic_plan_is_explicit_and_budget_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostic"
            minimum = controller.minimum_feasible_frame_budget()
            args = plan_args(
                root,
                budget=minimum - 1,
                allow_infeasible_diagnostic=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                controller.create_plan(args)

            plan = controller.read_json(root / "plan" / "collection-plan.json")
            self.assertTrue(plan["diagnostic_only"])
            self.assertFalse(plan["distribution_feasible"])
            self.assertIn("diagnostic only", plan["budget_completion_policy"])

            assignment = controller.read_json(
                sorted((root / "assignments").glob("*.json"))[0]
            )
            controller.write_new_json(
                root / "results" / "diagnostic.json",
                {
                    "technical_result": "validated",
                    "assignment_id": assignment["assignment_id"],
                    "accepted_observation_frames": minimum - 1,
                    "resolved_recipe_ids": [],
                    "semantic_failure_recipe_ids": [],
                    "credited_cells": [],
                },
            )
            inventory = controller.build_inventory(root)
            self.assertTrue(inventory["budget_reached"])
            self.assertFalse(inventory["coverage_complete"])
            self.assertTrue(inventory["complete"])

    def test_planned_frame_distribution_and_nested_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            create(root, budget=1200000)
            plan = controller.read_json(root / "plan" / "collection-plan.json")
            recipes = controller.read_jsonl(root / "plan" / "recipes.jsonl")[: plan["base_recipe_count"]]
            expected = {
                mission: sum(recipe["expected_credited_frames"] for recipe in recipes if recipe["mission"] == mission)
                for mission in controller.MISSION_COUNTS
            }
            total = sum(expected.values())
            for mission, target in controller.MISSION_FRAME_SHARES.items():
                self.assertAlmostEqual(expected[mission] / total, float(target), delta=0.002)

            object_recipes = [recipe for recipe in recipes if recipe["mission"] == "object_view"]
            modes = {}
            for recipe in object_recipes:
                mode = recipe["cell"]["mode"]
                modes[mode] = modes.get(mode, 0.0) + recipe["expected_credited_frames"]
            object_frames = sum(modes.values())
            for mode, target in controller.OBJECT_MODE_SHARES.items():
                self.assertAlmostEqual(modes[mode] / object_frames, float(target), delta=0.012)

            contact = [recipe for recipe in recipes if recipe["mission"] == "contact_recovery"]
            facings = {}
            for recipe in contact:
                facing = recipe["cell"]["facing"]
                facings[facing] = facings.get(facing, 0.0) + recipe["expected_credited_frames"]
            # The mandatory contact catalog is deliberately finite and complete;
            # a smaller budget need not manufacture a near-uniform nested mix.
            self.assertEqual(set(facings), set(controller.FACING_SHARES))
            self.assertTrue(all(frames > 0 for frames in facings.values()))

    def test_nested_feasibility_and_tail_assignment_boundary(self) -> None:
        requirements = controller.feasibility_requirements()
        self.assertEqual(
            controller.minimum_feasible_frame_budget(),
            requirements["static_no_input"]["minimum_total_budget"],
        )
        self.assertEqual(
            requirements["static_no_input"]["limiting_requirement"],
            "complete_mission_catalog",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            create(root, budget=1200000)
            plan = controller.read_json(root / "plan" / "collection-plan.json")
            assignments = [controller.read_json(path) for path in sorted((root / "assignments").glob("*.json"))]
            for assignment in assignments:
                phases = {recipe["schedule_phase"] for recipe in assignment["recipes"]}
                self.assertFalse("tail" in phases and len(phases) > 1)
            base_end = int(plan["base_recipe_count"]) - 1
            ending_assignment = next(
                assignment
                for assignment in assignments
                if any(int(recipe["recipe_index"]) == base_end for recipe in assignment["recipes"])
            )
            self.assertEqual(len(ending_assignment["recipes"]), 1)

    def test_seed_changes_immutable_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = base / "first"
            second = base / "second"
            create(first, seed_start=1000)
            create(second, seed_start=1001)
            first_plan = controller.read_json(first / "plan" / "collection-plan.json")
            second_plan = controller.read_json(second / "plan" / "collection-plan.json")
            self.assertNotEqual(first_plan["plan_id"], second_plan["plan_id"])

    def test_duplicate_validated_result_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            create(root)
            assignment = controller.read_json(sorted((root / "assignments").glob("*.json"))[0])
            result = {
                "technical_result": "validated",
                "assignment_id": assignment["assignment_id"],
                "accepted_observation_frames": 10,
                "resolved_recipe_ids": [recipe["recipe_id"] for recipe in assignment["recipes"]],
                "semantic_failure_recipe_ids": [],
                "credited_cells": [
                    {"mission": recipe["mission"], "scenario_index": recipe["scenario_index"]}
                    for recipe in assignment["recipes"]
                ],
            }
            controller.write_new_json(root / "results" / "a.json", result)
            controller.write_new_json(root / "results" / "b.json", result)
            inventory = controller.build_inventory(root)
            self.assertEqual(inventory["successful_assignment_count"], 1)
            self.assertEqual(inventory["accepted_observation_frames"], 10)
            self.assertEqual(inventory["duplicate_validated_results_ignored"], ["b.json"])

    def test_semantic_failure_frames_are_produced_but_not_credited(self) -> None:
        assignment = {
            "plan_id": "plan-test",
            "assignment_id": "assignment-test",
            "recipes": [
                {"recipe_id": "semi", "mission": "semi_markov", "scenario_index": 0},
                {"recipe_id": "guided-ok", "mission": "ramp_traverse", "scenario_index": 0},
                {"recipe_id": "guided-fail", "mission": "hoop_pass", "scenario_index": 0},
            ],
        }
        rows = [
            {"recipe_id": "semi", "observation_count": 201, "mission_success": False},
            {"recipe_id": "guided-ok", "observation_count": 45, "mission_success": True},
            {"recipe_id": "guided-fail", "observation_count": 42, "mission_success": False},
        ]
        with patch.object(worker, "read_episode_rows", return_value=rows):
            result = worker.build_validated_result(assignment, "attempt-000", "test", Path("unused"))
        self.assertEqual(result["produced_observation_frames"], 288)
        self.assertEqual(result["accepted_observation_frames"], 246)
        self.assertEqual(result["accepted_frames_by_mission"], {"semi_markov": 201, "ramp_traverse": 45})
        self.assertEqual(result["semantic_failure_recipe_ids"], ["guided-fail"])

    def test_stop_file_prevents_new_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            create(root)
            (root / "STOP").write_text("stop\n", encoding="utf-8")
            worker = Path(__file__).with_name("dataset_worker.py")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    str(root),
                    "--executable",
                    str(root / "not-used.exe"),
                    "--worker-id",
                    "0",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((root / "claims").exists())

    def test_orphaned_claim_advances_retry_and_is_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            create(root)
            assignment = controller.read_json(sorted((root / "assignments").glob("*.json"))[0])
            assignment_id = assignment["assignment_id"]
            self.assertIsNotNone(worker.claim_attempt(root, assignment_id, 0, "killed-worker"))
            self.assertEqual(worker.next_attempt(root, assignment_id), 1)
            inventory = controller.build_inventory(root)
            self.assertEqual(
                inventory["interrupted_claims_without_result"],
                [
                    {
                        "assignment_id": assignment_id,
                        "attempt_id": "attempt-000",
                        "executor_id": "killed-worker",
                        "claimed_utc": inventory["interrupted_claims_without_result"][0]["claimed_utc"],
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
