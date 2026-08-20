#!/usr/bin/env python3
"""Fast replacement-policy tests; does not launch Unreal."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_controller as v1
import resolve_plan_certification as resolver
import v2_dataset_controller as v2


class ReplacementPolicyTests(unittest.TestCase):
    def test_v1_attempts_are_distinct_same_slot_and_progressively_safer(self) -> None:
        calibration = v1.load_duration_calibration()
        failed = v1.build_recipe(
            calibration, "plan-test", 100, "hoop_pass", 3, 2,
            schedule_phase="base",
        )
        failed["planned_credited_frames"] = 100
        plan = {
            "plan_id": "plan-test",
            "generator": {"seed_start": 41_000},
        }
        candidates = resolver.v1_candidate_sequence(
            failed, plan, calibration, {100}, 3, 10
        )
        self.assertEqual(len(candidates), 10)
        self.assertEqual(len({item["recipe_id"] for item in candidates}), 10)
        self.assertTrue(all(item["mission"] == "hoop_pass" for item in candidates))
        self.assertTrue(all(item["scenario_index"] == 3 for item in candidates))
        scores = [
            resolver._v1_conservative_score(41_000, int(item["episode_index"]))
            for item in candidates
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_progressive_selection_ends_at_safest_candidate(self) -> None:
        selected = resolver.progressively_safer(
            [(f"candidate-{value}", float(value)) for value in range(100)], 10
        )
        self.assertEqual(len(selected), len(set(selected)))
        scores = [int(value.rsplit("-", 1)[1]) for value in selected]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(scores[-1], 0)

    def test_v2_replacements_extend_beyond_original_budget_sample_count(self) -> None:
        recipes = v2.build_recipes(40_000, 150, 20, 51_000, 10)
        distribution = v2.planned_distribution(recipes)
        failed = next(item for item in recipes if item["source"] == "mission")
        same_type = [
            item for item in recipes
            if item.get("mission_type") == failed["mission_type"]
        ]
        next_repetition = max(int(item["repetition_index"]) for item in same_type) + 1
        plan = {
            "generator": {
                "observation_rate": 20,
                "seed_start": 51_000,
                "evaluation_percent": 10,
            },
            "planned_distribution": distribution,
        }
        candidates = resolver.v2_candidate_sequence(
            failed, plan, {int(item["recipe_index"]) for item in recipes},
            next_repetition, 10,
        )
        self.assertEqual(len(candidates), 10)
        self.assertTrue(all(
            not v2.recipe_validation_errors(candidate, plan["generator"])
            for candidate in candidates
        ))

    def test_v2_resolved_collection_remains_structurally_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            destination = Path(temporary) / "resolved"
            args = argparse.Namespace(
                collection=source, frame_budget=40_000, workers=1,
                recipes_per_assignment=8, episode_seconds=150,
                observation_rate=20, width=64, height=64,
                storage_format="webp_parquet", webp_effort=0,
                seed_start=52_000, evaluation_percent=10, plan_id=None,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                v2.create_plan(args)
            plan = v2.read_json(source / "plan" / "collection-plan.json")
            recipes = v2.read_jsonl(source / "plan" / "recipes.jsonl")
            failed = next(item for item in recipes if item["source"] == "mission")
            same_type = [item for item in recipes if item.get("mission_type") == failed["mission_type"]]
            candidates = resolver.v2_candidate_sequence(
                failed, plan, {int(item["recipe_index"]) for item in recipes},
                max(int(item["repetition_index"]) for item in same_type) + 1, 1,
            )
            selected = [
                candidates[0] if item["recipe_id"] == failed["recipe_id"] else item
                for item in recipes
            ]
            resolver.write_resolved_collection(
                "v2", source, destination, selected, "resolution-test"
            )
            self.assertTrue(v2.verify_plan(destination)["valid"])

    def test_v1_resolved_collection_preserves_exact_slots_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            destination = Path(temporary) / "resolved"
            args = argparse.Namespace(
                collection=source, frame_budget=1_000_000, workers=1,
                recipes_per_assignment=32, tail_single_recipes=64,
                episode_seconds=10, observation_rate=20, width=64, height=64,
                storage_format="webp_parquet", webp_effort=0,
                seed_start=53_000, duration_calibration=v1.DEFAULT_DURATION_CALIBRATION,
                split="train", plan_id=None, allow_infeasible_diagnostic=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                v1.create_plan(args)
            plan = v1.read_json(source / "plan" / "collection-plan.json")
            all_recipes = v1.read_jsonl(source / "plan" / "recipes.jsonl")
            recipes = [item for item in all_recipes if item["active"]]
            failed = next(item for item in recipes if item["mission"] != "semi_markov")
            key_recipes = [
                item for item in all_recipes
                if item["mission"] == failed["mission"]
                and int(item["scenario_index"]) == int(failed["scenario_index"])
            ]
            candidates = resolver.v1_candidate_sequence(
                failed, plan, v1.load_duration_calibration(),
                {int(item["recipe_index"]) for item in all_recipes},
                max(int(item["repetition_index"]) for item in key_recipes) + 1, 1,
            )
            selected = [
                candidates[0] if item["recipe_id"] == failed["recipe_id"] else item
                for item in recipes
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                resolved_plan = resolver.write_resolved_collection(
                    "v1", source, destination, selected, "resolution-test"
                )
            base = selected[: int(resolved_plan["base_recipe_count"])]
            self.assertEqual(
                {
                    mission: sum(
                        int(item["planned_credited_frames"])
                        for item in base if item["mission"] == mission
                    )
                    for mission in v1.MISSION_COUNTS
                },
                resolved_plan["target_frames_by_mission"],
            )


if __name__ == "__main__":
    unittest.main()
