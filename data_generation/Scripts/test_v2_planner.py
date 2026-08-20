#!/usr/bin/env python3
"""Focused tests for the immutable combined V2 planner."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import v2_dataset_controller as controller
from v2_mission_catalog import (
    FAMILY_FRAME_SHARES,
    TYPE_FRAME_SHARE,
    mission_types,
    review_recipes,
)


def plan_args(root: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "collection": root, "frame_budget": 700_000, "workers": 1,
        "recipes_per_assignment": 8, "episode_seconds": 150,
        "observation_rate": 20, "width": 384, "height": 384,
        "storage_format": "webp_parquet", "webp_effort": 0,
        "seed_start": 200_000, "evaluation_percent": 10, "plan_id": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class V2PlannerTests(unittest.TestCase):
    def test_source_fingerprint_line_endings_are_platform_independent(self) -> None:
        expected = b"first line\nsecond line\nthird line\n"
        self.assertEqual(controller.canonical_source_bytes(expected), expected)
        self.assertEqual(
            controller.canonical_source_bytes(
                b"first line\r\nsecond line\r\nthird line\r\n"
            ),
            expected,
        )
        self.assertEqual(
            controller.canonical_source_bytes(
                b"first line\rsecond line\rthird line\r"
            ),
            expected,
        )

    def test_campaign_budget_rounds_to_exact_v2_targets(self) -> None:
        self.assertEqual(
            controller.rounded_source_targets(2_222_222),
            {"semi_markov": 1_555_555, "mission": 666_667},
        )

    def test_catalog_has_exactly_the_agreed_sixty_types_and_shares(self) -> None:
        values = mission_types()
        self.assertEqual(len(values), 62)
        self.assertEqual(len({item.slug for item in values}), 62)
        expected = {
            family: int(share / TYPE_FRAME_SHARE)
            for family, share in FAMILY_FRAME_SHARES.items()
        }
        self.assertEqual(Counter(item.family for item in values), Counter(expected))
        self.assertEqual(sum(FAMILY_FRAME_SHARES.values(), Fraction()), Fraction(3, 10))
        self.assertEqual(TYPE_FRAME_SHARE, Fraction(3, 620))

    def test_plan_is_exact_70_30_and_every_type_is_exactly_half_percent(self) -> None:
        recipes = controller.build_recipes(700_000, 150, 20, 200_000, 10)
        distribution = controller.planned_distribution(recipes)
        self.assertEqual(
            distribution["source_frames"],
            {"mission": 210_000, "semi_markov": 490_000},
        )
        self.assertEqual(len(distribution["mission_type_frames"]), 62)
        self.assertLessEqual(
            max(distribution["mission_type_frames"].values())
            - min(distribution["mission_type_frames"].values()), 1
        )

    def test_mandatory_pass_contains_every_type_before_deficit_scheduling(self) -> None:
        recipes = controller.build_recipes(700_000, 150, 20, 200_000, 10)
        mandatory = [item for item in recipes if item["schedule_phase"] == "mandatory_coverage"]
        self.assertEqual(len(mandatory), 63)  # one free-play opening plus 62 types
        mission_mandatory = [item for item in mandatory if item["source"] == "mission"]
        self.assertEqual(len({item["mission_type"] for item in mission_mandatory}), 62)
        first_deficit = next(index for index, item in enumerate(recipes) if item["schedule_phase"] == "frame_deficit")
        self.assertGreaterEqual(first_deficit, 61)
        first_families = {item["family"] for item in mission_mandatory[:7]}
        self.assertEqual(first_families, set(FAMILY_FRAME_SHARES))

    def test_largest_frame_deficit_schedule_is_deterministic(self) -> None:
        first = controller.build_recipes(700_000, 150, 20, 200_000, 10)
        second = controller.build_recipes(700_000, 150, 20, 200_000, 10)
        self.assertEqual(first, second)
        self.assertEqual(len({item["recipe_id"] for item in first}), len(first))
        self.assertEqual(len({item["replay_identity"] for item in first}), len(first))

    def test_calculated_floor_uses_mandatory_durations_and_family_shares(self) -> None:
        rate = 20
        grouped: dict[str, list] = defaultdict(list)
        for item in mission_types():
            grouped[item.family].append(item)
        candidates = [Fraction(150 * rate + 1, 1) / Fraction(7, 10)]
        for family, share in FAMILY_FRAME_SHARES.items():
            mandatory = sum(controller.mission_expected_frames(item, rate) for item in grouped[family])
            candidates.append(Fraction(mandatory, 1) / share)
        raw = max((item.numerator + item.denominator - 1) // item.denominator for item in candidates)
        expected = raw
        self.assertEqual(controller.minimum_feasible_frame_budget(150, rate), expected)
        self.assertEqual(expected, 33_274)
        with self.assertRaisesRegex(ValueError, "calculated minimum"):
            controller.build_recipes(expected - 200, 150, rate, 1, 10)

    def test_non_quantized_budget_is_accounted_exactly(self) -> None:
        recipes = controller.build_recipes(700_001, 150, 20, 1, 10)
        distribution = controller.planned_distribution(recipes)
        self.assertEqual(distribution["total_credited_frames"], 700_001)
        self.assertEqual(
            distribution["source_frames"],
            controller.rounded_source_targets(700_001),
        )

    def test_types_reach_both_splits_when_budget_permits(self) -> None:
        recipes = controller.build_recipes(700_000, 150, 20, 200_000, 10)
        splits: dict[str, set[str]] = defaultdict(set)
        for item in recipes:
            if item["source"] == "mission":
                splits[item["mission_type"]].add(item["split"])
        self.assertEqual(set(splits), {item.slug for item in mission_types()})
        self.assertTrue(all(value == {"train", "evaluation"} for value in splits.values()))

    def test_wall_variation_never_leaves_the_physical_wall_plane(self) -> None:
        recipes = controller.build_recipes(700_000, 150, 20, 200_000, 10)
        for recipe in recipes:
            mission = str(recipe.get("mission_type") or "")
            if "_wall_" not in mission or "two_wall" in mission:
                continue
            target = recipe["mission_solution"]["target_point"]
            if mission.startswith("north_"):
                self.assertEqual(target["x"], 1600.0, mission)
            elif mission.startswith("south_"):
                self.assertEqual(target["x"], -1600.0, mission)
            elif mission.startswith("east_"):
                self.assertEqual(target["y"], 1600.0, mission)
            elif mission.startswith("west_"):
                self.assertEqual(target["y"], -1600.0, mission)

    def test_repetitions_progressively_cover_cells_and_corner_orders(self) -> None:
        recipes = controller.build_recipes(700_000, 150, 20, 200_000, 10)
        cells: dict[str, set[str]] = defaultdict(set)
        orders: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        counts: Counter[str] = Counter()
        for recipe in recipes:
            if recipe["source"] != "mission":
                continue
            mission = recipe["mission_type"]
            solution = recipe["mission_solution"]
            counts[mission] += 1
            cells[mission].add(json.dumps(
                solution["variation"]["coverage_cell"], sort_keys=True
            ))
            if "two_wall" in mission:
                orders[mission].add(tuple(solution["expected_contact_order"]))
        self.assertTrue(all(len(cells[name]) >= min(16, count) for name, count in counts.items()))
        self.assertTrue(all(len(value) == 2 for value in orders.values()))

    def test_non_certified_observation_rates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 20 Hz"):
            controller.build_recipes(700_000, 150, 19, 1, 10)

    def test_assignment_mutation_and_replacement_fields_invalidate_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            with contextlib.redirect_stdout(io.StringIO()):
                controller.create_plan(plan_args(root))
            assignment_path = sorted((root / "assignments").glob("*.json"))[0]
            assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
            mission = next(item for item in assignment["recipes"] if item["source"] == "mission")
            mission["mission_solution"]["target_point"]["x"] += 1.0
            assignment["reserve_for"] = "friendlier-recipe"
            assignment_path.write_text(json.dumps(assignment), encoding="utf-8")
            result = controller.verify_plan(root)
            self.assertFalse(result["valid"])
            self.assertFalse(result["immutable_recipe_content"])
            self.assertFalse(result["no_replacement_fields"])

    def test_created_plan_verifies_and_has_no_replacement_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collection"
            with contextlib.redirect_stdout(io.StringIO()):
                controller.create_plan(plan_args(root))
            result = controller.verify_plan(root)
            self.assertTrue(result["valid"])
            self.assertTrue(result["mandatory_coverage_complete"])
            self.assertTrue(result["no_replacement_fields"])
            text = (root / "plan" / "recipes.jsonl").read_text(encoding="utf-8")
            for forbidden in ("reserve_for", "replacement_for", "candidate_seed"):
                self.assertNotIn(forbidden, text)

    def test_review_builder_has_three_separated_examples_for_every_type(self) -> None:
        values = review_recipes(20)
        self.assertEqual(len(values), 186)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for value in values:
            grouped[value["mission_type"]].append(value)
            self.assertEqual((value["width"], value["height"]), (384, 384))
        self.assertEqual(len(grouped), 62)
        for examples in grouped.values():
            self.assertEqual([item["repetition_index"] for item in examples], [0, 1, 2])
            for dimension in (
                "surface_u", "surface_v", "distance_u", "arc_u",
            ):
                self.assertEqual(
                    sorted(item["solution"]["variation"][dimension] for item in examples),
                    [0.0, 0.5, 1.0],
                )
            self.assertEqual(len({
                tuple(item["solution"]["player_spawn"].values()) for item in examples
            }), 3)

    def test_review_plan_command_writes_exactly_186_one_recipe_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "review"
            args = argparse.Namespace(
                collection=root,
                observation_rate=20,
                workers=1,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                controller.create_review_plan(args)
            recipes = [
                json.loads(line)
                for line in (root / "plan" / "recipes.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            assignments = sorted(
                (root / "assignments").glob("review-assignment-*.json")
            )
            self.assertEqual(len(recipes), 186)
            self.assertEqual(len(assignments), 186)
            self.assertTrue(all(
                len(json.loads(path.read_text(encoding="utf-8"))["recipes"]) == 1
                for path in assignments
            ))


if __name__ == "__main__":
    unittest.main()
