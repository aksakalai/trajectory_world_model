#!/usr/bin/env python3
"""Geometry, variation, timing, and camera-construction tests for all 62 types."""

from __future__ import annotations

import math
import unittest

from v2_mission_catalog import (
    CANONICAL_PHYSICS_ID,
    build_solution,
    mission_types,
)


class V2MissionCatalogTests(unittest.TestCase):
    def test_every_solution_uses_low_arc_and_one_canonical_identity(self) -> None:
        for item in mission_types():
            solution = build_solution(item, 0, 20)
            self.assertEqual(solution["ballistic_branch"], "low", item.slug)
            self.assertEqual(solution["launch_speed_cm_per_second"], 1400.0)
            self.assertEqual(solution["canonical_physics_id"], CANONICAL_PHYSICS_ID)
            self.assertNotIn("restitution", str(solution).lower())
            self.assertNotIn("damping", str(solution).lower())
            self.assertNotIn("bounce_count", solution)
            self.assertNotIn("rest_position", solution)

    def test_human_readable_timing_for_every_type(self) -> None:
        for item in mission_types():
            timing = build_solution(item, 0, 20)["timing"]
            self.assertGreaterEqual(timing["establish_steps"], 15, item.slug)
            self.assertGreaterEqual(timing["preview_dwell_steps"], 12, item.slug)
            # Isolated named missions open at their solved throwing rotation;
            # camera acquisition is covered by the semi-Markov recipes.
            self.assertEqual(timing["camera_adjust_steps"], 0, item.slug)
            self.assertGreater(
                timing["total_observation_frames"],
                timing["throw_source_frame"] + 20,
                item.slug,
            )

    def test_camera_railguards_are_frozen_construction_invariants(self) -> None:
        for item in mission_types():
            guards = build_solution(item, 3, 20)["camera_railguards"]
            self.assertEqual(
                guards,
                {
                    "region_visible_all_frames": True,
                    "preview_and_region_visible_during_q": True,
                    "opening_has_arena_context": True,
                },
                item.slug,
            )

    def test_ramp_crossovers_start_on_opposite_sides_and_cross_body(self) -> None:
        by_slug = {item.slug: item for item in mission_types()}
        left = build_solution(by_slug["ramp_crossover_left_to_right"], 0, 20)
        right = build_solution(by_slug["ramp_crossover_right_to_left"], 0, 20)
        self.assertLess(left["player_spawn"]["y"], -130.0)
        self.assertGreater(right["player_spawn"]["y"], 130.0)
        for solution in (left, right):
            target = solution["target_point"]
            self.assertLessEqual(abs(target["x"]), 250.0)
            self.assertLessEqual(abs(target["y"]), 130.0)

    def test_edge_and_apex_corridors_are_physical_not_zero_width(self) -> None:
        edges = [
            item for item in mission_types()
            if item.family == "object_edge_apex" or "edge" in item.event_kind
        ]
        self.assertTrue(edges)
        self.assertTrue(all(item.region_radius_cm >= 20.0 for item in edges))

    def test_rectangle_nw_edge_paths_stay_clear_of_central_ramp(self) -> None:
        item = next(
            value for value in mission_types()
            if value.slug == "rectangle_nw_vertical_edge"
        )
        for repetition in range(3):
            solution = build_solution(item, repetition, 20, sample_count=3)
            # The ramp spans only the central Y corridor.  Both launch and
            # target remain on the rectangle's west-side Y corridor.
            self.assertGreater(solution["player_spawn"]["y"], 200.0)
            self.assertGreater(solution["target_point"]["y"], 400.0)

    def test_budget_ramp_targets_follow_authored_downhill_plane(self) -> None:
        by_slug = {item.slug: item for item in mission_types()}
        for slug in (
            "ramp_uphill_surface",
            "ramp_downhill_surface",
            "ramp_left_side_edge",
            "ramp_right_side_edge",
        ):
            first = build_solution(by_slug[slug], 0, 20, sample_count=35)["target_point"]
            last = build_solution(by_slug[slug], 34, 20, sample_count=35)["target_point"]
            self.assertNotEqual(last["x"], first["x"], slug)
            self.assertLess(
                (last["x"] - first["x"]) * (last["z"] - first["z"]),
                0.0,
                slug,
            )

    def test_budget_apex_targets_remain_on_pyramid_surface(self) -> None:
        item = next(value for value in mission_types() if value.slug == "pyramid_apex")
        for repetition in range(37):
            target = build_solution(item, repetition, 20, sample_count=37)["target_point"]
            dx = abs(target["x"] - 700.0)
            dy = abs(target["y"] - 700.0)
            expected_z = 250.0 - (250.0 / 140.0) * max(dx, dy)
            self.assertAlmostEqual(target["z"], expected_z, places=6)

    def test_continuous_variations_are_deterministic_and_well_separated(self) -> None:
        for item in mission_types():
            first = [build_solution(item, repetition, 20) for repetition in (0, 5, 17)]
            second = [build_solution(item, repetition, 20) for repetition in (0, 5, 17)]
            self.assertEqual(first, second)
            positions = [solution["player_spawn"] for solution in first]
            distances = []
            for left_index in range(3):
                for right_index in range(left_index + 1, 3):
                    left, right = positions[left_index], positions[right_index]
                    distances.append(math.dist(
                        (left["x"], left["y"], left["z"]),
                        (right["x"], right["y"], right["z"]),
                    ))
            self.assertGreater(max(distances), 20.0, item.slug)


if __name__ == "__main__":
    unittest.main()
