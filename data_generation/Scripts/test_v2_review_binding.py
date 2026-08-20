#!/usr/bin/env python3
"""Exact plan/assignment/recipe binding for the V2 human-review renderer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_v2_review_set import result_output, sha256_file


class V2ReviewBindingTests(unittest.TestCase):
    def test_unrelated_validated_output_cannot_replace_planned_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "captured"
            output.mkdir()
            dataset = output / "dataset.json"
            dataset.write_text(json.dumps({
                "plan_id": "plan-a", "assignment_id": "assignment-a",
            }), encoding="utf-8")
            recipe = {
                "recipe_id": "recipe-a",
                "replay_identity": "replay-a",
                "mission_type": "rectangle_north_face",
                "mission_solution": {"target_point": {"x": 1}},
            }
            assignment = {
                "plan_id": "plan-a",
                "assignment_id": "assignment-a",
                "assignment_digest": "assignment-digest-a",
                "recipes": [recipe],
            }
            (root / "assignments").mkdir()
            (root / "assignments" / "assignment-a.json").write_text(
                json.dumps(assignment), encoding="utf-8"
            )
            (root / "results").mkdir()
            result_path = root / "results" / "assignment-a--attempt-000.json"
            result = {
                "technical_result": "validated",
                "plan_id": "plan-a",
                "assignment_id": "assignment-a",
                "assignment_digest": "assignment-digest-a",
                "resolved_recipe_ids": ["recipe-a"],
                "recipe_bindings": [{
                    "recipe_id": "recipe-a",
                    "replay_identity": "replay-a",
                    "mission_type": "rectangle_north_face",
                    "mission_solution": {"target_point": {"x": 1}},
                }],
                "output_directory": str(output),
                "output_dataset_sha256": sha256_file(dataset),
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            entry = {
                "assignment_id": "assignment-a",
                "recipe_id": "recipe-a",
                "mission_type": "rectangle_north_face",
                "solution": {"target_point": {"x": 1}},
            }
            row = {
                "plan_id": "plan-a", "assignment_id": "assignment-a",
                "recipe_id": "recipe-a", "v2_replay_identity": "replay-a",
                "v2_mission_type": "rectangle_north_face",
            }
            with patch("dataset_worker.read_episode_rows", return_value=[row]):
                self.assertEqual(result_output(root, entry, "plan-a"), output.resolve())

            result["recipe_bindings"][0]["mission_solution"] = {
                "target_point": {"x": 999}
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact recipe"):
                result_output(root, entry, "plan-a")


if __name__ == "__main__":
    unittest.main()
