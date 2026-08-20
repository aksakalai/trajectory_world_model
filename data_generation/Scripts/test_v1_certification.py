#!/usr/bin/env python3
"""Tests for V1 no-capture certification and recording gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_worker import (
    bind_execution_build,
    require_v1_certification,
    v1_certification_bindings,
    v1_certification_recipe_counts,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class V1CertificationTests(unittest.TestCase):
    def make_collection(self, root: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
        executable = root / "package" / "game.exe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"v1-runtime")
        plan = {
            "plan_id": "plan-v1",
            "plan_version": "movement-v1-test",
            "worker_count": 1,
            "active_recipe_count": 2,
            "assignment_count": 1,
        }
        write_json(root / "plan" / "collection-plan.json", plan)
        recipes = [
            {"recipe_id": "semi", "mission": "semi_markov", "active": True},
            {"recipe_id": "guided", "mission": "object_view", "active": True},
            {
                "recipe_id": "reserve",
                "mission": "contact_recovery",
                "active": False,
                "reserve_for": "guided",
            },
        ]
        recipes_path = root / "plan" / "recipes.jsonl"
        recipes_path.write_text(
            "".join(json.dumps(recipe) + "\n" for recipe in recipes),
            encoding="utf-8",
        )
        build = bind_execution_build(root, executable, plan)
        return plan, executable, build

    def test_recipe_inventory_certifies_guided_reserves_but_not_semi_markov(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_collection(root)
            self.assertEqual(
                v1_certification_recipe_counts(root / "plan" / "recipes.jsonl"),
                (2, 1),
            )

    def test_v1_recording_requires_prior_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, executable, build = self.make_collection(root)
            with self.assertRaisesRegex(ValueError, "requires certification"):
                require_v1_certification(root, plan, executable, build)

    def test_matching_zero_rejection_certificate_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, executable, build = self.make_collection(root)
            report = {
                "complete": True,
                "certified_count": 2,
                "failed_count": 0,
                "bindings": v1_certification_bindings(root, plan, executable, build),
            }
            write_json(
                root / "certification" / "certification-report-bound.json", report
            )
            require_v1_certification(root, plan, executable, build)

    def test_recipe_or_build_drift_invalidates_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, executable, build = self.make_collection(root)
            report = {
                "complete": True,
                "certified_count": 2,
                "failed_count": 0,
                "bindings": v1_certification_bindings(root, plan, executable, build),
            }
            write_json(
                root / "certification" / "certification-report-bound.json", report
            )
            with (root / "plan" / "recipes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"recipe_id": "new", "mission": "hoop_pass"}) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete|does not bind"):
                require_v1_certification(root, plan, executable, build)

if __name__ == "__main__":
    unittest.main()
