#!/usr/bin/env python3
"""Ledger tests for exact local packaged-build binding."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_worker import (
    bind_execution_build,
    build_validated_result,
    command_for,
    package_runtime_fingerprint,
    package_runtime_root,
    require_v2_certification,
)


class V2WorkerBuildBindingTests(unittest.TestCase):
    def test_v2_recording_requires_prior_bound_batch_certification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "game.exe"
            executable.write_bytes(b"game")
            (root / "plan").mkdir()
            (root / "assignments").mkdir()
            (root / "plan" / "collection-plan.json").write_text("{}\n")
            (root / "plan" / "recipes.jsonl").write_text("{}\n")
            plan = {
                "plan_id": "p",
                "plan_version": "trajectory-throw-v2-test",
                "assignment_count": 0,
                "active_recipe_count": 0,
                "generator_source_sha256": "source",
            }
            build = bind_execution_build(root, executable, plan)
            with self.assertRaisesRegex(ValueError, "requires certification"):
                require_v2_certification(root, plan, executable, build)

    def test_editor_command_opens_the_project_in_game_mode(self) -> None:
        command = command_for(
            Path("UnrealEditor-Cmd.exe"),
            Path("request.json"),
            Path("output"),
        )
        self.assertEqual(Path(command[1]).name, "he_grenade_game.uproject")
        self.assertEqual(command[2], "-game")

    def test_packaged_command_does_not_add_an_editor_project(self) -> None:
        command = command_for(Path("he_grenade_game.exe"), Path("request.json"), Path("output"))
        self.assertEqual(command[1], "-GenerateDataset")

    def test_collection_rejects_a_different_packaged_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            executable = package / "game.exe"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"bootstrap")
            runtime = package / "game" / "Binaries" / "Win64" / "game.exe"
            runtime.parent.mkdir(parents=True)
            runtime.write_bytes(b"runtime-one")
            generated_config = package / "game" / "Saved" / "Config" / "GameUserSettings.ini"
            generated_config.parent.mkdir(parents=True)
            generated_config.write_bytes(b"mutable-one")
            plan = {"plan_id": "plan-a", "generator_source_sha256": "source-a"}

            first = bind_execution_build(root / "collection", executable, plan)
            second = bind_execution_build(root / "collection", executable, plan)
            self.assertEqual(first, second)
            self.assertEqual(first["package_runtime_file_count"], 2)

            generated_config.write_bytes(b"mutable-two")
            self.assertEqual(
                first,
                bind_execution_build(root / "collection", executable, plan),
            )

            runtime.write_bytes(b"runtime-two")
            with self.assertRaisesRegex(ValueError, "immutable execution-build.json"):
                bind_execution_build(root / "collection", executable, plan)

    def test_linux_launcher_fingerprints_the_complete_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "Linux"
            launcher = package / "he_grenade_game.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_bytes(b"#!/bin/sh\n")
            elf = package / "he_grenade_game" / "Binaries" / "Linux" / "he_grenade_game"
            elf.parent.mkdir(parents=True)
            elf.write_bytes(b"elf-one")
            shared = package / "Engine" / "Binaries" / "ThirdParty" / "libRuntime.so"
            shared.parent.mkdir(parents=True)
            shared.write_bytes(b"shared-one")
            pak = package / "he_grenade_game" / "Content" / "Paks" / "game.pak"
            pak.parent.mkdir(parents=True)
            pak.write_bytes(b"pak-one")
            debug = elf.with_suffix(".debug")
            debug.write_bytes(b"debug-one")

            self.assertEqual(package_runtime_root(launcher), package)
            first = package_runtime_fingerprint(launcher)
            self.assertEqual(first[1], 4)

            debug.write_bytes(b"debug-two")
            self.assertEqual(package_runtime_fingerprint(launcher), first)

            shared.write_bytes(b"shared-two")
            self.assertNotEqual(package_runtime_fingerprint(launcher), first)

    def test_technically_valid_semi_markov_is_unconditionally_credited(self) -> None:
        assignment = {
            "plan_id": "p", "plan_version": "trajectory-throw-v2-sixty-missions-1",
            "assignment_id": "a", "recipes": [{
                "recipe_id": "r", "mission": "semi_markov",
                "mission_type": None, "source": "semi_markov", "family": "semi_markov",
                "scenario_index": 0, "planned_credited_frames": 100,
            }],
        }
        row = {
            "recipe_id": "r", "observation_count": 101,
            "v2_source": "semi_markov", "mission_success": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "dataset.json").write_text("{}\n", encoding="utf-8")
            with patch("dataset_worker.read_episode_rows", return_value=[row]):
                result = build_validated_result(assignment, "attempt-000", "worker", output)
        self.assertEqual(result["accepted_observation_frames"], 100)
        self.assertEqual(result["semantic_failure_recipe_ids"], [])

    def test_certified_mission_failure_is_recorded_without_replacement(self) -> None:
        assignment = {
            "plan_id": "p", "plan_version": "trajectory-throw-v2-sixty-missions-1",
            "assignment_id": "a", "recipes": [{
                "recipe_id": "r", "mission": "rectangle_north_face",
                "mission_type": "rectangle_north_face", "source": "mission",
                "family": "broad_object_surface", "scenario_index": 0,
                "planned_credited_frames": 100,
            }],
        }
        row = {
            "recipe_id": "r", "observation_count": 101,
            "v2_source": "mission", "v2_mission_type": "rectangle_north_face",
            "mission_success": False, "termination_reason": "mission_timeout",
            "v2_construction_certified": True, "v2_mission_event_frame": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "dataset.json").write_text("{}\n", encoding="utf-8")
            with patch("dataset_worker.read_episode_rows", return_value=[row]):
                result = build_validated_result(
                    assignment, "attempt-000", "worker", output
                )
        self.assertEqual(result["accepted_observation_frames"], 0)
        self.assertEqual(result["resolved_recipe_ids"], ["r"])
        self.assertEqual(result["semantic_failure_recipe_ids"], ["r"])
        self.assertEqual(result["semantic_result"], "resolved_with_failures")

    def test_visibility_degraded_mission_is_credited_and_tracked(self) -> None:
        assignment = {
            "plan_id": "p", "plan_version": "trajectory-throw-v2-sixty-missions-1",
            "assignment_id": "a", "recipes": [{
                "recipe_id": "r", "mission": "trajectory_manual_toggle_cycle",
                "mission_type": "trajectory_manual_toggle_cycle", "source": "mission",
                "family": "trajectory_control", "scenario_index": 0,
                "planned_credited_frames": 100,
            }],
        }
        row = {
            "recipe_id": "r", "observation_count": 101,
            "v2_source": "mission",
            "v2_mission_type": "trajectory_manual_toggle_cycle",
            "mission_success": True,
            "v2_visibility_degraded": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "dataset.json").write_text("{}\n", encoding="utf-8")
            with patch("dataset_worker.read_episode_rows", return_value=[row]):
                result = build_validated_result(
                    assignment, "attempt-000", "worker", output
                )
        self.assertEqual(result["accepted_observation_frames"], 100)
        self.assertEqual(result["semantic_failure_recipe_ids"], [])
        self.assertEqual(result["visibility_degraded_recipe_count"], 1)
        self.assertEqual(result["visibility_degraded_recipe_ids"], ["r"])


if __name__ == "__main__":
    unittest.main()
