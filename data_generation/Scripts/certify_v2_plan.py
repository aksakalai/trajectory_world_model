#!/usr/bin/env python3
"""Batch-certify every immutable V2 recipe in one Unreal session."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import v2_dataset_controller as controller
import dataset_worker


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("collection", type=Path)
    result.add_argument("--executable", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    root = args.collection.resolve()
    executable = args.executable.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"certification output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    plan_path = root / "plan" / "collection-plan.json"
    recipes_path = root / "plan" / "recipes.jsonl"
    plan = controller.read_json(plan_path)
    if not str(plan.get("plan_version", "")).startswith("trajectory-throw-v2"):
        raise ValueError("certification requires a Trajectory/Throw V2 plan")
    if plan.get("worker_count") != 1:
        raise ValueError("certification requires the repository-wide one-worker invariant")
    execution_build = dataset_worker.bind_execution_build(root, executable, plan)

    assignments: list[dict[str, Any]] = []
    for path in sorted((root / "assignments").glob("*.json")):
        assignment = controller.read_json(path)
        controller.validate_assignment_against_plan(root, assignment)
        if assignment.get("logical_worker_id") != 0:
            raise ValueError(f"{assignment['assignment_id']} is not owned by worker 0")
        assignments.append(assignment)
    if len(assignments) != int(plan["assignment_count"]):
        raise ValueError(
            f"plan requires {plan['assignment_count']} assignments; found {len(assignments)}"
        )
    recipes = [recipe for assignment in assignments for recipe in assignment["recipes"]]
    if len(recipes) != int(plan["active_recipe_count"]):
        raise ValueError(
            f"plan requires {plan['active_recipe_count']} active recipes; found {len(recipes)}"
        )

    manifest = {
        **assignments[0],
        "assignment_id": "plan-construction-certification",
        "assignment_number": 0,
        "dispatch_wave": 0,
        "logical_worker_id": 0,
        "recipes": recipes,
    }
    manifest.pop("assignment_digest", None)
    manifest["execution_build"] = execution_build
    manifest_path = output / "certification-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    command = [str(executable)]
    if "unrealeditor" in executable.name.lower():
        command.extend([str(Path(__file__).resolve().parent.parent / "he_grenade_game.uproject"), "-game"])
    command.extend([
        "-GenerateDataset",
        "-CertifyV2PlanOnly",
        f"-RecipeManifest={manifest_path}",
        f"-Output={output}",
        "-RenderOffscreen",
        "-unattended",
        "-nosound",
        "-NoSplash",
        "-NoVSync",
    ])
    completed = subprocess.run(command, check=False)
    engine_report_path = output / "certification-report.json"
    if not engine_report_path.exists():
        raise RuntimeError(
            f"Unreal exited {completed.returncode} without a certification report"
        )
    engine_report = controller.read_json(engine_report_path)
    assignment_digests = [assignment["assignment_digest"] for assignment in assignments]
    bindings = {
        "plan_id": plan["plan_id"],
        "plan_version": plan["plan_version"],
        "collection_plan_sha256": sha256_file(plan_path),
        "recipes_jsonl_sha256": sha256_file(recipes_path),
        "assignment_digest_sha256": hashlib.sha256(
            canonical_json(assignment_digests).encode("utf-8")
        ).hexdigest(),
        "assignment_count": len(assignments),
        "recipe_count": len(recipes),
        "executable_path": str(executable),
        "executable_sha256": sha256_file(executable),
        "executable_size_bytes": executable.stat().st_size,
        "generator_source_sha256": plan.get("generator_source_sha256"),
        "package_runtime_sha256": execution_build.get("package_runtime_sha256"),
        "package_runtime_file_count": execution_build.get("package_runtime_file_count"),
    }
    report = {**engine_report, "bindings": bindings}
    if (
        engine_report.get("plan_id") != plan["plan_id"]
        or engine_report.get("plan_version") != plan["plan_version"]
        or int(engine_report.get("recipe_count", -1)) != len(recipes)
    ):
        raise ValueError("Unreal certification report does not bind to the requested plan")
    bound_path = output / "certification-report-bound.json"
    bound_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "complete": bool(report.get("complete")),
        "certified_count": report.get("certified_count"),
        "failed_count": report.get("failed_count"),
        "report": str(bound_path),
    }, indent=2, sort_keys=True))
    return 0 if report.get("complete") and completed.returncode == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
