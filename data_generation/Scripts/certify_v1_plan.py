#!/usr/bin/env python3
"""Runtime-certify every immutable guided V1 mission without recording RGB."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import dataset_controller as controller
import dataset_worker


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    if not str(plan.get("plan_version", "")).startswith("movement-v1"):
        raise ValueError("certification requires a Movement V1 plan")
    if plan.get("worker_count") != 1:
        raise ValueError("certification requires the repository-wide one-worker invariant")

    # Reuse the existing structural verifier before entering Unreal. It checks
    # global recipe identity, complete discrete coverage, and active assignment
    # membership without recording any data.
    controller.verify_plan(argparse.Namespace(collection=root))
    execution_build = dataset_worker.bind_execution_build(root, executable, plan)

    all_recipes = read_jsonl(recipes_path)
    guided_recipes = [
        recipe for recipe in all_recipes if recipe.get("mission") != "semi_markov"
    ]
    semi_markov_count = len(all_recipes) - len(guided_recipes)
    if not guided_recipes:
        raise ValueError("V1 plan contains no guided mission recipes to certify")

    manifest = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "plan_version": plan["plan_version"],
        "assignment_id": "plan-runtime-mission-certification",
        "assignment_number": 0,
        "dispatch_wave": 0,
        "logical_worker_id": 0,
        "split": plan["split"],
        "generator": plan["generator"],
        "recipes": guided_recipes,
        "execution_build": execution_build,
    }
    manifest_path = output / "certification-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    command = [str(executable)]
    if "unrealeditor" in executable.name.lower():
        command.extend([
            str(Path(__file__).resolve().parent.parent / "he_grenade_game.uproject"),
            "-game",
        ])
    command.extend([
        "-GenerateDataset",
        "-CertifyV1PlanOnly",
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
    bindings = dataset_worker.v1_certification_bindings(
        root, plan, executable, execution_build
    )
    if bindings["plan_recipe_count"] != len(all_recipes):
        raise ValueError("V1 recipe inventory changed during certification")
    if bindings["guided_recipe_count"] != len(guided_recipes):
        raise ValueError("V1 guided recipe inventory changed during certification")
    if bindings["semi_markov_recipe_count"] != semi_markov_count:
        raise ValueError("V1 semi-Markov recipe inventory changed during certification")
    if (
        engine_report.get("plan_id") != plan["plan_id"]
        or engine_report.get("plan_version") != plan["plan_version"]
        or int(engine_report.get("recipe_count", -1)) != len(guided_recipes)
    ):
        raise ValueError("Unreal certification report does not bind to the requested plan")

    report = {**engine_report, "bindings": bindings}
    bound_path = output / "certification-report-bound.json"
    bound_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "complete": bool(report.get("complete")),
        "certified_count": report.get("certified_count"),
        "failed_count": report.get("failed_count"),
        "guided_recipe_count": len(guided_recipes),
        "semi_markov_recipe_count": semi_markov_count,
        "report": str(bound_path),
    }, indent=2, sort_keys=True))
    return 0 if report.get("complete") and completed.returncode == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
