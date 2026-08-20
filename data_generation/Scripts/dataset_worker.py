#!/usr/bin/env python3
"""Run immutable assignment manifests and publish immutable results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dataset_controller as movement_controller
import v2_dataset_controller as v2_controller


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_runtime_root(executable: Path) -> Path:
    """Locate a packaged-game root without mistaking an installed editor for one."""
    executable = executable.resolve()
    for candidate in (executable.parent, *executable.parents):
        if not (candidate / "Engine").is_dir():
            continue
        launchers = [
            path for path in candidate.iterdir()
            if path.is_file() and path.suffix.lower() in {".exe", ".sh"}
        ]
        packaged_projects = [
            path for path in candidate.iterdir()
            if path.is_dir() and (path / "Content" / "Paks").is_dir()
        ]
        if launchers and packaged_projects:
            return candidate
    return executable.parent


def package_runtime_fingerprint(executable: Path) -> tuple[str, int, int]:
    root = package_runtime_root(executable)
    excluded_suffixes = {".debug", ".pdb", ".sym"}
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in excluded_suffixes
        and "saved" not in {
            part.lower() for part in path.relative_to(root).parts[:-1]
        }
    )
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_size += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest(), len(files), total_size


def bind_execution_build(root: Path, executable: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Freeze the exact packaged binary used by every attempt in a collection."""
    executable = executable.resolve()
    if not executable.is_file():
        raise ValueError(f"packaged executable does not exist: {executable}")
    package_sha256, package_file_count, package_size = package_runtime_fingerprint(executable)
    identity = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "generator_source_sha256": plan.get("generator_source_sha256"),
        "executable_sha256": sha256_file(executable),
        "executable_size_bytes": executable.stat().st_size,
        "executable_name": executable.name,
        "package_runtime_sha256": package_sha256,
        "package_runtime_file_count": package_file_count,
        "package_runtime_size_bytes": package_size,
    }
    path = root / "execution-build.json"
    try:
        write_new_json(path, identity)
    except FileExistsError:
        frozen = read_json(path)
        if frozen != identity:
            raise ValueError(
                "packaged executable identity differs from the collection's "
                "immutable execution-build.json"
            )
    return identity


def require_v2_certification(
    root: Path,
    plan: dict[str, Any],
    executable: Path,
    execution_build: dict[str, Any],
) -> None:
    """Require a successful batch certificate bound to this plan and build."""
    report_path = root / "certification" / "certification-report-bound.json"
    if not report_path.is_file():
        raise ValueError(
            "V2 recording requires certification/certification-report-bound.json "
            "from certify_v2_plan.py"
        )
    report = read_json(report_path)
    bindings = report.get("bindings") or {}
    plan_path = root / "plan" / "collection-plan.json"
    recipes_path = root / "plan" / "recipes.jsonl"
    assignment_digests = [
        read_json(path)["assignment_digest"]
        for path in sorted((root / "assignments").glob("*.json"))
    ]
    assignment_digest_sha256 = hashlib.sha256(
        json.dumps(
            assignment_digests,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "plan_id": plan["plan_id"],
        "plan_version": plan["plan_version"],
        "collection_plan_sha256": sha256_file(plan_path),
        "recipes_jsonl_sha256": sha256_file(recipes_path),
        "assignment_digest_sha256": assignment_digest_sha256,
        "assignment_count": int(plan["assignment_count"]),
        "recipe_count": int(plan["active_recipe_count"]),
        "executable_path": str(executable.resolve()),
        "executable_sha256": execution_build["executable_sha256"],
        "executable_size_bytes": execution_build["executable_size_bytes"],
        "generator_source_sha256": execution_build.get("generator_source_sha256"),
        "package_runtime_sha256": execution_build["package_runtime_sha256"],
        "package_runtime_file_count": execution_build["package_runtime_file_count"],
    }
    if not bool(report.get("complete")) or int(report.get("failed_count", -1)) != 0:
        raise ValueError("V2 recording requires a complete zero-rejection certificate")
    if int(report.get("certified_count", -1)) != expected["recipe_count"]:
        raise ValueError("V2 certification recipe count is incomplete")
    mismatches = [key for key, value in expected.items() if bindings.get(key) != value]
    if mismatches:
        raise ValueError(
            "V2 certification does not bind to the current plan/build: "
            + ", ".join(mismatches)
        )


def v1_certification_recipe_counts(recipes_path: Path) -> tuple[int, int]:
    """Return (guided mission recipes, semi-Markov recipes) for a V1 plan."""
    guided = 0
    semi_markov = 0
    with recipes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            recipe = json.loads(line)
            if recipe.get("mission") == "semi_markov":
                semi_markov += 1
            else:
                guided += 1
    return guided, semi_markov


def v1_certification_bindings(
    root: Path,
    plan: dict[str, Any],
    executable: Path,
    execution_build: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable plan/build identity required by V1 recording."""
    plan_path = root / "plan" / "collection-plan.json"
    recipes_path = root / "plan" / "recipes.jsonl"
    guided_count, semi_markov_count = v1_certification_recipe_counts(recipes_path)
    return {
        "plan_id": plan["plan_id"],
        "plan_version": plan["plan_version"],
        "collection_plan_sha256": sha256_file(plan_path),
        "recipes_jsonl_sha256": sha256_file(recipes_path),
        "plan_recipe_count": guided_count + semi_markov_count,
        "guided_recipe_count": guided_count,
        "semi_markov_recipe_count": semi_markov_count,
        "executable_path": str(executable.resolve()),
        "executable_sha256": execution_build["executable_sha256"],
        "executable_size_bytes": execution_build["executable_size_bytes"],
        "generator_source_sha256": execution_build.get("generator_source_sha256"),
        "package_runtime_sha256": execution_build["package_runtime_sha256"],
        "package_runtime_file_count": execution_build["package_runtime_file_count"],
    }


def require_v1_certification(
    root: Path,
    plan: dict[str, Any],
    executable: Path,
    execution_build: dict[str, Any],
) -> None:
    """Require V1 guided missions to pass no-capture runtime certification."""
    report_path = root / "certification" / "certification-report-bound.json"
    if not report_path.is_file():
        raise ValueError(
            "V1 recording requires certification/certification-report-bound.json "
            "from certify_v1_plan.py"
        )
    report = read_json(report_path)
    expected = v1_certification_bindings(root, plan, executable, execution_build)
    if not bool(report.get("complete")) or int(report.get("failed_count", -1)) != 0:
        raise ValueError("V1 recording requires a complete zero-rejection certificate")
    if int(report.get("certified_count", -1)) != expected["guided_recipe_count"]:
        raise ValueError("V1 certification guided-recipe count is incomplete")
    bindings = report.get("bindings") or {}
    mismatches = [key for key, value in expected.items() if bindings.get(key) != value]
    if mismatches:
        raise ValueError(
            "V1 certification does not bind to the current plan/build: "
            + ", ".join(mismatches)
        )


def results_for(root: Path, assignment_id: str) -> list[dict[str, Any]]:
    paths = sorted((root / "results").glob(f"{assignment_id}--attempt-*.json"))
    return [read_json(path) for path in paths]


def next_attempt(root: Path, assignment_id: str) -> int:
    used: list[int] = []
    for directory in (root / "claims", root / "results"):
        if not directory.exists():
            continue
        for path in directory.glob(f"{assignment_id}--attempt-*.json"):
            try:
                used.append(int(path.stem.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(used, default=-1) + 1


def claim_attempt(root: Path, assignment_id: str, attempt_number: int, executor_id: str) -> Path | None:
    claim = root / "claims" / f"{assignment_id}--attempt-{attempt_number:03d}.json"
    try:
        write_new_json(claim, {
            "assignment_id": assignment_id,
            "attempt_number": attempt_number,
            "executor_id": executor_id,
            "claimed_utc": utc_now(),
        })
    except FileExistsError:
        return None
    return claim


def read_episode_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    dataset = read_json(dataset_dir / "dataset.json")
    shard = dataset_dir / dataset["shards"][0]
    with tarfile.open(shard, "r") as archive:
        member = archive.getmember("episodes.parquet")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("episodes.parquet could not be read from finalized shard")
        return pq.read_table(extracted).to_pylist()


def build_validated_result(
    assignment: dict[str, Any],
    attempt_id: str,
    executor_id: str,
    output: Path,
) -> dict[str, Any]:
    episodes = read_episode_rows(output)
    rows_by_recipe = {row.get("recipe_id"): row for row in episodes}
    resolved: list[str] = []
    successful: list[str] = []
    semantic_failures: list[str] = []
    semantic_failure_details: list[dict[str, Any]] = []
    visibility_degraded_recipe_ids: list[str] = []
    credited_cells: list[dict[str, Any]] = []
    accepted_frames = 0
    produced_frames = 0
    accepted_frames_by_mission: dict[str, int] = {}
    assignment_is_v2 = str(assignment.get("plan_version", "")).startswith(
        "trajectory-throw-v2"
    )
    for recipe in assignment["recipes"]:
        recipe_id = recipe["recipe_id"]
        row = rows_by_recipe.get(recipe_id)
        if row is None:
            raise ValueError(f"validated output omitted recipe {recipe_id}")
        resolved.append(recipe_id)
        observation_count = int(row["observation_count"])
        produced_frames += observation_count
        mission = recipe["mission"]
        is_v2 = assignment_is_v2
        source = str(recipe.get("source") or "")
        if is_v2:
            if source not in {"semi_markov", "mission"}:
                raise ValueError(f"V2 recipe {recipe_id} has invalid source {source!r}")
            if row.get("v2_source") != source:
                raise ValueError(f"V2 recipe {recipe_id} source disagrees with runtime output")
            if source == "semi_markov" and mission != "semi_markov":
                raise ValueError(f"V2 semi-Markov recipe has mission {mission!r}")
            if source == "mission" and mission != recipe.get("mission_type"):
                raise ValueError(f"V2 mission recipe {recipe_id} changed mission identity")
            if row.get("v2_replay_identity") != recipe.get("replay_identity"):
                raise ValueError(f"V2 recipe {recipe_id} changed replay identity")
            if source == "mission" and row.get("v2_mission_type") != recipe.get("mission_type"):
                raise ValueError(f"V2 recipe {recipe_id} changed runtime mission type")
            if bool(row.get("v2_visibility_degraded")):
                visibility_degraded_recipe_ids.append(recipe_id)
        creditable = source == "semi_markov" if is_v2 else mission == "semi_markov"
        creditable = creditable or bool(row["mission_success"])
        if creditable:
            successful.append(recipe_id)
            credited_frame_cap = int(recipe.get("planned_credited_frames", observation_count))
            credited_frames = min(observation_count, credited_frame_cap)
            accepted_frames += credited_frames
            accepted_frames_by_mission[mission] = accepted_frames_by_mission.get(mission, 0) + credited_frames
            credited_cells.append({
                "mission": mission,
                "mission_type": recipe.get("mission_type"),
                "source": source or ("semi_markov" if mission == "semi_markov" else "mission"),
                "family": recipe.get("family"),
                "scenario_index": int(recipe["scenario_index"]),
                "cell_id": recipe.get("cell_id"),
                "variation_cell_id": recipe.get("variation_cell_id"),
                "recipe_id": recipe_id,
                "credited_observation_frames": credited_frames,
                "produced_observation_frames": observation_count,
            })
        else:
            semantic_failures.append(recipe_id)
            semantic_failure_details.append({
                "recipe_id": recipe_id,
                "mission_type": recipe.get("mission_type"),
                "termination_reason": row.get("termination_reason"),
                "construction_certified": row.get("v2_construction_certified"),
                "mission_event_frame": row.get("v2_mission_event_frame"),
            })
    result = {
        "schema_version": 1,
        "plan_id": assignment["plan_id"],
        "assignment_id": assignment["assignment_id"],
        "attempt_id": attempt_id,
        "executor_id": executor_id,
        "completed_utc": utc_now(),
        "technical_result": "validated",
        "semantic_result": "success" if not semantic_failures else "resolved_with_failures",
        "output_directory": str(output),
        "accepted_observation_frames": accepted_frames,
        "produced_observation_frames": produced_frames,
        "accepted_frames_by_mission": accepted_frames_by_mission,
        "resolved_recipe_ids": resolved,
        "successful_recipe_ids": successful,
        "semantic_failure_recipe_ids": semantic_failures,
        "semantic_failure_details": semantic_failure_details,
        "credited_cells": credited_cells,
    }
    if assignment_is_v2:
        result.update({
            "output_dataset_sha256": sha256_file(output / "dataset.json"),
            "assignment_digest": assignment.get("assignment_digest"),
            "visibility_degraded_recipe_count": len(visibility_degraded_recipe_ids),
            "visibility_degraded_recipe_ids": visibility_degraded_recipe_ids,
            "recipe_bindings": [
                {
                    "recipe_id": recipe["recipe_id"],
                    "replay_identity": recipe.get("replay_identity"),
                    "mission_type": recipe.get("mission_type"),
                    "mission_solution": recipe.get("mission_solution"),
                }
                for recipe in assignment["recipes"]
            ],
        })
    return result


def command_for(executable: Path, runtime_manifest: Path, output: Path) -> list[str]:
    command = [str(executable)]
    if "unrealeditor" in executable.name.lower():
        project = Path(__file__).resolve().parent.parent / "he_grenade_game.uproject"
        command.extend([str(project), "-game"])
    command.extend([
        "-GenerateDataset",
        f"-RecipeManifest={runtime_manifest}",
        f"-Output={output}",
        "-RenderOffscreen",
        "-unattended",
        "-nosound",
        "-NoSplash",
        "-NoVSync",
    ])
    return command


def run_assignment(args: argparse.Namespace, assignment_path: Path) -> str:
    root = args.collection.resolve()
    assignment = read_json(assignment_path)
    if str(assignment.get("plan_version", "")).startswith("trajectory-throw-v2"):
        v2_controller.validate_assignment_against_plan(root, assignment)
    assignment_id = assignment["assignment_id"]
    previous = results_for(root, assignment_id)
    if any(result.get("technical_result") == "validated" for result in previous):
        return "already_validated"
    attempt_number = next_attempt(root, assignment_id)
    if attempt_number >= args.max_attempts:
        return "attempt_limit"
    if claim_attempt(root, assignment_id, attempt_number, args.executor_id) is None:
        return "claimed_elsewhere"
    attempt_id = f"attempt-{attempt_number:03d}"
    attempt_dir = root / "attempts" / assignment_id / attempt_id
    output = attempt_dir / "output"
    runtime_manifest = {
        **assignment,
        "attempt_id": attempt_id,
        "executor_id": args.executor_id,
        "execution_build": args.execution_build,
    }
    write_new_json(attempt_dir / "request.json", runtime_manifest)
    result_path = root / "results" / f"{assignment_id}--{attempt_id}.json"
    if args.simulate_interruption:
        write_new_json(result_path, {
            "schema_version": 1,
            "plan_id": assignment["plan_id"],
            "assignment_id": assignment_id,
            "attempt_id": attempt_id,
            "executor_id": args.executor_id,
            "completed_utc": utc_now(),
            "technical_result": "interrupted",
            "semantic_result": "not_evaluated",
            "resolved_recipe_ids": [],
            "accepted_observation_frames": 0,
            "error": "controlled interruption requested by worker test flag",
        })
        return "interrupted"

    output.mkdir(parents=True, exist_ok=False)
    command = command_for(args.executable.resolve(), attempt_dir / "request.json", output)
    started = utc_now()
    with (attempt_dir / "stdout.log").open("xb") as stdout_handle, (attempt_dir / "stderr.log").open("xb") as stderr_handle:
        completed = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, check=False)
    if completed.returncode != 0:
        write_new_json(result_path, {
            "schema_version": 1,
            "plan_id": assignment["plan_id"],
            "assignment_id": assignment_id,
            "attempt_id": attempt_id,
            "executor_id": args.executor_id,
            "started_utc": started,
            "completed_utc": utc_now(),
            "technical_result": "generator_failed",
            "semantic_result": "not_evaluated",
            "return_code": completed.returncode,
            "resolved_recipe_ids": [],
            "accepted_observation_frames": 0,
        })
        return "generator_failed"

    project = Path(__file__).resolve().parent.parent
    finalize = subprocess.run([sys.executable, str(project / "Scripts" / "finalize_production_dataset.py"), str(output)], check=False)
    validate = subprocess.run([sys.executable, str(project / "Scripts" / "review_dataset.py"), str(output), "--validate-only"], check=False) if finalize.returncode == 0 else None
    if finalize.returncode != 0 or validate is None or validate.returncode != 0:
        write_new_json(result_path, {
            "schema_version": 1,
            "plan_id": assignment["plan_id"],
            "assignment_id": assignment_id,
            "attempt_id": attempt_id,
            "executor_id": args.executor_id,
            "started_utc": started,
            "completed_utc": utc_now(),
            "technical_result": "validation_failed",
            "semantic_result": "not_evaluated",
            "finalizer_return_code": finalize.returncode,
            "validator_return_code": None if validate is None else validate.returncode,
            "resolved_recipe_ids": [],
            "accepted_observation_frames": 0,
        })
        return "validation_failed"

    try:
        result = build_validated_result(assignment, attempt_id, args.executor_id, output)
    except ValueError as error:
        write_new_json(result_path, {
            "schema_version": 1,
            "plan_id": assignment["plan_id"],
            "assignment_id": assignment_id,
            "attempt_id": attempt_id,
            "executor_id": args.executor_id,
            "started_utc": started,
            "completed_utc": utc_now(),
            "technical_result": "validation_failed",
            "semantic_result": "not_evaluated",
            "resolved_recipe_ids": [],
            "accepted_observation_frames": 0,
            "error": str(error),
        })
        return "validation_failed"
    result["execution_build"] = args.execution_build
    write_new_json(result_path, result)
    return "validated"


def prior_wave_complete(root: Path, assignment: dict[str, Any]) -> bool:
    """Do not let a fast worker run arbitrarily far beyond slower workers."""
    is_v2 = str(assignment.get("plan_version", "")).startswith("trajectory-throw-v2")
    if is_v2 and v2_controller._forbidden_keys(assignment):
        raise ValueError("V2 assignments cannot use reserve or replacement fields")
    if not is_v2 and "reserve_activation_for" in assignment:
        return True
    wave = int(assignment.get("dispatch_wave", 0))
    if wave == 0:
        return True
    validated = {
        result.get("assignment_id")
        for path in (root / "results").glob("*.json")
        if (result := read_json(path)).get("technical_result") == "validated"
    }
    for path in (root / "assignments").glob("*.json"):
        earlier = read_json(path)
        earlier_is_v2 = str(earlier.get("plan_version", "")).startswith("trajectory-throw-v2")
        if not earlier_is_v2 and "reserve_activation_for" in earlier:
            continue
        if int(earlier.get("dispatch_wave", 0)) < wave and earlier["assignment_id"] not in validated:
            return False
    return True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("collection", type=Path)
    result.add_argument("--executable", type=Path, required=True)
    result.add_argument("--worker-id", type=int, required=True)
    result.add_argument("--executor-id", default=f"{platform.system().lower()}-{os.getpid()}")
    result.add_argument("--max-attempts", type=int, default=3)
    result.add_argument("--simulate-interruption", action="store_true")
    result.add_argument("--one", action="store_true", help="process at most one eligible assignment")
    return result


def main() -> int:
    args = parser().parse_args()
    root = args.collection.resolve()
    plan = read_json(root / "plan" / "collection-plan.json")
    controller = (
        v2_controller
        if str(plan.get("plan_version", "")).startswith("trajectory-throw-v2")
        else movement_controller
    )
    statuses: list[dict[str, str]] = []
    if (root / "STOP").exists():
        print(json.dumps({
            "worker_id": args.worker_id,
            "executor_id": args.executor_id,
            "assignments": statuses,
        }, indent=2, sort_keys=True))
        return 0
    args.execution_build = bind_execution_build(root, args.executable, plan)
    plan_version = str(plan.get("plan_version", ""))
    if plan_version.startswith("trajectory-throw-v2"):
        require_v2_certification(root, plan, args.executable, args.execution_build)
    elif plan_version.startswith("movement-v1"):
        require_v1_certification(root, plan, args.executable, args.execution_build)
    for assignment_path in sorted((root / "assignments").glob("*.json")):
        if (root / "STOP").exists():
            break
        assignment = read_json(assignment_path)
        if int(assignment["logical_worker_id"]) != args.worker_id:
            continue
        inventory = controller.build_inventory(root)
        if inventory["complete"]:
            statuses.append({"assignment_id": assignment["assignment_id"], "status": "budget_complete"})
            break
        if not prior_wave_complete(root, assignment):
            statuses.append({"assignment_id": assignment["assignment_id"], "status": "waiting_for_prior_wave"})
            break
        status = run_assignment(args, assignment_path)
        statuses.append({"assignment_id": assignment["assignment_id"], "status": status})
        if status not in {"validated", "already_validated", "budget_complete"}:
            break
        if args.one or args.simulate_interruption or (root / "STOP").exists():
            break
    print(json.dumps({"worker_id": args.worker_id, "executor_id": args.executor_id, "assignments": statuses}, indent=2, sort_keys=True))
    return 0 if all(item["status"] in {"validated", "already_validated", "budget_complete", "waiting_for_prior_wave"} for item in statuses) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
