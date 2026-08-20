#!/usr/bin/env python3
"""Resolve V1/V2 runtime-certification failures with immutable candidate rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dataset_controller as v1
import v2_dataset_controller as v2
from v2_mission_catalog import build_solution, mission_types


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(value) + "\n" for value in values), encoding="utf-8")


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{hashlib.sha256(canonical(value).encode()).hexdigest()[:16]}"


def progressively_safer(values: list[tuple[Any, float]], count: int) -> list[Any]:
    """Select distinct candidates from diverse to conservative, ending safest."""
    if count <= 0 or not values:
        return []
    ranked = sorted(values, key=lambda value: (value[1], str(value[0])))
    if count == 1:
        return [ranked[0][0]]
    # Start around the 65th safety percentile rather than at an extreme. Each
    # retry is monotonically safer and the final retry is the safest candidate.
    positions = [round((len(ranked) - 1) * 0.65 * (count - 1 - i) / (count - 1)) for i in range(count)]
    chosen: list[Any] = []
    seen: set[int] = set()
    for position in positions:
        while position in seen and position > 0:
            position -= 1
        if position in seen:
            position = next(index for index in range(len(ranked)) if index not in seen)
        seen.add(position)
        chosen.append(ranked[position][0])
    return chosen


def run_certification_batch(
    version: str,
    plan: dict[str, Any],
    recipes: list[dict[str, Any]],
    executable: Path,
    output: Path,
) -> dict[str, Any]:
    """Certify exactly the supplied recipes in one no-capture Unreal session."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "plan_version": plan["plan_version"],
        "assignment_id": f"{version}-certification-{output.name}",
        "assignment_number": 0,
        "dispatch_wave": 0,
        "logical_worker_id": 0,
        "split": plan.get("split", "mixed"),
        "generator": plan["generator"],
        "recipes": recipes,
    }
    manifest_path = output / "certification-manifest.json"
    write_json(manifest_path, manifest)
    command = [str(executable)]
    if "unrealeditor" in executable.name.lower():
        command.extend([str(Path(__file__).resolve().parent.parent / "he_grenade_game.uproject"), "-game"])
    command.extend([
        "-GenerateDataset",
        "-CertifyV1PlanOnly" if version == "v1" else "-CertifyV2PlanOnly",
        f"-RecipeManifest={manifest_path}",
        f"-Output={output}",
        "-RenderOffscreen", "-unattended", "-nosound", "-NoSplash", "-NoVSync",
    ])
    completed = subprocess.run(command, check=False)
    report_path = output / "certification-report.json"
    if not report_path.exists():
        raise RuntimeError(f"Unreal exited {completed.returncode} without {report_path}")
    report = read_json(report_path)
    if int(report.get("recipe_count", -1)) != len(recipes):
        raise RuntimeError("certification report recipe count does not match its manifest")
    report["process_exit_code"] = completed.returncode
    report["manifest"] = str(manifest_path)
    write_json(output / "certification-report-round-bound.json", report)
    return report


def _mix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _name_hash(name: str) -> int:
    value = 1469598103934665603
    for character in name:
        value ^= ord(character)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def _parameter_unit(seed_start: int, episode_index: int, name: str, sample: int = 0) -> float:
    key = seed_start & 0xFFFFFFFF
    key = _mix64(key ^ (episode_index & 0xFFFFFFFF))
    key ^= _name_hash(name)
    key ^= _mix64((sample & 0xFFFFFFFF) + 0xD1B54A32D192ED03)
    return ((_mix64(key) >> 40) & 0xFFFFFF) / float(1 << 24)


def _v1_conservative_score(seed_start: int, episode_index: int) -> float:
    # Initial view offsets dominate recoverability for the rare guided failures.
    return abs(_parameter_unit(seed_start, episode_index, "initial_yaw_offset") - 0.5) + abs(
        _parameter_unit(seed_start, episode_index, "initial_pitch_offset") - 0.5
    )


def v1_candidate_sequence(
    failed: dict[str, Any],
    plan: dict[str, Any],
    calibration: dict[str, Any],
    used_indices: set[int],
    next_repetition: int,
    count: int,
) -> list[dict[str, Any]]:
    seed_start = int(plan["generator"]["seed_start"])
    start = max(used_indices, default=-1) + 1
    pool = [index for index in range(start, start + 512) if index not in used_indices]
    ordered = progressively_safer(
        [(index, _v1_conservative_score(seed_start, index)) for index in pool], count
    )
    result = []
    for offset, recipe_index in enumerate(ordered):
        recipe = v1.build_recipe(
            calibration,
            str(plan["plan_id"]),
            recipe_index,
            str(failed["mission"]),
            int(failed["scenario_index"]),
            next_repetition + offset,
            reserve_for=str(failed["recipe_id"]),
            schedule_phase=str(failed["schedule_phase"]),
        )
        recipe["active"] = True
        recipe["planned_credited_frames"] = int(
            failed.get("planned_credited_frames", round(float(failed["expected_credited_frames"])))
        )
        result.append(recipe)
    return result


def v1_donor_mutation(
    failed: dict[str, Any],
    donor: dict[str, Any],
    plan: dict[str, Any],
    calibration: dict[str, Any],
    used_indices: set[int],
    next_repetition: int,
) -> dict[str, Any]:
    """Preserve the failed slot while closely matching a certified view seed."""
    seed_start = int(plan["generator"]["seed_start"])
    donor_index = int(donor["episode_index"])
    target_yaw = _parameter_unit(seed_start, donor_index, "initial_yaw_offset")
    target_pitch = _parameter_unit(seed_start, donor_index, "initial_pitch_offset")
    start = max(used_indices, default=-1) + 1
    pool = [index for index in range(start, start + 2048) if index not in used_indices]
    recipe_index = min(pool, key=lambda index: (
        abs(_parameter_unit(seed_start, index, "initial_yaw_offset") - target_yaw)
        + abs(_parameter_unit(seed_start, index, "initial_pitch_offset") - target_pitch),
        index,
    ))
    recipe = v1.build_recipe(
        calibration, str(plan["plan_id"]), recipe_index,
        str(failed["mission"]), int(failed["scenario_index"]), next_repetition,
        reserve_for=str(failed["recipe_id"]),
        schedule_phase=str(failed["schedule_phase"]),
    )
    recipe["active"] = True
    recipe["planned_credited_frames"] = int(
        failed.get("planned_credited_frames", round(float(failed["expected_credited_frames"])))
    )
    return recipe


def _v2_safety_score(solution: dict[str, Any]) -> float:
    variation = solution["variation"]
    values = [variation.get(name, 0.5) for name in ("surface_u", "surface_v", "distance_u", "arc_u")]
    return sum(abs(float(value) - 0.5) for value in values)


def v2_candidate_sequence(
    failed: dict[str, Any],
    plan: dict[str, Any],
    used_indices: set[int],
    next_repetition: int,
    count: int,
) -> list[dict[str, Any]]:
    item = {value.slug: value for value in mission_types()}[str(failed["mission_type"])]
    rate = int(plan["generator"]["observation_rate"])
    type_target = int(plan["planned_distribution"]["mission_type_frames"][item.slug])
    sample_count = math.ceil(type_target / v2.mission_expected_frames(item, rate))
    candidate_sample_count = max(sample_count, next_repetition + 128)
    candidates = []
    for repetition in range(next_repetition, next_repetition + 128):
        solution = build_solution(item, repetition, rate, candidate_sample_count)
        candidates.append((repetition, solution, _v2_safety_score(solution)))
    ordered_repetitions = progressively_safer(
        [(repetition, score) for repetition, _, score in candidates], count
    )
    by_repetition = {repetition: solution for repetition, solution, _ in candidates}
    start_index = max(used_indices, default=-1) + 1
    result = []
    for offset, repetition in enumerate(ordered_repetitions):
        solution = by_repetition[repetition]
        draft = {
            "source": "mission",
            "family": item.family,
            "mission": item.slug,
            "mission_type": item.slug,
            "scenario_index": int(failed["scenario_index"]),
            "repetition_index": repetition,
            "schedule_phase": str(failed["schedule_phase"]),
            "expected_credited_frames": int(failed["expected_credited_frames"]),
            "planned_credited_frames": int(failed["planned_credited_frames"]),
            "mission_solution": solution,
        }
        recipe = v2._materialize_recipe(
            draft,
            start_index + offset,
            int(plan["generator"]["seed_start"]),
            int(plan["generator"].get("evaluation_percent", 10)),
        )
        recipe["split"] = failed["split"]
        recipe["recipe_id"] = v2.recipe_digest(recipe)
        recipe["replay_identity"] = v2.replay_digest(recipe)
        result.append(recipe)
    return result


def v2_donor_mutation(
    failed: dict[str, Any],
    donor: dict[str, Any],
    plan: dict[str, Any],
    used_indices: set[int],
    next_repetition: int,
) -> dict[str, Any]:
    """Choose the unused canonical solution nearest a certified same-type donor."""
    item = {value.slug: value for value in mission_types()}[str(failed["mission_type"])]
    rate = int(plan["generator"]["observation_rate"])
    type_target = int(plan["planned_distribution"]["mission_type_frames"][item.slug])
    sample_count = math.ceil(type_target / v2.mission_expected_frames(item, rate))
    candidate_sample_count = max(sample_count, next_repetition + 512)
    donor_variation = donor["mission_solution"]["variation"]
    names = ("surface_u", "surface_v", "distance_u", "arc_u")
    values: list[tuple[float, int, dict[str, Any]]] = []
    for repetition in range(next_repetition, next_repetition + 512):
        solution = build_solution(item, repetition, rate, candidate_sample_count)
        variation = solution["variation"]
        distance = sum(
            abs(float(variation.get(name, 0.5)) - float(donor_variation.get(name, 0.5)))
            for name in names
        )
        values.append((distance, repetition, solution))
    _, repetition, solution = min(values, key=lambda value: (value[0], value[1]))
    draft = {
        "source": "mission", "family": item.family,
        "mission": item.slug, "mission_type": item.slug,
        "scenario_index": int(failed["scenario_index"]),
        "repetition_index": repetition,
        "schedule_phase": str(failed["schedule_phase"]),
        "expected_credited_frames": int(failed["expected_credited_frames"]),
        "planned_credited_frames": int(failed["planned_credited_frames"]),
        "mission_solution": solution,
    }
    recipe = v2._materialize_recipe(
        draft, max(used_indices, default=-1) + 1,
        int(plan["generator"]["seed_start"]),
        int(plan["generator"].get("evaluation_percent", 10)),
    )
    recipe["split"] = failed["split"]
    recipe["recipe_id"] = v2.recipe_digest(recipe)
    recipe["replay_identity"] = v2.replay_digest(recipe)
    return recipe


def result_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(value["recipe_id"]): value for value in report.get("results", [])}


def run_final_bound_certification(
    version: str,
    collection: Path,
    executable: Path,
) -> dict[str, Any]:
    """Use the existing recorder gate to produce its standard bound certificate."""
    script = Path(__file__).resolve().parent / (
        "certify_v1_plan.py" if version == "v1" else "certify_v2_plan.py"
    )
    output = collection / "certification"
    command = [
        sys.executable, str(script), str(collection),
        "--executable", str(executable), "--output", str(output),
    ]
    completed = subprocess.run(command, check=False)
    report_path = output / "certification-report-bound.json"
    if not report_path.exists():
        raise RuntimeError(
            f"final {version.upper()} certification exited {completed.returncode} without {report_path}"
        )
    report = read_json(report_path)
    report["process_exit_code"] = completed.returncode
    return report


def write_resolved_collection(
    version: str,
    source: Path,
    destination: Path,
    selected: list[dict[str, Any]],
    resolution_id: str,
) -> dict[str, Any]:
    plan = read_json(source / "plan" / "collection-plan.json")
    resolved_plan_id = stable_id("resolved", {
        "candidate_plan_id": plan["plan_id"],
        "resolution_id": resolution_id,
        "recipes": [recipe["recipe_id"] for recipe in selected],
    })
    plan.update({
        "plan_id": resolved_plan_id,
        "candidate_plan_id": plan["plan_id"],
        "resolution_id": resolution_id,
        "active_recipe_count": len(selected),
    })
    for recipe in selected:
        recipe["active"] = True
    chunk = int(plan.get("recipes_per_assignment", 32))
    assignments = []
    for number, start in enumerate(range(0, len(selected), chunk)):
        value = {
            "schema_version": 3 if version == "v2" else 1,
            "plan_id": resolved_plan_id,
            "plan_version": plan["plan_version"],
            "assignment_id": f"assignment-{number:06d}",
            "assignment_number": number,
            "dispatch_wave": number,
            "logical_worker_id": 0,
            "split": "mixed" if version == "v2" else plan.get("split", "train"),
            "generator": plan["generator"],
            "recipes": selected[start:start + chunk],
        }
        if version == "v2":
            value["contract_version"] = plan["contract_version"]
            value = v2.seal_assignment(value)
        assignments.append(value)
    plan["assignment_count"] = len(assignments)
    if version == "v2":
        plan["planned_distribution"] = v2.planned_distribution(selected)
    write_json(destination / "plan" / "collection-plan.json", plan)
    write_jsonl(destination / "plan" / "recipes.jsonl", selected)
    for assignment in assignments:
        write_json(destination / "assignments" / f"{assignment['assignment_id']}.json", assignment)
    if version == "v1":
        v1.verify_plan(argparse.Namespace(collection=destination))
    else:
        verification = v2.verify_plan(destination)
        write_json(destination / "plan" / "structural-verification.json", verification)
        if not verification["valid"]:
            raise RuntimeError(
                "resolved V2 plan failed structural verification: "
                + canonical(verification)
            )
    return plan


def resolve_version(
    version: str,
    collection: Path,
    executable: Path,
    output: Path,
    max_attempts: int,
) -> dict[str, Any]:
    plan = read_json(collection / "plan" / "collection-plan.json")
    all_recipes = read_jsonl(collection / "plan" / "recipes.jsonl")
    active = [value for value in all_recipes if value.get("active")]
    certifiable = [
        value for value in active
        if (value["mission"] != "semi_markov" if version == "v1" else value.get("source") == "mission")
    ]
    initial = run_certification_batch(version, plan, certifiable, executable, output / "round-00-original")
    initial_results = result_map(initial)
    missing_results = [
        recipe["recipe_id"] for recipe in certifiable
        if recipe["recipe_id"] not in initial_results
    ]
    if missing_results:
        raise RuntimeError(f"certification omitted {len(missing_results)} recipe results")
    accepted: dict[str, dict[str, Any]] = {
        recipe["recipe_id"]: recipe
        for recipe in certifiable
        if initial_results[recipe["recipe_id"]].get("certified")
    }
    unresolved = [recipe for recipe in certifiable if recipe["recipe_id"] not in accepted]
    chosen: dict[str, dict[str, Any]] = dict(accepted)
    lineage: list[dict[str, Any]] = []
    used_indices = {int(value["recipe_index"]) for value in all_recipes}
    repetitions: Counter[tuple[str, int]] = Counter()
    for recipe in all_recipes:
        key = (str(recipe["mission"]), int(recipe["scenario_index"]))
        repetitions[key] = max(repetitions[key], int(recipe["repetition_index"]) + 1)

    still_unresolved = []
    for failed in unresolved:
        key = (str(failed["mission"]), int(failed["scenario_index"]))
        candidates = (
            v1_candidate_sequence(failed, plan, v1.load_duration_calibration(), used_indices, repetitions[key], max_attempts)
            if version == "v1"
            else v2_candidate_sequence(failed, plan, used_indices, repetitions[key], max_attempts)
        )
        repetitions[key] = max(
            repetitions[key],
            max((int(candidate["repetition_index"]) + 1 for candidate in candidates), default=repetitions[key]),
        )
        replacement = None
        for attempt, candidate in enumerate(candidates, 1):
            used_indices.add(int(candidate["recipe_index"]))
            round_output = output / "slots" / str(failed["recipe_id"]) / f"attempt-{attempt:02d}"
            report = run_certification_batch(version, plan, [candidate], executable, round_output)
            detail = report["results"][0]
            entry = {
                "slot_recipe_id": failed["recipe_id"],
                "rejected_recipe": failed,
                "candidate_recipe": candidate,
                "attempt": attempt,
                "strategy": "progressively_centered_resample",
                "certification_result": detail,
                "accepted": bool(detail.get("certified")),
            }
            lineage.append(entry)
            if entry["accepted"]:
                replacement = candidate
                break
        if replacement is None:
            donors = [
                value for value in accepted.values()
                if value["mission"] == failed["mission"]
            ]
            if donors:
                donor = min(
                    donors,
                    key=lambda value: (
                        int(value["scenario_index"]) != int(failed["scenario_index"]),
                        int(value["recipe_index"]),
                    ),
                )
                mutation = (
                    v1_donor_mutation(
                        failed, donor, plan, v1.load_duration_calibration(),
                        used_indices, repetitions[key],
                    )
                    if version == "v1"
                    else v2_donor_mutation(
                        failed, donor, plan, used_indices, repetitions[key]
                    )
                )
                used_indices.add(int(mutation["recipe_index"]))
                report = run_certification_batch(
                    version, plan, [mutation], executable,
                    output / "slots" / str(failed["recipe_id"]) / "accepted-mutation",
                )
                detail = report["results"][0]
                lineage.append({
                    "slot_recipe_id": failed["recipe_id"],
                    "rejected_recipe": failed,
                    "candidate_recipe": mutation,
                    "donor_recipe_id": donor["recipe_id"],
                    "attempt": max_attempts + 1,
                    "strategy": "mutation_from_certified_same_cell",
                    "certification_result": detail,
                    "accepted": bool(detail.get("certified")),
                })
                if detail.get("certified"):
                    replacement = mutation
        if replacement is None:
            still_unresolved.append(failed)
        else:
            chosen[failed["recipe_id"]] = replacement
            accepted[replacement["recipe_id"]] = replacement

    if still_unresolved:
        selected = []
    else:
        selected = []
        for recipe in active:
            if recipe in certifiable:
                selected.append(chosen[recipe["recipe_id"]])
            else:
                selected.append(recipe)
    resolution_id = stable_id("resolution", {
        "version": version,
        "plan_id": plan["plan_id"],
        "selected": [value["recipe_id"] for value in selected],
        "unresolved": [value["recipe_id"] for value in still_unresolved],
    })
    resolved_plan = None
    final_report = None
    if selected:
        resolved_root = output / "resolved-collection"
        resolved_plan = write_resolved_collection(
            version, collection, resolved_root, selected, resolution_id
        )
        final_report = run_final_bound_certification(version, resolved_root, executable)

    summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "version": version,
        "candidate_plan_id": plan["plan_id"],
        "resolution_id": resolution_id,
        "resolved_plan_id": resolved_plan.get("plan_id") if resolved_plan else None,
        "certified_original_count": len(accepted),
        "rejected_original_count": len(unresolved),
        "original_rejections": [
            {"recipe": recipe, "certification_result": initial_results[recipe["recipe_id"]]}
            for recipe in unresolved
        ],
        "replacement_attempt_count": len(lineage),
        "accepted_replacement_count": sum(
            1 for failed in unresolved if failed["recipe_id"] in chosen
        ),
        "unresolved_count": len(still_unresolved),
        "unresolved_recipes": still_unresolved,
        "lineage": lineage,
        "planned_frame_distribution": (
            v1.distribution_summary(selected[: int(plan["base_recipe_count"])]) if selected else None
        ) if version == "v1" else (
            v2.planned_distribution(selected) if selected else None
        ),
        "final_certification": final_report,
        "final_certification_complete": bool(
            final_report
            and final_report.get("complete")
            and int(final_report.get("failed_count", -1)) == 0
            and int(final_report.get("process_exit_code", -1)) == 0
        ),
        "resolved_collection": str(output / "resolved-collection") if selected else None,
    }
    write_json(output / "resolution-report.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=("v1", "v2"))
    parser.add_argument("collection", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.max_attempts <= 10:
        raise ValueError("max-attempts must be between 1 and 10")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"resolution output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = resolve_version(
        args.version,
        args.collection.resolve(),
        args.executable.resolve(),
        output,
        args.max_attempts,
    )
    print(json.dumps({
        "resolution_id": report["resolution_id"],
        "rejected_original_count": report["rejected_original_count"],
        "accepted_replacement_count": report["accepted_replacement_count"],
        "unresolved_count": report["unresolved_count"],
        "resolved_collection": report["resolved_collection"],
    }, indent=2, sort_keys=True))
    return 0 if (
        report["unresolved_count"] == 0
        and report["final_certification_complete"]
    ) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
