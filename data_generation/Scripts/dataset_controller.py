#!/usr/bin/env python3
"""Immutable central planner and inventory for Movement V1 dataset generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PLAN_VERSION = "movement-v1-persistent-semi-markov-5"
CATALOG_VERSION = "fixed-arena-r5-static-no-input"
DEFAULT_DURATION_CALIBRATION = Path(__file__).with_name("movement_v1_duration_calibration.json")
MISSION_COUNTS = {
    "semi_markov": 32,  # 8 initial behavior families x 4 hold-duration bands.
    "object_view": 120,
    "contact_recovery": 675,
    "ramp_traverse": 30,
    "hoop_pass": 30,
    "static_no_input": 15,
}
GUIDED_MISSIONS = set(MISSION_COUNTS) - {"semi_markov"}
MISSION_FRAME_SHARES = {
    "semi_markov": Fraction(70, 100),
    "object_view": Fraction(132, 1000),
    "contact_recovery": Fraction(99, 1000),
    "ramp_traverse": Fraction(33, 1000),
    "hoop_pass": Fraction(33, 1000),
    "static_no_input": Fraction(3, 1000),
}
FACING_SHARES = {
    "forward": Fraction(35, 100),
    "backward": Fraction(15, 100),
    "strafe_left": Fraction(15, 100),
    "strafe_right": Fraction(15, 100),
    "free_attention": Fraction(20, 100),
}
OBJECT_MODE_SHARES = {
    "approach_observe": Fraction(40, 100),
    "pass_by": Fraction(35, 100),
    "partial_orbit": Fraction(20, 100),
    "full_orbit": Fraction(5, 100),
}
OBJECT_GAZE_SHARES = {
    "target_center": Fraction(40, 100),
    "target_offset": Fraction(25, 100),
    "travel_direction": Fraction(20, 100),
    "roam_reacquire": Fraction(15, 100),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def rounded_frame_targets(frame_budget: int) -> dict[str, int]:
    """Largest-remainder whole-frame targets preserving the frozen shares."""
    exact = {
        mission: Fraction(frame_budget) * share
        for mission, share in MISSION_FRAME_SHARES.items()
    }
    targets = {
        mission: value.numerator // value.denominator
        for mission, value in exact.items()
    }
    remaining = frame_budget - sum(targets.values())
    order = sorted(
        MISSION_COUNTS,
        key=lambda mission: (
            exact[mission] - targets[mission],
            -list(MISSION_COUNTS).index(mission),
        ),
        reverse=True,
    )
    for mission in order[:remaining]:
        targets[mission] += 1
    return targets


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_new_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value))
            handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def export_duration_calibration(args: argparse.Namespace) -> None:
    """Export per-scenario credited-frame means from a validated collection."""
    root = args.collection.resolve()
    recipes = {recipe["recipe_id"]: recipe for recipe in read_jsonl(root / "plan" / "recipes.jsonl")}
    totals: dict[str, Counter[int]] = {mission: Counter() for mission in MISSION_COUNTS}
    samples: dict[str, Counter[int]] = {mission: Counter() for mission in MISSION_COUNTS}
    validated_assignments: set[str] = set()
    for path in result_files(root):
        result = read_json(path)
        assignment_id = str(result.get("assignment_id"))
        if result.get("technical_result") != "validated" or assignment_id in validated_assignments:
            continue
        validated_assignments.add(assignment_id)
        for cell in result.get("credited_cells", []):
            recipe = recipes.get(str(cell.get("recipe_id")))
            if recipe is None:
                continue
            mission = recipe["mission"]
            scenario = int(recipe["scenario_index"])
            frames = int(cell.get("credited_observation_frames", 0))
            if frames > 0:
                totals[mission][scenario] += frames
                samples[mission][scenario] += 1
    missing = [
        (mission, scenario)
        for mission, count in MISSION_COUNTS.items()
        for scenario in range(count)
        if samples[mission][scenario] == 0
    ]
    if missing:
        raise ValueError(f"cannot calibrate: {len(missing)} prescribed scenarios have no credited sample")
    calibration = {
        "schema_version": 1,
        "calibration_version": args.calibration_version,
        "source_plan_id": read_json(root / "plan" / "collection-plan.json")["plan_id"],
        "source_credited_frames": build_inventory(root)["accepted_observation_frames"],
        "expected_frames_by_scenario": {
            mission: [
                round(totals[mission][scenario] / samples[mission][scenario], 8)
                for scenario in range(count)
            ]
            for mission, count in MISSION_COUNTS.items()
        },
        "sample_counts_by_scenario": {
            mission: [samples[mission][scenario] for scenario in range(count)]
            for mission, count in MISSION_COUNTS.items()
        },
    }
    write_new_json(args.output.resolve(), calibration)
    print(canonical_json({
        "output": str(args.output.resolve()),
        "calibration_version": args.calibration_version,
        "credited_frames": calibration["source_credited_frames"],
        "scenario_count": sum(MISSION_COUNTS.values()),
    }))


@lru_cache(maxsize=None)
def spread_indices(count: int) -> tuple[int, ...]:
    """Return a deterministic farthest-gap ordering over [0, count)."""
    if count <= 0:
        return ()
    selected: list[int] = []
    remaining = set(range(count))
    distances = [math.inf] * count
    while remaining:
        candidate = max(remaining, key=lambda value: (distances[value], -value))
        selected.append(candidate)
        remaining.remove(candidate)
        for value in remaining:
            distances[value] = min(distances[value], abs(value - candidate))
    return tuple(selected)


def mandatory_cells() -> list[tuple[str, int]]:
    """Interleave mission families while spreading within every finite catalog."""
    orders = {mission: spread_indices(count) for mission, count in MISSION_COUNTS.items()}
    cells: list[tuple[str, int]] = []
    for ordinal in range(max(MISSION_COUNTS.values())):
        for mission in MISSION_COUNTS:
            if ordinal < len(orders[mission]):
                cells.append((mission, orders[mission][ordinal]))
    return cells


def load_duration_calibration(path: Path = DEFAULT_DURATION_CALIBRATION) -> dict[str, Any]:
    calibration = read_json(path.resolve())
    # Static control recipes have a contractual duration rather than an
    # empirical early-success duration: 200 observations at the required 20 Hz.
    calibration.setdefault("expected_frames_by_scenario", {})["static_no_input"] = [
        200.0
    ] * MISSION_COUNTS["static_no_input"]
    values = calibration.get("expected_frames_by_scenario", {})
    for mission, count in MISSION_COUNTS.items():
        if len(values.get(mission, [])) != count:
            raise ValueError(f"duration calibration has the wrong scenario count for {mission}")
        if any(float(value) <= 0 for value in values[mission]):
            raise ValueError(f"duration calibration contains a non-positive duration for {mission}")
    return calibration


def expected_recipe_frames(calibration: dict[str, Any], mission: str, scenario: int) -> Fraction:
    return Fraction(str(calibration["expected_frames_by_scenario"][mission][scenario]))


def expected_mandatory_frames(calibration: dict[str, Any] | None = None) -> dict[str, Fraction]:
    calibration = calibration or load_duration_calibration()
    return {
        mission: sum(
            (expected_recipe_frames(calibration, mission, scenario) for scenario in range(count)),
            Fraction(),
        )
        for mission, count in MISSION_COUNTS.items()
    }


def nested_frame_targets(mission: str) -> dict[str, dict[str, Fraction]]:
    equal = lambda names: {name: Fraction(1, len(names)) for name in names}
    if mission == "semi_markov":
        return {
            "initial_behavior_family": equal(["idle", "forward", "strafe", "camera_yaw", "camera_pitch", "movement_camera", "opposing_inputs", "deliberate_contact"]),
            "initial_hold_band": equal(["short", "medium", "long", "very_long"]),
        }
    if mission == "static_no_input":
        return {"location": {str(index): Fraction(1, MISSION_COUNTS[mission]) for index in range(MISSION_COUNTS[mission])}}
    if mission == "object_view":
        return {
            "target": equal(["rectangle", "pyramid", "sphere", "hoop", "ramp"]),
            "mode": OBJECT_MODE_SHARES,
            "gaze": OBJECT_GAZE_SHARES,
            # Directions are conditional on the 25% combined orbit share.
            "direction": {"clockwise": Fraction(1, 8), "counter_clockwise": Fraction(1, 8)},
        }
    if mission == "contact_recovery":
        return {
            "target": equal(["rectangle", "pyramid", "sphere", "hoop", "ramp", "north_wall", "south_wall", "east_wall", "west_wall"]),
            "recovery": equal(["backward", "strafe_left", "strafe_right", "diagonal_left", "diagonal_right"]),
            "approach": equal(["direct", "glance_left", "glance_right"]),
            "facing": FACING_SHARES,
        }
    path_names = (
        ["center", "diagonal_left_to_right", "diagonal_right_to_left"]
        if mission == "ramp_traverse"
        else ["center", "oblique_left_to_right", "oblique_right_to_left"]
    )
    directions = (
        ["uphill", "downhill"]
        if mission == "ramp_traverse"
        else ["positive_x_to_negative_x", "negative_x_to_positive_x"]
    )
    return {
        "direction": equal(directions),
        "path": {path_names[0]: Fraction(1, 2), path_names[1]: Fraction(1, 4), path_names[2]: Fraction(1, 4)},
        "facing": FACING_SHARES,
    }


def feasibility_requirements(calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    calibration = calibration or load_duration_calibration()
    requirements: dict[str, Any] = {}
    for mission, count in MISSION_COUNTS.items():
        mandatory_total = sum(
            (expected_recipe_frames(calibration, mission, scenario) for scenario in range(count)),
            Fraction(),
        )
        required_mission_frames = mandatory_total
        limiting = "complete_mission_catalog"
        for dimension, targets in nested_frame_targets(mission).items():
            bucket_frames: Counter[str] = Counter()
            for scenario in range(count):
                value = cell_details(mission, scenario).get(dimension)
                if value in targets:
                    bucket_frames[str(value)] += expected_recipe_frames(calibration, mission, scenario)
            for value, target in targets.items():
                required = Fraction(bucket_frames[value]) / target
                if required > required_mission_frames:
                    required_mission_frames = required
                    limiting = f"{dimension}={value}"
        requirements[mission] = {
            "minimum_mission_frames": float(required_mission_frames),
            "minimum_total_budget": math.ceil(required_mission_frames / MISSION_FRAME_SHARES[mission]),
            "limiting_requirement": limiting,
        }
    return requirements


def minimum_feasible_frame_budget(calibration: dict[str, Any] | None = None) -> int:
    """Smallest budget that can contain full coverage and all frozen aggregate shares."""
    return max(item["minimum_total_budget"] for item in feasibility_requirements(calibration).values())


def cell_details(mission: str, scenario: int) -> dict[str, Any]:
    if mission == "semi_markov":
        action_names = [
            "idle", "forward", "strafe", "camera_yaw", "camera_pitch",
            "movement_camera", "opposing_inputs", "deliberate_contact",
        ]
        hold_names = ["short", "medium", "long", "very_long"]
        return {
            "initial_behavior_family": action_names[scenario // 4],
            "initial_hold_band": hold_names[scenario % 4],
        }
    if mission == "static_no_input":
        return {
            "location": str(scenario),
            "duration_seconds": 10,
            "action_mask": 0,
        }
    if mission == "object_view":
        gaze = ["target_center", "target_offset", "travel_direction", "roam_reacquire"]
        targets = ["rectangle", "pyramid", "sphere", "hoop", "ramp"]
        if scenario < 20:
            return {"target": targets[scenario // 4], "mode": "approach_observe", "gaze": gaze[scenario % 4], "direction": None}
        if scenario < 40:
            local = scenario - 20
            return {"target": targets[local // 4], "mode": "pass_by", "gaze": gaze[local % 4], "direction": None}
        full = scenario >= 80
        local = scenario - (80 if full else 40)
        return {
            "target": targets[local // 8],
            "mode": "full_orbit" if full else "partial_orbit",
            "gaze": gaze[(local % 8) // 2],
            "direction": "clockwise" if local % 2 == 0 else "counter_clockwise",
        }
    if mission == "contact_recovery":
        targets = ["rectangle", "pyramid", "sphere", "hoop", "ramp", "north_wall", "south_wall", "east_wall", "west_wall"]
        recoveries = ["backward", "strafe_left", "strafe_right", "diagonal_left", "diagonal_right"]
        approaches = ["direct", "glance_left", "glance_right"]
        facings = ["forward", "backward", "strafe_left", "strafe_right", "free_attention"]
        base, facing = divmod(scenario, 5)
        target, local = divmod(base, 15)
        recovery, approach = divmod(local, 3)
        return {"target": targets[target], "recovery": recoveries[recovery], "approach": approaches[approach], "facing": facings[facing]}
    direction, local = divmod(scenario, 15)
    path, facing = divmod(local, 5)
    facings = ["forward", "backward", "strafe_left", "strafe_right", "free_attention"]
    if mission == "ramp_traverse":
        return {"direction": ["uphill", "downhill"][direction], "path": ["center", "diagonal_left_to_right", "diagonal_right_to_left"][path], "facing": facings[facing]}
    return {"direction": ["positive_x_to_negative_x", "negative_x_to_positive_x"][direction], "path": ["center", "oblique_left_to_right", "oblique_right_to_left"][path], "facing": facings[facing]}


def scenario_weight(mission: str, scenario: int) -> Fraction:
    """Frozen within-mission probability for a prescribed discrete cell."""
    cell = cell_details(mission, scenario)
    if mission == "semi_markov":
        return Fraction(1, MISSION_COUNTS[mission])
    if mission == "static_no_input":
        return Fraction(1, MISSION_COUNTS[mission])
    if mission == "object_view":
        weight = Fraction(1, 5) * OBJECT_MODE_SHARES[cell["mode"]] * OBJECT_GAZE_SHARES[cell["gaze"]]
        if cell["direction"] is not None:
            weight *= Fraction(1, 2)
        return weight
    if mission == "contact_recovery":
        return Fraction(1, 9) * Fraction(1, 5) * Fraction(1, 3) * FACING_SHARES[cell["facing"]]
    path_share = Fraction(1, 2) if cell["path"] == "center" else Fraction(1, 4)
    return Fraction(1, 2) * path_share * FACING_SHARES[cell["facing"]]


def scenario_recipe_weights(calibration: dict[str, Any], mission: str) -> tuple[Fraction, ...]:
    """Recipe probability needed to realize the desired probability in frame units."""
    raw = [
        scenario_weight(mission, index) / expected_recipe_frames(calibration, mission, index)
        for index in range(MISSION_COUNTS[mission])
    ]
    total = sum(raw, Fraction())
    return tuple(value / total for value in raw)


def next_weighted_scenario(weights: tuple[Fraction, ...], mission: str, counts: Counter[int]) -> int:
    """Weighted fair selection with alternating spread-order tie breaking."""
    count = MISSION_COUNTS[mission]
    selected = sum(counts.values())
    order = list(spread_indices(count))
    if (selected // count) % 2 == 1:
        order.reverse()
    rank = {scenario: position for position, scenario in enumerate(order)}
    next_total = selected + 1
    return max(
        range(count),
        key=lambda scenario: (
            weights[scenario] * next_total - counts[scenario],
            -rank[scenario],
        ),
    )


def build_candidate_schedule(calibration: dict[str, Any], frame_budget: int, tail_fraction: Fraction = Fraction(1, 10)) -> tuple[list[tuple[str, int, int, str]], int, Fraction]:
    """Return mandatory/base/tail candidates in one immutable weighted order."""
    schedule: list[tuple[str, int, int, str]] = []
    scenario_counts: dict[str, Counter[int]] = {mission: Counter() for mission in MISSION_COUNTS}
    recipe_weights = {mission: scenario_recipe_weights(calibration, mission) for mission in MISSION_COUNTS}
    mission_frames: dict[str, Fraction] = {mission: Fraction() for mission in MISSION_COUNTS}
    mission_capacity: Counter[str] = Counter()
    frame_targets = rounded_frame_targets(frame_budget)

    for mission, scenario in mandatory_cells():
        repetition = scenario_counts[mission][scenario]
        schedule.append((mission, scenario, repetition, "mandatory"))
        scenario_counts[mission][scenario] += 1
        expected = expected_recipe_frames(calibration, mission, scenario)
        mission_frames[mission] += expected
        mission_capacity[mission] += max(1, int(round(float(expected))))

    total_frames = sum(mission_frames.values(), Fraction())

    def append_weighted(phase: str) -> None:
        nonlocal total_frames
        # The smallest frames/share ratio is the most underserved mission.
        mission = min(
            MISSION_COUNTS,
            key=lambda name: (
                mission_frames[name] / MISSION_FRAME_SHARES[name],
                list(MISSION_COUNTS).index(name),
            ),
        )
        scenario = next_weighted_scenario(recipe_weights[mission], mission, scenario_counts[mission])
        repetition = scenario_counts[mission][scenario]
        schedule.append((mission, scenario, repetition, phase))
        scenario_counts[mission][scenario] += 1
        added = expected_recipe_frames(calibration, mission, scenario)
        mission_frames[mission] += added
        mission_capacity[mission] += max(1, int(round(float(added))))
        total_frames += added

    while any(mission_capacity[name] < frame_targets[name] for name in MISSION_COUNTS):
        append_weighted("base")
    base_count = len(schedule)
    base_expected = total_frames
    tail_target = max(Fraction(10000), Fraction(frame_budget) * tail_fraction)
    while total_frames < base_expected + tail_target:
        append_weighted("tail")
    return schedule, base_count, base_expected


def build_recipe(calibration: dict[str, Any], plan_id: str, recipe_index: int, mission: str, scenario: int, repetition: int, reserve_for: str | None = None, schedule_phase: str = "reserve") -> dict[str, Any]:
    identity = {
        "plan_id": plan_id,
        "recipe_index": recipe_index,
        "mission": mission,
        "scenario_index": scenario,
        "repetition_index": repetition,
        "reserve_for": reserve_for,
    }
    return {
        "recipe_id": stable_id("recipe", identity),
        "recipe_index": recipe_index,
        "episode_index": recipe_index,
        "mission": mission,
        "scenario_index": scenario,
        "cell": cell_details(mission, scenario),
        "continuous_sample_ordinal": repetition,
        "refinement_level": 0 if repetition == 0 else int(math.floor(math.log2(repetition))) + 1,
        "repetition_index": repetition,
        "reserve_for": reserve_for,
        "schedule_phase": schedule_phase,
        "expected_credited_frames": float(expected_recipe_frames(calibration, mission, scenario)),
        "active": reserve_for is None,
    }


def create_plan(args: argparse.Namespace) -> None:
    if args.workers != 1:
        raise ValueError("Movement V1 planning requires exactly one worker")
    root = args.collection.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"collection directory is not empty: {root}")
    calibration = load_duration_calibration(args.duration_calibration)
    # V1 free play now uses full persistent episodes. Its duration is prescribed
    # by the plan rather than inherited from the historical ten-second calibration.
    calibration["expected_frames_by_scenario"]["semi_markov"] = [
        args.episode_seconds * args.observation_rate + 1
    ] * MISSION_COUNTS["semi_markov"]
    calibration_identity = stable_id("calibration", calibration)
    requirements = feasibility_requirements(calibration)
    minimum_frames = minimum_feasible_frame_budget(calibration)
    infeasible_budget = args.frame_budget < minimum_frames
    if infeasible_budget and not getattr(args, "allow_infeasible_diagnostic", False):
        raise ValueError(
            f"frame budget {args.frame_budget} is infeasible: complete discrete coverage "
            f"and the frozen nested frame shares require at least {minimum_frames} credited observations"
        )
    candidates, base_count, base_expected = build_candidate_schedule(calibration, args.frame_budget)
    plan_identity = {
        "version": PLAN_VERSION,
        "catalog": CATALOG_VERSION,
        "frame_budget": args.frame_budget,
        "split": args.split,
        "stage": "movement_v1",
        "episode_seconds": args.episode_seconds,
        "observation_rate_hz": args.observation_rate,
        "width": args.width,
        "height": args.height,
        "storage_format": args.storage_format,
        "seed_start": args.seed_start,
        "duration_calibration": calibration_identity,
        "mission_frame_shares": {mission: float(share) for mission, share in MISSION_FRAME_SHARES.items()},
        "base_recipe_count": base_count,
        "candidate_recipe_count": len(candidates),
    }
    plan_id = args.plan_id or stable_id("plan", plan_identity)

    active: list[dict[str, Any]] = []
    for recipe_index, (mission, scenario, repetition, phase) in enumerate(candidates):
        active.append(build_recipe(calibration, plan_id, recipe_index, mission, scenario, repetition, schedule_phase=phase))

    # The base schedule has exact whole-frame obligations even though mission
    # episodes have calibrated (sometimes fractional) expected durations.
    # Recording credits no more than this cap; tail candidates remain available
    # if realized episodes are shorter than calibration.
    remaining_targets = rounded_frame_targets(args.frame_budget)
    for index, recipe in enumerate(active):
        mission = str(recipe["mission"])
        expected = max(1, int(round(float(recipe["expected_credited_frames"]))))
        if index < base_count:
            planned = min(expected, remaining_targets[mission])
            remaining_targets[mission] -= planned
        else:
            planned = expected
        recipe["planned_credited_frames"] = planned
    if any(remaining_targets.values()):
        raise AssertionError(f"base schedule did not fill exact frame targets: {remaining_targets}")

    reserves: list[dict[str, Any]] = []
    next_index = len(active)
    reserve_repetitions: dict[tuple[str, int], int] = Counter(
        (recipe["mission"], int(recipe["scenario_index"])) for recipe in active
    )
    for primary in active:
        if primary["mission"] not in GUIDED_MISSIONS:
            continue
        key = (primary["mission"], int(primary["scenario_index"]))
        repetition = reserve_repetitions[key]
        reserves.append(build_recipe(
            calibration,
            plan_id,
            next_index,
            primary["mission"],
            primary["scenario_index"],
            repetition,
            primary["recipe_id"],
        ))
        reserves[-1]["planned_credited_frames"] = int(
            primary.get("planned_credited_frames", round(primary["expected_credited_frames"]))
        )
        reserve_repetitions[key] += 1
        next_index += 1

    generator = {
        "stage": "movement_v1",
        "episode_seconds": args.episode_seconds,
        "observation_rate_hz": args.observation_rate,
        "rgb_width": args.width,
        "rgb_height": args.height,
        "storage_format": args.storage_format,
        "webp_lossless_effort": args.webp_effort,
        "seed_start": args.seed_start,
    }
    assignments: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(active):
        # Single-recipe blocks for the final base window and all tail work bound
        # budget overshoot even when empirical episode lengths differ.
        remaining_base = max(0, base_count - cursor)
        single_recipe_window = max(args.tail_single_recipes, args.recipes_per_assignment)
        block_size = 1 if cursor >= base_count or remaining_base <= single_recipe_window else args.recipes_per_assignment
        block_limit = len(active) if cursor >= base_count else base_count
        block = active[cursor : min(block_limit, cursor + block_size)]
        assignment_number = len(assignments)
        logical_worker_id = 0
        assignment_id = f"assignment-{assignment_number:06d}"
        assignments.append({
            "schema_version": 1,
            "plan_id": plan_id,
            "plan_version": PLAN_VERSION,
            "assignment_id": assignment_id,
            "assignment_number": assignment_number,
            "dispatch_wave": assignment_number,
            "logical_worker_id": logical_worker_id,
            "split": args.split,
            "schedule_phase": block[0]["schedule_phase"],
            "generator": generator,
            "recipes": block,
        })
        cursor += len(block)

    expected_base_by_mission = {
        mission: sum(
            (Fraction(int(recipe["planned_credited_frames"])) for recipe in active[:base_count] if recipe["mission"] == mission),
            Fraction(),
        )
        for mission in MISSION_COUNTS
    }

    plan = {
        "schema_version": 2,
        "plan_version": PLAN_VERSION,
        "plan_id": plan_id,
        "created_utc": utc_now(),
        "catalog_version": CATALOG_VERSION,
        "catalog_counts": MISSION_COUNTS,
        "duration_calibration_version": calibration["calibration_version"],
        "duration_calibration_id": calibration_identity,
        "duration_calibration_source_frames": calibration["source_credited_frames"],
        "mandatory_recipe_count": sum(MISSION_COUNTS.values()),
        "base_recipe_count": base_count,
        "active_recipe_count": len(active),
        "tail_recipe_count": len(active) - base_count,
        "reserve_recipe_count": len(reserves),
        "target_accepted_frames": args.frame_budget,
        "target_frames_by_mission": rounded_frame_targets(args.frame_budget),
        "minimum_feasible_accepted_frames": minimum_frames,
        "diagnostic_only": bool(infeasible_budget),
        "distribution_feasible": not infeasible_budget,
        "feasibility_requirements": requirements,
        "expected_base_credited_frames": float(base_expected),
        "expected_base_frames_by_mission": {mission: float(value) for mission, value in expected_base_by_mission.items()},
        "mission_frame_shares": {mission: float(share) for mission, share in MISSION_FRAME_SHARES.items()},
        "object_mode_shares": {name: float(share) for name, share in OBJECT_MODE_SHARES.items()},
        "object_gaze_shares": {name: float(share) for name, share in OBJECT_GAZE_SHARES.items()},
        "facing_shares": {name: float(share) for name, share in FACING_SHARES.items()},
        "budget_completion_policy": (
            "diagnostic only; count semantically credited observations; stop before new work once target is reached; complete discrete coverage is not required"
            if infeasible_budget else
            "count only semantically credited observations; require complete discrete coverage; stop before new work once target is reached"
        ),
        "candidate_exhaustion_policy": "insufficient_capacity; never invent runtime recipes",
        "worker_count": 1,
        "recipes_per_assignment": args.recipes_per_assignment,
        "tail_single_recipes": args.tail_single_recipes,
        "assignment_count": len(assignments),
        "split": args.split,
        "generator": generator,
    }
    write_new_json(root / "plan" / "collection-plan.json", plan)
    write_new_jsonl(root / "plan" / "recipes.jsonl", [*active, *reserves])
    for assignment in assignments:
        write_new_json(root / "assignments" / f"{assignment['assignment_id']}.json", assignment)
    print(canonical_json({
        "plan_id": plan_id,
        "mandatory_recipes": sum(MISSION_COUNTS.values()),
        "base_recipes": base_count,
        "tail_recipes": len(active) - base_count,
        "active_recipes": len(active),
        "reserve_recipes": len(reserves),
        "assignments": len(assignments),
        "collection": str(root),
    }))


def result_files(root: Path) -> list[Path]:
    results = root / "results"
    return sorted(results.glob("*.json")) if results.exists() else []


def distribution_summary(
    recipes: list[dict[str, Any]],
    credited_frames: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Summarize mission and nested distributions in frame units."""
    mission_frames: Counter[str] = Counter()
    nested: dict[str, dict[str, Counter[str]]] = {
        "semi_markov": {"initial_behavior_family": Counter(), "initial_hold_band": Counter()},
        "object_view": {"target": Counter(), "mode": Counter(), "gaze": Counter(), "orbit_direction": Counter()},
        "contact_recovery": {"target": Counter(), "recovery": Counter(), "approach": Counter(), "facing": Counter()},
        "ramp_traverse": {"direction": Counter(), "path": Counter(), "facing": Counter()},
        "hoop_pass": {"direction": Counter(), "path": Counter(), "facing": Counter()},
        "static_no_input": {"location": Counter()},
    }
    recipe_counts: Counter[str] = Counter()
    for recipe in recipes:
        recipe_id = recipe["recipe_id"]
        frames = (
            float(credited_frames.get(recipe_id, 0))
            if credited_frames is not None
            else float(recipe.get("planned_credited_frames", recipe["expected_credited_frames"]))
        )
        if frames <= 0:
            continue
        mission = recipe["mission"]
        recipe_counts[mission] += 1
        mission_frames[mission] += frames
        cell = recipe["cell"]
        for dimension in nested[mission]:
            source = "direction" if dimension == "orbit_direction" else dimension
            value = cell.get(source)
            if value is not None:
                nested[mission][dimension][str(value)] += frames

    total = float(sum(mission_frames.values()))
    result: dict[str, Any] = {
        "total_frames": round(total, 6),
        "mission_recipe_counts": {mission: recipe_counts[mission] for mission in MISSION_COUNTS},
        "mission_frames": {mission: round(float(mission_frames[mission]), 6) for mission in MISSION_COUNTS},
        "mission_shares": {
            mission: (round(float(mission_frames[mission]) / total, 8) if total else 0.0)
            for mission in MISSION_COUNTS
        },
        "nested": {},
    }
    for mission, dimensions in nested.items():
        result["nested"][mission] = {}
        for dimension, values in dimensions.items():
            denominator = float(sum(values.values()))
            result["nested"][mission][dimension] = {
                "frames": {name: round(float(frames), 6) for name, frames in sorted(values.items())},
                "shares": {
                    name: (round(float(frames) / denominator, 8) if denominator else 0.0)
                    for name, frames in sorted(values.items())
                },
            }
    return result


def plan_distribution_command(args: argparse.Namespace) -> None:
    root = args.collection.resolve()
    plan = read_json(root / "plan" / "collection-plan.json")
    recipes = read_jsonl(root / "plan" / "recipes.jsonl")
    base = [recipe for recipe in recipes if recipe["active"]][: int(plan["base_recipe_count"])]
    print(json.dumps(distribution_summary(base), indent=2, sort_keys=True))


def build_inventory(root: Path) -> dict[str, Any]:
    plan = read_json(root / "plan" / "collection-plan.json")
    recipes = read_jsonl(root / "plan" / "recipes.jsonl")
    active_ids = {recipe["recipe_id"] for recipe in recipes if recipe["active"]}
    successful_assignments: set[str] = set()
    technical_failures: list[dict[str, Any]] = []
    accepted_frames = 0
    produced_frames = 0
    accepted_frames_by_mission: Counter[str] = Counter()
    accepted_recipes: set[str] = set()
    credited_cells: set[tuple[str, int]] = set()
    semantic_failures: set[str] = set()
    duplicate_results: list[str] = []
    result_attempts: set[tuple[str, str]] = set()
    credited_frames_by_recipe: dict[str, int] = {}
    for path in result_files(root):
        result = read_json(path)
        result_attempts.add((str(result.get("assignment_id")), str(result.get("attempt_id"))))
        if result.get("technical_result") != "validated":
            technical_failures.append(result)
            continue
        assignment_id = result["assignment_id"]
        if assignment_id in successful_assignments:
            duplicate_results.append(path.name)
            continue
        successful_assignments.add(assignment_id)
        accepted_frames += int(result.get("accepted_observation_frames", 0))
        produced_frames += int(result.get("produced_observation_frames", result.get("accepted_observation_frames", 0)))
        accepted_frames_by_mission.update({
            mission: int(frames)
            for mission, frames in result.get("accepted_frames_by_mission", {}).items()
        })
        accepted_recipes.update(result.get("resolved_recipe_ids", []))
        semantic_failures.update(result.get("semantic_failure_recipe_ids", []))
        for cell in result.get("credited_cells", []):
            credited_cells.add((cell["mission"], int(cell["scenario_index"])))
            credited_frames_by_recipe[str(cell.get("recipe_id"))] = int(cell.get("credited_observation_frames", 0))
    expected_cells = {(mission, scenario) for mission, count in MISSION_COUNTS.items() for scenario in range(count)}
    missing_cells = sorted(expected_cells - credited_cells)
    assignments = sorted((root / "assignments").glob("*.json"))
    unresolved_active = active_ids - accepted_recipes
    candidate_exhausted = not unresolved_active
    orphaned_claims: list[dict[str, Any]] = []
    claims_dir = root / "claims"
    if claims_dir.exists():
        for path in sorted(claims_dir.glob("*.json")):
            claim = read_json(path)
            attempt_id = f"attempt-{int(claim['attempt_number']):03d}"
            if (str(claim["assignment_id"]), attempt_id) not in result_attempts:
                orphaned_claims.append({
                    "assignment_id": claim["assignment_id"],
                    "attempt_id": attempt_id,
                    "executor_id": claim.get("executor_id"),
                    "claimed_utc": claim.get("claimed_utc"),
                })
    inventory = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "reconstructed_utc": utc_now(),
        "target_accepted_frames": plan["target_accepted_frames"],
        "accepted_observation_frames": accepted_frames,
        "produced_observation_frames": produced_frames,
        "accepted_frames_by_mission": {mission: accepted_frames_by_mission[mission] for mission in MISSION_COUNTS},
        "accepted_frame_shares_by_mission": {
            mission: (accepted_frames_by_mission[mission] / accepted_frames if accepted_frames else 0.0)
            for mission in MISSION_COUNTS
        },
        "credited_distribution": distribution_summary(recipes, credited_frames_by_recipe),
        "budget_reached": accepted_frames >= plan["target_accepted_frames"],
        "active_recipe_count": len(active_ids),
        "resolved_active_recipe_count": len(active_ids & accepted_recipes),
        "successful_assignment_count": len(successful_assignments),
        "assignment_count": len(assignments),
        "technical_failure_attempt_count": len(technical_failures),
        "interrupted_claims_without_result": orphaned_claims,
        "semantic_failure_recipe_ids": sorted(semantic_failures),
        "credited_discrete_cell_count": len(credited_cells),
        "expected_discrete_cell_count": len(expected_cells),
        "missing_credited_cells": [{"mission": mission, "scenario_index": scenario} for mission, scenario in missing_cells],
        "coverage_complete": not missing_cells,
        "candidate_capacity_exhausted": candidate_exhausted,
        "insufficient_capacity": candidate_exhausted and (accepted_frames < plan["target_accepted_frames"] or bool(missing_cells)),
        "duplicate_validated_results_ignored": duplicate_results,
        "complete": accepted_frames >= plan["target_accepted_frames"] and (
            bool(plan.get("diagnostic_only")) or not missing_cells
        ),
    }
    return inventory


def inventory_command(args: argparse.Namespace) -> None:
    inventory = build_inventory(args.collection.resolve())
    if args.write_snapshot:
        snapshot = args.collection.resolve() / "snapshots" / f"inventory-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        write_new_json(snapshot, inventory)
        inventory["snapshot"] = str(snapshot)
    print(json.dumps(inventory, indent=2, sort_keys=True))


def activate_reserves(args: argparse.Namespace) -> None:
    root = args.collection.resolve()
    inventory = build_inventory(root)
    failed = set(inventory["semantic_failure_recipe_ids"])
    recipes = read_jsonl(root / "plan" / "recipes.jsonl")
    existing_assignments = [read_json(path) for path in (root / "assignments").glob("*.json")]
    already_assigned = {
        recipe["recipe_id"]
        for assignment in existing_assignments
        for recipe in assignment["recipes"]
    }
    reserves = [
        recipe
        for recipe in recipes
        if recipe.get("reserve_for") in failed and recipe["recipe_id"] not in already_assigned
    ]
    if not reserves:
        print(canonical_json({"activated": 0, "reason": "no unresolved semantic failures with reserves"}))
        return
    existing = {path.stem for path in (root / "assignments").glob("*.json")}
    activation_sources = sorted({str(recipe["reserve_for"]) for recipe in reserves})
    activation_id = stable_id("reserve", activation_sources)
    assignment_id = f"assignment-{activation_id}"
    if assignment_id in existing:
        print(canonical_json({"activated": 0, "reason": "already activated", "assignment_id": assignment_id}))
        return
    plan = read_json(root / "plan" / "collection-plan.json")
    assignment = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "plan_version": plan["plan_version"],
        "assignment_id": assignment_id,
        "assignment_number": max(
            (int(read_json(path).get("assignment_number", -1)) for path in (root / "assignments").glob("*.json")),
            default=-1,
        ) + 1,
        "logical_worker_id": args.worker_id,
        "split": plan["split"],
        "reserve_activation_for": activation_sources,
        "generator": plan["generator"],
        "recipes": [{**recipe, "active": True} for recipe in reserves],
    }
    write_new_json(root / "assignments" / f"{assignment_id}.json", assignment)
    write_new_json(root / "reserve-activations" / f"{activation_id}.json", {
        "activation_id": activation_id,
        "created_utc": utc_now(),
        "assignment_id": assignment_id,
        "failed_recipe_ids": activation_sources,
        "reserve_recipe_ids": [recipe["recipe_id"] for recipe in reserves],
    })
    print(canonical_json({"activated": len(reserves), "assignment_id": assignment_id}))


def verify_plan(args: argparse.Namespace) -> None:
    root = args.collection.resolve()
    plan = read_json(root / "plan" / "collection-plan.json")
    recipes = read_jsonl(root / "plan" / "recipes.jsonl")
    active = [recipe for recipe in recipes if recipe["active"]]
    ids = [recipe["recipe_id"] for recipe in recipes]
    indices = [recipe["episode_index"] for recipe in recipes]
    if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
        raise ValueError("recipe IDs and episode indices must be globally unique")
    cells = Counter((recipe["mission"], recipe["scenario_index"]) for recipe in active)
    missing = [
        (mission, scenario)
        for mission, count in MISSION_COUNTS.items()
        for scenario in range(count)
        if cells[(mission, scenario)] == 0
    ]
    assignments = [read_json(path) for path in sorted((root / "assignments").glob("*.json"))]
    assigned = [recipe["recipe_id"] for assignment in assignments for recipe in assignment["recipes"]]
    if Counter(assigned) != Counter(recipe["recipe_id"] for recipe in active):
        raise ValueError("active recipes are not assigned exactly once")
    target_frames = plan.get("target_frames_by_mission")
    planned_frames: Counter[str] = Counter()
    if target_frames is not None:
        base = active[: int(plan["base_recipe_count"])]
        planned_frames.update({
            mission: sum(
                int(recipe["planned_credited_frames"])
                for recipe in base if recipe["mission"] == mission
            )
            for mission in MISSION_COUNTS
        })
        if dict(planned_frames) != {name: int(target_frames[name]) for name in MISSION_COUNTS}:
            raise ValueError("base recipes do not exactly satisfy target_frames_by_mission")
    report = {
        "valid": not missing,
        "plan_id": plan["plan_id"],
        "active_recipes": len(active),
        "assignments": len(assignments),
        "discrete_cells": len(cells),
        "expected_discrete_cells": sum(MISSION_COUNTS.values()),
        "missing_cells": missing,
        "target_frames_by_mission": target_frames,
        "planned_base_frames_by_mission": dict(planned_frames) if target_frames is not None else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        raise ValueError("plan omits mandatory discrete cells")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="create an immutable collection plan")
    plan.add_argument("collection", type=Path)
    plan.add_argument("--frame-budget", type=int, required=True)
    plan.add_argument(
        "--allow-infeasible-diagnostic",
        action="store_true",
        help="allow a below-floor diagnostic plan that stops at its frame budget without claiming complete coverage",
    )
    plan.add_argument("--workers", type=int, choices=(1,), default=1)
    plan.add_argument("--recipes-per-assignment", type=int, default=32)
    plan.add_argument("--tail-single-recipes", type=int, default=64)
    plan.add_argument("--episode-seconds", type=int, default=150)
    plan.add_argument("--observation-rate", type=int, default=20)
    plan.add_argument("--width", type=int, default=256)
    plan.add_argument("--height", type=int, default=256)
    plan.add_argument("--storage-format", choices=("webp_parquet", "png_jsonl"), default="webp_parquet")
    plan.add_argument("--webp-effort", type=int, default=0)
    plan.add_argument("--seed-start", type=int, default=1000)
    plan.add_argument("--duration-calibration", type=Path, default=DEFAULT_DURATION_CALIBRATION)
    plan.add_argument("--split", default="train")
    plan.add_argument("--plan-id")
    plan.set_defaults(func=create_plan)
    inventory = commands.add_parser("inventory", help="reconstruct collection state from immutable results")
    inventory.add_argument("collection", type=Path)
    inventory.add_argument("--write-snapshot", action="store_true")
    inventory.set_defaults(func=inventory_command)
    reserves = commands.add_parser("activate-reserves", help="assign predefined reserves for semantic failures")
    reserves.add_argument("collection", type=Path)
    reserves.add_argument("--worker-id", type=int, default=0)
    reserves.set_defaults(func=activate_reserves)
    verify = commands.add_parser("verify-plan", help="validate complete catalog coverage and non-overlap")
    verify.add_argument("collection", type=Path)
    verify.set_defaults(func=verify_plan)
    distribution = commands.add_parser("plan-distribution", help="report expected base-plan mission and nested frame shares")
    distribution.add_argument("collection", type=Path)
    distribution.set_defaults(func=plan_distribution_command)
    calibration = commands.add_parser("export-duration-calibration", help="export per-scenario frame means from a complete collection")
    calibration.add_argument("collection", type=Path)
    calibration.add_argument("output", type=Path)
    calibration.add_argument("--calibration-version", required=True)
    calibration.set_defaults(func=export_duration_calibration)
    return root


def main() -> int:
    args = parser().parse_args()
    if (
        getattr(args, "workers", 1) < 1
        or getattr(args, "recipes_per_assignment", 1) < 1
        or getattr(args, "tail_single_recipes", 1) < 1
    ):
        raise ValueError("workers, recipes-per-assignment, and tail-single-recipes must be positive")
    args.func(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
