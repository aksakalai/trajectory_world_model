#!/usr/bin/env python3
"""Plan, verify, inventory, and prepare human review for combined V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from v2_mission_catalog import (
    CATALOG_VERSION,
    FAMILY_FRAME_SHARES,
    TYPE_FRAME_SHARE,
    build_solution,
    catalog_fingerprint,
    mission_types,
    review_recipes,
)


PLAN_VERSION = "trajectory-throw-v2-sixty-two-missions-10"
CONTRACT_VERSION = "shared-persistent-semi-markov-1+certified-sixty-two-missions-1"
SOURCE_FRAME_SHARES = {
    "semi_markov": Fraction(7, 10),
    "mission": Fraction(3, 10),
}
MINIMUM_EPISODE_SECONDS = 120
MAXIMUM_EPISODE_SECONDS = 180
PRODUCTION_BUDGET_QUANTUM = 1  # rounded whole-frame 70/30 targets support any budget
V2_OBSERVATION_RATE = 20
FORBIDDEN_V2_KEYS = {
    "candidate_seed", "reserve_activation_for", "reserve_for",
    "replacement_for", "replacement_recipe", "alternate_seed",
}
GENERATOR_PIPELINE_FILES = (
    "Scripts/build_v2_review_set.py",
    "Scripts/dataset_worker.py",
    "Scripts/finalize_production_dataset.py",
    "Scripts/review_dataset.py",
    "Scripts/v2_dataset_controller.py",
    "Scripts/v2_mission_catalog.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def rounded_source_targets(frame_budget: int) -> dict[str, int]:
    """Return whole-frame 70/30 targets whose sum is exactly the budget."""
    mission = (frame_budget * 3 + 5) // 10
    return {"semi_markov": frame_budget - mission, "mission": mission}


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def _forbidden_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return ({str(key) for key in value if str(key) in FORBIDDEN_V2_KEYS}
                | set().union(*(_forbidden_keys(item) for item in value.values()), set()))
    if isinstance(value, list):
        return set().union(*(_forbidden_keys(item) for item in value), set())
    return set()


def recipe_identity_payload(recipe: dict[str, Any]) -> dict[str, Any]:
    """All immutable execution and credit fields covered by a recipe identity."""
    keys = (
        "active", "recipe_index", "episode_index", "scenario_index",
        "continuous_sample_ordinal", "refinement_level", "repetition_index",
        "seed", "mission", "mission_type", "source", "family", "cell_id",
        "variation_cell_id", "mission_solution", "split", "schedule_phase",
        "planned_credited_frames", "expected_credited_frames", "review_variant",
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "catalog_version": CATALOG_VERSION,
        **{key: recipe.get(key) for key in keys},
    }


def recipe_digest(recipe: dict[str, Any]) -> str:
    return stable_id("v2recipe", recipe_identity_payload(recipe))


def replay_digest(recipe: dict[str, Any]) -> str:
    return stable_id("v2replay", recipe_identity_payload(recipe))


def assignment_digest(assignment: dict[str, Any]) -> str:
    payload = {key: value for key, value in assignment.items() if key != "assignment_digest"}
    return stable_id("v2assignment", payload, length=32)


def seal_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    result = dict(assignment)
    result["assignment_digest"] = assignment_digest(result)
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_new_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def canonical_source_bytes(value: bytes) -> bytes:
    """Normalize text line endings before hashing cross-platform source."""
    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def generator_source_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    runtime_sources = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "Source" / "he_grenade_game").rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".cpp", ".h", ".cs"}
        and not path.name.endswith("Tests.cpp")
    )
    for relative in (*GENERATOR_PIPELINE_FILES, *runtime_sources):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_source_bytes((root / relative).read_bytes()))
        digest.update(b"\0")
    return digest.hexdigest()


def validate_episode_seconds(seconds: int) -> None:
    if not MINIMUM_EPISODE_SECONDS <= seconds <= MAXIMUM_EPISODE_SECONDS:
        raise ValueError(
            f"semi-Markov episodes must last {MINIMUM_EPISODE_SECONDS}-"
            f"{MAXIMUM_EPISODE_SECONDS} seconds"
        )


def validate_observation_rate(observation_rate: int) -> None:
    if observation_rate != V2_OBSERVATION_RATE:
        raise ValueError(
            f"V2 certified recipes require exactly {V2_OBSERVATION_RATE} Hz; "
            "a different rate needs a separately certified catalog"
        )


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def mission_expected_frames(item: Any, observation_rate: int) -> int:
    return math.ceil(float(item.duration_seconds) * observation_rate) + 1


def minimum_feasible_frame_budget(
    episode_seconds: int = 150,
    observation_rate: int = 20,
) -> int:
    """Calculate the V2 floor from mandatory durations and family shares."""
    validate_episode_seconds(episode_seconds)
    validate_observation_rate(observation_rate)
    semi_markov_mandatory_frames = episode_seconds * observation_rate + 1
    candidates = [
        Fraction(semi_markov_mandatory_frames, 1)
        / SOURCE_FRAME_SHARES["semi_markov"]
    ]
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in mission_types():
        grouped[item.family].append(item)
    for family, share in FAMILY_FRAME_SHARES.items():
        mandatory = sum(
            mission_expected_frames(item, observation_rate)
            for item in grouped[family]
        )
        candidates.append(Fraction(mandatory, 1) / share)
    return _round_up(
        max(_ceil_fraction(candidate) for candidate in candidates),
        PRODUCTION_BUDGET_QUANTUM,
    )


def mandatory_mission_order() -> list[Any]:
    """Interleave families so an early partial run is spatially diverse."""
    grouped: dict[str, list[Any]] = {
        family: [item for item in mission_types() if item.family == family]
        for family in FAMILY_FRAME_SHARES
    }
    result: list[Any] = []
    maximum = max(len(values) for values in grouped.values())
    for ordinal in range(maximum):
        for family in FAMILY_FRAME_SHARES:
            values = grouped[family]
            if ordinal < len(values):
                # Alternating ends separates opposing regions during coverage.
                index = ordinal // 2 if ordinal % 2 == 0 else len(values) - 1 - ordinal // 2
                result.append(values[index])
    if len(result) != 62 or len({item.slug for item in result}) != 62:
        raise AssertionError("mandatory order must contain all 62 types exactly once")
    return result


def _split_for(
    source: str,
    mission_type: str,
    repetition: int,
    seed: int,
    evaluation_percent: int,
) -> str:
    if evaluation_percent <= 0:
        return "train"
    # When a type has at least two repetitions, it is deterministically in both
    # splits. Remaining work follows the requested stable percentage.
    if source == "mission" and repetition == 0:
        return "train"
    if source == "mission" and repetition == 1:
        return "evaluation"
    bucket = int(
        hashlib.sha256(
            f"v2-split:{source}:{mission_type}:{seed}".encode("ascii")
        ).hexdigest()[:8],
        16,
    ) % 100
    return "evaluation" if bucket < evaluation_percent else "train"


def _materialize_recipe(
    draft: dict[str, Any],
    recipe_index: int,
    seed_start: int,
    evaluation_percent: int,
) -> dict[str, Any]:
    seed = seed_start + recipe_index
    split = _split_for(
        draft["source"],
        str(draft.get("mission_type") or "semi_markov"),
        int(draft["repetition_index"]),
        seed,
        evaluation_percent,
    )
    recipe = {
        "active": True,
        "recipe_index": recipe_index,
        "episode_index": recipe_index,
        "scenario_index": int(draft["scenario_index"]),
        "continuous_sample_ordinal": int(draft["repetition_index"]),
        "refinement_level": 0,
        "repetition_index": int(draft["repetition_index"]),
        "seed": seed,
        "mission": draft["mission"],
        "mission_type": draft.get("mission_type"),
        "source": draft["source"],
        "family": draft["family"],
        "cell_id": draft.get("mission_type"),
        "variation_cell_id": (
            stable_id(
                "v2cell",
                draft["mission_solution"]["variation"]["coverage_cell"],
            )
            if draft.get("mission_solution") else None
        ),
        "mission_solution": draft.get("mission_solution"),
        "split": split,
        "schedule_phase": draft["schedule_phase"],
        "planned_credited_frames": int(draft["planned_credited_frames"]),
        "expected_credited_frames": int(draft["expected_credited_frames"]),
    }
    if "review_variant" in draft:
        recipe["review_variant"] = int(draft["review_variant"])
    recipe["recipe_id"] = recipe_digest(recipe)
    recipe["replay_identity"] = replay_digest(recipe)
    return recipe


def build_recipes(
    frame_budget: int,
    episode_seconds: int,
    observation_rate: int,
    seed_start: int,
    evaluation_percent: int,
) -> list[dict[str, Any]]:
    if frame_budget <= 0:
        raise ValueError("frame budget must be positive")
    validate_episode_seconds(episode_seconds)
    validate_observation_rate(observation_rate)
    if not 0 <= evaluation_percent <= 100:
        raise ValueError("evaluation percent must be between 0 and 100")
    floor = minimum_feasible_frame_budget(episode_seconds, observation_rate)
    if frame_budget < floor:
        raise ValueError(
            f"V2 frame budget {frame_budget} is below calculated minimum {floor}"
        )
    type_values = mission_types()
    type_index = {item.slug: index for index, item in enumerate(type_values)}
    source_targets = rounded_source_targets(frame_budget)
    mission_total = source_targets["mission"]
    base_type_target, extra_type_targets = divmod(mission_total, len(type_values))
    type_targets = {
        item.slug: base_type_target + (index < extra_type_targets)
        for index, item in enumerate(type_values)
    }
    semi_target = source_targets["semi_markov"]
    drafts: list[dict[str, Any]] = []
    credited: Counter[str] = Counter()
    repetitions: Counter[str] = Counter()
    mission_sample_counts = {
        item.slug: math.ceil(
            type_targets[item.slug] / mission_expected_frames(item, observation_rate)
        )
        for item in type_values
    }

    # The one mandatory semi-Markov opening makes its floor term concrete.
    semi_expected = episode_seconds * observation_rate + 1
    drafts.append({
        "source": "semi_markov", "family": "semi_markov",
        "mission": "semi_markov", "scenario_index": 0,
        "repetition_index": 0, "schedule_phase": "mandatory_coverage",
        "expected_credited_frames": semi_expected,
        "planned_credited_frames": semi_expected,
    })
    semi_credited = semi_expected

    # First pass: every mission type once, before any deficit work.
    for item in mandatory_mission_order():
        expected = mission_expected_frames(item, observation_rate)
        solution = build_solution(
            item, 0, observation_rate, mission_sample_counts[item.slug]
        )
        drafts.append({
            "source": "mission", "family": item.family,
            "mission": item.slug, "mission_type": item.slug,
            "scenario_index": type_index[item.slug], "repetition_index": 0,
            "schedule_phase": "mandatory_coverage",
            "expected_credited_frames": expected,
            "planned_credited_frames": expected,
            "mission_solution": solution,
        })
        credited[item.slug] += expected
        repetitions[item.slug] = 1

    # Deterministic largest frame-deficit scheduling. Ties use catalog order.
    while any(credited[item.slug] < type_targets[item.slug] for item in type_values):
        item = max(
            type_values,
            key=lambda value: (type_targets[value.slug] - credited[value.slug], -type_index[value.slug]),
        )
        deficit = type_targets[item.slug] - credited[item.slug]
        if deficit <= 0:
            break
        repetition = repetitions[item.slug]
        expected = mission_expected_frames(item, observation_rate)
        drafts.append({
            "source": "mission", "family": item.family,
            "mission": item.slug, "mission_type": item.slug,
            "scenario_index": type_index[item.slug],
            "repetition_index": repetition,
            "schedule_phase": "frame_deficit",
            "expected_credited_frames": expected,
            "planned_credited_frames": min(expected, deficit),
            "mission_solution": build_solution(
                item, repetition, observation_rate,
                mission_sample_counts[item.slug],
            ),
        })
        credited[item.slug] += min(expected, deficit)
        repetitions[item.slug] += 1

    semi_repetition = 1
    while semi_credited < semi_target:
        planned = min(semi_expected, semi_target - semi_credited)
        drafts.append({
            "source": "semi_markov", "family": "semi_markov",
            "mission": "semi_markov", "scenario_index": semi_repetition,
            "repetition_index": semi_repetition,
            "schedule_phase": "frame_deficit",
            "expected_credited_frames": semi_expected,
            "planned_credited_frames": planned,
        })
        semi_credited += planned
        semi_repetition += 1

    recipes = [
        _materialize_recipe(draft, index, seed_start, evaluation_percent)
        for index, draft in enumerate(drafts)
    ]
    if sum(int(item["planned_credited_frames"]) for item in recipes) != frame_budget:
        raise AssertionError("V2 recipe scheduler did not preserve the exact budget")
    return recipes


def planned_distribution(recipes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source_frames: Counter[str] = Counter()
    family_frames: Counter[str] = Counter()
    type_frames: Counter[str] = Counter()
    split_frames: Counter[str] = Counter()
    variation_cells: dict[str, set[str]] = defaultdict(set)
    total = 0
    for recipe in recipes:
        frames = int(recipe["planned_credited_frames"])
        total += frames
        source_frames[str(recipe["source"])] += frames
        family_frames[str(recipe["family"])] += frames
        if recipe["source"] == "mission":
            type_frames[str(recipe["mission_type"])] += frames
            variation_cells[str(recipe["mission_type"])].add(
                str(recipe["variation_cell_id"])
            )
        split_frames[str(recipe["split"])] += frames
    return {
        "total_credited_frames": total,
        "source_frames": dict(sorted(source_frames.items())),
        "source_shares": {
            key: value / total if total else 0.0
            for key, value in sorted(source_frames.items())
        },
        "family_frames": dict(sorted(family_frames.items())),
        "mission_type_frames": dict(sorted(type_frames.items())),
        "mission_type_variation_cell_counts": {
            key: len(value) for key, value in sorted(variation_cells.items())
        },
        "split_frames": dict(sorted(split_frames.items())),
    }


def create_plan(args: argparse.Namespace) -> None:
    root = args.collection.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"collection directory is not empty: {root}")
    if args.workers != 1:
        raise ValueError("Trajectory/Throw V2 planning requires exactly one worker")
    if args.recipes_per_assignment < 1:
        raise ValueError("recipes-per-assignment must be positive")
    if not 64 <= args.width <= 4096 or not 64 <= args.height <= 4096:
        raise ValueError("width and height must be between 64 and 4096")
    recipes = build_recipes(
        args.frame_budget, args.episode_seconds, args.observation_rate,
        args.seed_start, args.evaluation_percent,
    )
    source_fingerprint = generator_source_fingerprint()
    generator = {
        "stage": "trajectory_throw_v2",
        "episode_seconds": args.episode_seconds,
        "observation_rate": args.observation_rate,
        "width": args.width,
        "height": args.height,
        "storage_format": args.storage_format,
        "webp_effort": args.webp_effort,
        "seed_start": args.seed_start,
        "evaluation_percent": args.evaluation_percent,
    }
    identity = {
        "plan_version": PLAN_VERSION,
        "contract_version": CONTRACT_VERSION,
        "catalog_sha256": catalog_fingerprint(),
        "generator_source_sha256": source_fingerprint,
        "frame_budget": args.frame_budget,
        "generator": generator,
        "replays": [recipe["replay_identity"] for recipe in recipes],
    }
    plan_id = args.plan_id or stable_id("v2plan", identity)
    assignments: list[dict[str, Any]] = []
    for number, start in enumerate(range(0, len(recipes), args.recipes_per_assignment)):
        assignment_id = f"assignment-{number:06d}"
        assignments.append(seal_assignment({
            "schema_version": 3,
            "plan_id": plan_id,
            "plan_version": PLAN_VERSION,
            "contract_version": CONTRACT_VERSION,
            "assignment_id": assignment_id,
            "assignment_number": number,
            "dispatch_wave": number,
            "logical_worker_id": 0,
            "split": "mixed",
            "generator": generator,
            "recipes": recipes[start:start + args.recipes_per_assignment],
        }))
    plan = {
        "schema_version": 5,
        "plan_version": PLAN_VERSION,
        "plan_id": plan_id,
        "created_utc": utc_now(),
        "contract_version": CONTRACT_VERSION,
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": catalog_fingerprint(),
        "generator_source_sha256": source_fingerprint,
        "target_accepted_frames": args.frame_budget,
        "target_frames_by_source": rounded_source_targets(args.frame_budget),
        "minimum_feasible_frame_budget": minimum_feasible_frame_budget(
            args.episode_seconds, args.observation_rate
        ),
        "active_recipe_count": len(recipes),
        "assignment_count": len(assignments),
        "worker_count": 1,
        "recipes_per_assignment": args.recipes_per_assignment,
        "source_frame_shares": {key: float(value) for key, value in SOURCE_FRAME_SHARES.items()},
        "family_frame_shares": {key: float(value) for key, value in FAMILY_FRAME_SHARES.items()},
        "mission_type_frame_share": float(TYPE_FRAME_SHARE),
        "mandatory_mission_type_count": 62,
        "generator": generator,
        "planned_distribution": planned_distribution(recipes),
    }
    write_new_json(root / "plan" / "collection-plan.json", plan)
    write_new_jsonl(root / "plan" / "recipes.jsonl", recipes)
    for assignment in assignments:
        write_new_json(root / "assignments" / f"{assignment['assignment_id']}.json", assignment)
    print(canonical_json({
        "plan_id": plan_id, "recipes": len(recipes),
        "assignments": len(assignments), "collection": str(root),
        "minimum_feasible_frame_budget": plan["minimum_feasible_frame_budget"],
    }))


def recipe_validation_errors(
    recipe: dict[str, Any],
    generator: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    forbidden = _forbidden_keys(recipe)
    if forbidden:
        errors.append(f"forbidden V2 fields: {sorted(forbidden)}")
    base_keys = {
        "active", "recipe_id", "recipe_index", "episode_index",
        "scenario_index", "continuous_sample_ordinal", "refinement_level",
        "repetition_index", "seed", "mission", "mission_type", "source",
        "family", "cell_id", "variation_cell_id", "mission_solution",
        "replay_identity", "split",
        "schedule_phase", "planned_credited_frames", "expected_credited_frames",
    }
    allowed = base_keys | ({"review_variant"} if "review_variant" in recipe else set())
    extras = set(recipe) - allowed
    missing = base_keys - set(recipe)
    if extras:
        errors.append(f"unknown immutable recipe fields: {sorted(extras)}")
    if missing:
        errors.append(f"missing immutable recipe fields: {sorted(missing)}")
        return errors
    try:
        rate = int(generator["observation_rate"])
        validate_observation_rate(rate)
        index = int(recipe["recipe_index"])
        repetition = int(recipe["repetition_index"])
        if recipe["episode_index"] != index:
            errors.append("episode_index differs from recipe_index")
        if recipe["continuous_sample_ordinal"] != repetition:
            errors.append("continuous sample ordinal differs from repetition")
        if int(recipe["seed"]) != int(generator["seed_start"]) + index:
            errors.append("seed is not the deterministic plan seed")
        if recipe["active"] is not True:
            errors.append("V2 recipe is not active")
        source = str(recipe["source"])
        if source == "mission":
            values = list(mission_types())
            by_slug = {item.slug: (position, item) for position, item in enumerate(values)}
            mission_type = str(recipe["mission_type"])
            if mission_type not in by_slug:
                errors.append("unknown mission type")
            else:
                position, item = by_slug[mission_type]
                sample_count_value = (
                    recipe.get("mission_solution", {})
                    .get("variation", {})
                    .get("budget_sample_count")
                )
                sample_count = (
                    int(sample_count_value)
                    if sample_count_value is not None else None
                )
                expected_solution = build_solution(
                    item, repetition, rate, sample_count
                )
                expected_frames = mission_expected_frames(item, rate)
                expected_fields = {
                    "mission": mission_type,
                    "family": item.family,
                    "cell_id": mission_type,
                    "variation_cell_id": stable_id(
                        "v2cell", expected_solution["variation"]["coverage_cell"]
                    ),
                    "scenario_index": position,
                    "mission_solution": expected_solution,
                    "expected_credited_frames": expected_frames,
                }
                for key, expected in expected_fields.items():
                    if recipe.get(key) != expected:
                        errors.append(f"{key} differs from the canonical catalog")
        elif source == "semi_markov":
            expected = {
                "mission": "semi_markov", "mission_type": None,
                "family": "semi_markov", "cell_id": None,
                "variation_cell_id": None,
                "mission_solution": None,
            }
            for key, value in expected.items():
                if recipe.get(key) != value:
                    errors.append(f"semi-Markov {key} changed")
        else:
            errors.append(f"invalid source {source!r}")
        expected_frames = int(recipe["expected_credited_frames"])
        planned_frames = int(recipe["planned_credited_frames"])
        if not 0 < planned_frames <= expected_frames:
            errors.append("planned frame credit is outside the recipe duration")
        if recipe["recipe_id"] != recipe_digest(recipe):
            errors.append("recipe identity hash mismatch")
        if recipe["replay_identity"] != replay_digest(recipe):
            errors.append("replay identity hash mismatch")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"malformed recipe: {error}")
    return errors


def validate_assignment_against_plan(
    root: Path,
    assignment: dict[str, Any],
) -> None:
    root = root.resolve()
    plan = read_json(root / "plan" / "collection-plan.json")
    canonical = {
        recipe["recipe_id"]: recipe
        for recipe in read_jsonl(root / "plan" / "recipes.jsonl")
    }
    forbidden = _forbidden_keys(assignment)
    if forbidden:
        raise ValueError(f"V2 assignment has forbidden fields: {sorted(forbidden)}")
    if assignment.get("assignment_digest") != assignment_digest(assignment):
        raise ValueError("V2 assignment identity hash mismatch")
    for key in ("plan_id", "plan_version", "contract_version", "generator"):
        expected = plan.get(key)
        if key == "contract_version" and expected is None:
            expected = CONTRACT_VERSION
        if assignment.get(key) != expected:
            raise ValueError(f"V2 assignment {key} differs from its plan")
    seen: set[str] = set()
    for recipe in assignment.get("recipes", []):
        recipe_id = str(recipe.get("recipe_id"))
        if recipe_id in seen or recipe_id not in canonical:
            raise ValueError("V2 assignment has duplicate or unknown recipe identity")
        seen.add(recipe_id)
        if recipe != canonical[recipe_id]:
            raise ValueError(f"V2 assignment recipe {recipe_id} differs from recipes.jsonl")
        errors = recipe_validation_errors(recipe, plan["generator"])
        if errors:
            raise ValueError(f"V2 recipe {recipe_id} is invalid: {'; '.join(errors)}")
    if not seen:
        raise ValueError("V2 assignment contains no recipes")


def verify_plan(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan = read_json(root / "plan" / "collection-plan.json")
    recipes = read_jsonl(root / "plan" / "recipes.jsonl")
    assignments = [read_json(path) for path in sorted((root / "assignments").glob("*.json"))]
    assigned_ids = [item["recipe_id"] for assignment in assignments for item in assignment.get("recipes", [])]
    recipe_ids = [item["recipe_id"] for item in recipes]
    distribution = planned_distribution(recipes)
    mission_types_present = {
        str(item.get("mission_type")) for item in recipes if item.get("source") == "mission"
    }
    mandatory = [item for item in recipes if item.get("schedule_phase") == "mandatory_coverage"]
    source_targets = rounded_source_targets(int(plan["target_accepted_frames"]))
    mission_total = source_targets["mission"]
    base_type_target, extra_type_targets = divmod(mission_total, 62)
    exact_type_frames = sorted(distribution["mission_type_frames"].values()) == sorted(
        [base_type_target + 1] * extra_type_targets
        + [base_type_target] * (62 - extra_type_targets)
    )
    exact_source_frames = distribution["source_frames"] == source_targets
    exact_assignment = Counter(assigned_ids) == Counter(recipe_ids)
    unique = len(recipe_ids) == len(set(recipe_ids)) and len({item["replay_identity"] for item in recipes}) == len(recipes)
    complete_coverage = len(mission_types_present) == 62 and len(
        {item.get("mission_type") for item in mandatory if item.get("source") == "mission"}
    ) == 62
    no_mutable_fallbacks = not _forbidden_keys([plan, recipes, assignments])
    recipe_errors = {
        recipe["recipe_id"]: errors
        for recipe in recipes
        if (errors := recipe_validation_errors(recipe, plan["generator"]))
    }
    assignment_errors: list[str] = []
    for assignment in assignments:
        try:
            validate_assignment_against_plan(root, assignment)
        except ValueError as error:
            assignment_errors.append(f"{assignment.get('assignment_id')}: {error}")
    immutable_recipes = not recipe_errors and not assignment_errors
    valid = all((
        exact_assignment, unique, complete_coverage, exact_type_frames,
        exact_source_frames, no_mutable_fallbacks, immutable_recipes,
        distribution["total_credited_frames"] == int(plan["target_accepted_frames"]),
    ))
    return {
        "valid": valid,
        "plan_id": plan["plan_id"],
        "recipes": len(recipes),
        "mission_type_count": len(mission_types_present),
        "mandatory_coverage_complete": complete_coverage,
        "assigned_exactly_once": exact_assignment,
        "exact_frame_budget": distribution["total_credited_frames"] == int(plan["target_accepted_frames"]),
        "exact_70_30_sources": exact_source_frames,
        "exact_half_percent_types": exact_type_frames,
        "identities_unique": unique,
        "no_replacement_fields": no_mutable_fallbacks,
        "immutable_recipe_content": immutable_recipes,
        "recipe_validation_errors": recipe_errors,
        "assignment_validation_errors": assignment_errors,
        "minimum_feasible_frame_budget": plan["minimum_feasible_frame_budget"],
        "planned_distribution": distribution,
    }


def result_files(root: Path) -> list[Path]:
    directory = root.resolve() / "results"
    return sorted(directory.glob("*.json")) if directory.exists() else []


def build_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan = read_json(root / "plan" / "collection-plan.json")
    validated_assignments: set[str] = set()
    source_frames: Counter[str] = Counter()
    type_frames: Counter[str] = Counter()
    produced = 0
    resolved: set[str] = set()
    technical_failures = 0
    semantic_failure_details: list[dict[str, Any]] = []
    visibility_degraded_recipe_ids: set[str] = set()
    assignments = {
        value["assignment_id"]: value
        for path in sorted((root / "assignments").glob("*.json"))
        if (value := read_json(path))
    }
    for path in result_files(root):
        result = read_json(path)
        if result.get("technical_result") != "validated":
            technical_failures += 1
            continue
        assignment_id = str(result["assignment_id"])
        assignment = assignments.get(assignment_id)
        expected_bindings = [] if assignment is None else [
            {
                "recipe_id": recipe["recipe_id"],
                "replay_identity": recipe.get("replay_identity"),
                "mission_type": recipe.get("mission_type"),
                "mission_solution": recipe.get("mission_solution"),
            }
            for recipe in assignment["recipes"]
        ]
        if (
            assignment is None
            or result.get("plan_id") != plan["plan_id"]
            or result.get("assignment_digest") != assignment.get("assignment_digest")
            or result.get("recipe_bindings") != expected_bindings
            or result.get("resolved_recipe_ids")
                != [recipe["recipe_id"] for recipe in assignment["recipes"]]
        ):
            technical_failures += 1
            continue
        if assignment_id in validated_assignments:
            continue
        validated_assignments.add(assignment_id)
        produced += int(result.get("produced_observation_frames", 0))
        resolved.update(result.get("resolved_recipe_ids", []))
        semantic_failure_details.extend(result.get("semantic_failure_details", []))
        visibility_degraded_recipe_ids.update(
            str(value) for value in result.get("visibility_degraded_recipe_ids", [])
        )
        for cell in result.get("credited_cells", []):
            frames = int(cell.get("credited_observation_frames", 0))
            source = str(cell.get("source") or ("semi_markov" if cell.get("mission") == "semi_markov" else "mission"))
            source_frames[source] += frames
            if source == "mission":
                type_frames[str(cell.get("mission_type") or cell.get("mission"))] += frames
    credited = sum(source_frames.values())
    target = int(plan["target_accepted_frames"])
    return {
        "schema_version": 4,
        "plan_id": plan["plan_id"],
        "reconstructed_utc": utc_now(),
        "target_accepted_frames": target,
        "accepted_observation_frames": credited,
        "produced_observation_frames": produced,
        "credited_frames_by_source": dict(sorted(source_frames.items())),
        "credited_frames_by_mission_type": dict(sorted(type_frames.items())),
        "budget_reached": credited >= target,
        "technical_failure_attempt_count": technical_failures,
        "semantic_failure_count": len(semantic_failure_details),
        "semantic_failure_details": semantic_failure_details,
        "visibility_degraded_recipe_count": len(visibility_degraded_recipe_ids),
        "visibility_degraded_recipe_ids": sorted(visibility_degraded_recipe_ids),
        "validated_assignment_count": len(validated_assignments),
        "resolved_recipe_count": len(resolved),
        "complete": credited >= target,
    }


def create_review_plan(args: argparse.Namespace) -> None:
    if args.workers != 1:
        raise ValueError("Trajectory/Throw V2 review planning requires exactly one worker")
    """Write review manifests only; this command never launches the game."""
    root = args.collection.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"review directory is not empty: {root}")
    examples_per_type = int(getattr(args, "examples_per_type", 3))
    recipes_per_assignment = int(getattr(args, "recipes_per_assignment", 1))
    if recipes_per_assignment < 1:
        raise ValueError("recipes-per-assignment must be positive")
    entries = [
        entry for entry in review_recipes(args.observation_rate)
        if int(entry["review_variant"]) < examples_per_type
    ]
    generator = {
        "stage": "trajectory_throw_v2",
        "episode_seconds": 150,
        "observation_rate": args.observation_rate,
        "width": 384,
        "height": 384,
        "storage_format": "webp_parquet",
        "webp_effort": 0,
        "seed_start": 900000,
        "evaluation_percent": 0,
    }
    recipes: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        item = next(value for value in mission_types() if value.slug == entry["mission_type"])
        expected = mission_expected_frames(item, args.observation_rate)
        draft = {
            "source": "mission", "family": item.family,
            "mission": item.slug, "mission_type": item.slug,
            "scenario_index": list(mission_types()).index(item),
            "repetition_index": entry["repetition_index"],
            "schedule_phase": "human_review",
            "expected_credited_frames": expected,
            "planned_credited_frames": expected,
            "mission_solution": entry["solution"],
            "review_variant": entry["review_variant"],
        }
        recipe = _materialize_recipe(draft, index, 900000, 0)
        recipes.append(recipe)
    plan_id = stable_id("v2reviewplan", [item["replay_identity"] for item in recipes])
    for assignment_number, start in enumerate(
        range(0, len(recipes), recipes_per_assignment)
    ):
        assignment_id = f"review-assignment-{assignment_number:03d}"
        assignments.append(seal_assignment({
            "schema_version": 3,
            "plan_id": plan_id,
            "plan_version": f"{PLAN_VERSION}-review-1",
            "contract_version": CONTRACT_VERSION,
            "assignment_id": assignment_id,
            "assignment_number": assignment_number,
            "dispatch_wave": assignment_number,
            "logical_worker_id": 0,
            "split": "review",
            "generator": generator,
            "recipes": recipes[start:start + recipes_per_assignment],
        }))
    manifest = {
        "schema_version": 1,
        "purpose": "v2_human_review",
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": catalog_fingerprint(),
        "width": 384,
        "height": 384,
        "examples_per_type": examples_per_type,
        "mission_type_count": 62,
        "video_count": len(recipes),
        "plan_id": plan_id,
        "entries": [
            {
                **entry,
                "recipe_id": recipes[index]["recipe_id"],
                "replay_identity": recipes[index]["replay_identity"],
                "episode_id": f"p-e{index:09d}",
                "assignment_id": assignments[index // recipes_per_assignment]["assignment_id"],
                "assignment_digest": assignments[index // recipes_per_assignment]["assignment_digest"],
                "review_id": stable_id("v2review", {
                    "type": entry["mission_type"],
                    "variant": entry["review_variant"],
                    "solution": entry["solution"],
                }),
                "output_video": f"{entry['mission_type']}--{entry['review_variant'] + 1}.mp4",
            }
            for index, entry in enumerate(entries)
        ],
    }
    write_new_json(root / "plan" / "collection-plan.json", {
        "schema_version": 1,
        "plan_version": f"{PLAN_VERSION}-review-1",
        "plan_id": plan_id,
        "contract_version": CONTRACT_VERSION,
        "generator_source_sha256": generator_source_fingerprint(),
        "target_accepted_frames": sum(item["planned_credited_frames"] for item in recipes),
        "active_recipe_count": len(recipes),
        "assignment_count": len(assignments),
        "worker_count": 1,
        "human_review_only": True,
        "generator": generator,
    })
    write_new_jsonl(root / "plan" / "recipes.jsonl", recipes)
    for assignment in assignments:
        write_new_json(root / "assignments" / f"{assignment['assignment_id']}.json", assignment)
    write_new_json(root / "v2-review-plan.json", manifest)
    print(canonical_json({"review_recipes": len(recipes), "collection": str(root)}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("collection", type=Path)
    plan.add_argument("--frame-budget", type=int, required=True)
    plan.add_argument("--workers", type=int, choices=(1,), default=1)
    plan.add_argument("--recipes-per-assignment", type=int, default=8)
    plan.add_argument("--episode-seconds", type=int, default=150)
    plan.add_argument("--observation-rate", type=int, default=20)
    plan.add_argument("--width", type=int, default=384)
    plan.add_argument("--height", type=int, default=384)
    plan.add_argument("--storage-format", choices=("webp_parquet", "png_jsonl"), default="webp_parquet")
    plan.add_argument("--webp-effort", type=int, default=0)
    plan.add_argument("--seed-start", type=int, default=200000)
    plan.add_argument("--evaluation-percent", type=int, default=10)
    plan.add_argument("--plan-id")
    plan.set_defaults(func=create_plan)
    verify = commands.add_parser("verify-plan")
    verify.add_argument("collection", type=Path)
    verify.set_defaults(func=lambda args: print(json.dumps(verify_plan(args.collection), indent=2, sort_keys=True)))
    inventory = commands.add_parser("inventory")
    inventory.add_argument("collection", type=Path)
    inventory.add_argument("--write-snapshot", action="store_true")
    def inventory_command(args: argparse.Namespace) -> None:
        value = build_inventory(args.collection)
        if args.write_snapshot:
            write_new_json(args.collection.resolve() / "inventory" / f"inventory-{stable_id('snapshot', value)}.json", value)
        print(json.dumps(value, indent=2, sort_keys=True))
    inventory.set_defaults(func=inventory_command)
    review = commands.add_parser("review-plan")
    review.add_argument("collection", type=Path)
    review.add_argument("--observation-rate", type=int, default=20)
    review.add_argument("--workers", type=int, choices=(1,), default=1)
    review.add_argument("--examples-per-type", type=int, choices=(1, 2, 3), default=3)
    review.add_argument("--recipes-per-assignment", type=int, default=1)
    review.set_defaults(func=create_review_plan)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
