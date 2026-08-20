#!/usr/bin/env python3
"""Finalize native WebP captures with typed Parquet metadata in the same tar."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit(
        "PyArrow is required. Install it with: python -m pip install -r "
        "Scripts/requirements-production.txt"
    ) from error


VECTOR3 = pa.struct(
    [pa.field("x", pa.float64()), pa.field("y", pa.float64()), pa.field("z", pa.float64())]
)
GRENADE = pa.struct(
    [
        pa.field("id", pa.int32()),
        pa.field("position", VECTOR3),
        pa.field("velocity", VECTOR3),
        pa.field("resting", pa.bool_()),
    ]
)
V2_THROW = pa.struct(
    [
        pa.field("grenade_id", pa.int32(), nullable=False),
        pa.field("preview_start_frame", pa.int32(), nullable=False),
        pa.field("throw_source_frame", pa.int32(), nullable=False),
        pa.field("camera_yaw", pa.float64(), nullable=False),
        pa.field("camera_pitch", pa.float64(), nullable=False),
        pa.field("launch_position", VECTOR3),
        pa.field("launch_velocity", VECTOR3),
        pa.field("physics_config_identity", pa.string()),
        pa.field("realized_target", pa.string()),
        pa.field("realized_contact_order", pa.list_(pa.string()), nullable=False),
        pa.field("first_contact_frame", pa.int32()),
        pa.field("first_contact_position", VECTOR3),
        pa.field("first_contact_normal", VECTOR3),
        pa.field("first_contact_velocity", VECTOR3),
        pa.field("bounce_count", pa.int32(), nullable=False),
        pa.field("rest_frame", pa.int32()),
        pa.field("post_rest_tail_steps", pa.int32(), nullable=False),
        pa.field("arena_exit_frame", pa.int32()),
        pa.field("arena_exit_direction", pa.string()),
        pa.field("visible_observation_count", pa.int32(), nullable=False),
        pa.field("preview_to_realized_flight_parity", pa.bool_(), nullable=False),
    ]
)

FRAME_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("frame_index", pa.int32(), nullable=False),
        pa.field("simulation_step", pa.int32(), nullable=False),
        pa.field("rgb_key", pa.string(), nullable=False),
        pa.field("position", VECTOR3, nullable=False),
        pa.field("velocity", VECTOR3, nullable=False),
        pa.field(
            "camera",
            pa.struct(
                [
                    pa.field("yaw", pa.float64()),
                    pa.field("pitch", pa.float64()),
                    pa.field("roll", pa.float64()),
                ]
            ),
            nullable=False,
        ),
        pa.field("grounded", pa.bool_(), nullable=False),
        pa.field("contact", pa.bool_(), nullable=False),
        pa.field("contact_object", pa.string()),
        pa.field("crosshair_state", pa.string(), nullable=False),
        pa.field("cooldown_remaining_steps", pa.int32(), nullable=False),
        pa.field("q_visibility", pa.bool_(), nullable=False),
        pa.field("aim_lock_active", pa.bool_(), nullable=False),
        pa.field("trajectory_visible", pa.bool_(), nullable=False),
        pa.field("flying_grenade_count", pa.int32(), nullable=False),
        pa.field("resting_grenade_count", pa.int32(), nullable=False),
        pa.field("visible_grenade_count", pa.int32(), nullable=False),
        pa.field("total_grenade_count", pa.int32(), nullable=False),
        pa.field("v2_episode_phase", pa.string(), nullable=False),
        pa.field("v2_mission_type", pa.string()),
        pa.field("v2_mission_region_visible", pa.bool_(), nullable=False),
        pa.field("v2_preview_region_visible", pa.bool_(), nullable=False),
        pa.field("v2_visibility_degraded", pa.bool_(), nullable=False),
        pa.field("grenades", pa.list_(GRENADE), nullable=False),
        pa.field("collection_mission", pa.string(), nullable=False),
        pa.field("mission_phase", pa.string(), nullable=False),
        pa.field("mission_success_frame_index", pa.int32()),
        pa.field("coverage_target", pa.string()),
        pa.field("mission_review_slug", pa.string()),
        pa.field("object_view_mode", pa.string()),
        pa.field("object_gaze_pattern", pa.string()),
        pa.field("object_gaze_intent", pa.string()),
        pa.field("object_gaze_target", VECTOR3),
        pa.field("orbit_direction", pa.string()),
        pa.field("contact_phase", pa.string()),
        pa.field("contact_recovery_style", pa.string()),
        pa.field("contact_approach_profile", pa.string()),
        pa.field("guided_camera_style", pa.string()),
        pa.field("locomotion_facing_profile", pa.string()),
        pa.field("ramp_direction", pa.string()),
        pa.field("ramp_path_profile", pa.string()),
        pa.field("hoop_path_profile", pa.string()),
        pa.field("coverage_target_visible", pa.bool_(), nullable=False),
        pa.field("coverage_position_azimuth_bin", pa.int32()),
        pa.field("coverage_position_distance_band", pa.int32()),
        pa.field("coverage_waypoint_index", pa.int32(), nullable=False),
        pa.field("visited_azimuth_bins_mask", pa.uint32(), nullable=False),
        pa.field("required_azimuth_bins_mask", pa.uint32(), nullable=False),
        pa.field("required_azimuth_bin_count", pa.int32(), nullable=False),
        pa.field("visible_hold_steps", pa.int32(), nullable=False),
        pa.field("pitch_band", pa.int32(), nullable=False),
        pa.field("ramp_traversals", pa.int32(), nullable=False),
        pa.field("hoop_passes", pa.int32(), nullable=False),
        pa.field("contact_hold_steps", pa.int32(), nullable=False),
        pa.field("verified_contact_steps", pa.int32(), nullable=False),
        pa.field("recovery_steps", pa.int32(), nullable=False),
        pa.field("primary_objective_achieved", pa.bool_(), nullable=False),
        pa.field("post_objective_steps", pa.int32(), nullable=False),
        pa.field("required_post_objective_steps", pa.int32(), nullable=False),
        pa.field("post_success_steps", pa.int32(), nullable=False),
        pa.field("required_post_success_steps", pa.int32(), nullable=False),
        pa.field("post_success_style", pa.string()),
        pa.field("facing_moving_frames", pa.int32(), nullable=False),
        pa.field("facing_matched_frames", pa.int32(), nullable=False),
        pa.field("facing_match_ratio", pa.float64(), nullable=False),
        pa.field("movement_camera_yaw_delta_degrees", pa.float64(), nullable=False),
        pa.field("hoop_crossing_recorded", pa.bool_(), nullable=False),
        pa.field("hoop_crossing_y", pa.float64(), nullable=False),
        pa.field("hoop_crossing_z", pa.float64(), nullable=False),
        pa.field("mission_success", pa.bool_(), nullable=False),
        pa.field("mission_failed", pa.bool_(), nullable=False),
        pa.field("no_progress_steps", pa.int32(), nullable=False),
        pa.field("natural_play_contact_steps", pa.int32(), nullable=False),
        pa.field("natural_play_contact_limit_steps", pa.int32(), nullable=False),
        pa.field("natural_play_escape_active", pa.bool_(), nullable=False),
    ]
)

TRANSITION_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("source_frame_index", pa.int32(), nullable=False),
        pa.field("action_mask", pa.uint16(), nullable=False),
        *[
            pa.field(name, pa.bool_(), nullable=False)
            for name in (
                "w",
                "a",
                "s",
                "d",
                "arrow_up",
                "arrow_down",
                "arrow_left",
                "arrow_right",
                "q",
                "e",
            )
        ],
        *[
            pa.field(name, pa.float64(), nullable=False)
            for name in ("forward_axis", "right_axis", "pitch_axis", "yaw_axis")
        ],
        pa.field("e_request_edge", pa.bool_(), nullable=False),
        pa.field("e_accepted", pa.bool_(), nullable=False),
        pa.field("planar_movement_suppressed", pa.bool_(), nullable=False),
        pa.field("q_rising_edge", pa.bool_(), nullable=False),
        pa.field("q_falling_edge", pa.bool_(), nullable=False),
        pa.field("e_rejection_reason", pa.string(), nullable=False),
        pa.field("accepted_throw_grenade_id", pa.int32()),
        pa.field("cooldown_before_steps", pa.int32(), nullable=False),
        pa.field("cooldown_after_steps", pa.int32(), nullable=False),
        pa.field("cooldown_remaining_steps", pa.int32(), nullable=False),
        pa.field("observation_valid", pa.bool_(), nullable=False),
    ]
)

EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        *[
            pa.field(name, pa.int32(), nullable=False)
            for name in (
                "episode_index",
                "worker_id",
                "seed",
            )
        ],
        pa.field("prescribed", pa.bool_()),
        *[
            pa.field(name, pa.string())
            for name in (
                "plan_id",
                "plan_version",
                "assignment_id",
                "attempt_id",
                "executor_id",
                "split",
                "recipe_id",
                "v2_contract_version",
                "v2_source",
                "v2_replay_identity",
                "v2_mission_type",
                "v2_mission_family",
                "v2_event_kind",
                "v2_target_actor",
                "v2_target_region",
                "v2_canonical_physics_id",
            )
        ],
        *[
            pa.field(name, pa.int32())
            for name in (
                "continuous_sample_ordinal",
                "refinement_level",
                "repetition_index",
                "prescribed_scenario_index",
            )
        ],
        *[
            pa.field(name, pa.int32(), nullable=False)
            for name in (
                "requested_transitions",
                "actual_transitions",
                "observation_count",
            )
        ],
        pa.field("collection_mission", pa.string(), nullable=False),
        pa.field("mission_review_slug", pa.string()),
        pa.field("mission_observation_frames", pa.int32(), nullable=False),
        pa.field("post_success_observation_frames", pa.int32(), nullable=False),
        pa.field("mission_success_frame_index", pa.int32()),
        *[
            pa.field(name, pa.string())
            for name in (
                "coverage_target",
                "object_view_mode",
                "object_gaze_pattern",
            )
        ],
        pa.field("object_scenario_index", pa.int32()),
        pa.field("orbit_direction", pa.string()),
        pa.field("contact_scenario_index", pa.int32()),
        *[
            pa.field(name, pa.string())
            for name in (
                "contact_recovery_style",
                "contact_approach_profile",
                "guided_camera_style",
                "locomotion_facing_profile",
            )
        ],
        pa.field("ramp_scenario_index", pa.int32()),
        pa.field("ramp_direction", pa.string()),
        pa.field("ramp_path_profile", pa.string()),
        pa.field("hoop_scenario_index", pa.int32()),
        pa.field("hoop_path_profile", pa.string()),
        *[
            pa.field(name, pa.uint32(), nullable=False)
            for name in (
                "visited_azimuth_bins_mask",
                "visible_azimuth_bins_mask",
                "required_azimuth_bins_mask",
            )
        ],
        *[
            pa.field(name, pa.int32(), nullable=False)
            for name in (
                "visited_azimuth_bin_count",
                "required_azimuth_bin_count",
                "visible_azimuth_bin_count",
                "visible_hold_steps",
                "ramp_traversals",
                "hoop_passes",
                "contact_hold_steps",
                "verified_contact_steps",
                "recovery_steps",
            )
        ],
        pa.field("primary_objective_achieved", pa.bool_(), nullable=False),
        *[
            pa.field(name, pa.int32(), nullable=False)
            for name in (
                "post_objective_steps",
                "required_post_objective_steps",
                "post_success_steps",
                "required_post_success_steps",
            )
        ],
        pa.field("post_success_style", pa.string()),
        pa.field("facing_moving_frames", pa.int32(), nullable=False),
        pa.field("facing_matched_frames", pa.int32(), nullable=False),
        pa.field("facing_match_ratio", pa.float64(), nullable=False),
        pa.field("hoop_crossing_recorded", pa.bool_(), nullable=False),
        pa.field("hoop_crossing_y", pa.float64(), nullable=False),
        pa.field("hoop_crossing_z", pa.float64(), nullable=False),
        pa.field("final_distance_to_goal_cm", pa.float64(), nullable=False),
        pa.field("distance_to_goal_at_success_cm", pa.float64()),
        pa.field("natural_play_contact_escape_count", pa.int32(), nullable=False),
        pa.field("maximum_consecutive_contact_steps", pa.int32(), nullable=False),
        pa.field("accepted_for_balancing", pa.bool_(), nullable=False),
        pa.field("mission_required", pa.bool_(), nullable=False),
        pa.field("mission_success", pa.bool_(), nullable=False),
        pa.field("mission_parameters_json", pa.string(), nullable=False),
        pa.field("v2_accepted_throw_count", pa.int32()),
        pa.field("v2_mission_event_frame", pa.int32()),
        pa.field("v2_construction_certified", pa.bool_()),
        pa.field("v2_mission_region_visible_all_frames", pa.bool_()),
        pa.field("v2_preview_region_visible_all_q_frames", pa.bool_()),
        pa.field("v2_opening_arena_context_visible", pa.bool_()),
        pa.field("v2_visibility_degraded", pa.bool_(), nullable=False),
        pa.field("planned_credited_frames", pa.int32()),
        pa.field("v2_throws", pa.list_(V2_THROW)),
        pa.field("termination_reason", pa.string(), nullable=False),
    ]
)

STAGING = {
    "frames": ".frames.jsonl.staging",
    "transitions": ".transitions.jsonl.staging",
    "episodes": ".episodes.jsonl.staging",
}


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parquet_bytes(
    records: list[dict[str, Any]], schema: pa.Schema, *, episodes: bool = False
) -> bytes:
    records = [dict(record) for record in records]
    if "aim_lock_active" in schema.names:
        for record in records:
            record.setdefault("aim_lock_active", False)
            record.setdefault("trajectory_visible", bool(record.get("q_visibility")))
            grenades = record.get("grenades") or []
            record.setdefault(
                "flying_grenade_count",
                sum(not grenade.get("resting", False) for grenade in grenades),
            )
            record.setdefault(
                "resting_grenade_count",
                sum(bool(grenade.get("resting", False)) for grenade in grenades),
            )
            record.setdefault("visible_grenade_count", 0)
            record.setdefault("total_grenade_count", len(grenades))
            record.setdefault("v2_episode_phase", "not_applicable")
            record.setdefault("v2_mission_type", None)
            record.setdefault("v2_mission_region_visible", False)
            record.setdefault("v2_preview_region_visible", False)
            record.setdefault("v2_visibility_degraded", False)
    if "planar_movement_suppressed" in schema.names:
        for record in records:
            record.setdefault("planar_movement_suppressed", False)
            record.setdefault("q_rising_edge", False)
            record.setdefault("q_falling_edge", False)
            record.setdefault("e_rejection_reason", "none")
            record.setdefault("accepted_throw_grenade_id", None)
            cooldown = int(record.get("cooldown_remaining_steps", 0))
            record.setdefault("cooldown_before_steps", cooldown)
            record.setdefault("cooldown_after_steps", cooldown)
    if episodes:
        for record in records:
            record.setdefault("v2_mission_type", None)
            record.setdefault("v2_mission_family", None)
            record.setdefault("v2_event_kind", None)
            record.setdefault("v2_target_actor", None)
            record.setdefault("v2_target_region", None)
            record.setdefault("v2_canonical_physics_id", None)
            record.setdefault("v2_mission_event_frame", None)
            record.setdefault("v2_construction_certified", None)
            record.setdefault("v2_mission_region_visible_all_frames", None)
            record.setdefault("v2_preview_region_visible_all_q_frames", None)
            record.setdefault("v2_opening_arena_context_visible", None)
            record.setdefault("v2_visibility_degraded", False)
            record["mission_parameters_json"] = json.dumps(
                record.pop("mission_parameters"),
                separators=(",", ":"),
                sort_keys=True,
            )
    table = pa.Table.from_pylist(records, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=True,
        version="2.6",
        write_statistics=True,
    )
    return sink.getvalue().to_pybytes()


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    tar.addfile(info, io.BytesIO(payload))


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append typed Parquet metadata to a native WebP dataset shard."
    )
    parser.add_argument("dataset", type=Path)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    dataset_dir = parse_args().dataset.resolve()
    dataset_path = dataset_dir / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("rgb_format") != "lossless_webp":
        raise SystemExit("Dataset is not a native lossless-WebP capture.")
    if dataset.get("metadata_format") != "parquet_pending":
        raise SystemExit("Dataset is not waiting for Parquet finalization.")
    shards = dataset.get("shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise SystemExit("Production v1 finalizer requires exactly one shard.")

    staging_paths = {
        name: dataset_dir / filename for name, filename in STAGING.items()
    }
    missing = [str(path) for path in staging_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing staging metadata: {', '.join(missing)}")

    records = {name: read_json_lines(path) for name, path in staging_paths.items()}
    expected = {
        "frames": int(dataset["observation_count"]),
        "transitions": int(dataset["transition_count"]),
        "episodes": int(dataset["completed_episode_count"]),
    }
    for name, count in expected.items():
        if len(records[name]) != count:
            raise SystemExit(
                f"{name} staging row count {len(records[name])} != expected {count}"
            )

    payloads = {
        "frames.parquet": parquet_bytes(records["frames"], FRAME_SCHEMA),
        "transitions.parquet": parquet_bytes(
            records["transitions"], TRANSITION_SCHEMA
        ),
        "episodes.parquet": parquet_bytes(
            records["episodes"], EPISODE_SCHEMA, episodes=True
        ),
    }
    manifest = {
        "schema_version": dataset["schema_version"],
        "curriculum_version": dataset["curriculum_version"],
        "complete": True,
        "worker_id": dataset["worker_id"],
        "episode_count": expected["episodes"],
        "transition_count": expected["transitions"],
        "observation_count": expected["frames"],
        "image_format": "lossless_webp",
        "webp_lossless_effort": int(dataset.get("webp_lossless_effort", 0)),
        "metadata_format": "parquet",
        "metadata_compression": "zstd",
        "tables": {
            name: {"rows": expected[name.removesuffix(".parquet")], "bytes": len(data)}
            for name, data in payloads.items()
        },
        "field_encoding": {
            "episodes.mission_parameters_json": "canonical JSON for the mission-specific parameter object"
        },
    }
    payloads["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    shard_path = dataset_dir / shards[0]
    with tarfile.open(shard_path, mode="a") as tar:
        existing = {member.name for member in tar.getmembers()}
        duplicates = existing.intersection(payloads)
        if duplicates:
            raise SystemExit(
                f"Shard already contains production metadata: {sorted(duplicates)}"
            )
        for name, payload in payloads.items():
            add_bytes(tar, name, payload)

    checksum = md5_file(shard_path)
    (dataset_dir / "checksums.md5").write_text(
        f"{checksum}  {shard_path.name}\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    dataset.update(
        {
            "complete": True,
            "error": "",
            "metadata_format": "parquet",
            "metadata_compression": "zstd",
            "format_finalized_utc": datetime.now(timezone.utc).isoformat(),
            "parquet_finalization_seconds": round(elapsed, 6),
            "parquet_tables": manifest["tables"],
        }
    )
    dataset_path.write_text(
        json.dumps(dataset, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    for path in staging_paths.values():
        path.unlink()

    print(
        f"Finalized {expected['frames']} WebP observations and three Parquet "
        f"tables in {elapsed:.3f} seconds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
