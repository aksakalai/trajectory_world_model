import io
import json
import hashlib
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from world_model_trajectory.data.crosshair_canonicalization import (
    MIGRATION_VERSION,
    MigrationError,
    READY_GREEN,
    WHITE,
    canonical_json,
    canonicalize_frames_table,
    canonicalize_v1_rgb,
    crosshair_stencil,
    decode_rgb,
    encode_lossless_webp,
    inspect_stencil,
    migrate_v1_shard,
    normalize_frame_schema,
    validate_canonical_v1_shard,
    validate_migrated_v1_shard,
    validate_v2_table,
)


def image_with_stencil(color: np.ndarray = WHITE) -> np.ndarray:
    image = np.zeros((384, 384, 3), dtype=np.uint8)
    image[:, :] = (12, 34, 56)
    for x, y in crosshair_stencil(384, 384):
        image[y, x] = color
    return image


def frame_table(*, state: str = "Neutral", cooldown: int = 0) -> pa.Table:
    return pa.table(
        {
            "episode_id": ["episode-1"],
            "frame_index": pa.array([0], type=pa.int32()),
            "rgb_key": ["episodes/episode-1/frame-000000.webp"],
            "crosshair_state": [state],
            "cooldown_remaining_steps": pa.array([cooldown], type=pa.int32()),
            "q_visibility": [False],
            "aim_lock_active": [False],
            "trajectory_visible": [False],
        }
    )


def parquet_bytes(table: pa.Table) -> bytes:
    output = io.BytesIO()
    pq.write_table(table, output, compression="zstd")
    return output.getvalue()


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 1234
    archive.addfile(member, io.BytesIO(payload))


def build_shard(path: Path, *, bad_pixel: bool = False) -> None:
    image = image_with_stencil()
    if bad_pixel:
        x, y = crosshair_stencil(384, 384)[0]
        image[y, x] = (1, 2, 3)
    image_payload = encode_lossless_webp(image)
    frames = frame_table()
    transitions = pa.table(
        {"episode_id": ["episode-1"], "source_frame_index": pa.array([0], type=pa.int32())}
    )
    episodes = pa.table({"episode_id": ["episode-1"], "plan_version": ["movement-v1-test"]})
    manifest = {
        "observation_count": 1,
        "transition_count": 1,
        "episode_count": 1,
        "tables": {"frames.parquet": {"rows": 1, "bytes": len(parquet_bytes(frames))}},
    }
    with tarfile.open(path, "w") as archive:
        add_bytes(archive, "episodes/episode-1/frame-000000.webp", image_payload)
        add_bytes(archive, "frames.parquet", parquet_bytes(frames))
        add_bytes(archive, "transitions.parquet", parquet_bytes(transitions))
        add_bytes(archive, "episodes.parquet", parquet_bytes(episodes))
        add_bytes(archive, "manifest.json", canonical_json(manifest))


def test_exact_stencil_geometry() -> None:
    stencil = crosshair_stencil(384, 384)
    assert len(stencil) == 9 + 9 - 1 == 17
    assert {(x, y) for x, y in stencil if y == 192} == {(x, 192) for x in range(188, 197)}
    assert {(x, y) for x, y in stencil if x == 192} == {(192, y) for y in range(188, 197)}


def test_white_to_green_changes_only_stencil() -> None:
    source = image_with_stencil()
    corrected, count = canonicalize_v1_rgb(source)
    assert count == 17
    assert inspect_stencil(corrected) == "corrected"
    changed = np.any(source != corrected, axis=2)
    assert changed.sum() == 17
    assert np.all(corrected[changed] == READY_GREEN)


def test_rejects_one_incorrect_source_pixel() -> None:
    source = image_with_stencil()
    x, y = crosshair_stencil(384, 384)[7]
    source[y, x] = (0, 0, 0)
    with pytest.raises(MigrationError, match="invalid"):
        canonicalize_v1_rgb(source)


def test_idempotent_corrected_image_validation() -> None:
    source = image_with_stencil(READY_GREEN)
    corrected, count = canonicalize_v1_rgb(source)
    assert count == 0
    assert np.array_equal(corrected, source)


def test_lossless_webp_round_trip() -> None:
    source = image_with_stencil()
    assert np.array_equal(decode_rgb(encode_lossless_webp(source)), source)


def test_metadata_neutral_to_ready_and_nonzero_cooldown_rejection() -> None:
    corrected = canonicalize_frames_table(frame_table())
    assert corrected["crosshair_state"].to_pylist() == ["Ready"]
    assert corrected.schema == frame_table().schema
    with pytest.raises(MigrationError, match="nonzero cooldown"):
        canonicalize_frames_table(frame_table(cooldown=1))


@pytest.mark.parametrize(("state", "cooldown"), [("Ready", 0), ("Cooldown", 2)])
def test_v2_ready_and_cooldown_are_noops(state: str, cooldown: int) -> None:
    table = frame_table(state=state, cooldown=cooldown)
    assert validate_v2_table(table) is table


def test_schema_normalization_uses_only_explicit_defaults() -> None:
    source = pa.table({"a": pa.array([1], type=pa.int32())})
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("v2_visibility_degraded", pa.bool_())])
    normalized = normalize_frame_schema(source, schema, defaults={"v2_visibility_degraded": False})
    assert normalized.to_pydict() == {"a": [1], "v2_visibility_degraded": [False]}
    with pytest.raises(MigrationError, match="No explicit default"):
        normalize_frame_schema(source, schema)


def test_shard_migration_preserves_members_keys_and_supports_verified_resume(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    destination = tmp_path / "corrected.tar"
    build_shard(source)
    first = migrate_v1_shard(source, destination)
    second = migrate_v1_shard(source, destination)
    assert first == second
    assert first.image_count == first.frame_rows == 1
    assert first.changed_pixels_total == 17
    with tarfile.open(destination) as archive:
        names = [member.name for member in archive.getmembers()]
        assert names == [
            "episodes/episode-1/frame-000000.webp",
            "frames.parquet",
            "transitions.parquet",
            "episodes.parquet",
            "manifest.json",
        ]
        image = decode_rgb(archive.extractfile(names[0]).read())
        frames = pq.read_table(io.BytesIO(archive.extractfile("frames.parquet").read()))
        assert inspect_stencil(image) == "corrected"
        assert frames["rgb_key"].to_pylist() == [names[0]]
        assert frames["crosshair_state"].to_pylist() == ["Ready"]
    saved_report = json.loads((tmp_path / "corrected.tar.migration.json").read_text())
    assert saved_report == asdict(first)
    assert saved_report["migration_version"] == MIGRATION_VERSION
    validation = validate_migrated_v1_shard(source, destination)
    assert validation.image_count == 1
    assert validation.changed_pixels_total == 17
    assert validation.transition_rows == 1
    assert validation.episode_rows == 1
    canonical = validate_canonical_v1_shard(destination)
    assert canonical.image_count == 1
    assert canonical.zero_repairs == 0


def test_interrupted_partial_is_rebuilt_and_failure_is_cleaned(tmp_path: Path) -> None:
    source = tmp_path / "source.tar"
    destination = tmp_path / "corrected.tar"
    build_shard(source)
    partial = tmp_path / "corrected.tar.partial"
    partial.write_bytes(b"interrupted")
    migrate_v1_shard(source, destination)
    assert destination.is_file()
    assert not partial.exists()

    bad_source = tmp_path / "bad.tar"
    bad_destination = tmp_path / "bad-corrected.tar"
    build_shard(bad_source, bad_pixel=True)
    with pytest.raises(MigrationError):
        migrate_v1_shard(bad_source, bad_destination)
    assert not bad_destination.exists()
    assert not (tmp_path / "bad-corrected.tar.partial").exists()


def test_canonical_report_encoding_is_deterministic() -> None:
    value = {"z": 1, "a": [3, 2, 1]}
    assert canonical_json(value) == canonical_json(value)
    assert canonical_json(value).startswith(b'{\n  "a"')


def test_collection_cli_builds_resumable_corrected_collection(tmp_path: Path) -> None:
    source = tmp_path / "source-collection"
    source_output = source / "attempts/assignment-000000/attempt-000/output"
    source_output.mkdir(parents=True)
    shard = source_output / "shard-w000-000000.tar"
    build_shard(shard)
    dataset = {
        "schema_version": "movement_v1-test",
        "observation_count": 1,
        "parquet_tables": {"frames.parquet": {"rows": 1, "bytes": 0}},
        "webp_lossless_effort": 0,
    }
    (source_output / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    md5 = hashlib.md5(shard.read_bytes(), usedforsecurity=False).hexdigest()
    (source_output / "checksums.md5").write_text(
        f"{md5}  {shard.name}\n", encoding="utf-8"
    )
    (source / "execution-build.json").write_text('{"build": 1}\n', encoding="utf-8")
    corrected = tmp_path / "corrected-collection"
    script = Path(__file__).resolve().parents[1] / "scripts/migrate_crosshair_canonicalization.py"
    command = [
        sys.executable,
        str(script),
        "--source-collection",
        str(source),
        "--output-collection",
        str(corrected),
        "--expected-shards",
        "1",
        "--minimum-free-gib",
        "0",
        "--tool-commit",
        "test-commit",
        "--validation-level",
        "exhaustive",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert (corrected / "execution-build.json").read_bytes() == (
        source / "execution-build.json"
    ).read_bytes()
    corrected_output = corrected / "attempts/assignment-000000/attempt-000/output"
    corrected_dataset = json.loads((corrected_output / "dataset.json").read_text())
    assert corrected_dataset["source_schema_version"] == "movement_v1-test"
    assert corrected_dataset["crosshair_canonicalization"]["migration_version"] == MIGRATION_VERSION
    assert (corrected / "migration-release.json").is_file()
