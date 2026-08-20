"""Fail-closed V1 crosshair canonicalization for production tar shards."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tarfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


MIGRATION_VERSION = "v1-crosshair-canonicalization-1"
WHITE = np.array((242, 242, 242), dtype=np.uint8)
READY_GREEN = np.array((48, 242, 72), dtype=np.uint8)
COOLDOWN_RED = np.array((242, 48, 48), dtype=np.uint8)
REQUIRED_METADATA = frozenset(
    {"frames.parquet", "transitions.parquet", "episodes.parquet", "manifest.json"}
)


class MigrationError(RuntimeError):
    """Raised when a source invariant or validation gate fails."""


def crosshair_stencil(
    width: int,
    height: int,
    *,
    half_size: int = 4,
    thickness: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Return sorted ``(x, y)`` coordinates matching the Unreal integer loops."""
    if width <= 0 or height <= 0 or half_size < 0 or thickness < 1:
        raise ValueError("Invalid crosshair geometry")
    center_x, center_y = width // 2, height // 2
    coordinates: set[tuple[int, int]] = set()
    # C++ integer division truncates toward zero. Thickness is currently one,
    # but this expression preserves the documented renderer behavior.
    half_thickness = thickness // 2
    for offset in range(-half_size, half_size + 1):
        for t in range(-half_thickness, half_thickness + 1):
            coordinates.add(
                (min(max(center_x + offset, 0), width - 1), min(max(center_y + t, 0), height - 1))
            )
            coordinates.add(
                (min(max(center_x + t, 0), width - 1), min(max(center_y + offset, 0), height - 1))
            )
    return tuple(sorted(coordinates, key=lambda coordinate: (coordinate[1], coordinate[0])))


def inspect_stencil(rgb: np.ndarray) -> Literal["legacy", "corrected", "cooldown", "invalid"]:
    """Classify an exact center stencil without searching elsewhere in the image."""
    _validate_rgb_array(rgb)
    pixels = np.asarray([rgb[y, x] for x, y in crosshair_stencil(rgb.shape[1], rgb.shape[0])])
    if np.all(pixels == WHITE):
        return "legacy"
    if np.all(pixels == READY_GREEN):
        return "corrected"
    if np.all(pixels == COOLDOWN_RED):
        return "cooldown"
    return "invalid"


def canonicalize_v1_rgb(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace exactly the legacy V1 stencil; accept green input idempotently."""
    _validate_rgb_array(rgb)
    state = inspect_stencil(rgb)
    if state == "corrected":
        return rgb.copy(), 0
    if state != "legacy":
        raise MigrationError(f"Expected an all-white V1 stencil, found {state}")
    corrected = rgb.copy()
    stencil = crosshair_stencil(rgb.shape[1], rgb.shape[0])
    for x, y in stencil:
        corrected[y, x] = READY_GREEN
    difference = np.any(corrected != rgb, axis=2)
    if int(difference.sum()) != len(stencil):
        raise MigrationError("Pixel-difference count does not match the stencil")
    for x, y in stencil:
        difference[y, x] = False
    if difference.any():
        raise MigrationError("A non-stencil pixel changed")
    return corrected, len(stencil)


def _validate_rgb_array(rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise MigrationError("Expected an HxWx3 uint8 RGB array")
    if rgb.shape[:2] != (384, 384):
        raise MigrationError(f"Expected 384x384 input, found {rgb.shape[1]}x{rgb.shape[0]}")


def decode_rgb(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (384, 384):
            raise MigrationError(f"Expected 384x384 WebP, found {image.size[0]}x{image.size[1]}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def encode_lossless_webp(rgb: np.ndarray, *, method: int = 6) -> bytes:
    _validate_rgb_array(rgb)
    output = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        output, format="WEBP", lossless=True, method=method, exact=True
    )
    payload = output.getvalue()
    round_trip = decode_rgb(payload)
    if not np.array_equal(round_trip, rgb):
        raise MigrationError("Lossless WebP round trip changed decoded pixels")
    return payload


def canonicalize_frames_table(table: pa.Table, *, require_legacy: bool = False) -> pa.Table:
    """Validate V1 semantics and replace only Neutral with Ready."""
    required = {
        "crosshair_state",
        "cooldown_remaining_steps",
        "q_visibility",
        "aim_lock_active",
        "trajectory_visible",
    }
    missing = required.difference(table.column_names)
    if missing:
        raise MigrationError(f"frames.parquet is missing columns: {sorted(missing)}")
    values = {
        field: table[field].to_pylist()
        for field in required
    }
    for index in range(table.num_rows):
        if values["cooldown_remaining_steps"][index] != 0:
            raise MigrationError(f"V1 frame row {index} has nonzero cooldown")
        for field in ("q_visibility", "aim_lock_active", "trajectory_visible"):
            if values[field][index] is not False:
                raise MigrationError(f"V1 frame row {index} has {field} != false")
        state = values["crosshair_state"][index]
        if require_legacy and state != "Neutral":
            raise MigrationError(
                f"V1 source frame row {index} must be Neutral, found {state!r}"
            )
        if state not in ("Neutral", "Ready"):
            raise MigrationError(
                f"V1 frame row {index} has invalid crosshair_state "
                f"{state!r}"
            )
    field_index = table.schema.get_field_index("crosshair_state")
    field = table.schema.field(field_index)
    replacement = pa.chunked_array(
        [pa.array(["Ready"] * len(chunk), type=field.type) for chunk in table.column(field_index).chunks]
    )
    corrected = table.set_column(field_index, field, replacement)
    validate_frames_table_migration(table, corrected)
    return corrected


def validate_frames_table_migration(source: pa.Table, corrected: pa.Table) -> None:
    """Prove only Neutral/Ready canonicalization changed semantically."""
    if source.schema != corrected.schema:
        raise MigrationError("Corrected frames schema differs from source schema")
    if source.num_rows != corrected.num_rows:
        raise MigrationError("Corrected frames row count differs from source")
    for name in source.column_names:
        if name == "crosshair_state":
            expected = ["Ready"] * source.num_rows
            if corrected[name].to_pylist() != expected:
                raise MigrationError("Corrected crosshair_state values are not all Ready")
            if any(value not in ("Neutral", "Ready") for value in source[name].to_pylist()):
                raise MigrationError("Source crosshair_state contains an unexpected value")
        elif not source[name].equals(corrected[name]):
            raise MigrationError(f"Corrected frames field {name!r} changed unexpectedly")


def validate_v2_table(table: pa.Table) -> pa.Table:
    """Validate canonical V2 Ready/Cooldown semantics and return a no-op table."""
    required = {"crosshair_state", "cooldown_remaining_steps"}
    missing = required.difference(table.column_names)
    if missing:
        raise MigrationError(f"frames.parquet is missing columns: {sorted(missing)}")
    states = table["crosshair_state"].to_pylist()
    cooldowns = table["cooldown_remaining_steps"].to_pylist()
    for index, (state, cooldown) in enumerate(zip(states, cooldowns, strict=True)):
        expected = "Cooldown" if cooldown > 0 else "Ready"
        if state != expected:
            raise MigrationError(f"V2 frame row {index}: expected {expected}, found {state!r}")
    return table


def normalize_frame_schema(
    table: pa.Table, release_schema: pa.Schema, *, defaults: dict[str, Any] | None = None
) -> pa.Table:
    """Add explicitly defaulted release fields and order/cast all columns."""
    defaults = defaults or {}
    source = set(table.column_names)
    release = set(release_schema.names)
    unexpected = source - release
    if unexpected:
        raise MigrationError(f"Release schema would drop fields: {sorted(unexpected)}")
    arrays: list[pa.ChunkedArray] = []
    for field in release_schema:
        if field.name in source:
            arrays.append(table[field.name].cast(field.type, safe=True))
        elif field.name in defaults:
            arrays.append(pa.chunked_array([pa.array([defaults[field.name]] * table.num_rows, type=field.type)]))
        else:
            raise MigrationError(f"No explicit default for missing field {field.name!r}")
    return pa.Table.from_arrays(arrays, schema=release_schema)


def _parquet_bytes(table: pa.Table) -> bytes:
    output = io.BytesIO()
    pq.write_table(table, output, compression="zstd", version="2.6")
    return output.getvalue()


def _read_member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise MigrationError(f"Could not read tar member {member.name}")
    return extracted.read()


def _is_frame(name: str) -> bool:
    return name.startswith("episodes/") and "/frame-" in name and name.endswith(".webp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()


@dataclass(frozen=True)
class ShardReport:
    migration_version: str
    source_shard: str
    source_sha256: str
    corrected_shard: str
    corrected_sha256: str
    image_count: int
    frame_rows: int
    transition_rows: int
    episode_rows: int
    changed_pixels_per_image: int
    changed_pixels_total: int
    webp_method: int
    v2_unchanged: bool
    source_bytes: int
    corrected_bytes: int
    elapsed_seconds: float
    frames_schema: str


@dataclass(frozen=True)
class ShardValidationReport:
    migration_version: str
    source_shard: str
    corrected_shard: str
    source_sha256: str
    corrected_sha256: str
    image_count: int
    frame_rows: int
    transition_rows: int
    episode_rows: int
    changed_pixels_total: int
    unchanged_member_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class CanonicalShardReport:
    migration_version: str
    corrected_shard: str
    corrected_sha256: str
    image_count: int
    frame_rows: int
    zero_repairs: int
    elapsed_seconds: float


def _copy_info(member: tarfile.TarInfo, size: int) -> tarfile.TarInfo:
    result = copy.copy(member)
    result.size = size
    result.offset = 0
    result.offset_data = 0
    return result


def migrate_v1_shard(
    source: str | Path,
    destination: str | Path,
    *,
    webp_method: int = 6,
    progress_every: int = 0,
) -> ShardReport:
    """Build and atomically publish one validated corrected V1 shard."""
    started = time.monotonic()
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    report_path = destination.with_suffix(destination.suffix + ".migration.json")
    source_hash = _sha256(source)
    if destination.is_file() and report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("source_sha256") == source_hash
            and existing.get("corrected_sha256") == _sha256(destination)
            and existing.get("migration_version") == MIGRATION_VERSION
        ):
            return ShardReport(**existing)
        raise MigrationError(f"Existing output failed resume verification: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    with tarfile.open(source, "r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise MigrationError(f"Duplicate tar members: {duplicates[:5]}")
        missing = REQUIRED_METADATA.difference(names)
        if missing:
            raise MigrationError(f"Shard is missing members: {sorted(missing)}")
        payload_by_name = {
            name: _read_member_bytes(archive, archive.getmember(name)) for name in REQUIRED_METADATA
        }

    frames = pq.read_table(io.BytesIO(payload_by_name["frames.parquet"]))
    transitions = pq.read_table(io.BytesIO(payload_by_name["transitions.parquet"]))
    episodes = pq.read_table(io.BytesIO(payload_by_name["episodes.parquet"]))
    if "plan_version" not in episodes.column_names:
        raise MigrationError("episodes.parquet is missing plan_version")
    plan_versions = episodes["plan_version"].to_pylist()
    if not plan_versions or any(
        not isinstance(version, str) or not version.startswith("movement-v1")
        for version in plan_versions
    ):
        raise MigrationError("Shard provenance is not exclusively movement-v1")
    corrected_frames = canonicalize_frames_table(frames, require_legacy=True)
    corrected_frame_bytes = _parquet_bytes(corrected_frames)

    manifest = json.loads(payload_by_name["manifest.json"])
    if manifest.get("observation_count") != frames.num_rows:
        raise MigrationError("Manifest observation count does not match frames.parquet")
    if manifest.get("transition_count") != transitions.num_rows:
        raise MigrationError("Manifest transition count does not match transitions.parquet")
    if manifest.get("episode_count") != episodes.num_rows:
        raise MigrationError("Manifest episode count does not match episodes.parquet")
    manifest.setdefault("tables", {}).setdefault("frames.parquet", {})["bytes"] = len(corrected_frame_bytes)
    source_schema_version = manifest.get("schema_version")
    manifest["source_schema_version"] = source_schema_version
    manifest["schema_version"] = f"{source_schema_version}+{MIGRATION_VERSION}"
    manifest["crosshair_canonicalization"] = {
        "migration_version": MIGRATION_VERSION,
        "source_sha256": source_hash,
        "state_change": "Neutral -> Ready",
        "pixel_change": "(242,242,242) -> (48,242,72)",
    }
    replacement = {
        "frames.parquet": corrected_frame_bytes,
        "manifest.json": canonical_json(manifest),
    }

    frame_keys = frames["rgb_key"].to_pylist()
    if len(frame_keys) != len(set(frame_keys)):
        raise MigrationError("frames.parquet contains duplicate rgb_key values")
    expected_keys = set(frame_keys)
    ordered_image_members = [name for name in names if _is_frame(name)]
    if ordered_image_members != frame_keys:
        raise MigrationError("Tar image member order differs from frames.parquet rgb_key order")
    seen_keys: set[str] = set()
    changed_total = 0
    image_started = time.monotonic()
    recent_started = image_started
    recent_images = 0
    try:
        with tarfile.open(source, "r:*") as input_tar, tarfile.open(partial, "w") as output_tar:
            for member in input_tar:
                if not member.isfile():
                    output_tar.addfile(copy.copy(member))
                    continue
                payload = _read_member_bytes(input_tar, member)
                if _is_frame(member.name):
                    if member.name not in expected_keys:
                        raise MigrationError(f"Image member has no rgb_key row: {member.name}")
                    if member.name in seen_keys:
                        raise MigrationError(f"Duplicate image key: {member.name}")
                    seen_keys.add(member.name)
                    original = decode_rgb(payload)
                    corrected, changed = canonicalize_v1_rgb(original)
                    if changed != 17:
                        raise MigrationError(f"Source image was already corrected: {member.name}")
                    payload = encode_lossless_webp(corrected, method=webp_method)
                    changed_total += changed
                    recent_images += 1
                    if progress_every and len(seen_keys) % progress_every == 0:
                        now = time.monotonic()
                        image_elapsed = now - image_started
                        recent_elapsed = now - recent_started
                        average_rate = len(seen_keys) / image_elapsed if image_elapsed else 0.0
                        recent_rate = recent_images / recent_elapsed if recent_elapsed else 0.0
                        remaining = frames.num_rows - len(seen_keys)
                        print(
                            "SHARD_PROGRESS "
                            f"source={source} images={len(seen_keys):,}/{frames.num_rows:,} "
                            f"remaining={remaining:,} average_rate={average_rate:.1f} "
                            f"recent_rate={recent_rate:.1f} img/s "
                            f"elapsed={image_elapsed:.1f}s "
                            f"eta={(remaining / average_rate if average_rate else 0):.1f}s "
                            f"changed_pixels={changed_total:,} partial={partial}",
                            flush=True,
                        )
                        recent_started = now
                        recent_images = 0
                elif member.name in replacement:
                    payload = replacement[member.name]
                output_tar.addfile(_copy_info(member, len(payload)), io.BytesIO(payload))
        if seen_keys != expected_keys:
            missing_keys = sorted(expected_keys - seen_keys)
            raise MigrationError(f"rgb_key rows without image members: {missing_keys[:5]}")
        if len(seen_keys) != frames.num_rows:
            raise MigrationError("Image count does not match frame row count")
        corrected_hash = _sha256(partial)
        report = ShardReport(
            migration_version=MIGRATION_VERSION,
            source_shard=str(source),
            source_sha256=source_hash,
            corrected_shard=str(destination),
            corrected_sha256=corrected_hash,
            image_count=len(seen_keys),
            frame_rows=frames.num_rows,
            transition_rows=transitions.num_rows,
            episode_rows=episodes.num_rows,
            changed_pixels_per_image=17,
            changed_pixels_total=changed_total,
            webp_method=webp_method,
            v2_unchanged=True,
            source_bytes=source.stat().st_size,
            corrected_bytes=partial.stat().st_size,
            elapsed_seconds=time.monotonic() - started,
            frames_schema=str(frames.schema),
        )
        os.replace(partial, destination)
        report_path.write_bytes(canonical_json(asdict(report)))
        return report
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _member_metadata(member: tarfile.TarInfo) -> tuple[Any, ...]:
    return (
        member.name,
        member.mode,
        member.uid,
        member.gid,
        member.mtime,
        member.type,
        member.linkname,
        member.uname,
        member.gname,
        member.devmajor,
        member.devminor,
    )


def validate_migrated_v1_shard(
    source: str | Path,
    corrected: str | Path,
) -> ShardValidationReport:
    """Exhaustively compare one corrected shard with its legacy V1 source."""
    started = time.monotonic()
    source = Path(source).resolve()
    corrected = Path(corrected).resolve()
    changed_pixels_total = 0
    image_count = 0
    unchanged_member_count = 0
    with tarfile.open(source, "r:*") as source_tar, tarfile.open(corrected, "r:*") as corrected_tar:
        source_members = source_tar.getmembers()
        corrected_members = corrected_tar.getmembers()
        source_names = [member.name for member in source_members]
        corrected_names = [member.name for member in corrected_members]
        if source_names != corrected_names:
            raise MigrationError("Corrected tar member names/order differ from source")
        source_frames: pa.Table | None = None
        corrected_frames: pa.Table | None = None
        transitions_rows = 0
        episodes_rows = 0
        for source_member, corrected_member in zip(
            source_members, corrected_members, strict=True
        ):
            if _member_metadata(source_member) != _member_metadata(corrected_member):
                raise MigrationError(
                    f"Tar metadata changed unexpectedly for {source_member.name}"
                )
            if not source_member.isfile():
                continue
            source_payload = _read_member_bytes(source_tar, source_member)
            corrected_payload = _read_member_bytes(corrected_tar, corrected_member)
            if _is_frame(source_member.name):
                legacy_rgb = decode_rgb(source_payload)
                corrected_rgb = decode_rgb(corrected_payload)
                if inspect_stencil(legacy_rgb) != "legacy":
                    raise MigrationError(f"Source stencil is not legacy: {source_member.name}")
                if inspect_stencil(corrected_rgb) != "corrected":
                    raise MigrationError(
                        f"Corrected stencil is not Ready green: {source_member.name}"
                    )
                difference = np.any(legacy_rgb != corrected_rgb, axis=2)
                if int(difference.sum()) != 17:
                    raise MigrationError(
                        f"Expected 17 changed pixels in {source_member.name}, "
                        f"found {int(difference.sum())}"
                    )
                stencil = set(crosshair_stencil(384, 384))
                changed_coordinates = {
                    (int(x), int(y)) for y, x in np.argwhere(difference)
                }
                if changed_coordinates != stencil:
                    raise MigrationError(
                        f"Changed coordinates differ from stencil: {source_member.name}"
                    )
                changed_pixels_total += 17
                image_count += 1
            elif source_member.name == "frames.parquet":
                source_frames = pq.read_table(io.BytesIO(source_payload))
                corrected_frames = pq.read_table(io.BytesIO(corrected_payload))
                expected = canonicalize_frames_table(source_frames, require_legacy=True)
                if not expected.equals(corrected_frames, check_metadata=True):
                    raise MigrationError("Corrected frames.parquet differs from expected table")
            elif source_member.name == "manifest.json":
                manifest = json.loads(corrected_payload)
                canonicalization = manifest.get("crosshair_canonicalization", {})
                if canonicalization.get("migration_version") != MIGRATION_VERSION:
                    raise MigrationError("Corrected manifest lacks migration provenance")
            else:
                if source_payload != corrected_payload:
                    raise MigrationError(
                        f"Unrelated member bytes changed: {source_member.name}"
                    )
                unchanged_member_count += 1
                if source_member.name == "transitions.parquet":
                    transitions_rows = pq.read_metadata(io.BytesIO(source_payload)).num_rows
                elif source_member.name == "episodes.parquet":
                    episodes_rows = pq.read_metadata(io.BytesIO(source_payload)).num_rows
        if source_frames is None or corrected_frames is None:
            raise MigrationError("Missing frames.parquet during validation")
        if image_count != source_frames.num_rows:
            raise MigrationError("Validated image count differs from frames row count")

    return ShardValidationReport(
        migration_version=MIGRATION_VERSION,
        source_shard=str(source),
        corrected_shard=str(corrected),
        source_sha256=_sha256(source),
        corrected_sha256=_sha256(corrected),
        image_count=image_count,
        frame_rows=source_frames.num_rows,
        transition_rows=transitions_rows,
        episode_rows=episodes_rows,
        changed_pixels_total=changed_pixels_total,
        unchanged_member_count=unchanged_member_count,
        elapsed_seconds=time.monotonic() - started,
    )


def validate_canonical_v1_shard(corrected: str | Path) -> CanonicalShardReport:
    """Validate a corrected shard is canonical and an idempotent repair makes zero changes."""
    started = time.monotonic()
    corrected = Path(corrected).resolve()
    image_count = 0
    with tarfile.open(corrected, "r:*") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        if duplicates:
            raise MigrationError(f"Corrected shard has duplicate members: {duplicates[:5]}")
        frames_payload = _read_member_bytes(archive, archive.getmember("frames.parquet"))
        frames = pq.read_table(io.BytesIO(frames_payload))
        recanonicalized = canonicalize_frames_table(frames)
        if not recanonicalized.equals(frames, check_metadata=True):
            raise MigrationError("Corrected frames table is not idempotently canonical")
        expected_keys = frames["rgb_key"].to_pylist()
        image_names = [name for name in names if _is_frame(name)]
        if image_names != expected_keys:
            raise MigrationError("Corrected image member order differs from frame keys")
        for name in image_names:
            rgb = decode_rgb(_read_member_bytes(archive, archive.getmember(name)))
            recanonicalized_rgb, changed = canonicalize_v1_rgb(rgb)
            if changed != 0 or not np.array_equal(recanonicalized_rgb, rgb):
                raise MigrationError(f"Corrected image is not idempotent: {name}")
            image_count += 1
    return CanonicalShardReport(
        migration_version=MIGRATION_VERSION,
        corrected_shard=str(corrected),
        corrected_sha256=_sha256(corrected),
        image_count=image_count,
        frame_rows=frames.num_rows,
        zero_repairs=0,
        elapsed_seconds=time.monotonic() - started,
    )


def write_checksum_file(shards: list[Path], destination: Path) -> None:
    lines = [f"{_md5(shard)}  {shard.name}\n" for shard in sorted(shards)]
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text("".join(lines), encoding="utf-8")
    os.replace(temporary, destination)
