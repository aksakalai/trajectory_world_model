#!/usr/bin/env python3
"""Frozen V2 mission catalog and deterministic certified-region sampling.

The catalog describes named geometry and player-controlled initial conditions.
It deliberately contains no restitution, damping, bounce-count, settling, or
replacement parameters.  Final launch certification is performed in Unreal by
the same ``FGrenadeSim`` path used by preview and realized grenades.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable


CATALOG_VERSION = "trajectory-throw-v2-progressive-certified-regions-10"
CANONICAL_PHYSICS_ID = "grenade-sim-config-r1+launch-1400cmps+cooldown-2s"
TYPE_FRAME_SHARE = Fraction(3, 620)
FAMILY_FRAME_SHARES = {
    "broad_object_surface": Fraction(39, 620),
    "object_edge_apex": Fraction(39, 620),
    "wall_corner_rebound": Fraction(36, 620),
    "hoop": Fraction(30, 620),
    "ramp": Fraction(24, 620),
    "out_of_bounds": Fraction(12, 620),
    "trajectory_control": Fraction(6, 620),
}


@dataclass(frozen=True)
class MissionType:
    slug: str
    family: str
    event_kind: str
    target_actor: str
    target_region: str
    center: tuple[float, float, float]
    approach: tuple[float, float]
    duration_seconds: Fraction
    region_radius_cm: float = 34.0
    variation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    variation_extent_cm: float = 30.0
    secondary_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    secondary_extent_cm: float = 18.0
    expected_contact_order: tuple[str, ...] = ()
    boundary: str = ""
    direction: str = ""

    def as_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        for key, item in list(value.items()):
            if isinstance(item, tuple):
                value[key] = list(item)
            elif isinstance(item, Fraction):
                value[key] = float(item)
        value["type_frame_share"] = float(TYPE_FRAME_SHARE)
        return value


def _m(
    slug: str,
    family: str,
    event: str,
    actor: str,
    region: str,
    center: tuple[float, float, float],
    approach: tuple[float, float],
    duration: str,
    **kwargs: Any,
) -> MissionType:
    return MissionType(
        slug, family, event, actor, region, center, approach,
        Fraction(duration), **kwargs,
    )


def mission_types() -> tuple[MissionType, ...]:
    b = "broad_object_surface"
    e = "object_edge_apex"
    w = "wall_corner_rebound"
    h = "hoop"
    r = "ramp"
    o = "out_of_bounds"
    c = "trajectory_control"
    rect = "CurriculumObject_Rectangle"
    pyramid = "CurriculumObject_Pyramid"
    sphere = "CurriculumObject_Sphere"
    hoop = "CurriculumObject_Hoop"
    ramp = "CurriculumObject_Ramp"
    result = (
        # Broad rectangle faces and top, safely inset from every edge.
        _m("rectangle_north_face", b, "contact_region", rect, "north_face", (-615, 700, 135), (1, 0), "6", variation_axis=(0, 1, 0), variation_extent_cm=58, secondary_axis=(0, 0, 1), secondary_extent_cm=55),
        _m("rectangle_south_face", b, "contact_region", rect, "south_face", (-785, 700, 135), (-1, 0), "6", variation_axis=(0, 1, 0), variation_extent_cm=58, secondary_axis=(0, 0, 1), secondary_extent_cm=55),
        _m("rectangle_east_face", b, "contact_region", rect, "east_face", (-700, 840, 135), (0, 1), "6", variation_axis=(1, 0, 0), variation_extent_cm=32, secondary_axis=(0, 0, 1), secondary_extent_cm=55),
        _m("rectangle_west_face", b, "contact_region", rect, "west_face", (-700, 560, 135), (0, -1), "6", variation_axis=(1, 0, 0), variation_extent_cm=32, secondary_axis=(0, 0, 1), secondary_extent_cm=55),
        _m("rectangle_top_surface", b, "contact_region", rect, "top_inset", (-700, 700, 250), (-1, 0), "6", variation_axis=(1, 0, 0), variation_extent_cm=30, secondary_axis=(0, 1, 0), secondary_extent_cm=55),
        # Pyramid is 280x280x250; points lie in the interior of each slope.
        _m("pyramid_north_face", b, "contact_region", pyramid, "north_slope", (770, 700, 125), (1, 0), "6", variation_axis=(0, 1, 0), variation_extent_cm=38, secondary_axis=(-0.49, 0, 0.87), secondary_extent_cm=25),
        _m("pyramid_south_face", b, "contact_region", pyramid, "south_slope", (630, 700, 125), (-1, 0), "6", variation_axis=(0, 1, 0), variation_extent_cm=38, secondary_axis=(0.49, 0, 0.87), secondary_extent_cm=25),
        _m("pyramid_east_face", b, "contact_region", pyramid, "east_slope", (700, 770, 125), (0, 1), "6", variation_axis=(1, 0, 0), variation_extent_cm=38, secondary_axis=(0, -0.49, 0.87), secondary_extent_cm=25),
        _m("pyramid_west_face", b, "contact_region", pyramid, "west_slope", (700, 630, 125), (0, -1), "6", variation_axis=(1, 0, 0), variation_extent_cm=38, secondary_axis=(0, 0.49, 0.87), secondary_extent_cm=25),
        _m("sphere_north_quadrant", b, "contact_region", sphere, "north_quadrant", (-580, -700, 125), (1, 0), "6", variation_axis=(0, 1, 0), variation_extent_cm=28, secondary_axis=(0, 0, 1), secondary_extent_cm=30),
        _m("sphere_south_quadrant", b, "contact_region", sphere, "south_quadrant", (-820, -700, 125), (-1, 0), "6", variation_axis=(0, 1, 0), variation_extent_cm=28, secondary_axis=(0, 0, 1), secondary_extent_cm=30),
        _m("sphere_east_quadrant", b, "contact_region", sphere, "east_quadrant", (-700, -580, 125), (0, 1), "6", variation_axis=(1, 0, 0), variation_extent_cm=28, secondary_axis=(0, 0, 1), secondary_extent_cm=30),
        _m("sphere_west_quadrant", b, "contact_region", sphere, "west_quadrant", (-700, -820, 125), (0, -1), "6", variation_axis=(1, 0, 0), variation_extent_cm=28, secondary_axis=(0, 0, 1), secondary_extent_cm=30),
        # Narrow but physical grenade-radius-aware edge corridors.
        _m("rectangle_ne_vertical_edge", e, "edge_contact", rect, "north_east_vertical", (-615, 840, 135), (1, 1), "13/2", variation_axis=(0, 0, 1), variation_extent_cm=55, region_radius_cm=22),
        # A shallow diagonal retains true two-face edge incidence while keeping
        # every budget endpoint outside the central ramp corridor.
        _m("rectangle_nw_vertical_edge", e, "edge_contact", rect, "north_west_vertical", (-615, 560, 135), (1, -.3), "13/2", variation_axis=(0, 0, 1), variation_extent_cm=55, secondary_extent_cm=0, region_radius_cm=22),
        _m("rectangle_se_vertical_edge", e, "edge_contact", rect, "south_east_vertical", (-785, 840, 135), (-1, 1), "13/2", variation_axis=(0, 0, 1), variation_extent_cm=55, region_radius_cm=22),
        _m("rectangle_sw_vertical_edge", e, "edge_contact", rect, "south_west_vertical", (-785, 560, 135), (-1, -1), "13/2", variation_axis=(0, 0, 1), variation_extent_cm=55, region_radius_cm=22),
        _m("rectangle_north_upper_edge", e, "edge_contact", rect, "north_upper", (-615, 700, 250), (1, 0), "13/2", variation_axis=(0, 1, 0), variation_extent_cm=62, region_radius_cm=22),
        _m("rectangle_south_upper_edge", e, "edge_contact", rect, "south_upper", (-785, 700, 250), (-1, 0), "13/2", variation_axis=(0, 1, 0), variation_extent_cm=62, region_radius_cm=22),
        _m("rectangle_east_upper_edge", e, "edge_contact", rect, "east_upper", (-700, 840, 250), (0, 1), "13/2", variation_axis=(1, 0, 0), variation_extent_cm=34, region_radius_cm=22),
        _m("rectangle_west_upper_edge", e, "edge_contact", rect, "west_upper", (-700, 560, 250), (0, -1), "13/2", variation_axis=(1, 0, 0), variation_extent_cm=34, region_radius_cm=22),
        _m("pyramid_ne_ridge", e, "edge_contact", pyramid, "north_east_ridge", (750, 750, 160), (1, 1), "13/2", variation_axis=(-0.35, -0.35, 0.87), variation_extent_cm=48, region_radius_cm=24),
        _m("pyramid_nw_ridge", e, "edge_contact", pyramid, "north_west_ridge", (750, 650, 160), (1, -1), "13/2", variation_axis=(-0.35, 0.35, 0.87), variation_extent_cm=48, region_radius_cm=24),
        _m("pyramid_se_ridge", e, "edge_contact", pyramid, "south_east_ridge", (650, 750, 160), (-1, 1), "13/2", variation_axis=(0.35, -0.35, 0.87), variation_extent_cm=48, region_radius_cm=24),
        _m("pyramid_sw_ridge", e, "edge_contact", pyramid, "south_west_ridge", (650, 650, 160), (-1, -1), "13/2", variation_axis=(0.35, 0.35, 0.87), variation_extent_cm=48, region_radius_cm=24),
        _m("pyramid_apex", e, "apex_contact", pyramid, "apex_corridor", (700, 700, 250), (-1, 0), "13/2", region_radius_cm=28, variation_axis=(1, 0, 0), variation_extent_cm=12, secondary_axis=(0, 1, 0), secondary_extent_cm=12),
        # Four direct, four oblique, and four two-wall corner interactions.
        *tuple(_m(f"{side}_wall_direct", w, "wall_contact", f"CurriculumWall_{side.title()}", f"{side}_direct", point, approach, "7", variation_axis=tangent, variation_extent_cm=380, secondary_axis=(0, 0, 1), secondary_extent_cm=90, boundary=side) for side, point, approach, tangent in (
            ("north", (1600, 0, 190), (-1, 0), (0, 1, 0)), ("south", (-1600, 0, 190), (1, 0), (0, 1, 0)),
            ("east", (0, 1600, 190), (0, -1), (1, 0, 0)), ("west", (0, -1600, 190), (0, 1), (1, 0, 0)),)),
        *tuple(_m(f"{side}_wall_oblique", w, "wall_contact", f"CurriculumWall_{side.title()}", f"{side}_oblique", point, approach, "7", variation_axis=tangent, variation_extent_cm=330, secondary_axis=(0, 0, 1), secondary_extent_cm=90, boundary=side) for side, point, approach, tangent in (
            ("north", (1600, -240, 190), (-1, .42), (0, 1, 0)), ("south", (-1600, 240, 190), (1, -.42), (0, 1, 0)),
            ("east", (240, 1600, 190), (-.42, -1), (1, 0, 0)), ("west", (-240, -1600, 190), (.42, 1), (1, 0, 0)),)),
        *tuple(_m(f"{corner}_corner_two_wall", w, "two_wall_contact", "CurriculumArena", f"{corner}_corner", point, approach, "15/2", region_radius_cm=70, variation_axis=axis, variation_extent_cm=35, secondary_axis=(0, 0, 1), secondary_extent_cm=35, expected_contact_order=order, boundary=corner) for corner, point, approach, axis, order in (
            ("north_east", (1592, 1592, 175), (-1, -1), (1, -1, 0), ("CurriculumWall_North", "CurriculumWall_East")),
            ("north_west", (1592, -1592, 175), (-1, 1), (1, 1, 0), ("CurriculumWall_North", "CurriculumWall_West")),
            ("south_east", (-1592, 1592, 175), (1, -1), (-1, -1, 0), ("CurriculumWall_South", "CurriculumWall_East")),
            ("south_west", (-1592, -1592, 175), (1, 1), (-1, 1, 0), ("CurriculumWall_South", "CurriculumWall_West")),)),
        # Hoop plane is YZ and throws travel along X.
        _m("hoop_clean_negx_to_posx", h, "hoop_passage", hoop, "safe_opening", (700, -700, 145), (-1, 0), "7", direction="negative_x_to_positive_x", variation_axis=(0, 1, 0), variation_extent_cm=32, secondary_axis=(0, 0, 1), secondary_extent_cm=32),
        _m("hoop_clean_posx_to_negx", h, "hoop_passage", hoop, "safe_opening", (700, -700, 145), (1, 0), "7", direction="positive_x_to_negative_x", variation_axis=(0, 1, 0), variation_extent_cm=32, secondary_axis=(0, 0, 1), secondary_extent_cm=32),
        *tuple(_m(f"hoop_{rim}_{short}", h, "rim_contact", hoop, f"{rim}_rim", point, approach, "7", direction=direction, region_radius_cm=49, variation_axis=axis, variation_extent_cm=24, secondary_extent_cm=0) for direction, short, approach in (("negative_x_to_positive_x", "negx_to_posx", (-1, 0)), ("positive_x_to_negative_x", "posx_to_negx", (1, 0))) for rim, point, axis in (("upper", (700, -700, 250), (0, 1, 0)), ("lower", (700, -700, 40), (0, 1, 0)), ("left", (700, -805, 145), (0, 0, 1)), ("right", (700, -595, 145), (0, 0, 1)))),
        # Ramp surface, crossover, and physical edge/lip corridors.
        _m("ramp_uphill_surface", r, "ramp_contact", ramp, "uphill_surface", (80, 0, 82), (1, 0), "7", variation_axis=(0, 1, 0), variation_extent_cm=62, secondary_axis=(1, 0, -.325), secondary_extent_cm=55),
        _m("ramp_downhill_surface", r, "ramp_contact", ramp, "downhill_surface", (160, 0, 54), (-1, 0), "7", region_radius_cm=52, variation_axis=(0, 1, 0), variation_extent_cm=62, secondary_axis=(1, 0, -.325), secondary_extent_cm=45),
        _m("ramp_crossover_left_to_right", r, "ramp_crossover", ramp, "body_crossing", (0, 0, 150), (0, -1), "15/2", variation_axis=(1, 0, 0), variation_extent_cm=100, direction="left_to_right"),
        _m("ramp_crossover_right_to_left", r, "ramp_crossover", ramp, "body_crossing", (0, 0, 150), (0, 1), "15/2", variation_axis=(1, 0, 0), variation_extent_cm=100, direction="right_to_left"),
        _m("ramp_left_side_edge", r, "edge_contact", ramp, "left_side_edge", (0, -130, 108), (0, -1), "7", variation_axis=(1, 0, -.325), variation_extent_cm=95, region_radius_cm=24),
        _m("ramp_right_side_edge", r, "edge_contact", ramp, "right_side_edge", (0, 130, 108), (0, 1), "7", variation_axis=(1, 0, -.325), variation_extent_cm=95, region_radius_cm=24),
        _m("ramp_high_end_lip", r, "edge_contact", ramp, "high_end_lip", (-250, 0, 190), (-1, 0), "7", variation_axis=(0, 1, 0), variation_extent_cm=70, region_radius_cm=25),
        _m("ramp_low_end_lip", r, "edge_contact", ramp, "low_end_lip", (250, 0, 28), (1, 0), "7", variation_axis=(0, 1, 0), variation_extent_cm=70, region_radius_cm=25),
        # Boundary exits are aimed above the wall, retaining the named wall in view.
        *tuple(_m(f"{side}_boundary_exit", o, "arena_exit", f"CurriculumWall_{side.title()}", f"{side}_boundary", point, approach, "8", variation_axis=tangent, variation_extent_cm=300, secondary_axis=(0, 0, 1), secondary_extent_cm=90, boundary=side) for side, point, approach, tangent in (
            ("north", (1900, 0, 650), (-1, 0), (0, 1, 0)), ("south", (-1900, 0, 650), (1, 0), (0, 1, 0)),
            ("east", (0, 1900, 650), (0, -1), (1, 0, 0)), ("west", (0, -1900, 650), (0, 1), (1, 0, 0)),)),
        # Explicit V2 trajectory-control demonstrations.
        _m("trajectory_manual_toggle_cycle", c, "trajectory_manual_toggle", "CurriculumArena", "safe_control_view", (0, -300, 110), (1, 0), "6", variation_axis=(0, 1, 0), variation_extent_cm=260, secondary_axis=(1, 0, 0), secondary_extent_cm=180),
        _m("trajectory_reload_reopen_cycle", c, "trajectory_reload_reopen", "CurriculumArena", "safe_control_view", (0, 300, 110), (-1, 0), "8", variation_axis=(0, 1, 0), variation_extent_cm=260, secondary_axis=(1, 0, 0), secondary_extent_cm=180),
    )
    validate_catalog(result)
    return result


def validate_catalog(types: Iterable[MissionType]) -> None:
    values = tuple(types)
    if len(values) != 62:
        raise ValueError(f"V2 catalog must contain 62 mission types, got {len(values)}")
    slugs = [item.slug for item in values]
    if len(slugs) != len(set(slugs)):
        raise ValueError("V2 mission type slugs must be unique")
    counts = Counter(item.family for item in values)
    expected = {family: int(share / TYPE_FRAME_SHARE) for family, share in FAMILY_FRAME_SHARES.items()}
    if counts != Counter(expected):
        raise ValueError(f"V2 mission family counts changed: {dict(counts)}")
    if sum(FAMILY_FRAME_SHARES.values(), Fraction()) != Fraction(3, 10):
        raise ValueError("V2 mission family shares must total 30%")
    if any(item.duration_seconds <= 0 for item in values):
        raise ValueError("mission durations must be positive")


def radical_inverse(index: int, base: int) -> float:
    """Return a deterministic low-discrepancy coordinate in [0, 1)."""
    value = 0.0
    denominator = 1.0
    while index:
        index, digit = divmod(index, base)
        denominator *= base
        value += digit / denominator
    return value


def _normalized_xy(value: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*value)
    if length <= 1e-9:
        raise ValueError("mission approach vector must be non-zero")
    return value[0] / length, value[1] / length


def _coverage_bin(value: float, count: int) -> int:
    """Quantize a progressive coordinate for auditable within-type coverage."""
    return min(count - 1, max(0, math.floor(value * count)))


def _ramp_top_surface_z(world_x: float) -> float:
    """Mirror the fixed authored ramp's top plane for supported player spawns."""
    pitch = math.radians(-18.0)
    support = abs(math.sin(pitch)) * 500.0 * 0.5 + abs(math.cos(pitch)) * 36.0 * 0.5
    return support + math.tan(pitch) * world_x + (36.0 * 0.5) / math.cos(pitch)


def _wall_normal(boundary: str) -> tuple[float, float]:
    return {
        "north": (-1.0, 0.0),
        "south": (1.0, 0.0),
        "east": (0.0, -1.0),
        "west": (0.0, 1.0),
    }[boundary]


def _project_sphere_target(
    item: MissionType,
    u: float,
    v: float,
) -> list[float]:
    """Keep sphere samples on the physical 120 cm sphere, not inside it."""
    sphere_center = (-700.0, -700.0, 120.0)
    radius = 120.0
    ax, ay = _normalized_xy(item.approach)
    tx, ty = -ay, ax
    tangent = (2.0 * u - 1.0) * item.variation_extent_cm
    vertical = 5.0 + (2.0 * v - 1.0) * item.secondary_extent_cm
    horizontal_radius = math.sqrt(max(0.0, radius * radius - vertical * vertical))
    direction_x = ax * radius + tx * tangent
    direction_y = ay * radius + ty * tangent
    direction_length = math.hypot(direction_x, direction_y)
    return [
        sphere_center[0] + direction_x / direction_length * horizontal_radius,
        sphere_center[1] + direction_y / direction_length * horizontal_radius,
        sphere_center[2] + vertical,
    ]


def _project_hoop_rim_target(item: MissionType, u: float) -> list[float]:
    """Keep rim samples on the authored ring centerline."""
    ring_radius = 120.0
    offset = (2.0 * u - 1.0) * item.variation_extent_cm
    radial = math.sqrt(max(0.0, ring_radius * ring_radius - offset * offset))
    region = item.target_region
    if region == "upper_rim":
        return [700.0, -700.0 + offset, 145.0 + radial]
    if region == "lower_rim":
        return [700.0, -700.0 + offset, 145.0 - radial]
    if region == "left_rim":
        return [700.0, -700.0 - radial, 145.0 + offset]
    if region == "right_rim":
        return [700.0, -700.0 + radial, 145.0 + offset]
    raise ValueError(f"unknown hoop rim region {region!r}")


def _project_pyramid_apex_target(u: float, v: float) -> list[float]:
    """Project apex-corridor variation onto the authored pyramid surface.

    The pyramid has a 280 cm square base and is 250 cm high. Moving laterally
    away from its apex while retaining z=250 places the requested target in
    empty air, so endpoint-inclusive budget sampling must lower z onto the
    corresponding triangular face.
    """
    offset_x = (2.0 * u - 1.0) * 12.0
    offset_y = (2.0 * v - 1.0) * 12.0
    height = 250.0 - (250.0 / 140.0) * max(abs(offset_x), abs(offset_y))
    return [700.0 + offset_x, 700.0 + offset_y, height]


def _budget_coverage_units(sample_index: int, sample_count: int) -> tuple[float, ...]:
    """Evenly cover every parameter range for the exact allocated count.

    Each dimension is an endpoint-inclusive Latin permutation.  Thus two
    samples use opposing extremes, three use both extremes plus the midpoint,
    and larger budgets divide every range into progressively finer slices.
    """
    if sample_count < 1 or not 0 <= sample_index < sample_count:
        raise ValueError("invalid budget coverage sample")
    if sample_count == 1:
        return (0.5,) * 5
    coprime = [
        value for value in range(1, sample_count)
        if math.gcd(value, sample_count) == 1
    ]
    units: list[float] = []
    for dimension in range(5):
        if dimension == 0:
            stride, offset = 1, 0
        elif dimension == 1:
            stride, offset = sample_count - 1, sample_count - 1
        else:
            stride = coprime[(dimension * 3) % len(coprime)]
            offset = (dimension * max(1, sample_count // 3)) % sample_count
        rank = (sample_index * stride + offset) % sample_count
        units.append(rank / (sample_count - 1))
    return tuple(units)


def build_solution(
    item: MissionType,
    repetition: int,
    observation_rate: int,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Sample one immutable point in a known-working certified region.

    Repetition zero is central. Later points use separated Halton coordinates;
    the seed affects identity, never mission validity.
    """
    if repetition < 0 or observation_rate <= 0:
        raise ValueError("invalid mission repetition or observation rate")
    if sample_count is not None:
        u, v, distance_u, arc_u, oblique_unit = _budget_coverage_units(
            repetition, sample_count
        )
    elif repetition == 0:
        u = v = distance_u = arc_u = 0.5
        oblique_unit = 0.5
    else:
        ordinal = repetition
        u, v = radical_inverse(ordinal, 2), radical_inverse(ordinal, 3)
        distance_u, arc_u = radical_inverse(ordinal, 5), radical_inverse(ordinal, 7)
        oblique_unit = radical_inverse(repetition, 11)
    center = list(item.center)
    for coordinate in range(3):
        center[coordinate] += item.variation_axis[coordinate] * (2 * u - 1) * item.variation_extent_cm
        center[coordinate] += item.secondary_axis[coordinate] * (2 * v - 1) * item.secondary_extent_cm
    if item.target_actor == "CurriculumObject_Sphere":
        center = _project_sphere_target(item, u, v)
    elif item.event_kind == "rim_contact":
        center = _project_hoop_rim_target(item, u)
    elif item.slug == "pyramid_apex":
        center = _project_pyramid_apex_target(u, v)

    expected_contact_order = list(item.expected_contact_order)
    contact_order_variant: str | None = None
    if item.event_kind == "two_wall_contact":
        # The first recipe uses the catalog order. Later repetitions alternate
        # the wall approached first, with a physical corner bias matching it.
        reverse_order = repetition > 0 and repetition % 2 == 1
        contact_order_variant = "reverse" if reverse_order else "catalog"
        if reverse_order:
            expected_contact_order.reverse()
        bias = 22.0 if repetition == 0 else 14.0 + 18.0 * distance_u
        sign = -1.0 if reverse_order else 1.0
        center = [
            item.center[index]
            + item.variation_axis[index] * sign * bias
            + item.secondary_axis[index] * (2 * v - 1) * item.secondary_extent_cm
            for index in range(3)
        ]
    ax, ay = _normalized_xy(item.approach)
    distance = 430.0 + 260.0 * distance_u
    # Approach points outward from geometry toward the player.
    spawn = [center[0] + ax * distance, center[1] + ay * distance, 100.0]
    # Parameter-specific progressive lateral variation. Wall modes get explicit
    # incidence bands so direct and oblique recipes cannot collapse together.
    oblique = (2 * oblique_unit - 1) * 115.0
    approach_side = (
        "center" if abs(oblique) < 1e-6
        else "left" if oblique < 0 else "right"
    )
    if item.event_kind == "wall_contact":
        nx, ny = _wall_normal(item.boundary)
        tx, ty = -ny, nx
        if item.slug.endswith("_oblique"):
            side = -1.0 if repetition % 2 else 1.0
            lateral_ratio = 0.32 + 0.16 * arc_u
            spawn = [
                center[0] + nx * distance + tx * side * distance * lateral_ratio,
                center[1] + ny * distance + ty * side * distance * lateral_ratio,
                100.0,
            ]
            oblique = side * distance * lateral_ratio
            approach_side = "left" if side < 0 else "right"
        else:
            oblique = (2 * oblique_unit - 1) * 70.0
            spawn = [
                center[0] + nx * distance + tx * oblique,
                center[1] + ny * distance + ty * oblique,
                100.0,
            ]
    else:
        spawn[0] += -ay * oblique
        spawn[1] += ax * oblique
    if item.event_kind == "arena_exit":
        distance = 1050.0 + 180.0 * distance_u
        spawn = [center[0] + ax * distance, center[1] + ay * distance, 100.0]
    if item.event_kind == "hoop_passage" or item.event_kind == "rim_contact":
        side = -1.0 if item.direction == "negative_x_to_positive_x" else 1.0
        spawn = [center[0] + side * (560.0 + 100.0 * distance_u), center[1], 100.0]
        spawn[1] += oblique * 0.35
    if item.event_kind == "ramp_crossover":
        side = -1.0 if item.direction == "left_to_right" else 1.0
        spawn = [center[0], center[1] + side * (520.0 + 80.0 * distance_u), 100.0]
    if item.slug == "rectangle_top_surface":
        # A ground-level low branch necessarily meets the rectangle's vertical
        # side before its top. Use the authored ramp as a legitimate elevated,
        # supported launch position so the low branch descends onto the top.
        support_x = -220.0 + 24.0 * (2.0 * distance_u - 1.0)
        support_y = 55.0 * (2.0 * oblique_unit - 1.0)
        spawn = [support_x, support_y, _ramp_top_surface_z(support_x) + 98.0]
        approach_side = "left" if support_y < 0 else "right" if support_y > 0 else "center"
    elif item.slug == "ramp_downhill_surface":
        # Stand on the walkable high half of the ramp and throw down its slope.
        # Unlike the old rectangle-top construction, this pose is naturally
        # reachable by walking up the ramp.
        support_x = -220.0 + 24.0 * (2.0 * distance_u - 1.0)
        support_y = 45.0 * (2.0 * oblique_unit - 1.0)
        spawn = [support_x, support_y, _ramp_top_surface_z(support_x) + 98.0]
        approach_side = "left" if support_y < 0 else "right" if support_y > 0 else "center"
    duration_frames = math.ceil(float(item.duration_seconds) * observation_rate) + 1
    establish = round((0.95 + 0.4 * radical_inverse(repetition + 1, 13)) * observation_rate)
    # Named missions begin at their solved launch rotation. Camera acquisition
    # and turning behavior belong to the semi-Markov portion of V2.
    adjust = 0
    preview = round((0.75 + 0.3 * arc_u) * observation_rate)
    throw_source = establish + adjust + preview
    return {
        "catalog_version": CATALOG_VERSION,
        "canonical_physics_id": CANONICAL_PHYSICS_ID,
        "mission_type": item.slug,
        "family": item.family,
        "event_kind": item.event_kind,
        "target_actor": item.target_actor,
        "target_region": item.target_region,
        "target_point": {"x": center[0], "y": center[1], "z": center[2]},
        "player_spawn": {"x": spawn[0], "y": spawn[1], "z": spawn[2]},
        "region_radius_cm": item.region_radius_cm,
        "expected_contact_order": expected_contact_order,
        "boundary": item.boundary or None,
        "direction": item.direction or None,
        "ballistic_branch": "low",
        "launch_speed_cm_per_second": 1400.0,
        "timing": {
            "establish_steps": establish,
            "camera_adjust_steps": adjust,
            "preview_dwell_steps": preview,
            "throw_source_frame": throw_source,
            "total_observation_frames": duration_frames,
        },
        "variation": {
            "repetition": repetition,
            "budget_sample_count": sample_count,
            "coverage_strategy": (
                "budget_endpoint_latin_v1" if sample_count is not None
                else "legacy_progressive_v1"
            ),
            "surface_u": u,
            "surface_v": v,
            "distance_u": distance_u,
            "arc_u": arc_u,
            "oblique_offset_cm": oblique,
            "approach_side": approach_side,
            "contact_order_variant": contact_order_variant,
            "coverage_cell": {
                "surface_u_bin": _coverage_bin(u, 8),
                "surface_v_bin": _coverage_bin(v, 4),
                "distance_bin": _coverage_bin(distance_u, 4),
                "arc_bin": _coverage_bin(arc_u, 4),
                "approach_side": approach_side,
                "contact_order": contact_order_variant,
            },
        },
        "event_constraints": {
            "maximum_contact_distance_cm": item.region_radius_cm
                + 8.0,
            "hoop_safe_passage_radius_cm": 58.0
                if item.event_kind == "hoop_passage" else None,
            "ramp_crossing_half_width_cm": 250.0
                if item.event_kind == "ramp_crossover" else None,
            "ramp_opposite_landing_offset_cm": 150.0
                if item.event_kind == "ramp_crossover" else None,
        },
        "camera_railguards": {
            "region_visible_all_frames": True,
            "preview_and_region_visible_during_q": True,
            "opening_has_arena_context": True,
        },
    }


def catalog_fingerprint() -> str:
    payload = json.dumps(
        [item.as_dict() for item in mission_types()],
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def review_recipes(observation_rate: int = 20) -> list[dict[str, Any]]:
    """Exactly three separated 384x384 review examples per mission type."""
    values: list[dict[str, Any]] = []
    for item in mission_types():
        for review_variant in range(3):
            repetition = review_variant
            values.append({
                "mission_type": item.slug,
                "family": item.family,
                "review_variant": review_variant,
                "repetition_index": repetition,
                "solution": build_solution(
                    item, repetition, observation_rate, sample_count=3
                ),
                "width": 384,
                "height": 384,
            })
    if len(values) != 186:
        raise AssertionError("V2 review set must contain exactly 186 recipes")
    return values
