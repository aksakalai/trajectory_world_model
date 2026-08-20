from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_episode_split_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_episode_split_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_grouped_assignment_is_deterministic_and_never_splits_shared_episode() -> None:
    records = {
        "v1": [
            {"episode_id": f"e{index}", "observation_count": 20 + index}
            for index in range(40)
        ],
        "v2": [
            {"episode_id": f"e{index}", "observation_count": 30 + 2 * index}
            for index in range(35)
        ],
    }
    first = MODULE.assign_grouped_episodes(records, refinement_iterations=20_000)
    second = MODULE.assign_grouped_episodes(records, refinement_iterations=20_000)

    assert first == second
    assert set(first) == {f"e{index}" for index in range(40)}
    assert {name: list(first.values()).count(name) for name in MODULE.FRACTIONS} == (
        MODULE._episode_quotas(40)
    )
    for episode in records["v2"]:
        assert first[episode["episode_id"]] in MODULE.FRACTIONS
