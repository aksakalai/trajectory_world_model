from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_autoencoder_selection_pipeline.py"
SPEC = importlib.util.spec_from_file_location("autoencoder_selection_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result(candidate: str, latent: int, score: float, *, eligible: bool = True) -> dict[str, object]:
    return {
        "candidate": candidate,
        "latent_values": latent,
        "selection_score": score,
        "eligible": eligible,
    }


def test_pareto_candidates_preserve_smaller_useful_latent() -> None:
    rows = [
        result("G", 27_648, 0.314),
        result("B", 46_080, 0.291),
        result("H", 46_080, 0.292),
        result("F", 110_592, 0.303),
    ]

    assert MODULE.pareto_candidates(rows) == ["G", "B"]


def test_efficiency_winner_uses_smallest_latent_within_one_percent() -> None:
    rows = [
        result("F", 110_592, 0.2055),
        result("B", 46_080, 0.2060),
        result("I", 55_296, 0.2090),
    ]

    winner, decision = MODULE.efficiency_winner(rows, 0.01)

    assert winner == "B"
    assert decision["quality_equivalent_candidates"] == ["B", "F"]


def test_efficiency_winner_spends_capacity_for_material_gain() -> None:
    rows = [
        result("B", 46_080, 0.2100),
        result("F", 110_592, 0.2050),
    ]

    winner, _ = MODULE.efficiency_winner(rows, 0.01)

    assert winner == "F"
