#!/usr/bin/env python3
"""Apply the frozen promotion gate to the exact 256K comparison stage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


MIN_RELATIVE_F1 = 0.75
MIN_SUPPORT_F1 = 0.60
MIN_SUPPORT_PRECISION = 0.45
MIN_SUPPORT_RECALL = 0.70
MAX_FINAL_L1 = 0.05
MAX_FOREGROUND_MULTIPLE = 5.0
MAX_ABSOLUTE_FOREGROUND_FRACTION = 0.02


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("decision_path", type=Path)
    parser.add_argument("shared_comparison", type=Path)
    args = parser.parse_args()

    summary = json.loads((args.run_dir / "run-summary.json").read_text())
    comparison = json.loads(args.shared_comparison.read_text())
    if comparison.get("comparison_contract") != "same-probe-same-final-rgb-metric-v1":
        raise RuntimeError("Shared comparison contract is missing or invalid")
    baseline_f1 = float(comparison["old"]["trajectory_f1"])
    new_shared_f1 = float(comparison["new"]["trajectory_f1"])
    records = [json.loads(line) for line in (args.run_dir / "probe-metrics.jsonl").read_text().splitlines()]
    initial, final = records[0], records[-1]
    support = final["trajectory"]
    exact = final["trajectory_exact_core"]
    foreground_limit = max(
        MAX_ABSOLUTE_FOREGROUND_FRACTION,
        MAX_FOREGROUND_MULTIPLE * support["target_foreground_fraction"],
    )
    checks = {
        "exact_image_budget": summary.get("images_seen") == 256_000,
        "completed": summary.get("status") == "diagnostic_complete",
        "support_f1_absolute": support["f1"] >= MIN_SUPPORT_F1,
        "shared_rgb_f1_vs_historical_single_latent": new_shared_f1 >= baseline_f1 * MIN_RELATIVE_F1,
        "support_precision": support["precision"] >= MIN_SUPPORT_PRECISION,
        "support_recall": support["recall"] >= MIN_SUPPORT_RECALL,
        "not_predicting_everything": support["predicted_foreground_fraction"] <= foreground_limit,
        "final_rgb_l1": final["components"]["final_l1"] <= MAX_FINAL_L1,
        "rgb_improved": final["components"]["final_l1"] < initial["components"]["final_l1"],
        "crosshair_exact": final["components"]["final_crosshair_rgb"] <= 1e-8,
        "strict_core_has_precision": exact["precision"] > 0.0,
    }
    payload = {
        "decision": "promote" if all(checks.values()) else "stop",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "historical_single_latent": {
            "shared_comparison": str(args.shared_comparison),
            "model": comparison["old"]["model"],
            "trajectory_f1": baseline_f1,
            "trajectory_precision": comparison["old"]["trajectory_precision"],
            "trajectory_recall": comparison["old"]["trajectory_recall"],
        },
        "new_shared_rgb_metrics": comparison["new"],
        "thresholds": {
            "minimum_relative_f1": MIN_RELATIVE_F1,
            "minimum_support_f1": MIN_SUPPORT_F1,
            "minimum_support_precision": MIN_SUPPORT_PRECISION,
            "minimum_support_recall": MIN_SUPPORT_RECALL,
            "maximum_final_l1": MAX_FINAL_L1,
            "foreground_limit": foreground_limit,
        },
        "checks": checks,
        "final_probe": final,
    }
    args.decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["decision"] == "promote" else 2


if __name__ == "__main__":
    raise SystemExit(main())
