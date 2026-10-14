#!/usr/bin/env python3
"""Focused runner for PITSA and DECS mechanism ablations (AB-1 to AB-8)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis import analyze_study1b
from src.pitsa import evaluate_pitsa_and_ablations
from src.simulation import simulate_study1b_cohort


def main() -> int:
    benchmark_path = ROOT_DIR / "configs" / "soc_itpi_24_benchmark.json"
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    pitsa_ablation = evaluate_pitsa_and_ablations(benchmark, seed=20261014)
    df_1b = simulate_study1b_cohort(benchmark, seed=20261015)
    study1b_summary = analyze_study1b(df_1b)

    with open(results_dir / "pitsa_stress_results.json", "w", encoding="utf-8") as f:
        json.dump(pitsa_ablation, f, indent=2)

    print("PITSA Stress-1000 F1:", pitsa_ablation["stress_1000_ablation"]["LLR_full"]["f1"])
    print(
        "Study 1B Contrasts (C4 vs AB1 d):",
        study1b_summary["contrasts"]["Stage2_PITSA_Cue_Increment_C4_vs_AB1"]["cohens_d"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
