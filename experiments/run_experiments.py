#!/usr/bin/env python3
"""Main end-to-end experiment runner for Study 1, Study 1B, and Study 2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis import analyze_study1, analyze_study1b, evaluate_study2_qualitative
from src.pitsa import evaluate_pitsa_and_ablations
from src.simulation import simulate_study1_cohort, simulate_study1b_cohort


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full multi-method evaluation (PITSA/DECS, Study 1, Study 1B, and Study 2)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "configs" / "default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export trial-level behavioral CSV files to outputs/",
    )
    args = parser.parse_args()

    benchmark_path = ROOT_DIR / "configs" / "soc_itpi_24_benchmark.json"
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    exp1 = evaluate_pitsa_and_ablations(benchmark, seed=20261014)
    with open(results_dir / "pitsa_stress_results.json", "w", encoding="utf-8") as f:
        json.dump(exp1, f, indent=2)

    df_s1 = simulate_study1_cohort(benchmark, seed=20261014)
    exp2 = analyze_study1(df_s1, boot_seed=20261014)

    seed_list = [20261014, 20261015, 20261016, 20261017, 20261018]
    multiseed_metrics = []
    for s in seed_list:
        df_tmp = simulate_study1_cohort(benchmark, seed=s)
        c2_tmp = df_tmp[
            (df_tmp["condition"] == "C2_Fluent_Standard") & (df_tmp["is_contested"] == 1)
        ]
        c4_tmp = df_tmp[
            (df_tmp["condition"] == "C4_Fluent_DECS") & (df_tmp["is_contested"] == 1)
        ]
        multiseed_metrics.append(
            {
                "seed": s,
                "C2_contested_override": round(float(c2_tmp["override_ai"].mean()), 4),
                "C4_contested_override": round(float(c4_tmp["override_ai"].mean()), 4),
                "C2_V_depth": round(
                    float(
                        df_tmp[df_tmp["condition"] == "C2_Fluent_Standard"]["V_depth"].mean()
                    ),
                    4,
                ),
                "C4_V_depth": round(
                    float(df_tmp[df_tmp["condition"] == "C4_Fluent_DECS"]["V_depth"].mean()),
                    4,
                ),
            }
        )

    df_1b = simulate_study1b_cohort(benchmark, seed=20261015)
    exp3 = analyze_study1b(df_1b)

    exp4 = evaluate_study2_qualitative()
    with open(results_dir / "study2_thematic_coding_matrix.json", "w", encoding="utf-8") as f:
        json.dump(exp4, f, indent=2)

    if args.export_csv:
        out_dir = ROOT_DIR / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        df_s1.to_csv(out_dir / "study1_trial_level_data.csv", index=False)
        df_1b.to_csv(out_dir / "study1b_ablation_data.csv", index=False)

    full_summary = {
        "seed": 20261014,
        "experiment_1_pitsa_and_ablations": exp1,
        "experiment_2_study1_main": exp2,
        "experiment_2_multiseed_stability": multiseed_metrics,
        "experiment_3_study1b_ablation": exp3,
        "experiment_4_study2_qualitative": exp4,
    }
    with open(results_dir / "statistical_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    print("Evaluation completed. Summary written to results/statistical_results_summary.json")
    print(
        "SOC-ITPI-24 R3 Recall [Full / Unpart / NoLin / NoSig / NoNet]:",
        exp1["soc_itpi_24_ablation"]["full"]["R3_ITPI_recall"],
        exp1["soc_itpi_24_ablation"]["unpartitioned"]["R3_ITPI_recall"],
        exp1["soc_itpi_24_ablation"]["no_lineage"]["R3_ITPI_recall"],
        exp1["soc_itpi_24_ablation"]["no_binary_sig"]["R3_ITPI_recall"],
        exp1["soc_itpi_24_ablation"]["no_net_task"]["R3_ITPI_recall"],
    )
    print(
        "Study 1 H1 d:",
        exp2["hypothesis_tests"]["H1"]["cohens_d"],
        "p_holm:",
        exp2["hypothesis_tests"]["H1"]["p_holm"],
    )
    print(
        "Study 1 H2 d:",
        exp2["hypothesis_tests"]["H2"]["cohens_d"],
        "p_holm:",
        exp2["hypothesis_tests"]["H2"]["p_holm"],
    )
    print(
        "Study 1 H3 d_OC:",
        exp2["hypothesis_tests"]["H3a_H3b"]["cohens_d_OC"],
        "p_holm:",
        exp2["hypothesis_tests"]["H3a_H3b"]["p_holm"],
    )
    print(
        "Study 1 H5 d_restore:",
        exp2["hypothesis_tests"]["H5_Accountability"]["c4_vs_c2_cohens_d"],
        "p_holm:",
        exp2["hypothesis_tests"]["H5_Accountability"]["c4_vs_c2_p_holm"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
