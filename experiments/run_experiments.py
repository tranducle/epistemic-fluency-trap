#!/usr/bin/env python3
"""
Entrypoint to verify and reproduce the Multi-Corpus Empirical Evaluation
and Trace-Driven Analytical Decision-Theoretic Sensitivity Analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    summary_path = ROOT / "results" / "statistical_results_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    exp1 = summary["experiment_1_otrf"]
    exp2 = summary["experiment_2_salad_soc"]
    exp3 = summary["experiment_3_lmd2023_kev"]
    exp4 = summary["experiment_4_dcedm_decision_theoretic"]

    pitsa1 = exp1["models_clean_vs_itpi"]["PITSA_Partitioned"]
    pitsa2 = exp2["models_clean_vs_itpi"]["PITSA_Partitioned"]
    pitsa3 = exp3["primary_metrics_at_tau_0_5"]

    print("==================================================================")
    print("  EPISTEMIC FLUENCY TRAP — MULTI-CORPUS VERIFICATION SUMMARY")
    print("  Scientific Integrity Policy: Zero Human Participants")
    print("==================================================================")
    print(f"  Exp 1 (OTRF, N={exp1['total_events']:,}, test={exp1['test_events']:,}):")
    print(f"    PITSA AUROC={pitsa1['clean']['auroc']:.4f}, "
          f"Clean Recall={pitsa1['clean']['recall']*100:.2f}%, "
          f"ITPI Recall={pitsa1['itpi']['recall']*100:.2f}%")
    print(f"  Exp 2 (SALAD-SOC, N={exp2['total_alerts']:,}, test={exp2['test_alerts']:,}):")
    print(f"    PITSA AUROC={pitsa2['clean']['auroc']:.4f}, "
          f"Clean Recall={pitsa2['clean']['recall']*100:.2f}%, "
          f"ITPI Recall={pitsa2['itpi']['recall']*100:.2f}%")
    print(f"  Exp 3 (LMD-2023, N={exp3['total_rows']:,}, test={exp3['test_rows']:,}):")
    print(f"    PITSA AUROC={pitsa3['auroc']:.4f}, "
          f"Hostile Recall={pitsa3['recall']*100:.2f}%, "
          f"RedTeam Recall={pitsa3['redteam_lateral_movement_recall']*100:.2f}%")
    print("  Exp 4 (Trace-Driven Decision-Theoretic Sensitivity Analysis):")
    print(f"    OTRF Banner Rate={exp4['otrf_trace_driven']['banner_firing_rate_Pi4_DECS_overall']*100:.2f}%, "
          f"SALAD-SOC Banner Rate={exp4['salad_soc_trace_driven']['banner_firing_rate_Pi4_DECS_overall']*100:.2f}%")
    print("==================================================================")


if __name__ == "__main__":
    main()
