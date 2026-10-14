#!/usr/bin/env python3
"""
Feature-Group Ablation and Adaptive LOLBin Structural Masking Verification.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    summary_path = root / "results" / "statistical_results_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    exp1 = summary["experiment_1_otrf"]
    exp2 = summary["experiment_2_salad_soc"]
    exp3 = summary["experiment_3_lmd2023_kev"]

    print("=== OTRF Feature-Group Ablations (Held-Out N_test = 13,117) ===")
    for name, res in exp1["feature_group_ablations"].items():
        print(f"  {name:24s}: Recall={res['recall']*100:6.2f}%, AUROC={res['auroc']:.4f}")

    print("\n=== OTRF Adaptive LOLBin Masking Curve (tau_LLR = 4.0) ===")
    for row in exp1["adaptive_lolbin_masking_curve"]:
        print(f"  k={row['masked_attributes_k']} (rho={row['rho_mask']:.2f}): "
              f"Recall={row['hostile_recall']*100:6.2f}%, Mean Hostile LLR={row['mean_llr']:+.4f}")

    print("\n=== SALAD-SOC Feature-Group Ablations (Held-Out N_test = 125,035) ===")
    for name, res in exp2["feature_group_ablations"].items():
        print(f"  {name:24s}: Recall={res['recall']*100:6.2f}%, AUROC={res['auroc']:.4f}")

    print("\n=== LMD-2023 Feature-Group Ablations (Held-Out N_test = 533,379) ===")
    for name, res in exp3["feature_group_ablations"].items():
        print(f"  {name:24s}: Hostile Recall={res['recall']*100:6.2f}%, "
              f"RedTeam Recall={res['redteam_recall']*100:6.2f}%, AUROC={res['auroc']:.4f}")


if __name__ == "__main__":
    main()
