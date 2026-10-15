from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import write_json


ATTACK_PREVALENCE_GRID = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
COST_RATIO_GRID = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]


def _select_pitsa_metrics(metrics_path: Path) -> dict[str, float]:
    frame = pd.read_csv(metrics_path)
    subset = frame[
        (frame["model"] == "PITSA_FullTrain")
        & (frame["condition"] == "clean")
    ]
    if subset.empty:
        raise ValueError("PITSA_FullTrain clean metrics are missing")
    row = subset.iloc[0]
    return {
        "gate_true_positive_rate": float(row["recall"]),
        "gate_false_positive_rate": float(row["fpr"]),
    }


def _llm_rates(llm_results: dict[str, Any]) -> list[dict[str, float | str]]:
    rates: list[dict[str, float | str]] = []
    for item in llm_results.get("model_summaries", []):
        conditions = item.get("conditions", {})
        attacked = conditions.get("attacked_no_cue", {})
        cued = conditions.get("attacked_with_cue", {})
        if not attacked or not cued:
            continue
        rates.append(
            {
                "model": str(item["model"]),
                "unassisted_attack_miss_rate": 1.0 - float(attacked["escalation_rate"]),
                "cued_attack_miss_rate": 1.0 - float(cued["escalation_rate"]),
                "attack_escalation_rate": float(attacked["escalation_rate"]),
                "cued_escalation_rate": float(cued["escalation_rate"]),
            }
        )
    if not rates:
        raise ValueError("LLM attacked and cued rates are missing")
    return rates


def _expected_losses(
    attack_prevalence: float,
    cost_ratio: float,
    gate_tpr: float,
    gate_fpr: float,
    unassisted_miss: float,
    cued_miss: float,
) -> dict[str, float]:
    p = attack_prevalence
    miss_cost = cost_ratio
    cue_cost = 1.0
    selective_miss = (1.0 - gate_tpr) * unassisted_miss + gate_tpr * cued_miss
    selective_cue_rate = p * gate_tpr + (1.0 - p) * gate_fpr
    unassisted_loss = p * miss_cost * unassisted_miss
    static_loss = p * miss_cost * cued_miss + cue_cost
    selective_loss = p * miss_cost * selective_miss + cue_cost * selective_cue_rate
    return {
        "unassisted_loss": float(unassisted_loss),
        "static_cue_loss": float(static_loss),
        "selective_cue_loss": float(selective_loss),
        "selective_attack_miss_rate": float(selective_miss),
        "selective_cue_rate": float(selective_cue_rate),
        "selective_loss_reduction_vs_unassisted_pct": float(
            100.0 * (unassisted_loss - selective_loss) / max(1e-12, unassisted_loss)
        ),
        "selective_loss_reduction_vs_static_pct": float(
            100.0 * (static_loss - selective_loss) / max(1e-12, static_loss)
        ),
    }


def _break_even_ratio(
    attack_prevalence: float,
    gate_tpr: float,
    gate_fpr: float,
    unassisted_miss: float,
    cued_miss: float,
) -> float | None:
    benefit_per_miss_cost = (
        attack_prevalence
        * gate_tpr
        * (unassisted_miss - cued_miss)
    )
    cue_rate = (
        attack_prevalence * gate_tpr
        + (1.0 - attack_prevalence) * gate_fpr
    )
    if benefit_per_miss_cost <= 0.0:
        return None
    return float(cue_rate / benefit_per_miss_cost)


def run_governance_model(paper_root: Path) -> dict[str, Any]:
    result_dir = paper_root / "results"
    metrics_path = result_dir / "salad_metrics_long.csv"
    llm_path = result_dir / "llm_results.json"
    if not metrics_path.exists() or not llm_path.exists():
        raise FileNotFoundError("SALAD and LLM results are required before governance analysis")
    pitsa = _select_pitsa_metrics(metrics_path)
    import json

    llm_results = json.loads(llm_path.read_text(encoding="utf-8"))
    rates = _llm_rates(llm_results)
    rows: list[dict[str, Any]] = []
    break_even: list[dict[str, Any]] = []
    for rate in rates:
        model = str(rate["model"])
        unassisted_miss = float(rate["unassisted_attack_miss_rate"])
        cued_miss = float(rate["cued_attack_miss_rate"])
        for prevalence in ATTACK_PREVALENCE_GRID:
            threshold = _break_even_ratio(
                prevalence,
                pitsa["gate_true_positive_rate"],
                pitsa["gate_false_positive_rate"],
                unassisted_miss,
                cued_miss,
            )
            break_even.append(
                {
                    "model": model,
                    "attack_prevalence": prevalence,
                    "break_even_miss_to_cue_cost_ratio": threshold,
                    "gate_true_positive_rate": pitsa["gate_true_positive_rate"],
                    "gate_false_positive_rate": pitsa["gate_false_positive_rate"],
                    "unassisted_attack_miss_rate": unassisted_miss,
                    "cued_attack_miss_rate": cued_miss,
                }
            )
            for cost_ratio in COST_RATIO_GRID:
                losses = _expected_losses(
                    prevalence,
                    cost_ratio,
                    pitsa["gate_true_positive_rate"],
                    pitsa["gate_false_positive_rate"],
                    unassisted_miss,
                    cued_miss,
                )
                rows.append(
                    {
                        "model": model,
                        "attack_prevalence": prevalence,
                        "miss_to_cue_cost_ratio": cost_ratio,
                        "gate_true_positive_rate": pitsa["gate_true_positive_rate"],
                        "gate_false_positive_rate": pitsa["gate_false_positive_rate"],
                        "unassisted_attack_miss_rate": unassisted_miss,
                        "cued_attack_miss_rate": cued_miss,
                        **losses,
                    }
                )
    loss_frame = pd.DataFrame(rows)
    break_even_frame = pd.DataFrame(break_even)
    loss_path = result_dir / "governance_loss_sensitivity.csv"
    break_even_path = result_dir / "governance_break_even.csv"
    loss_frame.to_csv(loss_path, index=False)
    break_even_frame.to_csv(break_even_path, index=False)
    finite_thresholds = break_even_frame["break_even_miss_to_cue_cost_ratio"].dropna()
    payload = {
        "model_scope": (
            "Evidence-linked operational scenario analysis. It combines observed local LLM "
            "disposition rates with official-split PITSA gate rates. It does not estimate human behavior."
        ),
        "assumptions": {
            "cue_cost": 1.0,
            "miss_cost": "swept as a ratio to cue cost",
            "attack_prevalence": ATTACK_PREVALENCE_GRID,
            "cost_ratio": COST_RATIO_GRID,
            "gated_attack_outcome": "uses the attacked with cue LLM miss rate",
            "ungated_attack_outcome": "uses the attacked without cue LLM miss rate",
            "benign_gate_burden": "uses the official SALAD PITSA false positive rate",
        },
        "pitsa_gate_rates": pitsa,
        "llm_rates": rates,
        "finite_break_even_range": {
            "minimum": float(finite_thresholds.min()) if not finite_thresholds.empty else None,
            "maximum": float(finite_thresholds.max()) if not finite_thresholds.empty else None,
        },
        "loss_sensitivity_file": str(loss_path),
        "break_even_file": str(break_even_path),
        "claim_boundary": (
            "Break-even values are conditional on the evaluated models, stimuli, official SALAD gate rates, "
            "attack prevalence scenarios, and normalized costs. They are not universal behavioral laws."
        ),
    }
    write_json(result_dir / "governance_results.json", payload)
    return payload
