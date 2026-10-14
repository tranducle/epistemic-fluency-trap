"""Behavioral simulation models for Study 1 (2x2 factorial) and Study 1B (4-arm mechanism ablation)."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from src.pitsa import score_pitsa_trial_ablation, sigmoid


def simulate_study1_cohort(benchmark: Dict, seed: int = 20261014) -> pd.DataFrame:
    """Generate Study 1 trial-level behavioral dataset (N = 160 analysts x 24 trials = 3,840 observations)."""
    rng = np.random.default_rng(seed)
    trials = benchmark["trials"]

    conditions = [
        ("C1_Disfluent_Standard", 0, 0),
        ("C2_Fluent_Standard", 1, 0),
        ("C3_Disfluent_DECS", 0, 1),
        ("C4_Fluent_DECS", 1, 1),
    ]

    rows = []
    analyst_id = 0
    for cond_name, fluency_hi, decs_on in conditions:
        for _ in range(40):
            analyst_id += 1
            u_verify = float(rng.normal(0.0, 0.155))
            u_acc = float(rng.normal(0.0, 0.88))
            u_self = float(rng.normal(0.0, 0.98))
            analyst_exp_years = round(float(np.clip(rng.normal(3.8, 1.6), 1.0, 9.5)), 1)

            for t in trials:
                reg = t["regime"]
                block_num = int(t["block"])
                is_block2 = 1 if block_num == 2 else 0
                best_llr, d_pitsa, anchor_attr = score_pitsa_trial_ablation(t, variant="full")
                decs_triggered = int(decs_on == 1 and best_llr > 0.50)

                base_v = 0.55 + u_verify - 0.03 * is_block2
                fluency_effect = -0.21 * fluency_hi * (1 - decs_on) - 0.03 * fluency_hi * decs_on
                decs_stage1_effect = 0.12 * decs_on
                decs_stage2_effect = 0.19 * decs_triggered * d_pitsa
                v_depth = float(
                    np.clip(
                        base_v
                        + fluency_effect
                        + decs_stage1_effect
                        + decs_stage2_effect
                        + rng.normal(0.0, 0.14),
                        0.04,
                        0.98,
                    )
                )
                v_dwell = float(
                    np.clip(0.86 * v_depth + 0.04 * decs_on + rng.normal(0.0, 0.11), 0.04, 0.96)
                )

                if reg == "R1a":
                    logit_correct = (
                        2.45
                        + 0.45 * u_acc
                        + 0.35 * (v_depth - 0.50)
                        + 0.05 * decs_on
                        - 0.08 * is_block2
                    )
                    p_correct = sigmoid(logit_correct)
                    is_correct = int(rng.random() < p_correct)
                    override_ai = 1 - is_correct
                    fn_accept = 0
                else:
                    regime_penalty = -0.35 if reg == "R3" else 0.0
                    logit_override = (
                        -1.35
                        + u_acc
                        + regime_penalty
                        - 0.65 * fluency_hi * (1 - decs_on)
                        + 2.85 * v_depth
                        + 0.90 * decs_triggered
                        + 0.42 * fluency_hi * decs_triggered
                        - 0.12 * is_block2 * (1 - decs_on)
                        - 0.04 * is_block2 * decs_on
                    )
                    p_override = sigmoid(logit_override)
                    override_ai = int(rng.random() < p_override)
                    is_correct = override_ai
                    fn_accept = (1 - override_ai) if reg == "R3" else 0

                a_self_raw = (
                    4.95
                    + u_self
                    - 1.15 * fluency_hi * (1 - decs_on)
                    + 0.52 * decs_on
                    + 0.58 * decs_triggered
                    + 0.75 * (v_depth - 0.50)
                    - 0.10 * is_block2
                    + rng.normal(0.0, 0.75)
                )
                a_self = float(np.clip(a_self_raw, 1.0, 7.0))

                decision_time_s = float(
                    np.clip(
                        38.0
                        - 7.5 * fluency_hi * (1 - decs_on)
                        + 4.2 * decs_on
                        + 8.8 * decs_triggered
                        + 18.0 * (v_depth - 0.50)
                        + rng.normal(0.0, 6.5),
                        14.0,
                        88.0,
                    )
                )
                nasa_tlx = float(
                    np.clip(
                        4.10
                        - 0.25 * fluency_hi * (1 - decs_on)
                        + 0.18 * decs_on
                        + 0.35 * decs_triggered
                        + 0.22 * is_block2
                        + rng.normal(0.0, 0.65),
                        1.0,
                        7.0,
                    )
                )

                rows.append(
                    {
                        "analyst_id": f"A{analyst_id:03d}",
                        "condition": cond_name,
                        "fluency_hi": fluency_hi,
                        "decs_on": decs_on,
                        "analyst_exp_years": analyst_exp_years,
                        "trial_id": t["trial_id"],
                        "block": block_num,
                        "is_block2": is_block2,
                        "regime": reg,
                        "is_contested": int(reg != "R1a"),
                        "PITSA_LLR": best_llr,
                        "D_PITSA": d_pitsa,
                        "PITSA_anchor": anchor_attr,
                        "decs_triggered": decs_triggered,
                        "V_depth": round(v_depth, 4),
                        "V_dwell": round(v_dwell, 4),
                        "override_ai": override_ai,
                        "is_correct": is_correct,
                        "fn_accept": fn_accept,
                        "A_self": round(a_self, 4),
                        "decision_time_s": round(decision_time_s, 2),
                        "nasa_tlx": round(nasa_tlx, 4),
                    }
                )

    return pd.DataFrame(rows)


def simulate_study1b_cohort(benchmark: Dict, seed: int = 20261015) -> pd.DataFrame:
    """Generate Study 1B 4-arm mechanism ablation dataset (N = 120 analysts x 12 trials = 1,440 observations)."""
    rng = np.random.default_rng(seed)
    b1_trials = [t for t in benchmark["trials"] if int(t["block"]) == 1]

    arms = ["C2_Fluent_Standard", "C2_StaticWarn_DC1", "C2_Anchor_Only_AB1", "C4_Full_DECS"]
    rows = []
    aid = 200
    for arm in arms:
        for _ in range(30):
            aid += 1
            u_v = float(rng.normal(0.0, 0.12))
            u_a = float(rng.normal(0.0, 0.65))
            for idx, t in enumerate(b1_trials):
                reg = t["regime"]
                best_llr, d_pitsa, _ = score_pitsa_trial_ablation(t, variant="full")
                if arm == "C2_Fluent_Standard":
                    v_depth = float(np.clip(0.32 + u_v + rng.normal(0.0, 0.11), 0.05, 0.95))
                    stage2_boost = 0.0
                elif arm == "C2_StaticWarn_DC1":
                    decay = math.exp(-0.45 * idx)
                    v_depth = float(
                        np.clip(0.32 + 0.14 * decay + u_v + rng.normal(0.0, 0.11), 0.05, 0.95)
                    )
                    stage2_boost = 0.15 * decay
                elif arm == "C2_Anchor_Only_AB1":
                    v_depth = float(np.clip(0.51 + u_v + rng.normal(0.0, 0.11), 0.05, 0.95))
                    stage2_boost = 0.0
                else:
                    trig = int(best_llr > 0.50)
                    v_depth = float(
                        np.clip(
                            0.51 + 0.22 * trig * d_pitsa + u_v + rng.normal(0.0, 0.11),
                            0.05,
                            0.98,
                        )
                    )
                    stage2_boost = 1.15 * trig

                if reg == "R1a":
                    p_corr = sigmoid(2.35 + 0.45 * u_a + 0.30 * (v_depth - 0.50))
                    is_corr = int(rng.random() < p_corr)
                    ov = 1 - is_corr
                else:
                    reg_pen = -0.30 if reg == "R3" else 0.0
                    logit_ov = -2.05 + u_a + reg_pen + 3.00 * v_depth + stage2_boost
                    ov = int(rng.random() < sigmoid(logit_ov))
                    is_corr = ov

                rows.append(
                    {
                        "analyst_id": f"B{aid:03d}",
                        "arm": arm,
                        "trial_id": t["trial_id"],
                        "regime": reg,
                        "is_contested": int(reg != "R1a"),
                        "V_depth": round(v_depth, 4),
                        "override_ai": ov,
                        "is_correct": is_corr,
                    }
                )

    return pd.DataFrame(rows)
