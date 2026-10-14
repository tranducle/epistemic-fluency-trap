#!/usr/bin/env python3
"""
Trace-Driven Analytical Bayesian Decision-Theoretic Sensitivity Evaluator
for the Dual-Cue Epistemic Defeasibility Model (DCEDM) and Two-Stage DECS.

IMPORTANT SCIENTIFIC INTEGRITY NOTE:
This module implements a purely mathematical Bayesian decision-theoretic
sensitivity analysis over empirical held-out PITSA log-likelihood ratio
streams. No human participants, synthetic survey cohorts, or practitioner
interviews are involved.
"""

from __future__ import annotations

from typing import Dict, Sequence
import numpy as np


def evaluate_dcedm_trace_policies(
    llr_scores: np.ndarray,
    labels: np.ndarray,
    tau_llr: float = 1.0,
    lambda_f: float = 0.40,
    cost_ratio: float = 20.0,
    gamma_hab: float = 0.22,
) -> Dict[str, float]:
    """
    Evaluate the four formal DCEDM governance policies (Pi_1..Pi_4) on an
    empirical held-out LLR trace stream:
      - Pi_1: Disfluent / Structured Baseline (Delta F = 0)
      - Pi_2: Unassisted Fluent LLM Triage (Delta F = 0.70, no contestation)
      - Pi_3: Static Warning Banner on 100% of Alerts (habituation gamma_hab)
      - Pi_4: Selective Two-Stage DECS Gated by PITSA (I(LLR > tau_llr))
    """
    y = np.asarray(labels, dtype=int)
    llr_clipped = np.clip(np.asarray(llr_scores, dtype=float), -12.0, 12.0)
    gate = (llr_clipped > tau_llr).astype(float)

    beta_s = 0.28
    delta_f = 0.70
    kappa_f = 2.60
    eta_1 = 0.35
    eta_2 = 0.18
    delta_reb = 0.45
    beta_cost = 0.18 * np.log(max(cost_ratio, 1.0) / 20.0)

    z_pi1 = beta_s * llr_clipped + beta_cost
    z_pi2 = beta_s * llr_clipped - kappa_f * lambda_f * delta_f + beta_cost
    z_pi3 = beta_s * llr_clipped - kappa_f * (1.0 - gamma_hab) * lambda_f * delta_f + beta_cost
    z_pi4 = (
        gate * (beta_s * llr_clipped + delta_reb - eta_2 * lambda_f * delta_f + beta_cost)
        + (1.0 - gate) * (beta_s * llr_clipped - kappa_f * eta_1 * lambda_f * delta_f + beta_cost)
    )

    v_pi1 = 1.0 / (1.0 + np.exp(-z_pi1))
    v_pi2 = 1.0 / (1.0 + np.exp(-z_pi2))
    v_pi3 = 1.0 / (1.0 + np.exp(-z_pi3))
    v_pi4 = 1.0 / (1.0 + np.exp(-z_pi4))

    hostile_mask = y == 1
    cmr_pi1 = float(np.mean(1.0 - v_pi1[hostile_mask]))
    cmr_pi2 = float(np.mean(1.0 - v_pi2[hostile_mask]))
    cmr_pi3 = float(np.mean(1.0 - v_pi3[hostile_mask]))
    cmr_pi4 = float(np.mean(1.0 - v_pi4[hostile_mask]))

    p_hostile = float(np.mean(hostile_mask))
    loss_pi1 = float(p_hostile * cost_ratio * cmr_pi1 + np.mean(v_pi1))
    loss_pi2 = float(p_hostile * cost_ratio * cmr_pi2 + np.mean(v_pi2))
    loss_pi3 = float(p_hostile * cost_ratio * cmr_pi3 + np.mean(v_pi3))
    loss_pi4 = float(p_hostile * cost_ratio * cmr_pi4 + np.mean(v_pi4))

    return {
        "banner_firing_rate_pi4": float(np.mean(gate)),
        "cmr_pi1": cmr_pi1,
        "cmr_pi2": cmr_pi2,
        "cmr_pi3": cmr_pi3,
        "cmr_pi4": cmr_pi4,
        "fluency_trap_cmr_increase_pp": float((cmr_pi2 - cmr_pi1) * 100.0),
        "decs_cmr_reduction_vs_pi2_pp": float((cmr_pi2 - cmr_pi4) * 100.0),
        "loss_pi1": loss_pi1,
        "loss_pi2": loss_pi2,
        "loss_pi3": loss_pi3,
        "loss_pi4": loss_pi4,
        "decs_loss_reduction_vs_pi2_pct": float((loss_pi2 - loss_pi4) / max(loss_pi2, 1e-12) * 100.0),
    }
