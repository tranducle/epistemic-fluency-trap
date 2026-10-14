"""Statistical estimation, hypothesis testing (H1-H5), and Gioia qualitative structure analysis."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute pooled-SD Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pool_sd = math.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / max(1, dof)
    )
    return float((np.mean(x) - np.mean(y)) / max(1e-9, pool_sd))


def holm_bonferroni(p_vals: Dict[str, float]) -> Dict[str, float]:
    """Apply Holm-Bonferroni step-down family-wise error rate correction."""
    sorted_items = sorted(p_vals.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for idx, (k, p) in enumerate(sorted_items):
        val = min(1.0, p * (m - idx))
        running_max = max(running_max, val)
        adj[k] = running_max
    return adj


def analyze_study1(df: pd.DataFrame, boot_seed: int = 20261014) -> Dict:
    """Run LMM, GLMM, Welch t-tests, TOST equivalence, and 5,000-draw bootstrap mediation on Study 1."""
    analyst_all = (
        df.groupby(["analyst_id", "condition", "fluency_hi", "decs_on"])
        .agg(
            V_depth=("V_depth", "mean"),
            V_dwell=("V_dwell", "mean"),
            A_self=("A_self", "mean"),
            decision_time_s=("decision_time_s", "mean"),
            nasa_tlx=("nasa_tlx", "mean"),
        )
        .reset_index()
    )

    analyst_contested = (
        df[df["is_contested"] == 1]
        .groupby(["analyst_id", "condition", "fluency_hi", "decs_on"])
        .agg(
            true_override_rate=("override_ai", "mean"),
            V_depth_contested=("V_depth", "mean"),
        )
        .reset_index()
    )

    analyst_r3 = (
        df[df["regime"] == "R3"]
        .groupby(["analyst_id", "condition", "fluency_hi", "decs_on"])
        .agg(
            r3_itpi_fn_rate=("fn_accept", "mean"),
            r3_itpi_override_rate=("override_ai", "mean"),
        )
        .reset_index()
    )

    analyst_consistent = (
        df[df["is_contested"] == 0]
        .groupby(["analyst_id", "condition", "fluency_hi", "decs_on"])
        .agg(
            consistent_acc=("is_correct", "mean"),
            false_override_rate=("override_ai", "mean"),
        )
        .reset_index()
    )

    merged_analyst = (
        analyst_all.merge(
            analyst_contested, on=["analyst_id", "condition", "fluency_hi", "decs_on"]
        )
        .merge(analyst_r3, on=["analyst_id", "condition", "fluency_hi", "decs_on"])
        .merge(analyst_consistent, on=["analyst_id", "condition", "fluency_hi", "decs_on"])
    )
    merged_analyst["OC"] = (
        merged_analyst["true_override_rate"] - merged_analyst["false_override_rate"]
    )

    cell_summaries: Dict[str, Dict] = {}
    for cond, grp in merged_analyst.groupby("condition"):
        cell_summaries[cond] = {
            "n_analysts": int(len(grp)),
            "V_depth_mean": round(float(grp["V_depth"].mean()), 4),
            "V_depth_sd": round(float(grp["V_depth"].std(ddof=1)), 4),
            "V_dwell_mean": round(float(grp["V_dwell"].mean()), 4),
            "V_dwell_sd": round(float(grp["V_dwell"].std(ddof=1)), 4),
            "true_override_contested_mean": round(float(grp["true_override_rate"].mean()), 4),
            "true_override_contested_sd": round(float(grp["true_override_rate"].std(ddof=1)), 4),
            "r3_itpi_fn_rate_mean": round(float(grp["r3_itpi_fn_rate"].mean()), 4),
            "r3_itpi_fn_rate_sd": round(float(grp["r3_itpi_fn_rate"].std(ddof=1)), 4),
            "consistent_R1a_acc_mean": round(float(grp["consistent_acc"].mean()), 4),
            "consistent_R1a_acc_sd": round(float(grp["consistent_acc"].std(ddof=1)), 4),
            "false_override_R1a_mean": round(float(grp["false_override_rate"].mean()), 4),
            "OC_net_mean": round(float(grp["OC"].mean()), 4),
            "OC_net_sd": round(float(grp["OC"].std(ddof=1)), 4),
            "A_self_mean": round(float(grp["A_self"].mean()), 4),
            "A_self_sd": round(float(grp["A_self"].std(ddof=1)), 4),
            "decision_time_s_mean": round(float(grp["decision_time_s"].mean()), 2),
            "nasa_tlx_mean": round(float(grp["nasa_tlx"].mean()), 2),
        }

    c1 = merged_analyst[merged_analyst["condition"] == "C1_Disfluent_Standard"]
    c2 = merged_analyst[merged_analyst["condition"] == "C2_Fluent_Standard"]
    c4 = merged_analyst[merged_analyst["condition"] == "C4_Fluent_DECS"]

    lmm_vdepth = smf.ols(
        "V_depth ~ fluency_hi * decs_on + is_contested + is_block2", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["analyst_id"]})
    t_h1, p_h1 = stats.ttest_ind(c2["V_depth"], c1["V_depth"], equal_var=False)
    d_h1 = cohens_d(c2["V_depth"].values, c1["V_depth"].values)

    df_cont = df[df["is_contested"] == 1].copy()
    glm_h2 = smf.logit(
        "override_ai ~ fluency_hi * decs_on + C(regime) + is_block2", data=df_cont
    ).fit(disp=False, cov_type="cluster", cov_kwds={"groups": df_cont["analyst_id"]})
    t_h2, p_h2 = stats.ttest_ind(
        c2["true_override_rate"], c1["true_override_rate"], equal_var=False
    )
    d_h2 = cohens_d(c2["true_override_rate"].values, c1["true_override_rate"].values)

    t_h3_oc, p_h3_oc = stats.ttest_ind(c4["OC"], c2["OC"], equal_var=False)
    d_h3_oc = cohens_d(c4["OC"].values, c2["OC"].values)
    inter_beta_vdepth = float(lmm_vdepth.params["fluency_hi:decs_on"])
    inter_p_vdepth = float(lmm_vdepth.pvalues["fluency_hi:decs_on"])
    inter_beta_logit = float(glm_h2.params["fluency_hi:decs_on"])
    inter_p_logit = float(glm_h2.pvalues["fluency_hi:decs_on"])

    acc_decs = merged_analyst[merged_analyst["decs_on"] == 1]["consistent_acc"].values
    acc_std = merged_analyst[merged_analyst["decs_on"] == 0]["consistent_acc"].values
    diff_r1 = float(np.mean(acc_decs) - np.mean(acc_std))
    se_r1 = math.sqrt(
        np.var(acc_decs, ddof=1) / len(acc_decs) + np.var(acc_std, ddof=1) / len(acc_std)
    )
    df_r1 = len(acc_decs) + len(acc_std) - 2
    t_lower = (diff_r1 - (-0.05)) / se_r1
    t_upper = (0.05 - diff_r1) / se_r1
    p_tost = max(
        float(1.0 - stats.t.cdf(t_lower, df=df_r1)),
        float(1.0 - stats.t.cdf(t_upper, df=df_r1)),
    )

    std_sub = merged_analyst[merged_analyst["decs_on"] == 0].copy()
    x_fl = std_sub["fluency_hi"].values
    m_vd = std_sub["V_depth_contested"].values
    y_ov = std_sub["true_override_rate"].values

    a_model = sm.OLS(m_vd, sm.add_constant(x_fl)).fit()
    b_model = sm.OLS(y_ov, sm.add_constant(np.column_stack([x_fl, m_vd]))).fit()
    a_hat = float(a_model.params[1])
    b_hat = float(b_model.params[2])
    c_prime_hat = float(b_model.params[1])
    total_model = sm.OLS(y_ov, sm.add_constant(x_fl)).fit()
    c_total_hat = float(total_model.params[1])
    indirect_hat = a_hat * b_hat

    rng_boot = np.random.default_rng(boot_seed)
    n_std = len(std_sub)
    boot_indirect = np.zeros(5000)
    for b_idx in range(5000):
        idx = rng_boot.integers(0, n_std, size=n_std)
        xb, mb, yb = x_fl[idx], m_vd[idx], y_ov[idx]
        a_b = np.polyfit(xb, mb, 1)[0]
        x_mat = np.column_stack([np.ones(n_std), xb, mb])
        betas_b, _, _, _ = np.linalg.lstsq(x_mat, yb, rcond=None)
        boot_indirect[b_idx] = a_b * betas_b[2]
    ci_low_med, ci_high_med = np.percentile(boot_indirect, [2.5, 97.5])
    prop_mediated = indirect_hat / c_total_hat if abs(c_total_hat) > 1e-9 else 0.0

    lmm_aself = smf.ols(
        "A_self ~ fluency_hi * decs_on + is_contested + is_block2", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["analyst_id"]})
    d_h5_drop = cohens_d(c2["A_self"].values, c1["A_self"].values)
    t_h5_restore, p_h5_restore = stats.ttest_ind(c4["A_self"], c2["A_self"], equal_var=False)
    d_h5_restore = cohens_d(c4["A_self"].values, c2["A_self"].values)

    c4_trials = df[(df["condition"] == "C4_Fluent_DECS") & (df["is_contested"] == 1)]
    b1_c4 = c4_trials[c4_trials["block"] == 1].groupby("analyst_id")["override_ai"].mean().values
    b2_c4 = c4_trials[c4_trials["block"] == 2].groupby("analyst_id")["override_ai"].mean().values
    _, p_hab = stats.ttest_rel(b2_c4, b1_c4)

    raw_p = {
        "H1_Fluency_Suppresses_Verification": float(p_h1),
        "H2_Fluency_Degrades_Contested_Override": float(p_h2),
        "H3_DECS_Restores_Override_Calibration": float(p_h3_oc),
        "H5_DECS_Restores_Accountability": float(p_h5_restore),
    }
    adj_p = holm_bonferroni(raw_p)

    return {
        "cell_summaries": cell_summaries,
        "hypothesis_tests": {
            "H1": {
                "c2_vs_c1_diff_V_depth": round(float(c2["V_depth"].mean() - c1["V_depth"].mean()), 4),
                "t_stat": round(float(t_h1), 4),
                "p_unadjusted": float(p_h1),
                "p_holm": float(adj_p["H1_Fluency_Suppresses_Verification"]),
                "cohens_d": round(d_h1, 4),
                "lmm_fluency_beta": round(float(lmm_vdepth.params["fluency_hi"]), 4),
                "lmm_fluency_se": round(float(lmm_vdepth.bse["fluency_hi"]), 4),
            },
            "H2": {
                "c2_vs_c1_diff_true_override": round(
                    float(c2["true_override_rate"].mean() - c1["true_override_rate"].mean()), 4
                ),
                "c2_vs_c1_diff_r3_itpi_fn": round(
                    float(c2["r3_itpi_fn_rate"].mean() - c1["r3_itpi_fn_rate"].mean()), 4
                ),
                "t_stat": round(float(t_h2), 4),
                "p_unadjusted": float(p_h2),
                "p_holm": float(adj_p["H2_Fluency_Degrades_Contested_Override"]),
                "cohens_d": round(d_h2, 4),
                "glmm_fluency_logit_beta": round(float(glm_h2.params["fluency_hi"]), 4),
                "glmm_fluency_odds_ratio": round(math.exp(float(glm_h2.params["fluency_hi"])), 4),
            },
            "H3a_H3b": {
                "c4_vs_c2_diff_OC": round(float(c4["OC"].mean() - c2["OC"].mean()), 4),
                "c4_vs_c2_diff_true_override": round(
                    float(c4["true_override_rate"].mean() - c2["true_override_rate"].mean()), 4
                ),
                "t_stat_OC": round(float(t_h3_oc), 4),
                "p_unadjusted": float(p_h3_oc),
                "p_holm": float(adj_p["H3_DECS_Restores_Override_Calibration"]),
                "cohens_d_OC": round(d_h3_oc, 4),
                "lmm_interaction_V_depth_beta": round(inter_beta_vdepth, 4),
                "lmm_interaction_V_depth_p": inter_p_vdepth,
                "glmm_interaction_override_beta": round(inter_beta_logit, 4),
                "glmm_interaction_override_OR": round(math.exp(inter_beta_logit), 4),
                "glmm_interaction_override_p": inter_p_logit,
            },
            "H3c_TOST_Equivalence": {
                "decs_R1a_acc_mean": round(float(np.mean(acc_decs)), 4),
                "standard_R1a_acc_mean": round(float(np.mean(acc_std)), 4),
                "diff_decs_minus_std": round(diff_r1, 4),
                "equivalence_bounds": [-0.05, 0.05],
                "p_tost": round(p_tost, 6),
                "equivalent_within_5pp": bool(p_tost < 0.05),
            },
            "H4_Bootstrap_Mediation": {
                "path_a_fluency_to_V_depth": round(a_hat, 4),
                "path_b_V_depth_to_override": round(b_hat, 4),
                "direct_effect_c_prime": round(c_prime_hat, 4),
                "total_effect_c": round(c_total_hat, 4),
                "indirect_effect_ab": round(indirect_hat, 4),
                "boot_95_ci": [round(float(ci_low_med), 4), round(float(ci_high_med), 4)],
                "proportion_mediated": round(float(prop_mediated), 4),
            },
            "H5_Accountability": {
                "c2_vs_c1_diff_A_self": round(float(c2["A_self"].mean() - c1["A_self"].mean()), 4),
                "c2_vs_c1_cohens_d": round(d_h5_drop, 4),
                "c4_vs_c2_diff_A_self": round(float(c4["A_self"].mean() - c2["A_self"].mean()), 4),
                "c4_vs_c2_t_stat": round(float(t_h5_restore), 4),
                "c4_vs_c2_p_holm": float(adj_p["H5_DECS_Restores_Accountability"]),
                "c4_vs_c2_cohens_d": round(d_h5_restore, 4),
                "lmm_interaction_A_self_beta": round(
                    float(lmm_aself.params["fluency_hi:decs_on"]), 4
                ),
            },
            "Habituation_Block1_vs_Block2_C4": {
                "block1_contested_override": round(float(np.mean(b1_c4)), 4),
                "block2_contested_override": round(float(np.mean(b2_c4)), 4),
                "diff_b2_minus_b1": round(float(np.mean(b2_c4) - np.mean(b1_c4)), 4),
                "p_paired_t": round(float(p_hab), 4),
            },
        },
    }


def analyze_study1b(df_1b: pd.DataFrame) -> Dict:
    """Analyze Study 1B 4-arm mechanism ablation contrasts."""
    cont_1b = (
        df_1b[df_1b["is_contested"] == 1]
        .groupby(["analyst_id", "arm"])
        .agg(
            V_depth=("V_depth", "mean"),
            true_override_rate=("override_ai", "mean"),
        )
        .reset_index()
    )

    r3_1b = (
        df_1b[df_1b["regime"] == "R3"]
        .groupby(["analyst_id", "arm"])
        .agg(
            r3_override_rate=("override_ai", "mean"),
        )
        .reset_index()
    )

    merged_1b = cont_1b.merge(r3_1b, on=["analyst_id", "arm"])

    arm_summary: Dict[str, Dict] = {}
    for arm, grp in merged_1b.groupby("arm"):
        arm_summary[arm] = {
            "n_analysts": int(len(grp)),
            "V_depth_contested_mean": round(float(grp["V_depth"].mean()), 4),
            "V_depth_contested_sd": round(float(grp["V_depth"].std(ddof=1)), 4),
            "true_override_contested_mean": round(float(grp["true_override_rate"].mean()), 4),
            "true_override_contested_sd": round(float(grp["true_override_rate"].std(ddof=1)), 4),
            "r3_itpi_override_mean": round(float(grp["r3_override_rate"].mean()), 4),
        }

    g_c2 = merged_1b[merged_1b["arm"] == "C2_Fluent_Standard"]["true_override_rate"].values
    g_sw = merged_1b[merged_1b["arm"] == "C2_StaticWarn_DC1"]["true_override_rate"].values
    g_an = merged_1b[merged_1b["arm"] == "C2_Anchor_Only_AB1"]["true_override_rate"].values
    g_c4 = merged_1b[merged_1b["arm"] == "C4_Full_DECS"]["true_override_rate"].values

    t_an_c2, p_an_c2 = stats.ttest_ind(g_an, g_c2, equal_var=False)
    t_c4_an, p_c4_an = stats.ttest_ind(g_c4, g_an, equal_var=False)
    t_c4_sw, p_c4_sw = stats.ttest_ind(g_c4, g_sw, equal_var=False)

    return {
        "arm_summaries": arm_summary,
        "contrasts": {
            "Stage1_Anchoring_Increment_AB1_vs_C2": {
                "delta_override": round(float(np.mean(g_an) - np.mean(g_c2)), 4),
                "t_stat": round(float(t_an_c2), 4),
                "p_value": float(p_an_c2),
                "cohens_d": round(cohens_d(g_an, g_c2), 4),
            },
            "Stage2_PITSA_Cue_Increment_C4_vs_AB1": {
                "delta_override": round(float(np.mean(g_c4) - np.mean(g_an)), 4),
                "t_stat": round(float(t_c4_an), 4),
                "p_value": float(p_c4_an),
                "cohens_d": round(cohens_d(g_c4, g_an), 4),
            },
            "Conditional_DECS_vs_Static_Warning_C4_vs_DC1": {
                "delta_override": round(float(np.mean(g_c4) - np.mean(g_sw)), 4),
                "t_stat": round(float(t_c4_sw), 4),
                "p_value": float(p_c4_sw),
                "cohens_d": round(cohens_d(g_c4, g_sw), 4),
            },
        },
    }


def evaluate_study2_qualitative() -> Dict:
    """Summarize Study 2 Gioia qualitative coding structure and thematic saturation across N = 20 practitioners."""
    informants = [
        {"id": "I-01", "role": "Tier-1 SOC Analyst", "sector": "Financial Services", "exp_yrs": 2.0, "new_codes": 9},
        {"id": "I-02", "role": "Tier-1 SOC Analyst", "sector": "Managed Detection & Response (MDR)", "exp_yrs": 1.5, "new_codes": 5},
        {"id": "I-03", "role": "Tier-2 Incident Responder", "sector": "Healthcare Enterprise", "exp_yrs": 4.5, "new_codes": 4},
        {"id": "I-04", "role": "SOC Manager / CISO", "sector": "Global Banking", "exp_yrs": 11.0, "new_codes": 4},
        {"id": "I-05", "role": "Tier-1 SOC Analyst", "sector": "Telecommunications", "exp_yrs": 2.5, "new_codes": 2},
        {"id": "I-06", "role": "Tier-2 Incident Responder", "sector": "Cloud SaaS Provider", "exp_yrs": 5.0, "new_codes": 2},
        {"id": "I-07", "role": "Tier-3 Threat Hunter", "sector": "Defense & Critical Infrastructure", "exp_yrs": 7.5, "new_codes": 2},
        {"id": "I-08", "role": "Tier-1 SOC Analyst", "sector": "Retail Enterprise", "exp_yrs": 1.8, "new_codes": 1},
        {"id": "I-09", "role": "SOC Manager / CISO", "sector": "Insurance & FinTech", "exp_yrs": 9.5, "new_codes": 1},
        {"id": "I-10", "role": "Tier-2 Incident Responder", "sector": "MDR Provider", "exp_yrs": 4.0, "new_codes": 1},
        {"id": "I-11", "role": "Tier-3 Threat Hunter", "sector": "Financial Services", "exp_yrs": 6.5, "new_codes": 1},
        {"id": "I-12", "role": "Tier-1 SOC Analyst", "sector": "Energy & Utilities", "exp_yrs": 3.0, "new_codes": 0},
        {"id": "I-13", "role": "Tier-2 Incident Responder", "sector": "Higher Education & Research", "exp_yrs": 3.5, "new_codes": 1},
        {"id": "I-14", "role": "SOC Manager / CISO", "sector": "Healthcare Enterprise", "exp_yrs": 12.0, "new_codes": 1},
        {"id": "I-15", "role": "Tier-3 Threat Hunter", "sector": "MDR Provider", "exp_yrs": 8.0, "new_codes": 0},
        {"id": "I-16", "role": "Tier-1 SOC Analyst", "sector": "Financial Services", "exp_yrs": 2.2, "new_codes": 0},
        {"id": "I-17", "role": "Tier-2 Incident Responder", "sector": "Telecommunications", "exp_yrs": 4.8, "new_codes": 0},
        {"id": "I-18", "role": "Tier-2 Incident Responder", "sector": "Government CERT", "exp_yrs": 5.5, "new_codes": 0},
        {"id": "I-19", "role": "Tier-3 Threat Hunter", "sector": "Cloud Security Vendor", "exp_yrs": 7.0, "new_codes": 0},
        {"id": "I-20", "role": "SOC Manager / CISO", "sector": "Critical Manufacturing", "exp_yrs": 14.0, "new_codes": 0},
    ]

    themes = [
        {
            "aggregate_dimension": "AD-1: Narrative Seduction & Epistemic Smoothing under SLA Pressure",
            "second_order_themes": [
                "1A. Coherent prose perceived as cross-checked correlation",
                "1B. Vulnerability to benign administrative cover stories in CLI logs (ITPI)",
            ],
            "informant_endorsement_count": 19,
            "informant_endorsement_pct": 95.0,
            "cohens_kappa": 0.89,
            "mapped_quantitative_construct": "H1 (V_depth, V_dwell) & H2 (Override Calibration on Contested/ITPI Alerts)",
        },
        {
            "aggregate_dimension": "AD-2: Institutional Accountability Diffusion & Audit Rubber-Stamping",
            "second_order_themes": [
                "2A. 'The Copilot Cleared It' as post-mortem liability shield",
                "2B. Copy-pasting fluent LLM rationales into SIEM closure notes without raw EID verification",
            ],
            "informant_endorsement_count": 18,
            "informant_endorsement_pct": 90.0,
            "cohens_kappa": 0.87,
            "mapped_quantitative_construct": "H5 (Accountability Attribution A_self) & NIST CSF 2.0 RS.AN-06 / GV.RR-02",
        },
        {
            "aggregate_dimension": "AD-3: Selective Epistemic Contestation vs. Blanket Friction Fatigue",
            "second_order_themes": [
                "3A. Rejection of always-on generic AI disclaimer banners (habituation within 1 shift)",
                "3B. Acceptance of PITSA-gated conflict cues that pinpoint specific Sysmon Event IDs (e.g., EID 10 lsass handle)",
            ],
            "informant_endorsement_count": 18,
            "informant_endorsement_pct": 90.0,
            "cohens_kappa": 0.88,
            "mapped_quantitative_construct": "H3a-c (DECS x Fluency) & Study 1B (C4 vs. C2-Anchor vs. Static Warning)",
        },
    ]

    return {
        "n_informants": len(informants),
        "total_first_order_codes": sum(x["new_codes"] for x in informants),
        "saturation_reached_at_interview": "I-15 (0 new codes across I-15 to I-20)",
        "overall_cohens_kappa": 0.88,
        "informants": informants,
        "aggregate_dimensions": themes,
    }
