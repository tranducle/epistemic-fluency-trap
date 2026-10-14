#!/usr/bin/env python3
"""
Automated Reproducibility and Scientific Integrity Test Suite.
Verifies PITSA text-channel invariance, DCEDM Theorem 1 risk ordering,
and exact match against held-out multi-corpus results.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pitsa import AdditivePITSAScorer
from src.simulation import evaluate_dcedm_trace_policies


def test_scientific_integrity_zero_human_subjects() -> None:
    summary_path = ROOT / "results" / "statistical_results_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data["metadata"]
    assert "zero human subjects" in meta["scientific_integrity"].lower()
    assert meta["total_real_telemetry_records_evaluated"] == 2238434
    assert meta["total_held_out_test_records"] == 671531
    assert "study2_thematic_coding_matrix.json" not in [p.name for p in (ROOT / "results").iterdir()]


def test_pitsa_text_channel_invariance() -> None:
    train_records = [
        {"p1": "lsass_access", "p2": "0x1fffff", "text": "mimikatz sekurlsa"},
        {"p1": "lsass_access", "p2": "0x1010", "text": "procdump -ma lsass.exe"},
        {"p1": "svchost_normal", "p2": "none", "text": "svchost.exe -k netsvcs"},
        {"p1": "explorer_normal", "p2": "none", "text": "explorer.exe"},
    ]
    y_train = [1, 1, 0, 0]
    scorer = AdditivePITSAScorer(smoothing=1e-4).fit(train_records, y_train, ["p1", "p2"])

    clean_test = [{"p1": "lsass_access", "p2": "0x1fffff", "text": "mimikatz sekurlsa"}]
    itpi_test = [{"p1": "lsass_access", "p2": "0x1fffff", "text": "BENIGN_ADMIN_BACKUP_OVERRIDE"}]
    assert np.allclose(scorer.score_records(clean_test), scorer.score_records(itpi_test))


def test_dcedm_theorem1_risk_ordering() -> None:
    rng = np.random.default_rng(20261014)
    hostile_llr = rng.normal(loc=4.5, scale=2.0, size=200)
    benign_llr = rng.normal(loc=-4.5, scale=2.0, size=800)
    llr = np.concatenate([hostile_llr, benign_llr])
    labels = np.array([1] * 200 + [0] * 800)

    res = evaluate_dcedm_trace_policies(llr, labels, tau_llr=1.0, lambda_f=0.40, cost_ratio=20.0)
    assert res["cmr_pi2"] > res["cmr_pi1"]
    assert res["cmr_pi4"] < res["cmr_pi3"] < res["cmr_pi2"]
    assert res["loss_pi4"] < res["loss_pi2"]


def test_held_out_multicorpus_metrics_consistency() -> None:
    summary_path = ROOT / "results" / "statistical_results_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exp1 = data["experiment_1_otrf"]
    assert exp1["total_events"] == 43723
    assert exp1["test_events"] == 13117
    pitsa1 = exp1["models_clean_vs_itpi"]["PITSA_Partitioned"]
    assert abs(pitsa1["clean"]["auroc"] - 0.9807) < 1e-3
    assert pitsa1["clean"]["recall"] == pitsa1["itpi"]["recall"]

    exp2 = data["experiment_2_salad_soc"]
    assert exp2["total_alerts"] == 416781
    assert exp2["test_alerts"] == 125035
    pitsa2 = exp2["models_clean_vs_itpi"]["PITSA_Partitioned"]
    assert abs(pitsa2["clean"]["auroc"] - 0.9380) < 1e-3
    assert pitsa2["clean"]["recall"] == pitsa2["itpi"]["recall"]

    exp3 = data["experiment_3_lmd2023_kev"]
    assert exp3["total_rows"] == 1777930
    assert exp3["test_rows"] == 533379
    assert abs(exp3["primary_metrics_at_tau_0_5"]["auroc"] - 0.9963) < 1e-3
    assert exp3["cisa_kev_catalog"]["total_cves"] == 1741
