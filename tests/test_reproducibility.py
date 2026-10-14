"""Automated reproducibility and parity test suite for the Epistemic Fluency Trap package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis import analyze_study1, analyze_study1b, evaluate_study2_qualitative
from src.pitsa import evaluate_pitsa_and_ablations
from src.simulation import simulate_study1_cohort, simulate_study1b_cohort

BENCHMARK_PATH = ROOT_DIR / "configs" / "soc_itpi_24_benchmark.json"
LIKELIHOODS_PATH = ROOT_DIR / "configs" / "pitsa_empirical_likelihoods.json"


def test_benchmark_and_likelihood_configs_exist() -> None:
    assert BENCHMARK_PATH.exists(), "Missing configs/soc_itpi_24_benchmark.json"
    assert LIKELIHOODS_PATH.exists(), "Missing configs/pitsa_empirical_likelihoods.json"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    assert benchmark["total_trials"] == 24
    assert benchmark["regime_counts"] == {"R1a": 8, "R1b": 2, "R2": 8, "R3": 6}


def test_pitsa_ablation_and_stress_reproducibility() -> None:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    res = evaluate_pitsa_and_ablations(benchmark, seed=20261014)
    assert res["soc_itpi_24_ablation"]["full"]["R3_ITPI_recall"] == 1.0
    assert res["soc_itpi_24_ablation"]["unpartitioned"]["R3_ITPI_recall"] == 0.0
    assert res["stress_1000_ablation"]["LLR_full"]["f1"] == 0.9562
    assert res["stress_1000_ablation"]["LLR_full"]["false_friction_rate"] == 0.0283


def test_study1_hypotheses_parity() -> None:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    df_s1 = simulate_study1_cohort(benchmark, seed=20261014)
    assert len(df_s1) == 3840

    s1 = analyze_study1(df_s1, boot_seed=20261014)
    ht = s1["hypothesis_tests"]
    assert ht["H1"]["cohens_d"] == -1.6177
    assert ht["H2"]["cohens_d"] == -1.1786
    assert ht["H3a_H3b"]["cohens_d_OC"] == 1.6396
    assert ht["H3c_TOST_Equivalence"]["equivalent_within_5pp"] is True
    assert ht["H3c_TOST_Equivalence"]["p_tost"] == 0.012326
    assert ht["H4_Bootstrap_Mediation"]["proportion_mediated"] == 0.728
    assert ht["H5_Accountability"]["c4_vs_c2_cohens_d"] == 2.1715


def test_study1b_and_study2_parity() -> None:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    df_1b = simulate_study1b_cohort(benchmark, seed=20261015)
    assert len(df_1b) == 1440
    s1b = analyze_study1b(df_1b)
    assert s1b["contrasts"]["Stage1_Anchoring_Increment_AB1_vs_C2"]["cohens_d"] == 0.8501
    assert s1b["contrasts"]["Stage2_PITSA_Cue_Increment_C4_vs_AB1"]["cohens_d"] == 1.1782
    assert s1b["contrasts"]["Conditional_DECS_vs_Static_Warning_C4_vs_DC1"]["cohens_d"] == 1.3483

    s2 = evaluate_study2_qualitative()
    assert s2["n_informants"] == 20
    assert s2["total_first_order_codes"] == 34
    assert s2["overall_cohens_kappa"] == 0.88
