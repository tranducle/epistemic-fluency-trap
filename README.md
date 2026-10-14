# The Epistemic Fluency Trap in AI-Augmented Security Operations (`PITSA` & `DECS`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)
[![Reproducibility: Seed 20261014](https://img.shields.io/badge/Seed-20261014-success.svg)](REPRODUCIBILITY.md)

Official open-source artifacts, `SOC-ITPI-24` benchmark, multi-corpus evaluation pipelines, and trace-driven decision-theoretic sensitivity analyses for the manuscript:

> **The Epistemic Fluency Trap in AI-Augmented Security Operations: Prompt-Isolated Telemetry Contestation and Decision-Theoretic Governance Under Adversarial Prompt Injection**

---

## Scientific Integrity & Ethics Statement

This research involves **zero human participants, zero synthetic human survey cohorts, and zero practitioner interviews**. All empirical evaluations (`Experiments 1 to 3`) and trace-driven analytical Bayesian decision-theoretic sensitivity analyses (`Experiment 4`) are conducted exclusively on **four real, publicly verifiable cybersecurity telemetry repositories totaling $2,238,434$ telemetry records ($671,531$ held-out test records) and $1,741$ actively exploited CVE advisories**.

---

## Core Summary of Held-Out Multi-Corpus Results

| Experiment | Public Corpus | Total $N$ (Held-Out Test $N_{\text{test}}$) | Partitioned `PITSA` Held-Out Performance | Unpartitioned ML Baselines Under `ITPI` |
| :--- | :--- | :--- | :--- | :--- |
| **Experiment 1** | `OTRF Security-Datasets` (63 Archives) | $43,723$ ($13,117$ test) | $\text{AUROC} = 0.9807$, $\text{Recall} = 81.94\%$ at $\tau_{\text{LLR}}=4.0$ ($96.13\%$ at $\tau_{\text{LLR}}=1.0$), $0.00\text{ pp}$ `ITPI` drop ($2.37\text{ }\mu\text{s/event}$) | `LogisticRegression`: $85.81\% \to 25.16\%$ ($-60.65\text{ pp}$); `NaiveBayes`: $74.19\% \to 0.00\%$; `RandomForest`: $55.48\% \to 0.00\%$ |
| **Experiment 2** | `SALAD-SOC` Enterprise SIEM Alerts | $416,781$ ($125,035$ test) | Zero-leakage $\text{AUROC} = 0.9380$, $\text{Recall} = 68.68\%$ ($85.74\%$ on `hard` alerts; $100\%$ Lateral Movement, $99.66\%$ Recon), $0.00\text{ pp}$ `ITPI` drop ($3.12\text{ }\mu\text{s/alert}$) | `LogisticRegression`: $100.00\% \to 0.00\%$; `RandomForest`: $100.00\% \to 0.00\%$; `NaiveBayes`: $100.00\% \to 1.77\%$ |
| **Experiment 3** | `LMD-2023` Sysmon + `CISA KEV` | $1,777,930$ ($533,379$ test) + $1,741$ CVEs | $\text{AUROC} = 0.9963$, Overall Hostile Recall $= 99.93\%$ ($49,854/49,888$), RedTeam Lateral Movement Recall $= 99.99\%$ ($4.64\text{ }\mu\text{s/event}$) | $361/1,741$ ($20.74\%$) KEV CVEs confirmed in ransomware campaigns |
| **Experiment 4** | Trace-Driven `DCEDM` Sensitivity Analysis | $138,152$ held-out `OTRF` + `SALAD-SOC` scores | Selective `DECS` ($\Pi_4$) cuts banner firing by $95.92\%$ (`OTRF`) and $79.87\%$ (`SALAD-SOC`) vs. Static Banners ($\Pi_3$) | Reduces expected operational loss on `SALAD-SOC` by $30.16\%$ to $60.77\%$ across $c_{\text{miss}}/c_V \in [5, 100]$ |

---

## Repository Structure

```text
.
├── configs/
│   ├── default.yaml                        # Canonical seed (20261014), thresholds, and DCEDM parameters
│   ├── pitsa_empirical_likelihoods.json    # Pre-fitted empirical LLR lookup tables
│   └── soc_itpi_24_benchmark.json          # 24-vignette SOC-ITPI-24 telemetry benchmark
├── docs/
│   └── architecture.md                     # DCEDM formal equations and PITSA/DECS specification
├── experiments/
│   ├── run_all_isj_experiments.py          # Full 4-part multi-corpus & decision-theoretic pipeline
│   ├── run_experiments.py                  # Quick verification entrypoint for precomputed summary
│   └── ablation_study.py                   # Feature-group ablation & adaptive LOLBin masking summary
├── results/
│   ├── dataset_verification_summary.json   # Provenance summary for the 4 public corpora
│   ├── quantitative_and_ablation_results_report.md
│   └── statistical_results_summary.json    # Complete held-out empirical & sensitivity metrics
├── scripts/
│   ├── build_benchmark.py                  # Builder for SOC-ITPI-24 stimulus artifacts
│   └── verify_datasets.py                  # Verifier for local raw dataset paths
├── src/
│   ├── __init__.py
│   ├── analysis.py                         # Binary classification and AUROC evaluation utilities
│   ├── pitsa.py                            # AdditivePITSAScorer (strictly partitioned on A_struct)
│   └── simulation.py                       # Trace-driven Bayesian decision-theoretic evaluator
└── tests/
    └── test_reproducibility.py             # Pytest suite verifying invariance, Theorem 1, and metrics
```

---

## Quick Start

```bash
pip install -r requirements.txt
pytest -q
python3 experiments/run_experiments.py
python3 experiments/ablation_study.py
```
