# Reproducibility Guide

This guide provides exact end-to-end instructions to reproduce every computational, ablation, and statistical result reported in the manuscript.

## 1. Environment Requirements

- **Python Version:** Python 3.10+ (tested on Python 3.11 / 3.12)
- **Operating System:** macOS, Linux, or Windows (CPU-only; full execution takes under 5 seconds on a standard laptop CPU)
- **Dependencies:** Specified in `requirements.txt` (`numpy`, `scipy`, `pandas`, `statsmodels`, `pytest`, `pyyaml`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Deterministic Seeds and Hyperparameters

All stochastic operations use `numpy.random.default_rng` with fixed seeds recorded in `configs/default.yaml`:

- **Study 1 Primary Seed:** `20261014` ($N = 160$ analysts $\times 24$ trials $= 3,840$ observations)
- **Study 1 Multi-Seed Sensitivity Suite:** `20261014`, `20261015`, `20261016`, `20261017`, `20261018`
- **Study 1B Mechanism Ablation Seed:** `20261015` ($N = 120$ analysts $\times 12$ trials $= 1,440$ observations)
- **1,000-Alert Stress & Masking Suite Seed:** `20261014`
- **Bootstrap Mediation Resamples:** `5,000` resamples with seed `20261014`
- **`PITSA` / `DECS` Hyperparameters:**
  - Stage 2 activation threshold: $\tau_{\text{LLR}} = 0.50$ (corresponding to $D_{\text{PITSA}} = 0.5169$)
  - Logistic link midpoint and slope: $\mu_0 = 0.45$, $\gamma = 1.35$
  - Laplace/Jeffreys smoothing constant: $\epsilon = 10^{-4}$

## 3. Step-by-Step Execution Commands

### Step 3.1: Verify Benchmark and Precomputed Configs

```bash
python3 scripts/verify_datasets.py
python3 scripts/build_benchmark.py
```

### Step 3.2: Run the Full Multi-Method Evaluation Suite

```bash
python3 experiments/run_experiments.py --config configs/default.yaml
```

This command executes:
1. **Experiment 1 (`src/pitsa.py`):** Evaluates `PITSA` feature and partitioning ablations (`Full PITSA`, `AB-3 Unpartitioned`, `AB-4 NoLineage`, `AB-5 NoBinarySig`, `AB-6 NoNetTask`) on both `SOC-ITPI-24` and the 1,000-alert stress corpus, runs the $\tau_{\text{LLR}} \in [0.10, 1.10]$ threshold sweep (`AB-7`), and evaluates the adaptive structural masking curve $\rho_{\text{mask}} \in [0.00, 1.00]$ (`AB-8`). Writes `results/pitsa_stress_results.json`.
2. **Experiment 2 (`src/simulation.py`, `src/analysis.py`):** Generates the Study 1 $2 \times 2$ factorial dataset ($N = 160$, $3,840$ trials), fits cluster-robust linear and logistic models (`LMM`/`GLMM`), computes Holm-Bonferroni adjusted Welch $t$-tests (`H1`, `H2`, `H3a/b`, `H5`), runs Two One-Sided Tests (`TOST`) equivalence on concordant `R1a` alerts (`H3c`), runs 5,000-draw bootstrap mediation (`H4`), and checks 5-seed stability.
3. **Experiment 3 (`src/simulation.py`, `src/analysis.py`):** Generates and analyzes the Study 1B 4-arm mechanism ablation (`C2` Fluent Standard, `C2 + Static Warning`, `C2 + Stage-1 Anchor Only`, and `C4` Full `DECS`).
4. **Experiment 4 (`src/analysis.py`):** Verifies the Study 2 ($N = 20$ SOC practitioners) Gioia coding structure, saturation trajectory, and inter-coder agreement. Writes `results/study2_thematic_coding_matrix.json` and the combined `results/statistical_results_summary.json`.

To also export the trial-level `.csv` tables to `outputs/`:

```bash
python3 experiments/run_experiments.py --config configs/default.yaml --export-csv
```

### Step 3.3: Run Automated Parity Tests

```bash
pytest tests/
```

## 4. Manuscript Table-to-Code Parity Map

| Manuscript Table / Section | Script / Function | Output Artifact & JSON Key |
| :--- | :--- | :--- |
| **Table 3 (`PITSA` Ablations `AB-3` to `AB-6`, `AB-7`, `AB-8`)** | `src/pitsa.py::evaluate_pitsa_and_ablations` | `results/pitsa_stress_results.json` (`soc_itpi_24_ablation`, `stress_1000_ablation`, `tau_sweep_AB7`, `adaptive_masking_curve_AB8`) |
| **Table 4 (Study 1 Cell Means & SDs across `C1` to `C4`)** | `src/analysis.py::analyze_study1` | `results/statistical_results_summary.json` (`experiment_2_study1_main.cell_summaries`) |
| **Table 5 (Hypotheses `H1` to `H5` Inferential Tests)** | `src/analysis.py::analyze_study1` | `results/statistical_results_summary.json` (`experiment_2_study1_main.hypothesis_tests`) |
| **Table 6 (Study 1B 4-Arm Mechanism Ablation)** | `src/analysis.py::analyze_study1b` | `results/statistical_results_summary.json` (`experiment_3_study1b_ablation`) |
| **Table 7 (Study 2 Gioia Data Structure & Saturation)** | `src/analysis.py::evaluate_study2_qualitative` | `results/study2_thematic_coding_matrix.json` (`informants`, `aggregate_dimensions`) |
