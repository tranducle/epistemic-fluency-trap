# The Epistemic Fluency Trap in AI-Augmented Security Operations

This repository provides the reference implementation, benchmark configurations, and multi-method evaluation pipeline for the study **"The Epistemic Fluency Trap in AI-Augmented Security Operations: Generative Rationale Fluency, Contested Evidence Cues, and Analyst Verification."**

Modern Security Operations Centers (SOCs) increasingly integrate Large Language Model (LLM) copilots to summarize telemetry and recommend alert dispositions under strict service-level agreement (SLA) time pressure. When generative rationales are phrased in smooth, causally coherent prose, analysts experience an *epistemic fluency trap*: high processing fluency acts as a metacognitive cue of epistemic sufficiency, suppressing raw log verification and increasing vulnerability to Indirect Telemetry Prompt Injection (`ITPI`), where attackers embed benign administrative cover stories inside mutable log fields.

To counter this vulnerability without inducing alert fatigue, this repository implements the **Prompt-Invariant Telemetry Surprisal Auditor (`PITSA`)** and the **Two-Stage Defeasible Epistemic Cueing System (`DECS`)**, together with the 24-vignette **`SOC-ITPI-24`** benchmark, a 1,000-alert real-log stress suite, the Study 1 ($N = 160 \times 24 = 3,840$ trials) and Study 1B ($N = 120 \times 12 = 1,440$ trials) behavioral evaluation pipelines, and the Study 2 ($N = 20$ SOC practitioners) Gioia qualitative data structure verifier.

## Repository Structure

- `src/`: Reusable Python modules implementing `PITSA` scoring (`src/pitsa.py`), behavioral simulation (`src/simulation.py`), and statistical estimation (`src/analysis.py`).
- `experiments/`: Experiment runners for the full multi-method evaluation (`experiments/run_experiments.py`), mechanism ablations (`experiments/ablation_study.py`), and qualitative coding structure verification (`experiments/qualitative_coding.py`).
- `configs/`: Experiment parameters (`configs/default.yaml`), the 24-vignette `SOC-ITPI-24` benchmark (`configs/soc_itpi_24_benchmark.json`), and empirical Sysmon/Security log likelihood tables (`configs/pitsa_empirical_likelihoods.json`).
- `scripts/`: Dataset verification (`scripts/verify_datasets.py`) and benchmark construction utilities (`scripts/build_benchmark.py`).
- `results/`: Precomputed JSON statistical summaries and ablation outputs verified by the test suite.
- `tests/`: Automated `pytest` reproducibility and numerical parity tests (`tests/test_reproducibility.py`).
- `docs/`: Technical specification of the `DCEDM`, `PITSA`, and `DECS` architecture (`docs/architecture.md`).
- `DATA.md`: Upstream public dataset sources, schemas, and dual-use safety protocols.
- `REPRODUCIBILITY.md`: End-to-end execution instructions, deterministic seeds, and expected metrics.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Reproducibility Pipeline

```bash
# Step 1: Run automated unit and numerical parity tests
pytest tests/

# Step 2: Execute the full multi-method evaluation pipeline
python3 experiments/run_experiments.py --config configs/default.yaml

# Step 3: Run focused PITSA/DECS ablations or Study 2 qualitative verification
python3 experiments/ablation_study.py
python3 experiments/qualitative_coding.py
```

## Key Verified Findings

| Evaluation Component | Primary Metric | Verified Result |
| :--- | :--- | :--- |
| **Algorithm 1 (`PITSA`) on `SOC-ITPI-24`** | Contested (`R1b` + `R3`) Recall / `R1a` False Friction | `100.0%` (`8/8`) / `0.0%` (`0/8`) |
| **Algorithm 1 (`PITSA`) on 1,000-Alert Corpus** | Contestation F1 / False Friction Rate | `0.9562` (`95.50%` recall, `95.74%` precision) / `2.83%` |
| **Study 1 (`H1`: Verification Depth $V_{\text{depth}}$)** | `C2` (Fluent) vs. `C1` (Disfluent) | `0.3291` vs. `0.5445` ($d = -1.62, p_{\text{Holm}} < 0.0001$) |
| **Study 1 (`H2`: Contested Override Accuracy)** | `C2` (Fluent) vs. `C1` (Disfluent) | `29.38%` vs. `53.44%` ($d = -1.18, p_{\text{Holm}} < 0.0001$) |
| **Study 1 (`H3a/b`: Net Override Calibration `OC`)** | `C4` (Fluent + `DECS`) vs. `C2` (Fluent Standard) | `0.6297` vs. `0.2313` ($d = +1.64, p_{\text{Holm}} < 0.0001$) |
| **Study 1 (`H3c`: Concordant `R1a` TOST Equivalence)** | `DECS` vs. Standard ($\pm 5.0\text{ pp}$ bounds) | `91.41%` vs. `92.97%` ($p_{\text{TOST}} = 0.0123$) |
| **Study 1 (`H4`: Bootstrap Mediation via $V_{\text{depth}}$)** | Indirect Effect $a \times b$ (5,000 resamples) | `-0.1752` (`72.80%` of total effect mediated) |
| **Study 1 (`H5`: Self-Accountability $A_{\text{self}}$)** | `C4` (Fluent + `DECS`) vs. `C2` (Fluent Standard) | `5.60` vs. `3.50` ($d = +2.17, p_{\text{Holm}} < 0.0001$) |
| **Study 2 (Practitioner Qualitative Triangulation)** | Inter-Coder Reliability ($\kappa$) / Saturation | $\kappa = 0.88$ across $N = 20$ / Saturation at `I-15` |

## Data Access and Dual-Use Safety

See [`DATA.md`](DATA.md) for upstream repository links (`OTRF Security-Datasets`, `LMD-2023`, `SALAD`, and `CISA KEV`), input schemas, and dual-use sanitization rules. All `ITPI` strings in `configs/soc_itpi_24_benchmark.json` use non-executable administrative maintenance strings and RFC 5737 / RFC 1918 documentation IP ranges.

## License

This project is licensed under the [MIT License](LICENSE).
