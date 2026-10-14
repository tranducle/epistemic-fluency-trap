# Reproducibility Guide (`REPRODUCIBILITY.md`)

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Automated Verification (`pytest`)

To verify mathematical text-channel invariance (`Proposition 1`), `DCEDM` risk ordering (`Theorem 1`), and exact numerical consistency with `results/statistical_results_summary.json`:

```bash
pytest -v
python3 experiments/run_experiments.py
python3 experiments/ablation_study.py
```

## 3. Full End-to-End Multi-Corpus Re-Execution (`2,238,434` Records)

When the four public raw datasets (`OTRF Security-Datasets`, `SALAD-SOC`, `LMD-2023`, and `CISA KEV`) are placed under `datasets/raw/`, execute:

```bash
python3 experiments/run_all_isj_experiments.py
```

Total single-CPU runtime is approximately `44 seconds` (random seed `20261014`).
