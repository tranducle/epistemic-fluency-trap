# Quantitative Empirical & Trace-Driven Decision-Theoretic Results Report

## 1. Scientific Integrity & Corpus Provenance
- **Zero Human Participants / Zero Simulated Surveys**: This evaluation contains no human-subject trials, no synthetic participant cohorts, and no qualitative interviews.
- **100% Real Telemetry Corpora Evaluated**:
  1. **OTRF Security-Datasets**: `43,723` parsed Windows Sysmon/Security events across 63 attack archives (`30,606` train / `13,117` held-out test).
  2. **SALAD-SOC SIEM Corpus**: `416,781` real labeled SOC alerts (`291,746` train / `125,035` held-out test).
  3. **LMD-2023 Lateral Movement Sysmon Corpus**: `1,777,930` real Windows Sysmon rows (`1,244,551` train / `533,379` held-out test).
  4. **CISA KEV Catalog**: `1,741` actively exploited CVEs (`361` known ransomware campaign CVEs).

## 2. Experiment 1: OTRF Security-Datasets Held-Out Evaluation (`N_test = 13,117`)
- **PITSA Inference Latency**: `43.484 ms` total (`0.00332 ms/event`).

| Model | Clean Precision | Clean Recall | Clean F1 | Clean FPR | Clean AUROC | ITPI Recall | ITPI F1 | ITPI AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `PITSA_Partitioned` | 0.2374 | 0.8194 | 0.3681 | 0.0315 | 0.9807 | 0.8194 | 0.3681 | 0.9807 |
| `NaiveBayes_Unpartitioned` | 0.0688 | 0.7419 | 0.1259 | 0.1201 | 0.9195 | 0.0000 | 0.0000 | 0.7267 |
| `LogisticRegression_Unpartitioned` | 1.0000 | 0.8581 | 0.9236 | 0.0000 | 0.9956 | 0.2516 | 0.4021 | 0.9373 |
| `RandomForest_Unpartitioned` | 0.9773 | 0.5548 | 0.7078 | 0.0002 | 0.9940 | 0.0000 | 0.0000 | 0.9862 |
| `DenseNeuralMLP_Unpartitioned` | 0.9027 | 0.6581 | 0.7612 | 0.0008 | 0.9893 | 0.0000 | 0.0000 | 0.7430 |
| `AutoregressiveSubwordLM_Unpartitioned` | 1.0000 | 0.1484 | 0.2584 | 0.0000 | 0.9267 | 0.0000 | 0.0000 | 0.9568 |

## 3. Experiment 2: SALAD-SOC Held-Out Evaluation (`N_test = 125,035`)
- **PITSA Inference Latency**: `377.659 ms` total (`0.00302 ms/alert`).

| Model | Clean Precision | Clean Recall | Clean F1 | Clean FPR | Clean AUROC | ITPI Recall | ITPI F1 | ITPI AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `PITSA_Partitioned` | 0.7250 | 0.6868 | 0.7054 | 0.0703 | 0.9380 | 0.6868 | 0.7054 | 0.9380 |
| `NaiveBayes_Unpartitioned` | 0.7119 | 0.7865 | 0.7473 | 0.0859 | 0.9395 | 0.2939 | 0.3646 | 0.8453 |
| `LogisticRegression_Unpartitioned` | 0.7824 | 0.8087 | 0.7954 | 0.0607 | 0.9653 | 0.0292 | 0.0466 | 0.6643 |
| `RandomForest_Unpartitioned` | 0.7824 | 0.8437 | 0.8119 | 0.0633 | 0.9690 | 0.3605 | 0.4520 | 0.9278 |
| `DenseNeuralMLP_Unpartitioned` | 0.7738 | 0.8493 | 0.8098 | 0.0670 | 0.9686 | 0.2753 | 0.3614 | 0.8784 |
| `AutoregressiveSubwordLM_Unpartitioned` | 0.7911 | 0.7746 | 0.7828 | 0.0552 | 0.9581 | 0.0403 | 0.0648 | 0.7869 |

## 4. Experiment 3: LMD-2023 Held-Out Evaluation (`N_test = 533,379`)
- **PITSA AUROC**: `0.9963`
- **Overall Hostile Recall (`tau = 0.50`)**: `0.9993`
- **RedTeam Lateral Movement Recall (`tau = 0.50`)**: `0.9999`
- **False Positive Rate (`tau = 0.50`)**: `0.0468`
- **PITSA Inference Latency**: `4.312 us/event`

## 5. Experiment 4: Trace-Driven Analytical Decision-Theoretic Sensitivity Analysis (`DCEDM` & `DECS`)
- Driven directly by held-out empirical PITSA LLR distributions on OTRF and SALAD-SOC.
