# Data Documentation and Access Protocols

This document describes the benchmark artifacts bundled in this repository, the four upstream public cybersecurity repositories from which they are derived, the JSON data schemas, and the dual-use and human-participant governance protocols.

## 1. Bundled Benchmark and Likelihood Artifacts

Two self-contained JSON artifacts are included under `configs/` so that the entire computational and statistical evaluation pipeline can run immediately without downloading multi-gigabyte raw log archives:

1. **`configs/soc_itpi_24_benchmark.json` (`SOC-ITPI-24` Stimulus Benchmark, v2.0):**
   - Contains 24 balanced alert vignettes organized into two sequential 12-trial operational shift blocks (Block 1: Trials 1 to 12; Block 2: Trials 13 to 24).
   - Each block contains 4 concordant benign alerts (`R1a`), 1 non-adversarial false-positive contested alert (`R1b`), 4 concordant malicious alerts (`R2`), and 3 adversarial Indirect Telemetry Prompt Injection alerts (`R3`).
   - Each vignette record includes:
     - `trial_id`, `block`, `regime`, `y_true`, `y_hat_LLM`, `llm_confidence_G2`
     - `raw_telemetry_packet`: Windows Sysmon / Security event fields (`EventID`, `Hostname`, `Image`, `ParentImage`, `TargetImage`, `GrantedAccess`, `Signed`, `IntegrityLevel`, `DestinationPort`, `CommandLine`, and partitioned `structured_attrs`)
     - `G1_struct_display`: Disfluent bulleted telemetry key-value rationale
     - `G1_fluent_display`: High-fluency narrative prose rationale
     - `PITSA_counter_anchor`: Precomputed `y_opp`, `a_opp_star`, `llr_score`, and `all_candidate_llrs`

2. **`configs/pitsa_empirical_likelihoods.json` (`PITSA` Empirical Likelihood Tables):**
   - Contains the class-conditional structural attribute probabilities $P_0(a_k)$ (`P_0_benign_empirical_probs`, estimated across $33,434$ benign background events) and $P_1(a_k)$ (`P_1_malicious_empirical_probs`, estimated across $275$ attack events) extracted from the Open Threat Research Forge (`OTRF`) Windows Sysmon and Security event corpus.

## 2. Upstream Public Cybersecurity Datasets

Researchers who wish to inspect or rebuild the raw upstream extracts using `scripts/verify_datasets.py` and `scripts/build_benchmark.py` can obtain the four public datasets from their official providers and place them under `data/raw/`:

| Dataset | Official Source / Identifier | Role in Pipeline | Expected Local Path |
| :--- | :--- | :--- | :--- |
| **OTRF Security-Datasets** | [https://github.com/OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets) | Primary source of Windows Sysmon/Security host logs (`T1003` Credential Access, `T1021` Lateral Movement, `T1053` Persistence) and empirical likelihood estimation ($33,709$ parsed events across 60 archives). | `data/raw/otrf_security_datasets/repo/datasets/` |
| **LMD-2023** | [https://doi.org/10.5281/zenodo.7687817](https://doi.org/10.5281/zenodo.7687817) | Enterprise Windows Sysmon lateral movement benchmark ($78,367$ verified CSV rows across normal, red-team, and malware executions) used to cross-validate structural attribute distributions. | `data/raw/lmd_2023/` |
| **SALAD** | [https://github.com/nutalk/salad](https://github.com/nutalk/salad) (`arXiv:2511.06952`) | SOC Alert & Log Analysis Dataset ($7,939$ test-split alerts; $952,668$ full-corpus alerts) providing realistic triage decision, severity, and MITRE ATT&CK tactic distributions. | `data/raw/salad_soc/` |
| **CISA KEV Catalog** | [https://www.cisa.gov/known-exploited-vulnerabilities-catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) | Authoritative catalog of $1,568$ known exploited CVEs used to anchor realistic CVE and patch-ticket references in non-adversarial (`R1b`) and adversarial (`R3`) alert vignettes. | `data/raw/cisa_kev/` |

## 3. Generating Optional Trial-Level CSV Exports

In accordance with clean repository standards, large generated `.csv` files are excluded from version control via `.gitignore` and can be regenerated deterministically at any time by passing `--export-csv` to `experiments/run_experiments.py`:

```bash
python3 experiments/run_experiments.py --export-csv
```

This writes `outputs/study1_trial_level_data.csv` ($3,840$ rows) and `outputs/study1b_ablation_data.csv` ($1,440$ rows).

## 4. Dual-Use Safety and Human-Participant Governance

- **Cybersecurity Dual-Use Sanitization:** All `ITPI` strings in `configs/soc_itpi_24_benchmark.json` consist exclusively of non-weaponizable benign administrative cover text (synthetic change-control ticket tags and audit notes). All hostnames, usernames, and network addresses are mapped to RFC 5737 (`192.0.2.0/24`, `198.51.100.0/24`) and RFC 1918 documentation ranges. No live exploit code or malware binaries are distributed.
- **Study 2 Practitioner Anonymity:** Raw audio recordings and full verbatim interview transcripts from Study 2 ($N = 20$ SOC practitioners) are not publicly redistributed under participant confidentiality agreements. The complete anonymized Gioia coding structure, informant demographics, new-code saturation counts, and inter-coder reliability statistics are provided in `results/study2_thematic_coding_matrix.json`.
