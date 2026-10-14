# Dataset Provenance and Zero-Leakage Preprocessing (`DATA.md`)

All empirical evaluations in this study are grounded in **four publicly available cybersecurity datasets totaling $2,238,434$ telemetry records ($671,531$ held-out test records) and $1,741$ CVE advisories**. No human-subject data or interview transcripts exist in this repository.

## 1. Public Datasets Evaluated

1. **OTRF Security-Datasets (`N = 43,723` Windows Sysmon/Security Events)**
   * **Source**: Open Threat Research Forge (`https://github.com/OTRF/Security-Datasets`), 63 `.zip` attack-replay archives under `datasets/atomic/` and `datasets/compound/`.
   * **Split**: Stratified 70/30 split (`seed = 20261014`): $30,606$ training events, $13,117$ held-out test events ($516$ hostile total, $155$ hostile in test).
   * **Partitioning**: 5 structural pillars ($\mathcal{A}_{\text{struct}}$: `p1_lineage`, `p2_target_access`, `p3_sig_integ`, `p4_net_port`, `p5_task_reg`); untrusted text channel ($\mathcal{A}_{\text{text}}$: `CommandLine`, `ProcessCommandLine`, `Message`).

2. **SALAD-SOC (`N = 416,781` Enterprise SIEM Alerts)**
   * **Source**: `salad_test.csv` ($88,570$ malicious, $328,211$ benign alerts across 9 MITRE ATT&CK tactics).
   * **Split**: Stratified 70/30 split (`seed = 20261014`): $291,746$ training alerts, $125,035$ held-out test alerts ($26,571$ malicious in test).
   * **Strict Zero-Leakage Protocol**: Excludes all post-triage and SIEM-rule metadata columns (`correlation_key`, `severity`, `confidence`, `priority_score`, `alert_type`, `mitre_tactic`, `mitre_technique`, `kill_chain_phase`, `attack_category`, `triage_decision`). Extracts 5 pure flow/burst structural pillars (`s1_proto_service`, `s2_flow_vol`, `s3_dur_seg`, `s4_burst`, `s5_target_rep`).

3. **LMD-2023 (`N = 1,777,930` Windows Sysmon Lateral Movement Records)**
   * **Source**: `LMD-2023 [1.75M Elements][Labelled]full-csv.csv` ($1,611,637$ Normal, $110,746$ standard Malicious, $55,547$ RedTeam lateral movement).
   * **Split**: Stratified 70/30 split (`seed = 20261014`): $1,244,551$ training rows, $533,379$ held-out test rows ($49,888$ hostile in test, including $16,664$ RedTeam).

4. **CISA Known Exploited Vulnerabilities (`N = 1,741` CVEs)**
   * **Source**: `known_exploited_vulnerabilities.json` ($361$ CVEs confirmed in ransomware campaigns).
