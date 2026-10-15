# Research Highlights

**Manuscript Title:** *The Epistemic Fluency Trap in AI-Augmented Security Operations: Generative Rationale Fluency, Contested Evidence Cues, and Analyst Verification*  
**Target Journal:** *Journal of Management Information Systems* (*JMIS*, Taylor & Francis)

## Concise Highlights (<= 85 Characters Each)
- Formalizes the Epistemic Fluency Trap via Bayesian decision theory (DCEDM). (75 chars)
- Unpartitioned lexical, neural MLP, and subword LM scorers collapse under ITPI. (78 chars)
- Prompt-isolated PITSA guarantees zero recall loss across 2.24M real logs. (73 chars)
- Selective Two-Stage DECS cuts warning banners by 79.9% to 95.9% across streams. (79 chars)
- Exact cost-ratio crossovers ($\kappa^*$) and U-shaped habituation optima derived. (81 chars)

## Extended Executive Highlights
1. **Theory (DCEDM, Theorem 1 & Corollary 1):** Formalizes how generative narrative fluency ($\Delta F = 0.70$) shifts posterior compromise log-odds downward and inflates Compromise Miss Rates ($+14.97\text{ pp}$ on OTRF, $+26.02\text{ pp}$ on SALAD-SOC at $\lambda_F = 0.40$), while deriving exact cost-ratio dominance boundaries ($\kappa^* = 39.64$ on OTRF; $\kappa^* = 1.68$ on SALAD-SOC) and an interior U-shaped threshold optimum ($\tau^*_{\text{LLR}} = 1.0$) under unified endogenous warning habituation.
2. **Design Science Artifact (PITSA + Two-Stage DECS, DP1 to DP3):** Enforces an architectural projection $\Pi_{\text{struct}}$ ($\nabla_{\mathcal{A}_{\text{text}}} L(e) \equiv 0$) that isolates five structural telemetry pillars from untrusted text strings and extracts the top-contributing structural defeater pillar $k^*(e)$ in $2.12$ to $4.31\text{ }\mu\text{s/event}$ on a single CPU.
3. **Multi-Corpus Empirical Validation ($N = 2,238,434$ Real Telemetry Records):** Evaluates Partitioned PITSA against five unpartitioned baselines (Naive Bayes, Logistic Regression, Random Forest, Dense Semantic Neural MLP, and Subword LM) across OTRF (`43,723` events), SALAD-SOC (`416,781` alerts), LMD-2023 (`1,777,930` rows), and CISA KEV (`1,741` CVEs), proving $0.00\text{ pp}$ ITPI degradation ($\text{AUROC} = 0.9807$, $0.9380$, and $0.9963$).
