# Architectural Specification: `DCEDM`, `PITSA`, and Two-Stage `DECS`

## 1. Telemetry Decomposition and Indirect Telemetry Prompt Injection (`ITPI`)

Each SOC alert $e$ consists of a composite telemetry tuple $\mathcal{A}(e) = (\mathcal{A}_{\text{struct}}(e), \mathcal{A}_{\text{text}}(e))$:
* **Trusted Structural Channel ($\mathcal{A}_{\text{struct}}(e)$)**: Kernel-enforced process creation and handle-access records (`Sysmon Event IDs 1, 3, 7, 8, 10`), binary cryptographic signatures, and network flow headers.
* **Untrusted Text Channel ($\mathcal{A}_{\text{text}}(e)$)**: Attacker-modifiable user-space strings (`CommandLine`, `ProcessCommandLine`, `alert_description`).

## 2. Prompt-Isolated Telemetry Discrepancy Scorer (`PITSA`)

`PITSA` applies the strict projection $\Pi_{\text{struct}}(\mathcal{A}(e)) = \mathcal{A}_{\text{struct}}(e) = (a_1(e), \dots, a_5(e))$ and computes the additive Laplace-smoothed log-likelihood ratio:

$$L(e) = \sum_{k=1}^{5} \ln \frac{\hat{P}_{\text{train}}(a_k(e) \mid y = 1, \epsilon)}{\hat{P}_{\text{train}}(a_k(e) \mid y = 0, \epsilon)}$$

Because $\nabla_{\mathcal{A}_{\text{text}}} L(e) \equiv 0$, `PITSA` is mathematically invariant to arbitrary prompt-injection payloads in $\mathcal{A}_{\text{text}}(e)$.

## 3. Two-Stage Defeasible Epistemic Cueing System (`DECS`)

* **Stage 1 (Pre-Rationale Structural Anchor $\mathcal{H}_1$)**: Displays the 5-pillar structural telemetry summary prior to the generative narrative.
* **Stage 2 (Conditional Toulmin Rebuttal Cue $\mathcal{R}_2$)**: Fires a high-contrast rebuttal alert naming the top-contributing structural pillar $k^*(e) = \arg\max_k \ell_k(a_k(e))$ strictly when $\hat{y}_{\text{LLM}}(e) = 0$ and $L(e) > \tau_{\text{LLR}}$.
