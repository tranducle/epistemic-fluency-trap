# Architecture Specification: DCEDM, PITSA, and DECS

This document specifies the formal architecture of the Dual-Cue Epistemic Defeasibility Model (`DCEDM`), the Prompt-Invariant Telemetry Surprisal Auditor (`PITSA`), and the Two-Stage Defeasible Epistemic Cueing System (`DECS`).

## 1. Dual-Cue Epistemic Defeasibility Model (DCEDM)

When an analyst inspects an alert $x_i = (\mathcal{A}_{\text{struct}}, \mathcal{A}_{\text{text}})$ alongside a generative copilot output $g_i = (\hat{y}_{\text{LLM}}, c_{\text{LLM}}, r_{\text{LLM}})$, `DCEDM` models decision making through two competing metacognitive pathways:

1. **System 1 Heuristic Pathway (Narrative Fluency):** High linguistic cohesion and causal completeness in $r_{\text{LLM}}$ elevate perceived epistemic sufficiency, lowering telemetry verification depth ($V_{\text{depth}}$, $V_{\text{dwell}}$) and inducing premature closure on contested alerts (`R1b` and `R3`).
2. **System 2 Analytical Pathway (Defeasible Contestation):** Combining Stage 1 mandatory pre-rationale structural anchoring with a conditional Stage 2 defeasible rebuttal banner triggered only when structural surprisal exceeds $\tau_{\text{LLR}} = 0.50$ recruits analytical scrutiny without inducing alert fatigue on concordant alerts (`R1a` and `R2`).

## 2. Prompt-Invariant Telemetry Surprisal Auditor (PITSA)

Every SIEM/EDR alert packet is partitioned at the ingestion boundary into two disjoint subsets:

$$\mathcal{A}(x_i) = \mathcal{A}_{\text{struct}}(x_i) \cup \mathcal{A}_{\text{text}}(x_i), \quad \mathcal{A}_{\text{struct}}(x_i) \cap \mathcal{A}_{\text{text}}(x_i) = \emptyset$$

- **Partitioned Structural Subset ($\mathcal{A}_{\text{struct}}$):** Immutable kernel, process-tree, and network header fields (`integrity_level`, `parent_child_pair`, `unsigned_binary_hash`, `external_dest_port`, `scheduled_task_mutation`).
- **Untrusted Textual Subset ($\mathcal{A}_{\text{text}}$):** User- or attacker-mutable strings (`CommandLine`, HTTP user-agent headers, registry string payloads, and ticket annotations).

Given the copilot's binary triage recommendation $\hat{y}_{\text{LLM}} \in \{0, 1\}$ and the opposing disposition $y_{\text{opp}} = 1 - \hat{y}_{\text{LLM}}$, `PITSA` evaluates each structural attribute $a_k \in \mathcal{A}_{\text{struct}}(x_i)$ using empirical class-conditional likelihood tables $P_0(a_k)$ and $P_1(a_k)$ estimated from historical Windows Sysmon and Security event logs (`configs/pitsa_empirical_likelihoods.json`):

$$\text{LLR}(a_k; \hat{y}_{\text{LLM}}) = \log \frac{P(a_k \mid Y = 1 - \hat{y}_{\text{LLM}}) + \epsilon}{P(a_k \mid Y = \hat{y}_{\text{LLM}}) + \epsilon}$$

The maximum structural counter-anchor is selected via canonical tie-breaking:

$$a_{\text{opp}}^* = \arg\max_{a_k \in \mathcal{A}_{\text{struct}}(x_i)} \text{LLR}(a_k; \hat{y}_{\text{LLM}}), \quad S_{\text{PITSA}}(x_i, \hat{y}_{\text{LLM}}) = \max_{a_k \in \mathcal{A}_{\text{struct}}(x_i)} \text{LLR}(a_k; \hat{y}_{\text{LLM}})$$

The raw surprisal score is mapped to a normalized divergence index $D_{\text{PITSA}} \in [0, 1]$ via a calibrated logistic link ($\mu_0 = 0.45$, $\gamma = 1.35$):

$$D_{\text{PITSA}}(x_i, \hat{y}_{\text{LLM}}) = \sigma\left(\gamma \cdot \left(S_{\text{PITSA}}(x_i, \hat{y}_{\text{LLM}}) - \mu_0\right)\right)$$

## 3. Two-Stage Defeasible Epistemic Cueing System (DECS)

- **Stage 1 (Pre-Rationale Structural Anchoring):** Before the generative rationale $r_{\text{LLM}}$ is rendered, the analyst must log one primary telemetry indicator from $\mathcal{A}_{\text{struct}}(x_i)$.
- **Stage 2 (Conditional Structural Rebuttal Banner):** If $S_{\text{PITSA}}(x_i, \hat{y}_{\text{LLM}}) > \tau_{\text{LLR}}$ ($\tau_{\text{LLR}} = 0.50$), `DECS` displays a Toulmin rebuttal banner juxtaposing the copilot's claim against the specific structural counter-anchor $a_{\text{opp}}^*$. When $S_{\text{PITSA}} \le \tau_{\text{LLR}}$, Stage 2 remains silent.
