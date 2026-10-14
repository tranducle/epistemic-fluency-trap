#!/usr/bin/env python3
"""
Prompt-Isolated Telemetry Discrepancy Scorer (PITSA).

Implements the strictly partitioned additive Laplace-smoothed log-likelihood
ratio (LLR) scorer over structured kernel and network headers (A_struct),
completely excluding untrusted text fields (A_text) to ensure mathematical
invariance against Indirect Telemetry Prompt Injection (ITPI).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np


class AdditivePITSAScorer:
    """
    Empirical Laplace-smoothed Naive Bayes log-likelihood ratio (LLR) scorer
    fitted strictly on structured telemetry pillars (A_struct).
    """

    def __init__(self, smoothing: float = 1e-4) -> None:
        self.smoothing = float(smoothing)
        self.feature_names: List[str] = []
        self.llr_tables: Dict[str, Dict[str, float]] = {}
        self.benign_modes: Dict[str, str] = {}
        self.prior_log_odds: float = 0.0

    def fit(self, records: Sequence[Dict[str, str]], labels: Sequence[int], feature_names: Sequence[str]) -> "AdditivePITSAScorer":
        self.feature_names = list(feature_names)
        y = np.asarray(labels, dtype=int)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        self.prior_log_odds = float(np.log((n_pos + 1.0) / (n_neg + 1.0)))

        for f in self.feature_names:
            pos_counts: Counter[str] = Counter()
            neg_counts: Counter[str] = Counter()
            vocab = set()
            for rec, lbl in zip(records, y):
                val = str(rec.get(f, "MISSING"))
                vocab.add(val)
                if lbl == 1:
                    pos_counts[val] += 1
                else:
                    neg_counts[val] += 1
            self.benign_modes[f] = neg_counts.most_common(1)[0][0] if neg_counts else "MISSING"
            v_size = max(len(vocab), 1)
            table: Dict[str, float] = {}
            for val in vocab:
                p_pos = (pos_counts[val] + self.smoothing * n_pos) / (n_pos * (1.0 + self.smoothing * v_size))
                p_neg = (neg_counts[val] + self.smoothing * n_neg) / (n_neg * (1.0 + self.smoothing * v_size))
                table[val] = float(np.log(p_pos / p_neg))
            self.llr_tables[f] = table
        return self

    def score_records(
        self,
        records: Sequence[Dict[str, str]],
        active_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        feats = list(active_features) if active_features is not None else self.feature_names
        scores = np.zeros(len(records), dtype=float)
        for i, rec in enumerate(records):
            s = 0.0
            for f in feats:
                val = str(rec.get(f, "MISSING"))
                s += self.llr_tables[f].get(val, 0.0)
            scores[i] = s
        return scores

    def explain_event(self, record: Dict[str, str], threshold: float = 4.0) -> Tuple[float, int, str]:
        """Return (total_llr, contestation_flag, primary_defeater_pillar)."""
        best_pillar = self.feature_names[0] if self.feature_names else "none"
        best_val = -float("inf")
        total = 0.0
        for f in self.feature_names:
            val = str(record.get(f, "MISSING"))
            llr_k = self.llr_tables[f].get(val, 0.0)
            total += llr_k
            if llr_k > best_val:
                best_val = llr_k
                best_pillar = f
        flag = int(total > threshold)
        return float(total), flag, best_pillar
