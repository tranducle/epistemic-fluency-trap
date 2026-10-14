#!/usr/bin/env python3
"""Runner for Study 2 Gioia qualitative coding structure and thematic saturation verification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis import evaluate_study2_qualitative


def main() -> int:
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out = evaluate_study2_qualitative()
    with open(results_dir / "study2_thematic_coding_matrix.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(
        f"Study 2 Gioia structure verified: N={out['n_informants']} informants, "
        f"{out['total_first_order_codes']} first-order codes, kappa={out['overall_cohens_kappa']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
