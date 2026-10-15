from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from audit_splits import run_split_audits
from governance_model import run_governance_model
from llm_pipeline import run_llm_pipeline
from lmd_pipeline import run_lmd_pipeline
from otrf_stimuli import build_otrf_stimuli
from salad_pipeline import run_salad_pipeline


def load_config() -> dict:
    path = ROOT / "configs/experiment_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducibility experiment pipeline")
    parser.add_argument(
        "stage",
        choices=["splits", "salad", "lmd", "stimuli", "llm", "governance", "all"],
    )
    parser.add_argument("llm_limit", nargs="?", type=int)
    parser.add_argument("llm_models", nargs="?")
    args = parser.parse_args()
    config = load_config()
    output: dict[str, object] = {}
    if args.stage in {"splits", "all"}:
        output["splits"] = run_split_audits(ROOT)
    if args.stage in {"salad", "all"}:
        output["salad"] = run_salad_pipeline(ROOT, config)
    if args.stage in {"lmd", "all"}:
        output["lmd"] = run_lmd_pipeline(ROOT, config)
    if args.stage in {"stimuli", "all"}:
        output["stimuli"] = build_otrf_stimuli(ROOT, int(config["llm_stimulus_target"]))
    if args.stage in {"llm", "all"}:
        output["llm"] = run_llm_pipeline(
            ROOT,
            config,
            models=args.llm_models.split(",") if args.llm_models else None,
            limit=args.llm_limit,
        )
    if args.stage in {"governance", "all"}:
        output["governance"] = run_governance_model(ROOT)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
