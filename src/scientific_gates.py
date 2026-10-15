from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core import write_json


def _finite_metric_frame(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    numeric_columns = [
        column
        for column in [
            "precision",
            "recall",
            "f1",
            "fpr",
            "specificity",
            "balanced_accuracy",
            "auroc",
            "auprc",
        ]
        if column in frame.columns
    ]
    problems: list[str] = []
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            problems.append(column + " contains a missing or nonfinite value")
    return frame, problems


def _prediction_diversity(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=False)
    report: dict[str, Any] = {}
    for name in data.files:
        if name.endswith("scores"):
            values = np.asarray(data[name], dtype=np.float64)
            report[name] = {
                "rows": int(values.size),
                "unique_values": int(np.unique(values).size),
                "standard_deviation": float(np.std(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    return report


def _gate_report(
    gate_id: str,
    stage: str,
    verdict: str,
    artifacts: list[str],
    metrics: dict[str, Any],
    criteria: dict[str, Any],
    failed: list[str],
    next_step: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "stage": stage,
        "verdict": verdict,
        "input_artifacts": artifacts,
        "metrics": metrics,
        "pass_criteria": criteria,
        "failed_criteria": failed,
        "next_allowed_step": next_step if verdict in {"PASS", "PASS_WITH_WARNINGS"} else None,
        "fail_interpretation": interpretation,
    }


def run_g2_and_g3(paper_root: Path) -> dict[str, Any]:
    result_dir = paper_root / "results"
    gate_dir = paper_root / "verification"
    gate_dir.mkdir(parents=True, exist_ok=True)
    salad_path = result_dir / "salad_metrics_long.csv"
    lmd_path = result_dir / "lmd_metrics_long.csv"
    salad, salad_problems = _finite_metric_frame(salad_path)
    lmd, lmd_problems = _finite_metric_frame(lmd_path)
    failed = salad_problems + lmd_problems
    required_salad = {
        "PITSA_FullTrain",
        "PITSA_BudgetMatched",
        "StructuralLogistic",
        "StructuralRandomForest",
        "JointWordLogistic",
        "CharacterNgramLogistic",
        "JointNaiveBayes",
    }
    required_lmd = {
        "PITSA_Core",
        "CoreLogistic",
        "CoreRandomForest",
        "ContextOnlyLogistic",
        "ContextAugmentedLogistic",
    }
    missing_salad = sorted(required_salad - set(salad["model"].astype(str)))
    missing_lmd = sorted(required_lmd - set(lmd["model"].astype(str)))
    if missing_salad:
        failed.append("missing SALAD models: " + ", ".join(missing_salad))
    if missing_lmd:
        failed.append("missing LMD models: " + ", ".join(missing_lmd))
    stochastic_runs = pd.concat([salad, lmd], ignore_index=True)
    stochastic_runs = (
        stochastic_runs[stochastic_runs["seed"].astype(str) != "deterministic"]
        .groupby(["dataset", "split", "model", "condition"])
        .size()
        .to_dict()
    )
    for key, count in stochastic_runs.items():
        if int(count) < 3:
            failed.append("fewer than three stochastic runs for " + str(key))
    diversity = {}
    salad_prediction_path = paper_root / "outputs/salad/primary_predictions.npz"
    lmd_prediction_path = paper_root / "outputs/lmd/signature_disjoint_primary_predictions.npz"
    for label, path in [("SALAD", salad_prediction_path), ("LMD", lmd_prediction_path)]:
        diversity[label] = _prediction_diversity(path)
        for name, values in diversity[label].items():
            if values["unique_values"] < 3 or values["standard_deviation"] <= 0.0:
                failed.append(label + " prediction collapse in " + name)
    verdict = "PASS" if not failed else "FAIL_REPAIR"
    g2 = _gate_report(
        "G2_baseline_sanity",
        "baseline_sanity",
        verdict,
        [str(salad_path), str(lmd_path), str(salad_prediction_path), str(lmd_prediction_path)],
        {
            "salad_rows": int(len(salad)),
            "lmd_rows": int(len(lmd)),
            "stochastic_run_counts": {str(key): int(value) for key, value in stochastic_runs.items()},
            "prediction_diversity": diversity,
        },
        {
            "required_models_present": True,
            "numeric_metrics_finite": True,
            "stochastic_runs_per_configuration": 3,
            "prediction_unique_values_minimum": 3,
        },
        failed,
        "method_mechanism",
        "A model, metric, or prediction distribution is missing, degenerate, or not traceable.",
    )
    write_json(gate_dir / "baseline_sanity.json", g2)

    mechanism_failed: list[str] = []
    salad_npz = np.load(salad_prediction_path, allow_pickle=False)
    pitsa_clean = np.asarray(salad_npz["PITSA_FullTrain_clean_scores"])
    pitsa_attack = np.asarray(salad_npz["PITSA_FullTrain_attack_scores"])
    exact_difference = float(np.max(np.abs(pitsa_clean - pitsa_attack)))
    pitsa_row = salad[
        (salad["model"] == "PITSA_FullTrain")
        & (salad["condition"] == "clean")
    ].iloc[0]
    prevalence = float(pitsa_row["positive_rows"] / (pitsa_row["positive_rows"] + pitsa_row["negative_rows"]))
    if exact_difference != 0.0:
        mechanism_failed.append("PITSA score changed under text only attack")
    if float(pitsa_row["auprc"]) <= prevalence:
        mechanism_failed.append("PITSA AUPRC does not exceed class prevalence")
    if float(pitsa_row["balanced_accuracy"]) <= 0.5:
        mechanism_failed.append("PITSA balanced accuracy does not exceed chance")
    g3_verdict = "PASS" if not mechanism_failed and g2["verdict"] == "PASS" else "FAIL_REPAIR"
    g3 = _gate_report(
        "G3_method_mechanism",
        "method_mechanism",
        g3_verdict,
        [str(salad_prediction_path), str(salad_path)],
        {
            "maximum_text_attack_score_difference": exact_difference,
            "PITSA_AUPRC": float(pitsa_row["auprc"]),
            "test_prevalence": prevalence,
            "PITSA_balanced_accuracy": float(pitsa_row["balanced_accuracy"]),
            "PITSA_score_unique_values": int(np.unique(pitsa_clean).size),
        },
        {
            "text_only_score_difference": 0.0,
            "AUPRC_above_prevalence": True,
            "balanced_accuracy_above_chance": True,
            "G2_pass": True,
        },
        mechanism_failed,
        "ablation_validity",
        "The proposed prompt isolation mechanism is not invariant, discriminative, or supported by a sane baseline pipeline.",
    )
    write_json(gate_dir / "method_mechanism.json", g3)
    return {"G2": g2, "G3": g3}


def run_g4(paper_root: Path) -> dict[str, Any]:
    result_dir = paper_root / "results"
    gate_dir = paper_root / "verification"
    path = result_dir / "ablation_results.csv"
    frame, problems = _finite_metric_frame(path)
    failed = list(problems)
    for dataset in ["SALAD_SOC", "LMD_2023"]:
        subset = frame[frame["dataset"] == dataset]
        if "Full" not in set(subset["configuration"].astype(str)):
            failed.append(dataset + " full configuration missing")
        if not subset["configuration"].astype(str).str.startswith("Drop_").any():
            failed.append(dataset + " leave one pillar out configurations missing")
        if subset["auroc"].nunique() < 2 and subset["f1"].nunique() < 2:
            failed.append(dataset + " ablation outputs are invariant across configurations")
    verdict = "PASS" if not failed else "FAIL_REPAIR"
    report = _gate_report(
        "G4_ablation_validity",
        "ablation_validity",
        verdict,
        [str(path)],
        {
            "rows": int(len(frame)),
            "datasets": sorted(frame["dataset"].astype(str).unique().tolist()),
            "configurations": sorted(frame["configuration"].astype(str).unique().tolist()),
        },
        {
            "full_and_drop_configurations_present": True,
            "same_declared_split_and_tuning_rule": True,
            "at_least_one_metric_changes": True,
        },
        failed,
        "robustness_validity",
        "The ablation set is missing, degenerate, or changes more than the named mechanism.",
    )
    write_json(gate_dir / "ablation_validity.json", report)
    return report


def run_g5(paper_root: Path) -> dict[str, Any]:
    result_dir = paper_root / "results"
    gate_dir = paper_root / "verification"
    salad = json.loads((result_dir / "salad_results.json").read_text(encoding="utf-8"))
    lmd = json.loads((result_dir / "lmd_results.json").read_text(encoding="utf-8"))
    llm = json.loads((result_dir / "llm_results.json").read_text(encoding="utf-8"))
    failed: list[str] = []
    curves = salad.get("graded_attack_curves", {})
    for model in ["PITSA_FullTrain", "JointWordLogistic", "CharacterNgramLogistic"]:
        if len(curves.get(model, [])) < 5:
            failed.append("graded attack curve missing for " + model)
    for name in ["signature_disjoint", "temporal_future", "station_holdout"]:
        if name not in lmd.get("split_report", {}):
            failed.append("LMD robustness split missing: " + name)
    model_summaries = llm.get("model_summaries", [])
    if len(model_summaries) < 2:
        failed.append("fewer than two actual instruction models evaluated")
    for item in model_summaries:
        for condition in ["clean_no_cue", "attacked_no_cue", "attacked_with_cue"]:
            details = item.get("conditions", {}).get(condition, {})
            if float(details.get("parse_valid_rate", 0.0)) < 0.90:
                failed.append(str(item.get("model")) + " parse validity below 0.90 in " + condition)
    verdict = "PASS" if not failed else "FAIL_REPAIR"
    report = _gate_report(
        "G5_robustness_validity",
        "robustness_validity",
        verdict,
        [
            str(result_dir / "salad_results.json"),
            str(result_dir / "lmd_results.json"),
            str(result_dir / "llm_results.json"),
        ],
        {
            "graded_curve_models": sorted(curves),
            "LMD_split_views": sorted(lmd.get("split_report", {})),
            "actual_model_count": int(len(model_summaries)),
            "actual_model_summaries": model_summaries,
        },
        {
            "graded_attack_budgets_present": True,
            "source_time_station_or_signature_views_present": True,
            "actual_instruction_models_minimum": 2,
            "JSON_parse_valid_rate_minimum": 0.90,
        },
        failed,
        "result_integrity",
        "Robustness evidence is incomplete, unparsable, or not matched to declared stress budgets.",
    )
    write_json(gate_dir / "robustness_validity.json", report)
    return report


def run_g6(paper_root: Path) -> dict[str, Any]:
    result_dir = paper_root / "results"
    gate_dir = paper_root / "verification"
    metric_files = [
        result_dir / "salad_metrics_long.csv",
        result_dir / "lmd_metrics_long.csv",
        result_dir / "ablation_results.csv",
        result_dir / "llm_results_long.csv",
        result_dir / "governance_loss_sensitivity.csv",
    ]
    failed: list[str] = []
    rows: dict[str, int] = {}
    detection_required = [
        "threshold", "precision", "recall", "f1", "fpr", "specificity",
        "balanced_accuracy", "auroc", "auprc", "tp", "fp", "tn", "fn",
        "positive_rows", "negative_rows",
    ]
    governance_required = [
        "attack_prevalence", "miss_to_cue_cost_ratio", "gate_true_positive_rate",
        "gate_false_positive_rate", "unassisted_attack_miss_rate",
        "cued_attack_miss_rate", "unassisted_loss", "static_cue_loss",
        "selective_cue_loss", "selective_attack_miss_rate", "selective_cue_rate",
        "selective_loss_reduction_vs_unassisted_pct",
        "selective_loss_reduction_vs_static_pct",
    ]
    for path in metric_files:
        if not path.exists():
            failed.append("missing result file: " + path.name)
            continue
        frame = pd.read_csv(path)
        rows[path.name] = int(len(frame))
        if path.name in {"salad_metrics_long.csv", "lmd_metrics_long.csv", "ablation_results.csv"}:
            required = detection_required
            missing_columns = [column for column in required if column not in frame.columns]
            if missing_columns:
                failed.append(path.name + " missing required columns: " + ", ".join(missing_columns))
                continue
            for column in required:
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                    failed.append(path.name + " has missing or nonfinite required values in " + column)
            if path.name == "ablation_results.csv":
                optional_subclasses = [column for column in ["EoRS_recall", "EoHT_recall"] if column in frame.columns]
                for column in optional_subclasses:
                    observed = pd.to_numeric(frame[column], errors="coerce").dropna()
                    if not observed.empty and not np.isfinite(observed.to_numpy(dtype=float)).all():
                        failed.append(path.name + " has nonfinite observed subclass values in " + column)
        elif path.name == "llm_results_long.csv":
            required = ["elapsed_seconds", "parse_valid"]
            missing_columns = [column for column in required if column not in frame.columns]
            if missing_columns:
                failed.append(path.name + " missing required columns: " + ", ".join(missing_columns))
                continue
            elapsed = pd.to_numeric(frame["elapsed_seconds"], errors="coerce")
            if elapsed.isna().any() or not np.isfinite(elapsed.to_numpy(dtype=float)).all():
                failed.append(path.name + " has missing or nonfinite elapsed time")
            valid_mask = frame["parse_valid"].astype(str).str.lower().isin(["true", "1"])
            for column in ["confidence", "eval_count", "prompt_eval_count"]:
                values = pd.to_numeric(frame.loc[valid_mask, column], errors="coerce")
                if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                    failed.append(path.name + " has missing or nonfinite values for valid responses in " + column)
        elif path.name == "governance_loss_sensitivity.csv":
            missing_columns = [column for column in governance_required if column not in frame.columns]
            if missing_columns:
                failed.append(path.name + " missing required columns: " + ", ".join(missing_columns))
                continue
            for column in governance_required:
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                    failed.append(path.name + " has missing or nonfinite required values in " + column)
    salad_npz = paper_root / "outputs/salad/primary_predictions.npz"
    lmd_npz = paper_root / "outputs/lmd/signature_disjoint_primary_predictions.npz"
    for path in [salad_npz, lmd_npz]:
        if not path.exists():
            failed.append("missing prediction provenance: " + path.name)
    verdict = "PASS" if not failed else "FAIL_REPAIR"
    report = _gate_report(
        "G6_result_integrity",
        "result_integrity",
        verdict,
        [str(path) for path in metric_files] + [str(salad_npz), str(lmd_npz)],
        {"result_rows": rows, "prediction_files_present": salad_npz.exists() and lmd_npz.exists()},
        {
            "all_declared_result_files_present": True,
            "all_numeric_evidence_finite": True,
            "prediction_level_provenance_present": True,
        },
        failed,
        "visual_evidence",
        "A table or claim would depend on a missing, nonfinite, or untraceable result.",
    )
    write_json(gate_dir / "result_integrity.json", report)
    return report


def run_all_empirical_gates(paper_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result.update(run_g2_and_g3(paper_root))
    if (paper_root / "results/ablation_results.csv").exists():
        result["G4"] = run_g4(paper_root)
    if all((paper_root / "results" / name).exists() for name in ["salad_results.json", "lmd_results.json", "llm_results.json"]):
        result["G5"] = run_g5(paper_root)
    if all((paper_root / "results" / name).exists() for name in ["salad_metrics_long.csv", "lmd_metrics_long.csv", "ablation_results.csv", "llm_results_long.csv", "governance_loss_sensitivity.csv"]):
        result["G6"] = run_g6(paper_root)
    return result
