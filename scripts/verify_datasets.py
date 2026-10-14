#!/usr/bin/env python3
"""Verify local raw dataset extracts (OTRF, SALAD, LMD-2023, CISA KEV) when downloaded."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def find_files(root_dir: Path, ext: str) -> list[str]:
    matches = []
    if not root_dir.exists():
        return matches
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(ext.lower()):
                matches.append(os.path.join(r, fn))
    return sorted(matches)


def inspect_cisa_kev(base_dir: Path) -> dict:
    kev_json_path = base_dir / "cisa_kev" / "known_exploited_vulnerabilities.json"
    if not kev_json_path.exists():
        return {"status": "OPTIONAL_RAW_EXTRACT_NOT_PRESENT", "expected_path": "data/raw/cisa_kev"}
    with open(kev_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vulns = data.get("vulnerabilities", [])
    vendors = Counter(v.get("vendorProject", "Unknown") for v in vulns)
    ransomware = Counter(v.get("knownRansomwareCampaignUse", "Unknown") for v in vulns)
    return {
        "status": "USABLE_VERIFIED",
        "json_size_bytes": os.path.getsize(kev_json_path),
        "catalog_version": data.get("catalogVersion"),
        "date_released": data.get("dateReleased"),
        "total_vulnerabilities": len(vulns),
        "top_5_vendors": vendors.most_common(5),
        "ransomware_campaign_breakdown": dict(ransomware),
    }


def inspect_salad_soc(base_dir: Path) -> dict:
    csv_path = base_dir / "salad_soc" / "salad_test.csv"
    if not csv_path.exists():
        return {"status": "OPTIONAL_RAW_EXTRACT_NOT_PRESENT", "expected_path": "data/raw/salad_soc"}
    row_count = 0
    label_counts: Counter = Counter()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for row in reader:
            row_count += 1
            label_counts[row.get("is_malicious", "")] += 1
    return {
        "status": "USABLE_VERIFIED",
        "verified_test_split_rows": row_count,
        "column_count": len(columns),
        "is_malicious_distribution": dict(label_counts),
    }


def inspect_lmd_2023(base_dir: Path) -> dict:
    lmd_root = base_dir / "lmd_2023"
    csv_files = find_files(lmd_root, ".csv")
    if not csv_files:
        return {"status": "OPTIONAL_RAW_EXTRACT_NOT_PRESENT", "expected_path": "data/raw/lmd_2023"}
    total_rows = 0
    event_ids: Counter = Counter()
    for p in csv_files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                eid = row.get("EventID") or row.get("System.EventID") or "Unknown"
                event_ids[str(eid).strip()] += 1
    return {
        "status": "USABLE_VERIFIED",
        "csv_files_count": len(csv_files),
        "total_verified_csv_rows": total_rows,
        "sysmon_event_id_distribution": dict(event_ids.most_common(15)),
    }


def inspect_otrf(base_dir: Path) -> dict:
    otrf_root = base_dir / "otrf_security_datasets" / "repo" / "datasets"
    zip_files = find_files(otrf_root, ".zip")
    if not zip_files:
        return {
            "status": "OPTIONAL_RAW_EXTRACT_NOT_PRESENT",
            "expected_path": "data/raw/otrf_security_datasets",
        }
    return {
        "status": "USABLE_VERIFIED",
        "total_zip_bundles_downloaded": len(zip_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify raw cybersecurity datasets and precomputed benchmark artifacts."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw",
        help="Directory containing downloaded raw datasets (default: data/raw)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT_DIR / "results" / "dataset_verification_summary.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    benchmark_path = ROOT_DIR / "configs" / "soc_itpi_24_benchmark.json"
    likelihoods_path = ROOT_DIR / "configs" / "pitsa_empirical_likelihoods.json"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bm = json.load(f)
    with open(likelihoods_path, "r", encoding="utf-8") as f:
        lk = json.load(f)

    summary = {
        "precomputed_artifacts": {
            "soc_itpi_24_benchmark_trials": bm.get("total_trials"),
            "soc_itpi_24_regimes": bm.get("regime_counts"),
            "pitsa_benign_events_count": lk.get("benign_events_count"),
            "pitsa_malicious_events_count": lk.get("malicious_events_count"),
        },
        "raw_extract_checks": {
            "cisa_kev": inspect_cisa_kev(args.raw_dir),
            "salad_soc": inspect_salad_soc(args.raw_dir),
            "lmd_2023": inspect_lmd_2023(args.raw_dir),
            "otrf_security_datasets": inspect_otrf(args.raw_dir),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
