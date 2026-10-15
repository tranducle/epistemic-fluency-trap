from __future__ import annotations

import json
from typing import Any


def _coerce_payload(payload: dict[str, Any], parse_status: str) -> dict[str, Any]:
    disposition = str(payload.get("disposition", "uncertain")).strip().lower()
    if disposition not in {"escalate", "dismiss", "uncertain"}:
        disposition = "uncertain"
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = payload.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    return {
        "disposition": disposition,
        "confidence": confidence,
        "evidence": [str(item) for item in evidence],
        "rationale": str(payload.get("rationale", "")),
        "parse_status": parse_status,
    }


def parse_triage_response(raw_text: str) -> dict[str, Any]:
    candidate = str(raw_text).strip()
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return _coerce_payload(payload, "valid_json")
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(candidate[start : end + 1])
            if isinstance(payload, dict):
                return _coerce_payload(payload, "recovered_json")
        except json.JSONDecodeError:
            pass
    return {
        "disposition": "uncertain",
        "confidence": 0.0,
        "evidence": [],
        "rationale": candidate,
        "parse_status": "invalid",
    }
