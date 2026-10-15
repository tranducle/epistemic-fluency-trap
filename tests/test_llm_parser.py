from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_itpi import parse_triage_response


def test_parse_triage_response_accepts_valid_json() -> None:
    raw = '{"disposition":"escalate","confidence":91,"evidence":["EventID 10"],"rationale":"LSASS access"}'
    parsed = parse_triage_response(raw)
    assert parsed["disposition"] == "escalate"
    assert parsed["confidence"] == 91.0
    assert parsed["parse_status"] == "valid_json"


def test_parse_triage_response_recovers_json_from_wrapped_text() -> None:
    raw = "Result follows: {\"disposition\":\"dismiss\",\"confidence\":80,\"evidence\":[],\"rationale\":\"routine\"}"
    parsed = parse_triage_response(raw)
    assert parsed["disposition"] == "dismiss"
    assert parsed["parse_status"] == "recovered_json"


def test_parse_triage_response_marks_unparseable_output() -> None:
    parsed = parse_triage_response("I cannot decide")
    assert parsed["disposition"] == "uncertain"
    assert parsed["parse_status"] == "invalid"
