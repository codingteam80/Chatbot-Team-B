from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.verified_answer_contract import enforce_verified_answer_contract


def _result(rule_id: str):
    return [{"metadata": {"section_type": "rule", "rule_id": rule_id}}]


def _check(name, condition, detail, failures):
    if condition:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}: {detail}")
        failures.append(name)


def main() -> int:
    failures = []

    answer, change = enforce_verified_answer_contract(
        answer="No.",
        question="Kung kumpleto ang switch cases pero walang default, compliant ba iyon sa MISRA?",
        results=_result("16.4"),
        yes_no_intent=True,
        require_reference=True,
    )
    _check("missing verified reference repaired", "Rule 16.4" in answer and answer.startswith("No."), answer, failures)
    _check("repair recorded", change.get("reason") == "verified_reference_missing_from_generated_answer", change, failures)

    answer2, change2 = enforce_verified_answer_contract(
        answer="Yes. 根据提供的上下文，适用的 MISRA 规则是 Rule 21.20。",
        question="May pointer mula sa strerror tapos tumawag ulit bago gamitin ang lumang pointer. Anong MISRA rule ang applicable?",
        results=_result("21.20"),
        yes_no_intent=False,
        require_reference=True,
    )
    _check("unrelated script removed", "根据" not in answer2 and "Rule 21.20" in answer2, answer2, failures)
    _check("script drift recorded", change2.get("reason") == "unexpected_language_script", change2, failures)

    original = "No. Rule 16.4 states that every switch statement shall have a default label."
    answer3, change3 = enforce_verified_answer_contract(
        answer=original,
        question="Is a switch without a default label acceptable under MISRA C?",
        results=_result("16.4"),
        yes_no_intent=True,
        require_reference=True,
    )
    _check("already compliant answer unchanged", answer3 == original and not change3, {"answer": answer3, "change": change3}, failures)

    answer4, change4 = enforce_verified_answer_contract(
        answer="The code needs review.",
        question="What does the code require?",
        results=[
            {"metadata": {"section_type": "rule", "rule_id": "1.1"}},
            {"metadata": {"section_type": "rule", "rule_id": "1.2"}},
        ],
        yes_no_intent=False,
        require_reference=True,
    )
    _check("multiple references are not guessed", answer4 == "The code needs review." and not change4, {"answer": answer4, "change": change4}, failures)

    payload = {
        "validation": "PASS" if not failures else "FAIL",
        "checks": 6,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
