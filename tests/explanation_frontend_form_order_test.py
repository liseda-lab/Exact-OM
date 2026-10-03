"""The frontend's legacy questionnaire order must equal the frozen v1 declaration order."""

import json
import re
from pathlib import Path

from exact_inspect.study import forms

TABLE = Path(__file__).resolve().parents[1] / "explanations_visualizer/src/study/legacyFormOrder.ts"


def declared_order():
    """The Python declaration order that canonical sorted-key storage loses."""
    definition = forms.definitions()
    order = {}
    for group in ("background", "consultation", "final"):
        for question in definition[group]:
            entry = {}
            if question["options"]:
                entry["options"] = list(question["options"])
            if question["matrix"]:
                entry["rows"] = list(question["matrix"])
            if entry:
                order[question["id"]] = entry
    return {definition["version"]: order}


def frontend_order():
    text = TABLE.read_text(encoding="utf-8")
    match = re.search(r"/\* json \*/ (\{.*\});\s*$", text, re.S)
    assert match, "legacyFormOrder.ts must keep its JSON object literal"
    return json.loads(match.group(1))


def test_legacy_form_order_matches_frozen_declaration():
    assert frontend_order() == declared_order()


def test_ordinal_scales_are_not_alphabetical():
    order = frontend_order()["exact-study-forms/1"]
    assert order["component_usefulness"]["options"] == [
        "not_helpful",
        "slightly",
        "moderately",
        "very",
        "cannot_judge",
    ]
    assert order["mental_effort"]["options"][0] == "very_low"
