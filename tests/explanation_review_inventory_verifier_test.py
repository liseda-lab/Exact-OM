"""Full-index verification must prove count, identity and cursor completeness."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.verify_review_environment import Journey, VerificationError, environment, inventory


def pages():
    return [
        {
            "items": [
                {
                    "entity": {"ontology_version_id": "one", "iri": "urn:a", "kind": "class"},
                    "preferred_label": {"text": "A"},
                }
            ],
            "returned_count": 1,
            "total_count": 2,
            "next_cursor": "second",
        },
        {
            "items": [
                {
                    "entity": {"ontology_version_id": "one", "iri": "urn:b", "kind": "class"},
                    "preferred_label": {"text": "B"},
                }
            ],
            "returned_count": 1,
            "total_count": 2,
            "next_cursor": None,
        },
    ]


def test_full_inventory_exhaustion_is_deterministic_and_retains_only_sample_identities():
    requested = []

    def fetch(cursor):
        requested.append(cursor)
        return pages()[cursor is not None]

    result, samples = inventory(fetch)
    assert result["entities"] == result["pages"] == 2
    assert requested == [None, "second"]
    assert samples == [p["items"][0]["entity"] for p in pages()]
    assert inventory(lambda cursor: pages()[cursor is not None])[0] == result
    changed = pages()
    changed[1]["items"][0]["preferred_label"]["text"] = "Changed"
    assert inventory(lambda cursor: changed[cursor is not None])[0]["sha256"] != result["sha256"]


@pytest.mark.parametrize(
    "change,label",
    [
        (lambda p: p[1].update(total_count=3), "inventory_total_stable"),
        (lambda p: p[1].update(items=deepcopy(p[0]["items"])), "inventory_unique_typed_entities"),
        (lambda p: p[0].update(returned_count=2), "inventory_page_count"),
        (lambda p: p[0].update(next_cursor=None), "inventory_exhaustive_count"),
        (lambda p: p[1].update(next_cursor="second"), "inventory_cursor_progress"),
        (lambda p: p[0].update(total_count=None), "inventory_exact_total_available"),
    ],
)
def test_partial_duplicate_or_unstable_inventory_cannot_pass(change, label):
    saved = pages()
    change(saved)
    with pytest.raises(VerificationError, match=label):
        inventory(lambda cursor: saved[cursor is not None])


@pytest.mark.parametrize("defect", [None, "empty", "count", "parity", "presence"])
def test_read_surfaces_require_real_class_edges_and_check_continuation(monkeypatch, defect):
    source = dict(ontology_version_id="one", iri="urn:class", kind="class")
    sample = dict(ontology_version_id="one", iri="urn:annotation", kind="annotation_property")
    journey = Journey.__new__(Journey)
    journey.config = {"ports": {"exploration": 19185}}
    journey.definition = {"cases": [{"source": source, "candidates": []}]}
    journey.report = {}
    monkeypatch.setattr(environment, "processes", lambda config: {"exploration": {"pid": 1}})
    monkeypatch.setattr(environment, "owned", lambda *args: True)
    hierarchy_probes = []

    def request(method, path, *, params, label):
        if path.endswith("/entities"):
            result = dict(items=[dict(entity=sample)], returned_count=1, total_count=1)
        elif params["direction"] == "ancestors":
            result = dict(items=[], nodes=[source])
        else:
            hierarchy_probes.append({key: params[key] for key in source})
            rows = (
                [] if defect == "empty" else [{"edge": "second" if "cursor" in params else "first"}]
            )
            result = dict(items=rows, returned_count=len(rows))
            if "cursor" not in params and rows:
                result["next_cursor"] = "next"
            if path.startswith("https:"):
                if defect == "presence":
                    result.pop("next_cursor", None)
                if "cursor" in params:
                    if defect == "count":
                        result["returned_count"] += 1
                    elif defect == "parity":
                        result["items"] = [{"edge": "changed"}]
        return SimpleNamespace(json=lambda: result)

    journey.request = request
    if defect:
        label = {
            "empty": "nonempty_class_hierarchy_exercised",
            "count": "hierarchy_continuation_count",
            "parity": "same_hierarchy_continuation",
            "presence": "same_hierarchy_continuation_presence",
        }[defect]
        with pytest.raises(VerificationError, match=label):
            journey.complete_read_surfaces("/workspace", {"ontology_versions": ["one"]})
    else:
        journey.complete_read_surfaces("/workspace", {"ontology_versions": ["one"]})
        result = journey.report["complete_read_surfaces"]["ontologies"][0]
        assert result["search_probe_outside_focal_cases"]
        assert result["hierarchy"] == {
            direction: dict(first_page_edges=1, continuation_edges=1, continuation_exercised=True)
            for direction in ("parents", "children")
        }
    assert hierarchy_probes and all(probe == source for probe in hierarchy_probes)
