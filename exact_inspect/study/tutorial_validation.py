"""Safe tutorial rejection metadata and predicates over admitted frozen resources.

Participant strings never become error detail. IDs identify the authored requirement and
finite action kind; API middleware supplies the request correlation identity separately.
"""

from __future__ import annotations

from typing import Any


class TutorialValidationError(ValueError):
    def __init__(self, code: str, detail: str, *, requirement_ids=(), action_ids=()):
        super().__init__(detail)
        self.code = code
        self.requirement_ids = list(dict.fromkeys(requirement_ids))
        self.action_ids = list(dict.fromkeys(action_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "requirement_ids": self.requirement_ids,
            "action_ids": self.action_ids,
        }


# Evidence actions never accept a graph channel/bridge or an unlinked selected feature.
ACTION_MATRIX = {
    "open_citation": "admitted_fact_cited_by_prepared_claim",
    "open_original_axiom": "admitted_original_fact",
    "locate_in_evidence_list": "available_selected_feature_linked_to_inspected_fact",
    "inspect_graph_or_list": "available_selected_feature_linked_to_inspected_fact",
    "search_entity": "admitted_typed_entity",
    "navigate_parent": "recorded_typed_parent_endpoint",
    "navigate_child": "recorded_typed_child_endpoint",
    "return_to_compared": "compared_typed_entity_after_navigation_away",
    "copy_iri": "admitted_typed_entity",
}
FACT_ACTIONS = {
    "open_citation",
    "open_original_axiom",
    "locate_in_evidence_list",
    "inspect_graph_or_list",
}
ENTITY_ACTIONS = {
    "search_entity",
    "navigate_parent",
    "navigate_child",
    "return_to_compared",
    "copy_iri",
}
SELECTED_ACTIONS = {"locate_in_evidence_list", "inspect_graph_or_list"}


def identity(entity):
    if entity.get("kind") not in {"class", "object_property", "data_property", "individual"}:
        raise TutorialValidationError(
            "tutorial_entity_kind_required", "Choose a typed ontology entity."
        )
    return entity["ontology_version_id"], entity["iri"], entity["kind"]


def all_entities(value):
    if isinstance(value, dict):
        if {"ontology_version_id", "iri", "kind"} <= value.keys():
            yield value
        for child in value.values():
            yield from all_entities(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_entities(child)


def _fail(code, detail, action):
    raise TutorialValidationError(
        code, detail, requirement_ids=[action["requirement_id"]], action_ids=[action["action"]]
    )


def validate_resource_action(action, resources, case, interaction, completed):
    """Check optional scopes strictly; retain valid historical fact-only payloads.

    Resources are already admitted by publication provenance checks. Their contents,
    not participant claims about graph nodes or bridge labels, define the universe.
    """
    kind = action["action"]
    candidates = {c["candidate_id"]: c["entity"] for c in case["candidates"]}
    compared = {identity(case["source"]), *(identity(c) for c in candidates.values())}
    entities = {identity(e) for e in all_entities(resources)} | compared
    entity = action.get("entity")
    key = identity(entity) if entity else None
    if key is not None and key not in entities:
        _fail(
            "tutorial_entity_out_of_scope",
            "Action entity is outside the synthetic tutorial",
            action,
        )
    if kind in ENTITY_ACTIONS and key is None:
        _fail("tutorial_entity_required", "This action needs the selected ontology entity.", action)
    facts = {
        fact["fact_id"]
        for resource in resources
        for fact in resource.get("facts", []) + resource.get("referenced_labels", [])
    }
    fact_id = action.get("fact_id")
    if fact_id and fact_id not in facts:
        _fail("tutorial_fact_out_of_scope", "Action fact is outside the synthetic tutorial", action)
    if kind in FACT_ACTIONS and not fact_id:
        _fail(
            "tutorial_fact_required",
            "Open the linked original fact to complete this action.",
            action,
        )
    if kind in {"navigate_parent", "navigate_child"}:
        endpoint = "parent" if kind == "navigate_parent" else "child"
        if key not in {
            identity(edge[endpoint]) for resource in resources for edge in resource["hierarchy"]
        }:
            _fail(
                "tutorial_navigation_unrecorded", "Choose a recorded hierarchy neighbour.", action
            )
    if kind == "return_to_compared":
        if key not in compared:
            _fail("tutorial_return_out_of_scope", "Return to a compared practice entity.", action)
        if not interaction.get("navigated_away") and action["requirement_id"] not in completed:
            _fail(
                "tutorial_return_requires_navigation",
                "Browse another entity before returning.",
                action,
            )
        interaction["navigated_away"] = False
    elif kind in {"search_entity", "navigate_parent", "navigate_child"} and key not in compared:
        interaction["navigated_away"] = True
    if kind == "open_citation":
        claims = [
            claim
            for resource in resources
            for claim in resource["entity_profiles"] + resource["pair_comparison"]
        ]
        if action.get("candidate_id"):
            scopes = {identity(case["source"]), identity(candidates[action["candidate_id"]])}
            claims = [
                claim
                for claim in claims
                if {identity(e) for e in claim.get("scoped_entities", [])} <= scopes
            ]
        if fact_id not in {fid for claim in claims for fid in claim["fact_ids"]}:
            _fail(
                "tutorial_citation_unrecorded",
                "Open a citation from the displayed prepared text.",
                action,
            )
    if kind in SELECTED_ACTIONS or action.get("evidence_id") is not None:
        links = [
            item
            for resource in resources
            for item in resource["evidence"]
            if item["status"] == "available" and fact_id in item["fact_ids"]
        ]
        if action.get("candidate_id"):
            links = [item for item in links if item["candidate_id"] == action["candidate_id"]]
        if action.get("evidence_id"):
            links = [item for item in links if item["evidence_id"] == action["evidence_id"]]
            if not action.get("candidate_id") and len({x["candidate_id"] for x in links}) > 1:
                _fail(
                    "tutorial_evidence_scope_required",
                    "Identify the candidate for this selected feature.",
                    action,
                )
        if not links:
            _fail(
                "tutorial_evidence_unlinked",
                "Open an original fact linked to the selected practice feature.",
                action,
            )


def validate_reachability(tutorial, resources):
    """Every selectable candidate must permit the authored mandatory workspace actions.

    Checks concern reachable admitted controls/data. Optional graph requirements never
    become a gate; the accessible feature list is the required evidence route.
    """
    case = tutorial["case"]
    source = identity(case["source"])
    facts = {
        f["fact_id"]: f
        for resource in resources
        for f in resource["facts"] + resource.get("referenced_labels", [])
    }
    entities = {identity(e) for e in all_entities(resources)}
    claims = [c for r in resources for c in r["entity_profiles"] + r["pair_comparison"]]
    edges = [edge for r in resources for edge in r["hierarchy"]]
    links = [item for r in resources for item in r["evidence"]]
    mandatory = {
        r["action"]: r["requirement_id"]
        for lesson in tutorial["lessons"]
        if not lesson.get("optional")
        for r in lesson["requirements"]
    }
    for candidate in case["candidates"]:
        pair = {source, identity(candidate["entity"])}
        failures = set()
        if not pair <= entities:
            failures.update(set(ENTITY_ACTIONS) & mandatory.keys())
        # Navigation can take multiple admitted steps: a leaf can go to its parent,
        # then inspect that parent's children. Require a path from the compared pair,
        # not an unrelated edge elsewhere in the frozen resource.
        reachable = set(pair)
        changed = True
        admitted_edges = [edge for edge in edges if set(edge["fact_ids"]) <= facts.keys()]
        while changed:
            changed = False
            for edge in admitted_edges:
                endpoints = {identity(edge["child"]), identity(edge["parent"])}
                if endpoints & reachable and not endpoints <= reachable:
                    reachable.update(endpoints)
                    changed = True
        for action, origin, endpoint in (
            ("navigate_parent", "child", "parent"),
            ("navigate_child", "parent", "child"),
        ):
            if action in mandatory and not any(
                identity(edge[origin]) in reachable
                and identity(edge[endpoint]) != identity(edge[origin])
                for edge in admitted_edges
            ):
                failures.add(action)
        if "search_entity" in mandatory and not entities - pair:
            failures.add("search_entity")
        if "open_citation" in mandatory and not any(
            claim["fact_ids"]
            and set(claim["fact_ids"]) <= facts.keys()
            and {identity(e) for e in claim.get("scoped_entities", [])} <= pair
            for claim in claims
        ):
            failures.add("open_citation")
        if "open_original_axiom" in mandatory and not any(
            identity(f["subject"]) in pair for f in facts.values()
        ):
            failures.add("open_original_axiom")
        if "locate_in_evidence_list" in mandatory and not any(
            link["candidate_id"] == candidate["candidate_id"]
            and link["status"] == "available"
            and link["fact_ids"]
            and set(link["fact_ids"]) <= facts.keys()
            for link in links
        ):
            failures.add("locate_in_evidence_list")
        if failures:
            raise TutorialValidationError(
                "tutorial_unreachable_actions",
                "The prepared practice workspace cannot complete all required actions.",
                requirement_ids=[mandatory[action] for action in sorted(failures)],
                action_ids=sorted(failures),
            )
