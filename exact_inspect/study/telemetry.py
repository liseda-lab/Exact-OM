"""Versioned optional observations, independent of authoritative study mutations.

Late v2 intervals are acknowledged as unavailable, recorded and excluded from durations.
An interval received after its stage ends cannot establish where its client clock lay
relative to a server transition. We deliberately do not manufacture that relationship.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, get_args

from pydantic import Field, model_validator

from .models import Event, Identifier, StrictModel, TimingSegment
from .store import StudyError, canonical, digest, utcnow

V1_EVENT_TYPES = get_args(Event.model_fields["type"].annotation)
V2_EVENT_TYPES = (
    "tab_open",
    "hierarchy_navigate",
    "search_select",
    "graph_reset",
    "graph_edge_open",
    "copy_iri",
    "resources_open",
    "help_open",
)
WORKSPACE_EVENTS = {
    "hierarchy_expand",
    "hierarchy_collapse",
    "definition_open",
    "axiom_open",
    "evidence_open",
    "table_open",
    "graph_open",
    "comparison_open",
    "graph_zoom",
    "graph_fit",
    "graph_reset",
    "graph_edge_open",
    "hierarchy_navigate",
    "search_select",
    "tab_open",
}
COMPONENTS = {
    "details",
    "citation",
    "workspace",
    "hierarchy_source",
    "hierarchy_target",
    "graph",
    "evidence_list",
    "tutorial_help",
    "source",
    "target",
    "ranking",
    "resources",
    "downloads",
    "comparison",
    "meaning",
    "hierarchy",
    "evidence",
    "identity",
}
RANK_EVENTS = {
    "candidate_inspected",
    "rank_add",
    "rank_remove",
    "rank_move",
    "keep_initial_order",
    "response_type_change",
    "revision",
    "submit",
}
ACTION_COMPONENTS = {
    **{action: {"ranking"} for action in RANK_EVENTS},
    **{
        action: {"hierarchy_source", "hierarchy_target", "hierarchy", "workspace"}
        for action in {
            "hierarchy_expand",
            "hierarchy_collapse",
            "hierarchy_navigate",
            "search_select",
        }
    },
    "definition_open": {"source", "target", "meaning", "workspace", "citation"},
    "axiom_open": {"source", "target", "meaning", "workspace", "citation"},
    "evidence_open": {"graph", "evidence_list", "evidence"},
    "table_open": {"details", "evidence_list", "evidence"},
    "graph_open": {"details", "graph"},
    "comparison_open": {"details", "comparison"},
    **{
        action: {"graph"}
        for action in {"graph_zoom", "graph_fit", "graph_reset", "graph_edge_open"}
    },
    "tab_open": {"details"},
    "copy_iri": {"workspace", "source", "target", "identity"},
    "resources_open": {"resources", "downloads"},
    "external_resource_link": {"resources", "downloads"},
    "help_open": {"tutorial_help"},
    **{action: set() for action in {"case_ready", "pause", "resume", "visibility"}},
}
ACTION_RESOURCES = {
    **{
        action: ("hierarchy", {"hierarchy"})
        for action in {
            "hierarchy_expand",
            "hierarchy_collapse",
            "hierarchy_navigate",
            "search_select",
        }
    },
    "definition_open": ("context", {"original_context"}),
    "evidence_open": ("evidence", {"evidence_table", "evidence_graph"}),
    "table_open": ("evidence", {"evidence_table"}),
    **{
        action: ("evidence", {"evidence_graph"})
        for action in {"graph_open", "graph_zoom", "graph_fit", "graph_reset", "graph_edge_open"}
    },
    "comparison_open": ("comparison", {"pair_comparison"}),
}


def _element_alias(value):
    """Match the frontend's bounded identifier encoding, without observing URLs."""
    return re.sub(r"[^A-Za-z0-9_.:/-]", "_", value)[:128].replace("://", "_")


def _case_identifiers(case):
    """Collect public frozen identities only, never researcher keys or answer text."""
    result = {
        case["case_id"],
        *case.get("explanation_refs", []),
        *case.get("ontology_resource_ids", []),
    }
    for entity in [case["source"], *(candidate["entity"] for candidate in case["candidates"])]:
        result.update((entity["iri"], entity["ontology_version_id"]))
    result.update(candidate["candidate_id"] for candidate in case["candidates"])
    result.update(resource["asset_id"] for resource in case.get("ontology_resources", []))
    return result


def _element_scope(case, event, *, synthetic=False, lessons=()):
    """Check meaningful typed targets; optional observations may omit a target."""
    element, action = event.get("element_id"), event["type"]
    if element is None:
        return
    candidates = {_element_alias(candidate["candidate_id"]) for candidate in case["candidates"]}
    resources = {_element_alias(value) for value in case.get("ontology_resource_ids", [])}
    resources.update(
        _element_alias(value["asset_id"]) for value in case.get("ontology_resources", [])
    )
    targets = {
        **{
            name: candidates
            for name in {"candidate_inspected", "rank_add", "rank_remove", "rank_move"}
        },
        "response_type_change": {"ranked_candidates", "none_of_these", "insufficient_evidence"},
        "revision": {"undo"},
        "copy_iri": {"source", "target"},
        "tab_open": {"hierarchy", "evidence", "graph", "comparison"},
        "external_resource_link": resources,
    }
    if action in targets and element not in targets[action]:
        raise StudyError(422, "Observation target is unavailable for this action")
    if synthetic and action not in targets:
        # Practice currently sends no fact/IRI identifiers. Admit known synthetic
        # identities only, rather than accepting arbitrary scored or free-text IDs.
        admitted = {_element_alias(value) for value in _case_identifiers(case)} | set(lessons)
        if element not in admitted:
            raise StudyError(422, "Observation target is outside the synthetic tutorial")


class EventV2(Event):
    """Tutorial/help observations carry synthetic identities instead of fake cases."""

    case_id: Identifier | None = None
    presentation_id: Identifier | None = None
    type: Literal[
        "case_ready",
        "candidate_inspected",
        "rank_add",
        "rank_remove",
        "rank_move",
        "keep_initial_order",
        "response_type_change",
        "hierarchy_expand",
        "hierarchy_collapse",
        "definition_open",
        "axiom_open",
        "evidence_open",
        "table_open",
        "graph_open",
        "comparison_open",
        "graph_zoom",
        "graph_fit",
        "external_resource_link",
        "pause",
        "resume",
        "visibility",
        "submit",
        "revision",
        "tab_open",
        "hierarchy_navigate",
        "search_select",
        "graph_reset",
        "graph_edge_open",
        "copy_iri",
        "resources_open",
        "help_open",
    ]
    scope: Literal["case", "tutorial", "help"] = "case"
    tutorial_version: Identifier | None = None
    lesson_id: Identifier | None = None

    @model_validator(mode="after")
    def check_scope(self):
        if self.scope == "case":
            if (
                not self.case_id
                or not self.presentation_id
                or self.tutorial_version
                or self.lesson_id
            ):
                raise ValueError("Case observations require only a case and presentation")
        elif self.case_id or self.presentation_id or not self.tutorial_version:
            raise ValueError("Tutorial/help observations require a tutorial version and no case")
        elif self.type == "case_ready":
            raise ValueError("Tutorial observations cannot start a scored case timer")
        return self


class EventBatchV2(StrictModel):
    events: Annotated[list[EventV2], Field(min_length=1, max_length=100)]


class TimingSegmentV2(TimingSegment):
    stage: Literal["setup", "background", "practice", "tutorial", "case", "consultation", "final"]


class TimingAcknowledgementV2(StrictModel):
    acknowledged_segment_id: Identifier
    availability: Literal["observed", "unavailable"]
    reason: Literal["stage_no_longer_current", "case_not_ready"] | None = None


def declaration():
    return {
        "event_types": [*V1_EVENT_TYPES, *V2_EVENT_TYPES],
        "scopes": ["case", "tutorial", "help"],
        "late_timing_policy": "acknowledged_unavailable",
    }


def _presentations(state):
    return {
        p["case_id"]: p
        for p in (state["assignment"] or {}).get("presentations", [])[: state["case_index"] + 1]
    }


def _scope(store, state, study, event):
    scope = event.get("scope", "case")
    action = event["type"]
    lesson = None
    tutorial = study.get("tutorial") or {}
    lessons = {item["lesson_id"]: item for item in tutorial.get("lessons", [])}
    if scope != "case":
        if event.get("tutorial_version") != tutorial.get("version"):
            raise StudyError(409, "Tutorial observation version is unavailable")
        if event.get("lesson_id") is not None and event["lesson_id"] not in lessons:
            raise StudyError(422, "Unknown tutorial lesson")
        lesson = lessons.get(event.get("lesson_id"))
        if lesson is None and action in WORKSPACE_EVENTS | RANK_EVENTS:
            raise StudyError(422, "Practice observations require a tutorial lesson")
        if state["stage"] not in {"tutorial", "case", "consultation", "paused", "final"}:
            raise StudyError(403, "Tutorial observations unavailable at this stage")
        if event.get("case_id") or event.get("presentation_id"):
            raise StudyError(403, "Synthetic help cannot name a scored case")
        case = tutorial["case"]
        if lesson and lesson["view"] == "baseline" and action in WORKSPACE_EVENTS:
            raise StudyError(403, "Component unavailable in baseline practice")
        element = event.get("element_id")
        if element:
            scored = {
                _element_alias(value)
                for scored_case in study["cases"]
                for value in _case_identifiers(scored_case)
            }
            if element in scored:
                raise StudyError(403, "Synthetic observations cannot name scored resources")
    else:
        presentation = _presentations(state).get(event["case_id"])
        if not presentation or presentation["presentation_id"] != event["presentation_id"]:
            raise StudyError(403, "Event presentation is unauthorized")
        if presentation["condition"] == "ontology_baseline" and event["type"] in WORKSPACE_EVENTS:
            raise StudyError(403, "Component unavailable in baseline")
        if event["type"] == "case_ready" and (
            state["stage"] != "case"
            or (store._current(state) or {}).get("presentation_id") != event["presentation_id"]
        ):
            raise StudyError(409, "Case timer cannot start outside its ranking step")
        case = next(item for item in study["cases"] if item["case_id"] == event["case_id"])
    component = event.get("component_id")
    if component is not None:
        if component not in COMPONENTS and not (scope != "case" and component in lessons):
            raise StudyError(422, "Unknown observation component")
        if component in lessons and scope != "case":
            if component != event.get("lesson_id"):
                raise StudyError(422, "Observation component names a different lesson")
        elif component not in ACTION_COMPONENTS.get(action, set()):
            raise StudyError(422, "Observation component does not match its action")
    _element_scope(case, event, synthetic=scope != "case", lessons=lessons)
    requirement = ACTION_RESOURCES.get(action)
    if action == "tab_open" and event.get("element_id"):
        requirement = ACTION_RESOURCES.get(
            {
                "hierarchy": "hierarchy_expand",
                "evidence": "table_open",
                "graph": "graph_open",
                "comparison": "comparison_open",
            }[event["element_id"]]
        )
    if requirement:
        resource, components = requirement
        workspace = next(
            (
                item
                for item in study.get("workspace_scopes", [])
                if item["case_id"] == case["case_id"]
                and item["kind"] == ("case" if scope == "case" else "tutorial")
            ),
            None,
        )
        if workspace is None or resource not in workspace["components"]:
            raise StudyError(403, "Observation resource was not admitted for this workspace")
        if scope == "case" and not components.intersection(study["components"]):
            raise StudyError(403, "Observation component was not admitted for this publication")
    compatible = (study.get("tutorial") or {}).get("compatible_builds", [])
    if compatible and event["build_version"] not in compatible:
        raise StudyError(409, "Observation build is incompatible with this publication")


def save_events(store, sid, generation, batch):
    """Admit bounded v2 events; retain the legacy contract for legacy sessions."""
    with store.transaction() as db:
        _, state, study = store._session(db, sid, generation)
        legacy = study["contract_version"] == "exact-study/1.0"
    if legacy:
        from .models import EventBatch

        return store.events(
            sid, generation, EventBatch.model_validate(batch.model_dump(mode="json"))
        )
    batch = EventBatchV2.model_validate(batch.model_dump(mode="json"))
    with store.transaction() as db:
        _, state, study = store._session(db, sid, generation)
        if not state["consent"] or not state["consent"]["accepted"]:
            raise StudyError(403, "Consent is required")
        acknowledged, gaps = [], []
        for model in batch.events:
            event = model.model_dump(mode="json")
            payload_hash = digest(canonical(event))
            existing = db.execute(
                "SELECT payload_hash FROM events WHERE session_id = ? AND event_id = ?",
                (sid, event["event_id"]),
            ).fetchone()
            if existing:
                if existing["payload_hash"] != payload_hash:
                    raise StudyError(409, "Event ID reused for different content")
                acknowledged.append(event["event_id"])
                continue
            _scope(store, state, study, event)
            if db.execute(
                "SELECT event_id FROM events WHERE session_id = ? AND page_id = ? AND sequence = ?",
                (sid, event["page_instance_id"], event["sequence"]),
            ).fetchone():
                raise StudyError(409, "Event sequence already acknowledged")
            last = db.execute(
                "SELECT MAX(sequence) AS sequence FROM events WHERE session_id = ? AND page_id = ?",
                (sid, event["page_instance_id"]),
            ).fetchone()["sequence"]
            expected = last + 1 if last is not None else 0
            gap = (
                {
                    "page_instance_id": event["page_instance_id"],
                    "from": expected,
                    "to": event["sequence"] - 1,
                }
                if event["sequence"] > expected
                else None
            )
            if gap:
                gaps.append(gap)
            event.update(server_received_at=utcnow(), sequence_gap_before=gap)
            db.execute(
                "INSERT INTO events(session_id, event_id, page_id, sequence, payload_hash, payload, received_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    event["event_id"],
                    event["page_instance_id"],
                    event["sequence"],
                    payload_hash,
                    canonical(event),
                    event["server_received_at"],
                ),
            )
            acknowledged.append(event["event_id"])
        return {"acknowledged_event_ids": acknowledged, "sequence_gaps": gaps}


def save_timing(store, sid, generation, model):
    with store.transaction() as db:
        _, state, study = store._session(db, sid, generation)
        legacy = study["contract_version"] == "exact-study/1.0"
    if legacy:
        return store.timing(
            sid, generation, TimingSegment.model_validate(model.model_dump(mode="json"))
        )
    segment = TimingSegmentV2.model_validate(model.model_dump(mode="json")).model_dump(mode="json")
    payload_hash = digest(canonical(segment))
    with store.transaction() as db:
        _, state, _ = store._session(db, sid, generation)
        if not state["consent"] or not state["consent"]["accepted"]:
            raise StudyError(403, "Consent is required")
        previous = db.execute(
            "SELECT payload_hash, payload FROM timing_segments WHERE session_id = ? AND segment_id = ?",
            (sid, segment["segment_id"]),
        ).fetchone()
        if previous:
            if previous["payload_hash"] != payload_hash:
                raise StudyError(409, "Timing segment ID reused for different content")
            import json

            return _timing_receipt(json.loads(previous["payload"]))
        if segment["stage"] in {"case", "consultation"}:
            presentation = _presentations(state).get(segment["case_id"])
            if not presentation or presentation["presentation_id"] != segment["presentation_id"]:
                raise StudyError(403, "Timing presentation is unauthorized")
        elif segment["case_id"] is not None or segment["presentation_id"] is not None:
            raise StudyError(422, "Non-case timing must not name a case")
        current = store._current(state) or {}
        reason = None
        if state["stage"] != segment["stage"] or (
            segment["case_id"] is not None
            and current.get("presentation_id") != segment["presentation_id"]
        ):
            reason = "stage_no_longer_current"
        elif segment["stage"] == "case" and not any(
            e["type"] == "case_ready"
            and e.get("presentation_id") == segment["presentation_id"]
            and e["page_instance_id"] == segment["page_instance_id"]
            and e["client_monotonic_ms"] <= segment["monotonic_start_ms"]
            for e in store._records(db, "events", sid)
        ):
            reason = "case_not_ready"
        for earlier in store._records(db, "timing_segments", sid):
            if earlier.get("availability", "observed") == "unavailable" or reason:
                continue
            if earlier["page_instance_id"] == segment["page_instance_id"] and max(
                earlier["monotonic_start_ms"], segment["monotonic_start_ms"]
            ) < min(earlier["monotonic_end_ms"], segment["monotonic_end_ms"]):
                raise StudyError(409, "Observed timing segments must not overlap")
        segment.update(
            server_received_at=utcnow(),
            availability="unavailable" if reason else "observed",
            reason=reason,
        )
        db.execute(
            "INSERT INTO timing_segments(session_id, segment_id, payload_hash, payload, received_at) VALUES (?, ?, ?, ?, ?)",
            (
                sid,
                segment["segment_id"],
                payload_hash,
                canonical(segment),
                segment["server_received_at"],
            ),
        )
        return _timing_receipt(segment)


def _timing_receipt(segment):
    return {
        "acknowledged_segment_id": segment["segment_id"],
        "availability": segment["availability"],
        "reason": segment["reason"],
    }
