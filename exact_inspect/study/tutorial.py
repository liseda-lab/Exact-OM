"""Frozen synthetic tutorial validation, server grading and resumable preparation state."""

from __future__ import annotations

import hashlib
import json

CORE_ITEMS = [
    "score_meaning",
    "equivalence_scope",
    "response_states",
    "evidence_origin",
    "external_methods",
]
REQUIRED_ACTIONS = {
    "identity": {"inspect_other_candidate", "return_to_candidate"},
    "context": {"search_entity", "navigate_parent", "navigate_child", "return_to_compared"},
    "evidence": {"open_citation", "open_original_axiom", "locate_in_evidence_list"},
    "graph": {"inspect_graph_or_list", "change_graph_view", "rank_with_details_open"},
    "answers": {
        "add_rank",
        "move_rank",
        "remove_rank",
        "undo_rank",
        "keep_initial_order",
        "check_partial_ranking",
        "choose_none",
        "choose_insufficient",
    },
    "baseline": {"copy_iri", "locate_downloads", "report_multiple_methods", "report_no_methods"},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tutorial_hash(definition):
    return (
        "sha256:"
        + hashlib.sha256(
            canonical({k: v for k, v in definition.items() if k != "hash"}).encode()
        ).hexdigest()
    )


def response_value(response):
    return {k: v for k, v in response.items() if v is not None}


def validate_response(item, response, *, complete):
    """Validate codes and typed response shape; drafts may omit unanswered parts."""
    response = response_value(response)
    fields = {
        "single": {"choice"},
        "multiple": {"choices"},
        "match": {"matches"},
        "match_and_single": {"matches", "part_b"},
    }[item["kind"]]
    if set(response) - fields or (complete and set(response) != fields):
        raise ValueError("Assessment response does not match its question type")
    if "choice" in response and response["choice"] not in {
        o["code"] for o in item.get("options") or []
    }:
        raise ValueError("Unknown assessment choice")
    if "choices" in response:
        values = response["choices"]
        if (
            len(values) != len(set(values))
            or not set(values) <= {o["code"] for o in item.get("options") or []}
            or (complete and not values)
        ):
            raise ValueError("Invalid assessment choices")
    if "matches" in response:
        values = response["matches"]
        rows = {row["row_id"] for row in item.get("rows") or []}
        options = {o["code"] for o in item.get("row_options") or []}
        if (
            set(values) - rows
            or any(v not in options for v in values.values())
            or (complete and set(values) != rows)
        ):
            raise ValueError("Invalid assessment matching response")
    if "part_b" in response and response["part_b"] not in {
        o["code"] for o in item["part_b"]["options"]
    }:
        raise ValueError("Unknown assessment part B choice")
    return response


def validate_definition(tutorial):
    lessons = tutorial["lessons"]
    if len({x["lesson_id"] for x in lessons}) != len(lessons):
        raise ValueError("Duplicate tutorial lesson")
    by_lesson = {x["lesson_id"]: x for x in lessons}
    if not set(REQUIRED_ACTIONS) <= set(by_lesson):
        raise ValueError("All six required tutorial lessons must be authored")
    for lid, actions in REQUIRED_ACTIONS.items():
        lesson = by_lesson[lid]
        if not actions <= {r["action"] for r in lesson["requirements"]}:
            raise ValueError(f"Tutorial lesson {lid} lacks required interactions")
        if lid != "graph" and lesson.get("optional"):
            raise ValueError("Only the graph lesson may be optional")
    if by_lesson["baseline"]["view"] != "baseline" or any(
        by_lesson[k]["view"] != "explanation" for k in REQUIRED_ACTIONS if k != "baseline"
    ):
        raise ValueError("Training must include both condition workflows")
    requirements = [r for lesson in lessons for r in lesson["requirements"]]
    if len({r["requirement_id"] for r in requirements}) != len(requirements):
        raise ValueError("Duplicate tutorial requirement")
    for requirement in requirements:
        alternatives = requirement.get("alternatives", [])
        allowed = (
            {"inspect_graph_or_list"} if requirement["action"] == "change_graph_view" else set()
        )
        if len(alternatives) != len(set(alternatives)) or not set(alternatives) <= allowed:
            raise ValueError("Undeclared accessibility-equivalent action predicate")
    for r in by_lesson["graph"]["requirements"]:
        if r["action"] == "change_graph_view" and "inspect_graph_or_list" not in r["alternatives"]:
            raise ValueError("Graph interaction requires an accessible list alternative")
    if [q["question_id"] for q in tutorial["assessment"]] != CORE_ITEMS or set(
        tutorial["grading"]
    ) != set(CORE_ITEMS):
        raise ValueError("Exactly the five core assessment items must be frozen in order")
    for item in tutorial["assessment"]:
        if item["lesson_id"] not in by_lesson:
            raise ValueError("Assessment references an unknown lesson")
        for key, code in (("options", "code"), ("rows", "row_id"), ("row_options", "code")):
            values = item.get(key) or []
            if len(values) != len({v[code] for v in values}):
                raise ValueError("Assessment option and row orders must have unique codes")
        kind = item["kind"]
        if (
            (kind in {"single", "multiple"} and not item.get("options"))
            or (
                kind in {"match", "match_and_single"}
                and not (item.get("rows") and item.get("row_options"))
            )
            or (kind == "match_and_single" and not item.get("part_b"))
        ):
            raise ValueError("Incomplete assessment controls")
        validate_response(item, tutorial["grading"][item["question_id"]]["response"], complete=True)
    predicates = {k: response_value(v["response"]) for k, v in tutorial["grading"].items()}
    if (
        predicates["score_meaning"] != {"choice": "advice"}
        or predicates["equivalence_scope"] != {"choice": "broader"}
        or set(predicates["external_methods"].get("choices", []))
        != {"optional", "combine_change", "per_case"}
    ):
        raise ValueError("Core assessment grading contradicts the approved protocol")
    if predicates["response_states"] != {
        "matches": {
            "judged_none": "none_of_these",
            "cannot_judge": "insufficient_evidence",
            "not_answered": "unanswered",
        }
    } or predicates["evidence_origin"] != {
        "matches": {"statement": "original", "feature": "matcher", "summary": "generated"},
        "part_b": "no",
    }:
        raise ValueError("Core matching predicates contradict the approved protocol")
    candidates = tutorial["case"]["candidates"]
    if (
        len({entity_key(c["entity"]) for c in candidates}) != 5
        or len({c["candidate_id"] for c in candidates}) != 5
        or [c["display_position"] for c in candidates] != list(range(1, 6))
    ):
        raise ValueError("Tutorial must have five unique ordered candidates")
    if tutorial["hash"] != tutorial_hash(tutorial):
        raise ValueError("Tutorial hash does not bind its complete frozen definition")


def entity_key(entity):
    if entity.get("kind") not in {"class", "object_property", "data_property", "individual"}:
        raise ValueError("Entity evidence requires an explicit recorded kind")
    return (entity["ontology_version_id"], entity["iri"], entity["kind"])


def walk_entities(value):
    """Include embedded facts, references, fillers and generated scope identities."""
    if isinstance(value, dict):
        if {"ontology_version_id", "iri"} <= value.keys():
            yield value
        for child in value.values():
            yield from walk_entities(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_entities(child)


def validate_disjoint_definition(study):
    tutorial = study["tutorial"]
    case = tutorial["case"]
    scored = study["cases"]
    scored_entities = list(walk_entities(scored))
    scored_ids = (
        {c["case_id"] for c in scored}
        | {x["candidate_id"] for c in scored for x in c["candidates"]}
        | {c["transfer_group"] for c in scored}
    )
    practice_ids = {
        tutorial["tutorial_id"],
        tutorial["practice_id"],
        tutorial["transfer_group"],
    } | {x["candidate_id"] for x in case["candidates"]}
    if scored_ids & practice_ids:
        raise ValueError(
            "Tutorial identities and transfer groups must be disjoint from scored material"
        )
    # Ontology separation also excludes matching IRIs in a differently versioned resource.
    scored_iris = {e["iri"] for e in scored_entities}
    scored_ontologies = {e["ontology_version_id"] for e in scored_entities}
    if any(
        e["iri"] in scored_iris or e["ontology_version_id"] in scored_ontologies
        for e in walk_entities(case)
    ):
        raise ValueError("Tutorial ontology identities must be disjoint from scored material")
    practice_assets = {a["asset_id"] for a in case["ontology_resources"]} | set(
        case["explanation_refs"]
    )
    scored_assets = {a for c in scored for a in c["ontology_resource_ids"] + c["explanation_refs"]}
    if practice_assets & scored_assets:
        raise ValueError("Tutorial resources must be disjoint from scored resources")
    assets = {a["asset_id"]: a for a in study["assets"]}
    if not practice_assets <= assets.keys() or any(
        assets[a]["kind"] != "explanation" for a in case["explanation_refs"]
    ):
        raise ValueError("Tutorial must reference admitted publication resources")
    for resource in case["ontology_resources"]:
        asset = assets[resource["asset_id"]]
        if any(asset.get(k) != v for k, v in resource.items()):
            raise ValueError("Tutorial download metadata differs from publication asset")


def validate_publication(store, study, admitted_ontologies, admitted_explanations):
    """Check full tutorial fact graph after provenance/policy admission."""
    if study["contract_version"] != "exact-study/2.0":
        return
    tutorial = study["tutorial"]
    tcase = tutorial["case"]
    tutorial_ontology_ids = {a["ontology_version_id"] for a in tcase["ontology_resources"]}
    all_cases = [
        *study["cases"],
        {**tcase, "ontology_resource_ids": [a["asset_id"] for a in tcase["ontology_resources"]]},
    ]
    roles = {}
    for case in all_cases:
        roles.setdefault(case["source"]["ontology_version_id"], set()).add("source")
        for candidate in case["candidates"]:
            roles.setdefault(candidate["entity"]["ontology_version_id"], set()).add("target")
    for case in all_cases:
        targets = {c["entity"]["ontology_version_id"] for c in case["candidates"]}
        required = {case["source"]["ontology_version_id"]} | targets
        available = set()
        for aid in case["ontology_resource_ids"]:
            asset = next(a for a in study["assets"] if a["asset_id"] == aid)
            oid = admitted_ontologies[aid]
            expected_role = (
                "both"
                if len(roles.get(oid, set())) == 2
                else next(iter(roles.get(oid, {"unknown"})))
            )
            if (
                oid not in required
                or asset["ontology_version_id"] != oid
                or asset["role"] != expected_role
            ):
                raise ValueError("Download ontology metadata or role disagrees with the case")
            available.add(oid)
        if not required <= available:
            raise ValueError("Missing tutorial or case download")
    scored_resources = [
        admitted_explanations[a].model_dump(mode="json")
        for c in study["cases"]
        for a in c["explanation_refs"]
    ]
    scored_iris = {e["iri"] for e in walk_entities(scored_resources)} | {
        e["iri"] for e in walk_entities(study["cases"])
    }
    candidates = {c["candidate_id"] for c in tcase["candidates"]}
    for aid in tcase["explanation_refs"]:
        resource = admitted_explanations[aid].model_dump(mode="json")
        if not resource["facts"] or not resource["hierarchy"] or not resource["entity_profiles"]:
            raise ValueError("Tutorial requires prepared facts, hierarchy and cited explanations")
        if any(
            e["ontology_version_id"] not in tutorial_ontology_ids or e["iri"] in scored_iris
            for e in walk_entities(resource)
        ):
            raise ValueError("Tutorial resource graph includes scored or non-synthetic entities")
        if any(e["candidate_id"] not in candidates for e in resource["evidence"]):
            raise ValueError("Tutorial evidence includes a scored candidate")


def initial_progress(tutorial):
    progress = {
        "tutorial_version": tutorial["version"],
        "current_lesson_id": tutorial["lessons"][0]["lesson_id"],
        "position": {
            "view": "lesson",
            "lesson_id": tutorial["lessons"][0]["lesson_id"],
            "question_id": None,
        },
        "completed_requirements": [],
        "practice": {},
        "assessment_drafts": {},
        "attempts": [],
        "passed_items": [],
        "outstanding": [],
        "help_opened": 0,
        "completed_at": None,
    }
    refresh_outstanding(tutorial, progress)
    return progress


def normalize_position(tutorial, progress):
    """Normalize a detached state on read; storage changes only on a valid mutation."""
    if "position" not in progress:
        lesson = progress.get("current_lesson_id")
        if lesson in {item["lesson_id"] for item in tutorial["lessons"]}:
            position = {"view": "lesson", "lesson_id": lesson, "question_id": None}
        elif progress.get("assessment_drafts") or progress.get("attempts"):
            position = {"view": "assessment", "lesson_id": None, "question_id": None}
        else:
            position = {
                "view": "lesson",
                "lesson_id": tutorial["lessons"][0]["lesson_id"],
                "question_id": None,
            }
        progress["position"] = position
    progress["current_lesson_id"] = progress["position"]["lesson_id"]
    return progress


def patch_position(tutorial, progress, data):
    normalize_position(tutorial, progress)
    if "position" in data:
        position = data["position"]
        from .v2_models import AssessmentPosition, LessonPosition

        model = LessonPosition if position["view"] == "lesson" else AssessmentPosition
        position = model.model_validate(position).model_dump(mode="json")
        if position["view"] == "lesson" and position["lesson_id"] not in {
            l["lesson_id"] for l in tutorial["lessons"]
        }:
            raise ValueError("Unknown tutorial position lesson")
        if (
            position["view"] == "assessment"
            and position["question_id"] is not None
            and position["question_id"] not in {q["question_id"] for q in tutorial["assessment"]}
        ):
            raise ValueError("Unknown tutorial position question")
        progress["position"] = position
    elif data.get("current_lesson_id") is not None:
        progress["position"] = {
            "view": "lesson",
            "lesson_id": data["current_lesson_id"],
            "question_id": None,
        }
    progress["current_lesson_id"] = progress["position"]["lesson_id"]


def refresh_outstanding(tutorial, progress):
    completed = set(progress["completed_requirements"])
    progress["outstanding"] = [
        l["lesson_id"]
        for l in tutorial["lessons"]
        if not l.get("optional")
        and any(r["requirement_id"] not in completed for r in l["requirements"])
    ] + [
        q["question_id"]
        for q in tutorial["assessment"]
        if q["question_id"] not in progress["passed_items"]
    ]


def public_tutorial(tutorial):
    return {
        k: tutorial[k]
        for k in (
            "tutorial_id",
            "version",
            "hash",
            "synthetic",
            "intro",
            "lessons",
            "assessment",
            "case",
        )
    }


def validate_practice(case, answer):
    ids = answer["ranked_candidate_ids"]
    if (
        len(ids) != len(set(ids))
        or not set(ids) <= {c["candidate_id"] for c in case["candidates"]}
        or (answer["response_type"] == "ranked_candidates") != bool(ids)
    ):
        raise ValueError("Invalid synthetic practice response")


def apply_progress(store, study, progress, data, interaction_state=None):
    tutorial = study["tutorial"]
    interaction_state = interaction_state if interaction_state is not None else {}
    visited = set(
        interaction_state.get(
            "inspected_candidates", [tutorial["case"]["candidates"][0]["candidate_id"]]
        )
    )
    lesson_ids = {l["lesson_id"] for l in tutorial["lessons"]}
    for field in ("current_lesson_id", "lesson_id"):
        if data.get(field) is not None and data[field] not in lesson_ids:
            raise ValueError("Unknown tutorial lesson")
    requirements = {r["requirement_id"]: r for l in tutorial["lessons"] for r in l["requirements"]}
    requested = set(data["completed_requirements"])
    if (
        len(requested) != len(data["completed_requirements"])
        or not requested <= requirements.keys()
    ):
        raise ValueError("Unknown or duplicate completed tutorial requirement")
    completed = set(progress["completed_requirements"])
    allowed_candidates = {c["candidate_id"] for c in tutorial["case"]["candidates"]}
    resources = None
    for action in data["actions"]:
        requirement = requirements.get(action["requirement_id"])
        if not requirement or action["action"] not in {
            requirement["action"],
            *requirement.get("alternatives", []),
        }:
            raise ValueError("Action does not satisfy a declared tutorial requirement")
        kind = action["action"]
        if (
            action.get("candidate_id") is not None
            and action["candidate_id"] not in allowed_candidates
        ):
            raise ValueError("Action candidate is outside the synthetic tutorial")
        if kind in {"inspect_other_candidate", "return_to_candidate"} and not action.get(
            "candidate_id"
        ):
            raise ValueError("Candidate inspection requires a synthetic candidate identity")
        if kind == "inspect_other_candidate":
            if action["candidate_id"] == tutorial["case"]["candidates"][0]["candidate_id"]:
                raise ValueError("Inspect a candidate other than the initial first candidate")
            visited.add(action["candidate_id"])
        if kind == "return_to_candidate" and action["candidate_id"] not in visited:
            raise ValueError("Return requires a previously inspected synthetic candidate")
        if (
            kind
            in {
                "search_entity",
                "navigate_parent",
                "navigate_child",
                "return_to_compared",
                "copy_iri",
                "open_citation",
                "open_original_axiom",
                "locate_in_evidence_list",
                "inspect_graph_or_list",
            }
            or action.get("entity")
            or action.get("fact_id")
        ):
            if resources is None:
                from .resources import ExplanationResource

                # Historical admitted resource bytes may omit legacy class defaults.
                # Normalize through their original model, without rewriting frozen assets
                # or applying those defaults to new participant action evidence.
                resources = [
                    ExplanationResource.model_validate_json(
                        store._asset_bytes(next(a for a in study["assets"] if a["asset_id"] == aid))
                    ).model_dump(mode="json")
                    for aid in tutorial["case"]["explanation_refs"]
                ]
            identities = {entity_key(e) for e in walk_entities(resources)} | {
                entity_key(e) for e in walk_entities(tutorial["case"])
            }
            if action.get("entity") and entity_key(action["entity"]) not in identities:
                raise ValueError("Action entity is outside the synthetic tutorial")
            if kind in {
                "search_entity",
                "navigate_parent",
                "navigate_child",
                "return_to_compared",
                "copy_iri",
            } and not action.get("entity"):
                raise ValueError("Entity action requires its synthetic typed identity")
            facts = {
                f["fact_id"]
                for resource in resources
                for f in resource.get("facts", []) + resource.get("referenced_labels", [])
            }
            if action.get("fact_id") and action["fact_id"] not in facts:
                raise ValueError("Action fact is outside the synthetic tutorial")
            if kind in {
                "open_citation",
                "open_original_axiom",
                "locate_in_evidence_list",
                "inspect_graph_or_list",
            } and not action.get("fact_id"):
                raise ValueError("Evidence action requires an admitted original fact identity")
            if kind in {"navigate_parent", "navigate_child"}:
                endpoint = "parent" if kind == "navigate_parent" else "child"
                if entity_key(action["entity"]) not in {
                    entity_key(edge[endpoint])
                    for resource in resources
                    for edge in resource["hierarchy"]
                }:
                    raise ValueError(
                        "Navigation action is not a recorded synthetic hierarchy endpoint"
                    )
            if kind == "return_to_compared" and entity_key(action["entity"]) not in {
                entity_key(tutorial["case"]["source"]),
                *(entity_key(c["entity"]) for c in tutorial["case"]["candidates"]),
            }:
                raise ValueError("Return must restore a compared synthetic entity")
            if kind == "open_citation" and action["fact_id"] not in {
                fid
                for resource in resources
                for claim in resource["entity_profiles"] + resource["pair_comparison"]
                for fid in claim["fact_ids"]
            }:
                raise ValueError("Citation action requires a fact cited by prepared synthetic text")
            if kind in {"locate_in_evidence_list", "inspect_graph_or_list"} and action[
                "fact_id"
            ] not in {
                fid
                for resource in resources
                for evidence in resource["evidence"]
                for fid in evidence["fact_ids"]
            }:
                raise ValueError("Evidence-list action requires a recorded synthetic evidence link")
        if kind in {
            "rank_with_details_open",
            "add_rank",
            "move_rank",
            "remove_rank",
            "undo_rank",
            "keep_initial_order",
            "check_partial_ranking",
            "choose_none",
            "choose_insufficient",
        }:
            if action.get("ranked_candidate_ids") is None:
                raise ValueError("Rank action requires a typed synthetic answer")
            validate_practice(tutorial["case"], action)
            if kind == "check_partial_ranking" and not 1 <= len(action["ranked_candidate_ids"]) < 5:
                raise ValueError("Partial ranking requires one to four candidates")
            if kind == "keep_initial_order" and action["ranked_candidate_ids"] != [
                c["candidate_id"] for c in tutorial["case"]["candidates"]
            ]:
                raise ValueError("Keep-order action must preserve the initial order")
            if (
                kind in {"choose_none", "choose_insufficient"}
                and action["response_type"]
                != {
                    "choose_none": "none_of_these",
                    "choose_insufficient": "insufficient_evidence",
                }[kind]
            ):
                raise ValueError("Practice response differs from the action")
        if kind in {"report_multiple_methods", "report_no_methods"}:
            methods = action.get("methods")
            if (
                methods is None
                or len(methods) != len(set(methods))
                or (kind == "report_multiple_methods" and len(methods) < 2)
                or (kind == "report_no_methods" and methods)
            ):
                raise ValueError("Synthetic consultation does not satisfy the practice action")
        if kind == "locate_downloads" and set(action.get("asset_ids") or []) != {
            a["asset_id"] for a in tutorial["case"]["ontology_resources"]
        }:
            raise ValueError("Locate both synthetic ontology downloads")
        completed.add(requirement["requirement_id"])
    if not requested <= completed:
        raise ValueError(
            "Completed IDs require typed action evidence; checkbox claims do not count"
        )
    interaction_state["inspected_candidates"] = sorted(visited)
    progress["completed_requirements"] = [r for r in requirements if r in completed]
    patch_position(tutorial, progress, data)
    if data.get("practice"):
        answer = data["practice"]
        if answer["key"] != tutorial["practice_id"]:
            raise ValueError("Unknown synthetic practice answer")
        validate_practice(tutorial["case"], answer)
        progress["practice"][answer["key"]] = {k: v for k, v in answer.items() if k != "key"}
    if data.get("assessment_draft"):
        draft = data["assessment_draft"]
        item = next(
            (i for i in tutorial["assessment"] if i["question_id"] == draft["question_id"]), None
        )
        if item is None:
            raise ValueError("Unknown assessment question")
        progress["assessment_drafts"][item["question_id"]] = validate_response(
            item, draft["response"], complete=False
        )
    if data.get("help_opened"):
        progress["help_opened"] += 1
    refresh_outstanding(tutorial, progress)


def attempt(tutorial, progress, data, now):
    item = next(
        (q for q in tutorial["assessment"] if q["question_id"] == data["question_id"]), None
    )
    if item is None:
        raise ValueError("Unknown assessment question")
    response = validate_response(item, data["response"], complete=True)
    prior = next((a for a in progress["attempts"] if a["attempt_id"] == data["attempt_id"]), None)
    if prior:
        if (
            prior["question_id"] != data["question_id"]
            or response_value(prior["response"]) != response
        ):
            raise ValueError("Attempt ID already belongs to different content")
        return prior
    rule = tutorial["grading"][item["question_id"]]
    expected = response_value(rule["response"])
    correct = (
        (set(response.get("choices", [])) == set(expected["choices"]))
        if item["kind"] == "multiple"
        else response == expected
    )
    receipt = {
        "attempt_id": data["attempt_id"],
        "question_id": item["question_id"],
        "response": response,
        "correct": correct,
        "feedback": rule["correct_feedback"] if correct else rule["incorrect_feedback"],
        "revisit_lesson_id": None if correct else item["lesson_id"],
        "submitted_at": now,
    }
    progress["attempts"].append(receipt)
    if correct and item["question_id"] not in progress["passed_items"]:
        progress["passed_items"].append(item["question_id"])
    refresh_outstanding(tutorial, progress)
    return receipt
