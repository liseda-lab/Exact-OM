"""Exercise a test-only development review through certificate-verified HTTPS.

This is an automated API journey, not a browser/usability or participant study.
The tool uses researcher-only frozen tutorial grading for test answers, never
research case adjudication. It writes only counts, hashes and HTTP diagnostics;
requests, answers, cookies, invitations and researcher credentials stay private.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import ssl
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit
from uuid import uuid4

import httpx

from exact_inspect.contracts import canonical_hash
from exact_inspect.evidence_metadata import evidence_completeness
from tools import review_environment as environment


class VerificationError(RuntimeError):
    """A fixed diagnostic label safe to include in the token-free receipt."""


def require(value, label):
    if not value:
        raise VerificationError(label)


def fingerprint(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def validate_response_csp(response, *, study_origin, label):
    """Require promised study/document policies without inventing an exploration API policy."""
    url = urlsplit(str(response.url))
    study = f"{url.scheme}://{url.netloc}" == study_origin.rstrip("/")
    document = response.headers.get("content-type", "").split(";", 1)[0] == "text/html"
    if not (study or document):
        return "not_promised_by_exploration_api"
    raw = response.headers.get("content-security-policy", "")
    require(bool(raw.strip()), label + ":csp_present")
    directives = {}
    for part in raw.split(";"):
        tokens = part.split()
        if not tokens:
            continue
        require(tokens[0] not in directives, label + ":csp_unique_directives")
        directives[tokens[0]] = tokens[1:]
    for name, expected in (
        ("default-src", ["'self'"]),
        ("frame-ancestors", ["'none'"]),
        ("object-src", ["'none'"]),
        ("base-uri", ["'none'"]),
    ):
        require(directives.get(name) == expected, label + ":csp_" + name)
    require("unsafe-eval" not in raw, label + ":csp_no_unsafe_eval")
    if document:
        scripts = directives.get("script-src", [])
        require(
            "'self'" in scripts
            and all(
                token == "'self'" or re.fullmatch(r"'sha256-[A-Za-z0-9+/]{43}='", token)
                for token in scripts
            ),
            label + ":csp_document_scripts",
        )
        for name in ("connect-src", "font-src", "form-action"):
            require(directives.get(name) == ["'self'"], label + ":csp_" + name)
        require(
            directives.get("style-src") == (["'self'"] if study else ["'self'", "'unsafe-inline'"]),
            label + ":csp_document_styles",
        )
    return "strict_document" if document else "study_response"


def answers_for(form):
    """Select valid non-identifying test codes from the actual versioned form."""
    answers = {}
    for question in form:
        if any(
            (
                (expected["contains"] not in (answers.get(key) or []))
                if isinstance(expected, dict)
                else answers.get(key) != expected
            )
            for key, expected in (question.get("show_if") or {}).items()
        ):
            continue
        if not question["required"]:
            continue
        if question["options"] is None:
            answers[question["id"]] = "Automated local verification"
            continue
        options = question["option_order"]
        selected = "cannot_judge" if "cannot_judge" in options else options[0]
        if question["matrix"]:
            value = {row: selected for row in question["row_order"]}
        else:
            value = [selected] if question["multiple"] else selected
        answers[question["id"]] = value
    return answers


def wrong_answer(item, correct):
    result = copy.deepcopy(correct)
    if item["kind"] == "single":
        result["choice"] = next(
            o["code"] for o in item["options"] if o["code"] != correct["choice"]
        )
    elif item["kind"] == "multiple":
        options = [o["code"] for o in item["options"]]
        result["choices"] = next([code] for code in options if {code} != set(correct["choices"]))
    else:
        row = item["rows"][0]["row_id"]
        result["matches"][row] = next(
            o["code"] for o in item["row_options"] if o["code"] != correct["matches"][row]
        )
    return result


def actions_for(tutorial, resource):
    """Generate the frozen required action payloads from fetched synthetic facts."""
    case = tutorial["case"]
    candidate = case["candidates"][1]["candidate_id"]
    initial = [c["candidate_id"] for c in case["candidates"]]
    fact = resource["facts"][0]["fact_id"]
    for lesson in tutorial["lessons"]:
        actions = []
        for requirement in lesson["requirements"]:
            kind = requirement["action"]
            action = {"requirement_id": requirement["requirement_id"], "action": kind}
            if kind in {"inspect_other_candidate", "return_to_candidate"}:
                action["candidate_id"] = candidate
            elif kind in {
                "search_entity",
                "navigate_parent",
                "navigate_child",
                "return_to_compared",
                "copy_iri",
            }:
                action["entity"] = (
                    resource["hierarchy"][0]["parent" if kind == "navigate_parent" else "child"]
                    if kind in {"navigate_parent", "navigate_child"}
                    else case["source"]
                )
            elif kind in {
                "open_citation",
                "open_original_axiom",
                "locate_in_evidence_list",
                "inspect_graph_or_list",
            }:
                action["fact_id"] = (
                    resource["evidence"][0]["fact_ids"][0]
                    if kind in {"locate_in_evidence_list", "inspect_graph_or_list"}
                    else fact
                )
            elif kind in {
                "add_rank",
                "move_rank",
                "remove_rank",
                "undo_rank",
                "keep_initial_order",
                "check_partial_ranking",
                "choose_none",
                "choose_insufficient",
                "rank_with_details_open",
            }:
                action.update(
                    response_type="ranked_candidates",
                    ranked_candidate_ids=initial if kind == "keep_initial_order" else [candidate],
                )
                if kind in {"choose_none", "choose_insufficient"}:
                    action.update(
                        response_type=(
                            "none_of_these" if kind == "choose_none" else "insufficient_evidence"
                        ),
                        ranked_candidate_ids=[],
                    )
            elif kind in {"report_multiple_methods", "report_no_methods"}:
                action["methods"] = (
                    ["plain_files", "other_method"] if kind == "report_multiple_methods" else []
                )
                if action["methods"]:
                    action["other_method"] = "Automated verification practice"
            elif kind == "locate_downloads":
                action["asset_ids"] = [a["asset_id"] for a in case["ontology_resources"]]
            actions.append(action)
        yield lesson["lesson_id"], actions


def validate_participant_catalogue(state, *, scope):
    """Participant state must never disclose the frozen future-case catalogue."""
    about = state.get("review_bundle")
    if scope == "synthetic_contract_fixture" and about is None:
        return False
    require(isinstance(about, dict), "participant_review_metadata_present")
    require(about.get("cases") == [], "participant_catalogue_not_exposed")
    return True


def validate_catalogues(study_about, main_about, definition, expected_cases):
    """Bind main's full catalogue while checking the participant-safe projection."""
    validate_participant_catalogue({"review_bundle": study_about}, scope="real_bundle")
    require(
        {key: value for key, value in study_about.items() if key != "cases"}
        == {key: value for key, value in main_about.items() if key != "cases"},
        "shared_public_review_provenance",
    )
    require(
        study_about.get("purpose") == "development_review"
        and study_about.get("correctness_metrics") == "disabled",
        "catalogue_development_purpose",
    )
    expected = {
        case["case_id"]: {
            "case_id": case["case_id"],
            "source": case["source"],
            "targets": [candidate["entity"] for candidate in case["candidates"]],
        }
        for case in definition["cases"]
    }
    actual = {case["case_id"]: case for case in main_about["cases"]}
    require(
        len(main_about["cases"]) == len(actual) == len(expected) == expected_cases,
        "catalogue_exact_case_count",
    )
    require(actual == expected, "catalogue_frozen_typed_identities")
    require(
        study_about["source_run_id"] == definition["review_provenance"]["source_run_id"],
        "catalogue_source_run",
    )
    return {
        "case_count": len(actual),
        "target_count": sum(len(case["targets"]) for case in actual.values()),
        "main_sha256": fingerprint(main_about),
        "participant_sha256": fingerprint(study_about),
        "same_public_review_provenance": True,
        "participant_case_catalogue_withheld": True,
    }


def validate_pair_metadata(study, main, source, candidate, run, rows):
    """Compare exact public contracts, including typed pair and selected-row counts."""
    require(study["candidate_id"] == candidate["candidate_id"], "metadata_candidate_binding")
    require(study["source"] == main["source"] == source, "metadata_source_binding")
    require(study["target"] == main["target"] == candidate["entity"], "metadata_target_binding")
    require(
        main["run_id"] == run
        and main["pair_id"]
        == canonical_hash(
            {
                "run_id": run,
                "source": source,
                "target": candidate["entity"],
            }
        ),
        "metadata_main_pair_binding",
    )
    require(study["evidence_metadata"] == main["evidence_metadata"], "same_public_pair_metadata")
    require(len({row["evidence_id"] for row in rows}) == len(rows), "unique_selected_evidence")
    require(
        study["evidence_metadata"]["counts"] == evidence_completeness(rows),
        "metadata_selected_status_counts",
    )
    for row in rows:
        require(
            row["entity"] == (source if row["side"] == "source" else candidate["entity"]),
            "selected_evidence_typed_subject",
        )


def inventory(request_page):
    """Exhaust one frozen entity index with bounded, non-repeating cursor traversal."""
    digest, seen, cursors, samples = hashlib.sha256(), set(), set(), []
    cursor, total, pages = None, None, 0
    while True:
        page = request_page(cursor)
        require(type(page.get("total_count")) is int, "inventory_exact_total_available")
        require(total is None or page["total_count"] == total, "inventory_total_stable")
        total = page["total_count"]
        require(0 <= total <= 1_000_000, "inventory_verification_bound")
        require(page["returned_count"] == len(page["items"]), "inventory_page_count")
        for row in page["items"]:
            key = canonical_hash(row["entity"])
            require(key not in seen, "inventory_unique_typed_entities")
            seen.add(key)
            digest.update(canonical_hash(row).encode() + b"\n")
        if page["items"]:
            if not samples:
                samples.append(page["items"][0]["entity"])
            if len(samples) == 1:
                samples.append(page["items"][-1]["entity"])
            else:
                samples[-1] = page["items"][-1]["entity"]
        pages += 1
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        require(bool(page["items"]) and cursor not in cursors, "inventory_cursor_progress")
        require(pages <= total, "inventory_page_bound")
        cursors.add(cursor)
    require(len(seen) == total, "inventory_exhaustive_count")
    return {"entities": total, "pages": pages, "sha256": digest.hexdigest()}, samples


class Journey:
    def __init__(self, config, *, receipt, expected_cases=24, scope="real_bundle"):
        require(not Path(receipt).exists(), "receipt_must_be_new")
        self.config, self.receipt = config, receipt
        self.expected_cases, self.scope = expected_cases, scope
        self.publication = environment.read_json(config["publication"])
        self.definition = self.publication["definition"]
        environment.validate_publication(self.publication)
        require(
            expected_cases > 0 and (scope != "real_bundle" or expected_cases == 24),
            "case_count_scope",
        )
        self.origin = f"https://127.0.0.1:{config['ports']['study']}"
        context = ssl.create_default_context(cafile=config["certificate"])
        self.client = httpx.Client(
            base_url=self.origin,
            verify=context,
            trust_env=False,
            timeout=httpx.Timeout(90),
            headers={"Origin": self.origin, "Accept-Encoding": "identity"},
        )
        self.admin = {"Authorization": "Bearer " + config["researcher_token"]}
        self.state = None
        self.started = time.monotonic()
        self.report = {
            "schema": "exact-review-api-verification/1",
            "status": "running",
            "scope": scope,
            "boundary": "Actual certificate-verified HTTPS API journey; not a browser or human-activity test",
            "verifier_sha256": environment.file_hash(__file__),
            "publication_sha256": environment.file_hash(config["publication"]),
            "certificate_sha256": environment.file_hash(config["certificate"]),
            "client_contract": "study-review/1",
            "checks": [],
            "interruptions": [],
        }
        self.checkpoint = "initialization"
        self.study_record = environment.processes(config).get("study")
        require(
            self.study_record and environment.owned(config, "study", self.study_record),
            "owned_study_process",
        )
        self.study_log = Path(config["root"]) / "logs" / "study.log"
        self.log_offset = self.study_log.stat().st_size

    def record(self):
        self.report["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        self.report["check_counts"] = dict(Counter(item["check"] for item in self.report["checks"]))
        environment.private_json(self.receipt, self.report)

    def request(self, method, path, *, label, expected=200, **kwargs):
        self.checkpoint = label
        start = time.monotonic()
        response = self.client.request(method, path, **kwargs)
        length = response.headers.get("content-length")
        require(length is None or int(length) == len(response.content), label + ":body_length")
        csp = validate_response_csp(
            response, study_origin=str(self.client.base_url).rstrip("/"), label=label
        )
        self.report["checks"].append(
            {
                "check": label,
                "status": response.status_code,
                "csp_contract": csp,
                "bytes": len(response.content),
                "sha256": fingerprint(response.content),
                "seconds": round(time.monotonic() - start, 3),
            }
        )
        require(response.status_code == expected, label + ":http_status")
        if csp != "not_promised_by_exploration_api" and response.headers.get(
            "content-type", ""
        ).startswith("application/json"):
            data = response.json()
            if isinstance(data, dict) and "session_id" in data and "stage" in data:
                if validate_participant_catalogue(data, scope=self.scope):
                    self.report["participant_states_with_catalogue_withheld"] = (
                        self.report.get("participant_states_with_catalogue_withheld", 0) + 1
                    )
        return response

    def mutate(self, method, path, **payload):
        body = {
            "idempotency_key": uuid4().hex,
            "expected_revision": self.state["revision"],
            **payload,
        }
        self.state = self.request(
            method, "/api/v1/study/" + path, label="mutation:" + path.split("/")[0], json=body
        ).json()
        return self.state

    def timing(self, stage, page, start, end, case=None):
        require(end >= start, "monotonic_timing")
        while start < end:
            stop = min(start + 60000, end)
            payload = {
                "segment_id": uuid4().hex,
                "page_instance_id": page,
                "stage": stage,
                "monotonic_start_ms": start,
                "monotonic_end_ms": stop,
            }
            if case:
                payload.update(case_id=case["case_id"], presentation_id=case["presentation_id"])
            ack = self.request(
                "POST", "/api/v1/study/timing", label="actual_elapsed_timing", json=payload
            ).json()
            require(ack["availability"] == "observed", "timing_observed")
            start = stop

    def pair_metadata(self, prefix, case, capabilities):
        contract = capabilities.get("evidence_metadata_contract")
        if contract is None:
            require(self.scope == "synthetic_contract_fixture", "real_pair_metadata_available")
            for candidate in case["candidates"]:
                self.request(
                    "GET",
                    prefix + "/evidence-metadata",
                    params={"candidate_id": candidate["candidate_id"]},
                    label="legacy_pair_metadata_not_exported",
                    expected=404,
                )
            return
        require(contract == "exact-evidence-metadata/1", "pair_metadata_contract")
        record = environment.processes(self.config).get("exploration")
        require(
            record and environment.owned(self.config, "exploration", record),
            "owned_exploration_process",
        )
        run = self.definition["review_provenance"]["source_run_id"]
        main_prefix = (
            f"https://127.0.0.1:{self.config['ports']['exploration']}"
            + "/api/v1/runs/"
            + quote(run, safe="")
        )
        for candidate in case["candidates"]:
            query = {"candidate_id": candidate["candidate_id"]}
            metadata = self.request(
                "GET",
                prefix + "/evidence-metadata",
                params=query,
                label="study_pair_metadata",
            ).json()
            rows, cursor, seen = [], None, set()
            while True:
                page = self.request(
                    "GET",
                    prefix + "/evidence",
                    params={**query, "limit": 100, **({"cursor": cursor} if cursor else {})},
                    label="study_selected_evidence",
                ).json()
                require(
                    page["returned_count"] == len(page["items"]), "selected_evidence_page_count"
                )
                rows.extend(page["items"])
                cursor = page.get("next_cursor")
                if cursor is None:
                    break
                require(cursor not in seen, "selected_evidence_cursor_progress")
                seen.add(cursor)
            pair_id = canonical_hash(
                {"run_id": run, "source": case["source"], "target": candidate["entity"]}
            )
            main = self.request(
                "GET",
                main_prefix + "/pair",
                params={"pair_id": pair_id},
                label="main_pair_metadata",
            ).json()
            validate_pair_metadata(metadata, main, case["source"], candidate, run, rows)
        require(
            environment.owned(self.config, "exploration", record), "exploration_identity_preserved"
        )
        self.request(
            "GET",
            prefix + "/evidence-metadata",
            params={"candidate_id": "not-a-case-candidate"},
            label="foreign_candidate_metadata_denied",
            expected=403,
        )

    def hierarchy_probe(self, prefix, main, version, candidates):
        """Find a real nonempty class without assuming every focal class has edges."""
        unique = {}
        for entity in candidates:
            if entity["ontology_version_id"] == version and entity["kind"] == "class":
                unique.setdefault(canonical_hash(entity), entity)
        for attempted, hierarchy_probe in enumerate(list(unique.values())[:200], start=1):
            entity_query = {**hierarchy_probe, "limit": 100, "basis": "literal_asserted"}
            hierarchy_counts = {}
            for direction in ("parents", "children"):
                responses, continuations = [], []
                for path, label in (
                    (prefix + "/hierarchy", "study_hierarchy_page"),
                    (main + "/hierarchy", "main_hierarchy_page"),
                ):
                    page = self.request(
                        "GET", path, params={**entity_query, "direction": direction}, label=label
                    ).json()
                    require(
                        page.get("status") in {"available", "absent_in_scope"},
                        "hierarchy_page_available",
                    )
                    require(page["returned_count"] == len(page["items"]), "hierarchy_page_count")
                    responses.append(page)
                    if page.get("next_cursor"):
                        continuation = self.request(
                            "GET",
                            path,
                            params={
                                **entity_query,
                                "direction": direction,
                                "cursor": page["next_cursor"],
                            },
                            label=label + "2",
                        ).json()
                        require(
                            continuation.get("status") in {"available", "absent_in_scope"},
                            "hierarchy_continuation_available",
                        )
                        require(
                            continuation["returned_count"] == len(continuation["items"]),
                            "hierarchy_continuation_count",
                        )
                        continuations.append(continuation["items"])
                        require(
                            not (
                                {canonical_hash(r) for r in page["items"]}
                                & {canonical_hash(r) for r in continuation["items"]}
                            ),
                            "hierarchy_page_continuation",
                        )
                    else:
                        continuations.append([])
                require(
                    bool(responses[0].get("next_cursor")) == bool(responses[1].get("next_cursor")),
                    "same_hierarchy_continuation_presence",
                )
                require(responses[0]["status"] == responses[1]["status"], "same_hierarchy_status")
                require(responses[0]["items"] == responses[1]["items"], "same_hierarchy_edges")
                require(continuations[0] == continuations[1], "same_hierarchy_continuation")
                hierarchy_counts[direction] = {
                    "first_page_edges": len(responses[0]["items"]),
                    "continuation_edges": len(continuations[0]),
                    "continuation_exercised": bool(responses[0].get("next_cursor")),
                }
            if any(value["first_page_edges"] for value in hierarchy_counts.values()):
                return hierarchy_probe, hierarchy_counts, attempted
        raise VerificationError("nonempty_class_hierarchy_exercised")

    def complete_read_surfaces(self, prefix, capabilities):
        """Check full indexed data and graph inputs; this does not render a graph."""
        record = environment.processes(self.config).get("exploration")
        require(
            record and environment.owned(self.config, "exploration", record),
            "owned_exploration_process",
        )
        main = f"https://127.0.0.1:{self.config['ports']['exploration']}/api/v1"
        summaries = []
        focal_entities = [
            entity
            for case in self.definition["cases"]
            for entity in [case["source"], *[c["entity"] for c in case["candidates"]]]
        ]
        focal = {canonical_hash(entity) for entity in focal_entities}
        for version in capabilities["ontology_versions"]:
            started = time.monotonic()
            query = {"ontology_version_id": version, "limit": 100}

            def pages(path, label):
                return inventory(
                    lambda cursor: self.request(
                        "GET",
                        path,
                        params={**query, **({"cursor": cursor} if cursor else {})},
                        label=label,
                    ).json()
                )

            study, samples = pages(prefix + "/entities", "study_full_index_page")
            exploration, _ = pages(main + "/entities", "main_full_index_page")
            require(study == exploration, "same_full_typed_entity_inventory")
            require(bool(samples), "full_ontology_inventory_nonempty")
            probe = next((e for e in samples if canonical_hash(e) not in focal), samples[0])
            for path, label in (
                (prefix + "/entities", "study_exact_iri_search"),
                (main + "/entities", "main_exact_iri_search"),
            ):
                found = self.request(
                    "GET",
                    path,
                    params={**query, "term": probe["iri"], "kind": probe["kind"]},
                    label=label,
                ).json()
                require(probe in [r["entity"] for r in found["items"]], "exact_iri_search_identity")
            hierarchy_probe, hierarchy_counts, attempted = self.hierarchy_probe(
                prefix, main, version, [*focal_entities, *samples]
            )
            entity_query = {**hierarchy_probe, "limit": 100, "basis": "literal_asserted"}
            graph = self.request(
                "GET",
                main + "/hierarchy",
                params={**entity_query, "direction": "ancestors"},
                label="main_bounded_ancestor_graph",
            ).json()
            require(
                len(graph["nodes"]) <= 100 and hierarchy_probe in graph["nodes"],
                "ancestor_graph_node_bound",
            )
            summaries.append(
                {
                    "ontology_sha256": fingerprint(version),
                    **study,
                    "same_study_and_exploration_inventory": True,
                    "search_probe_outside_focal_cases": canonical_hash(probe) not in focal,
                    "ancestor_graph_nodes": len(graph["nodes"]),
                    "ancestor_graph_edges": len(graph["items"]),
                    "hierarchy": hierarchy_counts,
                    "hierarchy_probes_attempted": attempted,
                    "hierarchy_empty_probes": attempted - 1,
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
        require(
            environment.owned(self.config, "exploration", record), "exploration_identity_preserved"
        )
        self.report["complete_read_surfaces"] = {
            "ontologies": summaries,
            "scope": "Full entity-index pagination, exact-IRI search, hierarchy and bounded ancestor graph API inputs; no frontend graph rendering claim",
        }

    def workspace(self, scope_id, case, *, interrupt=False):
        source = case["source"]
        prefix = "/api/v1/study/workspace/" + quote(scope_id, safe="")
        capabilities = self.request(
            "GET", prefix + "/capabilities", label="workspace_capabilities"
        ).json()
        require(capabilities["scope_id"] == scope_id, "workspace_scope_binding")
        self.pair_metadata(prefix, case, capabilities)
        if interrupt and self.scope == "real_bundle":
            self.complete_read_surfaces(prefix, capabilities)
        for version in capabilities["ontology_versions"]:
            query = {"ontology_version_id": version, "kind": "class", "limit": 1}
            first = self.request(
                "GET", prefix + "/entry-points", params=query, label="entry_points_page1"
            ).json()
            require(first["returned_count"] == len(first["items"]), "entry_point_count")
            if first.get("next_cursor"):
                second = self.request(
                    "GET",
                    prefix + "/entry-points",
                    params={**query, "cursor": first["next_cursor"]},
                    label="entry_points_page2",
                ).json()
                require(
                    not (
                        {fingerprint(x) for x in first["items"]}
                        & {fingerprint(x) for x in second["items"]}
                    ),
                    "entry_point_continuation",
                )
        query = {key: source[key] for key in ("ontology_version_id", "iri", "kind")}
        target = prefix + "/entity-context"
        original = self.request("GET", target, params=query, label="entity_context_full")
        facts = self.request(
            "GET", prefix + "/entity-facts", params={**query, "limit": 1}, label="entity_facts_page"
        ).json()
        if facts["items"]:
            fact = facts["items"][0]
            self.request(
                "GET",
                prefix + "/facts/" + quote(fact["fact_id"], safe=""),
                params=query,
                label="original_fact",
            )
            self.request(
                "GET",
                prefix + "/axioms/" + quote(fact["axiom_ref"], safe=""),
                params=query,
                label="original_axiom",
            )
        if set(capabilities["components"]) & {"profiles", "comparison"}:
            explanations = self.request(
                "GET",
                prefix + "/explanations",
                params={**query, "limit": 2},
                label="prepared_explanations_page",
            ).json()
            for explanation in explanations["items"]:
                self.request(
                    "GET",
                    prefix + "/explanations/" + quote(explanation["explanation_id"], safe=""),
                    label="prepared_explanation_full",
                )
        if interrupt:
            for _ in range(3):
                start = time.monotonic()
                with self.client.stream("GET", target, params=query) as response:
                    require(response.status_code == 200, "interrupted_response_status")
                    validate_response_csp(
                        response, study_origin=self.origin, label="interrupted_response"
                    )
                    declared = int(response.headers.get("content-length", "0"))
                    chunk = next(response.iter_raw(chunk_size=128))
                    require(len(chunk) < declared, "partial_response_probe_required")
                recovered = self.request(
                    "GET", target, params=query, label="post_interrupt_full_body"
                )
                require(recovered.content == original.content, "interrupted_response_recovery_hash")
                self.request("GET", "/api/v1/study/state", label="post_interrupt_state")
                self.report["interruptions"].append(
                    {
                        "scenario": "authenticated_workspace_response_closed_after_first_128_bytes",
                        "declared_bytes": declared,
                        "received_first_chunk_bytes": len(chunk),
                        "full_body_sha256": fingerprint(recovered.content),
                        "seconds": round(time.monotonic() - start, 3),
                        "limitation": "Client closed the response early; this does not prove server-side cancellation or browser reload behavior",
                    }
                )

    def catalogue(self):
        about = self.state.get("review_bundle")
        if self.scope == "synthetic_contract_fixture" and not (about or {}).get("cases"):
            self.report["public_catalogue"] = {"status": "not_in_synthetic_contract_fixture"}
            return
        validate_participant_catalogue(self.state, scope=self.scope)
        record = environment.processes(self.config).get("exploration")
        require(
            record and environment.owned(self.config, "exploration", record),
            "owned_exploration_process",
        )
        main = self.request(
            "GET",
            f"https://127.0.0.1:{self.config['ports']['exploration']}/api/v1/review-bundle",
            label="public_exploration_catalogue",
        ).json()
        require(
            environment.owned(self.config, "exploration", record), "exploration_identity_preserved"
        )
        self.report["public_catalogue"] = validate_catalogues(
            about, main, self.definition, self.expected_cases
        )

    def prepare(self, invitation=None):
        page = self.request("GET", "/participate/", label="served_participation_page")
        require(bool(page.headers.get("content-security-policy")), "normal_csp_present")
        self.request("GET", "/api/ready", label="study_ready")
        if invitation:
            invitation = environment.read_json(invitation)
        else:
            invitation = self.request(
                "POST",
                f"/api/v1/admin/studies/{quote(self.config['study_revision'], safe='')}/invitations",
                label="test_invitation",
                headers=self.admin,
                json={"test": True, "count": 1},
            ).json()["invitations"][0]
        url = invitation.get("invitation_url") or invitation["invitation"]
        parsed = urlsplit(url)
        query = parse_qs(parsed.fragment or parsed.query)
        secret = query["invite"][0]
        self.request(
            "POST",
            "/api/v1/study/session",
            label="legacy_client_refused",
            expected=409,
            json={"secret": secret},
        )
        self.state = self.request(
            "POST",
            "/api/v1/study/session",
            label="negotiated_session",
            json={"secret": secret, "client_contract": "study-review/1"},
        ).json()
        self.client.headers["X-Study-Session"] = self.state["session_id"]
        require(
            self.state["purpose"] == "development_review"
            and self.state["correctness_metrics"] == "disabled",
            "development_only",
        )
        self.report["session_sha256"] = fingerprint(self.state["session_id"])
        self.catalogue()
        self.mutate(
            "PUT",
            "consent",
            accepted=True,
            information_version=self.definition["information_version"],
        )
        self.mutate(
            "PUT",
            "setup",
            setup_version="setup/3",
            instructions_acknowledged=True,
            external_inspection_optional_understood=True,
            resource_access="available",
            submitted=True,
            familiar_methods=["plain_files", "other_method"],
            familiar_other_method="API verification method",
        )
        self.mutate(
            "PUT",
            "questionnaires/background",
            form_version="exact-study-forms/3",
            answers=answers_for(self.state["forms"]["background"]),
            submitted=True,
        )
        require(self.state["stage"] == "tutorial", "tutorial_stage")

    def tutorial(self):
        started = time.monotonic()
        page = uuid4().hex
        tutorial = self.definition["tutorial"]
        require("grading" not in self.state["tutorial"], "tutorial_grading_private")
        resource = self.request(
            "GET",
            "/api/v1/study/resources/" + quote(tutorial["case"]["explanation_refs"][0], safe=""),
            label="frozen_tutorial_resource",
        ).json()
        action_count = 0
        for lesson, actions in actions_for(tutorial, resource):
            self.mutate(
                "PUT",
                "tutorial/progress",
                tutorial_version=tutorial["version"],
                lesson_id=lesson,
                current_lesson_id=lesson,
                position={"view": "lesson", "lesson_id": lesson, "question_id": None},
                actions=actions,
                completed_requirements=[a["requirement_id"] for a in actions],
            )
            action_count += len(actions)
        for item in tutorial["assessment"]:
            correct = {
                key: value
                for key, value in tutorial["grading"][item["question_id"]]["response"].items()
                if value is not None
            }
            for answer, expected in ((wrong_answer(item, correct), False), (correct, True)):
                self.mutate(
                    "POST",
                    "tutorial/assessment",
                    tutorial_version=tutorial["version"],
                    question_id=item["question_id"],
                    attempt_id=uuid4().hex,
                    response=answer,
                )
                require(
                    self.state["tutorial_progress"]["attempts"][-1]["correct"] is expected,
                    "tutorial_assessment_feedback",
                )
        # Keep an honest single-client clock; no synthetic elapsed values or parallel-page sum.
        while time.monotonic() - started < 120:
            time.sleep(max(0, min(1, 120 - (time.monotonic() - started))))
        elapsed = (time.monotonic() - started) * 1000
        self.timing("tutorial", page, 0, elapsed)
        self.mutate("POST", "tutorial/complete", tutorial_version=tutorial["version"])
        require(self.state["stage"] == "case", "allocation_after_tutorial")
        self.report["tutorial"] = {
            "required_actions": action_count,
            "wrong_attempts": 5,
            "correct_attempts": 5,
            "actual_elapsed_seconds": round(elapsed / 1000, 3),
            "observation_kind": "Automated API client elapsed time, not human engagement",
        }
        self.record()

    def cases(self):
        conditions = Counter()
        interrupted = False
        for index in range(self.expected_cases):
            require(self.state["stage"] == "case", "expected_case_stage")
            case = self.request("GET", "/api/v1/study/cases/current", label="current_case").json()
            conditions[case["condition"]] += 1
            page, started = uuid4().hex, time.monotonic()
            if case["condition"] == "explanation":
                require(case.get("workspace"), "explanation_workspace_present")
                self.workspace(case["workspace"]["scope_id"], case, interrupt=not interrupted)
                interrupted = True
                for ref in case["explanation_refs"]:
                    self.request(
                        "GET",
                        "/api/v1/study/resources/" + quote(ref, safe=""),
                        label="current_explanation_resource",
                    )
            else:
                require(
                    case.get("workspace") is None and case["explanation_refs"] == [],
                    "baseline_filtered",
                )
                scope = next(
                    s
                    for s in self.definition["workspace_scopes"]
                    if s.get("case_id") == case["case_id"]
                )
                self.request(
                    "GET",
                    "/api/v1/study/workspace/"
                    + quote(scope["scope_id"], safe="")
                    + "/capabilities",
                    label="baseline_workspace_denied",
                    expected=403,
                )
                for candidate in case["candidates"]:
                    self.request(
                        "GET",
                        "/api/v1/study/workspace/"
                        + quote(scope["scope_id"], safe="")
                        + "/evidence-metadata",
                        params={"candidate_id": candidate["candidate_id"]},
                        label="baseline_pair_metadata_denied",
                        expected=403,
                    )
            ready = (time.monotonic() - started) * 1000
            event = {
                "event_id": uuid4().hex,
                "page_instance_id": page,
                "sequence": 0,
                "scope": "case",
                "case_id": case["case_id"],
                "presentation_id": case["presentation_id"],
                "type": "case_ready",
                "client_monotonic_ms": ready,
                "build_version": self.definition["software_version"],
            }
            self.request(
                "POST",
                "/api/v1/study/events",
                label="case_usable_observation",
                json={"events": [event]},
            )
            kind = ("ranked_candidates", "none_of_these", "insufficient_evidence")[index % 3]
            ranked = (
                [c["candidate_id"] for c in case["candidates"][:2]]
                if kind == "ranked_candidates"
                else []
            )
            self.timing("case", page, ready, (time.monotonic() - started) * 1000, case)
            self.mutate(
                "POST",
                f"cases/{case['case_id']}/submit",
                presentation_id=case["presentation_id"],
                response_type=kind,
                ranked_candidate_ids=ranked,
            )
            require(self.state["stage"] == "consultation", "consultation_stage")
            body = {
                "form_version": "exact-study-forms/3",
                "consultation_version": "consultation/3",
                "presentation_id": case["presentation_id"],
            }
            if index % 3 == 0:
                body.update(
                    consulted_external_ontologies=True,
                    methods=["protege", "plain_files", "other_method"],
                    other_method="API verification method",
                )
            elif index % 3 == 1:
                body.update(
                    consulted_external_ontologies=False,
                    methods=["other_method"],
                    other_method="Hidden verification value",
                )
            else:
                body.update(consulted_external_ontologies=True, methods=["plain_files"])
            self.mutate("PUT", f"cases/{case['case_id']}/consultation/draft", **body)
            if body["consulted_external_ontologies"] is False:
                require(
                    self.state["consultation_draft"]["methods"] == []
                    and self.state["consultation_draft"]["other_method"] is None,
                    "hidden_consultation_cleared",
                )
            self.mutate("PUT", f"cases/{case['case_id']}/consultation", **body)
            self.report["completed_cases"] = index + 1
            self.report["conditions"] = dict(conditions)
            self.record()
        require(self.state["stage"] == "final", "all_cases_before_final")
        require(set(conditions) == {"explanation", "ontology_baseline"}, "both_assigned_conditions")
        self.mutate(
            "PUT",
            "questionnaires/final",
            form_version="exact-study-forms/3",
            answers=answers_for(self.state["forms"]["final"]),
            submitted=True,
        )
        self.mutate("POST", "complete")
        require(self.state["stage"] == "completed", "completed_stage")

    def exports(self):
        sid = self.state["session_id"]
        records = []
        for named in (False, True):
            exported = self.request(
                "POST",
                f"/api/v1/admin/studies/{quote(self.config['study_revision'], safe='')}/exports",
                label="restricted_export" if named else "default_export",
                headers=self.admin,
                params={"include_test": "true", "include_named_methods": str(named).lower()},
            ).json()
            manifest, data = exported["manifest"], exported["data"]
            require(
                manifest["schema"] == "exact-study-analysis/4"
                and manifest["includes_named_method_details"] is named,
                "export_contract",
            )
            require(
                data["correctness_metrics"] == "disabled"
                and data["summary"] == {"status": "disabled_development_review"},
                "correctness_disabled",
            )
            session = next(s for s in data["sessions"] if s["session_id"] == sid)
            require(
                session["test"]
                and session["stage"] == "completed"
                and len(session["cases"]) == self.expected_cases,
                "export_completed_test_session",
            )
            require(
                all(row["first_response"] and row["consultation"] for row in session["cases"]),
                "all_responses_exported",
            )
            for row in session["cases"]:
                require(
                    row["timing"]["active_duration_known"] is False
                    and row["timing"]["unique_elapsed_coverage_seconds"] is None,
                    "analysis3_clock_uncertainty_preserved",
                )
            content = json.dumps(exported)
            require("Hidden verification value" not in content, "hidden_method_excluded")
            require(("API verification method" in content) is named, "named_method_opt_in")
            export_path = "/api/v1/admin/exports/" + manifest["export_id"]
            saved = self.request(
                "GET", export_path, label="immutable_json_export", headers=self.admin
            ).json()
            require(fingerprint(saved) == fingerprint(exported), "saved_json_immutable")
            csv = self.request(
                "GET",
                export_path,
                label="immutable_csv_export",
                headers=self.admin,
                params={"format": "csv"},
            )
            repeat = self.request(
                "GET",
                export_path,
                label="repeat_csv_export",
                headers=self.admin,
                params={"format": "csv"},
            )
            require(csv.content == repeat.content, "saved_csv_immutable")
            with zipfile.ZipFile(io.BytesIO(csv.content)) as archive:
                require("method_responses.csv" in archive.namelist(), "method_csv_table")
                require(
                    (b"API verification method" in archive.read("method_responses.csv")) is named,
                    "csv_named_method_opt_in",
                )
                tables = sorted(archive.namelist())
            records.append(
                {
                    "named_methods": named,
                    "json_sha256": fingerprint(exported),
                    "csv_sha256": fingerprint(csv.content),
                    "tables": tables,
                }
            )
        self.report["exports"] = records

    def run(self, invitation=None):
        try:
            self.prepare(invitation)
            print(json.dumps({"phase": "tutorial", "wait": "120 real elapsed seconds"}), flush=True)
            self.tutorial()
            print(json.dumps({"phase": "cases", "count": self.expected_cases}), flush=True)
            self.cases()
            self.exports()
            require(
                environment.owned(self.config, "study", self.study_record),
                "study_process_identity_preserved",
            )
            require(self.study_log.stat().st_size >= self.log_offset, "service_log_not_truncated")
            with self.study_log.open("rb") as stream:
                stream.seek(self.log_offset)
                log = stream.read()
            signatures = {
                "content_length_protocol_errors": b"Too little data for declared Content-Length",
                "h11_local_protocol_errors": b"LocalProtocolError",
                "asgi_exceptions": b"Exception in ASGI application",
                "tracebacks": b"Traceback (most recent call last)",
            }
            counts = {key: log.count(pattern) for key, pattern in signatures.items()}
            self.report["service_log_observation"] = {
                "new_bytes": len(log),
                "sha256": fingerprint(log),
                "counts": counts,
                "scope": "Owned study process log during this API journey; no historical-causation claim",
            }
            require(not any(counts.values()), "no_server_protocol_exceptions_during_journey")
            self.report["status"] = "passed"
        except Exception as error:
            self.report.update(
                status="failed", failed_check=self.checkpoint, exception_type=type(error).__name__
            )
            if isinstance(error, VerificationError):
                self.report["failure"] = str(error)
            raise
        finally:
            self.record()
            self.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--invitation", type=Path)
    parser.add_argument("--expected-cases", type=int, default=24)
    parser.add_argument(
        "--scope", choices=("real_bundle", "synthetic_contract_fixture"), default="real_bundle"
    )
    args = parser.parse_args()
    try:
        journey = Journey(
            environment.load(args.root),
            receipt=args.receipt,
            expected_cases=args.expected_cases,
            scope=args.scope,
        )
        journey.run(args.invitation)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "receipt": str(args.receipt),
                }
            ),
            flush=True,
        )
        return 1
    print(json.dumps({"status": "passed", "receipt": str(args.receipt)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
