"""Verify predecessor preservation and actual HTTPS parity in test-only study sessions.

This extends the existing API journey. It does not claim rendered frontend acceptance.
Run under the shared storage guard. No ontology classification or provider dispatch occurs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.descriptions import preparation_metadata
from exact_inspect.models import SelectedEvidence
from exact_inspect.study.resources import ExplanationResource
from tools import review_environment as environment
from tools.verify_review_environment import Journey, require, validate_pair_metadata

BASIS = "elk_inferred_partial"


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(root, case, candidate):
    bundle = read(root / "package/package.json")
    run = root / "package" / next(iter(bundle["runs"].values())) / "decisions.sqlite"
    with sqlite3.connect("file:" + str(run) + "?mode=ro", uri=True) as db:
        value = db.execute(
            "SELECT id,evidence,trace FROM pairs WHERE source=? AND source_kind=? AND target=? AND target_kind=?",
            (
                case["source"]["iri"],
                case["source"]["kind"],
                candidate["entity"]["iri"],
                candidate["entity"]["kind"],
            ),
        ).fetchone()
    require(value is not None, "presented_pair_in_predecessor")
    return value[0], json.loads(value[1]), json.loads(value[2])


def taxonomy_audit(root):
    """Compare every served member and edge with the independently pinned native output."""
    bundle = read(root / "package/package.json")
    out = []
    for version, locator in bundle["ontologies"].items():
        context = root / "package" / locator
        manifest = read(context / "manifest.json")
        info = manifest["partial_navigation"]["inference"]
        name = manifest["name"].lower()
        native = root.parent / "elk/results/14455-run" / name / "taxonomy.partial.json"
        require(file_hash(native) == info["artifact_sha256"], "retained_native_taxonomy_identity")
        data = read(native)
        with sqlite3.connect(
            "file:" + str(context / "context.sqlite") + "?mode=ro", uri=True
        ) as c, sqlite3.connect(
            "file:" + str(context / "partial-navigation.sqlite") + "?mode=ro", uri=True
        ) as n:
            classes = {x[0] for x in c.execute("SELECT iri FROM entities WHERE kind='class'")}
            expected = {
                (a, b)
                for child, parent in data["direct_edges"]
                for a in child
                for b in parent
                if a in classes and b in classes
            }
            actual = set(n.execute("SELECT child,parent FROM edges"))
            require(actual == expected, "all_native_direct_edges_preserved_without_closure")
            expected_members = {
                (
                    iri,
                    "class",
                    canonical_hash(sorted(group)),
                    int(sorted(group) == sorted(data["top"])),
                    int(sorted(group) == sorted(data["bottom"])),
                )
                for group in data["nodes"]
                for iri in group
                if iri in classes
            }
            require(
                set(n.execute("SELECT * FROM members")) == expected_members,
                "native_equivalence_bounds_membership",
            )
            literal = set(
                c.execute(
                    "SELECT child,parent FROM edges WHERE kind='class' AND basis='literal_asserted'"
                )
            )
            structural = set(
                c.execute(
                    "SELECT child,parent FROM edges WHERE kind='class' AND basis='structural_navigation'"
                )
            )
            added = sorted(actual - literal)
            seen = set()
            examples = []

            def label(iri):
                row = c.execute(
                    "SELECT term,fact_id FROM terms WHERE iri=? AND kind='class' AND category='labels' ORDER BY (language='') DESC,language,fact_id LIMIT 1",
                    (iri,),
                ).fetchone()
                return {"value": row[0], "fact_id": row[1]} if row else None

            for child, parent in added:
                if child not in seen:
                    examples.append(
                        {
                            "entity": {
                                "ontology_version_id": version,
                                "iri": child,
                                "kind": "class",
                            },
                            "additional_parent": parent,
                            "child_label": label(child),
                            "parent_label": label(parent),
                        }
                    )
                    seen.add(child)
                if len(examples) == 3:
                    break
            out.append(
                {
                    "ontology_version_id": version,
                    "name": name,
                    "native_taxonomy_sha256": file_hash(native),
                    "native_group_edges": len(data["direct_edges"]),
                    "served_named_edges": len(actual),
                    "literal_named_edges": len(literal),
                    "structural_named_edges": len(structural),
                    "additional_partial_direct_pairs": len(actual - literal),
                    "asserted_pairs_absent_from_partial": len(literal - actual),
                    "unrepresented_classes": sorted(classes - {x[0] for x in expected_members}),
                    "nontrivial_native_equivalence_groups": sum(len(g) > 1 for g in data["nodes"]),
                    "useful_difference_examples": examples,
                    "limitation": "Directness is relative to the returned partial taxonomy; differences imply no biomedical equivalence or contradiction.",
                }
            )
    return out


def static_audit(combined, evidence, prose):
    publication = read(combined / "publication.json")
    bundle = read(combined / "package/package.json")
    old = read(evidence / "package/package.json")
    run = next(iter(bundle["runs"]))
    database = combined / "package" / bundle["runs"][run] / "decisions.sqlite"
    require(
        file_hash(database)
        == file_hash(evidence / "package" / old["runs"][run] / "decisions.sqlite"),
        "full_recovered_database_preserved",
    )
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as db:
        require(db.execute("SELECT count(*) FROM pairs").fetchone()[0] == 30374, "full_pool_30374")
        require(db.execute("SELECT count(*) FROM sources").fetchone()[0] == 300, "source_pool_300")
    counts = Counter()
    bridges = Counter()
    cases = []
    examples = []
    unavailable = []
    approved = []
    before = read(prose / "package/package.json")
    for key, locator in before["explanations"].items():
        original = read(prose / "package" / locator)
        if original.get("support_status") == "independently_reviewed":
            require(key in bundle["explanations"], "approved_explanation_identity_preserved")
            require(
                file_hash(prose / "package" / locator)
                == file_hash(combined / "package" / bundle["explanations"][key]),
                "approved_explanation_bytes_preserved",
            )
            approved.extend(c["claim_id"] for c in original["claims"])
    require(len(approved) == 68, "all_68_approved_claims")
    for case in publication["definition"]["cases"]:
        asset = next(
            a
            for a in publication["definition"]["assets"]
            if a["asset_id"] == case["explanation_refs"][0]
        )
        resource = read(combined / asset["path"])
        ExplanationResource.model_validate(resource)
        pairs = []
        for candidate in case["candidates"]:
            pid, ev, trace = rows(evidence, case, candidate)
            _, current, ct = rows(combined, case, candidate)
            require(ev == current and trace == ct, "exact_predecessor_features_and_bridges")
            frozen = {
                e["evidence_id"]: e
                for e in resource["evidence"]
                if e["candidate_id"] == candidate["candidate_id"]
            }
            require(set(frozen) == {e["evidence_id"] for e in ev}, "frozen_feature_identity")
            for e in ev:
                f = frozen[e["evidence_id"]]
                require(
                    all(
                        f[k] == e[k] for k in ["fact_ids", "origin_resolution", "graph_projection"]
                    ),
                    "frozen_original_links",
                )
                counts[e["origin_resolution"]] += 1
            require(
                resource["pair_metadata"][candidate["candidate_id"]]["evidence_metadata"]
                == trace["evidence_metadata"],
                "frozen_bridge_parity",
            )
            bridges.update(b["status"] for b in trace["evidence_metadata"]["bridges"])
            pairs.append(
                {"candidate_id": candidate["candidate_id"], "pair_id": pid, "features": len(ev)}
            )
        cases.append({"case_id": case["case_id"], "pairs": pairs})
    for key, locator in bundle["explanations"].items():
        result = read(combined / "package" / locator)
        meta = preparation_metadata(result)
        if result["preparation_kind"] == "factual_fallback":
            unavailable.append(
                {
                    "id": key,
                    "entities": result["entities"],
                    "task": result["task"],
                    "reasons": result.get("support_reasons", []),
                }
            )
        if meta["presentation"]["kind"] in [
            "necessary_conditions",
            "source_only_profile",
            "target_only_profile",
            "paired_context",
        ]:
            examples.append(
                {
                    "id": key,
                    "entities": result["entities"],
                    "task": result["task"],
                    "presentation": meta["presentation"],
                    "claims": result["claims"],
                }
            )
    require(counts == Counter(resolved=1079, unresolved=2364), "recovered_origin_counts")
    require(bridges == Counter(recorded=812, withheld=2517), "recovered_bridge_counts")
    return {
        "publication_sha256": file_hash(combined / "publication.json"),
        "package_sha256": file_hash(combined / "package/package.json"),
        "evidence_predecessor": file_hash(evidence / "publication.json"),
        "prose_predecessor": file_hash(prose / "publication.json"),
        "selected_feature_counts": dict(counts),
        "bridge_counts": dict(bridges),
        "approved_claims": len(approved),
        "cases": cases,
        "presentation_examples": examples,
        "remaining_fallbacks": unavailable,
        "taxonomy": taxonomy_audit(combined),
    }


class CombinedJourney(Journey):
    def __init__(self, config, *, receipt, combined, evidence, shared):
        super().__init__(config, receipt=receipt)
        self.combined, self.evidence, self.shared = combined, evidence, shared
        self.main = f"https://127.0.0.1:{config['ports']['exploration']}/api/v1"
        self.bundle = read(combined / "package/package.json")

    def pages(self, path, params, label):
        rows = []
        cursor = None
        seen = set()
        pages = 0
        first = None
        while True:
            page = self.request(
                "GET",
                path,
                params={**params, **({"cursor": cursor} if cursor else {})},
                label=label,
            ).json()
            require(page["returned_count"] == len(page["items"]), "page_returned_count")
            first = first or page
            rows += page["items"]
            pages += 1
            require(pages <= 100, "bounded_verifier_page_count")
            cursor = page.get("next_cursor")
            if not cursor:
                break
            require(cursor not in seen, "cursor_advances")
            seen.add(cursor)
        require(len({canonical_hash(x) for x in rows}) == len(rows), "no_duplicate_page_items")
        return rows, pages, first

    def request(self, method, path, *, label, expected=200, **kwargs):
        response = super().request(method, path, label=label, expected=expected, **kwargs)
        if label == "baseline_workspace_denied":
            prefix = path.removesuffix("/capabilities")
            entity = self.definition["cases"][0]["source"]
            for route in ["entry-points", "hierarchy", "equivalents"]:
                super().request(
                    "GET",
                    prefix + "/" + route,
                    params={**entity, "basis": BASIS},
                    label="baseline_partial_" + route + "_denied",
                    expected=403,
                )
        return response

    def pair_metadata(self, prefix, case, capabilities):
        require(
            capabilities.get("evidence_metadata_contract") == "exact-evidence-metadata/1",
            "pair_metadata_contract",
        )
        owned = environment.processes(self.config).get("exploration")
        require(
            owned and environment.owned(self.config, "exploration", owned),
            "owned_exploration_process",
        )
        run = self.definition["review_provenance"]["source_run_id"]
        resource = self.request(
            "GET",
            "/api/v1/study/resources/" + quote(case["explanation_refs"][0], safe=""),
            label="frozen_combined_resource",
        ).json()
        for candidate in case["candidates"]:
            pid, expected, trace = rows(self.evidence, case, candidate)
            expected = sorted(
                [SelectedEvidence.model_validate(x).model_dump(mode="json") for x in expected],
                key=lambda x: x["evidence_id"],
            )
            study, _, _ = self.pages(
                prefix + "/evidence",
                {"candidate_id": candidate["candidate_id"], "limit": 17},
                "scoped_evidence_continuation",
            )
            main, _, _ = self.pages(
                self.main
                + "/runs/"
                + quote(self.definition["review_provenance"]["source_run_id"], safe="")
                + "/pair-evidence",
                {"pair_id": pid, "limit": 17},
                "main_evidence_continuation",
            )
            require(main == study == expected, "main_scoped_predecessor_selected_parity")
            metadata = self.request(
                "GET",
                prefix + "/evidence-metadata",
                params={"candidate_id": candidate["candidate_id"]},
                label="scoped_recorded_bridges",
            ).json()
            main_pair = self.request(
                "GET",
                self.main + "/runs/" + quote(run, safe="") + "/pair",
                params={"pair_id": pid},
                label="main_pair_metadata",
            ).json()
            validate_pair_metadata(metadata, main_pair, case["source"], candidate, run, study)
            require(
                metadata["evidence_metadata"]
                == resource["pair_metadata"][candidate["candidate_id"]]["evidence_metadata"]
                == trace["evidence_metadata"],
                "main_scoped_frozen_predecessor_bridge_parity",
            )
            frozen = {
                x["evidence_id"]: x
                for x in resource["evidence"]
                if x["candidate_id"] == candidate["candidate_id"]
            }
            require(
                set(frozen) == {x["evidence_id"] for x in expected},
                "https_frozen_evidence_inventory",
            )
            require(
                all(frozen[x["evidence_id"]]["fact_ids"] == x["fact_ids"] for x in expected),
                "https_frozen_evidence_fact_links",
            )
            self.shared["pairs"].add(pid)
        require(
            environment.owned(self.config, "exploration", owned), "exploration_identity_preserved"
        )
        self.request(
            "GET",
            prefix + "/evidence-metadata",
            params={"candidate_id": "not-a-case-candidate"},
            label="foreign_candidate_metadata_denied",
            expected=403,
        )
        ids = set()
        for entity in [case["source"], *[c["entity"] for c in case["candidates"]]]:
            descriptions, _, _ = self.pages(
                prefix + "/explanations", {**entity, "limit": 100}, "scoped_description_inventory"
            )
            ids.update(x["explanation_id"] for x in descriptions)
        frozen_claims = {
            x["claim_id"]: x for x in resource["entity_profiles"] + resource["pair_comparison"]
        }
        for key in sorted(ids):
            main = self.request(
                "GET",
                self.main + "/explanations/" + quote(key, safe=""),
                label="main_bound_description",
            ).json()
            study = self.request(
                "GET",
                prefix + "/explanations/" + quote(key, safe=""),
                label="scoped_bound_description",
            ).json()
            require(
                main["claims"] == study["claims"] and main["presentation"] == study["presentation"],
                "main_scoped_description_parity",
            )
            for claim in main["claims"]:
                frozen = frozen_claims[claim["claim_id"]]
                require(
                    all(frozen[k] == claim[k] for k in ["text", "fact_ids", "category"]),
                    "frozen_clause_citation_parity",
                )
                require(
                    frozen["presentation"] == main["presentation"], "frozen_presentation_parity"
                )
            self.shared["descriptions"].add(key)
        self.shared["cases"].add(case["case_id"])

    def complete_read_surfaces(self, prefix, capabilities):
        # Earlier full-inventory scans are retained evidence. This run exercises
        # small-page continuation, directness and policy on the changed hierarchy.
        for version in capabilities["ontology_versions"]:
            audit = next(x for x in self.shared["taxonomy"] if x["ontology_version_id"] == version)
            for example in audit["useful_difference_examples"]:
                observed = {}
                for basis in ["literal_asserted", "structural_navigation", BASIS]:
                    params = {**example["entity"], "basis": basis, "limit": 100}
                    scoped, _, _ = self.pages(
                        prefix + "/hierarchy", params, "scoped_useful_basis_difference"
                    )
                    main, _, _ = self.pages(
                        self.main + "/hierarchy", params, "main_useful_basis_difference"
                    )
                    require(main == scoped, "basis_difference_main_study_parity")
                    observed[basis] = {x["parent"]["iri"] for x in main}
                require(
                    example["additional_parent"] in observed[BASIS] - observed["literal_asserted"],
                    "actual_useful_partial_difference",
                )
                self.shared["basis_examples"].append(
                    {**example, "parents": {k: sorted(v) for k, v in observed.items()}}
                )
            root = Path(self.config["package"]).parent / self.bundle["ontologies"][version]
            with sqlite3.connect(
                "file:" + str(root / "partial-navigation.sqlite") + "?mode=ro", uri=True
            ) as db:
                probes = {
                    direction: db.execute(
                        "SELECT "
                        + column
                        + " FROM edges GROUP BY "
                        + column
                        + " HAVING count(*) BETWEEN 2 AND 8 LIMIT 1"
                    ).fetchone()[0]
                    for direction, column in [("parents", "child"), ("children", "parent")]
                }
            for direction, iri in probes.items():
                for basis in ["literal_asserted", "structural_navigation", BASIS]:
                    params = {
                        "ontology_version_id": version,
                        "iri": iri,
                        "kind": "class",
                        "direction": direction,
                        "basis": basis,
                        "limit": 1,
                    }
                    study, n, first = self.pages(
                        prefix + "/hierarchy", params, "scoped_hierarchy_continuation"
                    )
                    main, m, _ = self.pages(
                        self.main + "/hierarchy", params, "main_hierarchy_continuation"
                    )
                    require(main == study, "real_hierarchy_ordered_page_union")
                    if basis == BASIS:
                        require(
                            n > 1 and first["status"] == "partial", "real_partial_multiple_pages"
                        )
                        with sqlite3.connect(
                            "file:" + str(root / "partial-navigation.sqlite") + "?mode=ro", uri=True
                        ) as db:
                            column = "child" if direction == "parents" else "parent"
                            original = [
                                json.loads(x[0])
                                for x in db.execute(
                                    "SELECT payload FROM edges WHERE " + column + "=? ORDER BY id",
                                    (iri,),
                                )
                            ]
                        require(main == original, "partial_direct_index_edges")
                        rejected = self.request(
                            "GET",
                            prefix + "/hierarchy",
                            params={
                                **params,
                                "basis": "reasoner_inferred",
                                "cursor": first["next_cursor"],
                            },
                            label="cross_basis_cursor_denied",
                            expected=409,
                        ).json()
                        require(rejected["code"] == "stale_cursor", "cross_basis_cursor_error_code")
                    self.shared["hierarchy"].append(
                        {
                            "ontology": version,
                            "entity": iri,
                            "basis": basis,
                            "direction": direction,
                            "pages": n,
                            "edges": len(main),
                            "ordered_sha256": canonical_hash(main),
                        }
                    )
            entity = {
                "ontology_version_id": version,
                "iri": probes["parents"],
                "kind": "class",
                "basis": BASIS,
                "limit": 1,
            }
            a, _, _ = self.pages(prefix + "/equivalents", entity, "scoped_partial_equivalents")
            b, _, _ = self.pages(
                self.main + "/ontologies/" + quote(version, safe="") + "/equivalents",
                entity,
                "main_partial_equivalents",
            )
            require(a == b, "partial_equivalence_parity")
            ep = self.request(
                "GET",
                prefix + "/entry-points",
                params={"ontology_version_id": version, "basis": BASIS, "limit": 1},
                label="scoped_partial_entry",
            ).json()
            mp = self.request(
                "GET",
                self.main + "/ontologies/" + quote(version, safe="") + "/entry-points",
                params={"basis": BASIS, "limit": 1},
                label="main_partial_entry",
            ).json()
            require(
                ep["items"] == mp["items"] and ep["inference"] == mp["inference"],
                "partial_entry_parity",
            )
            root_entity = ep["default_entity"]
            empty = self.request(
                "GET",
                prefix + "/hierarchy",
                params={**root_entity, "basis": BASIS},
                label="partial_empty_unknown",
            ).json()
            require(
                empty["status"] == "partial" and not empty["items"], "partial_empty_is_not_negative"
            )
            graph = self.request(
                "GET",
                prefix + "/hierarchy",
                params={**entity, "direction": "ancestors", "limit": 10},
                label="bounded_partial_ancestor_graph",
            ).json()
            require(
                graph["status"] == "partial" and len(graph["nodes"]) <= 10,
                "bounded_partial_ancestor_nodes",
            )
            foreign = next(
                s
                for s in self.definition["workspace_scopes"]
                if s["kind"] == "case" and s["scope_id"] != prefix.rsplit("/", 1)[1]
            )
            self.request(
                "GET",
                "/api/v1/study/workspace/" + foreign["scope_id"] + "/entry-points",
                params={"ontology_version_id": version, "basis": BASIS},
                label="foreign_partial_scope_denied",
                expected=403,
            )


def run(config, combined, evidence, prose, output):
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "exact-combined-review-verification/1",
        "verifier_sha256": environment.file_hash(__file__),
        "status": "running",
        "boundary": "Actual HTTPS/PostgreSQL API and frozen artifacts; not rendered browser acceptance",
        "static": static_audit(combined, evidence, prose),
    }
    environment.private_json(output / "static.json", result["static"])
    shared = {
        "pairs": set(),
        "cases": set(),
        "descriptions": set(),
        "hierarchy": [],
        "basis_examples": [],
        "taxonomy": result["static"]["taxonomy"],
    }
    try:
        for index in range(4):
            journey = CombinedJourney(
                config,
                receipt=output / f"journey-{index+1}.json",
                combined=combined,
                evidence=evidence,
                shared=shared,
            )
            try:
                journey.prepare()
                journey.tutorial()
                current = journey.request(
                    "GET", "/api/v1/study/cases/current", label="allocated_schedule_probe"
                ).json()
                possible = [
                    s
                    for s in journey.definition["schedules"]
                    if s["blocks"][0]["condition"] == current["condition"]
                    and current["case_id"] in s["blocks"][0]["case_ids"]
                ]
                require(len(possible) == 1, "known_server_allocated_schedule")
                explanation_cases = set(
                    next(
                        b["case_ids"]
                        for b in possible[0]["blocks"]
                        if b["condition"] == "explanation"
                    )
                )
                if explanation_cases <= shared["cases"]:
                    journey.report["status"] = "allocation_only"
                    journey.report["limitation"] = (
                        "Test session retained at first case; duplicate condition assignment was not repeated."
                    )
                else:
                    journey.cases()
                    journey.exports()
                    journey.report["status"] = "passed"
                with journey.study_log.open("rb") as stream:
                    stream.seek(journey.log_offset)
                    log = stream.read()
                signatures = {
                    "content_length_protocol_errors": b"Too little data for declared Content-Length",
                    "h11_protocol_errors": b"LocalProtocolError",
                    "asgi_exceptions": b"Exception in ASGI application",
                }
                observed = {key: log.count(value) for key, value in signatures.items()}
                journey.report["service_log_observation"] = {"bytes": len(log), "counts": observed}
                require(not any(observed.values()), "no_new_protocol_exceptions")
                journey.record()
            except Exception as error:
                journey.report.update(
                    status="failed", failure=str(error), failed_check=journey.checkpoint
                )
                journey.record()
                raise
            finally:
                journey.client.close()
            if len(shared["pairs"]) == 120:
                break
        require(
            len(shared["pairs"]) == 120 and len(shared["cases"]) == 24,
            "all_120_pairs_both_counterbalances",
        )
        require(len(shared["descriptions"]) == 260, "all_260_served_descriptions")
        result.update(
            status="passed",
            actual_pairs=120,
            actual_cases=24,
            actual_descriptions=260,
            hierarchy=shared["hierarchy"],
            basis_examples=shared["basis_examples"],
        )
    except Exception as error:
        result.update(status="failed", failure=str(error))
        raise
    finally:
        result["covered_pairs"] = len(shared["pairs"])
        result["covered_cases"] = sorted(shared["cases"])
        environment.private_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["environment", "combined", "evidence", "prose", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    run(environment.load(a.environment), a.combined, a.evidence, a.prose, a.output)
