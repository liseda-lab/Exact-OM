"""Read completed Exact 2.0 stages into an inspection index, without executing models.

Producer files are read-only. Receipts bind each stage to its predecessor and to
its actual saved output. Local query occurrences remain separate from alignment
membership. The public index contains only selected evidence and allowlisted data.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from .artifacts import atomic_json, relative_path
from .contracts import CONTRACT_VERSION, DomainError, canonical_hash, file_hash
from .decisions import (
    DEPENDENCY_INVENTORY_SCHEMA,
    _missing_event,
    _scores,
    dependency_inventory_hash,
    selected_evidence,
)
from .evidence_metadata import exact_match_basis, pair_evidence_metadata

FORMAT = "exact-staged/1"
DATABASE_SCHEMA = "exact-decision-index/2"
# Frozen predicate of exact-decisions-plan/1, verified without loading its inference runtime.
DECISION_QUESTION = {
    "equivalent": {
        "type": "noul",
        "instructions": "You are a binary classifier for ontology alignment. The source and target entities are equivalent.",
    }
}
STAGES = (
    "retrieval",
    "prefilter",
    "pair_scoring",
    "llm_signal",
    "selection",
    "threshold",
    "cardinality",
    "extraction",
    "relation_typing",
    "repair",
)
# Fields written by exact decisions apply. All other pair data must be unchanged.
MIX_FIELDS = frozenset(
    (
        "S_pair_final",
        "S_final",
        "p_llm",
        "llm_used",
        "llm_weight",
        "decision_request_id",
        "cached_response_reused",
        "local_model_call_attempted",
        "live_provider_call_attempted",
        "effective_pair_score_contribution",
        "selector_stage_consultation",
    )
)
SCORE_FIELDS = frozenset(
    (
        "S_final",
        "S_base",
        "S_pair_final",
        "S_struct",
        "U",
        "U_ind",
        "U_dis",
        "s_label",
        "s_hier",
        "s_attr",
        "s_sim",
        "s_diff",
        "s_strsim",
        "cand_sim",
        "repair_distinctive",
        "p_llm",
    )
)


def fail(message):
    raise DomainError("invalid_staged_input", message, 422)


def digest(value):
    if not isinstance(value, str):
        fail("Missing artifact checksum")
    value = value.removeprefix("sha256:")
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail("Invalid artifact checksum")
    return "sha256:" + value


def unique_fields(items):
    result = {}
    for key, value in items:
        if key in result:
            fail("Duplicate JSON field in staged input")
        result[key] = value
    return result


def read_json(path):
    # Pair evidence is streamed; JSON control/result documents have an explicit bound.
    if path.stat().st_size > 512 * 1024 * 1024:
        fail("Staged JSON metadata exceeds the 512 MiB preparation limit")

    return json.loads(
        path.read_bytes(),
        object_pairs_hook=unique_fields,
        parse_constant=lambda _: fail("Nonfinite staged JSON value"),
    )


def json_rows(path):
    with path.open("rb") as stream:
        while line := stream.readline(64 * 1024 * 1024 + 1):
            if len(line) > 64 * 1024 * 1024:
                fail("Staged pair record exceeds 64 MiB")
            if line.strip():
                row = json.loads(
                    line,
                    object_pairs_hook=unique_fields,
                    parse_constant=lambda _: fail("Nonfinite staged pair value"),
                )
                if not isinstance(row, dict):
                    fail("Expected a staged pair object")
                yield row


def check_hash(path, expected):
    if file_hash(path) != digest(expected):
        fail("Staged artifact checksum mismatch: " + path.name)


def discover_inputs(options, *, input_root):
    try:
        return _discover_inputs(options, input_root=input_root)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise DomainError(
            "invalid_staged_input", "Incomplete or malformed staged metadata", 422
        ) from exc


def _discover_inputs(options, *, input_root):
    """Locate and verify only the completed artifacts consumed by this adapter."""

    def locate(name):
        value = options.get(name)
        if value is None:
            return None
        path = Path(value).expanduser()
        return (path if path.is_absolute() else input_root / path).resolve()

    if set(options) - {
        "native_dir",
        "scored_pairs",
        "decision_plan",
        "acceptance_dir",
        "rankings",
        "queries",
        "model_manifest",
    }:
        fail("Unknown production stage input")
    native = locate("native_dir")
    if native is None:
        fail("A production import requires native_dir")
    paths = {
        "native_identity": native / "identity.json",
        "native_complete": native / "complete.json",
        "native_pairs": native / "pairs.jsonl",
        "sources": native / "sources.txt",
        "native_anchors": native / "anchors.json",
    }
    receipt, identity = read_json(paths["native_complete"]), read_json(paths["native_identity"])
    if (
        receipt.get("status") != "complete"
        or receipt.get("mode") not in {"global", "local"}
        or receipt["mode"] != identity.get("mode")
    ):
        fail("Native scoring is not a completed supported production run")
    check_hash(paths["native_identity"], receipt.get("identity_sha256"))
    outputs = receipt.get("outputs", {})
    for name in ("pairs.jsonl", "sources.txt", "anchors.json"):
        if name not in outputs:
            fail("Native completion receipt omits " + name)
    for name, checksum in outputs.items():
        path = relative_path(native, name)
        check_hash(path, checksum)
        paths["native_output/" + name] = path
    scored = locate("scored_pairs")
    plan = locate("decision_plan")
    if scored is not None and scored != paths["native_pairs"]:
        paths.update(scored_pairs=scored, decision_receipt=Path(str(scored) + ".receipt.json"))
        if plan is None:
            fail("--scored-pairs requires --decision-plan to verify the saved decision application")
        paths.update(decision_plan=plan, decision_requests=plan.parent / "requests.jsonl")
        application = read_json(paths["decision_receipt"])
        frozen = read_json(plan)
        if application.get("schema") != "exact-decisions-application/1":
            fail("Unsupported decision application receipt")
        if frozen["identity"].get("schema") != "exact-decisions-plan/1":
            fail("Unsupported decision plan")
        if frozen["identity"].get("anchors_sha256"):
            check_hash(paths["native_anchors"], frozen["identity"]["anchors_sha256"])
        if (
            application.get("fusion", {}).get("mode") != "beta_u"
            or number(application["fusion"]["beta"], "fusion beta") < 0
        ):
            fail("Unsupported decision score mixture")
        for path, expected in (
            (scored, application.get("output_sha256")),
            (paths["native_pairs"], application.get("candidates_sha256")),
            (plan, application.get("plan_sha256")),
            (paths["native_pairs"], frozen["identity"]["candidates_sha256"]),
            (paths["sources"], frozen["identity"]["sources_sha256"]),
            (paths["decision_requests"], frozen["requests_sha256"]),
        ):
            check_hash(path, expected)
    elif plan is not None:
        fail("A decision plan alone is not an applied result; pass --scored-pairs too")
    chosen = paths.get("scored_pairs", paths["native_pairs"])
    acceptance, ranking = locate("acceptance_dir"), locate("rankings")
    queries = locate("queries")
    model = locate("model_manifest")
    if acceptance and ranking:
        fail("Import one final output: acceptance or ranking, not both")
    if (acceptance or ranking) and model is None:
        fail("Final outputs require --model-manifest to verify their frozen policy identity")
    if model is not None:
        if not (acceptance or ranking):
            fail("--model-manifest requires --acceptance-dir or --rankings")
        paths["model_manifest"] = model
        model_info = read_json(model)
    if acceptance:
        if receipt["mode"] != "global":
            fail("Global acceptance must use a global native run")
        paths["acceptance_receipt"] = acceptance / "completion.json"
        result = read_json(paths["acceptance_receipt"])
        if result.get("status") != "complete":
            fail("Acceptance publication is incomplete")
        check_hash(chosen, result["policy"]["candidates_sha256"])
        policy = result["policy"]
        public_policy(policy)
        check_hash(model, policy["model_manifest_sha256"])
        if model_info.get("kind") == "edge_acceptance":
            setting = model_info.get("policies", {}).get(policy["name"])
            if setting is None or any(policy.get(key) != value for key, value in setting.items()):
                fail("Acceptance output differs from its frozen model policy")
        elif model_info.get("kind") == "retained_winner_acceptance":
            if (
                policy.get("name") != "retained_full"
                or policy.get("mode") != "winner"
                or policy.get("threshold") != model_info["head"]["accept_threshold"]
                or policy.get("pair_floor") != model_info["head"]["absolute_pair_floor"]
            ):
                fail("Retained acceptance output differs from its frozen policy")
        else:
            fail("Unsupported acceptance model manifest")
        for name in ("mappings.csv", "alignment.rdf"):
            check_hash(relative_path(acceptance, name), result["outputs"][name])
            paths["acceptance/" + name] = relative_path(acceptance, name)
    if ranking:
        if receipt["mode"] != "local" or queries is None:
            fail("Local ranking requires a local native run and --queries")
        paths.update(
            rankings=ranking,
            ranking_scores=ranking.with_suffix(".scores.json"),
            ranking_receipt=ranking.with_suffix(".receipt.json"),
            queries=queries,
        )
        result = read_json(paths["ranking_receipt"])
        if result.get("acceptance") is not False:
            fail("Ranking receipt must explicitly exclude alignment acceptance")
        check_hash(model, result["model_sha256"])
        if (
            model_info.get("kind") != "local_query_pairwise_ranker"
            or model_info.get("fit_identity") != result["fit_identity"]
            or model_info.get("acceptance") is not False
        ):
            fail("Ranking result differs from its frozen model identity")
        for path, expected in (
            (ranking, result["output_sha256"]),
            (paths["ranking_scores"], result["scores_sha256"]),
            (chosen, result["input_sha256"]),
            (queries, result["queries_sha256"]),
        ):
            check_hash(path, expected)
        if result.get("anchors_sha256"):
            check_hash(paths["native_anchors"], result["anchors_sha256"])
    elif queries:
        fail("--queries requires --rankings")
    return paths


def public_policy(policy):
    """Only operating settings and digests belong in the portable manifest."""
    if policy.get("name") not in {"full", "partial", "retained_full"} or policy.get("mode") not in {
        "greedy",
        "winner",
        "assignment",
    }:
        fail("Unsupported acceptance policy")
    result = {
        "name": policy["name"],
        "mode": policy["mode"],
        "threshold": number(policy["threshold"], "acceptance threshold"),
    }
    if not 0 <= result["threshold"] <= 1:
        fail("Invalid acceptance threshold")
    for key in ("model_manifest_sha256", "candidates_sha256"):
        result[key] = digest(policy[key])
    if "pair_floor" in policy:
        result["pair_floor"] = number(policy["pair_floor"], "pair floor")
        if not 0 <= result["pair_floor"] <= 1:
            fail("Invalid pair floor")
    if "complete_candidate_pairs" in policy:
        value = policy["complete_candidate_pairs"]
        if type(value) is not int or value < 0:
            fail("Invalid complete candidate count")
        result["complete_candidate_pairs"] = value
    return result


def bind_staged(options, *, input_root):
    if "native_complete" in options:
        validate_bound(options)
        return options
    paths = discover_inputs(options, input_root=input_root)
    return {
        name: {"path": str(path), "sha256": file_hash(path), "size": path.stat().st_size}
        for name, path in paths.items()
    }


def validate_bound(bindings):
    """Recheck the companion graph even for a manually supplied execution lock."""
    try:
        paths = {name: Path(item["path"]).resolve() for name, item in bindings.items()}
        for name, path in paths.items():
            check_hash(path, bindings[name]["sha256"])
            if path.stat().st_size != bindings[name]["size"]:
                fail("Staged artifact size mismatch")
        options = {"native_dir": str(paths["native_complete"].parent)}
        for role, option in (
            ("scored_pairs", "scored_pairs"),
            ("decision_plan", "decision_plan"),
            ("rankings", "rankings"),
            ("queries", "queries"),
            ("model_manifest", "model_manifest"),
        ):
            if role in paths:
                options[option] = str(paths[role])
        if "acceptance_receipt" in paths:
            options["acceptance_dir"] = str(paths["acceptance_receipt"].parent)
        expected = discover_inputs(options, input_root=Path.cwd())
        if expected != paths:
            fail("Staged lock does not pin the exact companion artifact inventory")
        return paths
    except (KeyError, TypeError, AttributeError) as exc:
        raise DomainError("invalid_staged_input", "Incomplete staged execution lock", 422) from exc


def number(value, name):
    if type(value) not in {float, int} or not math.isfinite(value):
        fail("Invalid finite number: " + name)
    return value


def pair(row):
    values = (
        row.get("Src"),
        row.get("SrcKind", "class"),
        row.get("Tgt"),
        row.get("TgtKind", "class"),
    )
    if (
        any(not isinstance(x, str) or not x for x in values)
        or values[1] not in {"class", "object_property", "data_property", "individual"}
        or values[3] not in {"class", "object_property", "data_property", "individual"}
    ):
        fail("Invalid typed production pair")
    return values


def saved_request_id(payload, identity):
    if (
        not isinstance(payload, dict)
        or set(payload) != {"state", "questions"}
        or payload.get("questions") != DECISION_QUESTION
        or not isinstance(payload.get("state"), dict)
        or set(payload["state"]) != {"evidence"}
        or not isinstance(payload["state"]["evidence"], str)
    ):
        fail("Decision request uses an incompatible predicate")
    # The production decision protocol uses ASCII-escaped canonical JSON.
    value = {"model": identity["model"], "revision": identity["revision"], "payload": payload}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def retrieval_channels(row):
    value = row.get("cand_channels")
    if value in (None, ""):
        return []
    if not isinstance(value, str):
        fail("Retrieval channels must be saved channel names")
    names = value.split("|")
    if any(name not in {"semantic", "lexical", "cross_encoder"} for name in names):
        fail("Unsupported native retrieval channel")
    return names


def queries_from(path):
    if path.suffix.lower() == ".json":
        rows = read_json(path)
    else:
        with path.open(newline="") as stream:
            rows = [
                dict(qid=i, source=r["SrcEntity"], candidates=ast.literal_eval(r["TgtCandidates"]))
                for i, r in enumerate(csv.DictReader(stream, delimiter="\t"))
            ]
    if not isinstance(rows, list):
        fail("Queries must be a list")
    seen = set()
    for i, row in enumerate(rows):
        qid = row.get("qid")
        if (
            type(qid) is not int
            or qid < 0
            or qid in seen
            or not isinstance(row.get("source"), str)
            or not row["source"]
            or not isinstance(row.get("candidates"), list)
            or any(not isinstance(t, str) or not t for t in row["candidates"])
        ):
            fail("Invalid original query identity or candidates")
        if len(row["candidates"]) != len(set(row["candidates"])):
            fail("Production ranking queries require unique target identities")
        seen.add(qid)
        # Application references/gold columns are deliberately never copied.
        yield {"qid": qid, "source": row["source"], "candidates": row["candidates"]}


def event(stage, reason, *, outcome="retained", status="completed", values=None, refs=()):
    return dict(
        stage=stage,
        status=status,
        outcome=outcome,
        reason_code=reason,
        scores=_scores(values or {}, stage),
        competitor_pair_ids=[],
        artifact_refs=list(refs),
        implementation_id="exact-staged-import/1",
        provenance_status="derived_from_saved_artifacts",
    )


def import_staged(bindings, output_dir, **kwargs):
    try:
        return _import_staged(bindings, output_dir, **kwargs)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise DomainError(
            "invalid_staged_input", "Incomplete or malformed staged records", 422
        ) from exc


def _import_staged(bindings, output_dir, *, source, target, run_id=None, evidence_resolver=None):
    """Build a bounded-query SQLite index using verified immutable stage inputs."""
    paths = validate_bound(bindings)
    dependencies = {name: item["sha256"] for name, item in bindings.items()}
    for name, path in paths.items():
        check_hash(path, dependencies[name])
    identity, native = read_json(paths["native_identity"]), read_json(paths["native_complete"])
    for side, context in (("source", source), ("target", target)):
        if digest(identity[side]["sha256"]) != context.manifest["identity"]["root_sha256"]:
            fail("Production run uses a different " + side + " ontology")
    run_id = run_id or "staged-" + canonical_hash(dependencies)[7:23]
    inventory = dependency_inventory_hash(dependencies)
    source_id, target_id = source.ontology_version_id, target.ontology_version_id
    ranking = "rankings" in paths
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "decisions.sqlite"
    if database.exists():
        fail("Staged decision index already exists")
    source_universe = paths["sources"].read_text().splitlines()
    if len(source_universe) != len(set(source_universe)) or any(not s for s in source_universe):
        fail("Invalid frozen source universe")
    source_set = set(source_universe)
    anchor_rows = read_json(paths["native_anchors"])
    anchors = {pair(a): a for a in anchor_rows}
    if any(
        number(row.get("S_final"), "anchor score") != 1 or key[0] not in source_set
        for key, row in anchors.items()
    ):
        fail("Invalid native hard anchor")
    if len(anchors) != len(anchor_rows):
        fail("Duplicate native anchor")
    requests = {}
    plan = read_json(paths["decision_plan"]) if "decision_plan" in paths else None
    if plan:
        for request in json_rows(paths["decision_requests"]):
            key = tuple(request["pair"])
            if (
                len(key) != 2
                or any(not isinstance(v, str) or not v for v in key)
                or key in requests
            ):
                fail("Invalid or duplicate applied decision request")
            if saved_request_id(request["payload"], plan["identity"]) != request["id"]:
                fail("Decision request payload differs from its recorded identity")
            requests[key] = request["id"]
        if len(requests) != plan["request_count"] or len(source_universe) != plan["source_count"]:
            fail("Decision plan population mismatch")
    application = read_json(paths["decision_receipt"]) if "decision_receipt" in paths else None
    acceptance = read_json(paths["acceptance_receipt"]) if "acceptance_receipt" in paths else None
    accepted = {}
    if acceptance:
        with paths["acceptance/mappings.csv"].open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["SrcEntity"], "class", row["TgtEntity"], "class")
                if key in accepted or row.get("Relation") != "=":
                    fail("Invalid acceptance mapping")
                accepted[key] = number(float(row["Score"]), "acceptance score")
        if (
            len(accepted) != acceptance["mappings"]
            or len({k[0] for k in accepted}) != len(accepted)
            or len({k[2] for k in accepted}) != len(accepted)
        ):
            fail("Acceptance membership/count/cardinality mismatch")
        import xml.etree.ElementTree as ET

        rdf, ns = (
            "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}",
            "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment#}",
        )
        restored = {}
        for _, cell in ET.iterparse(paths["acceptance/alignment.rdf"]):
            if cell.tag != ns + "Cell":
                continue
            key = (
                cell.find(ns + "entity1").get(rdf + "resource"),
                "class",
                cell.find(ns + "entity2").get(rdf + "resource"),
                "class",
            )
            value = number(float(cell.find(ns + "measure").text), "RDF acceptance score")
            if key in restored or cell.find(ns + "relation").text != "=":
                fail("Invalid RDF mapping")
            restored[key] = value
            cell.clear()
        if restored != accepted:
            fail("Acceptance CSV and RDF disagree")
    counts = dict(pairs=0, selected_evidence=0, sources=0, queries=0)
    db = sqlite3.connect(database)
    try:
        db.executescript(
            """CREATE TABLE pairs(id TEXT PRIMARY KEY,source TEXT NOT NULL,source_kind TEXT NOT NULL,target TEXT NOT NULL,target_kind TEXT NOT NULL,candidate TEXT NOT NULL,trace TEXT NOT NULL,evidence TEXT NOT NULL,query_id INTEGER);
CREATE INDEX pair_source ON pairs(source,source_kind,query_id,id);
CREATE TABLE sources(id TEXT PRIMARY KEY,iri TEXT,kind TEXT,payload TEXT,query_id INTEGER);
CREATE INDEX source_order ON sources(id);
CREATE UNIQUE INDEX source_query ON sources(query_id);
CREATE TABLE input_pairs(id TEXT PRIMARY KEY,checksum TEXT,scores TEXT,seen INTEGER DEFAULT 0);
CREATE TABLE occurrences(pair TEXT,query_id INTEGER,position INTEGER,payload TEXT,PRIMARY KEY(query_id,position));
CREATE INDEX occurrence_pair ON occurrences(pair);
"""
        )

        def add_source(iri, kind="class", qid=None, count=None, query_position=None):
            entity = dict(ontology_version_id=source_id, iri=iri, kind=kind)
            key = canonical_hash(entity if qid is None else {"entity": entity, "query_id": qid})
            payload = dict(entity_id=key, entity=entity)
            if qid is not None:
                payload.update(query_id=qid, candidate_count=count, query_position=query_position)
            db.execute(
                "INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?)",
                (key, iri, kind, json.dumps(payload), qid),
            )

        if ranking:
            output = read_json(paths["ranking_scores"])
            receipt = read_json(paths["ranking_receipt"])
            original = list(queries_from(paths["queries"]))
            if (
                output.get("kind") != "local_query_rankings"
                or output.get("schema_version") != 1
                or output.get("complete") is not True
                or output.get("acceptance") is not False
                or output.get("fit_identity") != receipt["fit_identity"]
                or len(output.get("queries", [])) != len(original)
                or len(original) != receipt["query_count"]
            ):
                fail("Invalid completed ranking result")
            with paths["rankings"].open(newline="") as stream:
                published = list(csv.DictReader(stream, delimiter="\t"))
            if len(published) != len(original):
                fail("Ranking TSV query count mismatch")
            protected_pairs = set()
            for query_position, (query, result, saved) in enumerate(
                zip(original, output["queries"], published, strict=True)
            ):
                qid, iri = query["qid"], query["source"]
                targets = [c["target"] for c in result["candidates"]]
                if (
                    iri not in source_set
                    or result["qid"] != qid
                    or result["source"] != iri
                    or Counter(targets) != Counter(query["candidates"])
                    or saved["SrcEntity"] != iri
                    or ast.literal_eval(saved["TgtCandidates"]) != targets
                ):
                    fail("Ranking changed query identity, membership or saved order")
                if result.get("fixed_anchor_count") != sum(
                    bool(c.get("protected_exact")) for c in result["candidates"]
                ):
                    fail("Ranking anchor count mismatch")
                add_source(iri, qid=qid, count=len(targets), query_position=query_position)
                for position, candidate in enumerate(result["candidates"], 1):
                    number(candidate["score"], "ranking score")
                    if candidate.get("utility") is not None:
                        number(candidate["utility"], "ranking utility")
                    key = (iri, "class", candidate["target"], "class")
                    if bool(candidate.get("protected_exact")) != bool(
                        receipt.get("anchors_sha256") and key in anchors
                    ):
                        fail("Ranking hard anchor differs from the bound native anchor sidecar")
                    if candidate.get("protected_exact"):
                        if candidate["score"] != 1 or candidate.get("utility") is not None:
                            fail("Ranking anchor must retain its saved score and absent utility")
                        protected_pairs.add(key)
                    payload = dict(
                        query_position=query_position,
                        score=candidate["score"],
                        utility=candidate.get("utility"),
                        protected_exact=bool(candidate.get("protected_exact")),
                        count=len(targets),
                    )
                    db.execute(
                        "INSERT INTO occurrences VALUES (?,?,?,?)",
                        (canonical_hash(key), qid, position, json.dumps(payload)),
                    )
            count = db.execute("SELECT count(*) FROM occurrences").fetchone()[0]
            if (
                count != receipt["candidate_occurrences"]
                or count != output["candidate_occurrences"]
                or len(original) != output["original_query_count"]
                or len({q["source"] for q in original}) != output["source_count"]
                or len(protected_pairs) != output["unchanged_anchor_pairs"]
            ):
                fail("Ranking occurrence count mismatch")
            counts["queries"] = len(original)
        else:
            for iri in source_universe:
                add_source(iri)
        total_native = 0
        for row in json_rows(paths["native_pairs"]):
            key = pair(row)
            if key[0] not in source_set:
                fail("Native pair outside its frozen source universe")
            number(row["S_final"], "native final score")
            check = canonical_hash({k: v for k, v in row.items() if k not in MIX_FIELDS})
            try:
                db.execute(
                    "INSERT INTO input_pairs(id,checksum,scores) VALUES (?,?,?)",
                    (
                        canonical_hash(key),
                        check,
                        json.dumps(
                            {
                                k: number(v, k)
                                for k, v in row.items()
                                if k in SCORE_FIELDS and v is not None
                            }
                        ),
                    ),
                )
            except sqlite3.IntegrityError:
                fail("Duplicate native typed pair")
            total_native += 1
        if (
            total_native != native["candidate_count"]
            or len(source_universe) != native["source_count"]
        ):
            fail("Native receipt population mismatch")
        if (
            ranking
            and db.execute(
                "SELECT count(*) FROM input_pairs WHERE id NOT IN (SELECT pair FROM occurrences)"
            ).fetchone()[0]
        ):
            fail("Native candidates fall outside the original ranking support")
        applied = changed = selected_count = total = 0
        for row in json_rows(paths.get("scored_pairs", paths["native_pairs"])):
            key = pair(row)
            pk = canonical_hash(key)
            original = db.execute(
                "SELECT checksum,seen,scores FROM input_pairs WHERE id=?", (pk,)
            ).fetchone()
            if (
                original is None
                or original[1]
                or original[0]
                != canonical_hash({k: v for k, v in row.items() if k not in MIX_FIELDS})
            ):
                fail("Decision application changed native membership or evidence")
            db.execute("UPDATE input_pairs SET seen=1 WHERE id=?", (pk,))
            total += 1
            if application:
                request_id = requests.get((key[0], key[2]))
                if (
                    row.get("llm_used") is not (request_id is not None)
                    or row.get("decision_request_id") != request_id
                ):
                    fail("Decision application differs from its frozen requests")
                if request_id:
                    if "decision_request" in row:
                        payload = row["decision_request"]["payload"]
                    else:
                        field = (
                            "llm_structured_evidence_packet"
                            if plan["identity"]["policy"].startswith("global")
                            else "llm_evidence_packet"
                        )
                        labels = row["selected_labels"]
                        text = f"Source entity: {labels['source']}\nTarget entity: {labels['target']}\n\nPair brief:\n{row[field]}"
                        payload = {
                            "state": {"evidence": text},
                            "questions": DECISION_QUESTION,
                        }
                    if saved_request_id(payload, plan["identity"]) != request_id:
                        fail("Decision request evidence differs from the saved native pair")
                base = number(row["S_base"], "base score")
                weight = (
                    min(
                        1.0,
                        max(0.0, application["fusion"]["beta"] * number(row["U"], "uncertainty")),
                    )
                    if request_id
                    else 0.0
                )
                probability = row.get("p_llm")
                if request_id and not 0 <= number(probability, "decision probability") <= 1:
                    fail("Invalid saved decision probability")
                if not request_id and probability is not None:
                    fail("Unrequested pair contains a decision probability")
                expected = (1.0 - weight) * base + weight * probability if request_id else base
                if (
                    row["S_final"] != expected
                    or row["S_pair_final"] != expected
                    or row["llm_weight"] != weight
                ):
                    fail("Saved decision scores disagree with the recorded mixture")
                changed += expected != base
                applied += int(request_id is not None)
            if key in accepted:
                selected_count += 1
            occurrences = (
                db.execute(
                    "SELECT query_id,position,payload FROM occurrences WHERE pair=? ORDER BY query_id,position",
                    (pk,),
                ).fetchall()
                if ranking
                else [(None, None, None)]
            )
            for qid, position, raw_rank in occurrences:
                rank = json.loads(raw_rank) if raw_rank else None
                s = dict(ontology_version_id=source_id, iri=key[0], kind=key[1])
                t = dict(ontology_version_id=target_id, iri=key[2], kind=key[3])
                pid = canonical_hash(
                    dict(run_id=run_id, source=s, target=t, query_id=qid, occurrence=position)
                )
                record = {
                    **(row.get("selected_attributions") or {}),
                    "selected_labels": row.get("selected_labels", {}),
                }
                evidence = selected_evidence(record, s, t, resolver=evidence_resolver)
                from .models import SelectedEvidence

                for item in evidence:
                    if item["channel"] not in {
                        "hierarchy",
                        "similarity",
                        "difference",
                        "attributes",
                    }:
                        fail("Unsupported selected evidence channel")
                    for name, value in item["values"].items():
                        number(value, "selected evidence " + name)
                    SelectedEvidence.model_validate(item)
                scores = {
                    k: number(v, k) for k, v in row.items() if k in SCORE_FIELDS and v is not None
                }
                if row.get("llm_weight") is not None:
                    scores["w_llm"] = number(row["llm_weight"], "model mixture weight")
                if key in accepted:
                    scores["P_match"] = accepted[key]
                if rank:
                    scores["S_rank"] = rank["score"]
                    if rank["utility"] is not None:
                        scores["rank_utility"] = rank["utility"]
                refs = [inventory]
                events = [event("retrieval", "staged_candidate_recorded", refs=refs)]
                protected = bool(rank and rank["protected_exact"]) or bool(
                    not acceptance
                    and not ranking
                    and native["mode"] == "local"
                    and row.get("exact_anchor")
                    and key in anchors
                )
                if protected:
                    e = event("prefilter", "exact_match_protected", outcome="selected", refs=refs)
                    e["exact_basis"] = exact_match_basis(e, source=s, target=t)
                    events.append(e)
                # Native local profiles may score an anchor before the ranker protects it.
                # Only appended anchor rows lack the scorer's selected-attribution record.
                skipped_scoring = protected and "selected_attributions" not in row
                events.append(
                    event(
                        "pair_scoring",
                        (
                            "exact_match_protected"
                            if skipped_scoring
                            else "native_pair_scores_recorded"
                        ),
                        status="not_run" if skipped_scoring else "completed",
                        outcome="not_applicable" if skipped_scoring else "retained",
                        values={k: v for k, v in json.loads(original[2]).items() if k != "p_llm"},
                        refs=refs,
                    )
                )
                if application:
                    events.append(
                        event(
                            "llm_signal",
                            (
                                "staged_decision_applied"
                                if row["llm_used"]
                                else "staged_decision_not_requested"
                            ),
                            status="completed" if row["llm_used"] else "not_run",
                            outcome="retained" if row["llm_used"] else "not_applicable",
                            values={
                                k: v
                                for k, v in scores.items()
                                if k in {"p_llm", "w_llm", "S_pair_final"}
                            },
                            refs=refs,
                        )
                    )
                ranked = (
                    dict(
                        mode="local_ranking",
                        query_id=qid,
                        query_position=rank["query_position"],
                        candidate_count=rank["count"],
                        rank=position,
                        saved_output_member=True,
                        artifact_ref=dependencies["rankings"],
                    )
                    if rank
                    else None
                )
                if rank:
                    events.append(
                        event(
                            "selection",
                            "local_ranking_applied",
                            values={"S_rank": rank["score"]},
                            refs=refs,
                        )
                    )
                elif acceptance:
                    events.append(
                        event(
                            "selection",
                            (
                                "staged_acceptance_selected"
                                if key in accepted
                                else "staged_acceptance_not_selected"
                            ),
                            outcome="selected" if key in accepted else "not_selected",
                            values={"P_match": accepted[key]} if key in accepted else {},
                            refs=refs,
                        )
                    )

                recorded = {e["stage"] for e in events}
                events.extend(_missing_event(stage) for stage in STAGES if stage not in recorded)
                events.sort(key=lambda e: STAGES.index(e["stage"]))
                membership = key in accepted if acceptance else None
                trace = dict(
                    artifact_type="pair_decision_trace",
                    contract_version=CONTRACT_VERSION,
                    run_id=run_id,
                    pair_id=pid,
                    source=s,
                    target=t,
                    events=events,
                    saved_alignment_member=membership,
                    relation="equivalent" if membership else None,
                    relation_basis="default_convention" if membership else "unavailable",
                    assumption_refs=[],
                    fixture_provenance="derived_from_saved_production_stages",
                    ranking=ranked,
                )
                trace["evidence_metadata"] = pair_evidence_metadata(
                    record, evidence, s, t, [inventory], events
                )
                score_list = _scores(scores, "saved_artifact")
                for item in score_list:
                    item["stage"] = (
                        "selection"
                        if item["name"] in {"P_match", "S_rank", "rank_utility"}
                        else (
                            "llm_signal"
                            if application
                            and item["name"] in {"p_llm", "w_llm", "S_final", "S_pair_final"}
                            else "pair_scoring"
                        )
                    )
                candidate = dict(
                    pair_id=pid,
                    output_mode=(
                        "local_ranking"
                        if ranking
                        else "global_acceptance" if acceptance else "native_scores"
                    ),
                    source=s,
                    target=t,
                    scores=score_list,
                    retrieval_channels=retrieval_channels(row),
                    ordinal_ranks=dict(
                        candidate_joint_rank=position,
                        nil_rank=None,
                        joint_ordering="saved_query_output" if rank else None,
                        joint_tie_rule=None,
                        retrieval_rank=row.get("cand_rank"),
                        retrieval_ordering=None,
                        retrieval_tie_rule=None,
                        retrieval_provenance=(
                            "recorded" if row.get("cand_rank") is not None else None
                        ),
                    ),
                    evidence_counts=[],
                    nil={},
                    saved_alignment_member=membership,
                    membership_provenance=dict(
                        status="derived_from_saved_artifacts" if acceptance else "unavailable",
                        artifact_hash=dependencies.get("acceptance/mappings.csv"),
                    ),
                    status="available",
                    decision_status="available",
                    ranking=ranked,
                )
                from .models import Candidate, PairResponse

                Candidate.model_validate(candidate)
                PairResponse.model_validate(trace)
                db.execute(
                    "INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        pid,
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        json.dumps(candidate),
                        json.dumps(trace),
                        json.dumps(evidence),
                        qid,
                    ),
                )
                if not ranking:
                    add_source(key[0], key[1])
                counts["pairs"] += 1
                counts["selected_evidence"] += len(evidence)
        if total != total_native or selected_count != len(accepted):
            fail("Final output contains candidates absent from native scoring")
        if application and (
            total != application["rows"]
            or applied != application["judgments_applied"]
            or applied != len(requests)
            or changed != application["changed_scores"]
        ):
            fail("Decision receipt population mismatch")
        if (
            ranking
            and counts["pairs"] != db.execute("SELECT count(*) FROM occurrences").fetchone()[0]
        ):
            fail("A ranked occurrence has no native pair evidence")
        expected_population = (
            acceptance["policy"].get("complete_candidate_pairs", acceptance["candidate_pairs"])
            if acceptance
            else None
        )
        if acceptance and expected_population != total:
            fail("Acceptance candidate population mismatch")
        counts["sources"] = db.execute("SELECT count(*) FROM sources").fetchone()[0]
        db.executescript("DROP TABLE input_pairs; DROP TABLE occurrences;")
        db.commit()
    finally:
        db.close()
    for name, path in paths.items():
        check_hash(path, dependencies[name])
    result = dict(
        contract_version=CONTRACT_VERSION,
        artifact_type="run_decision_package",
        implementation="staged-adapter/1",
        database_schema=DATABASE_SCHEMA,
        query_scoped=ranking,
        mode="local_ranking" if ranking else "global_acceptance" if acceptance else "native_scores",
        run_id=run_id,
        revision=canonical_hash(
            dict(adapter=FORMAT, dependencies=dependencies, source=source_id, target=target_id)
        ),
        source_ontology_version_id=source_id,
        target_ontology_version_id=target_id,
        context_revision=canonical_hash(
            [source.manifest["artifacts"], target.manifest["artifacts"]]
        ),
        ontology_input_binding=dict(
            status="verified",
            source_root_sha256=digest(identity["source"]["sha256"]),
            target_root_sha256=digest(identity["target"]["sha256"]),
        ),
        input_binding=dict(status="verified", scope="all_consumed_production_stages"),
        dependencies=dependencies,
        dependency_inventory_schema=DEPENDENCY_INVENTORY_SCHEMA,
        dependency_inventory_hash=inventory,
        counts=counts,
        database="decisions.sqlite",
        database_hash=file_hash(database),
        path=".",
        status="available",
        acceptance_policy=public_policy(acceptance["policy"]) if acceptance else None,
    )
    atomic_json(directory / "decisions.manifest.json", result)
    return result
