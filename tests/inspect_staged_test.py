"""Contract and adversarial checks for importing saved production stages.

These small documents exercise reader boundaries. The separate pipeline test
creates its inputs through the actual production CLI and offline encoders.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="Viewer needs Python 3.11")

from exact_inspect.contracts import DomainError, file_hash
from exact_inspect.decisions import DecisionStore
from exact_inspect.models import Candidate, PairResponse
from exact_inspect.staged import bind_staged, import_staged, validate_bound


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def read(path):
    return json.loads(path.read_text())


def native_fixture(root, *, mode="local", empty=False):
    native = root / "native"
    native.mkdir(parents=True)
    contexts = []
    for side in ("s", "t"):
        ontology = root / (side + ".ofn")
        ontology.write_text("Ontology(Declaration(Class(<urn:" + side + ":heart>)))")
        contexts.append(
            SimpleNamespace(
                ontology_version_id="sha256:" + side * 64,
                manifest={"identity": {"root_sha256": file_hash(ontology)}, "artifacts": {}},
            )
        )
    rows = (
        []
        if empty
        else [
            dict(
                Src="urn:s:heart",
                Tgt="urn:t:" + target,
                S_final=score,
                S_pair_final=score,
                S_base=score,
                U=0.2,
                exact_anchor=target == "heart" and mode == "local",
                selected_attributions={},
                gold_target="PRIVATE_GOLD",
                local_path="PRIVATE_PATH",
            )
            for target, score in (("heart", 1.0), ("lung", 0.3))
        ]
    )
    (native / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (native / "sources.txt").write_text("urn:s:heart\nurn:s:empty\n")
    write(
        native / "anchors.json",
        [] if empty else [dict(Src="urn:s:heart", Tgt="urn:t:heart", S_final=1.0)],
    )
    write(
        native / "identity.json",
        dict(
            mode=mode,
            source={"sha256": file_hash(root / "s.ofn")},
            target={"sha256": file_hash(root / "t.ofn")},
        ),
    )
    write(
        native / "complete.json",
        dict(
            status="complete",
            mode=mode,
            source_count=2,
            candidate_count=len(rows),
            identity_sha256=file_hash(native / "identity.json"),
            outputs={
                name: file_hash(native / name)
                for name in ("pairs.jsonl", "sources.txt", "anchors.json")
            },
        ),
    )
    return {"native_dir": str(native)}, contexts


def ranking_fixture(root):
    options, contexts = native_fixture(root)
    queries = [
        dict(
            qid=9,
            source="urn:s:heart",
            candidates=["urn:t:lung", "urn:t:heart"],
            gold_target="PRIVATE_GOLD",
        ),
        dict(qid=4, source="urn:s:heart", candidates=["urn:t:heart"]),
        dict(qid=22, source="urn:s:empty", candidates=[]),
    ]
    write(root / "queries.json", queries)
    model = write(
        root / "model.json",
        dict(
            kind="local_query_pairwise_ranker",
            fit_identity="fit1",
            acceptance=False,
            training_sources=["PRIVATE_TRAIN"],
        ),
    )
    ranked = root / "ranked.tsv"
    ranked.write_text(
        "SrcEntity\tTgtCandidates\nurn:s:heart\t['urn:t:heart', 'urn:t:lung']\nurn:s:heart\t['urn:t:heart']\nurn:s:empty\t[]\n"
    )
    results = [
        dict(
            qid=q["qid"],
            source=q["source"],
            fixed_anchor_count=int(bool(q["candidates"])),
            candidates=[
                dict(
                    target=t,
                    score=1.0 if t.endswith("heart") else 0.4,
                    utility=None if t.endswith("heart") else 0.12,
                    protected_exact=t.endswith("heart"),
                )
                for t in sorted(q["candidates"])
            ],
        )
        for q in queries
    ]
    write(
        ranked.with_suffix(".scores.json"),
        dict(
            schema_version=1,
            kind="local_query_rankings",
            fit_identity="fit1",
            queries=results,
            unchanged_anchor_pairs=1,
            candidate_occurrences=3,
            original_query_count=3,
            source_count=2,
            complete=True,
            acceptance=False,
        ),
    )
    write(
        ranked.with_suffix(".receipt.json"),
        dict(
            fit_identity="fit1",
            candidate_occurrences=3,
            query_count=3,
            acceptance=False,
            output_sha256=file_hash(ranked),
            scores_sha256=file_hash(ranked.with_suffix(".scores.json")),
            model_sha256=file_hash(model),
            queries_sha256=file_hash(root / "queries.json"),
            input_sha256=file_hash(Path(options["native_dir"]) / "pairs.jsonl"),
            anchors_sha256=file_hash(Path(options["native_dir"]) / "anchors.json"),
        ),
    )
    options.update(
        rankings=str(ranked), queries=str(root / "queries.json"), model_manifest=str(model)
    )
    return options, contexts


def build(options, contexts, destination):
    bindings = bind_staged(options, input_root=destination.parent)
    import_staged(bindings, destination, source=contexts[0], target=contexts[1])
    return DecisionStore(destination)


def test_repeated_query_empty_query_scores_and_anchor_identity(tmp_path):
    options, contexts = ranking_fixture(tmp_path)
    store = build(options, contexts, tmp_path / "view")
    sources = {s["query_id"]: s for s in store.sources(limit=100)["items"]}
    assert set(sources) == {9, 4, 22}
    with pytest.raises(DomainError, match="No saved query"):
        store.candidates("urn:s:heart", query_id=22)
    with pytest.raises(DomainError, match="No saved query"):
        store.candidates("urn:s:heart", query_id=777)
    assert sources[22]["candidate_count"] == 0 and sources[4]["query_position"] == 1
    assert store.candidates("urn:s:empty", query_id=22)["items"] == []
    with pytest.raises(DomainError, match="Choose an original"):
        store.candidates("urn:s:heart")
    first = store.candidates("urn:s:heart", query_id=9)["items"]
    second = store.candidates("urn:s:heart", query_id=4)["items"]
    assert len(first) == 2 and len(second) == 1
    assert second[0]["pair_id"] not in {x["pair_id"] for x in first}
    for item in first + second:
        Candidate.model_validate(item)
        trace = store.pair(item["pair_id"])
        PairResponse.model_validate(trace)
        assert item["saved_alignment_member"] is None and trace["relation"] is None
        assert item["ranking"]["rank"] in {1, 2}
        assert item["ranking"]["query_id"] in {9, 4}
        assert next(s for s in item["scores"] if s["name"] == "S_rank")["stage"] == "selection"
    content = (tmp_path / "view" / "decisions.sqlite").read_bytes()
    assert all(
        value not in content for value in (b"PRIVATE_GOLD", b"PRIVATE_PATH", b"PRIVATE_TRAIN")
    )
    page = store.candidates("urn:s:heart", query_id=9, limit=1)
    with pytest.raises(DomainError):
        store.candidates("urn:s:heart", query_id=4, cursor=page["next_cursor"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_count", 1),
        ("original_query_count", 5),
        ("unchanged_anchor_pairs", 0),
        ("candidate_occurrences", 2),
        ("complete", False),
        ("acceptance", True),
    ],
)
def test_rehashed_ranking_semantic_mismatches_rejected(tmp_path, field, value):
    options, contexts = ranking_fixture(tmp_path)
    ranked = Path(options["rankings"])
    scores = read(ranked.with_suffix(".scores.json"))
    scores[field] = value
    write(ranked.with_suffix(".scores.json"), scores)
    receipt = read(ranked.with_suffix(".receipt.json"))
    receipt["scores_sha256"] = file_hash(ranked.with_suffix(".scores.json"))
    write(ranked.with_suffix(".receipt.json"), receipt)
    with pytest.raises(DomainError):
        build(options, contexts, tmp_path / "view")


@pytest.mark.parametrize(
    "role",
    ["native_identity", "native_pairs", "sources", "model_manifest", "ranking_scores", "queries"],
)
def test_pinned_artifacts_cannot_be_changed_or_rebound(tmp_path, role):
    options, _ = ranking_fixture(tmp_path)
    bound = bind_staged(options, input_root=tmp_path)
    Path(bound[role]["path"]).write_text("{}")
    with pytest.raises(DomainError):
        validate_bound(bound)
    with pytest.raises(DomainError):
        bind_staged(bound, input_root=tmp_path)


def test_handcrafted_lock_cannot_bypass_receipts(tmp_path):
    options, contexts = ranking_fixture(tmp_path)
    bound = bind_staged(options, input_root=tmp_path)
    broken = read(Path(bound["ranking_receipt"]["path"]))
    broken["input_sha256"] = "0" * 64
    path = write(Path(bound["ranking_receipt"]["path"]), broken)
    bound["ranking_receipt"].update(sha256=file_hash(path), size=path.stat().st_size)
    with pytest.raises(DomainError, match="checksum mismatch"):
        import_staged(bound, tmp_path / "view", source=contexts[0], target=contexts[1])


def test_native_only_has_unknown_membership_and_preserves_empty_sources(tmp_path):
    options, contexts = native_fixture(tmp_path, mode="global")
    store = build(options, contexts, tmp_path / "view")
    assert len(store.sources()["items"]) == 2
    for item in store.candidates("urn:s:heart")["items"]:
        assert item["saved_alignment_member"] is None and item["ranking"] is None
        assert store.pair(item["pair_id"])["relation"] is None
    empty_options, empty_contexts = native_fixture(tmp_path / "zero", mode="global", empty=True)
    empty = build(empty_options, empty_contexts, tmp_path / "empty-view")
    assert empty.manifest()["counts"]["pairs"] == 0
    assert empty.candidates("urn:s:heart")["items"] == []


def test_duplicate_fields_and_path_escape_are_rejected(tmp_path):
    options, _ = native_fixture(tmp_path)
    path = Path(options["native_dir"]) / "complete.json"
    path.write_text('{"status":"incomplete","status":"complete"}')
    with pytest.raises(DomainError, match="Duplicate JSON"):
        bind_staged(options, input_root=tmp_path)
    options, _ = native_fixture(tmp_path / "escape")
    path = Path(options["native_dir"]) / "complete.json"
    receipt = read(path)
    receipt["outputs"]["../outside"] = "0" * 64
    write(path, receipt)
    with pytest.raises(DomainError, match="safe relative"):
        bind_staged(options, input_root=tmp_path)


def acceptance_fixture(root, retained=False):
    options, contexts = native_fixture(root, mode="global")
    native = Path(options["native_dir"])
    model = write(
        root / "model.json",
        (
            dict(
                kind="retained_winner_acceptance",
                head=dict(accept_threshold=0.5, absolute_pair_floor=0.1),
            )
            if retained
            else dict(kind="edge_acceptance", policies={"full": dict(mode="greedy", threshold=0.5)})
        ),
    )
    result = root / "accepted"
    result.mkdir()
    (result / "mappings.csv").write_text(
        "SrcEntity,TgtEntity,Score,Relation\nurn:s:heart,urn:t:lung,0.75,=\n"
    )
    (result / "alignment.rdf").write_text(
        """<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"><Alignment><map><Cell><entity1 rdf:resource="urn:s:heart"/><entity2 rdf:resource="urn:t:lung"/><measure>0.75</measure><relation>=</relation></Cell></map></Alignment></rdf:RDF>"""
    )
    policy = dict(
        name="retained_full" if retained else "full",
        mode="winner" if retained else "greedy",
        threshold=0.5,
        candidates_sha256=file_hash(native / "pairs.jsonl"),
        model_manifest_sha256=file_hash(model),
        private_notes={"gold": "PRIVATE_GOLD", "path": "PRIVATE_PATH"},
    )
    if retained:
        policy.update(pair_floor=0.1, complete_candidate_pairs=2)
    write(
        result / "completion.json",
        dict(
            status="complete",
            policy=policy,
            mappings=1,
            candidate_pairs=1 if retained else 2,
            outputs={name: file_hash(result / name) for name in ("mappings.csv", "alignment.rdf")},
        ),
    )
    options.update(acceptance_dir=str(result), model_manifest=str(model))
    return options, contexts


@pytest.mark.parametrize("retained", [False, True])
def test_acceptance_membership_roundtrip_and_private_policy_redaction(tmp_path, retained):
    from exact_inspect.contracts import VisibilityPolicy
    from exact_inspect.decisions_export import export_policy_run

    options, contexts = acceptance_fixture(tmp_path, retained)
    store = build(options, contexts, tmp_path / "view")
    exported = export_policy_run(
        store, tmp_path / "portable", contexts={}, policy=VisibilityPolicy()
    )
    assert exported.manifest()["counts"]["sources"] == 2
    candidates = exported.candidates("urn:s:heart")["items"]
    assert {row["target"]["iri"] for row in candidates if row["saved_alignment_member"]} == {
        "urn:t:lung"
    }
    for row in candidates:
        trace = exported.pair(row["pair_id"])
        PairResponse.model_validate(trace)
        event = next(e for e in trace["events"] if e["stage"] == "selection")
        assert event["reason_code"] == (
            "staged_acceptance_selected"
            if row["saved_alignment_member"]
            else "staged_acceptance_not_selected"
        )
        assert (
            next(e for e in trace["events"] if e["stage"] == "threshold")["status"]
            == "not_recorded"
        )
        assert trace["relation"] == ("equivalent" if row["saved_alignment_member"] else None)
    content = b"".join(p.read_bytes() for p in (tmp_path / "portable").iterdir())
    assert b"PRIVATE_GOLD" not in content and b"PRIVATE_PATH" not in content


def test_rehashed_csv_rdf_disagreement_is_rejected(tmp_path):
    options, contexts = acceptance_fixture(tmp_path)
    root = Path(options["acceptance_dir"])
    path = root / "alignment.rdf"
    path.write_text(path.read_text().replace("0.75", "0.76"))
    receipt = read(root / "completion.json")
    receipt["outputs"]["alignment.rdf"] = file_hash(path)
    write(root / "completion.json", receipt)
    with pytest.raises(DomainError, match="CSV and RDF disagree"):
        build(options, contexts, tmp_path / "view")


def test_scored_anchor_is_not_reported_as_a_skipped_scoring_stage(tmp_path):
    options, contexts = ranking_fixture(tmp_path)
    store = build(options, contexts, tmp_path / "view")
    anchor = next(
        c
        for c in store.candidates("urn:s:heart", query_id=9)["items"]
        if c["target"]["iri"] == "urn:t:heart"
    )
    trace = store.pair(anchor["pair_id"])
    assert next(e for e in trace["events"] if e["stage"] == "pair_scoring")["status"] == "completed"
    assert (
        next(e for e in trace["events"] if e["stage"] == "prefilter")["reason_code"]
        == "exact_match_protected"
    )


def decisions_fixture(root):
    import hashlib

    options, contexts = native_fixture(root, mode="global")
    native = Path(options["native_dir"])
    rows = [json.loads(line) for line in (native / "pairs.jsonl").read_text().splitlines()]
    for row in rows:
        row.update(
            selected_labels={"source": "heart", "target": row["Tgt"]},
            llm_structured_evidence_packet="saved ontology evidence",
        )
    (native / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    complete = read(native / "complete.json")
    complete["outputs"]["pairs.jsonl"] = file_hash(native / "pairs.jsonl")
    write(native / "complete.json", complete)
    directory = root / "decisions"
    directory.mkdir()
    payload = {
        "state": {
            "evidence": "Source entity: heart\nTarget entity: urn:t:heart\n\nPair brief:\nsaved ontology evidence"
        },
        "questions": {
            "equivalent": {
                "type": "noul",
                "instructions": "You are a binary classifier for ontology alignment. The source and target entities are equivalent.",
            }
        },
    }
    identity = {
        "schema": "exact-decisions-plan/1",
        "policy": "global",
        "model": "saved-model",
        "revision": "saved-revision",
        "candidates_sha256": file_hash(native / "pairs.jsonl"),
        "sources_sha256": file_hash(native / "sources.txt"),
    }
    request_id = hashlib.sha256(
        json.dumps(
            {"model": identity["model"], "revision": identity["revision"], "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    request = dict(id=request_id, pair=["urn:s:heart", "urn:t:heart"], payload=payload)
    (directory / "requests.jsonl").write_text(json.dumps(request) + "\n")
    plan = write(
        directory / "plan.json",
        dict(
            identity=identity,
            requests_sha256=file_hash(directory / "requests.jsonl"),
            request_count=1,
            source_count=2,
        ),
    )
    for i, row in enumerate(rows):
        weight = 0.8 * row["U"] if i == 0 else 0.0
        score = (1 - weight) * row["S_base"] + weight * 0.5 if i == 0 else row["S_base"]
        row.update(
            S_final=score,
            S_pair_final=score,
            p_llm=0.5 if i == 0 else None,
            llm_used=i == 0,
            llm_weight=weight,
            decision_request_id=request_id if i == 0 else None,
        )
    fused = root / "fused.jsonl"
    fused.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(
        Path(str(fused) + ".receipt.json"),
        dict(
            schema="exact-decisions-application/1",
            plan_sha256=file_hash(plan),
            candidates_sha256=file_hash(native / "pairs.jsonl"),
            output_sha256=file_hash(fused),
            rows=2,
            judgments_applied=1,
            changed_scores=1,
            fusion=dict(mode="beta_u", beta=0.8),
        ),
    )
    options.update(scored_pairs=str(fused), decision_plan=str(plan))
    return options, contexts


def test_native_event_keeps_pre_decision_scores(tmp_path):
    options, contexts = decisions_fixture(tmp_path)
    store = build(options, contexts, tmp_path / "view")
    row = next(
        c for c in store.candidates("urn:s:heart")["items"] if c["target"]["iri"] == "urn:t:heart"
    )
    trace = store.pair(row["pair_id"])
    native = next(e for e in trace["events"] if e["stage"] == "pair_scoring")
    assert next(s["value"] for s in native["scores"] if s["name"] == "S_final") == 1.0
    assert next(s["value"] for s in row["scores"] if s["name"] == "S_final") < 1.0


def test_changed_request_payload_cannot_reuse_id_even_with_updated_outer_hashes(tmp_path):
    options, contexts = decisions_fixture(tmp_path)
    plan = Path(options["decision_plan"])
    requests = plan.parent / "requests.jsonl"
    row = json.loads(requests.read_text())
    row["payload"]["state"]["evidence"] = "changed private evidence"
    requests.write_text(json.dumps(row) + "\n")
    data = read(plan)
    data["requests_sha256"] = file_hash(requests)
    write(plan, data)
    receipt = Path(options["scored_pairs"] + ".receipt.json")
    data = read(receipt)
    data["plan_sha256"] = file_hash(plan)
    write(receipt, data)
    with pytest.raises(DomainError, match="request payload"):
        build(options, contexts, tmp_path / "view")


@pytest.mark.parametrize("field", ["cand_channels", "support", "cand_rank"])
def test_nested_private_data_cannot_enter_public_scalar_fields(tmp_path, field):
    options, contexts = native_fixture(tmp_path, mode="global")
    native = Path(options["native_dir"])
    rows = [json.loads(line) for line in (native / "pairs.jsonl").read_text().splitlines()]
    private = {"path": "PRIVATE_PATH", "gold": "PRIVATE_GOLD"}
    if field == "support":
        rows[0]["selected_attributions"] = {
            "attributes": {
                "source": [
                    {
                        "property_iri": "urn:property",
                        "value": "ontology literal",
                        "support": private,
                    }
                ],
                "target": [],
            }
        }
    else:
        rows[0][field] = private
    (native / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    complete = read(native / "complete.json")
    complete["outputs"]["pairs.jsonl"] = file_hash(native / "pairs.jsonl")
    write(native / "complete.json", complete)
    with pytest.raises(ValueError):
        build(options, contexts, tmp_path / "view")
    assert not (tmp_path / "view" / "decisions.manifest.json").exists()


def test_rehashed_changed_predicate_is_rejected(tmp_path):
    import hashlib

    options, contexts = decisions_fixture(tmp_path)
    plan = Path(options["decision_plan"])
    requests = plan.parent / "requests.jsonl"
    row = json.loads(requests.read_text())
    row["payload"]["questions"]["equivalent"][
        "instructions"
    ] = "The source and target entities are NOT equivalent."
    document = read(plan)
    identity = document["identity"]
    row["id"] = hashlib.sha256(
        json.dumps(
            {
                "model": identity["model"],
                "revision": identity["revision"],
                "payload": row["payload"],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    requests.write_text(json.dumps(row) + "\n")
    document["requests_sha256"] = file_hash(requests)
    write(plan, document)
    fused = Path(options["scored_pairs"])
    rows = [json.loads(line) for line in fused.read_text().splitlines()]
    rows[0]["decision_request_id"] = row["id"]
    fused.write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = Path(str(fused) + ".receipt.json")
    document = read(receipt)
    document.update(plan_sha256=file_hash(plan), output_sha256=file_hash(fused))
    write(receipt, document)
    with pytest.raises(DomainError, match="incompatible predicate"):
        build(options, contexts, tmp_path / "view")
