"""Whole-source admission is separate from references and closed split labels."""

import itertools
import json
import zipfile

import pyowl_core as owl
import pytest

from exact.repair.records import read_record
from exact.repair.workers import CallResult
from tools.repair import corrective_conference as conference


def alignment(left="urn:left:A", right="urn:right:B", relation="=", score="0.7"):
    return f"""<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
       xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#">
       <Alignment><map><Cell><entity1 rdf:resource="{left}"/><entity2 rdf:resource="{right}"/>
       <relation>{relation}</relation><measure>{score}</measure></Cell></map></Alignment></rdf:RDF>""".encode()


@pytest.fixture
def release(tmp_path, monkeypatch):
    previous = tmp_path / "previous"
    previous.mkdir()
    names = ("cmt", "conference", "confOf", "edas", "ekaw", "iasted", "sigkdd")
    files, pairs = [], []
    for name in names:
        path = previous / (name + ".owl")
        if name != "ekaw":
            path.write_text("Ontology(<urn:" + name + "> Declaration(Class(<urn:" + name + ":A>)))")
            digest = conference.binding(path)["sha256"]
        else:
            digest = "closed-holdout-byte-binding"
        files.append(dict(id=name, path=str(path), sha256=digest))
    archive = tmp_path / "matchers.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for left, right in itertools.combinations(names, 2):
            pair_id = f"conference_2025:{left}:{right}"
            split = (
                "test"
                if "ekaw" in (left, right) or (left, right) == ("edas", "sigkdd")
                else "development" if (left, right) == ("conference", "iasted") else "train"
            )
            reference = previous / f"never-open-reference-{left}-{right}.rdf"
            files.append(dict(id=pair_id, path=str(reference), sha256="reference-binding-only"))
            pairs.append(
                dict(
                    id=pair_id,
                    group_id=pair_id,
                    cohort="conference_2025",
                    ontology_names=[left, right],
                    ontology_assets=[left, right],
                    reference_assets=[pair_id],
                    split=split,
                )
            )
            stream.writestr(
                f"LogMap-{left}-{right}.rdf",
                (
                    alignment(f"urn:{left}:A", f"urn:{right}:A")
                    if split == "train"
                    else b"CLOSED ROWS MUST NOT BE EXTRACTED"
                ),
            )
    (previous / "conference.zip").write_bytes(b"opaque ontology archive including closed bytes")
    monkeypatch.setattr(conference, "MATCHER_SHA", conference.binding(archive)["sha256"])
    monkeypatch.setattr(
        conference, "ONTOLOGY_SHA", conference.binding(previous / "conference.zip")["sha256"]
    )
    for name, value in {
        "assets.json": dict(files=files),
        "pair-splits.json": dict(pairs=pairs),
        "conference-receipt.json": dict(
            value=dict(url="https://publisher/conference.zip", ontology_hashes={})
        ),
    }.items():
        (previous / name).write_text(json.dumps(value))
    output = tmp_path / "output"
    manifest = conference.prepare(previous, archive, output)
    return manifest, output


def test_prepare_preserves_all_pairs_but_only_opens_train(release):
    manifest, output = release
    data = json.loads(conference.bound_bytes(manifest))
    assert data["scheduled"] == 21 and data["train_count"] == 13
    assert len(list((output / "train").glob("*/LogMap.rdf"))) == 13
    assert not (output / "ontologies/ekaw.owl").exists()
    for row in data["rows"]:
        assert row["reference_rows_opened"] is False
        if row["split"] != "train":
            assert "matcher" not in row
            with pytest.raises(ValueError, match="Only frozen TRAIN"):
                conference.train_row(manifest, row["id"])


def test_full_input_native_roundtrip_has_no_invented_teacher_target(release):
    manifest, _ = release
    value = conference.construct_native(manifest, "conference_2025:cmt:conference")
    problem = read_record(value["problem"])
    assert value["status"] == "constructed"
    assert value["mapping_count"] == 1
    assert value["complete_source_axiom_counts"] == [1, 1]
    assert problem.fixed_axioms == (*problem.source_axioms, *problem.target_axioms)
    assert len(problem.objects[0].candidates) == 2
    assert dict(problem.evidence)[problem.objects[0].object_id]["score"] == 0.7
    assert all(
        row["backend"] == "native" and row["complete_imports"] for row in value["source_routes"]
    )
    assert value["known_intended_theory"] is False
    assert value["reference_rows_opened"] is False


@pytest.mark.parametrize("relation,score", [("?", "0.7"), ("=", "nan"), ("=", "2")])
def test_unsupported_or_invalid_matcher_rows_are_not_silently_dropped(relation, score):
    left = owl.load_snapshot(b"Ontology(Declaration(Class(<urn:left:A>)))")
    right = owl.load_snapshot(b"Ontology(Declaration(Class(<urn:right:B>)))")
    with pytest.raises(ValueError):
        conference._mapping_axioms(alignment(relation=relation, score=score), left, right)


def test_reference_cannot_be_relabelled_matcher(release, tmp_path):
    manifest, _ = release
    data = json.loads(conference.bound_bytes(manifest))
    data["rows"][0]["matcher"] = data["rows"][0]["reference_bindings"][0]
    tampered = conference.immutable(tmp_path / "tampered.json", data)
    with pytest.raises(ValueError, match="Reference alignment"):
        conference.train_row(tampered, data["rows"][0]["id"])


def test_empty_publisher_alignment_remains_a_valid_scheduled_input():
    view = owl.load_snapshot(b"Ontology(Declaration(Class(<urn:left:A>)))")
    raw = b'<Alignment xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"/>'
    assert conference._mapping_axioms(raw, view, view) == []
    with pytest.raises(ValueError, match="Alignment envelope"):
        conference._mapping_axioms(b"<unrelated/>", view, view)


def test_qualification_resume_does_not_replay_completed_unknowns(release, tmp_path, monkeypatch):
    manifest, _ = release
    count = []

    def call(function, *args, **kwargs):
        count.append(function.__name__)
        value = (
            function(*args)
            if function is conference.construct_native
            else dict(logical_status="UNKNOWN", support=dict(input_supported=False))
        )
        return CallResult("complete", value=value)

    monkeypatch.setattr(conference, "bounded_call", call)
    directory = tmp_path / "qualified"
    report = conference.qualify(manifest, "conference_2025:cmt:conference", directory)
    assert len(count) == 5
    assert conference.qualify(manifest, "conference_2025:cmt:conference", directory) == json.loads(
        json.dumps(report)
    )
    assert len(count) == 5
    with pytest.raises(ValueError, match="Incompatible"):
        conference.qualify(manifest, "conference_2025:cmt:conference", directory, seconds=600)


def test_expired_stage_does_not_begin_native_check(release, tmp_path, monkeypatch):
    manifest, _ = release
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", "1")

    def expired(function, *args, **kwargs):
        assert kwargs["timeout"] < 0
        return CallResult("timeout", detail="stage deadline exhausted")

    monkeypatch.setattr(conference, "bounded_call", expired)
    result = conference.qualify(manifest, "conference_2025:cmt:conference", tmp_path / "expired")
    assert result["checks"] == {}
    assert "input" not in result
