"""Small ontology projection provenance, isolation and resumability contracts."""
import dataclasses
import json
from pathlib import Path

import pytest
import pyowl_core as owl

from exact.repair.records import canonical_hash, read_record
from exact.repair.workers import CallResult
from tools.repair import real_modules as modules
from tools.repair.expanded_corpus import binding, bound
from tools.repair.expanded_profile import checkpoint
from tools.repair.prepare import case_from_dict, case_to_dict


def write(path, value):
    path.write_text(json.dumps(value))
    return binding(path)


def xml_document(prefix):
    return f'''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#" xml:base="http://example.org/{prefix}">
      <owl:Class rdf:about="#A"><rdfs:subClassOf rdf:resource="#B"/></owl:Class>
      <owl:Class rdf:about="#C"><rdfs:subClassOf rdf:resource="#D"/></owl:Class>
    </rdf:RDF>'''


@pytest.fixture
def plan(tmp_path):
    pair = dict(id="bioml_2026_whole:NCIT:DOID", group_id="pair:doid:ncit", split="train",
                cohort="bioml_2026_whole", ontology_assets=["ncit", "doid"],
                ontology_names=["NCIT", "DOID"])
    captured = tmp_path / "release-license.txt"
    captured.write_text("Captured public release and attribution terms")
    provenance = dict(source=str(captured), sha256=modules.sha(captured))
    assets = []
    for name in pair["ontology_assets"]:
        path = tmp_path / (name + ".owl")
        path.write_text(xml_document(name))
        asset = dict(id=name, role="ontology", revision=modules.REVISION,
                     requires_real_matcher_inputs=False, **binding(path))
        assets.append(dict(asset=asset, lineage=dict(license_status="captured_public_terms",
                            license_claims=[provenance], release_links=[provenance])))
    protocol = write(tmp_path / "protocol.json", {})
    model = tmp_path / "frozen.pt"
    model.write_bytes(b"\x80\x02binary checkpoint must not be JSON decoded")
    arms = [dict(id=f"model-{i}", kind="learned", model=binding(model), protocol=protocol,
                 status="available") for i in range(6)]
    arms += [dict(id=name, kind="control", protocol=protocol, status="available")
             for name in ("symbolic_rich_action", "uniform", "deletion")]
    return dict(provenance=write(tmp_path / "provenance.json", dict(pairs=[pair], assets=assets)),
                splits=write(tmp_path / "splits.json", dict(pairs=[pair])), pair_id=pair["id"],
                parents=2, matcher_inputs_consumed=0, attribution="NCIT and DOID attribution",
                model_schedule=write(tmp_path / "models.json", dict(arms=arms, control_definition={})),
                program=write(tmp_path / "program.json", {"scope": "authorized"}),
                authorization=write(tmp_path / "authorization.json", {"api_spend": 0}))


def rebind(plan, key, mutate):
    path = Path(plan[key]["path"])
    value = json.loads(path.read_text())
    mutate(value)
    plan[key] = write(path, value)


@pytest.mark.parametrize("key,value", [
    ("split", "test"), ("group_id", "new-group"),
    ("ontology_assets", ["doid", "ncit"]), ("cohort", "conference_2025"),
])
def test_previous_source_partition_and_split_are_authoritative(plan, key, value):
    rebind(plan, "provenance", lambda p: p["pairs"][0].update({key: value}))
    with pytest.raises(ValueError, match="frozen non-test pair split"):
        modules.validate_plan(plan)


def test_prior_test_pair_cannot_be_reclassified_by_matching_metadata(plan):
    for key in ("provenance", "splits"):
        rebind(plan, key, lambda p: p["pairs"][0].update(split="test"))
    with pytest.raises(ValueError, match="non-test"):
        modules.validate_plan(plan)


@pytest.mark.parametrize("change", ["license", "release", "revision", "matcher", "ontology", "captured"])
def test_release_license_and_ontology_byte_pins_are_enforced(plan, change):
    document = bound(plan["provenance"])
    row = document["assets"][0]
    if change == "license": row["lineage"]["license_status"] = "unresolved"
    elif change == "release": row["lineage"]["release_links"] = []
    elif change == "revision": row["asset"]["revision"] = "2025"
    elif change == "matcher": row["asset"]["requires_real_matcher_inputs"] = True
    elif change == "ontology": Path(row["asset"]["path"]).write_text("changed")
    elif change == "captured": Path(row["lineage"]["license_claims"][0]["source"]).write_text("changed")
    plan["provenance"] = write(Path(plan["provenance"]["path"]), document)
    with pytest.raises(ValueError):
        modules.validate_plan(plan)


def test_extraction_selects_named_lexical_edges_and_ignores_nested_restrictions(plan):
    pair, assets = modules.validate_plan(plan)
    asset = assets[0]
    path = Path(asset["path"])
    document = path.read_text().replace("</rdf:RDF>", '''
       <owl:Class rdf:about="#A"><rdfs:subClassOf rdf:resource="#B"/></owl:Class>
       <owl:Class rdf:about="#Z"><rdfs:subClassOf><owl:Restriction>
          <rdfs:subClassOf rdf:resource="#ZERO"/>
       </owl:Restriction></rdfs:subClassOf></owl:Class>
       <owl:Class rdf:about="#Self"><rdfs:subClassOf rdf:resource="#Self"/></owl:Class>
       <owl:Class rdf:about="#Top"><rdfs:subClassOf rdf:resource="http://www.w3.org/2002/07/owl#Thing"/></owl:Class>
       </rdf:RDF>''')
    path.write_text(document)
    asset = {**asset, **binding(path)}
    result = modules.extract_edges(asset, 2)
    assert result["edges"] == [("http://example.org/ncit#A", "http://example.org/ncit#B"),
                               ("http://example.org/ncit#C", "http://example.org/ncit#D")]
    assert result["explicit_named_subclass_statements"] == 3
    assert pair["split"] == "train"


def test_clean_corrupt_cases_preserve_source_partition_and_evaluator_isolation(plan):
    pair, _ = modules.validate_plan(plan)
    edges = [["urn:ncit:A", "urn:ncit:B"], ["urn:doid:C", "urn:doid:D"]]
    cases = [modules.make_case(pair, edges, 0, variant)
             for variant in ("coherent", "strengthen_first", "strengthen_both")]
    assert len({case.structural_parent for case in cases}) == 1
    assert cases[0].structural_parent == pair["group_id"]
    assert all(case.split == "train" for case in cases)
    assert [[len(obj.original_axioms) for obj in case.problem.objects] for case in cases] == [[1, 1], [2, 1], [2, 2]]
    for case in cases:
        problem = case.problem
        assert set(problem.fixed_axioms) == set(problem.source_axioms + problem.target_axioms)
        assert problem.source_identity == canonical_hash(problem.source_axioms)
        assert problem.target_identity == canonical_hash(problem.target_axioms)
        assert problem.matcher_identity == "declared_synthetic_no_matcher"
        selected = tuple(axiom for obj, index in zip(problem.objects, case.intended_assignment)
                         for axiom in obj.candidates[index].axioms)
        assert set(case.intended_theory) == set(problem.fixed_axioms + selected)
        observable = problem.to_dict()
        assert not any(key in observable for key in ("probes", "intended_assignment", "intended_theory"))
        assert read_record(observable) == problem
        assert case_from_dict(case_to_dict(case)) == case
    # Logical witness: adding the reverse c <= a forces the synthetic x below
    # a while the immutable source asserts a disjoint x. The clean forward
    # assignment omits that reverse and preserves both intended bridges.
    corrupt = cases[1]
    negative = next(probe for probe in corrupt.probes if not probe.desired)
    assert negative.axiom in corrupt.problem.objects[0].original_axioms
    assert negative.axiom not in corrupt.intended_theory


@pytest.fixture
def cheap_native(monkeypatch):
    calls = []
    def call(function, *args, **kwargs):
        calls.append((function.__name__, args))
        return CallResult("complete", value=function(*args))
    monkeypatch.setattr("exact.repair.workers.bounded_call", call)
    monkeypatch.setattr(modules, "verify_theory", lambda case, scope: {"authorizes": True, "report": {"scope": scope}})
    monkeypatch.setattr("exact.repair.study.runtime_manifest", lambda: {"native": "test", "code": "v1"})
    return calls


def test_preparation_54_rows_binary_weights_and_valid_resume(plan, tmp_path, cheap_native):
    path = tmp_path / "plan.json"
    write(path, plan)
    output = tmp_path / "prepared"
    completion = modules.prepare(path, output)
    assert completion["scheduled_cases"] == 6
    assert completion["scheduled_rows"] == 54
    schedule = bound(completion["schedule"])
    assert len(schedule["arms"]) == 9
    assert schedule["independent_pair_count"] == 1
    assert all(row["status"] == "materialized" for row in schedule["cases"])
    assert len(cheap_native) == 10  # two extraction + four qualification calls per parent
    assert modules.prepare(path, output) == completion
    assert len(cheap_native) == 10
    observable = Path(schedule["cases"][0]["observable"]["path"])
    observable.write_text("{}")
    with pytest.raises(ValueError, match="binding changed"):
        modules.prepare(path, output)


def test_resume_revalidates_model_dependencies_before_any_work(plan, tmp_path, cheap_native):
    path = tmp_path / "plan.json"
    write(path, plan)
    modules.prepare(path, tmp_path / "prepared")
    schedule = bound(plan["model_schedule"])
    Path(schedule["arms"][0]["model"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen inference artifact changed"):
        modules.prepare(path, tmp_path / "prepared")
    assert len(cheap_native) == 10


def test_failed_parent_is_not_refilled_and_stays_in_every_arm_denominator(plan, tmp_path, cheap_native, monkeypatch):
    monkeypatch.setattr(modules, "verify_theory", lambda case, scope: {"authorizes": False, "report": {}})
    path = tmp_path / "plan.json"
    write(path, plan)
    completion = modules.prepare(path, tmp_path / "prepared")
    schedule = bound(completion["schedule"])
    assert len(schedule["rows"]) == 54
    assert len(schedule["cases"]) == 6
    assert all(row["status"] == "parent_qualification_unknown_or_failed" for row in schedule["cases"])
    assert all("observable" not in row for row in schedule["cases"])


def test_saved_call_guards_corruption_dependencies_and_unreconciled_cleanup(tmp_path, monkeypatch):
    path = tmp_path / "call.json"
    checkpoint(path, "identity", status="complete", value={"answer": 1}, cleanup_complete=True)
    monkeypatch.setattr("exact.repair.workers.bounded_call", lambda *a, **k: pytest.fail("Must reuse or reject"))
    assert modules.saved_call(path, "identity", None)["value"] == {"answer": 1}
    with pytest.raises(ValueError, match="dependencies changed"):
        modules.saved_call(path, "different", None)
    guard = path.with_suffix(".inflight.json")
    guard.write_text("unreconciled native owner")
    with pytest.raises(RuntimeError, match="ownership/budget"):
        modules.saved_call(path, "identity", None)
    guard.unlink()
    checkpoint(path, "identity", status="failed", value=None, cleanup_complete=False)
    with pytest.raises(RuntimeError, match="Prior native cleanup incomplete"):
        modules.saved_call(path, "identity", None)
    saved = json.loads(path.read_text())
    saved["cleanup_complete"] = True
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="content changed"):
        modules.saved_call(path, "identity", None)


@pytest.mark.slow
def test_native_projection_qualification_and_symbolic_witness():
    """Bounded smoke for the exact constructed theory, not full NCIT/DOID."""
    from exact.repair.workers import bounded_call
    pair = dict(id="NCIT:DOID", group_id="pair:doid:ncit", split="train")
    edges = [["urn:ncit:A", "urn:ncit:B"], ["urn:doid:C", "urn:doid:D"]]
    clean = modules.make_case(pair, edges, 0, "coherent")
    for scope in ("source", "target", "union", "intended"):
        outcome = bounded_call(modules.verify_theory, clean, scope,
                               timeout=60, memory_mb=8192, cpu_seconds=120)
        assert outcome.cleanup_complete
        assert outcome.status == "complete", outcome.detail
        assert outcome.value["authorizes"] is True, outcome.value
    for variant in ("strengthen_first", "strengthen_both"):
        corrupt = modules.make_case(pair, edges, 0, variant)
        original_theory = corrupt.problem.fixed_axioms + tuple(
            axiom for obj in corrupt.problem.objects for axiom in obj.original_axioms)
        exposed = dataclasses.replace(corrupt, intended_theory=original_theory)
        outcome = bounded_call(modules.verify_theory, exposed, "intended",
                               timeout=60, memory_mb=8192, cpu_seconds=120)
        assert outcome.cleanup_complete
        assert outcome.status == "complete", outcome.detail
        assert outcome.value["authorizes"] is False
        report = outcome.value["report"]
        assert report["logical_status"] == "VERIFIED_INFEASIBLE"
        assert any(row["kind"] == "class_satisfiability" and row["verdict"] is False
                   and row["class_iri"].endswith(":witness") for row in report["obligations"])



def test_preparation_stage_lock_prevents_concurrent_direct_resume(tmp_path, monkeypatch):
    import fcntl
    output = tmp_path / "prepared"
    def guarded(plan, directory):
        assert directory == output
        with (output / "stage.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {"status": "lock_verified"}
    monkeypatch.setattr(modules, "_prepare", guarded)
    assert modules.prepare("plan.json", output) == {"status": "lock_verified"}
    with (output / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            modules.prepare("plan.json", output)



@pytest.mark.parametrize("status", ["error", "worker_error"])
def test_native_software_error_is_persisted_then_raised_without_reexecution(tmp_path, monkeypatch, status):
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        return CallResult(status, detail="AttributeError: missing qualification field", cleanup_complete=True)
    monkeypatch.setattr("exact.repair.workers.bounded_call", failed)
    path = tmp_path / "qualified.json"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Recorded native software failure"):
            modules.saved_call(path, "identity", None)
    assert calls == [1]
    saved = json.loads(path.read_text())
    assert saved["status"] == status
    assert saved["cleanup_complete"] is True
    assert not path.with_suffix(".inflight.json").exists()


@pytest.mark.parametrize("status", ["timeout", "unknown"])
def test_native_unknown_and_timeout_remain_recorded_unavailable_outcomes(tmp_path, monkeypatch, status):
    monkeypatch.setattr("exact.repair.workers.bounded_call", lambda *a, **k: CallResult(status))
    path = tmp_path / "qualified.json"
    assert modules.saved_call(path, "identity", None)["status"] == status
