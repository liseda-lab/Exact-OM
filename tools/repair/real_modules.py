"""Controlled native-v3 repairs on explicit 2026 ontology projections.

These small derived theories omit imports and unselected axioms. They are not
locality modules, source-coherence certificates, or production matcher outputs.
"""
from __future__ import annotations
import argparse
import fcntl
import heapq
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit
import xml.etree.ElementTree as ET

from exact.repair.records import canonical_hash
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound, immutable
from tools.repair.expanded_profile import checked_checkpoint, checkpoint

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
XML_BASE = "{http://www.w3.org/XML/1998/namespace}base"
SCOPE = "Explicit named-subclass projections plus synthetic context/bridges; omitted source axioms/imports are outside this new theory, no full-source or locality-module claim"
REVISION = "c454644a334ab43754bc1070c0fb7fdd56a90a1d"

def iri(value, base):
    value = urljoin(base, value)
    if not urlsplit(value).scheme or re.search(r'[<>"{}|^`\\\s]', value):
        return None
    if value in {"http://www.w3.org/2002/07/owl#Nothing", "http://www.w3.org/2002/07/owl#Thing"}:
        return None
    return value


def extract_edges(asset, count):
    """Stream only explicit top-level owl:Class/named rdfs:subClassOf statements.

    No inferred edges, nested restrictions, equivalences, labels, imports, or
    correspondence rows are read as supervision. Select lexical first distinct
    edges before any semantic verification; never refill after a failed parent.
    """
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("Explicit projection extraction requires one to eight edges")
    if asset["role"] != "ontology" or asset.get("requires_real_matcher_inputs", False):
        raise ValueError("Only ontology structure may be consumed")
    if sha(asset["path"]) != asset["sha256"]:
        raise ValueError("Ontology bytes changed")
    stack, bases, selected, seen, total = [], [], [], set(), 0
    for event, node in ET.iterparse(asset["path"], events=("start", "end")):
        if event == "start":
            parent_base = bases[-1] if bases else Path(asset["path"]).as_uri()
            bases.append(
                urljoin(parent_base, node.attrib[XML_BASE])
                if XML_BASE in node.attrib
                else parent_base
            )
            stack.append(node)
            continue
        if (
            len(stack) == 3
            and stack[0].tag == RDF + "RDF"
            and stack[1].tag == OWL + "Class"
            and node.tag == RDFS + "subClassOf"
            and RDF + "resource" in node.attrib
        ):
            subject = stack[1].get(RDF + "about")
            if subject is None and RDF + "ID" in stack[1].attrib:
                subject = "#" + stack[1].get(RDF + "ID")
            left = iri(subject, bases[-2]) if subject is not None else None
            right = iri(node.get(RDF + "resource"), bases[-1])
            if left and right and left != right:
                total += 1
                edge = (left, right)
                if edge not in seen:
                    # At most count selected edges are retained in memory.
                    selected = heapq.nsmallest(count, [*selected, edge])
                    seen = set(selected)
        stack.pop()
        bases.pop()
        node.clear()
        if stack:
            stack[-1].remove(node)
    return dict(
        asset=asset,
        edges=selected,
        explicit_named_subclass_statements=total,
        extraction_scope="top-level explicit named subclass statements only; no imports",
        selection="lexical first distinct edges; no semantic refill",
    )


def validate_plan(plan):
    provenance, splits = bound(plan["provenance"]), bound(plan["splits"])
    pair = next(p for p in provenance["pairs"] if p["id"] == plan["pair_id"])
    saved = next(p for p in splits["pairs"] if p["id"] == plan["pair_id"])
    if (any(pair.get(key) != saved.get(key) for key in
            ("split", "group_id", "cohort", "ontology_assets", "ontology_names"))
            or pair["split"] not in {"train", "development"}):
        raise ValueError("Real structure must inherit the previously frozen non-test pair split")
    if (pair["cohort"] != "bioml_2026_whole" or pair.get("ontology_names") != ["NCIT", "DOID"]
            or plan["matcher_inputs_consumed"] != 0):
        raise ValueError("Only pinned 2026 ontology structure is in scope")
    assets = []
    for key in pair["ontology_assets"]:
        record = next(r for r in provenance["assets"] if r["asset"]["id"] == key)
        asset, lineage = record["asset"], record["lineage"]
        if (asset["role"] != "ontology" or asset["revision"] != REVISION
                or asset.get("requires_real_matcher_inputs") is not False
                or lineage["license_status"] != "captured_public_terms"
                or not lineage["release_links"] or not lineage["license_claims"]):
            raise ValueError("Missing release/license or ontology-only provenance")
        for link in lineage["release_links"] + lineage["license_claims"]:
            if sha(link["source"]) != link["sha256"]:
                raise ValueError("Captured release/license provenance changed")
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("Pinned ontology changed")
        assets.append(asset)
    if (len(assets) != 2 or len({asset["id"] for asset in assets}) != 2
            or type(plan["parents"]) is not int or not 1 <= plan["parents"] <= 8):
        raise ValueError("Require a small explicitly bounded pair construction")
    return pair, assets


def make_case(pair, edges, ordinal, variant):
    import pyowl_core as owl
    from exact.repair.candidates import make_candidate, replacement_cost_features
    from exact.repair.learning import TeacherProbe
    from exact.repair.records import RepairInputV2, RevisionObjectV2, PolicyV2, promote_input_v3
    from tools.repair.corpus import GeneratedCase
    cls=lambda name: owl.Class(owl.IRI(name))
    parent=canonical_hash((pair["id"],edges,ordinal,"20261004-projection"))
    a,b,c,d=map(cls,(*edges[0],*edges[1]))
    if len({a,b,c,d}) != 4:
        raise ValueError("Projection endpoints overlap; no semantic refill")
    x=cls("urn:exact:real-projection:"+parent+":witness")
    source=(owl.SubClassOf(a,b),owl.DisjointClasses((a,x)))
    target=(owl.SubClassOf(c,d),owl.SubClassOf(x,c))
    clean=(owl.SubClassOf(a,c),owl.SubClassOf(b,d))
    if variant not in {"coherent","strengthen_first","strengthen_both"}:
        raise ValueError("Unknown frozen corruption")
    objects=[];intended=[]
    for index,(left,right) in enumerate(((a,c),(b,d))):
        original=(clean[index],)
        if variant == "strengthen_both" or (variant == "strengthen_first" and index == 0):
            original += (owl.SubClassOf(right,left),)
        oid=f"bridge-{index}"
        choices=[("keep",original),("delete",())]
        if len(original)==2:choices += [("forward",(original[0],)),("reverse",(original[1],))]
        candidates=tuple(make_candidate(oid,axioms,(tag,),cost_features=replacement_cost_features(original,axioms,kind="mapping",authorship="generated")) for tag,axioms in choices)
        objects.append(RevisionObjectV2(oid,"mapping",original,candidates,source_entity=left,target_entity=right,authorship="generated"))
        intended.append(next(i for i,candidate in enumerate(candidates) if candidate.axioms==(clean[index],)))
    problem=promote_input_v3(RepairInputV2((*source,*target),tuple(objects),PolicyV2((a,b,c,d,x)),source_axioms=source,target_axioms=target,source_identity=canonical_hash(source),target_identity=canonical_hash(target),matcher_identity="declared_synthetic_no_matcher",evidence=tuple((o.object_id,{"score":0.5}) for o in objects)))
    probes=tuple(TeacherProbe(f"desired-{i}",axiom,"subsumption") for i,axiom in enumerate(clean))+(TeacherProbe("unwanted-reverse",owl.SubClassOf(c,a),"subsumption",False),)
    return GeneratedCase(parent+":"+variant,pair["group_id"],"real_projection_directional_strengthening",pair["split"],problem,probes,tuple(intended),20261004,False,origin="real_structure",control="coherent" if variant=="coherent" else "corrupted",intended_theory=(*source,*target,*clean),variation=(("derived_parent",parent),("scope",SCOPE),("variant",variant)),schema_revision="v3")


def verify_theory(case, scope):
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms
    theory={"source":case.problem.source_axioms,"target":case.problem.target_axioms,"union":case.problem.fixed_axioms,"intended":case.intended_theory}[scope]
    report=OwlVerifier(timeout_seconds=45).check_theory(snapshot_from_axioms(theory))
    return dict(authorizes=report.complete and report.logical_status == "VERIFIED_FEASIBLE",report=report.to_dict())


def saved_call(path, identity, function, *args, seconds=60):
    from exact.repair.workers import bounded_call
    saved=checked_checkpoint(path,identity)
    guard=path.with_suffix(".inflight.json")
    if guard.exists():raise RuntimeError("Incomplete native call requires ownership/budget reconciliation")
    if saved is not None:
        if not saved.get("cleanup_complete"):
            raise RuntimeError("Prior native cleanup incomplete")
        return saved
    checkpoint(guard,identity,started_epoch=time.time())
    outcome=bounded_call(function,*args,timeout=seconds,memory_mb=8192,cpu_seconds=120)
    saved=checkpoint(path,identity,status=outcome.status,value=outcome.value,detail=outcome.detail,cleanup_complete=outcome.cleanup_complete,resources=dict(outcome.resource_usage))
    if not outcome.cleanup_complete:raise RuntimeError("Native cleanup incomplete")
    guard.unlink()
    return saved


def prepare(plan_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(plan_path, output)


def _prepare(plan_path, output):
    from exact.repair.study import runtime_manifest
    from tools.repair.prepare import case_to_dict
    from tools.repair.fresh_evaluation import SCHEMA, make_rows
    plan=read(plan_path);pair,assets=validate_plan(plan);output=Path(output)
    source=bound(plan["model_schedule"])
    arms=source["arms"]
    from tools.repair.fresh_evaluation import CONTROLS
    if (len(arms)!=9 or len({arm["id"] for arm in arms}) != 9
            or sum(arm["kind"] == "learned" for arm in arms) != 6
            or {arm["id"] for arm in arms if arm["kind"] == "control"} != set(CONTROLS)):
        raise ValueError("Six fixed pilot models and three distinct controls required")
    for arm in arms:
        for key in ("model","protocol","completion","training_report"):
            if key in arm and sha(arm[key]["path"]) != arm[key]["sha256"]:
                raise ValueError("Frozen inference artifact changed: " + key)
    for key in ("program", "authorization"):
        bound(plan[key])
    runtime=runtime_manifest()
    dependencies={name:sha(Path(__file__).with_name(name)) for name in
                  ("real_modules.py","expanded_corpus.py","expanded_profile.py",
                   "prepare.py","fresh_evaluation.py","corpus.py","batch.py")}
    identity=canonical_hash((plan,dependencies,runtime))
    output.mkdir(parents=True,exist_ok=True)
    existing=checked_checkpoint(output/"completion.json",identity)
    if existing:
        for item in existing["artifacts"]:bound(item)
        return existing
    artifacts=[]
    def keep(path,value):
        ref=immutable(path,value);artifacts.append(ref);return ref
    extractions=[]
    for asset in assets:
        path=output/"extractions"/(canonical_hash(asset)+".json")
        row=saved_call(path,canonical_hash((identity,asset)),extract_edges,asset,plan["parents"],seconds=120)
        artifacts.append(binding(path));extractions.append(row)
    cases=[];qualifications=[]
    for ordinal in range(plan["parents"]):
        edges=None
        if all(r["status"]=="complete" and len(r["value"]["edges"])>ordinal for r in extractions):
            edges=[r["value"]["edges"][ordinal] for r in extractions]
        clean=None;reason="extraction_unavailable";qualified=False
        if edges:
            try:clean=make_case(pair,edges,ordinal,"coherent")
            except ValueError as exc:reason=str(exc)
        if clean:
            receipts=[]
            for scope in ("source","target","union","intended"):
                path=output/"qualification"/f"{ordinal}-{scope}.json"
                row=saved_call(path,canonical_hash((identity,ordinal,scope)),verify_theory,clean,scope)
                receipts.append(row);artifacts.append(binding(path))
            qualified=all(r["status"]=="complete" and r["value"]["authorizes"] for r in receipts)
            reason="qualified" if qualified else "parent_qualification_unknown_or_failed"
        qualifications.append(dict(parent=ordinal,status=reason,qualified=qualified))
        for variant in ("coherent","strengthen_first","strengthen_both"):
            item=dict(case_id=f"real-projection-{ordinal}-{variant}",structural_parent=pair["group_id"],group_id=pair["group_id"],family="real_projection_directional_strengthening",family_exposure="historical_train_pair_exploratory",split=pair["split"],condition=variant,status="materialized" if qualified else reason)
            if qualified:
                case=make_case(pair,edges,ordinal,variant);item["case_id"]=case.case_id
                item["observable"]=keep(output/"observables"/(case.case_id+".json"),case.problem.to_dict())
                item["evaluator"]=keep(output/"evaluators"/(case.case_id+".json"),case_to_dict(case))
                item["input_hash"]=case.problem.content_hash
            cases.append(item)
    prepared=keep(output/"preparation.json",dict(schema="exact-repair/real-projections/v1",identity=identity,scope=SCOPE,pair=pair,assets=assets,qualifications=qualifications,case_count=len(cases),matcher_inputs_consumed=0,labels_used_for_fitting=0,source_coherence_qualified=False,attribution=plan["attribution"],split_inheritance=plan["splits"],independent_pair_count=1))
    schedule=dict(schema=SCHEMA,program=plan["program"],authorization=plan["authorization"],corpus_completion=prepared,split_schedule=plan["splits"],cases=cases,arms=arms,rows=make_rows(cases,arms),followup="xr21-expanded-real-modules-001",scope=SCOPE,source_coherence_qualified=False,seed=20261004,api_spend_usd=0,cache_policy="cold per row",independent_pair_count=1,semantic_interpretation="Known synthetic intent for constructed theory only; no production matcher or real patch gold",control_definition=source["control_definition"])
    schedule_ref=keep(output/"schedule.json",schedule)
    return checkpoint(output/"completion.json",identity,status="complete",schema="exact-repair/real-projections-completion/v1",schedule=schedule_ref,qualifications=qualifications,scheduled_cases=len(cases),scheduled_rows=len(schedule["rows"]),artifacts=artifacts,runtime=runtime,dependencies=dependencies,study_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("plan",type=Path);p.add_argument("output",type=Path);a=p.parse_args();prepare(a.plan,a.output)

if __name__=="__main__":main()
