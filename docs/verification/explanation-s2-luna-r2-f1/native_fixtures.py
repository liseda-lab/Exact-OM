"""Native-parser fixtures for typed citation identity (R2-F1): a punned label and a shared axiom.

Usage: PYTHONPATH=<checkout> python native_fixtures.py <empty output directory>
Writes citation-identity-fixtures.json, committed as
explanations_visualizer/src/lib/workspace/fixtures/citation-identity.json.
"""
import json, sys
from pathlib import Path

import pyowl_core as core
from exact_inspect.context import build_context_package
from exact_inspect.contracts import EntityRef, VisibilityPolicy

out = Path(sys.argv[1])
policy = VisibilityPolicy()

def package(name, text):
    return build_context_package(core.load_snapshot(text), out / name)

def shared_facts(context, subjects):
    """Fact records retained under every subject with one original fact ID, by category."""
    found = {}
    for category in ["labels", "definitions", "synonyms", "hierarchy", "restrictions", "axioms", "characteristics", "comments"]:
        per = []
        for subject in subjects:
            try:
                items = context.facts(subject, category=category, policy=policy)["items"]
            except Exception:
                items = []
            per.append({item["fact_id"]: item for item in items})
        common = set(per[0]).intersection(*per[1:])
        for fact_id in sorted(common):
            found[fact_id] = {"category": category, "facts": [p[fact_id] for p in per]}
    return found

punned = package("punned", b'''Ontology(
  Declaration(Class(<urn:punned>))
  Declaration(NamedIndividual(<urn:punned>))
  AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:punned> "Punned entity")
)''')
p_subjects = [EntityRef(ontology_version_id=punned.ontology_version_id, iri="urn:punned", kind=kind) for kind in ["class", "individual"]]
p_shared = shared_facts(punned, p_subjects)

shared = package("shared", b'''Ontology(
  Declaration(Class(<urn:a>))
  Declaration(Class(<urn:b>))
  AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:a> "Class A")
  AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:b> "Class B")
  EquivalentClasses(<urn:a> <urn:b>)
)''')
s_subjects = [EntityRef(ontology_version_id=shared.ontology_version_id, iri=iri, kind="class") for iri in ["urn:a", "urn:b"]]
s_shared = shared_facts(shared, s_subjects)

def entry(context, subjects, found):
    fact_id, value = next(iter(found.items()))
    return {
        "ontology_version_id": context.ontology_version_id,
        "subjects": [s.model_dump(mode="json") for s in subjects],
        "fact_id": fact_id,
        "category": value["category"],
        "facts": value["facts"],
        "axiom": context.axiom(fact_id, policy=policy),
    }

result = {
    "generator": "Native pyowl_core parser + exact_inspect.context.build_context_package (docs/verification/explanation-s2-luna-r2-f1/native_fixtures.py); same punned record as the R2-F1 review fixture",
    "punned_label": entry(punned, p_subjects, p_shared),
    "shared_axiom": entry(shared, s_subjects, s_shared),
}
(out / "citation-identity-fixtures.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: {"fact_id": v["fact_id"], "category": v["category"], "subjects": [f["subject"]["iri"] + "|" + f["subject"]["kind"] for f in v["facts"]], "axiom_type": v["axiom"]["ast"]["type"]} for k, v in result.items() if k != "generator"}, indent=1))
