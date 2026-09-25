"""Build a small inspection package for developing the Exact Explain frontend.

Ontology facts are verbatim excerpts of pinned NCIT/DOID entities recorded in
``specs/explanation-framework/evidence/real-entity-context-examples.json``. The
matcher run, its scores and every generated text are synthetic fixtures produced
offline: nothing here is an Exact result, a provider response or a study case.

The package goes through the real ``bind-lock`` -> ``prepare`` -> portable export
path, so the frontend reads genuine API shapes, statuses and cursors.

    python -m tools.build_explanation_ui_fixture --output /tmp/exact-ui-fixture
    exact-inspect serve --package /tmp/exact-ui-fixture/prepared/stages/<id>/package.json

Requires ``pyowl-core`` plus the ``exact`` run-store dependencies (pandas,
zstandard). No network access, model or GPU is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
OBO = "http://purl.obolibrary.org/obo/"
OIO = "http://www.geneontology.org/formats/oboInOwl#"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
FIXTURE_NOTE = "Synthetic frontend fixture: not an Exact result"


def _literal(text: str, language: str | None = None) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"' + (f"@{language}" if language else "")


def _annotation(predicate: str, subject: str, text: str, language: str | None = None) -> str:
    return f"AnnotationAssertion(<{predicate}> <{subject}> {_literal(text, language)})"


def ncit_excerpt() -> str:
    """Original NCIT annotations/axioms for three development sources."""
    axioms = []
    classes = {
        "C116573": "Dravet Syndrome",
        "C114940": "Malignant Cardiovascular Neoplasm",
        "C101223": "Complete Trisomy 13 Syndrome",
        "C28193": "Syndrome",
        "C3020": "Seizure Disorder",
    }
    for code, label in classes.items():
        axioms += [f"Declaration(Class(<{NCIT}{code}>))", _annotation(LABEL, NCIT + code, label)]
    # Fillers whose labels were not captured by the evidence profile stay unlabelled.
    for code in ("C4784", "C9305", "C12686", "C102447", "C36529", "C13208"):
        axioms.append(f"Declaration(Class(<{NCIT}{code}>))")
    for code, label in {
        "R100": "Disease_Has_Associated_Anatomic_Site",
        "R106": "Disease_Has_Molecular_Abnormality",
        "R174": "Disease_Mapped_To_Chromosome",
        "R176": "Disease_Mapped_To_Gene",
    }.items():
        axioms += [
            f"Declaration(ObjectProperty(<{NCIT}{code}>))",
            _annotation(LABEL, NCIT + code, label),
        ]
    axioms += [
        _annotation(
            NCIT + "P97",
            NCIT + "C116573",
            "A severe form of epilepsy that presents in early childhood and is characterized by "
            "frequent, prolonged febrile or myoclonic seizures that may progress to status "
            "epilepticus and poor development of language, motor, and socialization skills.",
        ),
        _annotation(NCIT + "P90", NCIT + "C116573", "Dravet Syndrome"),
        _annotation(NCIT + "P90", NCIT + "C116573", "SMEI"),
        _annotation(NCIT + "P90", NCIT + "C116573", "Severe Myoclonic Epilepsy of Infancy"),
        f"SubClassOf(<{NCIT}C116573> <{NCIT}C28193>)",
        f"SubClassOf(<{NCIT}C116573> <{NCIT}C3020>)",
        f"SubClassOf(<{NCIT}C116573> ObjectSomeValuesFrom(<{NCIT}R176> <{NCIT}C102447>))",
        _annotation(
            NCIT + "P97",
            NCIT + "C114940",
            "A primary or metastatic malignant neoplasm involving the cardiovascular system.",
        ),
        _annotation(NCIT + "P90", NCIT + "C114940", "Malignant Cardiovascular Neoplasm"),
        f"EquivalentClasses(<{NCIT}C114940> ObjectIntersectionOf(<{NCIT}C4784> <{NCIT}C9305> "
        f"ObjectSomeValuesFrom(<{NCIT}R100> <{NCIT}C12686>)))",
        _annotation(
            NCIT + "P97",
            NCIT + "C101223",
            "A syndrome characterized by the presence of three complete copies of genetic "
            "material for chromosome 13, instead of the normal two. It leads to a variety of "
            "abnormalities that include mental retardation, microcephaly, low-set ears, eye "
            "structural defects, polydactyly, and limb abnormalities.",
        ),
        _annotation(NCIT + "P90", NCIT + "C101223", "Complete Trisomy 13 Syndrome"),
        _annotation(NCIT + "P90", NCIT + "C101223", "Patau Syndrome"),
        f"SubClassOf(<{NCIT}C101223> <{NCIT}C28193>)",
        f"SubClassOf(<{NCIT}C101223> ObjectSomeValuesFrom(<{NCIT}R106> <{NCIT}C36529>))",
        f"SubClassOf(<{NCIT}C101223> ObjectSomeValuesFrom(<{NCIT}R174> <{NCIT}C13208>))",
    ]
    return "Ontology(<urn:exact-ui-fixture:ncit-excerpt>\n" + "\n".join(axioms) + "\n)\n"


def doid_excerpt() -> str:
    """Original DOID annotations/axioms, including a class without a definition."""
    axioms = []
    classes = {
        "DOID_0080422": "Dravet syndrome",
        "DOID_0112202": "developmental and epileptic encephalopathy",
        "DOID_0050736": "autosomal dominant disease",
        "DOID_176": "cardiovascular cancer",
        "DOID_0050686": "organ system cancer",
        "DOID_1287": "cardiovascular system disease",
        "DOID_162": "cancer",
        "DOID_11665": "Patau syndrome",
        "DOID_0080014": "chromosomal disease",
    }
    for code, label in classes.items():
        axioms += [f"Declaration(Class(<{OBO}{code}>))", _annotation(LABEL, OBO + code, label)]
    for iri in ("UBERON_0004535", "GENO_0000147"):
        axioms.append(f"Declaration(Class(<{OBO}{iri}>))")
    axioms += [
        f"Declaration(ObjectProperty(<{OBO}RO_0004026>))",
        f"Declaration(ObjectProperty(<{OBO}IDO_0000664>))",
        _annotation(LABEL, OBO + "IDO_0000664", "has material basis in"),
    ]
    definition = OBO + "IAO_0000115"
    axioms += [
        _annotation(
            definition,
            OBO + "DOID_0080422",
            "A developmental and epileptic encephalopathy characterized by onset of seizures "
            "that are usually refractory to treatment in the first year of life after normal "
            "early development and impaired psychomoter development starting around the second "
            "year of life that has_material_basis_in heterozygous mutation in the SCN1A gene on "
            "chromosome 2q24.",
            "en",
        ),
        *(
            _annotation(OIO + "hasExactSynonym", OBO + "DOID_0080422", text)
            for text in (
                "DEE6",
                "DEE6A",
                "developmental and epileptic encephalopathy 6",
                "developmental and epileptic encephalopathy 6A",
                "early infantile epileptic encephalopathy 6",
                "severe myoclonic epilepsy of infancy",
            )
        ),
        f"SubClassOf(<{OBO}DOID_0080422> <{OBO}DOID_0050736>)",
        f"SubClassOf(<{OBO}DOID_0080422> <{OBO}DOID_0112202>)",
        f"SubClassOf(<{OBO}DOID_0080422> ObjectSomeValuesFrom(<{OBO}IDO_0000664> <{OBO}GENO_0000147>))",
        _annotation(
            definition,
            OBO + "DOID_176",
            "An organ system cancer that located_in the heart and blood vessels.",
            "en",
        ),
        _annotation(OIO + "hasExactSynonym", OBO + "DOID_176", "Cardiovascular tumors"),
        _annotation(OIO + "hasExactSynonym", OBO + "DOID_176", "cardiovascular neoplasm"),
        f"SubClassOf(<{OBO}DOID_176> <{OBO}DOID_0050686>)",
        f"SubClassOf(<{OBO}DOID_176> <{OBO}DOID_1287>)",
        f"EquivalentClasses(<{OBO}DOID_176> ObjectIntersectionOf(<{OBO}DOID_162> "
        f"ObjectSomeValuesFrom(<{OBO}RO_0004026> <{OBO}UBERON_0004535>)))",
        _annotation(OIO + "hasExactSynonym", OBO + "DOID_11665", "D1 Trisomy"),
        _annotation(OIO + "hasExactSynonym", OBO + "DOID_11665", "trisomy 13"),
        _annotation(OIO + "hasRelatedSynonym", OBO + "DOID_11665", "Patau's syndrome"),
        f"SubClassOf(<{OBO}DOID_11665> <{OBO}DOID_0080014>)",
        f"SubClassOf(<{OBO}DOID_0050686> <{OBO}DOID_162>)",
    ]
    return "Ontology(<urn:exact-ui-fixture:doid-excerpt>\n" + "\n".join(axioms) + "\n)\n"


# Synthetic candidate pools: five targets per source, with invented ordering values.
POOLS = {
    NCIT
    + "C116573": [
        "DOID_0080422",
        "DOID_0112202",
        "DOID_0050736",
        "DOID_11665",
        "DOID_0080014",
    ],
    NCIT + "C114940": ["DOID_176", "DOID_0050686", "DOID_1287", "DOID_162", "DOID_0112202"],
    NCIT + "C101223": ["DOID_11665", "DOID_0080014", "DOID_0050736", "DOID_0080422", "DOID_162"],
}
SYNTHETIC_SCORES = (0.9, 0.8, 0.7, 0.6, 0.5)
PARENT_EVIDENCE = {
    (NCIT + "C116573", "DOID_0080422"): (NCIT + "C3020", "DOID_0112202"),
    (NCIT + "C114940", "DOID_176"): None,
    (NCIT + "C101223", "DOID_11665"): (NCIT + "C28193", "DOID_0080014"),
}


def build_run(run_dir: Path) -> None:
    """Write a synthetic Exact run through the real run-store writers."""
    import pandas as pd

    from exact.runs import ExplanationStore, RunLayout, refresh_manifest
    from exact.runs.decisions import (
        append_event,
        event,
        observe_selection,
        write_candidate_decisions,
    )

    store = ExplanationStore.create(run_dir)
    records, rows = [], []
    for source, targets in POOLS.items():
        for rank, (code, score) in enumerate(zip(targets, SYNTHETIC_SCORES), start=1):
            target = OBO + code
            record = {
                "src_iri": source,
                "tgt_iri": target,
                "src_kind": "class",
                "tgt_kind": "class",
                "confidences": {"S_final": score},
                "fixture_note": FIXTURE_NOTE,
            }
            parents = PARENT_EVIDENCE.get((source, code))
            if parents:
                record["triple_attributions"] = {
                    "hierarchy": {
                        "is_a": {
                            "source": [
                                {
                                    "subject_iri": source,
                                    "object_iri": parents[0],
                                    "rel_iri": SUBCLASS,
                                }
                            ],
                            "target": [
                                {
                                    "subject_iri": target,
                                    "object_iri": OBO + parents[1],
                                    "rel_iri": SUBCLASS,
                                }
                            ],
                        }
                    }
                }
            records.append(record)
            row = {
                "Src": source,
                "Tgt": target,
                "SrcKind": "class",
                "TgtKind": "class",
                "S_final": score,
                "candidate_joint_rank": rank,
                "selection_winner": rank == 1,
            }
            row["candidate_decision"] = append_event(
                row,
                event(
                    "retrieval",
                    implementation="synthetic-fixture",
                    reason="provided_candidate_pool",
                    values={
                        "cand_rank": rank,
                        "cand_ordering": "synthetic_fixture_order",
                        "cand_tie_rule": "not_applicable",
                        "cand_rank_provenance": "synthetic_fixture",
                    },
                ),
            )
            rows.append(row)
    store.append(records)
    frame = pd.DataFrame(rows)
    observe_selection(frame, implementation="synthetic-fixture", config={"fixture": FIXTURE_NOTE})
    winners = frame[frame["selection_winner"]]
    mappings = pd.DataFrame(
        [
            {
                "SrcEntity": row["Src"],
                "TgtEntity": row["Tgt"],
                "Score": row["S_final"],
                "Relation": "=",
                "SrcKind": "class",
                "TgtKind": "class",
            }
            for row in winners.to_dict("records")
        ]
    )
    layout = RunLayout.open(run_dir)
    mappings.to_csv(layout.mapping_path("global"), sep="\t", index=False)
    write_candidate_decisions(run_dir, frame, mappings, policy={"relation_prediction": "none"})
    refresh_manifest(layout, run_id="synthetic-ui-fixture")


def offline_provider(self, messages, profile):
    """Return exact excerpts or the backend's own comparison templates; nothing else validates."""
    packet = json.loads(messages[1]["content"])
    if packet.get("task") == "pair_comparison":
        claims = [
            {"text": claim["text"], "fact_ids": claim["fact_ids"], "category": claim["category"]}
            for claim in packet.get("supported_comparison_templates", [])
        ]
        return {
            "model": profile.model,
            "provider": "offline-fixture",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"claims": claims, "limitations": [], "relation": None}
                        )
                    }
                }
            ],
        }
    literals = [
        fact
        for fact in packet["facts"]
        if (fact.get("value") or {}).get("term_type") == "literal"
        and fact.get("category") in {"definitions", "synonyms", "labels"}
    ]
    claims = [
        {
            "text": fact["value"]["lexical_form"],
            "fact_ids": [fact["fact_id"]],
            "category": "meaning" if fact["category"] == "definitions" else "key_fact",
        }
        for fact in literals[:3]
    ]
    return {
        "model": profile.model,
        "provider": "offline-fixture",
        "choices": [
            {
                "message": {
                    "content": json.dumps({"claims": claims, "limitations": [], "relation": None})
                }
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Choose an empty or new output directory")
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "ncit-excerpt.ofn").write_text(ncit_excerpt())
    (inputs / "doid-excerpt.ofn").write_text(doid_excerpt())
    build_run(inputs / "run")

    from exact.runs import RunLayout
    from exact_inspect.artifacts import atomic_json
    from exact_inspect.contracts import canonical_hash
    from exact_inspect.generation import ExplanationJobs
    from exact_inspect.preparation import Preparation, bind_execution_lock

    entities, pairs = {}, []
    for source, targets in POOLS.items():
        for code in targets:
            pair = {}
            for side, name, iri in (("source", "NCIT", source), ("target", "DOID", OBO + code)):
                entry = {"ontology": name, "iri": iri, "kind": "class"}
                key = canonical_hash(entry)
                entities[key] = {"id": key, **entry}
                pair[side] = key
            pairs.append(pair)
    template = {
        "design_revision": "exact-explain-ui-fixture/1",
        "audience": "development_demo",
        "ontologies": [
            {"name": "NCIT", "root": {"path": "inputs/ncit-excerpt.ofn"}, "scope": "root"},
            {"name": "DOID", "root": {"path": "inputs/doid-excerpt.ofn"}, "scope": "root"},
        ],
        "run": {
            "binding": {"path": str(RunLayout.open(inputs / "run").manifest_path)},
            "source_ontology": "NCIT",
            "target_ontology": "DOID",
            "run_id": "synthetic-ui-fixture",
        },
        "entities": list(entities.values()),
        "pairs": pairs,
        "profile": {"name": "offline-exact-excerpts", "model": "fixture/exact-excerpts"},
    }
    lock = bind_execution_lock(template, input_root=root)
    atomic_json(root / "execution.json", lock)
    ExplanationJobs._provider = offline_provider
    report = Preparation(lock, root / "prepared", input_root=root).run()
    package = root / "prepared" / report["outputs"]["portable-export"] / "package.json"
    print(json.dumps({"package": str(package), "note": FIXTURE_NOTE}, indent=2))


if __name__ == "__main__":
    main()
