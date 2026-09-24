"""Portable evidence redaction must protect bytes as well as HTTP responses."""

import json

from exact.runs import ExplanationStore
from exact_inspect.context import prepare_context
from exact_inspect.context_export import export_policy_context
from exact_inspect.contracts import VisibilityPolicy
from exact_inspect.decisions import DecisionStore, import_run
from exact_inspect.decisions_export import export_policy_run


def test_portable_run_removes_unverified_display_and_preserves_decisions(tmp_path):
    ontology = tmp_path / "source.ofn"
    ontology.write_text(
        'Ontology(Declaration(Class(<urn:s>)) Declaration(Class(<urn:t>)) AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:s> "Source"))'
    )
    context = prepare_context(ontology, tmp_path / "raw-context")
    policy = VisibilityPolicy()
    visible = export_policy_context(context, tmp_path / "visible-context", policy)
    entity = {"ontology_version_id": context.ontology_version_id, "iri": "urn:s", "kind": "class"}
    fact_id = context.facts(entity, category="labels")["items"][0]["fact_id"]
    run = tmp_path / "run"
    ExplanationStore.create(run).append(
        [
            {
                "src_iri": "urn:s",
                "tgt_iri": "urn:t",
                "src_kind": "class",
                "tgt_kind": "class",
                "confidences": {"S_final": 0.91},
                "attributes": {
                    "source": [
                        {"text": "forbidden-target", "item_id": "legacy-only"},
                        {
                            "property_iri": "http://www.w3.org/2000/01/rdf-schema#label",
                            "value": "Source",
                            "text": "untrusted-rendering",
                            "support": 0.73,
                        },
                    ],
                    "target": [],
                },
            }
        ]
    )
    import_run(
        run,
        tmp_path / "raw-run",
        source_ontology_version_id=context.ontology_version_id,
        target_ontology_version_id=context.ontology_version_id,
        evidence_resolver=lambda item, entity: {"fact_ids": [fact_id]},
    )
    source = DecisionStore(tmp_path / "raw-run")
    pair_id = source.candidates("urn:s")["items"][0]["pair_id"]
    assert "forbidden-target" in json.dumps(source.evidence(pair_id))
    exported = export_policy_run(
        source,
        tmp_path / "portable-run",
        contexts={context.ontology_version_id: visible},
        policy=policy,
    )
    content = (tmp_path / "portable-run" / "decisions.sqlite").read_bytes()
    assert b"forbidden-target" not in content
    assert b"untrusted-rendering" not in content
    assert exported.pair(pair_id) == source.pair(pair_id)
    assert len(exported.evidence(pair_id)) == 1
    assert exported.evidence(pair_id)[0]["values"]["support"] == 0.73
    assert exported.evidence(pair_id)[0]["fact_ids"] == [fact_id]
