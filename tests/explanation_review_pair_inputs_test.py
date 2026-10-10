"""Study pair inputs copy only the exact public export, without decision fields."""

from types import SimpleNamespace

import pytest

from exact.runs import ExplanationStore
from exact_inspect.contracts import VisibilityPolicy, canonical_hash
from exact_inspect.decisions import DecisionStore, import_run
from exact_inspect.decisions_export import export_policy_run
from tools.review_pair_inputs import frozen_case_pair_inputs


def test_actual_export_pair_metadata_is_scoped_and_excludes_decision_fields(tmp_path):
    raw = tmp_path / "raw"
    ExplanationStore.create(raw).append(
        [
            {
                "src_iri": "urn:S",
                "tgt_iri": "urn:T",
                "src_kind": "class",
                "tgt_kind": "class",
                "confidences": {"S_final": 0.87},
            }
        ]
    )
    import_run(
        raw,
        tmp_path / "private",
        source_ontology_version_id="source",
        target_ontology_version_id="target",
    )
    private = DecisionStore(tmp_path / "private")
    public = export_policy_run(private, tmp_path / "public", contexts={}, policy=VisibilityPolicy())
    source = {"ontology_version_id": "source", "iri": "urn:S", "kind": "class"}
    target = {"ontology_version_id": "target", "iri": "urn:T", "kind": "class"}
    candidate = {"candidate_id": "candidate-one", "entity": target}
    evidence, metadata = frozen_case_pair_inputs(public, source, [candidate])
    assert evidence == {"candidate-one": []}
    envelope = metadata["candidate-one"]
    assert set(envelope) == {"source", "target", "evidence_metadata"}
    assert envelope["evidence_metadata"]["scope"] == "prepared_visibility_policy"
    pair_id = canonical_hash(
        {"run_id": public.manifest()["run_id"], "source": source, "target": target}
    )
    assert envelope["evidence_metadata"] == public.pair(pair_id)["evidence_metadata"]
    with pytest.raises(ValueError, match="policy-exported"):
        frozen_case_pair_inputs(private, source, [candidate])
    forged = SimpleNamespace(
        manifest=public.manifest, pair=lambda _: {**public.pair(pair_id), "target": source}
    )
    with pytest.raises(ValueError, match="typed pair"):
        frozen_case_pair_inputs(forged, source, [candidate])
