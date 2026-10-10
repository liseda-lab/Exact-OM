"""The HTTPS verifier binds matching metadata to exact typed pairs and row counts."""

from copy import deepcopy

import pytest

from exact_inspect.contracts import canonical_hash
from exact_inspect.evidence_metadata import evidence_completeness
from tools.verify_review_environment import Journey, VerificationError, validate_pair_metadata


def metadata_inputs():
    source = dict(ontology_version_id="source-version", iri="urn:source", kind="class")
    target = dict(ontology_version_id="target-version", iri="urn:target", kind="class")
    candidate = dict(candidate_id="candidate-one", entity=target)
    rows = [
        dict(
            evidence_id="selected-one",
            side="source",
            entity=source,
            origin_resolution="unresolved",
            graph_projection="not_recorded",
        )
    ]
    metadata = dict(
        status="selected",
        counts=evidence_completeness(rows),
        bridge_status="not_recorded",
        bridges=[],
    )
    study = dict(
        candidate_id="candidate-one", source=source, target=target, evidence_metadata=metadata
    )
    main = dict(
        run_id="saved-run",
        pair_id=canonical_hash(dict(run_id="saved-run", source=source, target=target)),
        source=source,
        target=target,
        evidence_metadata=deepcopy(metadata),
    )
    return study, main, source, candidate, "saved-run", rows


def test_identical_public_pair_metadata_preserves_counts_and_typed_identity():
    validate_pair_metadata(*metadata_inputs())


@pytest.mark.parametrize(
    "change,label",
    [
        (lambda v: v[0].update(candidate_id="foreign"), "metadata_candidate_binding"),
        (
            lambda v: v[0].update(target={**v[3]["entity"], "kind": "individual"}),
            "metadata_target_binding",
        ),
        (lambda v: v[1].update(pair_id="foreign"), "metadata_main_pair_binding"),
        (
            lambda v: v[1]["evidence_metadata"].update(status="not_exported"),
            "same_public_pair_metadata",
        ),
        (lambda v: v[5].append(deepcopy(v[5][0])), "unique_selected_evidence"),
        (lambda v: v[5][0].update(origin_resolution="resolved"), "metadata_selected_status_counts"),
        (
            lambda v: v[5][0].update(entity={**v[2], "iri": "urn:foreign"}),
            "selected_evidence_typed_subject",
        ),
    ],
)
def test_mismatched_public_pair_contracts_fail(change, label):
    values = metadata_inputs()
    change(values)
    with pytest.raises(VerificationError, match=label):
        validate_pair_metadata(*values)


def test_only_explicit_legacy_synthetic_fixture_may_have_no_pair_metadata():
    journey = Journey.__new__(Journey)
    journey.scope = "synthetic_contract_fixture"
    calls = []
    journey.request = lambda *a, **kw: calls.append((a, kw))
    case = {"candidates": [{"candidate_id": "candidate-one"}]}
    journey.pair_metadata("/workspace", case, {"evidence_metadata_contract": None})
    assert calls[0][1]["expected"] == 404
    assert calls[0][1]["params"] == {"candidate_id": "candidate-one"}
    journey.scope = "real_bundle"
    with pytest.raises(VerificationError, match="real_pair_metadata_available"):
        journey.pair_metadata("/workspace", case, {"evidence_metadata_contract": None})
    assert len(calls) == 1
