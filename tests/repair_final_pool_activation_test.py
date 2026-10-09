"""Final omission distinguishes unchanged state from keep-tagged activation aliases."""

import pyowl_core as owl
import pytest

from exact.repair.candidates import make_candidate
from exact.repair.records import PolicyV2, RepairInputV2, RevisionObjectV2, promote_input_v3


@pytest.mark.parametrize("proposal", ["grammar_uniform", "bounded_enumeration"])
@pytest.mark.parametrize("target", ["activation", "changed_axioms", "unchanged"])
def test_removal_uses_full_bundle_identity(tmp_path, proposal, target):
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.retrieval import retrieve_vocabulary

    a, b, c = [owl.Class(owl.IRI("urn:removal:" + n)) for n in "ABC"]
    axiom = owl.SubClassOf(a, b)
    keep = make_candidate("o", (axiom,), ("keep",))
    activated = make_candidate("o", (axiom,), ("keep",), active_expressions=(a,))
    changed = make_candidate("o", (owl.SubClassOf(a, c),), ("keep",))
    obj = RevisionObjectV2(
        "o", "ontology_axiom", (axiom,), (keep, activated, changed), occurrence_id="fixture"
    )
    problem = promote_input_v3(RepairInputV2((), (obj,), PolicyV2((a,))))
    graph = EffectivePreparation(4096, 32768, 64, 128, revision="v3").graph(
        problem, retrieve_vocabulary(problem)
    )
    model = RepairModel(
        graph.metadata,
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0,
        revision="v3",
        plan_risk=False,
    )
    options = dict(
        proposal_arm=proposal,
        draws_per_object=8,
        max_depth=1,
        max_constructors=1,
        candidate_cap=32,
        contextual_filtering=False,
        compiler_cache_directory=str(tmp_path),
    )
    victim = dict(activation=activated, changed_axioms=changed, unchanged=keep)[target]
    before = problem.to_dict()
    if target == "unchanged":
        with pytest.raises(ValueError, match="cannot remove the unchanged mandatory state"):
            freeze_neural_round(
                problem, model, final_candidate_removals={"o": [victim.candidate_id]}, **options
            )
    else:
        baseline = freeze_neural_round(problem, model, **options)
        assert victim.candidate_id in {
            c.candidate_id for c in baseline.problem.objects[0].candidates
        }
        result = freeze_neural_round(
            problem, model, final_candidate_removals={"o": [victim.candidate_id]}, **options
        )
        ids = {c.candidate_id for c in result.problem.objects[0].candidates}
        assert victim.candidate_id not in ids and keep.candidate_id in ids
        assert baseline.graph_hash == result.graph_hash
    assert problem.to_dict() == before
