"""Corrective protocols freeze resolved controls without rewriting old evidence."""

import json
from pathlib import Path

import pytest

from exact.repair.protocol import (
    RepairProtocolV3,
    load_protocol_v3,
    training_projection_v3,
)

DIRECTORY = Path(__file__).resolve().parents[1] / "specs/exact-repair/protocol"
SAMPLE = DIRECTORY / "xr21-review-conformance.json"


def _load(tmp_path, mutate):
    value = json.loads(SAMPLE.read_text())
    mutate(value)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value))
    return load_protocol_v3(path)


def test_corrective_protocol_freezes_every_effective_identity_and_explicit_auxiliary():
    value = load_protocol_v3(SAMPLE)
    projection = training_projection_v3(value)
    assert value.identity.implementation_revision == "exact-repair/review-corrections/v1"
    assert value.model.preparation_schema == "exact-repair/effective-preparation/v3.1"
    assert value.model.support_admission_policy == "atomic-complete-support/v3.1"
    assert value.teacher.target_schema == "exact-repair/semantic-target/v3.1"
    assert value.teacher.aggregation == "weighted-family-means/v1"
    assert value.training.recovery_revision == "exact-phase-resume/v3.1"
    assert not value.model.support_enabled
    assert projection["graph"]["support_enabled"] is False
    assert projection["training"]["support_loss_weight"] == 0.2
    assert value.collection.plan_quotas == dict(
        utility=32, proposal=32, diversity=32, quartet=32, uniform=0
    )
    assert value.collection.quartet_budget_unit == "assignment_attempts"
    assert (
        json.loads((DIRECTORY / "schema-v3.2.json").read_text())
        == RepairProtocolV3.model_json_schema()
    )
    # The original manifest is still available as historical data, not reinterpreted.
    old = json.loads((DIRECTORY / "xr21-local-conformance.json").read_text())
    assert "implementation_revision" not in old["identity"]
    with pytest.raises(ValueError, match="implementation_revision"):
        load_protocol_v3(DIRECTORY / "xr21-local-conformance.json")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(exploration_fraction=0.5),
        lambda c: c.update(generator_fraction=0.5),
        lambda c: c.update(proposal_attempts=31),
        lambda c: c["plan_quotas"].update(quartet=31, uniform=1),
        lambda c: c["plan_quotas"].update(uniform=1),
        lambda c: c["plan_quotas"].pop("uniform"),
        lambda c: c["plan_quotas"].update(uniform=True),
    ],
)
def test_contradictory_fraction_quota_and_quartet_declarations_reject(tmp_path, mutation):
    with pytest.raises(ValueError):
        _load(tmp_path, lambda p: mutation(p["collection"]))


def test_nonunit_target_weights_and_nondefault_preparation_project_exactly(tmp_path):
    def mutate(value):
        value["teacher"]["family_weights"] = {"desired": 2.0, "unwanted": 0.3}
        value["model"].update(
            max_nodes=123, max_edges=234, max_supports=0, max_text_tokens=7, support_enabled=True
        )

    corrected = _load(tmp_path, mutate)
    projected = training_projection_v3(corrected)
    assert projected["teacher"]["desired_family_weight"] == 2.0
    assert projected["teacher"]["false_positive_weight"] == 0.3
    assert projected["graph"]["max_nodes"] == 123
    assert projected["graph"]["max_edges"] == 234
    assert projected["graph"]["max_explanations"] == 0
    assert projected["graph"]["max_text_tokens"] == 7
    assert projected["graph"]["support_enabled"] is True
    assert corrected.resolved_hash != load_protocol_v3(SAMPLE).resolved_hash


def test_progressive_language_cannot_shrink_while_preserving_old_pool(tmp_path):
    with pytest.raises(ValueError, match="nested language"):
        _load(tmp_path, lambda p: p["generation"]["stages"][1].update(classes_per_side=1))
