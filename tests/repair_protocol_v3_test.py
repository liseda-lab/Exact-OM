"""New protocols are explicit and never silently reinterpret frozen XR-2 inputs."""

import json
from pathlib import Path

import pytest

from exact.repair.protocol import RepairProtocolV3, load_protocol_v3

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "specs/exact-repair/protocol/xr21-review-conformance.json"


def test_manifest_is_complete_static_and_historical_inputs_stay_separate():
    value = load_protocol_v3(SAMPLE)
    assert value.identity.research_revision == "XR-2.1"
    assert not value.llm_labels.enabled
    assert value.training.checkpoint_criterion == "generated_pool_verified_quality_effort"
    with pytest.raises(PermissionError):
        load_protocol_v3(SAMPLE, for_execution=True)
    with pytest.raises(ValueError, match="explicitly declare v3"):
        load_protocol_v3(ROOT / "specs/exact-repair/protocol/pilot.json")
    assert (
        json.loads((SAMPLE.parent / "schema-v3.2.json").read_text())
        == RepairProtocolV3.model_json_schema()
    )


@pytest.mark.parametrize(
    "mutation", ["unknown", "missing", "nonfinite", "cap", "budget", "mask", "confirmatory", "llm"]
)
def test_invalid_manifest_cannot_relax_safeguards(tmp_path, mutation):
    p = json.loads(SAMPLE.read_text())
    if mutation == "unknown":
        p["resources"]["unlimited"] = True
    elif mutation == "missing":
        del p["resources"]["startup_seconds"]
    elif mutation == "nonfinite":
        p["resources"]["case_wall_seconds"] = float("inf")
    elif mutation == "cap":
        p["generation"]["stages"][-1]["candidate_cap"] = 64
    elif mutation == "budget":
        p["circuit"]["aggregate_seconds"] = 500.0
    elif mutation == "mask":
        p["losses"]["masks"] = "renormalize_known"
    elif mutation == "confirmatory":
        p["identity"]["purpose"] = "confirmatory"
    else:
        p["llm_labels"]["enabled"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_protocol_v3(path)


def test_duplicate_keys_cycle_and_inheritance_hashes(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema":"exact-repair/protocol/v3","schema":"x"}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_protocol_v3(bad)
    bad.write_text(json.dumps({"schema": "exact-repair/protocol/v3", "extends": "bad.json"}))
    with pytest.raises(ValueError, match="Cyclic"):
        load_protocol_v3(bad)
    bad.write_text(
        json.dumps(
            {
                "schema": "exact-repair/protocol/v3",
                "extends": str(SAMPLE),
                "selection": {"shortlist_size": 1},
            }
        )
    )
    loaded = load_protocol_v3(bad)
    assert loaded.selection.shortlist_size == 1
    assert loaded.resolved_hash != load_protocol_v3(SAMPLE).resolved_hash
    bad.write_text(json.dumps({"extends": str(SAMPLE)}))
    with pytest.raises(ValueError, match="explicitly declare v3"):
        load_protocol_v3(bad)


def test_training_projection_preserves_v3_and_resolves_every_optimizer_argument():
    from exact.repair.protocol import training_projection_v3
    from tools.repair.prepare import generated_from_protocol
    from tools.repair.train import _protocol_arguments, _protocol_profile

    protocol = load_protocol_v3(SAMPLE)
    projected = training_projection_v3(protocol)
    args = _protocol_arguments(projected)
    assert args["revision"] == "v3"
    assert args["risk_loss_weight"] == protocol.losses.risk_loss_weight
    assert projected["resolved_v3_protocol_hash"] == protocol.resolved_hash
    assert _protocol_profile(projected) == tuple(sorted(protocol.objective.edit_weights.items()))
    cases = generated_from_protocol(projected)
    assert cases and all(c.problem.schema_version.endswith("/v3") for c in cases)
    memberships = {}
    for case in cases:
        memberships.setdefault(case.structural_parent, set()).add(case.split)
    assert all(len(splits) == 1 for splits in memberships.values())


def test_protocol_cannot_claim_different_feature_or_label_schema(tmp_path):
    for key, value in (
        ("feature_schema", "old-score-features/v2"),
        ("label_schema", "exact-repair/semantic-fidelity-comparison/v3"),
    ):
        raw = json.loads(SAMPLE.read_text())
        raw["identity"][key] = value
        path = tmp_path / "wrong-schema.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            load_protocol_v3(path)


def test_authorization_cannot_make_unfrozen_inputs_executable(tmp_path):
    raw = json.loads(SAMPLE.read_text())
    raw["identity"].update(execution_authorized=True, code_hash="a" * 40, dirty_hash="b" * 64)
    path = tmp_path / "incomplete-freeze.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(PermissionError, match="Freeze code/input"):
        load_protocol_v3(path, for_execution=True)
