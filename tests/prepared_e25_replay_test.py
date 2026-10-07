"""A successful E25 replay cannot hide omitted sources or candidate rows."""

from copy import deepcopy

import pytest

from tools.prepared_batch import (
    _validate_e25_replay_world,
    validate_e25_replay_completion,
)


@pytest.fixture
def world():
    artifact = {
        "kind": "llm_oracle_replay",
        "outcome_semantics": "benchmark_reference",
        "teacher_binding": {
            "source_universe": ["judged", "unjudged", "protected", "empty"],
            "unscored_protected_pairs": [["protected", "p"]],
        },
        "population_rows": [
            {"Src": "judged", "Tgt": "a", "S_base": 0.8, "U": 0.2},
            {"Src": "judged", "Tgt": "b", "S_base": 0.6, "U": 0.3},
            {"Src": "unjudged", "Tgt": "u", "S_base": 0.4, "U": 0.7},
        ],
    }
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "source_universe": ["judged", "unjudged", "protected", "empty"],
        "records": [
            {
                "Src": source,
                "candidates": [
                    {"target": row["Tgt"], "S_base": row["S_base"], "U": row["U"]}
                    for row in artifact["population_rows"]
                    if row["Src"] == source
                ],
            }
            for source in artifact["teacher_binding"]["source_universe"]
        ],
    }
    trace["records"][2]["candidates"] = [
        {"target": "p", "S_base": None, "U": None, "protected_exact": True}
    ]
    return artifact, trace


def test_complete_world_preserves_empty_unjudged_and_protected_groups_order_independently(world):
    artifact, trace = world
    trace["source_universe"].reverse()
    trace["records"].reverse()
    for row in trace["records"]:
        row["candidates"].reverse()
    _validate_e25_replay_world(artifact, trace)


@pytest.mark.parametrize(
    "change", ["omit_source", "omit_empty", "duplicate_source", "extra_source"]
)
def test_changed_source_world_fails(world, change):
    artifact, trace = world
    if change == "omit_source":
        trace["source_universe"].remove("unjudged")
    elif change == "omit_empty":
        trace["records"].pop()
    elif change == "duplicate_source":
        trace["records"].append(deepcopy(trace["records"][0]))
    else:
        trace["source_universe"].append("extra")
        trace["records"].append({"Src": "extra", "candidates": []})
    with pytest.raises(ValueError, match="source"):
        _validate_e25_replay_world(artifact, trace)


@pytest.mark.parametrize(
    "change",
    ["omit_scored", "omit_unjudged", "omit_protected", "duplicate", "extra", "score", "protected"],
)
def test_changed_candidate_world_fails(world, change):
    artifact, trace = world
    if change == "omit_scored":
        trace["records"][0]["candidates"].pop()
    elif change == "omit_unjudged":
        trace["records"][1]["candidates"] = []
    elif change == "omit_protected":
        trace["records"][2]["candidates"] = []
    elif change == "duplicate":
        trace["records"][0]["candidates"].append(deepcopy(trace["records"][0]["candidates"][0]))
    elif change == "extra":
        trace["records"][0]["candidates"].append({"target": "extra"})
    elif change == "score":
        trace["records"][0]["candidates"][0]["S_base"] += 0.1
    else:
        trace["records"][2]["candidates"][0]["protected_exact"] = False
    with pytest.raises(ValueError, match="pair|score"):
        _validate_e25_replay_world(artifact, trace)


def test_non_e25_completion_keeps_existing_contract():
    validate_e25_replay_completion("E18", [{}])


def test_pre_extraction_trace_cannot_claim_completion(world):
    artifact, trace = world
    trace["stage"] = "before_extraction"
    with pytest.raises(ValueError, match="declared full source world"):
        _validate_e25_replay_world(artifact, trace)


def _report(tmp_path, artifact, trace, *, off=False):
    from exact.core.entities.configs.config import ConfigModel
    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments.harness import hash_payload
    from tools.prepared_batch import binding, write

    artifact_path = tmp_path / "artifact.json"
    write(artifact_path, artifact)
    gate = {"mode": "off"} if off else {"mode": "oracle_replay", "artifact": str(artifact_path)}
    config = ConfigModel.from_mapping(
        {"config_version": 2, "llm": {"experiment": {"enabled": True, "gate": gate}}}
    )
    inputs = tmp_path / "_inputs"
    inputs.mkdir()
    (inputs / "resolved.config.yaml").write_text(
        dump_yaml_document(config.model_dump(mode="json", by_alias=True))
    )
    write(tmp_path / "source_decisions.json", trace)
    return {
        "resolved_config_hash": hash_payload(config.model_dump(mode="json", by_alias=True)),
        "fingerprint_payload": {
            "output_dir": str(tmp_path),
            "artifacts": {"llm.experiment.gate.artifact": binding(artifact_path)},
        },
    }


def test_completion_uses_manifest_bound_gate_and_complete_saved_trace(tmp_path, world):
    artifact, trace = world
    report = _report(tmp_path, artifact, trace)
    validate_e25_replay_completion("E25-oracles", [report], require_artifact=True)
    (tmp_path / "artifact.json").write_text("{}")
    with pytest.raises(ValueError, match="binding changed"):
        validate_e25_replay_completion("E25-oracles", [report], require_artifact=True)


def test_completion_rejects_changed_resolved_configuration(tmp_path, world):
    report = _report(tmp_path, *world)
    report["resolved_config_hash"] = "changed"
    with pytest.raises(ValueError, match="configuration changed"):
        validate_e25_replay_completion("E25-trust", [report], require_artifact=True)


def test_legacy_off_allowed_but_amended_off_requires_a_population_artifact(tmp_path, world):
    report = _report(tmp_path, *world, off=True)
    validate_e25_replay_completion("E25-oracles", [report])
    with pytest.raises(ValueError, match="Every amended"):
        validate_e25_replay_completion("E25-oracles", [report], require_artifact=True)
