"""Audit repair preserves scoring while relocating only identical replay policies."""

import copy

import pytest

from exact.impl.trainer.checkpointing import CheckpointingMixin
from tools.prepared_batch import binding
from tools.recover_decision_audit import relocate


def checkpoint(tmp_path, *, additional):
    original, successor = tmp_path / "old.json", tmp_path / "new.json"
    original.write_text('{"kind":"llm_oracle_replay","rows":[1,2]}')
    successor.write_bytes(original.read_bytes())
    digest = CheckpointingMixin._hash_checkpoint_fingerprint_payload
    model = {
        "class": "PairAdaptiveSemanticScorer",
        "payload": {
            "pair_adaptive_channels": {
                "experiments": {
                    "llm": {"gate": {"mode": "oracle_replay", "artifact": str(original)}},
                    "gate_artifact": {**binding(original), "bytes": original.stat().st_size},
                }
            }
        },
    }
    model["fingerprint"] = digest(model["payload"])
    inner = {"models": [model], "dataset_signature": "unchanged"}
    key = "fingerprint_payload" if additional else "checkpoint_fingerprint_payload"
    value = {key: inner, "processed_examples": 5842, "numerical_rows": [0.75, 0.9]}
    if additional:
        value["candidate_records_path"] = (
            "inference_additional_models_" + digest(inner)[:12] + ".jsonl.zst"
        )
    else:
        value["checkpoint_fingerprint"] = digest(inner)
    return value, original, successor


@pytest.mark.parametrize("additional", [False, True])
@pytest.mark.parametrize("mode", ["oracle_replay", "oracle_perfect"])
def test_gate_relocation_retains_values_and_original(additional, mode, tmp_path):
    value, old, new = checkpoint(tmp_path, additional=additional)
    key = "fingerprint_payload" if additional else "checkpoint_fingerprint_payload"
    model = value[key]["models"][0]
    model["payload"]["pair_adaptive_channels"]["experiments"]["llm"]["gate"]["mode"] = mode
    digest = CheckpointingMixin._hash_checkpoint_fingerprint_payload
    model["fingerprint"] = digest(model["payload"])
    if additional:
        value["candidate_records_path"] = (
            "inference_additional_models_" + digest(value[key])[:12] + ".jsonl.zst"
        )
    else:
        value["checkpoint_fingerprint"] = digest(value[key])
    before = copy.deepcopy(value)
    result = relocate(value, new, additional=additional)
    assert value == before
    assert result["numerical_rows"] == value["numerical_rows"]
    assert result["processed_examples"] == 5842
    assert relocate(result, old, additional=additional) == value


@pytest.mark.parametrize("additional", [False, True])
def test_gate_relocation_rejects_changed_policy(additional, tmp_path):
    value, _, new = checkpoint(tmp_path, additional=additional)
    new.write_text('{"kind":"llm_oracle_replay","rows":[3]}')
    with pytest.raises(ValueError, match="differs numerically"):
        relocate(value, new, additional=additional)


def test_gate_relocation_rejects_corrupt_original_fingerprint(tmp_path):
    value, _, new = checkpoint(tmp_path, additional=False)
    value["checkpoint_fingerprint_payload"]["models"][0]["fingerprint"] = "bad"
    with pytest.raises(ValueError, match="fingerprint changed"):
        relocate(value, new)


@pytest.mark.parametrize(
    "experiment, arms, accepted",
    [
        ("E25-oracles", ["decision_off", "oracle_observed", "oracle_perfect"], True),
        ("E25-trust", ["trust_constant", "trust_shipped"], True),
        ("E25-trust", ["trust_shipped"], False),
        ("E25-trust", ["trust_shipped", "trust_constant", "source_first"], False),
        ("E25-trust", ["trust_shipped", "trust_constant", "trust_constant"], False),
        ("E25-oracles", ["decision_off", "oracle_observed"], False),
        ("E25-other", ["trust_shipped", "trust_constant"], False),
    ],
)
def test_repair_requires_full_comparison_and_producer(
    monkeypatch, tmp_path, experiment, arms, accepted
):
    from dataclasses import dataclass
    from types import SimpleNamespace as NS

    from exact.experiments import campaign, harness
    from tools.recover_decision_audit import cells_for_repair

    producer = NS(id="E25-forced", external_selection={"path": "bound"})
    config = NS(
        experiment_id=experiment,
        depends_on=[],
        frozen_constants={"campaign_v2": {"requires": ["E25-forced"]}},
    )
    source = NS(config=config)

    @dataclass
    class Suite:
        sources: list
        campaign: dict | None = None

    suite = Suite(sources=[source, NS(config=NS(experiment_id="E25-forced"))])
    monkeypatch.setattr(campaign, "load_campaign", lambda _: (NS(steps=[producer]), None))
    monkeypatch.setattr(campaign, "materialize_campaign", lambda *a, **k: suite)

    def external(*a, producer_manifests, **k):
        producer_manifests.append({"experiment_id": "E25-forced", "status": "complete"})
        return {}

    monkeypatch.setattr(campaign, "external_selection_result", external)
    monkeypatch.setattr(harness, "_bind_external_selection", lambda *a: {})
    monkeypatch.setattr(harness, "inherited_selection_overlay", lambda *a: {})

    def materialize(*args, external_manifests, **kwargs):
        assert args[1].campaign["root"] == str(tmp_path / "runtime")
        assert external_manifests == [{"experiment_id": "E25-forced", "status": "complete"}]
        return source, {}

    monkeypatch.setattr(harness, "_materialize_campaign_evidence", materialize)
    cells = [NS(arm_id=a, task_id="D0_E03-global_alignment", seed=17, source_cap=300) for a in arms]
    monkeypatch.setattr(harness, "build_cells", lambda *a, **k: cells)
    arguments = ({"scientific_step": experiment}, tmp_path / "campaign", tmp_path / "runtime")
    if accepted:
        assert cells_for_repair(*arguments)[1] == cells
    else:
        with pytest.raises(ValueError, match="complete original E25 comparison"):
            cells_for_repair(*arguments)
