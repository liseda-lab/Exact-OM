import json
from pathlib import Path

import pytest

from exact.experiments.oracle_policy import build_oracle_artifacts
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset


def forced_trace(tmp_path):
    sources = [source for source in ("good", "harm", "nil") for _ in range(2)]
    targets = ["a", "t"] * 3
    inputs = {
        "src_iris": sources,
        "tgt_iris": targets,
        "src_label_lists": [[source] for source in sources],
        "tgt_label_lists": [[target] for target in targets],
    }
    model = _scorer()
    model.attach_dataset(_TinyDataset())
    base = model(**inputs)
    rows = []
    for offset, source in enumerate(("good", "harm", "nil")):
        probabilities = {"a": 0.1, "t": 0.1 if source == "nil" else 0.9}
        judgment = {
            "source": source,
            "valid": True,
            "pair_probabilities": probabilities,
            "choice": "__NONE__" if source == "nil" else "t",
        }
        candidates = [
            {
                "target": target,
                "S_base": float(base["S_base"][offset * 2 + index]),
                "U": float(base["U"][offset * 2 + index]),
                "p_llm": probabilities[target],
                "llm_gate_invoked": True,
                "llm_grouped_decision": json.dumps(judgment),
            }
            for index, target in enumerate(("a", "t"))
        ]
        rows.append({"Src": source, "candidates": candidates})
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "dataset_signature": "tiny-signature",
        "source_universe": ["good", "harm", "nil"],
        "records": rows,
        "policy": {
            "threshold": 0.5,
            "llm": {
                "beta": 0.8,
                "pair_threshold": 0.5,
                "gate": {"mode": "forced_sample"},
                "forced_sample": {
                    "selected_sources": ["good", "harm", "nil"],
                    "population_fingerprint": "frozen-fixture",
                },
            },
        },
    }
    path = tmp_path / "source_decisions.json"
    path.write_text(json.dumps(trace))
    return path, inputs


def test_cached_producer_freezes_trust_sources_but_budget_selects_observed_oracle(
    tmp_path, monkeypatch
):
    trace, inputs = forced_trace(tmp_path)
    refs = [{"SrcEntity": "good", "TgtEntity": "t"}, {"SrcEntity": "harm", "TgtEntity": "a"}]
    result = build_oracle_artifacts(
        trace,
        refs,
        tmp_path / "policies",
        reference_role="development",
        negative_label_policy="complete_reference",
        budget=2,
        nil_sources=["nil"],
    )
    assert set(result["artifacts"]) == {
        "trust_shipped",
        "trust_constant",
        "trust_source",
        "oracle_observed",
        "oracle_perfect",
    }
    states = {
        name: json.loads(Path(path).read_text()) for name, path in result["artifacts"].items()
    }
    for name in ("trust_shipped", "trust_constant", "trust_source"):
        assert states[name]["selected_sources"] == ["good", "harm", "nil"]
    assert states["oracle_observed"]["selected_sources"] == ["good", "nil"]
    assert states["oracle_perfect"]["selected_sources"] == ["good", "nil"]
    assert (
        states["oracle_observed"]["decision_probs"].keys()
        == states["oracle_perfect"]["decision_probs"].keys()
    )
    for name, state in states.items():
        scorer = _scorer(
            llm_experiment_config={
                "enabled": True,
                "gate": {"mode": state["mode"], "artifact": result["artifacts"][name]},
                "fusion_weight": state["fixed_fusion"]["fusion_weight"],
                "constant_weight": 0.5,
                "decision": {"mode": "listwise" if name == "trust_source" else "binary"},
            }
        )
        scorer.attach_dataset(_TinyDataset())
        scorer.use_llm = True

        def forbidden(*args, **kwargs):
            raise AssertionError("Cached policy attempted a hosted call")

        for method in (
            "llm_grouped_decision_probs",
            "llm_yesno_probs_batched",
            "generate_pair_briefs_batched",
        ):
            monkeypatch.setattr(scorer, method, forbidden)
        output = scorer(**inputs)
        assert output["oracle_diagnostic"]["llm_invocations"] == 0
        if name == "trust_constant":
            assert output["S_final"].tolist() == pytest.approx([0.3, 0.7, 0.3, 0.7, 0.3, 0.3])
    assert result == build_oracle_artifacts(
        trace,
        refs,
        tmp_path / "policies",
        reference_role="development",
        negative_label_policy="complete_reference",
        budget=2,
        nil_sources=["nil"],
    )


def test_unknown_outcomes_allow_fixed_trust_but_never_outcome_selected_oracle(tmp_path):
    trace, _ = forced_trace(tmp_path)
    result = build_oracle_artifacts(
        trace,
        [],
        tmp_path / "policies",
        reference_role="development",
        negative_label_policy="unknown",
    )
    assert set(result["artifacts"]) == {"trust_shipped", "trust_constant", "trust_source"}
    assert set(result["unavailable"]) == {"oracle_observed", "oracle_perfect"}
    with pytest.raises(ValueError, match="development/diagnostic"):
        build_oracle_artifacts(
            trace, [], tmp_path, reference_role="test", negative_label_policy="unknown"
        )


def test_binary_cache_does_not_invent_a_comparative_choice(tmp_path):
    path, _ = forced_trace(tmp_path)
    trace = json.loads(path.read_text())
    for source in trace["records"]:
        for candidate in source["candidates"]:
            candidate.pop("llm_grouped_decision")
    path.write_text(json.dumps(trace))
    result = build_oracle_artifacts(
        path,
        [],
        tmp_path / "policies",
        reference_role="diagnostic",
        negative_label_policy="unknown",
    )
    assert set(result["artifacts"]) == {"trust_shipped", "trust_constant"}
    assert "comparative choices" in result["unavailable"]["trust_source"]


def test_unscored_protected_exact_pairs_stay_outside_replay_population(tmp_path):
    path, _ = forced_trace(tmp_path)
    trace = json.loads(path.read_text())
    trace["source_universe"].append("exact")
    trace["records"].append(
        {
            "Src": "exact",
            "candidates": [
                {"target": "exact_target", "protected_exact": True, "S_final": 1.0, "emitted": True}
            ],
        }
    )
    path.write_text(json.dumps(trace))
    result = build_oracle_artifacts(
        path,
        [],
        tmp_path / "policies",
        reference_role="diagnostic",
        negative_label_policy="unknown",
    )
    assert result["teacher_binding"]["unscored_protected_pairs"] == [["exact", "exact_target"]]
    artifact = json.loads(Path(result["artifacts"]["trust_constant"]).read_text())
    assert len(artifact["population_rows"]) == 6
    assert all(row["Src"] != "exact" for row in artifact["population_rows"])


def test_followup_binds_completed_recipe_and_refuses_population_or_role_changes(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from exact.core.entities.configs.config import ConfigModel
    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments import harness
    from exact.experiments.oracle_policy import materialize_followup
    from tests.label_policy_test import declaration

    trace, _ = forced_trace(tmp_path)
    source_path, target_path, universe = (
        tmp_path / name for name in ("source.owl", "target.owl", "sources.txt")
    )
    source_path.write_text("synthetic source")
    target_path.write_text("synthetic target")
    universe.write_text("good\nharm\nnil\n")
    reference = tmp_path / "valid.tsv"
    reference.write_text("SrcEntity\tTgtEntity\ngood\tt\nharm\ta\n")
    overlay = {
        "data": {
            "source": str(source_path),
            "target": str(target_path),
            "source_universe": str(universe),
            "refs": {"valid": str(reference)},
            "reference_role": "valid",
        }
    }
    producer = ConfigModel.from_mapping(
        harness.deep_merge(ConfigModel().model_dump(mode="json", by_alias=True), overlay)
    ).model_dump(mode="json", by_alias=True)
    output = tmp_path / "completed"
    (output / "_inputs").mkdir(parents=True)
    config_path = output / "_inputs/resolved.config.yaml"
    config_path.write_text(dump_yaml_document(producer))
    (output / "source_decisions.json").write_bytes(trace.read_bytes())
    arms = [
        {"id": "trust_shipped", "role": "baseline"},
        {"id": "trust_constant", "role": "candidate"},
        {"id": "trust_source", "role": "candidate"},
    ]
    source = declaration(tmp_path, "E25-trust", arms, overlay)
    suite = harness.LoadedSuite("fixture", "R_0", (source,), None, "suite", None, None, {})
    item = {
        "experiment_id": "E25-forced",
        "stage": "screen",
        "arm_id": "forced_sources",
        "task_id": "development",
        "seed": 17,
        "source_cap": source.config.screen.source_cap,
        "status": "complete",
        "reference_completeness": "known_incomplete",
        "resolved_config_hash": harness.hash_payload(producer),
        "fingerprint_payload": {"output_dir": str(output)},
    }
    bound = materialize_followup(source, suite, [item], {})
    assert bound.base_config_path == config_path
    assert bound.config.screen.source_cap == source.config.screen.source_cap
    assert all(
        arm.overlay["llm"]["experiment"]["gate"]["mode"] == "oracle_replay"
        for arm in bound.config.arms
    )
    metadata = bound.config.frozen_constants["resolved_oracle_policy"]
    assert (
        "oracle_perfect" in metadata["unavailable"]
    )  # training semantics never license report negatives
    assert materialize_followup(source, suite, [item], {}).raw_hash() == bound.raw_hash()
    changed = tmp_path / "changed.sources.txt"
    changed.write_text("good\n")
    task = source.config.screen.tasks[0].model_copy(
        update={"overlay": harness.deep_merge(overlay, {"data": {"source_universe": str(changed)}})}
    )
    modified = replace(
        source,
        config=source.config.model_copy(
            update={"screen": source.config.screen.model_copy(update={"tasks": [task]})}
        ),
    )
    with pytest.raises(ValueError, match="frozen source_universe"):
        materialize_followup(modified, suite, [item], {})
    # A consumer may relocate identical reference bytes, but cannot change the
    # effective role or reference labels before the adapter opens either table.
    relocated = tmp_path / "relocated-valid.tsv"
    relocated.write_bytes(reference.read_bytes())

    def with_reference(path, role="valid"):
        task = source.config.screen.tasks[0].model_copy(
            update={
                "reference_role": role,
                "overlay": harness.deep_merge(overlay, {"data": {"refs": {role: str(path)}}}),
            }
        )
        return replace(
            source,
            config=source.config.model_copy(
                update={"screen": source.config.screen.model_copy(update={"tasks": [task]})}
            ),
        )

    replay = materialize_followup(with_reference(relocated), suite, [item], {})
    assert (
        replay.config.frozen_constants["resolved_oracle_policy"]["manifest_sha256"]
        == metadata["manifest_sha256"]
    )
    relocated.write_text("SrcEntity\tTgtEntity\ngood\ta\n")
    with monkeypatch.context() as guard:
        guard.setattr(
            "exact.utils.data.read_table",
            lambda *args: pytest.fail("opened changed reference labels"),
        )
        with pytest.raises(ValueError, match="reference bytes"):
            materialize_followup(with_reference(relocated), suite, [item], {})
        with pytest.raises(ValueError, match="effective reference role"):
            materialize_followup(with_reference(reference, "development"), suite, [item], {})
    producer["data"]["reference_role"] = "test"
    config_path.write_text(dump_yaml_document(producer))
    item["resolved_config_hash"] = harness.hash_payload(producer)
    monkeypatch.setattr(
        "exact.utils.data.read_table", lambda *args: pytest.fail("read final labels")
    )
    with pytest.raises(ValueError, match="development reference role"):
        materialize_followup(source, suite, [item], {})


@pytest.mark.parametrize("field", ["records", "source_universe"])
def test_cached_policy_rejects_duplicate_source_identity_before_mapping(tmp_path, field):
    path, _ = forced_trace(tmp_path)
    trace = json.loads(path.read_text())
    trace[field].append(trace[field][0])
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="duplicate source identities"):
        build_oracle_artifacts(
            path,
            [],
            tmp_path / "policies",
            reference_role="development",
            negative_label_policy="unknown",
        )


def test_label_repairs_change_cached_policy_identity_without_reusing_old_artifacts(tmp_path):
    path, _ = forced_trace(tmp_path)
    results = []
    for refs, negatives, nil in [
        ([], [], []),
        ([{"Src": "good", "Tgt": "t"}], [], []),
        ([], [("good", "a")], []),
        ([], [], ["nil"]),
    ]:
        results.append(
            build_oracle_artifacts(
                path,
                refs,
                tmp_path / "policies",
                reference_role="development",
                negative_label_policy="unknown",
                confirmed_negatives=negatives,
                nil_sources=nil,
            )
        )
    assert len({result["identity"] for result in results}) == 4
    assert len({result["artifacts"]["trust_shipped"] for result in results}) == 4
