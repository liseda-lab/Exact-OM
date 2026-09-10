import json

import pytest
import torch

from exact.impl.models.selector.llm_gate import fit_forced_sample_artifact
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset


def test_forced_sample_keeps_whole_source_groups_and_replays_any_batch(tmp_path):
    rows = [
        {"source_iri": f"s{i}", "target_iri": f"t{j}", "U": 0.7 if i % 2 else 0.1}
        for i in range(4)
        for j in range(7)
    ]
    state = fit_forced_sample_artifact(
        rows,
        sample_size=2,
        seed=17,
        task_id="development",
        entity_kind="class",
        dataset_signature="tiny-signature",
    )
    assert state["sample_size"] == 2 and len(state["pairs"]) == 14
    assert state == fit_forced_sample_artifact(
        list(reversed(rows)),
        sample_size=2,
        seed=17,
        task_id="development",
        entity_kind="class",
        dataset_signature="tiny-signature",
    )
    assert len({state["source_strata"][source] for source in state["selected_sources"]}) == 2
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(state))
    scorer = _scorer(
        llm_experiment_config={
            "enabled": True,
            "gate": {"mode": "forced_sample", "forced_sample_size": 2, "artifact": str(path)},
        }
    )
    scorer.attach_dataset(_TinyDataset())
    sources = [state["selected_sources"][0]] * 7
    targets = [f"t{i}" for i in range(7)]
    mask, diagnostics = scorer._llm_gate_mask(
        U_ind=torch.ones(7),
        U_dis=torch.ones(7),
        U=torch.ones(7),
        S_base=torch.zeros(7),
        q_label=torch.ones(7),
        Q_struct=torch.ones(7),
        src_iris=sources,
        tgt_iris=targets,
        label=None,
    )
    assert mask.tolist() == [True] * 7 and diagnostics[0]["sample_unit"] == "source"
    assert diagnostics[0]["sample_size"] == 2
    bad = json.loads(json.dumps(state))
    bad["pairs"].pop()
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="every candidate"):
        _scorer(
            llm_experiment_config={
                "enabled": True,
                "gate": {"mode": "forced_sample", "forced_sample_size": 2, "artifact": str(path)},
            }
        )


def test_forced_limit_and_labeled_strata_require_explicit_provenance():
    rows = [{"source_iri": "s", "target_iri": "t", "U": 0.5}]
    with pytest.raises(ValueError, match="200 source"):
        fit_forced_sample_artifact(
            rows, sample_size=201, seed=17, task_id="dev", entity_kind="class"
        )
    with pytest.raises(ValueError, match="provenance"):
        fit_forced_sample_artifact(
            rows,
            sample_size=1,
            seed=17,
            task_id="dev",
            entity_kind="class",
            source_strata={"s": "confident_error"},
        )
    state = fit_forced_sample_artifact(
        rows,
        sample_size=200,
        seed=17,
        task_id="dev",
        entity_kind="class",
        source_strata={"s": "confident_error"},
        strata_provenance={
            "label_scope": "permitted_development",
            "negative_label_policy": "complete_reference",
        },
    )
    assert state["sample_size"] == 1 and state["requested_sample_size"] == 200
