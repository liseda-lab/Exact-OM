"""E24 scientific contract and executable producer, without network/model downloads."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from exact.experiments.directional_diagnostics import score_directions, summarize
from exact.experiments.inputs import nested_sources
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from exact.utils.provenance import sha256_path
from tests.pair_adaptive_experiments_test import _scorer
from tools.run_directional_diagnostic import (
    build_components,
    load_pairs,
    prepare_recipe,
    run_diagnostic,
    write,
)


def fact(relation):
    return {"triple": ("entity", relation, "neighbor"), "score": 1.0}


def scorer(matrix):
    model = _scorer(diff={"enabled": True, "formulation": "normalised"})
    model.use_context = True
    calls = []

    def support(source, target):
        calls.append((source, target))
        return matrix

    model._object_support_matrix = support
    return model, calls


def test_both_hypotheses_share_support_and_reversal_is_real():
    model, calls = scorer(torch.tensor([[1.0], [0.0]]))
    original = dict(model.diff_config)
    result = score_directions(model, [fact("r"), fact("other")], [fact("r")])
    assert len(calls) == 1
    assert result["scores"]["forward"] == {"<": 1.0, ">": 0.5}
    assert result["scores"]["reversed"] == {"<": 0.5, ">": 1.0}
    assert result["decision"] == "<"
    assert result["reversed_decision"] == ">"
    assert result["reversal_pass"]
    assert model.diff_config == original


@pytest.mark.parametrize(
    "matrix,source,target,status",
    [
        (torch.tensor([[1.0]]), [fact("r")], [fact("r")], "tie"),
        (torch.zeros((1, 0)), [fact("r")], [], "unsupported"),
        (torch.zeros((1, 1)), [fact("r")], [fact("other")], "unsupported"),
    ],
)
def test_ties_missing_and_unsupported_abstain(matrix, source, target, status):
    model, _ = scorer(matrix)
    result = score_directions(model, source, target)
    assert result["status"] == status
    assert result["decision"] is None
    assert result["reversal_pass"]


def test_invalid_support_and_hosted_use_fail_closed():
    model, _ = scorer(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="finite"):
        score_directions(model, [fact("r")], [fact("r")])
    model.use_llm = True
    with pytest.raises(ValueError, match="disabled"):
        score_directions(model, [], [])


def test_abstentions_count_against_balanced_accuracy_and_control_is_matched():
    model, _ = scorer(torch.tensor([[1.0], [0.0]]))
    less = score_directions(model, [fact("r"), fact("s")], [fact("r")])
    model, _ = scorer(torch.tensor([[1.0, 0.0]]))
    greater = score_directions(model, [fact("r")], [fact("r"), fact("s")])
    model, _ = scorer(torch.zeros((1, 0)))
    absent = score_directions(model, [fact("r")], [])
    rows = [
        {**less, "relation": "<"},
        {**greater, "relation": ">"},
        {**absent, "relation": ">"},
        {**less, "relation": "="},
    ]
    result = summarize(rows)
    assert result["directional"]["balanced_accuracy_including_abstentions"] == 0.75
    assert result["directional"]["balanced_accuracy_covered"] == 1.0
    assert result["directional"]["coverage"] == pytest.approx(2 / 3)
    control = result["matched_fixed_less_control"]
    assert control["coverage"] == result["directional"]["coverage"]
    assert control["balanced_accuracy_covered"] == 0.5
    assert result["equality"]["mean_absolute_direction_gap_supported"] == 0.5
    assert result["reversal"]["failures"] == 0
    # Relabeling changes evaluation only; the previously computed scores stay fixed.
    relabeled = summarize([{**row, "relation": "="} for row in rows])
    assert relabeled["directional"]["balanced_accuracy_covered"] is None
    assert relabeled["equality"]["pairs"] == 4


def fixture_recipe(tmp_path):
    def binding(path):
        return {"path": str(path), "sha256": sha256_path(path)}

    inputs = {}
    for name, prefix in (("source", "urn:s"), ("target", "urn:t")):
        path = tmp_path / name
        path.mkdir()
        (path / "entities.csv").write_text(
            "entity,kind\n" + "".join(f"{prefix}{i},class\n" for i in range(4))
        )
        (path / "triples.csv").write_text(
            f"src,rel,dst\n{prefix}0,part_of,{prefix}2\n{prefix}1,part_of,{prefix}3\n"
        )
        (path / "kg.yaml").write_text("triples_files: [triples.csv]\nentities_file: entities.csv\n")
        inputs[name] = binding(path)
    for role, index in (("train", 0), ("valid", 1)):
        path = tmp_path / f"{role}.tsv"
        path.write_text(f"SrcEntity\tTgtEntity\tRelation\nurn:s{index}\turn:t{index}\t<\n")
        inputs[role] = binding(path)
    config = tmp_path / "config.yaml"
    from exact.core.entities.configs.config import ConfigModel

    payload = ConfigModel().model_dump(mode="json")
    payload["pipeline"][0]["params"].update(
        lexical_model_revision="a" * 40, context_model_revision="b" * 40
    )
    config.write_text(yaml.safe_dump(payload))
    case = {
        "role": "development",
        "kind": "class",
        "capabilities": ["typed_reference", "csv_graph"],
        "source": inputs["source"],
        "target": inputs["target"],
        "references": {role: inputs[role] for role in ("train", "valid")},
    }
    bindings = tmp_path / "bindings.yaml"
    bindings.write_text(yaml.safe_dump({"cases": {"T0": case}}))
    recipe = prepare_recipe(bindings, config, tmp_path / "result", device="cpu")
    path = tmp_path / "recipe.json"
    write(path, recipe)
    return path, recipe


def test_executable_native_dataset_producer_resume_and_corrupt_cache(tmp_path):
    path, recipe = fixture_recipe(tmp_path)
    initialized = []

    def factory(config, inputs, output, device):
        dataset = PairAdaptiveContextDataset(
            output_path=output,
            input_format="csv-kg",
            verbaliser_name=None,
            verbalization_mode="deterministic",
            device=device,
        )
        dataset.load_ontologies(inputs["source"], inputs["target"])
        assert dataset._reference is None
        model, _ = scorer(torch.tensor([[1.0]]))
        # Use real extracted evidence shape; encoder calls alone are stubbed.
        model._object_support_matrix = lambda left, right: torch.ones((len(left), len(right)))
        initialized.append(dataset)
        return dataset, model

    first = run_diagnostic(path, component_factory=factory)
    assert first["computed_pairs"] == 2
    assert first["reused_pairs"] == 0
    assert len(initialized) == 1
    assert first["splits"]["valid"]["rows"][0]["source_facts"] > 0
    assert first["selection_eligible"] is False
    assert (tmp_path / "result" / "completion.json").is_file()
    second = run_diagnostic(
        path,
        component_factory=lambda *args: pytest.fail("cache should avoid encoder initialization"),
    )
    assert second["computed_pairs"] == 0
    assert second["reused_pairs"] == 2
    assert second["splits"] == first["splits"]
    with sqlite3.connect(tmp_path / "result" / "components.sqlite") as connection:
        connection.execute("UPDATE components SET sha256='corrupt'")
    with pytest.raises(ValueError, match="checksum"):
        run_diagnostic(path, component_factory=factory)


def test_recipe_rejects_final_inputs_changed_inputs_and_protocol_changes(tmp_path):
    path, recipe = fixture_recipe(tmp_path)
    recipe["inputs"]["test"] = recipe["inputs"]["valid"]
    write(path, recipe)
    with pytest.raises(ValueError, match="only source/target"):
        run_diagnostic(path)
    del recipe["inputs"]["test"]
    recipe["source_cap"] = 301
    write(path, recipe)
    with pytest.raises(ValueError, match="fixed diagnostic protocol"):
        run_diagnostic(path)
    recipe["source_cap"] = 300
    Path(recipe["inputs"]["valid"]["path"]).write_text("changed")
    write(path, recipe)
    with pytest.raises(ValueError, match="input changed"):
        run_diagnostic(path)


def test_stable_source_group_cap_never_uses_relation_labels(tmp_path):
    path = tmp_path / "pairs.tsv"
    sources = [f"s{index}" for index in range(305)]
    frame = pd.DataFrame({"SrcEntity": sources, "TgtEntity": ["t"] * 305, "Relation": ["<"] * 305})
    frame.to_csv(path, sep="\t", index=False)
    first, selected = load_pairs(path, 300, 17)
    assert set(selected) == set(nested_sources(sources, 300, seed=17))
    frame["Relation"] = ">"
    frame.sample(frac=1, random_state=9).to_csv(path, sep="\t", index=False)
    second, again = load_pairs(path, 300, 17)
    assert selected == again
    assert list(first.SrcEntity) == list(second.SrcEntity)


def test_production_factory_forces_no_hosted_calls_or_rationales(tmp_path, monkeypatch):
    from exact.core.entities.configs.config import ConfigModel
    from exact.impl.models import pair_adaptive_scorer

    path, recipe = fixture_recipe(tmp_path)
    config = ConfigModel().model_dump(mode="json")
    config_path = tmp_path / "production.yaml"
    config_path.write_text(yaml.safe_dump(config))
    captured = {}

    class Model:
        def __init__(self, **params):
            captured.update(params)

        def attach_dataset(self, dataset):
            assert dataset._reference is None

        def eval(self):
            pass

    monkeypatch.setattr(pair_adaptive_scorer, "PairAdaptiveSemanticScorer", Model)
    dataset, _ = build_components(
        config_path,
        {name: tmp_path / name for name in ("source", "target")},
        tmp_path / "factory",
        "cpu",
    )
    assert dataset.verbalization_mode == "deterministic"
    assert captured["use_llm"] is False
    assert captured["generate_llm_rationales"] is False
    assert captured["use_llm_calibration"] is False


def test_interrupted_producer_commits_each_pair_before_resuming(tmp_path):
    path, recipe = fixture_recipe(tmp_path)
    attempts = []

    def factory(config, inputs, output, device):
        model, _ = scorer(torch.tensor([[1.0]]))

        def features(entity, side):
            if entity == "urn:s1" and len(attempts) == 1:
                raise RuntimeError("interrupted between pairs")
            return {"object_triples": [fact("r")]}

        attempts.append(True)
        from types import SimpleNamespace

        return SimpleNamespace(get_entity_features=features), model

    with pytest.raises(RuntimeError, match="interrupted"):
        run_diagnostic(path, component_factory=factory)
    assert not (tmp_path / "result" / "completion.json").exists()
    resumed = run_diagnostic(path, component_factory=factory)
    assert resumed["reused_pairs"] == 1
    assert resumed["computed_pairs"] == 1


def test_unpinned_encoder_fails_before_scientific_execution(tmp_path):
    from tools.run_directional_diagnostic import encoder_bindings

    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    with pytest.raises(ValueError, match="frozen 40-hex"):
        encoder_bindings(config)
