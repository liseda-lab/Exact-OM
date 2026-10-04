"""The queued E23 preparer binds inputs and emits an ordinary non-selecting batch group."""

import json
from copy import deepcopy

import pytest

from tools import prepare_openea_pools as tool


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    original = {
        "cases": {"K0": {"task": "unchanged"}, tool.CASE_ID: {"task": "openea"}},
        "steps": {},
        "e23_natural_case": tool.CASE_ID,
    }
    paths = {}
    for name in ("config", "campaign_bindings", "blueprint", "preparation"):
        path = tmp_path / ("preparation.json" if name == "preparation" else name + ".json")
        path.write_text(json.dumps(original if name == "campaign_bindings" else {}))
        paths[name] = tool._binding(path)
    metadata = {
        "recipe": {
            "dataset": "OpenEA_v2.0",
            "task": tool.TASK,
            "split": tool.SPLIT,
            "training_cap": 2000,
            "source_cap": 300,
            "seed": 17,
            "schema_version": 2,
        },
        "archive": {"sha256": tool.ARCHIVE_SHA256},
    }
    monkeypatch.setattr(tool, "verify_prepared", lambda _: metadata)
    calls = {"pools": 0, "campaign": 0}

    def pools(config, directory, output, *, device):
        calls["pools"] += 1
        assert device == "cpu" and config == tmp_path / "config.json"
        output.mkdir(parents=True, exist_ok=True)
        (output / "pools.json").write_text("{}\n")
        return {
            "cases": {tool.CASE_ID: {"task": "openea", "frozen_pool": "own"}},
            "e23_natural_case": tool.CASE_ID,
        }

    def campaign(blueprint, config, bindings, output):
        calls["campaign"] += 1
        assert bindings["cases"]["K0"] == original["cases"]["K0"]
        assert bindings["cases"][tool.CASE_ID]["frozen_pool"] == "own"
        output.mkdir(parents=True)
        path = output / "campaign.lock.yaml"
        path.write_text(
            json.dumps(
                {
                    "cases": bindings["cases"],
                    "steps": [
                        {
                            "id": "E23",
                            "case": tool.CASE_ID,
                            "arms": [
                                {"id": name}
                                for name in (
                                    "natural_graph_off",
                                    "natural_inductive",
                                    "graph_only",
                                    "graph_shuffled",
                                )
                            ],
                        }
                    ],
                }
            )
        )
        return path

    monkeypatch.setattr(tool, "prepare_pools", pools)
    monkeypatch.setattr(tool, "prepare_campaign", campaign)
    recipe = {
        "schema_version": 1,
        "kind": "e23_pool_preparation",
        "case": tool.CASE_ID,
        "source_cap": 300,
        "training_source_cap": 2000,
        "seed": 17,
        "selection_eligible": False,
        "hosted_calls": False,
        "rationales": False,
        "inputs": paths,
        "output": str(tmp_path / "pools"),
        "campaign_output": str(tmp_path / "campaign"),
        "device": "cpu",
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path, recipe, calls


def test_queued_preparer_publishes_verified_group_and_resumes_metadata(prepared):
    path, recipe, calls = prepared
    result = tool.run_diagnostic(path)
    group = json.loads(tool._verified(result["group"]).read_text())
    assert set(group["cases"]) == {tool.CASE_ID}
    assert group["rows"][0]["step"] == "E23"
    assert group["base_config"] == recipe["inputs"]["config"]
    complete = json.loads((path.parent / "pools/completion.json").read_text())
    assert complete["selection_eligible"] is False
    assert json.loads(tool._verified(complete["diagnostic"]).read_text()) == result
    assert tool.run_diagnostic(path) == result
    assert calls["campaign"] == 1
    (path.parent / "campaign/campaign.lock.yaml").write_text("tampered")
    with pytest.raises(ValueError, match="campaign identity changed"):
        tool.run_diagnostic(path)


@pytest.mark.parametrize("change", ["extra_input", "extra_key", "train_cap", "case"])
def test_unapproved_protocol_rejected_before_retrieval(prepared, change):
    path, recipe, calls = prepared
    modified = deepcopy(recipe)
    if change == "extra_input":
        modified["inputs"]["test_reference"] = modified["inputs"]["config"]
    elif change == "extra_key":
        modified["test_reference"] = "forbidden"
    elif change == "train_cap":
        modified["training_source_cap"] = 3000
    else:
        modified["case"] = "K0"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError):
        tool.run_diagnostic(path)
    assert calls == {"pools": 0, "campaign": 0}


def test_binding_mutation_before_or_during_retrieval_prevents_publication(prepared, monkeypatch):
    path, recipe, calls = prepared
    target = path.parent / "config.json"
    before = target.read_bytes()
    target.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        tool.run_diagnostic(path)
    assert calls["pools"] == 0
    target.write_bytes(before)
    original = tool.prepare_pools

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        target.write_text("changed during retrieval")
        return result

    monkeypatch.setattr(tool, "prepare_pools", changing)
    with pytest.raises(ValueError, match="input changed"):
        tool.run_diagnostic(path)
    assert calls["pools"] == 1 and calls["campaign"] == 0
    assert not (path.parent / "pools/completion.json").exists()
