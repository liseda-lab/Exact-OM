"""Scientific independence, failure-inclusive profiling and compatible recovery."""

import dataclasses
import json

import pytest

from exact.repair.workers import CallResult
from tools.repair import expanded_profile as profile
from tools.repair.corpus import _case, coherent_control
from tools.repair.prepare import _with_object_count, case_to_dict


@pytest.mark.parametrize("family", profile.MECHANISMS_SELECTED)
def test_names_mirroring_and_disconnected_distractors_do_not_make_parents(family):
    original = profile.parent_case(family, 1, 13)
    fingerprint = profile.parent_fingerprints(original)
    assert fingerprint == profile.parent_fingerprints(profile.parent_case(family, 1, 73))
    mirror = _case(family + ":path-1", family, "test", 1, 1, 13)
    assert fingerprint == profile.parent_fingerprints(mirror)
    assert fingerprint == profile.parent_fingerprints(_with_object_count(original, [4]))
    assert set(fingerprint) & set(profile.parent_fingerprints(coherent_control(original)))
    assert fingerprint != profile.parent_fingerprints(profile.parent_case(family, 2, 13))


def test_aliases_and_transitive_exposure_cannot_cross_splits():
    counts = dict(development=1, fresh_evaluation=1, profile=1, test=1, train=1)
    candidates = [
        dict(key=str(i), family="papers", depth=i, fingerprints=[str(i)]) for i in range(1, 9)
    ]
    # An alias bridges 1 to exposed 2, even though 1 itself is not in history.
    candidates.append(dict(key="alias", family="papers", depth=9, fingerprints=["1", "2"]))
    selected, missing = profile.select_parents(
        candidates, [dict(key="old", fingerprints=["2"])], counts=counts, seed=17
    )
    assert not {"1", "2", "alias"} & {row["key"] for row in selected}
    assert next(row for row in selected if row["split"] == "profile")["key"] == "3"
    assert len({row["group_id"] for row in selected}) == 5
    assert len(missing) == 7 * 5


def test_actual_generator_aliases_share_observed_structure():
    aliases = ("directional_strengthening", "subclass_specialisation_target", "endpoint_confusion")
    fingerprints = [
        set(profile.parent_fingerprints(profile.parent_case(f, 2, 17))) for f in aliases
    ]
    assert set.intersection(*fingerprints)


def test_checkpoint_rejects_corruption_and_changed_dependencies(tmp_path):
    path = tmp_path / "row.json"
    profile.checkpoint(path, "identity", status="timeout")
    assert profile.checked_checkpoint(path, "identity")["status"] == "timeout"
    with pytest.raises(ValueError, match="dependencies"):
        profile.checked_checkpoint(path, "changed")
    value = json.loads(path.read_text())
    value["status"] = "complete"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="content"):
        profile.checked_checkpoint(path, "identity")


def settings():
    return dict(
        retrieval=dict(classes_per_side=2, properties_per_side=1, endpoints_per_side=2),
        max_depth=1,
        max_constructors=1,
        circuit=dict(
            aggregate_seconds=10.0,
            call_seconds=5.0,
            rss_mb=2048,
            allocated_node_limit=100000,
            live_node_limit=100000,
            reachable_node_limit=100000,
            element_limit=200000,
        ),
    )


def test_profile_retains_failed_rows_and_resumes_without_test_access(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                seed=13,
                parent_counts={"profile": 1},
                profile_settings=settings(),
                per_case_seconds=30,
                per_case_memory_mb=2048,
            )
        )
    )
    inventory = dict(
        selected=[
            dict(key="papers:path-1", family="papers", depth=1, split="profile"),
            dict(key="papers:path-2", family="papers", depth=2, split="test"),
        ],
        missing=[],
    )

    def prepare(manifest, output, identity):
        profile.write_artifact(output / "inventory.json", inventory)
        return inventory

    monkeypatch.setattr(profile, "prepare_inventory", prepare)
    calls = []

    def invoke(function, record, config, cache, **kwargs):
        assert record["case"]["split"] == "development"
        assert "path-1" in record["case"]["case_id"]
        calls.append(cache)
        return CallResult("timeout", detail="bounded probe")

    monkeypatch.setattr("exact.repair.workers.bounded_call", invoke)
    report = profile.run(manifest, tmp_path / "output")
    assert report["recorded"] == 4
    assert all(row["call_status"] == "timeout" for row in report["rows"])
    assert not report["corpus_finalized"] and not report["test_outcomes_opened"]
    assert calls[0] == calls[1] and calls[2] == calls[3]
    profile.run(manifest, tmp_path / "output")
    assert len(calls) == 4


@pytest.mark.parametrize("family", ["papers", "domain"])
def test_live_native_profile_serializes_all_template_outcomes(tmp_path, family):
    record = case_to_dict(profile.parent_case(family, 1, 13))
    result = profile.native_profile(record, settings(), tmp_path)
    assert result["status"] in {"complete", "partial"}
    assert result["objects"] and all(row["families"] for row in result["objects"])
    json.dumps(result)
    test_case = dataclasses.replace(profile.parent_case(family, 1, 13), split="test")
    with pytest.raises(ValueError, match="held-out"):
        profile.native_profile(case_to_dict(test_case), settings(), tmp_path)


def test_native_profile_crosses_bounded_transport(tmp_path):
    from exact.repair.workers import bounded_call

    result = bounded_call(
        profile.native_profile,
        case_to_dict(profile.parent_case("papers", 1, 13)),
        settings(),
        str(tmp_path),
        timeout=30,
        memory_mb=4096,
    )
    assert result.status == "complete", result.detail
    assert result.cleanup_complete
    assert result.value["objects"][0]["families"]
    assert dict(result.resource_usage)["cpu_seconds"] >= 0
    json.dumps(result.value)
