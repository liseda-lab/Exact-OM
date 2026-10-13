"""A bounded fixture can report numerical drift without admitting corrupt evidence."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from exact.runs.store import ExplanationStore, _atomic_json
from tools import benchmark_scoring_throughput as benchmark
from tools import validate_batched_scoring as validation


def panel():
    return {
        "namespace": "qualification_only",
        "queries": [
            {"qid": f"q{i}", "source": f"s{i}", "candidates": [f"t{j}" for j in range(20)]}
            for i in range(250)
        ],
    }


def test_fixed_bound_requires_complete_distinct_five_thousand_pairs():
    assert validation.validate_panel(panel()) == {"pairs": 5000, "source_groups": 250, "chunks": 25}
    for alteration in ("missing", "extra", "duplicate", "repeat_source", "repeat_query"):
        value = panel()
        if alteration == "missing":
            value["queries"][0]["candidates"].pop()
        elif alteration == "extra":
            value["queries"][0]["candidates"].append("extra")
        elif alteration == "duplicate":
            value["queries"][0]["candidates"][0] = "t1"
        elif alteration == "repeat_source":
            value["queries"][0]["source"] = "s1"
        else:
            value["queries"][0]["qid"] = "q1"
        with pytest.raises(ValueError, match="5,000"):
            validation.validate_panel(value)


def test_matmul_tokenizer_and_arithmetic_are_fixed_across_variants():
    flags = [validation.variant_flags(name) for name in validation.VARIANTS]
    for field, expected in (("MATRIX_KERNEL", "mm"), ("ARITHMETIC", "device")):
        assert {row["EXACT_PAIR_CONTEXT_" + field] for row in flags} == {expected}
    assert {row["EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY"] for row in flags} == {"0"}
    assert {key for key in flags[1] if flags[1][key] != flags[2][key]} == {
        "EXACT_ENCODER_CUDA_GRAPHS"
    }


def evidence_fixture():
    native = {
        "item_id": "item",
        "prop": "definition",
        "prop_iri": "p",
        "text": "definition: fact",
        "value": "fact",
        "entity_iri": "s",
        "source_axiom_refs": ["axiom"],
        "axiom_origins": [{"axiom_id": "axiom", "original_syntax": "original"}],
        "derivation": {"premises": ["axiom"]},
    }
    selected = {
        **native,
        "property": native["prop"],
        "property_iri": native["prop_iri"],
        "support": 0.7,
    }
    selected.pop("prop")
    selected.pop("prop_iri")
    record = {
        "src_iri": "s",
        "tgt_iri": "t",
        "confidences": {"S_final": 0.8},
        "attributes": {"source": [selected], "target": []},
        "triple_attributions": {
            "hierarchy": {},
            "similarity": {"source": [], "target": []},
            "difference": {"source": [], "target": []},
        },
        "cross_side_provenance": {
            "lexical": [{"source_ref": "__source__", "target_ref": "__target__"}],
            "hierarchy": {},
            "similarity": [],
            "difference": {"source": [], "target": []},
            "attributes": {
                "source": [{"item_id": "item", "anchor_ref": "__target__"}],
                "target": [],
            },
        },
    }
    dataset = SimpleNamespace(
        get_entity_features=lambda iri, side: {
            "attributes": [native] if side == "src" else [],
            "object_triples": [],
            "hierarchy": {},
        }
    )
    scorer = SimpleNamespace(_with_item_id=lambda channel, side, item, **kw: dict(item))
    return dataset, scorer, record


def test_evidence_scores_can_drift_but_every_fact_and_citation_stays_native():
    dataset, scorer, record = evidence_fixture()
    check = validation.EvidenceValidator()
    check(dataset, scorer, [record])
    changed = copy.deepcopy(record)
    changed["confidences"]["S_final"] = 0.2
    changed["attributes"]["source"][0]["support"] = 0.1
    check(dataset, scorer, [changed])
    assert check.counts == {"records": 2, "selected_items": 2, "citations": 2}


@pytest.mark.parametrize(
    "fault", ["text", "citation", "wrong_entity", "duplicate", "anchor", "nan"]
)
def test_corrupted_evidence_remains_a_hard_failure(fault):
    dataset, scorer, record = evidence_fixture()
    item = record["attributes"]["source"][0]
    if fault == "text":
        item["text"] = "invented fact retaining the valid item_id"
    elif fault == "citation":
        item["axiom_origins"][0]["original_syntax"] = "changed"
        # The source and selected fixtures initially share nested data; separate the source.
        dataset, scorer, _ = evidence_fixture()
    elif fault == "wrong_entity":
        item["item_id"] = "foreign entity item"
    elif fault == "duplicate":
        record["attributes"]["source"].append(copy.deepcopy(item))
    elif fault == "anchor":
        record["cross_side_provenance"]["attributes"]["source"][0]["anchor_ref"] = "foreign"
    else:
        item["support"] = float("nan")
    with pytest.raises(ValueError):
        validation.EvidenceValidator()(dataset, scorer, [record])


def saved_outputs(tmp_path):
    workload = tmp_path / "workload.json"
    _atomic_json(workload, {"queries": [{"qid": "q", "source": "s", "candidates": ["a", "b"]}]})
    manifest = tmp_path / "fitted.json"
    _atomic_json(manifest, {"workload": benchmark.binding(workload)})
    roots = [tmp_path / "left", tmp_path / "right"]
    for root in roots:
        _atomic_json(
            root / "measurement.json",
            {
                "namespace": "qualification_only",
                "paid_calls": 0,
                "fixture_outputs_promotable": False,
                "fitted_manifest": benchmark.binding(manifest),
            },
        )
        for mode in validation.MODES:
            records = [
                {
                    "src_iri": "s",
                    "tgt_iri": target,
                    "original_query_id": "q",
                    "confidences": {"S_final": 0.6},
                    "context_triples": ["native fact"],
                }
                for target in ("a", "b")
            ]
            if root == roots[1]:
                records[1]["confidences"]["S_final"] = 0.60000001
                records[1]["context_triples"] = ["another native fact"]
            ExplanationStore(root / mode / "ordinary-evidence").append(records)
            _atomic_json(
                root / (mode + ".json"),
                {"chunks": [{"query_ids": ["q"], "rows": 2, "exact_prefiltered_rows": 0}]},
            )
            _atomic_json(
                root / mode / "fitted-decisions.json",
                {
                    "namespace": "qualification_only",
                    "selector_scope": "bounded_cohort",
                    "hosted_quality_established": False,
                    "rows": [
                        {
                            "Src": "s",
                            "Tgt": row["tgt_iri"],
                            "S_final": row["confidences"]["S_final"],
                        }
                        for row in records
                    ],
                    "policy": {
                        "threshold": 0.7,
                        "source_cardinality": 1,
                        "target_cardinality": 1,
                        "extraction": {},
                    },
                    "protected_pairs": [],
                    "mappings": [["s", "a" if root == roots[0] else "b"]],
                },
            )
            _atomic_json(root / mode / "selector-evidence.json", records)
            _atomic_json(root / mode / "hosted-fixtures.json", [])
    return roots


def test_drift_reports_changed_winner_fact_and_mapping_without_equivalence_claim(tmp_path):
    roots = saved_outputs(tmp_path)
    drift = validation.drift(*roots, "cold")
    assert drift["ordinary"]["status"] == "failed"  # Strict comparator remains strict.
    assert drift["fitted"]["status"] == "failed"
    assert drift["mappings"]["added"] == [("s", "b")]
    assert drift["mappings"]["removed"] == [("s", "a")]
    assert "no scientific equivalence" in drift["interpretation"]
    assert validation.cache_check(roots[0])["status"] == "passed"


def test_exact_replay_detects_numeric_change_below_legacy_tolerance(tmp_path):
    root, _ = saved_outputs(tmp_path)
    path = root / "replay/fitted-decisions.json"
    result = json.loads(path.read_text())
    result["rows"][0]["S_final"] += 1e-9
    _atomic_json(path, result)
    assert validation.cache_check(root)["status"] == "failed"


def test_candidate_coverage_still_fails_in_drift_mode(tmp_path):
    left, right = saved_outputs(tmp_path)
    path = right / "cold/fitted-decisions.json"
    result = json.loads(path.read_text())
    result["rows"].pop()
    _atomic_json(path, result)
    with pytest.raises(ValueError, match="complete ordinary cohort"):
        validation.drift(left, right, "cold")


def test_valid_short_prefix_cannot_count_as_complete_declared_validation(tmp_path):
    root, _ = saved_outputs(tmp_path)
    with pytest.raises(ValueError, match="declared candidate population"):
        validation.validate_saved_mode(root, "cold")
    assert (
        validation.validate_saved_mode(root, "cold", expected_pairs=2, expected_chunks=1)["status"]
        == "passed"
    )


def test_nonfinite_nested_fitted_evidence_fails_self_validation(tmp_path):
    root, _ = saved_outputs(tmp_path)
    path = root / "cold/selector-evidence.json"
    value = json.loads(path.read_text())
    value[0]["nested"] = {"value": float("nan")}
    _atomic_json(path, value)
    with pytest.raises(ValueError, match="Invalid saved fitted outputs"):
        validation.validate_saved_mode(root, "cold", expected_pairs=2, expected_chunks=1)


def test_shared_native_state_cannot_masquerade_as_cold_measurement():
    with pytest.raises(ValueError, match="warm-public-feature"):
        benchmark.run(SimpleNamespace(pre_materialize_features=False), shared={})


@pytest.mark.parametrize("key", ["src_iri", "tgt_iri", "original_query_id"])
def test_pair_stamping_cannot_conceal_swapped_identity(key):
    row = {key: "wrong"}
    with pytest.raises(ValueError, match="conflicting"):
        benchmark.stamp_pair_identity(row, "source", "target", "query")
    assert row == {key: "wrong"}
    row = {}
    benchmark.stamp_pair_identity(row, "source", "target", "query")
    assert row[key] != "wrong"


def test_observer_counts_logical_cpu_tokens_without_model_hooks():
    import torch

    counts = {"calls": 0, "rows": 0, "tokens": 0, "batch_sizes": Counter(), "counter_seconds": 0.0}
    tokens = {"input_ids": torch.ones(2, 2), "attention_mask": torch.tensor([[1, 1], [1, 0]])}
    benchmark.observe_encoder(counts, object(), tokens)
    assert counts["calls"] == 1 and counts["rows"] == 2 and counts["tokens"] == 3
    assert counts["batch_sizes"] == {2: 1}


def test_public_features_materialized_once_per_entity_with_no_reference_access():
    calls = []
    dataset = SimpleNamespace(get_entity_features=lambda iri, side: calls.append((iri, side)))
    groups = [[{"source": "s", "candidates": ["a", "b"]}, {"source": "x", "candidates": ["a"]}]]
    result = benchmark.materialize_features(dataset, groups)
    assert calls == [("s", "src"), ("a", "tgt"), ("b", "tgt"), ("x", "src")]
    assert result["entities"] == 4


def test_preparation_binding_changes_with_scientific_input(tmp_path):
    args = SimpleNamespace(device="cpu", diagnostics="natural")
    for name in ("config", "workload", "templates", "fitted_manifest"):
        path = tmp_path / name
        path.write_text("first")
        setattr(args, name, path)
    before = benchmark.preparation_identity(args)
    args.config.write_text("changed")
    assert benchmark.preparation_identity(args) != before


def test_shared_preparation_time_is_counted_without_claiming_cold_gate():
    result = validation.timing_summary(
        {
            "cold": {
                "phases": {"seconds": {"structural_validation": 0.1}},
                "chunks": [{"elapsed_seconds": 2.0, "rows": 4900, "exact_prefiltered_rows": 100}],
            },
            "shared_native_preparation_seconds": 10.0,
            "shared_public_feature_preparation": {"seconds": 3.0},
            "model_loading_seconds": 1.0,
        }
    )
    assert result["model_scored_pairs"] == 4900
    assert result["candidate_occurrences"] == 5000
    assert result["warm_feature_cold_encoder_pairs_per_second"] == 2450
    assert result["standalone_preparation_inclusive_seconds"] == 16
    assert result["historical_cold_200_gate_established"] is False


def test_failure_receipt_is_durable_and_duplicate_run_preserves_it(tmp_path):
    workload = tmp_path / "bad.json"
    workload.write_text(json.dumps({"queries": []}))
    args = SimpleNamespace(output=tmp_path / "validation", workload=workload)
    with pytest.raises(ValueError):
        validation.run(args)
    path = args.output / "results.json"
    saved = path.read_bytes()
    assert json.loads(saved)["status"] == "failed"
    with pytest.raises(ValueError, match="never overwrite"):
        validation.run(args)
    assert path.read_bytes() == saved
