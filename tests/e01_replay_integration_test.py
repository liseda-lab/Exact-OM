"""Exercise replay artifacts through the campaign's ordinary result readers."""

import copy
import json
from types import SimpleNamespace

import pytest

from exact.core.entities.configs.yaml_io import dump_yaml_document
from exact.experiments import harness
from exact.experiments.extraction_replay import build_packet, run_replay
from tests.extraction_replay_test import replay


@pytest.mark.parametrize("mode,expected_f1", [("threshold", round(4 / 7, 3)), ("greedy", 1.0)])
def test_replay_artifacts_feed_standard_campaign_provenance_and_metrics(
    tmp_path, monkeypatch, mode, expected_f1
):
    config, source, _, _ = replay.__wrapped__(tmp_path)
    stats_path = source / "stats/run_stats.json"
    upstream = json.loads(stats_path.read_text())
    upstream.update(candidate_recall=0.9, candidate_recall_after_exact=1.0, mean_pool_size=1.0)
    stats_path.write_text(json.dumps(upstream))
    packet = tmp_path / "integration-packet.json"
    build_packet(
        source, packet, expected_sources=2, expected_scored_pairs=2, expected_protected_pairs=3
    )
    config = copy.deepcopy(config)
    config["matching"]["extraction"].update(mode=mode, anchor_conflict_policy="compete")
    if mode == "threshold":
        config["matching"].update(cardinality=None, target_cardinality=None)
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(dump_yaml_document(config))
    output = tmp_path / "output"
    wrapper = tmp_path / "worker.yaml"
    wrapper.write_text(
        dump_yaml_document({"job": {"config_file": str(config_path), "output_dir": str(output)}})
    )
    from exact.impl.trainer.runner import SemanticAlignmentRunner

    monkeypatch.setattr(
        SemanticAlignmentRunner,
        "__init__",
        lambda *a, **k: pytest.fail("replay constructed the scoring/model runner"),
    )
    run_replay(wrapper, packet)
    cell = SimpleNamespace(
        experiment_id="E01",
        output_dir=output,
        recovery=None,
        resolved_config=config,
        split_role="development",
        reference_role="valid",
        diagnostics=None,
    )
    provenance = harness._post_run_provenance(cell)
    assert provenance["candidate_pool_fingerprint"] == "pool"
    assert provenance["candidate_recall"] == 0.9
    assert provenance["candidate_recall_after_exact"] == 1.0
    assert provenance["explanation_reconstruction"]["failed_rows"] == 0
    assert provenance["observed_execution"]["mode"] == "extraction_only_replay"
    manifest = {
        "experiment_id": "E01",
        "stage": "screen",
        "arm_id": mode,
        "task_id": "D0-global_alignment",
        "seed": 17,
        "status": "complete",
        "execution_mode": "global_alignment",
        "fingerprint_payload": {"output_dir": str(output)},
        "wall_seconds": 0.1,
        **provenance,
    }
    harness._require_successful_cells([manifest], stage="screen")
    suite = SimpleNamespace(
        suite_id="fixture", campaign=None, confirmed_components_hash=None, sources=()
    )
    rows = harness.aggregate_stage(
        suite,
        stage="screen",
        output_root=tmp_path / "results",
        manifests=[manifest],
        finalize_reports=False,
    )
    assert len(rows) == 1
    assert rows[0]["metric_error"] is None
    assert harness._metric_value(rows[0]["metrics"], "F1") == pytest.approx(expected_f1)
    assert rows[0]["candidate_recall"] == 0.9
    assert rows[0]["wall_seconds"] == 0.1
