"""Recovery exercised through harness control flow and the real scorer cache seam."""

import json
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from exact.experiments import harness, runtime
from exact.experiments.harness import LoadedSuite, RunCell
from exact.experiments.schema import ResourceConfig
from exact.impl.models.scorer_common import PoolingMethod, ScorerCommonMixin
from exact.utils.provenance import file_provenance


def fixture(tmp_path, monkeypatch):
    reference = tmp_path / "reference.tsv"
    reference.write_text("SrcEntity\tTgtEntity\nsource:1\ttarget:1\nsource:2\ttarget:2\n")
    source = tmp_path / "source.owl"
    target = tmp_path / "target.owl"
    source.write_text("ontology source")
    target.write_text("ontology target")
    suite = LoadedSuite("suite", "baseline", (), None, "suite", None, None, {})
    root = tmp_path / "campaign"
    cell = RunCell(
        "suite",
        "E00",
        "screen",
        "production",
        "baseline",
        "fixture",
        "development",
        "dev",
        "complete",
        7,
        None,
        ResourceConfig(),
        root / "run",
        {
            "data": {
                "root": str(tmp_path),
                "source": str(source),
                "target": str(target),
                "refs": {"full": str(reference)},
                "execution_mode": "global_alignment",
            },
            "evaluation": {"backends": ["builtin"]},
        },
        "config",
        "experiment",
        "design",
        None,
        "target_label_free",
        {},
        "not_applicable",
        recovery={"root": str(root)},
    )
    revision = {"code": "prediction-v1", "evaluation": "evaluation-v1", "docs": "docs-v1"}

    def provenance(current, *_args, **_kwargs):
        payload = {
            "inputs": {
                "source": file_provenance(source),
                "target": file_provenance(target),
                "references": {"full": file_provenance(reference)},
            },
            "output_dir": str(current.output_dir),
            "docs": revision["docs"],
        }
        return {
            "fingerprint": harness.hash_payload(payload),
            "fingerprint_payload": payload,
            "packages": {},
            "git": {"commit": revision["docs"]},
        }

    monkeypatch.setattr(harness, "_provenance_payload", provenance)
    monkeypatch.setattr(
        runtime,
        "_code_identity",
        lambda root, evaluation: {"code": revision["evaluation" if evaluation else "code"]},
    )
    return cell, suite, revision


def write_outputs(output, cell):
    (output / "alignment").mkdir(exist_ok=True)
    (output / "dataset").mkdir(exist_ok=True)
    (output / "alignment/maps_global.tsv").write_text(
        "SrcEntity\tTgtEntity\tScore\nsource:1\ttarget:1\t0.9\nsource:2\ttarget:2\t0.8\n"
    )
    (output / "dataset/candidate_pool_manifest.json").write_text(
        json.dumps({"fingerprint": "pool-v1"})
    )
    # Invoke the actual existing evaluator, so replay checks meaningful persisted metrics.
    from exact.core.actions.evaluation import run_evaluation

    run_evaluation(
        output / "alignment/maps_global.tsv",
        output / "evaluation",
        full_reference_file_path=Path(cell.resolved_config["data"]["refs"]["full"]),
        error_on_fail=True,
        run_stats_path=output / "stats/run_stats.json",
    )


def test_harness_docs_relocation_and_evaluator_repair_make_no_new_model_calls(
    tmp_path, monkeypatch
):
    cell, suite, revision = fixture(tmp_path, monkeypatch)
    model_calls = []
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "7")

    def run(command, **kwargs):
        model_calls.append(command)
        assert kwargs["env"]["EXACT_EXPERIMENT_MODE"] == "1"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "7"
        output = Path(yaml.safe_load(Path(command[-1]).read_text())["job"]["output_dir"])
        write_outputs(output, cell)
        (output / "fitting").mkdir()
        (output / "fitting/training_units.json").write_text('{"units": 4}')
        (output / "source_decisions.json").write_text('{"schema_version": 2}')
        return 0, 0.1, None

    monkeypatch.setattr(harness, "_run_subprocess", run)
    first = harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)
    assert first["status"] == "complete"
    revision["docs"] = "edited-readme"
    moved_root = tmp_path / "relocated"
    moved = replace(
        cell,
        output_dir=moved_root / "run",
        recovery={"root": str(moved_root), "resume_from": cell.recovery["root"]},
    )
    replay = harness.execute_cell(moved, suite, workdir=tmp_path, resume=True)
    assert replay["status"] == "complete"
    assert len(model_calls) == 1
    assert json.loads((moved.output_dir / "fitting/training_units.json").read_text()) == {
        "units": 4
    }
    assert (moved.output_dir / "source_decisions.json").is_file()
    assert replay["recovery"]["reused_stages"] == ["evaluation", "extraction", "inputs"]
    revision["evaluation"] = "evaluator-bug-fixed"
    repaired = harness.execute_cell(
        replace(moved, recovery={"root": str(moved_root)}), suite, workdir=tmp_path, resume=True
    )
    assert repaired["status"] == "complete"
    assert len(model_calls) == 1
    assert repaired["recovery"]["reused_stages"] == ["extraction", "inputs"]
    assert harness.cell_metrics(moved.output_dir)["F1"] == pytest.approx(1.0)


def test_harness_interruption_imports_real_checkpoint_and_executes_only_missing_pairs(
    tmp_path, monkeypatch
):
    cell, suite, _ = fixture(tmp_path, monkeypatch)
    cell = replace(cell, recovery={**cell.recovery, "stop_after_checkpoint": True})
    encoded = []
    frame = pd.DataFrame({"Src": ["source:1", "source:2"], "Tgt": ["target:1", "target:2"]})

    def run(command, **kwargs):
        output = Path(yaml.safe_load(Path(command[-1]).read_text())["job"]["output_dir"])
        checkpoint = output / "checkpoints/inference.json"
        checkpoint.parent.mkdir(exist_ok=True)
        start = json.loads(checkpoint.read_text())["completed"] if checkpoint.exists() else 0
        for index in range(start, 2):
            encoded.append(index)
            checkpoint.write_text(json.dumps({"completed": index + 1}))
            runner = SimpleNamespace(dataset=SimpleNamespace(_active_dataframe=lambda: frame))
            with monkeypatch.context() as local:
                for key, value in kwargs["env"].items():
                    local.setenv(key, value)
                try:
                    runtime.runtime_checkpoint(runner, checkpoint, index + 1)
                except KeyboardInterrupt:
                    return 130, 0.1, None
        write_outputs(output, cell)
        return 0, 0.1, None

    monkeypatch.setattr(harness, "_run_subprocess", run)
    first = harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)
    assert first["status"] == "interrupted"
    moved_root = tmp_path / "continued"
    moved = replace(
        cell,
        output_dir=moved_root / "run",
        recovery={"root": str(moved_root), "resume_from": cell.recovery["root"]},
    )
    resumed = harness.execute_cell(moved, suite, workdir=tmp_path, resume=True)
    assert resumed["status"] == "complete"
    assert encoded == [0, 1]
    assert harness.cell_metrics(moved.output_dir)["F1"] == pytest.approx(1.0)


class FakeScorer(ScorerCommonMixin):
    device = torch.device("cpu")
    device_type = "cpu"
    fp16 = False
    pooling_method = PoolingMethod.MEAN
    _cache_tensor_dtype = torch.float32

    def __init__(self, counter):
        self.counter = counter
        self._cache_dirty = False

    def _encode_texts(self, tokenizer, model, texts, max_len):
        self.counter.extend(texts)
        return torch.tensor([[len(text), 1] for text in texts], dtype=torch.float32)


def test_fusion_repair_reuses_completed_encodings_even_with_new_batch_members(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path / "vectors"))
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "development")
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 1}, init_kwargs={})
    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="a" * 40, to_dict=lambda: {"hidden_size": 2}),
        training=False,
    )
    encoded = []
    first = FakeScorer(encoded)
    first._encode_with_cache(["a", "long"], tokenizer, model, 32, OrderedDict(), None)
    # A fresh scorer after changed fusion sees a different batch; only new text gets encoded.
    second = FakeScorer(encoded)
    replay = second._encode_with_cache(
        ["long", "a", "new"], tokenizer, model, 32, OrderedDict(), None
    )
    assert encoded == ["a", "long", "new"]
    assert replay.tolist() == [[4, 1], [1, 1], [3, 1]]
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "reporting")
    FakeScorer(encoded)._encode_with_cache(["a"], tokenizer, model, 32, OrderedDict(), None)
    assert encoded == ["a", "long", "new", "a"]


def test_changed_scoring_archives_incompatible_legacy_checkpoint(tmp_path, monkeypatch):
    cell, suite, revision = fixture(tmp_path, monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert not (cell.output_dir / "checkpoints/stale.json").exists()
        write_outputs(cell.output_dir, cell)
        return 0, 0.1, None

    monkeypatch.setattr(harness, "_run_subprocess", run)
    assert harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)["status"] == "complete"
    (cell.output_dir / "checkpoints").mkdir()
    (cell.output_dir / "checkpoints/stale.json").write_text(
        '{"legacy_model_fingerprint":"unchanged"}'
    )
    revision["code"] = "corrected-fusion"
    result = harness.execute_cell(cell, suite, workdir=tmp_path, resume=True)
    assert result["status"] == "complete"
    assert len(calls) == 2
    archive = (
        Path(cell.recovery["root"]) / "attempts" / result["recovery"]["attempt_id"] / "superseded"
    )
    assert (archive / "checkpoints/stale.json").is_file()


def test_current_result_set_rejects_superseded_attempts_missing_controls_and_mixed_repairs(
    tmp_path, monkeypatch
):
    cell, suite, revision = fixture(tmp_path, monkeypatch)

    def run(command, **kwargs):
        output = Path(yaml.safe_load(Path(command[-1]).read_text())["job"]["output_dir"])
        write_outputs(output, cell)
        return 0, 0.1, None

    monkeypatch.setattr(harness, "_run_subprocess", run)
    first = harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)
    second = harness.execute_cell(cell, suite, workdir=tmp_path, resume=True)
    root = Path(cell.recovery["root"])
    current = runtime.validate_campaign_results([second], [cell], root=root)
    assert current["cells"][0]["attempt_id"] == second["recovery"]["attempt_id"]
    with pytest.raises(ValueError, match="Superseded"):
        runtime.validate_campaign_results([first], [cell], root=root)
    with pytest.raises(ValueError, match="lacks planned"):
        runtime.validate_campaign_results([], [cell], root=root)
    revision["code"] = "new-fusion-code"
    candidate = replace(
        cell, arm_id="candidate", arm_role="candidate", output_dir=root / "candidate"
    )
    candidate_manifest = harness.execute_cell(candidate, suite, workdir=tmp_path, resume=False)
    with pytest.raises(ValueError, match="mix incompatible"):
        runtime.validate_campaign_results(
            [second, candidate_manifest], [cell, candidate], root=root
        )


def test_stop_prevents_queued_cells_from_starting(tmp_path, monkeypatch):
    cell, suite, _ = fixture(tmp_path, monkeypatch)
    root = Path(cell.recovery["root"])
    root.mkdir()
    (root / "STOP").touch()
    monkeypatch.setattr(
        harness, "execute_cell", lambda *_args, **_kwargs: pytest.fail("STOP scheduled work")
    )
    results = harness.run_cells([cell], suite, workdir=tmp_path, jobs=1, resume=True)
    assert results[0]["status"] == "interrupted"


@pytest.mark.parametrize("experiment", [False, True])
def test_experimental_subprocess_owns_session(tmp_path, experiment):
    import os
    import sys

    output = tmp_path / "stdout"
    code, _, _ = harness._run_subprocess(
        [sys.executable, "-c", "import os; print(os.getpid(), os.getsid(0))"],
        cwd=tmp_path,
        stdout_path=output,
        stderr_path=tmp_path / "stderr",
        env={"EXACT_EXPERIMENT_MODE": "1"} if experiment else {},
    )
    assert code == 0
    child, session = map(int, output.read_text().split())
    assert session == (child if experiment else os.getsid(0))


def test_directory_inputs_are_hashed_and_checkpointed_by_relative_file(tmp_path, monkeypatch):
    cell, _, _ = fixture(tmp_path, monkeypatch)
    graph = tmp_path / "csv-kg"
    graph.mkdir()
    (graph / "kg.yaml").write_text("triples: triples.csv\n")
    (graph / "triples.csv").write_text("source,predicate,target\na,r,b\n")
    config = {
        **cell.resolved_config,
        "data": {**cell.resolved_config["data"], "source": str(graph)},
    }
    cell = replace(cell, resolved_config=config)
    inputs = harness._path_provenance(config)
    first = runtime.CellRecovery(
        cell, {"fingerprint_payload": {"inputs": inputs}, "packages": {}}, tmp_path
    )
    assert inputs["source"]["files"].keys() == {"kg.yaml", "triples.csv"}
    assert "_locked_inputs/source/triples.csv" in first.input_files
    first.prepare()
    first.store.verify(first.identities["inputs"]["artifact_id"])
    (graph / "triples.csv").write_text("source,predicate,target\na,r,c\n")
    changed = runtime.CellRecovery(
        cell,
        {"fingerprint_payload": {"inputs": harness._path_provenance(config)}, "packages": {}},
        tmp_path,
    )
    assert first.identities["inputs"]["artifact_id"] != changed.identities["inputs"]["artifact_id"]


def test_explicit_deterministic_templates_never_call_a_generator(tmp_path, monkeypatch):
    from exact.core.entities.configs.config import ConfigModel
    from exact.impl.datasets.contextgraph import ContextDataset

    config = ConfigModel.from_mapping(
        {"config_version": 2, "dataset": {"verbalization_mode": "deterministic"}}, warn_v1=False
    )
    assert config.dataset_params.verbalization_mode == "deterministic"
    dataset = ContextDataset(output_path=tmp_path, verbalization_mode="deterministic")

    def forbidden(*args, **kwargs):
        pytest.fail("deterministic templates attempted generation")

    monkeypatch.setattr(dataset, "_batch_generate", forbidden)
    monkeypatch.setattr(dataset, "_ensure_verbaliser", forbidden)
    assert dataset._verbalize_triples([("A", "part_of", "B")]) == ["A part of B"]
    assert dataset._cache_fingerprint_payload()["verbalization_mode"] == "deterministic"


def test_local_evaluator_replay_joins_stripped_pool_without_models(tmp_path, monkeypatch):
    cell, _, _ = fixture(tmp_path, monkeypatch)
    pool = tmp_path / "pool.tsv"
    pool.write_text(
        "SrcEntity\tTgtEntity\tTgtCandidates\nsource:1\t\t['target:1', 'wrong']\nsource:2\t\t['target:2']\n"
    )
    config = {
        **cell.resolved_config,
        "data": {
            **cell.resolved_config["data"],
            "execution_mode": "local_ranking",
            "candidates": str(pool),
        },
    }
    cell = replace(cell, resolved_config=config)
    output = cell.output_dir / "alignment"
    output.mkdir(parents=True)
    (output / "maps_local.tsv").write_text(
        "SrcEntity\tTgtEntity\tTgtCandidates\nsource:1\t\t[('target:1', 0.9), ('wrong', 0.1)]\nsource:2\t\t[('target:2', 0.8)]\n"
    )
    provenance = {
        "fingerprint_payload": {"inputs": harness._path_provenance(config)},
        "packages": {},
    }
    recovery = runtime.CellRecovery(cell, provenance, tmp_path)
    recovery.evaluate()
    result = json.loads((cell.output_dir / "evaluation/evaluation_results.json").read_text())
    assert result["builtin"]["MRR"] == 1.0


def test_experiment_verbalization_errors_do_not_become_silent_text_fallback(monkeypatch):
    from exact.impl.models.pair_adaptive_evidence import PairAdaptiveEvidenceMixin

    def forbidden(*args):
        raise AssertionError("forbidden generation")

    scorer = PairAdaptiveEvidenceMixin()
    scorer._attached_dataset = SimpleNamespace(_verbalize_triples=forbidden)
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    with pytest.raises(AssertionError, match="forbidden generation"):
        scorer._verbalize_object_items([{"triple": ("a", "r", "b")}])


def test_relocation_keeps_original_inference_cost_not_cache_replay_wall(tmp_path, monkeypatch):
    cell, suite, _ = fixture(tmp_path, monkeypatch)
    timing = {
        "sessions": [
            {
                "stages": [
                    {"stage": "Total", "seconds": 500},
                    {"stage": "Dataset.Process", "seconds": 3},
                    {"stage": "Alignment.Fitting", "seconds": 400},
                    {"stage": "Alignment.Inference", "seconds": 5},
                    {"stage": "Alignment.PostInference", "seconds": 2},
                    {"stage": "Postprocess.Evaluation", "seconds": 10},
                ]
            }
        ]
    }

    def run(command, **kwargs):
        write_outputs(cell.output_dir, cell)
        (cell.output_dir / "timings.json").write_text(json.dumps(timing))
        return 0, 500, 1

    monkeypatch.setattr(harness, "_run_subprocess", run)
    original = harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)
    relocated = replace(
        cell,
        output_dir=tmp_path / "relocated/run",
        recovery={"root": str(tmp_path / "relocated"), "resume_from": str(tmp_path / "campaign")},
    )
    monkeypatch.setattr(
        harness, "_run_subprocess", lambda *args, **kwargs: pytest.fail("replay ran a model")
    )
    replay = harness.execute_cell(relocated, suite, workdir=tmp_path, resume=True)
    assert original["inference_seconds"] == replay["inference_seconds"] == 10
    assert replay["wall_seconds"] < original["wall_seconds"]
    assert json.loads((relocated.output_dir / "timings.json").read_text()) == timing


def test_published_comparator_uses_shared_recovery_and_replays_full_population_across_seeds(
    tmp_path, monkeypatch
):
    from exact.experiments import published_matcher

    cell, suite, revision = fixture(tmp_path, monkeypatch)
    binding = {
        "matcher": "logmap",
        "bundle": {"path": "/declared/bundle", "sha256": "a" * 64},
        "jar": "logmap.jar",
        "timeout_seconds": 30,
        "java_heap_gb": 1,
        "java_threads": 1,
    }
    cell = replace(cell, published_matcher=binding, seed=17)
    calls = []

    def run(current):
        calls.append(current.seed)
        write_outputs(current.output_dir, current)
        (current.output_dir / "published").mkdir()
        (current.output_dir / "published/raw.rdf").write_text("immutable external predictions")
        (current.output_dir / "evaluation_inputs").mkdir()
        (current.output_dir / "evaluation_inputs/reference.tsv").write_text("canonical reference")
        return 0, 1, 2

    monkeypatch.setattr(published_matcher, "run_cell", run)
    monkeypatch.setattr(
        harness,
        "_run_subprocess",
        lambda *args, **kwargs: pytest.fail("published comparator invoked Exact model runner"),
    )
    first = harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)
    assert first["status"] == "complete"
    second_cell = replace(cell, seed=29, output_dir=tmp_path / "campaign/second")
    second = harness.execute_cell(second_cell, suite, workdir=tmp_path, resume=False)
    assert second["recovery"]["artifacts"] == first["recovery"]["artifacts"]
    assert calls == [17]
    assert (
        second_cell.output_dir / "published/raw.rdf"
    ).read_text() == "immutable external predictions"
    assert (second_cell.output_dir / "evaluation_inputs/reference.tsv").is_file()
    revision["evaluation"] = "fixed-evaluator"
    evaluated = []
    monkeypatch.setattr(
        published_matcher, "evaluate_cell", lambda current: evaluated.append(current.seed)
    )
    repaired = harness.execute_cell(second_cell, suite, workdir=tmp_path, resume=True)
    assert repaired["status"] == "complete"
    assert evaluated == [29]
    assert calls == [17]


def test_e00_posthoc_attribution_uses_separate_development_reference(tmp_path, monkeypatch):
    cell, _, _ = fixture(tmp_path, monkeypatch)
    mapping = {**cell.resolved_config, "run": {"experiment_audit": True}}
    mapping["data"] = {**mapping["data"], "refs": {"dev": mapping["data"]["refs"]["full"]}}
    cell = replace(cell, resolved_config=mapping, negative_label_policy="complete_reference")
    cell.output_dir.mkdir(parents=True)
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "source_universe": ["source:1", "source:2"],
        "policy": {"source_cardinality": 1, "target_cardinality": 1},
        "records": [
            {
                "Src": "source:1",
                "candidates": [
                    {"target": "target:1", "S_final": 0.9, "emitted": True, "relation": "="}
                ],
                "pre_typing_targets": ["target:1"],
                "emitted_targets": ["target:1"],
            },
            {"Src": "source:2", "candidates": []},
        ],
    }
    (cell.output_dir / "source_decisions.json").write_text(json.dumps(trace))
    result = harness._post_run_provenance(cell)["error_attribution"]
    assert result["diagnostic_only"]
    assert result["observed"]["tp"] == 1
    assert result["observed"]["fn_known_positive"] == 1
    saved = json.loads((cell.output_dir / "diagnostics/error_attribution.json").read_text())
    assert saved["records"][1]["flags"]["candidate_loss"]
    monkeypatch.setattr(harness, "read_table", lambda *_: pytest.fail("opened reporting reference"))
    with pytest.raises(ValueError, match="development/diagnostic"):
        harness._post_run_provenance(replace(cell, split_role="reporting", reference_role="test"))


def test_posthoc_failure_reuses_completed_extraction_on_repair(tmp_path, monkeypatch):
    cell, suite, revision = fixture(tmp_path, monkeypatch)
    model_calls = []

    def run(command, **kwargs):
        model_calls.append(command)
        write_outputs(cell.output_dir, cell)
        return 0, 0.1, None

    def broken(_cell):
        raise ValueError("posthoc diagnostic bug")

    monkeypatch.setattr(harness, "_run_subprocess", run)
    monkeypatch.setattr(harness, "_error_attribution", broken)
    failed = harness.execute_cell(cell, suite, workdir=tmp_path, resume=False)
    assert failed["status"] == "failed"
    assert failed["failure"]["message"] == "posthoc diagnostic bug"
    assert "extraction" in failed["recovery"]["artifacts"]
    assert "evaluation" not in failed["recovery"]["artifacts"]
    revision["evaluation"] = "posthoc-fixed"
    monkeypatch.setattr(harness, "_error_attribution", lambda _: None)
    repaired = harness.execute_cell(cell, suite, workdir=tmp_path, resume=True)
    assert repaired["status"] == "complete"
    assert repaired["recovery"]["reused_stages"] == ["extraction", "inputs"]
    assert len(model_calls) == 1
    assert harness.cell_metrics(cell.output_dir)["F1"] == 1
