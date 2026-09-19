"""Tiny direct-proposal, checkpoint and external deadline integration."""

import json
import subprocess
import sys

import pyowl_core as owl
import pytest

from exact.repair.api import prepare_repair, write_artifact
from exact.repair.circuit import ConditionedMixture, ProposalEncoding
from exact.repair.compilation import compile_bounded
from exact.repair.graph import build_observable_graph
from exact.repair.owl import snapshot_from_axioms
from exact.repair.records import make_objective, read_record
from exact.repair.retrieval import retrieve_vocabulary


def neural_fixture():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pysdd")
    pytest.importorskip("torch_geometric")
    from exact.repair.model import RepairModel

    a, a1, b = (owl.Class(owl.IRI("urn:proposal:" + x)) for x in ("A", "A1", "B"))
    problem = prepare_repair(
        snapshot_from_axioms((owl.SubClassOf(a, a1),)),
        snapshot_from_axioms((owl.Declaration(b),)),
        [{"Src": a.iri.value, "Tgt": b.iri.value, "Score": 0.8, "object_id": "m"}],
    )
    retrieval = retrieve_vocabulary(problem)
    graph = build_observable_graph(
        problem.objects,
        fixed_axioms=problem.fixed_axioms,
        source_axioms=problem.source_axioms,
        target_axioms=problem.target_axioms,
        evidence=retrieval.graph_evidence(problem.evidence),
        retrieved_symbols=retrieval.symbols,
        explanations=retrieval.explanations,
    )
    torch.manual_seed(13)
    model = RepairModel(graph.metadata, hidden_dim=8, heads=2, layers=0, dropout=0)
    return problem, model


def test_public_circuit_serialization_preserves_probabilities_order_and_cache():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pysdd")
    encoding = ProposalEncoding((("x", ("a", "b")),), ((True, False), (False, True)), ("a", "b"))
    circuit = compile_bounded(encoding, seconds=5, variable_order=[2, 1])
    assert compile_bounded(encoding, seconds=5, variable_order=(2, 1)) is circuit
    distribution = ConditionedMixture(circuit, torch.zeros((1, 2)))
    assert float(distribution.log_probability((True, False)).exp()) == pytest.approx(0.5)
    with pytest.raises(TimeoutError):
        compile_bounded(encoding, seconds=0.0001)


def test_supervised_direct_round_preserves_parent_and_generates_bounded_controls():
    from exact.repair.pipeline import bounded_freeze_neural_round, model_digest

    problem, model = neural_fixture()
    previous = model_digest(model)
    outcome = bounded_freeze_neural_round(
        problem,
        model,
        seconds=20,
        compile_seconds=5,
        proposal_arm="grammar_uniform",
        draws_per_object=3,
        candidate_cap=32,
        max_depth=1,
        max_constructors=1,
    )
    assert outcome.status == "complete", outcome.detail
    frozen = outcome.value
    assert model.training and model_digest(model) == previous
    assert frozen.problem.graph_identity and frozen.model_hash == previous
    report = frozen.proposal_reports[0]
    assert report["attempted_draws"] == 3 and report["valid_draws"] == 3
    assert report["arm"] == "grammar_uniform"
    assert {c.candidate_id for c in problem.objects[0].candidates} <= {
        c.candidate_id for c in frozen.problem.objects[0].candidates
    }
    assert bounded_freeze_neural_round(problem, model, seconds=0.0001).status == "timeout"


def test_checkpoint_cli_runs_without_matching_and_writes_replayable_complex_bundle(tmp_path):
    torch = pytest.importorskip("torch")
    from exact.repair.kernel import replay_safety

    problem, model = neural_fixture()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "metadata": model.metadata, "config": model.config},
        checkpoint,
    )
    input_path, output = tmp_path / "input.json", tmp_path / "result.json"
    write_artifact(
        input_path,
        {"input": problem.to_dict(), "objective": make_objective(problem.objects).to_dict()},
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "exact.delivery.cli.repair",
            "--problem",
            str(input_path),
            "--model",
            str(checkpoint),
            "--output",
            str(output),
            "--seconds",
            "25",
            "--stage-seconds",
            "5",
            "--compile-seconds",
            "5",
            "--proposal-seconds",
            "15",
            "--grammar-depth",
            "1",
            "--constructors",
            "1",
            "--draws",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text())
    frozen, result = read_record(payload["input"]), read_record(payload["result"])
    assert result.logical_status == "VERIFIED_FEASIBLE"
    assert replay_safety(frozen, result, timeout=5)
    assert payload["stage_seconds"]["proposal"] > 0


def test_cli_preserves_captured_controls_when_checkpoint_proposal_fails(tmp_path):
    problem, _ = neural_fixture()
    input_path, output = tmp_path / "input.json", tmp_path / "fallback.json"
    write_artifact(
        input_path,
        {"input": problem.to_dict(), "objective": make_objective(problem.objects).to_dict()},
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "exact.delivery.cli.repair",
            "--problem",
            str(input_path),
            "--model",
            str(tmp_path / "missing.pt"),
            "--output",
            str(output),
            "--seconds",
            "15",
            "--stage-seconds",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text())
    result = read_record(payload["result"])
    assert result.logical_status == "VERIFIED_FEASIBLE"
    assert result.candidate_coverage == "captured_pool_fallback"
    assert any("proposal:" in failure for failure in result.failures)


def test_graph_edge_explanation_and_text_limits_record_omissions():
    from exact.repair.graph import GraphExplanation

    problem, _ = neural_fixture()
    minimal = build_observable_graph(problem.objects)
    with pytest.raises(ValueError, match="mandatory"):
        build_observable_graph(problem.objects, max_edges=len(minimal.edges) - 1)
    graph = build_observable_graph(
        problem.objects,
        fixed_axioms=problem.fixed_axioms,
        evidence={"m": {"text": "first second third fourth"}},
        max_text_tokens=2,
        explanations=(GraphExplanation("support", support_object_ids=("m",)),),
        max_explanations=0,
    )
    assert graph.omitted_supports == ("support",)
    assert any("text_tokens:" in omission for omission in graph.omitted_evidence)
    assert not any("token:third" in key for node in graph.nodes for key, _ in node.features)
