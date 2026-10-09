"""Second-review architecture authority and obligation-conditioned supervision."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from exact.repair.graph import (
    FEATURE_SCHEMA_V3,
    EffectivePreparation,
    build_observable_graph,
)
from exact.repair.learning import SemanticTargetSpec, support_loss, support_targets
from exact.repair.records import candidate_cost, canonical_hash
from exact.repair.workers import CallResult
from tests.repair_learning_v3_test import problem
from tests.repair_training_completion_test import cache_for
from tools.repair import train
from tools.repair.corpus import generate_corpus
from tools.repair.prepare import publish_label_cache


def protocol_fixture(tmp_path, enabled):
    template = (
        Path(__file__).parents[1] / "specs/exact-repair/protocol/xr21-review2-conformance.json"
    )
    protocol = json.loads(template.read_text().replace('"UNFROZEN"', '"conformance-fixture"'))
    protocol["identity"]["execution_authorized"] = True
    protocol["model"].update(
        pair_benefit=enabled,
        plan_risk=False,
        support_enabled=False,
        hidden_width=8,
        attention_heads=2,
        layers=1,
        dropout=0.0,
    )
    protocol["training"].update(max_epochs=1, patience=1, sampled_assignments=0, device="cpu")
    protocol["collection"]["rounds"] = 0
    protocol["circuit"]["cache_directory"] = str(tmp_path / "circuits")
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path, protocol


@pytest.mark.parametrize("enabled", [True, False])
def test_protocol_architecture_reaches_entrypoint_model_preparation_and_warm_start(
    tmp_path, monkeypatch, enabled
):
    import torch

    from exact.repair.model import RepairModel

    path, protocol = protocol_fixture(tmp_path, enabled)
    cases = generate_corpus(
        revision="v3",
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("interaction_complementary",),
    )
    constructions, pairs = [], []
    original_init, original_pairs = RepairModel.__init__, EffectivePreparation.pairs

    def construct(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        constructions.append((self.config.copy(), self.pair_head is not None))

    def prepare(self, *args, **kwargs):
        result = original_pairs(self, *args, **kwargs)
        pairs.append((kwargs["enabled"], result.pairs))
        return result

    monkeypatch.setattr(RepairModel, "__init__", construct)
    monkeypatch.setattr(EffectivePreparation, "pairs", prepare)
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: {"exact_regret": 0.0})
    monkeypatch.setattr(
        train,
        "generated_development",
        lambda *a, **k: dict(
            useful_candidate_coverage=1.0,
            decoded=dict(status="verified", selected_utility=0.5, checks=1),
        ),
    )
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)

    def worker(function, *args, **kwargs):
        if function.__name__ == "generated_from_protocol":
            value = cases
        elif function is train._label_payload:
            case = args[0]
            cache = cache_for(case)
            hashes = dict(cache.hashes)
            hashes.update(
                profile=canonical_hash(kwargs["profile"]),
                semantic_target=SemanticTargetSpec(canonical_hash(case.probes)).content_hash,
            )
            value = replace(
                cache,
                schema="exact-repair/teacher-cache/v3",
                hashes=tuple(sorted(hashes.items())),
                labels=tuple(
                    replace(
                        label,
                        cost=sum(
                            candidate_cost(obj, obj.candidates[choice], kwargs["profile"])
                            for obj, choice in zip(case.problem.objects, label.assignment)
                        ),
                    )
                    for label in cache.labels
                ),
            )
            value = publish_label_cache(value, args[1])
        elif function is train._train_payload:
            assert args[2]["pairwise"] is enabled
            value = function(*args)
        else:
            raise AssertionError(function)
        return CallResult("complete", value, resource_usage=(("cpu_seconds", 0.0),))

    monkeypatch.setattr(train, "bounded_call", worker)
    warm_path = None
    for name in ("cold", "warm"):
        output = tmp_path / name
        argv = ["repair-train", "--protocol", str(path), "--output", str(output)]
        if warm_path:
            argv += ["--warm-start", str(warm_path)]
        monkeypatch.setattr(sys, "argv", argv)
        assert train.main() == 0
        warm_path = output / "model.pt"
        checkpoint = torch.load(warm_path, weights_only=True)
        assert checkpoint["config"]["pairwise"] is enabled
        assert checkpoint["config"]["encoder"] == protocol["model"]["backbone"]
        report = json.loads((output / "report.json").read_text())
        assert report["model_configuration"] == checkpoint["config"]
        assert checkpoint["training_provenance"]["model_configuration"] == checkpoint["config"]
    assert len(constructions) == 2
    assert all(config["pairwise"] is enabled and head is enabled for config, head in constructions)
    assert pairs and all(value is enabled for value, _ in pairs)
    assert all(bool(selected) is enabled for _, selected in pairs)


def test_required_and_prohibited_supports_expose_distinct_obligation_inputs():
    import torch

    from exact.repair.kernel import materialize
    from exact.repair.model import RepairModel
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    p = problem()
    query = p.objects[0].original_axioms[0]
    assignment = (0, 0, 0)
    targets = []
    for kind in ("required_entailment", "prohibited_entailment"):
        policy = replace(
            p.policy,
            required=(query,) if kind == "required_entailment" else (),
            prohibited=(query,) if kind == "prohibited_entailment" else (),
        )
        case = replace(p, policy=policy)
        axioms, active = materialize(case, assignment)
        report = OwlVerifier("hermit", backend="python").check_theory(
            snapshot_from_axioms(axioms),
            policy.monitored_classes,
            required=policy.required,
            prohibited=policy.prohibited,
            activated=active,
        )
        targets.append(
            next(
                t
                for t in support_targets(case, assignment, report, axioms, active)
                if t.obligation_kind == kind
            )
        )
    assert targets[0].witness == targets[1].witness
    assert [t.violated for t in targets] == [False, True]
    assert all(t.eligible for t in targets)
    graph = build_observable_graph(p.objects, feature_schema=FEATURE_SCHEMA_V3)
    model = RepairModel(
        graph.metadata,
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0,
        revision="v3",
        support_enabled=True,
    )
    memory = model.encode(graph)
    inputs = []
    handle = model.support_head.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    logits = torch.stack(
        [
            model.support_violation_logit(
                p.objects,
                memory,
                t.assignment,
                query,
                obligation_kind=t.obligation_kind,
                expected_truth=t.expected_truth,
            )
            for t in targets
        ]
    )
    handle.remove()
    assert [t.expected_truth for t in targets] == [True, False]
    assert not torch.equal(inputs[0], inputs[1])
    assert support_loss(logits, targets)["eligible"] == 2
    with pytest.raises(ValueError, match="expected truth"):
        model.support_violation_logit(
            p.objects,
            memory,
            assignment,
            query,
            obligation_kind="required_entailment",
            expected_truth=False,
        )
    model.support_enabled = False
    with pytest.raises(ValueError, match="disabled"):
        model.support_violation_logit(
            p.objects,
            memory,
            assignment,
            query,
            obligation_kind="required_entailment",
            expected_truth=True,
        )


@pytest.mark.parametrize(
    "enabled,override",
    [(True, ["--no-pairwise"]), (False, ["--pairwise"]), (True, ["--encoder", "none"])],
)
def test_conflicting_cli_architecture_is_rejected_before_preparation(
    tmp_path, monkeypatch, enabled, override
):
    path, _ = protocol_fixture(tmp_path, enabled)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv", ["repair-train", "--protocol", str(path), "--output", str(output), *override]
    )
    monkeypatch.setattr(train, "bounded_call", lambda *a, **k: pytest.fail("worker dispatched"))
    with pytest.raises(ValueError, match="CLI .*conflicts with the protocol architecture"):
        train.main()
    assert not output.exists()


@pytest.mark.parametrize(
    "field,changed",
    [
        ("pairwise", False),
        ("encoder", "none"),
        ("hidden_dim", 16),
        ("layers", 2),
        ("heads", 4),
        ("dropout", 0.5),
        ("feature_dim", 64),
        ("revision", "v2"),
        ("plan_risk", True),
        ("support_enabled", True),
        ("pair_factor_bound", 2),
        ("support_readout_identity", None),
    ],
)
def test_conflicting_warm_architecture_is_rejected_before_preparation(
    tmp_path, monkeypatch, field, changed
):
    import torch

    from exact.repair.model import RepairModel
    from exact.repair.protocol import load_protocol_v3, training_projection_v3

    path, _ = protocol_fixture(tmp_path, True)
    config = train._protocol_arguments(training_projection_v3(load_protocol_v3(path)))
    keys = (
        "hidden_dim",
        "layers",
        "heads",
        "dropout",
        "encoder",
        "pairwise",
        "revision",
        "plan_risk",
        "support_enabled",
        "pair_factor_bound",
        "support_readout_identity",
    )
    model = RepairModel(
        (("class",), (("class", "self", "class"),)), **{key: config[key] for key in keys}
    )
    architecture = dict(model.config)
    if changed is None:
        del architecture[field]
    else:
        architecture[field] = changed
    warm_path = tmp_path / "warm.pt"
    torch.save(
        dict(
            model_schema="exact-repair/model/v3",
            config=architecture,
            training_provenance={"heldout_supervision": False},
            state_dict=model.state_dict(),
            metadata=model.metadata,
        ),
        warm_path,
    )
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repair-train",
            "--protocol",
            str(path),
            "--output",
            str(output),
            "--warm-start",
            str(warm_path),
        ],
    )
    monkeypatch.setattr(train, "bounded_call", lambda *a, **k: pytest.fail("worker dispatched"))
    with pytest.raises(ValueError, match=f"Warm-start architecture conflicts.*{field}"):
        train.main()
    assert not output.exists()


def test_unsupported_unary_architecture_cannot_be_silently_enabled(tmp_path, monkeypatch):
    path, protocol = protocol_fixture(tmp_path, False)
    protocol["model"]["unary_benefit"] = False
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(
        sys, "argv", ["repair-train", "--protocol", str(path), "--output", str(tmp_path / "run")]
    )
    with pytest.raises(ValueError, match="unary_benefit"):
        train.main()
