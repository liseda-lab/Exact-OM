from pathlib import Path

import pytest

from exact.core.entities.evaluation import EvaluationRequest
from exact.impl.evaluators import bioml


class FakeBioMLApi:
    version = "fake-1"

    def __init__(self, missing=()):
        self.missing = frozenset(missing)

    def evaluate_equivalence(self, request):
        return {"precision": 0.8, "recall": 0.5, "f1": 0.615}

    def evaluate_ranking(self, request):
        return {"mrr": 0.75, "hits_at_1": 0.5}

    def evaluate_typed(self, request):
        return {"typed_mrr": 0.6, "hierarchy_aware_ndcg_at_10": 0.7}

    def evaluate_coherence(self, request):
        return {
            "global_coherence": 0.0,
            "union_class_count": 2,
            "reasoner_used": "hermit",
            "provenance": {"schema": "coherence-provenance/1"},
        }


def test_bioml_adapter_uses_available_capabilities(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bioml, "_load_bioml_api", lambda: FakeBioMLApi())
    request = EvaluationRequest(
        alignment=tmp_path / "alignment.tsv",
        full_reference=tmp_path / "reference.tsv",
        source=object(),
        target=object(),
    )

    result = bioml.BioMLEvaluator.run(request)

    assert result.metrics["equivalence.f1"] == pytest.approx(0.615)
    assert result.metrics["coherence.global_coherence"] == 0.0
    assert result.metrics["coherence.union_class_count"] == 2.0
    assert result.details["coherence"]["reasoner_used"] == "hermit"
    assert result.details["coherence"]["provenance"] == {"schema": "coherence-provenance/1"}
    assert result.version == "fake-1"


def test_bioml_adapter_reports_missing_capability_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr(bioml, "_load_bioml_api", lambda: FakeBioMLApi(missing={"equivalence"}))

    result = bioml.BioMLEvaluator.run(EvaluationRequest(alignment=[], full_reference=[]))

    assert result.metrics == {
        "equivalence.precision": None,
        "equivalence.recall": None,
        "equivalence.f1": None,
    }
    assert set(result.skipped) == set(result.metrics)


def test_missing_optional_dependency_has_actionable_message(monkeypatch) -> None:
    real_import = bioml.importlib.import_module

    def missing(name):
        if name == "oaei_bioml_eval":
            raise ModuleNotFoundError(name="oaei_bioml_eval")
        return real_import(name)

    monkeypatch.setattr(bioml.importlib, "import_module", missing)

    with pytest.raises(bioml.BioMLDependencyError, match=r"exact-om\[bioml-eval\]"):
        bioml._load_bioml_api()
