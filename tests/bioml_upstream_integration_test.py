from pathlib import Path

import pytest

from exact.core.entities.evaluation import EvaluationRequest
from exact.impl.evaluators.bioml import BioMLEvaluator
from exact.ontology import load_ontology

pytestmark = pytest.mark.requires_data


def _write_mappings(path: Path) -> None:
    path.write_text(
        "SrcEntity\tTgtEntity\tScore\nhttp://s\thttp://t\t1.0\n",
        encoding="utf-8",
    )


def test_pinned_upstream_exposes_at_least_equivalence_metrics(tmp_path: Path) -> None:
    pytest.importorskip("oaei_bioml_eval")
    alignment = tmp_path / "alignment.tsv"
    reference = tmp_path / "reference.tsv"
    _write_mappings(alignment)
    _write_mappings(reference)

    result = BioMLEvaluator.run(EvaluationRequest(alignment=alignment, full_reference=reference))

    assert result.metrics["equivalence.f1"] == pytest.approx(1.0)


def test_pinned_upstream_scores_exacts_scored_local_cell_format(tmp_path: Path) -> None:
    pytest.importorskip("oaei_bioml_eval")
    alignment = tmp_path / "local.tsv"
    alignment.write_text(
        "SrcEntity\tTgtEntity\tTgtCandidates\n"
        "http://s\thttp://gold\t"
        '[("http://other", 0.9), ("http://gold", 0.8)]\n',
        encoding="utf-8",
    )

    result = BioMLEvaluator.run(EvaluationRequest(alignment=alignment, k=(1, 5)))

    assert result.metrics["ranking.mrr"] == pytest.approx(0.5)
    assert result.metrics["ranking.hits_at_1"] == pytest.approx(0.0)
    assert result.metrics["ranking.hits_at_5"] == pytest.approx(1.0)


def test_upstream_coherence_reuses_exact_snapshot_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = pytest.importorskip("oaei_bioml_eval.coherence.report")
    reasoner_module = pytest.importorskip("oaei_bioml_eval.coherence.reasoner")
    source = load_ontology(
        b"Ontology(<urn:exact:source> Declaration(Class(<urn:exact:S>)))",
        document_iri="urn:exact:source-document",
    )
    target = load_ontology(
        b"Ontology(<urn:exact:target> Declaration(Class(<urn:exact:T>)))",
        document_iri="urn:exact:target-document",
    )
    source_snapshot = source.owl_snapshot()
    target_snapshot = target.owl_snapshot()
    observed: dict[str, bool] = {}

    class RecordingReasoner:
        def unsatisfiable_classes_view(self, ontology, *, which, timeout_s):
            del timeout_s
            observed["source"] = ontology.members[0].view is source_snapshot
            observed["target"] = ontology.members[1].view is target_snapshot
            return reasoner_module.UnsatResult((), which, 0.0)

    monkeypatch.setattr(report, "load_reasoner", RecordingReasoner)
    alignment = tmp_path / "alignment.tsv"
    alignment.write_text(
        "SrcEntity\tTgtEntity\tScore\nurn:exact:S\turn:exact:T\t1.0\n",
        encoding="utf-8",
    )

    result = BioMLEvaluator.run(
        EvaluationRequest(
            alignment=alignment,
            full_reference=alignment,
            source=source,
            target=target,
            options={"coherence_reasoner": "hermit", "coherence_timeout_s": 30.0},
        )
    )

    assert observed == {"source": True, "target": True}
    assert result.metrics["coherence.global_coherence"] == 0.0
    assert result.metrics["coherence.union_class_count"] == 2.0
