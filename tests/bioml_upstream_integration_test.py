from pathlib import Path

import pytest

from exact.core.entities.evaluation import EvaluationRequest
from exact.impl.evaluators.bioml import BioMLEvaluator

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
