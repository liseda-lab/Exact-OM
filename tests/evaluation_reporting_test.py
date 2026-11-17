import json
from pathlib import Path

from exact.core.actions.evaluation import EvaluationAction
from exact.impl.evaluators import bioml
from exact.utils.provenance import sha256_file
from exact.utils.timing import TimingLedger


def _write_global(path: Path, rows: list[tuple[str, str, float]]) -> None:
    lines = ["SrcEntity\tTgtEntity\tScore"]
    lines.extend(f"{source}\t{target}\t{score}" for source, target, score in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FakeApi:
    version = "fake-2"
    missing = frozenset()

    def evaluate_equivalence(self, request):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def evaluate_ranking(self, request):
        return {"mrr": 1.0}

    def evaluate_typed(self, request):
        return {}

    def evaluate_coherence(self, request):
        return {
            "global_coherence": 0.0,
            "reasoner_used": "hermit",
            "provenance": {"schema": "coherence-provenance/1"},
        }


def test_builtin_only_csv_is_backward_compatible_and_json_is_canonical(tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.tsv"
    reference = tmp_path / "reference.tsv"
    _write_global(alignment, [("s", "t", 0.9)])
    _write_global(reference, [("s", "t", 1.0)])

    results = EvaluationAction.run(
        alignment=alignment,
        output_dir_path=tmp_path,
        full_reference_file_path=reference,
    )

    assert results == {"P": 1.0, "R": 1.0, "F1": 1.0}
    assert (tmp_path / "evaluation_results.csv").read_text(encoding="utf-8") == (
        "Metric,Value\nP,1.0\nR,1.0\nF1,1.0\n"
    )
    report = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))
    assert report["builtin"] == results
    assert report["meta"]["refs"]["full_reference"]["rows"] == 1
    assert report["meta"]["refs"]["full_reference"]["sha256"] == sha256_file(reference)
    sessions = TimingLedger.open(tmp_path).sessions()
    assert len(sessions) == 1
    assert sessions[0].command == "eval"
    assert [record.stage for record in sessions[0].stages] == ["Postprocess.Evaluation"]


def test_multiple_backends_are_namespaced(monkeypatch, tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.tsv"
    reference = tmp_path / "reference.tsv"
    _write_global(alignment, [("s", "t", 0.9)])
    _write_global(reference, [("s", "t", 1.0)])
    monkeypatch.setattr(bioml, "_load_bioml_api", lambda: _FakeApi())

    results = EvaluationAction.run(
        alignment=alignment,
        output_dir_path=tmp_path,
        full_reference_file_path=reference,
        backends=["builtin", "bioml"],
    )

    assert results["builtin.F1"] == 1.0
    assert results["bioml.equivalence.f1"] == 1.0
    csv_text = (tmp_path / "evaluation_results.csv").read_text(encoding="utf-8")
    assert "builtin.F1,1.0" in csv_text
    assert "bioml.equivalence.f1,1.0" in csv_text


def test_bioml_coherence_provenance_is_persisted(monkeypatch, tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.tsv"
    reference = tmp_path / "reference.tsv"
    _write_global(alignment, [("s", "t", 0.9)])
    _write_global(reference, [("s", "t", 1.0)])
    monkeypatch.setattr(bioml, "_load_bioml_api", lambda: _FakeApi())

    EvaluationAction.run(
        alignment=alignment,
        output_dir_path=tmp_path,
        full_reference_file_path=reference,
        source_file_path=object(),
        target_file_path=object(),
        backends=["bioml"],
    )

    report = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))
    coherence = report["meta"]["backend_details"]["bioml"]["coherence"]
    assert coherence["reasoner_used"] == "hermit"
    assert coherence["provenance"]["schema"] == "coherence-provenance/1"


def test_reference_hash_mismatch_is_recorded(tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.tsv"
    reference = tmp_path / "reference.tsv"
    train = tmp_path / "train.tsv"
    _write_global(alignment, [("s", "t", 0.9)])
    _write_global(reference, [("s", "t", 1.0)])
    _write_global(train, [("s0", "t0", 1.0)])
    stats = tmp_path / "existing_run_stats.json"
    stats.write_text(
        json.dumps({"selector_calibration": {"training_reference_sha256": "not-the-hash"}}),
        encoding="utf-8",
    )

    EvaluationAction.run(
        alignment=alignment,
        output_dir_path=tmp_path,
        full_reference_file_path=reference,
        train_reference_file_path=train,
        run_stats_path=stats,
    )

    report = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))
    assert "differs" in report["meta"]["warnings"][0]
    persisted = json.loads(stats.read_text(encoding="utf-8"))
    assert persisted["evaluation_inputs"]["train_reference"]["sha256"] == sha256_file(train)
