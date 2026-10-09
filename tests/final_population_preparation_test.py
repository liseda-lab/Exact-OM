import copy
from pathlib import Path

import pytest

from tools import prepare_final_populations as prep
from tools.prepared_batch import binding, read, write


def protocol(tmp_path):
    ontology = tmp_path / "ontology.ofn"
    ontology.write_text("Ontology(Declaration(Class(<urn:source>)))")
    queries = tmp_path / "public.tsv"
    queries.write_text("SrcEntity\tTgtCandidates\nurn:s\t['urn:t']\nurn:s\t['urn:v']\n")
    code = Path(prep.__file__).resolve().parents[1]
    names = [
        "tools/prepare_final_populations.py",
        "exact/experiments/public_inference.py",
        "exact/experiments/inputs.py",
        "exact/io/sources/owl.py",
        "exact/ontology/__init__.py",
        "exact/ontology/view_contract.py",
    ]
    payload = {
        "schema_version": 1,
        "kind": "final_native_populations",
        "entity_kinds": ["class"],
        "native_threads": 2,
        "selection_eligible": False,
        "hosted_calls": False,
        "rationales": False,
        "filter_ignored": True,
        "ontologies": {
            name: {"input": binding(ontology), "source_options": {}}
            for name in ["NCIT", "DOID", "SNOMED", "FMA"]
        },
        "cases": {
            name: {
                "source": s,
                "target": t,
                "public_queries": binding(queries),
                "query_count": 2,
                "source_count": 1,
            }
            for name, s, t in [
                ("H0", "NCIT", "DOID"),
                ("H1", "SNOMED", "FMA"),
                ("H2", "SNOMED", "NCIT"),
            ]
        },
        "source": {name: binding(code / name) for name in names},
        "output": str(tmp_path / "outputs"),
        "pause_paths": [str(tmp_path / "STOP")],
    }
    path = tmp_path / "protocol.json"
    write(path, payload)
    return path, payload


def test_interruption_reuses_native_population_and_preserves_repeated_queries(
    tmp_path, monkeypatch
):
    path, payload = protocol(tmp_path)
    calls = []

    def run(command, log_path, pauses):
        name = command[-1]
        if calls == ["DOID"]:
            calls.append("interrupted")
            raise RuntimeError("interrupted between populations")
        prep.prepare_one(path, name)
        calls.append(name)

    monkeypatch.setattr(prep, "_run", run)
    with pytest.raises(RuntimeError, match="interrupted between"):
        prep.run_diagnostic(path)
    saved = Path(payload["output"]) / "DOID/population.txt.manifest.json"
    before = binding(saved)
    prep.run_diagnostic(path)
    assert binding(saved) == before
    report = read(Path(payload["output"]) / "preparation-receipt.json")
    assert len(report["populations"]) == 4 and report["hosted_requests"] == 0
    for case in report["cases"].values():
        local = read(case["local_queries"]["path"])
        assert local["original_query_rows"] == 2 and local["sources"] == 1
        assert local["labels_exposed"] is False
        lines = Path(local["outputs"]["queries"]["path"]).read_text().splitlines()
        assert len(lines) == 2
    # Completed rerun must not reload any ontology or rewrite query artifacts.
    import exact.ontology

    monkeypatch.setattr(
        exact.ontology, "load_ontology", lambda *a, **kw: pytest.fail("native rerun")
    )
    prep.run_diagnostic(path)


def test_protocol_rejects_extra_reference_or_non_strict_imports(tmp_path):
    path, payload = protocol(tmp_path)
    changed = copy.deepcopy(payload)
    changed["cases"]["H0"]["references"] = {"test": "/private/not-opened"}
    write(path, changed)
    with pytest.raises(ValueError, match="public inputs only"):
        prep.validate_protocol(path)
    changed = copy.deepcopy(payload)
    changed["ontologies"]["DOID"]["source_options"]["import_policy"] = "ignore"
    write(path, changed)
    with pytest.raises(ValueError, match="strict complete"):
        prep.validate_protocol(path)


def test_stop_and_memory_guard(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"
    stop.write_text("User pause")
    with pytest.raises(RuntimeError, match="pause/STOP"):
        prep._check_resources([stop], 100)
    monkeypatch.setattr(prep, "process_tree_rss_bytes", lambda: 101)
    with pytest.raises(RuntimeError, match="memory allowance"):
        prep._check_resources([], 100)
