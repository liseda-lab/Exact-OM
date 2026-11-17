"""Small offline ontology/run fixture for API and browser regression tests."""

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

# Allow direct execution without installing the repository's tests as a package.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact.runs import ExplanationStore, RunLayout, refresh_manifest
from exact.runs.decisions import append_event, event, observe_selection, write_candidate_decisions
from exact_inspect.cli import main
from exact_inspect.contracts import file_hash


def create_fixture(tmp_path):
    import pandas as pd
    import yaml

    for side in ("source", "target"):
        (tmp_path / f"{side}.ofn").write_text(f"""Ontology(<urn:{side}>
Declaration(Class(<urn:{side}:root>)) Declaration(Class(<urn:{side}:leaf>))
Declaration(NamedIndividual(<urn:{side}:leaf>))
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:{side}:root> "{side.title()} root")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:{side}:leaf> "{side.title()} leaf")
AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:{side}:leaf> "A leaf concept in the {side} ontology.")
SubClassOf(<urn:{side}:leaf> <urn:{side}:root>)
)""")
    run = tmp_path / "run"
    store = ExplanationStore.create(run)
    record = dict(
        src_iri="urn:source:leaf",
        tgt_iri="urn:target:leaf",
        src_kind="class",
        tgt_kind="class",
        confidences={"S_final": 0.81},
    )
    record["triple_attributions"] = {
        "hierarchy": {
            "is_a": {
                side: [
                    {
                        "subject_iri": f"urn:{side}:leaf",
                        "object_iri": f"urn:{side}:root",
                        "rel_iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                    }
                ]
                for side in ("source", "target")
            }
        }
    }
    store.append(
        [
            record,
            {
                **record,
                "src_kind": "individual",
                "tgt_kind": "individual",
                "triple_attributions": {},
            },
        ]
    )
    row = dict(
        Src=record["src_iri"],
        Tgt=record["tgt_iri"],
        SrcKind="class",
        TgtKind="class",
        S_final=0.81,
        candidate_joint_rank=1,
        selection_winner=True,
    )
    row["candidate_decision"] = append_event(
        row,
        event(
            "retrieval",
            implementation="local-test",
            reason="provided_candidate_pool",
            values={"cand_rank": 1},
        ),
    )
    individual = {**row, "SrcKind": "individual", "TgtKind": "individual"}
    frame = pd.DataFrame([row, individual])
    observe_selection(frame, implementation="local-test", config={})
    mappings = pd.DataFrame(
        [
            dict(
                SrcEntity=row["Src"],
                TgtEntity=row["Tgt"],
                Score=0.81,
                Relation="=",
                SrcKind="class",
                TgtKind="class",
            )
        ]
    )
    mappings = pd.concat(
        [mappings, mappings.assign(SrcKind="individual", TgtKind="individual")], ignore_index=True
    )
    layout = RunLayout.open(run)
    mappings.to_csv(layout.mapping_path("global"), sep="\t", index=False)
    layout.config_path.write_text(
        yaml.safe_dump(
            {side + "_hash": file_hash(tmp_path / f"{side}.ofn") for side in ("source", "target")}
        )
    )
    write_candidate_decisions(run, frame, mappings, policy={"relation_prediction": "none"})
    refresh_manifest(layout, run_id="local-smoke")
    (run / "component-fit-folds.json.lock").touch()
    snapshot = {str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file()}
    args = [
        "prepare",
        "--source-ontology",
        str(tmp_path / "source.ofn"),
        "--target-ontology",
        str(tmp_path / "target.ofn"),
        "--run-dir",
        str(run),
        "--output",
        str(tmp_path / "prepared"),
    ]
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        assert main(args) == 0
    path = Path(json.loads(captured.getvalue())["package"])
    assert snapshot == {
        str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file()
    }
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        assert main(args) == 0
    assert Path(json.loads(captured.getvalue())["package"]) == path
    return path


if __name__ == "__main__":
    import uvicorn
    from exact_inspect.app import create_app
    from exact_inspect.artifacts import export_archive
    from exact_inspect.settings import InspectSettings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--input-archive", type=Path, help="Serve an existing production-test export")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="exact-ui-fixture-") as directory:
        root = Path(directory)
        if args.input_archive:
            from exact_inspect.artifacts import BundleLibrary
            library = BundleLibrary(root / "library")
            imported = library.import_archive(args.input_archive)
            manifest = library.select(imported.package_id)
        else:
            manifest = create_fixture(root)
        export_archive(manifest, args.archive)
        try:
            uvicorn.run(
                create_app(InspectSettings(package=manifest, library_dir=root / "library")),
                host="127.0.0.1",
                port=args.port,
            )
        finally:
            args.archive.unlink(missing_ok=True)
