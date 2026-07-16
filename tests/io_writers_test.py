import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

import exact.io.writers as writers_module
from exact.io.writers import WriterRegistry, write
from exact.io.writers.base import WriterOptionsError
from exact.io.writers.oaei_rdf import ALIGN_NS, RDF_NS, read_alignment
from exact.io.writers.typed_tsv import TYPED_COLUMNS, TYPED_RELATIONS
from tools.build_biokg_submission import build_submission

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mappings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Src": "http://source/B", "Tgt": "http://target/B", "Score": 0.4},
            {"Src": "http://source/A", "Tgt": "http://target/A", "Score": 0.9},
        ]
    )


def test_global_and_local_tsv_writers_are_legacy_byte_compatible(
    tmp_path: Path, mappings: pd.DataFrame
) -> None:
    global_path = write("tsv-global", mappings, tmp_path)
    assert global_path.name == "src2tgt.maps_global.tsv"
    assert global_path.read_bytes() == (
        b"SrcEntity\tTgtEntity\tScore\n"
        b"http://source/B\thttp://target/B\t0.4\n"
        b"http://source/A\thttp://target/A\t0.9\n"
    )

    local = pd.DataFrame(
        [
            {
                "SrcEntity": "http://source/A",
                "TgtEntity": "http://target/A",
                "TgtCandidates": "(('http://target/A', 0.9), ('http://target/B', 0.1))",
            }
        ]
    )
    local_path = write("tsv-local", local, tmp_path)
    assert local_path.read_bytes() == (
        b"SrcEntity\tTgtEntity\tTgtCandidates\n"
        b"http://source/A\thttp://target/A\t"
        b"(('http://target/A', 0.9), ('http://target/B', 0.1))\n"
    )


def test_oaei_rdf_writer_round_trips_with_legacy_namespace_parser(
    tmp_path: Path,
) -> None:
    mappings = pd.DataFrame(
        [
            {
                "SrcEntity": "http://source/A",
                "TgtEntity": "http://target/A",
                "Score": 0.875,
                "Relation": "=",
            },
            {
                "SrcEntity": "http://source/B",
                "TgtEntity": "http://target/Parent",
                "Score": 0.625,
                "Relation": "<",
            },
        ]
    )
    path = write(
        "oaei-rdf",
        mappings,
        tmp_path,
        options={"source_uri": "file:///source.owl", "target_uri": "file:///target.owl"},
    )
    namespace = {"al": ALIGN_NS, "rdf": RDF_NS}
    root = ET.parse(path).getroot()
    locations = [element.text for element in root.findall(".//al:location", namespace)]
    assert locations == ["file:///source.owl", "file:///target.owl"]
    cells = root.findall(".//al:Cell", namespace)
    assert len(cells) == 2
    assert cells[1].find("al:relation", namespace).text == "<"

    restored = read_alignment(path)
    pd.testing.assert_frame_equal(
        restored,
        mappings.reset_index(drop=True),
        check_dtype=False,
    )


def test_typed_tsv_maps_relations_validates_scores_and_sorts(
    tmp_path: Path,
) -> None:
    mappings = pd.DataFrame(
        [
            {
                "SrcEntity": "s2",
                "TgtEntity": "t2",
                "Relation": ">",
                "Score": 0.2,
            },
            {
                "SrcEntity": "s1",
                "TgtEntity": "t1",
                "Relation": "=",
                "Score": 0.9,
            },
            {
                "SrcEntity": "s3",
                "TgtEntity": "t3",
                "Relation": "<",
                "Score": 0.7,
            },
        ]
    )
    path = write("typed-tsv", mappings, tmp_path)
    typed = pd.read_csv(path, sep="\t")
    assert tuple(typed.columns) == TYPED_COLUMNS
    assert typed["Score"].tolist() == [0.9, 0.7, 0.2]
    assert set(typed["Relation"]) == TYPED_RELATIONS

    bad = mappings.copy()
    bad.loc[0, "Score"] = math.inf
    with pytest.raises(WriterOptionsError, match="finite"):
        write("typed-tsv", bad, tmp_path)
    bad = mappings.copy()
    bad.loc[0, "Relation"] = "related"
    with pytest.raises(WriterOptionsError, match="Unsupported relation"):
        write("typed-tsv", bad, tmp_path)


def test_json_writer_emits_normalized_programmatic_records(
    tmp_path: Path, mappings: pd.DataFrame
) -> None:
    mappings["SrcKind"] = "class"
    path = write("json", mappings, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == [
        {
            "Src": "http://source/B",
            "Tgt": "http://target/B",
            "Score": 0.4,
            "Relation": "=",
            "Kind": "class",
        },
        {
            "Src": "http://source/A",
            "Tgt": "http://target/A",
            "Score": 0.9,
            "Relation": "=",
            "Kind": "class",
        },
    ]


def test_writer_entry_point_contract(
    tmp_path: Path, mappings: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixtureWriter:
        name = "fixture-writer"
        default_filename = "fixture.txt"

        def write(self, values, path: Path, *, options=None) -> Path:
            assert len(values) == 2
            assert options == {"marker": "ok"}
            path.write_text("plugin\n", encoding="utf-8")
            return path

    class EntryPoint:
        name = "fixture-writer"

        @staticmethod
        def load():
            return FixtureWriter()

    class EntryPoints(list):
        def select(self, *, group: str):
            assert group == "exact.writers"
            return self

    monkeypatch.setattr(
        writers_module.metadata,
        "entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )
    registry = WriterRegistry()
    path = registry.write("fixture-writer", mappings, tmp_path, options={"marker": "ok"})
    assert path.read_text(encoding="utf-8") == "plugin\n"


def test_submission_builder_checks_candidates_and_concatenates(tmp_path: Path) -> None:
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    pd.DataFrame(
        [
            {
                "SrcEntity": "http://example.org/kg/atrium",
                "TgtEntity": "http://example.org/kg/atrium",
                "Relation": "equivalent",
                "Score": 0.9,
            }
        ],
        columns=TYPED_COLUMNS,
    ).to_csv(first, sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "SrcEntity": "s2",
                "TgtEntity": "t2",
                "Relation": "source_subsumes_target",
                "Score": 0.7,
            }
        ],
        columns=TYPED_COLUMNS,
    ).to_csv(second, sep="\t", index=False)
    output = build_submission(
        {"kg": first, "other": second},
        tmp_path / "submission.tsv",
        candidate_files={"kg": FIXTURES / "kg_csv" / "candidates.tsv"},
    )
    submission = pd.read_csv(output, sep="\t")
    assert tuple(submission.columns) == TYPED_COLUMNS
    assert submission["Score"].tolist() == [0.9, 0.7]

    outside = pd.read_csv(first, sep="\t")
    outside.loc[0, "TgtEntity"] = "not-a-candidate"
    outside.to_csv(first, sep="\t", index=False)
    with pytest.raises(WriterOptionsError, match="outside its candidate pool"):
        build_submission(
            {"kg": first},
            tmp_path / "invalid.tsv",
            candidate_files={"kg": FIXTURES / "kg_csv" / "candidates.tsv"},
        )
