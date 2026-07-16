from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import yaml

import exact.tracks as tracks_module
from exact.tracks import TrackRegistry, UserSuppliedFilesError
from exact.tracks.archive import safe_extract_zip
from exact.tracks.descriptor import TrackDescriptor
from exact.tracks.hf import HfProvider
from exact.tracks.http import DeclarativeHttpProvider
from exact.tracks.lockfile import LOCKFILE_NAME
from exact.tracks.provider import (
    IntegrityError,
    TaskLayout,
    TrackProvider,
    TrackStatus,
    VerificationReport,
)
from exact.tracks.sampling import create_validation_samples
from exact.tracks.transforms import (
    alignment_rdf_to_tsv,
    flatten_refs_equiv,
    pools_jsonl_to_cands_tsv,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tracks"


class FakeHfClient:
    def __init__(self, snapshot: Path, commit: str = "commit-a"):
        self.snapshot = snapshot
        self.commit = commit
        self.downloads: list[tuple[str, str]] = []

    def resolve_revision(self, repo_id: str, revision: str) -> str:
        return self.commit

    def snapshot_download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        *,
        allow_patterns: Sequence[str] | None = None,
        ignore_patterns: Sequence[str] | None = None,
    ) -> Path:
        self.downloads.append((repo_id, revision))
        return self.snapshot


class FakeHttpTransport:
    def __init__(self, bodies: Mapping[str, bytes], *, etag: str = '"fixture-a"'):
        self.bodies = dict(bodies)
        self.etag = etag

    def download(self, url: str, destination: Path) -> Mapping[str, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.bodies[url])
        return {"ETag": self.etag, "Last-Modified": "Wed, 15 Jul 2026 12:00:00 GMT"}

    def head(self, url: str) -> Mapping[str, str]:
        return {"etag": self.etag, "last-modified": "Wed, 15 Jul 2026 12:00:00 GMT"}


def _hf_descriptor(*, licensed: bool = False) -> TrackDescriptor:
    task = {
        "source": "licensed/source.owl" if licensed else "source.owl",
        "target": "target.owl",
        "refs": {
            "test": {
                "path": "references/test.rdf",
                "transform": "alignment_rdf_to_tsv",
            }
        },
        "candidates": {
            "path": "pools/pools.jsonl",
            "transform": "pools_jsonl_to_cands_tsv",
        },
        "extras": {"edition": "mini"},
    }
    descriptor = {
        "descriptor_version": 1,
        "name": "fake-hf-licensed" if licensed else "fake-hf",
        "provider": "hf",
        "provider_version": "test-1",
        "upstream": {
            "repo_id": "fixture/example",
            "revision": "stable",
            "checksum_manifest": "SHA256SUMS",
        },
        "tasks": {"demo": task},
    }
    if licensed:
        descriptor["user_supplied"] = {
            "source": {
                "path": "source.owl",
                "destination": "licensed/source.owl",
                "sha256_from": {
                    "path": "resolved_versions.json",
                    "key": "ontologies.source.sha256",
                },
                "help": "Obtain the licensed source ontology from its publisher.",
            }
        }
        task["user_supplied"] = ["source"]
    return TrackDescriptor.from_mapping(descriptor)


def _archive_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for path in sorted((FIXTURES / "http_archive").iterdir()):
            archive.writestr(path.name, path.read_bytes())
    return stream.getvalue()


def _http_descriptor(archive: bytes) -> TrackDescriptor:
    return TrackDescriptor.from_mapping(
        {
            "descriptor_version": 1,
            "name": "fake-http",
            "provider": "http",
            "provider_version": "test-1",
            "upstream": {
                "revision": "fixture",
                "urls": [
                    {
                        "url": "https://example.invalid/conference.zip",
                        "filename": "conference.zip",
                        "sha256": hashlib.sha256(archive).hexdigest(),
                        "extract": True,
                        "destination": "conference",
                    }
                ],
            },
            "tasks": {
                "demo": {
                    "source": "conference/source.owl",
                    "target": "conference/target.owl",
                    "refs": {
                        "test": {
                            "path": "conference/reference.rdf",
                            "transform": "alignment_rdf_to_tsv",
                        }
                    },
                }
            },
        }
    )


def test_hf_materialize_lock_verify_drift_and_update(tmp_path: Path) -> None:
    client = FakeHfClient(FIXTURES / "hf_snapshot")
    provider = HfProvider(_hf_descriptor(), client=client)

    assert provider.status("demo", tmp_path) == "not-materialized"
    layout = provider.materialize("demo", tmp_path)

    assert layout.source.read_text(encoding="utf-8").startswith("<?xml")
    assert layout.refs["test"].read_text(encoding="utf-8").splitlines()[0] == (
        "SrcEntity\tTgtEntity\tScore"
    )
    candidates = layout.candidates.read_text(encoding="utf-8")
    assert "/NIL" not in candidates
    assert layout.extras["nil"] is True
    assert layout.provenance["revision"] == "commit-a"
    assert provider.verify("demo", tmp_path).status == "ok"
    with pytest.raises(IntegrityError, match="use update=True"):
        provider.materialize("demo", tmp_path, revision="different-tag")

    lock = json.loads((tmp_path / LOCKFILE_NAME).read_text(encoding="utf-8"))
    entry = lock["tasks"]["fake-hf/demo"]
    assert entry["provider"] == "fake-hf"
    assert entry["upstream_manifest_hashes"]
    assert all(record["sha256"] for record in entry["files"].values())

    layout.source.write_text("corrupt", encoding="utf-8")
    assert provider.status("demo", tmp_path) == "local-drift"
    repaired = provider.materialize("demo", tmp_path, update=True)
    assert repaired.source.read_text(encoding="utf-8").startswith("<?xml")
    assert provider.status("demo", tmp_path) == "ok"

    client.commit = "commit-b"
    assert provider.status("demo", tmp_path) == "upstream-moved"
    pinned = provider.materialize("demo", tmp_path)
    assert pinned.provenance["revision"] == "commit-a"
    updated = provider.materialize("demo", tmp_path, update=True)
    assert updated.provenance["revision"] == "commit-b"
    assert provider.status("demo", tmp_path) == "ok"


def test_missing_and_mismatched_user_supplied_file_are_actionable(tmp_path: Path) -> None:
    client = FakeHfClient(FIXTURES / "hf_snapshot")
    provider = HfProvider(_hf_descriptor(licensed=True), client=client)

    with pytest.raises(UserSuppliedFilesError, match="place .*source.owl.*publisher"):
        provider.materialize("demo", tmp_path)

    supplied = tmp_path / "user_supplied" / provider.name / "source.owl"
    supplied.parent.mkdir(parents=True)
    supplied.write_text("licensed ontology", encoding="utf-8")
    with pytest.warns(UserWarning, match="differs from its published pin"):
        layout = provider.materialize("demo", tmp_path)

    assert layout.source.read_text(encoding="utf-8") == "licensed ontology"
    report = provider.verify("demo", tmp_path)
    assert report.status == "ok"
    assert any("published pin" in message for message in report.warnings)


def test_licensed_file_pin_is_resolved_from_snapshot_manifest(tmp_path: Path) -> None:
    client = FakeHfClient(FIXTURES / "hf_snapshot")
    provider = HfProvider(_hf_descriptor(licensed=True), client=client)
    supplied = tmp_path / "user_supplied" / provider.name / "source.owl"
    supplied.parent.mkdir(parents=True)
    supplied.write_text("licensed ontology\n", encoding="utf-8")

    layout = provider.materialize("demo", tmp_path)
    report = provider.verify("demo", tmp_path)

    assert layout.source.read_bytes() == supplied.read_bytes()
    assert report.status == "ok"
    assert report.warnings == ()
    declared = report.lock_entry["declared_hashes"]
    source_record = next(record for path, record in declared.items() if path.endswith("source.owl"))
    assert source_record == {
        "sha256": "3e3d1d0f32ad127387c94be537117fe47f496e1f467bdbf6debddd7551e41156",
        "enforce": False,
        "origin": "user-supplied pin source",
    }


def test_http_archive_materialization_and_upstream_movement(tmp_path: Path) -> None:
    archive = _archive_bytes()
    transport = FakeHttpTransport({"https://example.invalid/conference.zip": archive})
    provider = DeclarativeHttpProvider(_http_descriptor(archive), transport=transport)

    layout = provider.materialize("demo", tmp_path)

    assert layout.source.read_text(encoding="utf-8") == "source ontology fixture\n"
    assert provider.status("demo", tmp_path) == "ok"
    transport.etag = '"fixture-b"'
    assert provider.status("demo", tmp_path) == "upstream-moved"
    changed_stream = io.BytesIO(archive)
    with zipfile.ZipFile(changed_stream, "a") as changed_archive:
        changed_archive.writestr("release-note.txt", "updated")
    transport.bodies["https://example.invalid/conference.zip"] = changed_stream.getvalue()

    updated = provider.materialize("demo", tmp_path, update=True)

    assert updated.provenance["revision"] != layout.provenance["revision"]
    report = provider.verify("demo", tmp_path)
    assert report.status == "ok"
    assert any("explicit update repin" in warning for warning in report.warnings)


def test_alignment_transform_accepts_anatomy_rdf_without_locations(tmp_path: Path) -> None:
    source = tmp_path / "reference.rdf"
    source.write_text(
        """<?xml version="1.0"?>
<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <Alignment><map><Cell>
    <entity1 rdf:resource="http://mouse/A"/>
    <entity2 rdf:resource="http://human/B"/>
    <measure>1.0</measure><relation>=</relation>
  </Cell></map></Alignment>
</rdf:RDF>
""",
        encoding="utf-8",
    )

    result = alignment_rdf_to_tsv(source, tmp_path / "reference.tsv")

    assert result.path.read_text(encoding="utf-8") == (
        "SrcEntity\tTgtEntity\tScore\nhttp://mouse/A\thttp://human/B\t1.0\n"
    )


def test_custom_yaml_descriptor_materializes_without_provider_code(tmp_path: Path) -> None:
    archive = _archive_bytes()
    value = {
        "descriptor_version": 1,
        "name": "custom-yaml",
        "provider": "http",
        "upstream": {
            "urls": [
                {
                    "url": "https://example.invalid/custom.zip",
                    "filename": "custom.zip",
                    "sha256": hashlib.sha256(archive).hexdigest(),
                    "extract": True,
                    "destination": "payload",
                }
            ]
        },
        "tasks": {
            "only": {
                "source": "payload/source.owl",
                "target": "payload/target.owl",
                "refs": {"test": "payload/reference.rdf"},
            }
        },
    }
    descriptor_path = tmp_path / "custom.yaml"
    descriptor_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    descriptor = TrackDescriptor.load(descriptor_path)
    provider = DeclarativeHttpProvider(
        descriptor,
        transport=FakeHttpTransport({"https://example.invalid/custom.zip": archive}),
    )

    layout = provider.materialize("only", tmp_path / "data")

    assert layout.source.is_file()
    assert layout.target.is_file()
    assert set(layout.refs) == {"test"}


def test_transforms_cover_alignment_nil_pools_and_flattening(tmp_path: Path) -> None:
    alignment = alignment_rdf_to_tsv(
        FIXTURES / "http_archive" / "reference.rdf", tmp_path / "reference.tsv"
    )
    assert alignment.path.read_text(encoding="utf-8").splitlines()[-1] == (
        "https://example.org/source#A\thttps://example.org/target#A\t1.0"
    )

    candidates = pools_jsonl_to_cands_tsv(
        FIXTURES / "hf_snapshot" / "pools" / "pools.jsonl",
        tmp_path / "candidates.tsv",
    )
    assert candidates.extras["nil"] is True
    assert "NIL" not in candidates.path.read_text(encoding="utf-8")

    flattened = flatten_refs_equiv(FIXTURES / "refs_equiv", tmp_path / "flat")
    assert sorted(path.name for path in flattened.path.iterdir()) == ["test.tsv", "train.tsv"]


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", "unsafe")

    with pytest.raises(IntegrityError, match="unsafe member"):
        safe_extract_zip(archive_path, tmp_path / "output")

    assert not (tmp_path / "escaped.txt").exists()


def test_builtin_registry_and_aliases_are_declarative() -> None:
    registry = TrackRegistry(discover_plugins=False)

    assert registry.names() == [
        "anatomy",
        "biokg",
        "bioml_hf",
        "bioml_zenodo",
        "conference",
        "diso",
        "oaei_kg",
    ]
    assert registry.get("bioml").name == "bioml_hf"
    assert len(registry.get("conference").tasks()) == 21
    assert isinstance(registry.get("diso"), TrackProvider)


def test_entry_point_provider_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    class PluginProvider:
        name = "plugin-track"

        def tasks(self) -> list[str]:
            return ["task"]

        def materialize(
            self,
            task: str,
            data_root: Path,
            *,
            revision: str | None = None,
            update: bool = False,
        ) -> TaskLayout:
            raise AssertionError("not used")

        def verify(self, task: str, data_root: Path) -> VerificationReport:
            return VerificationReport(self.name, task, "not-materialized")

        def status(self, task: str, data_root: Path) -> TrackStatus:
            return "not-materialized"

    class EntryPoint:
        name = "plugin-track"

        @staticmethod
        def load() -> PluginProvider:
            return PluginProvider()

    class EntryPoints(list):
        def select(self, *, group: str) -> "EntryPoints":
            assert group == "exact.tracks"
            return self

    monkeypatch.setattr(tracks_module.metadata, "entry_points", lambda: EntryPoints([EntryPoint()]))
    registry = TrackRegistry(load_builtins=False)

    assert registry.names() == ["plugin-track"]
    assert registry.get("plugin-track").tasks() == ["task"]


def test_validation_sampling_moved_from_legacy_script(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = dataset / "test.cands.tsv"
    source.write_text(
        "SrcEntity\tTgtEntity\tTgtCandidates\n" "s1\tt1\t('t1',)\n" "s2\tt2\t('t2',)\n",
        encoding="utf-8",
    )

    outputs = create_validation_samples(0.5, 7, [dataset])

    assert outputs == [dataset / "test.cands.val.tsv"]
    assert len(outputs[0].read_text(encoding="utf-8").splitlines()) == 2
