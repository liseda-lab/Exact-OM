import importlib.util
import json
import sys
import types
from pathlib import Path

from exact.core.contracts.knowledge import KnowledgeSource

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


def _load_base_module(monkeypatch):
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    path = Path(__file__).parents[1] / "exact" / "impl" / "datasets" / "base.py"
    spec = importlib.util.spec_from_file_location("_exact_dataset_base_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_dataset_loads_knowledge_sources_without_runtime_java(tmp_path, monkeypatch):
    module = _load_base_module(monkeypatch)

    class BackendSmokeDataset(module.BaseAlignmentDataset):
        def __getitem__(self, idx):
            raise IndexError(idx)

        def __len__(self):
            return 0

        def get_features(self, df):
            return df

        def plot_feature_distributions(self, *args, **kwargs):
            return None

        def log_sanity_examples(self, *args, **kwargs):
            return None

    dataset = BackendSmokeDataset(
        output_path=tmp_path,
        filter_exact_matches=True,
        filter_ignored_alignment_classes=True,
        reasoner="asserted",
    )
    dataset.load_ontologies(
        FIXTURES / "mini_src.owl",
        FIXTURES / "mini_tgt.owl",
    )

    assert isinstance(dataset.source, KnowledgeSource)
    assert isinstance(dataset.target, KnowledgeSource)
    assert len(dataset.source_graph) == 42
    assert dataset.source_ignored_alignment_classes == {
        "http://example.org/mini/src#DeprecatedConcept",
        "http://example.org/mini/src#IgnoredConcept",
    }
    fingerprint = dataset._cache_fingerprint_payload()
    assert fingerprint["reasoner"] == "asserted"
    assert fingerprint["ontology_backend_version"] == 5
    assert fingerprint["projector"]["encoded_contract"]["core"]["descriptor_sha256"]
    assert fingerprint["projector"]["profile"] == "mowl-d993536-v1"

    provenance = dataset.ontology_stack_provenance()
    assert provenance["source"]["kind"] == "owl"
    assert provenance["target"]["kind"] == "owl"
    assert provenance["cache"] == {
        "schema_version": 4,
        "ontology_backend_version": 5,
        "state": "cold",
    }
    assert provenance["source"]["core"]["shared_snapshot"] is True
    assert provenance["source"]["reasoner"]["selection"]["effective"] == "asserted"

    dataset.get_exact_matches()
    exact_pairs = set(dataset.exact_matches[["Src", "Tgt"]].itertuples(index=False, name=None))
    assert (
        "http://example.org/mini/src#Heart",
        "http://example.org/mini/tgt#CardiacOrgan",
    ) in exact_pairs


def test_schema_1_dataset_cache_is_rejected_before_payload_interpretation(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_base_module(monkeypatch)

    class CacheBoundaryDataset(module.BaseAlignmentDataset):
        def __getitem__(self, idx):
            raise IndexError(idx)

        def __len__(self):
            return 0

        def get_features(self, df):
            return df

        def plot_feature_distributions(self, *args, **kwargs):
            return None

        def log_sanity_examples(self, *args, **kwargs):
            return None

    dataset = CacheBoundaryDataset(output_path=tmp_path, cache_ok=True)
    dataset._df_save_path.write_bytes(b"schema-1 dense-id payload must not be interpreted")
    dataset._cache_meta_path.write_text(
        json.dumps(
            {
                "cache_schema_version": 1,
                "ontology_backend_version": 1,
                "fingerprint": dataset.cache_fingerprint,
            }
        ),
        encoding="utf-8",
    )

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("schema-1 cache payload must not be interpreted")

    monkeypatch.setattr(module.pd, "read_csv", unexpected_read)

    assert dataset.has_cache() is False
    assert dataset.dataframe is None
    assert dataset.ontology_stack_provenance()["cache"] == {
        "schema_version": 4,
        "ontology_backend_version": 5,
        "state": "invalidated",
    }
    warning = capsys.readouterr().out.lower()
    assert "schema" in warning
    assert "rebuild" in warning
