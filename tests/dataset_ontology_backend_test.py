import importlib.util
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
    assert fingerprint["ontology_backend_version"] == 3
    assert fingerprint["projector"]["profile"] == "mowl-d993536-v1"

    provenance = dataset.ontology_stack_provenance()
    assert provenance["source"]["kind"] == "owl"
    assert provenance["target"]["kind"] == "owl"
    assert provenance["source"]["core"]["shared_snapshot"] is True
    assert provenance["source"]["reasoner"]["selection"]["effective"] == "asserted"

    dataset.get_exact_matches()
    exact_pairs = set(dataset.exact_matches[["Src", "Tgt"]].itertuples(index=False, name=None))
    assert (
        "http://example.org/mini/src#Heart",
        "http://example.org/mini/tgt#CardiacOrgan",
    ) in exact_pairs
