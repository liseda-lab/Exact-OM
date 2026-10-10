from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from exact.impl.retrieval.training import fit_retrieval_artifact, mine_training_examples


def pool():
    return pd.DataFrame(
        [
            {
                "Src": f"s{i}",
                "Tgt": f"t{i}-{j}",
                "src_text": str(i),
                "tgt_text": str(i if j < 2 else i + 4),
                "cand_sim": 1.0 - j * 0.1,
            }
            for i in range(6)
            for j in range(3)
        ]
    )


def refs():
    return {(f"s{i}", f"t{i}-{j}") for i in range(6) for j in (0, 1)}


APPLICATION = {"negative_label_policy": "complete_reference", "source_ids": ["report"]}


def test_mining_excludes_all_alternative_positives_and_unknown_negatives():
    mined = mine_training_examples(pool(), refs(), application=APPLICATION)
    assert all(
        (row["source"], row["negative"]) not in refs()
        for row in mined["train"] + mined["validation"]
    )
    assert not {row["source"] for row in mined["train"]} & {
        row["source"] for row in mined["validation"]
    }
    with pytest.raises(ValueError, match="positive-unlabelled"):
        mine_training_examples(
            pool(), refs(), application={**APPLICATION, "negative_label_policy": "unknown"}
        )
    confirmed = pool()
    confirmed["confirmed_label"] = [0 if target.endswith("2") else None for target in confirmed.Tgt]
    known = mine_training_examples(
        confirmed,
        refs(),
        application={**APPLICATION, "negative_label_policy": "confirmed_negatives"},
    )
    assert known["negative_label_policy"] == "confirmed_negatives"
    with pytest.raises(ValueError, match="overlap"):
        mine_training_examples(pool(), refs(), application={**APPLICATION, "source_ids": ["s1"]})


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)

    def tokenize(self, texts):
        return {"ids": torch.tensor([int(text) for text in texts])}

    def forward(self, features):
        return {"sentence_embedding": self.embedding(features["ids"])}

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")


class TinyCrossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, features):
        return SimpleNamespace(logits=self.linear(features.float()))


class TinyCross:
    def __init__(self):
        self.model = TinyCrossModel()

    def tokenizer(self, left, right, **kwargs):
        return {"features": torch.tensor([[int(a), int(b)] for a, b in zip(left, right)])}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(path) / "model.pt")


@pytest.mark.parametrize(
    "kind,builder", [("contrastive_encoder", TinyEncoder), ("cross_encoder", TinyCross)]
)
def test_real_optimizer_checkpoint_resume_matches_uninterrupted(tmp_path, kind, builder):
    def fitted(path, bundle, stop=None):
        return fit_retrieval_artifact(
            pool(),
            refs(),
            path,
            kind=kind,
            base_model="tiny-local",
            revision="fixture-v1",
            application=APPLICATION,
            epochs=1,
            max_steps=3,
            batch_size=2,
            accumulation=1,
            checkpoint_steps=1,
            patience=3,
            device="cpu",
            model_bundle=bundle,
            stop_after_steps=stop,
        )

    torch.manual_seed(12)
    uninterrupted = fitted(tmp_path / "whole", builder())
    torch.manual_seed(12)
    with pytest.raises(InterruptedError, match="committed"):
        fitted(tmp_path / "resume", builder(), stop=1)
    checkpoint = tmp_path / "resume" / "checkpoints" / "checkpoint-1"
    assert (checkpoint / "optimizer.pt").exists()
    assert (checkpoint / "rng_state.pth").exists()
    assert (checkpoint / "committed.json").exists()
    torch.manual_seed(91)
    resumed = fitted(tmp_path / "resume", builder())
    assert resumed.metadata["completed_steps"] == uninterrupted.metadata["completed_steps"] == 3
    a = torch.load(uninterrupted.model_path / "model.pt", weights_only=True)
    b = torch.load(resumed.model_path / "model.pt", weights_only=True)
    assert a.keys() == b.keys()
    assert all(torch.equal(a[name], b[name]) for name in a)
    assert not set(resumed.metadata["validation_sources"]) & {
        row["source"] for row in resumed.metadata["training_pairs"]
    }


@pytest.mark.parametrize("top_k", [1, 2])
def test_fixed_pool_cross_encoder_preserves_candidates(tmp_path, monkeypatch, top_k):
    from tests.retrieval_experiments_test import SRC, _artifact, _dataset

    dataset = _dataset(tmp_path, monkeypatch, name="fixed-cross")
    target = "http://example.org/mini/tgt#"
    frame = pd.DataFrame(
        {
            "Src": [SRC + "Person", SRC + "Person"],
            "Tgt": [target + "Human", target + "Disease"],
            "cand_sim": [0.8, 0.7],
        }
    )
    path = tmp_path / "pool.tsv"
    frame.to_csv(path, sep="\t", index=False)
    artifact = _artifact(tmp_path / "cross", kind="cross_encoder")
    dataset.load_candidates(
        path,
        cross_encoder={"mode": "on", "artifact": artifact, "top_k": top_k},
        device=torch.device("cpu"),
    )
    assert set(dataset.candidates[["Src", "Tgt"]].itertuples(index=False, name=None)) == set(
        frame[["Src", "Tgt"]].itertuples(index=False, name=None)
    )
    assert "cand_sim_cross_encoder" in dataset.candidates
    assert dataset.candidate_pool_manifest["origin"] == "provided"


def test_supplied_cross_encoder_keeps_large_overlapping_query_membership(tmp_path, monkeypatch):
    from exact.core.entities.kinds import EntityKind
    from exact.utils.mappings import candidate_table_views
    from tests.retrieval_experiments_test import SRC, _artifact, _dataset

    dataset = _dataset(tmp_path, monkeypatch, name="large-fixed-cross")
    targets = sorted(dataset.target.entities(EntityKind.CLASS))
    assert len(targets) >= 23
    # Query 1 itself exceeds the generated retrieval cap; queries 2 and 3
    # repeat a source with distinct membership, overlap, and a repeated query.
    queries = pd.DataFrame(
        [(SRC + "Person", "urn:query:1", repr(targets[:21])),
         (SRC + "Person", "urn:query:2", repr(targets[19:23])),
         (SRC + "Person", "urn:query:2", repr(targets[19:23])),
         (SRC + "Patient", "urn:query:3", repr(targets[:2]))],
        columns=["Src", "Tgt", "Candidates"],
    )
    path = tmp_path / "local-queries.tsv"
    queries.to_csv(path, sep="\t", index=False)
    original_bytes = path.read_bytes()
    expected_pairs, expected_queries = candidate_table_views(queries)
    artifact = _artifact(tmp_path / "cross", kind="cross_encoder")

    dataset.load_candidates(
        path, cross_encoder={"mode": "on", "artifact": artifact, "top_k": 20},
        encode_batch_size=3, device=torch.device("cpu"),
    )

    actual_pairs = dataset.candidates[["Src", "Tgt"]]
    assert set(actual_pairs.itertuples(index=False, name=None)) == set(
        expected_pairs[["Src", "Tgt"]].itertuples(index=False, name=None)
    )
    assert len(actual_pairs) == len(expected_pairs) == 25
    assert dataset.candidates.cand_sim_cross_encoder.notna().all()
    assert path.read_bytes() == original_bytes
    _, writer_queries = candidate_table_views(pd.read_csv(path, sep="\t"))
    pd.testing.assert_frame_equal(writer_queries, expected_queries)


def test_generated_cross_encoder_keeps_retrieval_cap(tmp_path, monkeypatch):
    from tests.retrieval_experiments_test import _artifact, _dataset

    dataset = _dataset(tmp_path, monkeypatch, name="bounded-global-cross")
    artifact = _artifact(tmp_path / "cross", kind="cross_encoder")
    with pytest.raises(ValueError, match="cross_encoder.top_k must be at least"):
        dataset.generate_candidates(
            top_k=2, cross_encoder={"mode": "on", "artifact": artifact, "top_k": 1},
            device=torch.device("cpu"), use_amp=False,
        )

    dataset.generate_candidates(
        top_k=1, cross_encoder={"mode": "on", "artifact": artifact, "top_k": 2},
        device=torch.device("cpu"), use_amp=False,
    )
    assert not dataset.candidates.empty
    assert dataset.candidates.groupby("Src").size().max() == 1
    assert dataset.candidate_pool_manifest["origin"] == "generated"
