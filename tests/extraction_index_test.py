"""Global competitors, duplicate edges and exact tie parity across input chunks."""

import random

import pytest

from exact.core.entities.mappings.entity import EntityMapping
from exact.impl.extraction import extract_global_alignment


def signature(result):
    return ([(m.head, m.tail, m.src_kind, m.tgt_kind, m.score, m.relation)
             for m in result.mappings], result.diagnostics)


@pytest.mark.parametrize("mode,source_n,target_n", [
    ("greedy", 1, 1), ("greedy", 3, 2), ("greedy", None, 2),
    ("greedy", 2, None), ("greedy", None, None),
    ("mutual_best", 1, 1), ("threshold", None, None),
])
@pytest.mark.parametrize("threshold", [None, 0.5])
def test_indexed_global_competition_matches_memory(tmp_path, monkeypatch, mode, source_n, target_n, threshold):
    rng = random.Random(20261009)
    # Equal scores, duplicate typed IRIs, duplicate edges and conflicting anchors
    # force deterministic discrete decisions; tolerance cannot hide a wrong winner.
    mappings = [EntityMapping(
        "s" + str(rng.randrange(21)), "t" + str(rng.randrange(17)),
        relation=rng.choice(["=", "<"]), score=rng.choice([0., .4, .5, .9]),
        src_kind=rng.choice(["class", "object_property"]),
        tgt_kind=rng.choice(["class", "object_property"]),
    ) for _ in range(503)]
    kwargs = dict(mode=mode, threshold=threshold, source_cardinality=source_n,
                  target_cardinality=target_n, anchor_conflict_policy="compete",
                  protected_pairs={("s0", "t0"), ("s0", "t1"), ("s1", "t0"), ("s9", "t8")})
    monkeypatch.delenv("EXACT_EXTRACTION_SQLITE_DIR", raising=False)
    expected = signature(extract_global_alignment(mappings, **kwargs))
    monkeypatch.setenv("EXACT_EXTRACTION_SQLITE_DIR", str(tmp_path))
    assert signature(extract_global_alignment(mappings, **kwargs)) == expected
    assert list(tmp_path.iterdir()) == []
    # Ingest order and chunk size do not change the full global competition.
    rng.shuffle(mappings)
    assert signature(extract_global_alignment(mappings, **kwargs)) == expected


def test_index_cleanup_on_conflicting_protection(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_EXTRACTION_SQLITE_DIR", str(tmp_path))
    mappings = [EntityMapping("s", "t", score=1), EntityMapping("s", "u", score=1)]
    with pytest.raises(ValueError, match="Conflicting protected"):
        extract_global_alignment(mappings, protected_pairs={("s", "t"), ("s", "u")})
    assert list(tmp_path.iterdir()) == []
