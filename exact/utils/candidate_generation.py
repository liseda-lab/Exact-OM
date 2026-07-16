from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class CandidateLabel:
    iri: str
    text: str
    normalized: str
    tokens: Tuple[str, ...]
    grams: Tuple[str, ...]


def normalize_candidate_text(text: str) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    raw = re.sub(r"\b([A-Za-z0-9]+)'s\b", r"\1", raw)
    raw = re.sub(r"\b([A-Za-z0-9]+)s'\b", r"\1s", raw)
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    raw = raw.lower().replace("_", " ").replace("-", " ")
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def candidate_tokens(normalized: str) -> Tuple[str, ...]:
    return tuple(sorted({token for token in normalized.split() if token}))


def candidate_token_key(text: str) -> Tuple[str, ...]:
    normalized = normalize_candidate_text(text)
    return candidate_tokens(normalized)


def compact_candidate_text_key(text: str) -> str:
    raw = "" if text is None else str(text)
    return re.sub(r"[\W_]+", "", raw.lower())


def candidate_annotation_property_key(prop_iri: str) -> str:
    local = str(prop_iri or "").strip().rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return normalize_candidate_text(local)


def candidate_annotation_priority(literal: str, prop_iri: str) -> Optional[float]:
    text = str(literal or "").strip()
    normalized = normalize_candidate_text(text)
    if not normalized:
        return None
    tokens = [token for token in normalized.split() if token]
    if not tokens or len(tokens) > 12:
        return None
    if re.search(r"https?://|www\.", text, flags=re.IGNORECASE):
        return None
    if re.match(r"^[A-Za-z_]+:[A-Za-z0-9_.:-]+$", text):
        return None
    if len(tokens) <= 2 and normalized.endswith("ontology"):
        return None

    prop_text = str(prop_iri or "")
    prop_key = candidate_annotation_property_key(prop_text)
    prop_compact = compact_candidate_text_key(prop_text.rsplit("#", 1)[-1].rsplit("/", 1)[-1])
    prop_terms = set(prop_key.split())

    if prop_compact == "p90":
        base_priority = 0.0
    elif prop_compact in {"p107", "p108"}:
        base_priority = 0.05
    elif prop_key == "has exact synonym":
        base_priority = 0.0
    elif prop_key in {"has related synonym", "has narrow synonym", "has broad synonym"}:
        base_priority = 0.15
    else:
        rejected_terms = {
            "alternativeid",
            "comment",
            "created",
            "creator",
            "curator",
            "date",
            "definition",
            "deprecated",
            "description",
            "editor",
            "format",
            "id",
            "identifier",
            "namespace",
            "note",
            "provenance",
            "review",
            "reviewer",
            "semantic",
            "slim",
            "source",
            "status",
            "subset",
            "type",
            "version",
            "xref",
        }
        explicit_rejected = {
            "nhc0",
            "p97",
            "p106",
            "p207",
            "p322",
            "p325",
            "p363",
            "iao0000115",
            "iao0000231",
        }
        if prop_compact in explicit_rejected:
            return None
        if prop_terms.intersection(rejected_terms):
            return None
        alias_terms = {"alt", "label", "name", "pref", "synonym", "term", "title"}
        if not prop_terms.intersection(alias_terms):
            return None
        base_priority = 0.30

    length_penalty = abs(min(len(tokens), 6) - 3) / 30.0
    return float(base_priority + length_penalty)


def candidate_annotation_property_cap(prop_iri: str) -> int:
    prop_text = str(prop_iri or "")
    prop_key = candidate_annotation_property_key(prop_text)
    prop_compact = compact_candidate_text_key(prop_text.rsplit("#", 1)[-1].rsplit("/", 1)[-1])
    if prop_compact == "p90" or prop_key == "has exact synonym":
        return 8
    if prop_compact in {"p107", "p108"}:
        return 2
    if prop_key in {"has related synonym", "has narrow synonym", "has broad synonym"}:
        return 4
    return 4


def select_candidate_annotation_literals(
    annotations: Sequence[Tuple[str, str]],
    seen_normalized: Optional[Set[str]] = None,
    overall_cap: int = 12,
) -> List[str]:
    ranked: List[Tuple[float, str, int, str, str]] = []
    for prop_iri, literal in annotations:
        priority = candidate_annotation_priority(literal, prop_iri)
        if priority is None:
            continue
        prop_key = candidate_annotation_property_key(prop_iri)
        cap = candidate_annotation_property_cap(prop_iri)
        ranked.append(
            (
                float(priority),
                prop_key,
                int(cap),
                normalize_candidate_text(literal),
                str(literal).strip(),
            )
        )

    ranked.sort(key=lambda item: (item[0], item[1], item[3], item[4]))
    selected: List[str] = []
    selected_seen = set(seen_normalized or set())
    per_property_counts: Dict[str, int] = defaultdict(int)
    limit = max(0, int(overall_cap))
    for _, prop_key, cap, _, literal in ranked:
        if len(selected) >= limit:
            break
        normalized = compact_candidate_text_key(literal)
        if not normalized or normalized in selected_seen:
            continue
        if per_property_counts[prop_key] >= cap:
            continue
        selected_seen.add(normalized)
        per_property_counts[prop_key] += 1
        selected.append(literal)
    return selected


def candidate_char_grams(normalized: str, n: int = 3) -> Tuple[str, ...]:
    compact = re.sub(r"\s+", " ", normalized).strip()
    if not compact:
        return tuple()
    padded = f" {compact} "
    if len(padded) <= n:
        return (padded,)
    return tuple(sorted({padded[idx : idx + n] for idx in range(len(padded) - n + 1)}))


def make_candidate_labels(
    iris: Sequence[str],
    labels_by_iri: Mapping[str, Sequence[str]],
) -> List[CandidateLabel]:
    records: List[CandidateLabel] = []
    for iri in iris:
        seen = set()
        labels = list(labels_by_iri.get(iri) or [])
        if not labels:
            labels = [str(iri).rsplit("/", 1)[-1].rsplit("#", 1)[-1]]
        for label in labels:
            normalized = normalize_candidate_text(label)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            records.append(
                CandidateLabel(
                    iri=str(iri),
                    text=str(label),
                    normalized=normalized,
                    tokens=candidate_tokens(normalized),
                    grams=candidate_char_grams(normalized),
                )
            )
    return records


def lexical_candidate_pair_scores(
    src_records: Sequence[CandidateLabel],
    tgt_records: Sequence[CandidateLabel],
    per_source_limit: int,
) -> Dict[Tuple[str, str], float]:
    if not src_records or not tgt_records or per_source_limit <= 0:
        return {}

    token_index, token_df = _build_inverted_index([record.tokens for record in tgt_records])
    gram_index, gram_df = _build_inverted_index([record.grams for record in tgt_records])
    token_index = _drop_overly_common_features(token_index, len(tgt_records))
    gram_index = _drop_overly_common_features(gram_index, len(tgt_records))

    token_idf = _idf_by_feature(token_df, len(tgt_records), set(token_index))
    gram_idf = _idf_by_feature(gram_df, len(tgt_records), set(gram_index))
    tgt_token_norms = [_feature_norm(record.tokens, token_idf) for record in tgt_records]
    tgt_gram_norms = [_feature_norm(record.grams, gram_idf) for record in tgt_records]

    by_source: Dict[str, Dict[str, float]] = defaultdict(dict)
    for record in src_records:
        token_acc: Dict[int, float] = defaultdict(float)
        gram_acc: Dict[int, float] = defaultdict(float)
        for token in record.tokens:
            weight = token_idf.get(token)
            if weight is None:
                continue
            for tgt_idx in token_index.get(token, ()):
                token_acc[tgt_idx] += weight * weight
        for gram in record.grams:
            weight = gram_idf.get(gram)
            if weight is None:
                continue
            for tgt_idx in gram_index.get(gram, ()):
                gram_acc[tgt_idx] += weight * weight

        src_token_norm = _feature_norm(record.tokens, token_idf)
        src_gram_norm = _feature_norm(record.grams, gram_idf)
        candidate_indices = set(token_acc).union(gram_acc)
        for tgt_idx in candidate_indices:
            token_score = _safe_cosine(
                token_acc.get(tgt_idx, 0.0),
                src_token_norm,
                tgt_token_norms[tgt_idx],
            )
            gram_score = _safe_cosine(
                gram_acc.get(tgt_idx, 0.0),
                src_gram_norm,
                tgt_gram_norms[tgt_idx],
            )
            score = max(token_score, 0.85 * gram_score, (0.65 * token_score) + (0.35 * gram_score))
            if score <= 0.0:
                continue
            tgt_iri = tgt_records[tgt_idx].iri
            current = by_source[record.iri].get(tgt_iri, 0.0)
            if score > current:
                by_source[record.iri][tgt_iri] = float(min(1.0, score))

    return _limit_scores_by_source(by_source, per_source_limit)


def rank_channel_scores(
    sources: Sequence[str],
    semantic_scores: Mapping[Tuple[str, str], float],
    lexical_scores: Mapping[Tuple[str, str], float],
    top_k: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    by_source: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (src, tgt), score in semantic_scores.items():
        by_source[str(src)].setdefault(str(tgt), {})["semantic"] = float(score)
    for (src, tgt), score in lexical_scores.items():
        by_source[str(src)].setdefault(str(tgt), {})["lexical"] = float(score)

    for src in sources:
        entries = []
        for tgt, channel_scores in by_source.get(str(src), {}).items():
            semantic = float(channel_scores.get("semantic", 0.0))
            lexical = float(channel_scores.get("lexical", 0.0))
            score = max(semantic, lexical)
            channels = [
                name
                for name, value in (("semantic", semantic), ("lexical", lexical))
                if value > 0.0
            ]
            entries.append((tgt, score, semantic, lexical, "|".join(channels)))
        entries.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        for tgt, score, semantic, lexical, channels in entries[: max(0, int(top_k))]:
            rows.append(
                {
                    "Src": str(src),
                    "Tgt": str(tgt),
                    "Label": 0,
                    "cand_sim": float(score),
                    "cand_sim_semantic": float(semantic),
                    "cand_sim_lexical": float(lexical),
                    "cand_channels": channels,
                }
            )
    return rows


def _build_inverted_index(
    features_by_record: Sequence[Sequence[str]],
) -> Tuple[Dict[str, List[int]], Counter]:
    index: Dict[str, List[int]] = defaultdict(list)
    df: Counter = Counter()
    for idx, features in enumerate(features_by_record):
        unique_features = set(features)
        for feature in unique_features:
            index[feature].append(idx)
            df[feature] += 1
    return dict(index), df


def _drop_overly_common_features(
    index: Mapping[str, Sequence[int]], n_records: int
) -> Dict[str, List[int]]:
    max_df = max(10, int(math.ceil(0.20 * max(1, n_records))))
    return {feature: list(indices) for feature, indices in index.items() if len(indices) <= max_df}


def _idf_by_feature(df: Counter, n_records: int, allowed: Iterable[str]) -> Dict[str, float]:
    n = max(1, int(n_records))
    return {
        feature: math.log((1.0 + n) / (1.0 + float(df.get(feature, 0)))) + 1.0
        for feature in allowed
    }


def _feature_norm(features: Sequence[str], idf: Mapping[str, float]) -> float:
    value = sum(float(idf[feature]) ** 2 for feature in set(features) if feature in idf)
    return math.sqrt(value) if value > 0.0 else 0.0


def _safe_cosine(overlap_weight: float, left_norm: float, right_norm: float) -> float:
    denom = left_norm * right_norm
    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(overlap_weight) / denom))


def _limit_scores_by_source(
    scores_by_source: Mapping[str, Mapping[str, float]],
    per_source_limit: int,
) -> Dict[Tuple[str, str], float]:
    limited: Dict[Tuple[str, str], float] = {}
    limit = max(0, int(per_source_limit))
    for src, tgt_scores in scores_by_source.items():
        ranked = sorted(tgt_scores.items(), key=lambda item: (-item[1], item[0]))
        for tgt, score in ranked[:limit]:
            limited[(str(src), str(tgt))] = float(score)
    return limited
