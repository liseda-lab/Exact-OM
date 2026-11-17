"""Deterministic label normalization and hybrid candidate-retrieval helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from exact.core.entities.kinds import EntityKind

_DEFAULT_REJECTED_ALIAS_TERMS = {
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

_DEFAULT_REJECTED_ALIAS_PROPERTIES = {
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


def _option(options: Optional[Mapping[str, Any]], name: str, default: Any) -> Any:
    return default if options is None or name not in options else options[name]


@dataclass(frozen=True)
class CandidateLabel:
    """Normalized label features associated with one kind-aware entity IRI."""

    iri: str
    text: str
    normalized: str
    tokens: Tuple[str, ...]
    grams: Tuple[str, ...]
    kind: EntityKind = EntityKind.CLASS


def normalize_candidate_text(text: str) -> str:
    """Normalize a label into lowercase, accent-free alphanumeric tokens."""

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
    """Return the sorted unique tokens from normalized candidate text."""

    return tuple(sorted({token for token in normalized.split() if token}))


def candidate_token_key(text: str) -> Tuple[str, ...]:
    """Return a stable token key for an arbitrary candidate label."""

    normalized = normalize_candidate_text(text)
    return candidate_tokens(normalized)


def compact_candidate_text_key(text: str) -> str:
    """Return a punctuation-free lowercase key for alias deduplication."""

    raw = "" if text is None else str(text)
    return re.sub(r"[\W_]+", "", raw.lower())


def candidate_annotation_property_key(prop_iri: str) -> str:
    """Normalize the local name of an annotation-property IRI."""

    local = str(prop_iri or "").strip().rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return normalize_candidate_text(local)


def candidate_annotation_priority(
    literal: str,
    prop_iri: str,
    alias_config: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    """Return an alias preference score, or ``None`` when it should be excluded."""

    text = str(literal or "").strip()
    normalized = normalize_candidate_text(text)
    if not normalized:
        return None
    tokens = [token for token in normalized.split() if token]
    if not tokens or len(tokens) > int(_option(alias_config, "max_tokens", 12)):
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

    exact_properties = set(_option(alias_config, "exact_properties", ["p90", "has exact synonym"]))
    preferred_properties = set(_option(alias_config, "preferred_properties", ["p107", "p108"]))
    related_properties = set(
        _option(
            alias_config,
            "related_properties",
            ["has related synonym", "has narrow synonym", "has broad synonym"],
        )
    )
    if prop_compact in exact_properties or prop_key in exact_properties:
        base_priority = float(_option(alias_config, "exact_priority", 0.0))
    elif prop_compact in preferred_properties or prop_key in preferred_properties:
        base_priority = float(_option(alias_config, "preferred_priority", 0.05))
    elif prop_compact in related_properties or prop_key in related_properties:
        base_priority = float(_option(alias_config, "related_priority", 0.15))
    else:
        rejected_terms = set(
            _option(alias_config, "rejected_property_terms", _DEFAULT_REJECTED_ALIAS_TERMS)
        )
        explicit_rejected = set(
            _option(
                alias_config,
                "rejected_properties",
                _DEFAULT_REJECTED_ALIAS_PROPERTIES,
            )
        )
        if prop_compact in explicit_rejected:
            return None
        if prop_terms.intersection(rejected_terms):
            return None
        alias_terms = set(
            _option(
                alias_config,
                "accepted_property_terms",
                ["alt", "label", "name", "pref", "synonym", "term", "title"],
            )
        )
        if not prop_terms.intersection(alias_terms):
            return None
        base_priority = float(_option(alias_config, "default_priority", 0.30))

    length_cap = int(_option(alias_config, "priority_length_cap", 6))
    length_target = int(_option(alias_config, "priority_length_target", 3))
    length_divisor = float(_option(alias_config, "priority_length_divisor", 30.0))
    length_penalty = abs(min(len(tokens), length_cap) - length_target) / length_divisor
    return float(base_priority + length_penalty)


def candidate_annotation_property_cap(
    prop_iri: str, alias_config: Optional[Mapping[str, Any]] = None
) -> int:
    """Return the configured per-property alias limit for ``prop_iri``."""

    prop_text = str(prop_iri or "")
    prop_key = candidate_annotation_property_key(prop_text)
    prop_compact = compact_candidate_text_key(prop_text.rsplit("#", 1)[-1].rsplit("/", 1)[-1])
    exact = set(_option(alias_config, "exact_properties", ["p90", "has exact synonym"]))
    preferred = set(_option(alias_config, "preferred_properties", ["p107", "p108"]))
    related = set(
        _option(
            alias_config,
            "related_properties",
            ["has related synonym", "has narrow synonym", "has broad synonym"],
        )
    )
    if prop_compact in exact or prop_key in exact:
        return int(_option(alias_config, "exact_property_cap", 8))
    if prop_compact in preferred or prop_key in preferred:
        return int(_option(alias_config, "preferred_property_cap", 2))
    if prop_compact in related or prop_key in related:
        return int(_option(alias_config, "related_property_cap", 4))
    return int(_option(alias_config, "default_property_cap", 4))


def select_candidate_annotation_literals(
    annotations: Sequence[Tuple[str, str]],
    seen_normalized: Optional[Set[str]] = None,
    overall_cap: int = 12,
    alias_config: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Select deterministic, deduplicated annotation aliases within configured caps."""

    ranked: List[Tuple[float, str, int, str, str]] = []
    for prop_iri, literal in annotations:
        priority = candidate_annotation_priority(literal, prop_iri, alias_config)
        if priority is None:
            continue
        prop_key = candidate_annotation_property_key(prop_iri)
        cap = candidate_annotation_property_cap(prop_iri, alias_config)
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
    """Return sorted unique boundary-aware character n-grams."""

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
    *,
    kind: EntityKind | str = EntityKind.CLASS,
) -> List[CandidateLabel]:
    """Build normalized label records for a kind-local entity pool."""

    normalized_kind = EntityKind(kind)
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
                    kind=normalized_kind,
                )
            )
    return records


def lexical_candidate_pair_scores(
    src_records: Sequence[CandidateLabel],
    tgt_records: Sequence[CandidateLabel],
    per_source_limit: int,
    fusion_config: Optional[Mapping[str, Any]] = None,
) -> Dict[Tuple[str, str], float]:
    """Score lexical source-target candidates without crossing entity kinds."""

    if not src_records or not tgt_records or per_source_limit <= 0:
        return {}

    scores: Dict[Tuple[str, str], float] = {}
    kinds = list(dict.fromkeys(record.kind for record in src_records))
    for kind in kinds:
        src_for_kind = [record for record in src_records if record.kind == kind]
        tgt_for_kind = [record for record in tgt_records if record.kind == kind]
        scores.update(
            _lexical_candidate_pair_scores_one_kind(
                src_for_kind,
                tgt_for_kind,
                per_source_limit,
                fusion_config,
            )
        )
    return scores


def _lexical_candidate_pair_scores_one_kind(
    src_records: Sequence[CandidateLabel],
    tgt_records: Sequence[CandidateLabel],
    per_source_limit: int,
    fusion_config: Optional[Mapping[str, Any]] = None,
) -> Dict[Tuple[str, str], float]:
    """Score one kind-local lexical index."""

    if not src_records or not tgt_records:
        return {}

    token_index, token_df = _build_inverted_index([record.tokens for record in tgt_records])
    gram_index, gram_df = _build_inverted_index([record.grams for record in tgt_records])
    token_index = _drop_overly_common_features(token_index, len(tgt_records), fusion_config)
    gram_index = _drop_overly_common_features(gram_index, len(tgt_records), fusion_config)

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
            token_weight = float(_option(fusion_config, "token_weight", 1.0))
            gram_weight = float(_option(fusion_config, "gram_weight", 0.85))
            blend_token_weight = float(_option(fusion_config, "blend_token_weight", 0.65))
            blend_gram_weight = float(_option(fusion_config, "blend_gram_weight", 0.35))
            score = max(
                token_weight * token_score,
                gram_weight * gram_score,
                (blend_token_weight * token_score) + (blend_gram_weight * gram_score),
            )
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
    fusion_config: Optional[Mapping[str, Any]] = None,
    adaptive_config: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, object]]:
    """Fuse retrieval channels and apply a deterministic per-source pool limit."""

    rows: List[Dict[str, object]] = []
    by_source: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (src, tgt), score in semantic_scores.items():
        by_source[str(src)].setdefault(str(tgt), {})["semantic"] = float(score)
    for (src, tgt), score in lexical_scores.items():
        by_source[str(src)].setdefault(str(tgt), {})["lexical"] = float(score)

    mode = str(_option(fusion_config, "mode", "max") or "max").lower()
    if mode not in {"max", "rrf", "weighted"}:
        raise ValueError(f"Unsupported candidate fusion mode: {mode!r}")
    semantic_ranks = _channel_ranks(semantic_scores) if mode == "rrf" else {}
    lexical_ranks = _channel_ranks(lexical_scores) if mode == "rrf" else {}

    for src in sources:
        entries = []
        for tgt, channel_scores in by_source.get(str(src), {}).items():
            semantic = float(channel_scores.get("semantic", 0.0))
            lexical = float(channel_scores.get("lexical", 0.0))
            if mode == "max":
                semantic_weight = float(_option(fusion_config, "semantic_channel_weight", 1.0))
                lexical_weight = float(_option(fusion_config, "lexical_channel_weight", 1.0))
                score = max(semantic_weight * semantic, lexical_weight * lexical)
            elif mode == "weighted":
                semantic_weight = float(_option(fusion_config, "weighted_semantic", 0.5))
                lexical_weight = float(_option(fusion_config, "weighted_lexical", 0.5))
                weight_total = semantic_weight + lexical_weight
                if semantic_weight < 0.0 or lexical_weight < 0.0 or weight_total <= 0.0:
                    raise ValueError(
                        "weighted candidate fusion requires non-negative shares "
                        "with a positive total"
                    )
                score = ((semantic_weight * semantic) + (lexical_weight * lexical)) / weight_total
            else:
                rrf_constant = float(_option(fusion_config, "rrf_constant", 60.0))
                if rrf_constant <= 0.0:
                    raise ValueError("candidate fusion rrf_constant must be greater than zero")
                score = 0.0
                semantic_rank = semantic_ranks.get((str(src), str(tgt)))
                lexical_rank = lexical_ranks.get((str(src), str(tgt)))
                if semantic_rank is not None:
                    score += 1.0 / (rrf_constant + semantic_rank)
                if lexical_rank is not None:
                    score += 1.0 / (rrf_constant + lexical_rank)
            channels = [
                name
                for name, value in (("semantic", semantic), ("lexical", lexical))
                if value > 0.0
            ]
            entries.append((tgt, score, semantic, lexical, "|".join(channels)))
        entries.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        limit = adaptive_candidate_count(
            [float(item[1]) for item in entries],
            top_k=top_k,
            adaptive_config=adaptive_config,
        )
        for tgt, score, semantic, lexical, channels in entries[:limit]:
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


def adaptive_candidate_count(
    ranked_scores: Sequence[float],
    *,
    top_k: int,
    adaptive_config: Optional[Mapping[str, Any]] = None,
) -> int:
    """Return the number of ranked candidates retained for one source.

    Gap mode grows from ``k_min`` until the first sufficiently separated rank
    boundary. Entropy mode grows while the retained score distribution remains
    diffuse; concentrated prefixes stop at the minimum. Both modes are bounded
    by ``k_max`` and contain no reference/gold information.
    """

    available = len(ranked_scores)
    if not bool(_option(adaptive_config, "enabled", False)):
        return min(available, max(0, int(top_k)))
    if available == 0:
        return 0

    k_min = int(_option(adaptive_config, "k_min", top_k))
    k_max = int(_option(adaptive_config, "k_max", top_k))
    if k_min < 1 or k_max < 1 or k_min > k_max:
        raise ValueError("adaptive_k requires 1 <= k_min <= k_max")
    count = min(available, k_min)
    cap = min(available, k_max)
    if count >= cap:
        return count

    criterion = str(_option(adaptive_config, "criterion", "gap") or "gap").lower()
    if criterion == "gap":
        threshold = float(_option(adaptive_config, "gap", 0.05))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("adaptive_k.gap must be between zero and one")
        while count < cap:
            boundary_gap = float(ranked_scores[count - 1]) - float(ranked_scores[count])
            if boundary_gap >= threshold:
                break
            count += 1
        return count
    if criterion == "entropy":
        threshold = float(_option(adaptive_config, "entropy", 0.75))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("adaptive_k.entropy must be between zero and one")
        while count < cap:
            if normalized_score_entropy(ranked_scores[:count]) <= threshold:
                break
            count += 1
        return count
    raise ValueError(f"Unsupported adaptive_k criterion: {criterion!r}")


def normalized_score_entropy(scores: Sequence[float]) -> float:
    """Return Shannon entropy in ``[0, 1]`` for non-negative score mass."""

    if len(scores) <= 1:
        return 0.0
    mass = [max(0.0, float(score)) for score in scores]
    total = sum(mass)
    if total <= 0.0:
        return 1.0
    probabilities = [value / total for value in mass if value > 0.0]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return max(0.0, min(1.0, entropy / math.log(len(scores))))


def _channel_ranks(
    scores: Mapping[Tuple[str, str], float],
) -> Dict[Tuple[str, str], int]:
    by_source: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (src, tgt), score in scores.items():
        by_source[str(src)].append((str(tgt), float(score)))
    ranks: Dict[Tuple[str, str], int] = {}
    for src, values in by_source.items():
        values.sort(key=lambda item: (-item[1], item[0]))
        for rank, (tgt, _) in enumerate(values, start=1):
            ranks[(src, tgt)] = rank
    return ranks


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
    index: Mapping[str, Sequence[int]],
    n_records: int,
    fusion_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, List[int]]:
    floor = int(_option(fusion_config, "df_ceiling_floor", 10))
    ratio = float(_option(fusion_config, "df_ceiling_ratio", 0.20))
    max_df = max(floor, int(math.ceil(ratio * max(1, n_records))))
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
