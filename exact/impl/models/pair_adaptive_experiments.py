"""Small runtime helpers for default-off pair-adaptive experiments.

The paper experiment suite deliberately keeps fitted artifacts simple.  This
module therefore supports deterministic JSON only; it never fits a model and
never deserializes executable Python objects.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from exact.utils.candidate_generation import candidate_tokens, normalize_candidate_text
from exact.utils.provenance import file_provenance


def config_dict(value: Any) -> Dict[str, Any]:
    """Return a plain copy of a nested config object."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="python"))
    raise TypeError(f"experiment config must be a mapping, got {type(value).__name__}")


def _longest_common_substring(left: str, right: str) -> tuple[int, int, int]:
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    match = matcher.find_longest_match(0, len(left), 0, len(right))
    return int(match.a), int(match.b), int(match.size)


def isub_similarity(left: str, right: str) -> float:
    """Return the I-Sub string similarity used by ontology matchers.

    The implementation follows the published commonality, unmatched-string
    dissimilarity, and Winkler-prefix terms.  Matches shorter than two
    characters are ignored, as in the usual ontology-matching implementation.
    """

    first = normalize_candidate_text(left).replace(" ", "")
    second = normalize_candidate_text(right).replace(" ", "")
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0

    work_first = first
    work_second = second
    common = 0
    while work_first and work_second:
        first_at, second_at, size = _longest_common_substring(work_first, work_second)
        if size <= 1:
            break
        common += size
        work_first = work_first[:first_at] + work_first[first_at + size :]
        work_second = work_second[:second_at] + work_second[second_at + size :]

    commonality = (2.0 * common) / max(1.0, float(len(first) + len(second)))
    unmatched_first = (len(first) - common) / float(len(first))
    unmatched_second = (len(second) - common) / float(len(second))
    unmatched_sum = unmatched_first + unmatched_second
    unmatched_product = unmatched_first * unmatched_second
    denominator = 0.6 + 0.4 * (unmatched_sum - unmatched_product)
    dissimilarity = unmatched_product / denominator if denominator > 0.0 else 0.0

    prefix = 0
    for first_char, second_char in zip(first[:4], second[:4]):
        if first_char != second_char:
            break
        prefix += 1
    winkler = prefix * 0.1 * (1.0 - commonality)
    return max(0.0, min(1.0, commonality - dissimilarity + winkler))


def jaro_winkler_similarity(left: str, right: str) -> float:
    """Return a dependency-free Jaro-Winkler similarity in ``[0, 1]``."""

    first = normalize_candidate_text(left).replace(" ", "")
    second = normalize_candidate_text(right).replace(" ", "")
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    radius = max(len(first), len(second)) // 2 - 1
    radius = max(0, radius)
    first_hits = [False] * len(first)
    second_hits = [False] * len(second)
    matches = 0
    for first_idx, char in enumerate(first):
        start = max(0, first_idx - radius)
        end = min(first_idx + radius + 1, len(second))
        for second_idx in range(start, end):
            if second_hits[second_idx] or second[second_idx] != char:
                continue
            first_hits[first_idx] = True
            second_hits[second_idx] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    first_matched = [char for char, hit in zip(first, first_hits) if hit]
    second_matched = [char for char, hit in zip(second, second_hits) if hit]
    transpositions = sum(a != b for a, b in zip(first_matched, second_matched)) / 2.0
    jaro = (
        matches / len(first) + matches / len(second) + (matches - transpositions) / matches
    ) / 3.0
    prefix = 0
    for first_char, second_char in zip(first[:4], second[:4]):
        if first_char != second_char:
            break
        prefix += 1
    return max(0.0, min(1.0, jaro + prefix * 0.1 * (1.0 - jaro)))


def token_set_similarity(left: str, right: str) -> float:
    """Return a length-aware token-set ratio in ``[0, 1]``.

    The usual token-set maximum is multiplied by token-count coverage.  This
    retains its word-order robustness without assigning a perfect score when a
    short label is merely contained in a long compositional label.
    """

    first_tokens = set(candidate_tokens(normalize_candidate_text(left)))
    second_tokens = set(candidate_tokens(normalize_candidate_text(right)))
    if not first_tokens or not second_tokens:
        return 0.0
    intersection = sorted(first_tokens & second_tokens)
    first_only = sorted(first_tokens - second_tokens)
    second_only = sorted(second_tokens - first_tokens)
    common = " ".join(intersection)
    first_combined = " ".join(intersection + first_only)
    second_combined = " ".join(intersection + second_only)

    def ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(SequenceMatcher(None, a, b, autojunk=False).ratio())

    raw = max(
        ratio(common, first_combined),
        ratio(common, second_combined),
        ratio(first_combined, second_combined),
    )
    coverage = min(len(first_tokens), len(second_tokens)) / max(
        len(first_tokens), len(second_tokens)
    )
    return max(0.0, min(1.0, raw * coverage))


def abbreviation_similarity(left: str, right: str) -> float:
    """Conservatively score an initialism or ordered compound abbreviation."""

    normalized_left = normalize_candidate_text(left)
    normalized_right = normalize_candidate_text(right)
    if not normalized_left or not normalized_right:
        return 0.0

    def one_way(short_text: str, expansion_text: str) -> float:
        short = short_text.replace(" ", "")
        expansion_tokens = [token for token in expansion_text.split() if token]
        if not (2 <= len(short) <= 10) or not expansion_tokens:
            return 0.0
        expansion = "".join(expansion_tokens)
        if not expansion or short[0] != expansion[0]:
            return 0.0
        if len(expansion_tokens) >= 2:
            initials = "".join(token[0] for token in expansion_tokens)
            if short == initials:
                return 1.0
            initialism = jaro_winkler_similarity(short, initials)
            if initialism >= 0.9 and abs(len(short) - len(initials)) <= 1:
                return 0.9 * initialism
        position = 0
        for char in expansion:
            if position < len(short) and char == short[position]:
                position += 1
        if position != len(short) or len(expansion) < len(short) + 3:
            return 0.0
        length_penalty = min(1.0, len(short) / max(2.0, math.sqrt(len(expansion))))
        return 0.75 * length_penalty

    return max(
        one_way(normalized_left, normalized_right), one_way(normalized_right, normalized_left)
    )


def deterministic_fraction(seed: int | None, *parts: str) -> float:
    """Map stable identifiers and a seed to a reproducible half-open unit interval."""

    payload = "\u241f".join([str(seed if seed is not None else 0), *map(str, parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


@dataclass(frozen=True)
class JsonExperimentArtifact:
    """Validated immutable JSON artifact plus file provenance."""

    path: Path
    payload: Dict[str, Any]
    provenance: Dict[str, Any]

    @classmethod
    def load(cls, value: Any, *, expected_mode: str, kind: str) -> "JsonExperimentArtifact":
        if value is None:
            raise ValueError(f"{kind} mode {expected_mode!r} requires an artifact")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{kind} artifact does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{kind} artifact is not valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{kind} artifact root must be an object: {path}")
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"{kind} artifact schema_version must be 1: {path}")
        if str(payload.get("mode", "")) != str(expected_mode):
            raise ValueError(
                f"{kind} artifact mode {payload.get('mode')!r} does not match {expected_mode!r}"
            )
        return cls(path=path, payload=payload, provenance=file_provenance(path))

    def validate_dataset(self, dataset_signature: str | None, *, required: bool = False) -> None:
        expected = self.payload.get("dataset_signature")
        if required and not expected:
            raise ValueError(f"artifact {self.path} must declare dataset_signature")
        if expected and expected != dataset_signature:
            raise ValueError(
                f"artifact {self.path} dataset_signature mismatch: "
                f"expected {expected!r}, got {dataset_signature!r}"
            )

    def require_fields(self, *names: str, kind: str = "artifact") -> None:
        """Fail closed when scientific artifact provenance is incomplete."""

        missing = [name for name in names if self.payload.get(name) in (None, "", [], {})]
        if missing:
            raise ValueError(
                f"{kind} artifact {self.path} is missing required field(s): {', '.join(missing)}"
            )

    def scoped_payload(self, field: str, *, scope_key: str) -> Mapping[str, Any]:

        value = self.payload.get(field)
        if not isinstance(value, Mapping):
            raise ValueError(f"artifact {self.path} must contain object field {field!r}")
        if scope_key in value or "global" in value:
            selected = value.get(scope_key, value.get("global"))
            if not isinstance(selected, Mapping):
                raise ValueError(
                    f"artifact {self.path} has no {field!r} entry for scope {scope_key!r}"
                )
            return selected
        return value


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity remapped from ``[-1, 1]`` to ``[0, 1]``."""

    if len(left) != len(right) or not left:
        raise ValueError("graph embedding vectors must be non-empty and have equal dimensions")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.5
    return max(0.0, min(1.0, 0.5 * (1.0 + dot / (left_norm * right_norm))))


__all__ = [
    "JsonExperimentArtifact",
    "abbreviation_similarity",
    "config_dict",
    "cosine_similarity",
    "deterministic_fraction",
    "isub_similarity",
    "jaro_winkler_similarity",
    "token_set_similarity",
]
