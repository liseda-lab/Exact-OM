"""Opt-in repair-only wire controls; shared matcher routing stays unchanged."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from functools import lru_cache
from collections.abc import Mapping
from exact.repair.records import canonical_json

SCHEMA_VERSION = "semantic-fidelity-response/v1"


def response_format(packet, swapped=False):
    """Shape constraints only. Grounding, scores and abstention still need validation."""

    def obj(properties):
        return dict(
            type="object",
            properties=properties,
            required=list(properties),
            additionalProperties=False,
        )

    text = dict(type="string")
    decision = dict(type="string", enum=["A", "B", "tie", "abstain"])
    citation = dict(type="string", enum=sorted(packet.evidence))
    criterion = obj(
        dict(
            criterion_id=dict(type="string", enum=sorted(packet.criterion_weights)),
            status=dict(type="string", enum=["decided", "unknown"]),
            preference=decision,
            a_score=dict(type=["number", "null"], minimum=0, maximum=1),
            b_score=dict(type=["number", "null"], minimum=0, maximum=1),
            evidence_ids=dict(type="array", items=citation),
            reason=text,
            quotes=dict(type="array", items=obj(dict(evidence_id=citation, span=text))),
            symbolic_claims=dict(
                type="array",
                items=obj(
                    dict(evidence_id=citation, value=dict(type="string", enum=["true", "false"]))
                ),
            ),
        )
    )
    context = packet.judge_payload(swapped=swapped)["context"]
    properties = {key: dict(type="string", enum=[value]) for key, value in context.items()}
    properties.update(
        decision=decision,
        criteria=dict(type="array", items=criterion),
        abstention_reason=dict(type=["string", "null"]),
    )
    return dict(
        type="json_schema",
        json_schema=dict(name="semantic_fidelity_v1", strict=True, schema=obj(properties)),
    )


def validate_wire_controls(controls):
    if not isinstance(controls, Mapping) or set(controls) not in (
        {"response_format"},
        {"reasoning", "response_format"},
    ):
        raise ValueError(
            "Repair controls require strict response_format and optional supported reasoning"
        )
    reasoning = controls.get("reasoning")
    if "reasoning" in controls and dict(reasoning or {}) not in (
        {"effort": "low"},
        {"enabled": False},
    ):
        raise ValueError("Unsupported repair reasoning controls")
    form = controls["response_format"]
    if (
        form.get("type") != "json_schema"
        or form.get("json_schema", {}).get("strict") is not True
        or form["json_schema"].get("name") != "semantic_fidelity_v1"
        or form["json_schema"].get("schema", {}).get("additionalProperties") is not False
    ):
        raise ValueError("Repair controls require strict JSON schema, not json_object")
    return json.loads(canonical_json(controls))


def controlled_completion(client, profile, messages, max_tokens, role, controls):
    """Reuse durable shared generation, without modifying its public chat API."""
    controls = validate_wire_controls(controls)
    if (
        profile.provider.get("allow_fallbacks") is not False
        or profile.provider.get("require_parameters") is not True
    ):
        raise ValueError("Controlled annotation requires fail-closed provider routing")
    return client._generation(
        profile,
        dict(
            model=profile.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            provider=dict(profile.provider),
            **controls,
        ),
        "chat/completions",
        role=role,
        requested_seed=None,
    )


@lru_cache(maxsize=4)
def _tokenizer(path, digest):
    from tokenizers import Tokenizer

    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Pinned annotation tokenizer changed")
    return Tokenizer.from_str(raw.decode())


def input_token_bound(messages, controls=None, tokenizer=None):
    # Legacy bytes are unchanged. Controlled requests count message text with a
    # pinned model tokenizer, plus 256 framing tokens and a conservative byte
    # bound for the *entire* response schema/reasoning object. This is a local
    # admission estimate, not a claim about the provider's billed count.
    plain = json.loads(canonical_json(messages))
    extra = len(json.dumps(json.loads(canonical_json(controls))).encode()) if controls else 0
    if tokenizer:
        path, digest = tokenizer["path"], tokenizer["sha256"]
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned annotation tokenizer changed")
        text = json.dumps(plain, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode()) + extra > 32768:
            raise ValueError("Annotation input exceeds independent byte cap")
        return len(_tokenizer(path, digest).encode(text).ids) + 256 + extra
    return len(json.dumps(plain).encode()) + 256 + extra
