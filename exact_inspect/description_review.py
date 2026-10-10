"""Offline independent review of saved prose; provider output cannot approve itself."""

from __future__ import annotations

import copy
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from .contracts import WireModel, canonical_hash
from .generation import FactPacket, parse_provider_output

Digest = Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
REVIEW_VERSION = "exact-description-review/1"


class ClaimReviewDecision(WireModel):
    ordinal: int = Field(ge=0, le=23)
    claim_id: Digest
    claim_hash: Digest
    decision: Literal["approved", "rejected"]
    support_explanation: str = Field(min_length=20, max_length=4000)


class IndependentDescriptionReview(WireModel):
    contract_version: Literal["exact-description-review/1"] = REVIEW_VERSION
    reviewer_id: str = Field(min_length=1, max_length=256)
    reviewer_kind: Literal["human", "independent_agent"]
    review_provenance: str = Field(min_length=20, max_length=2000)
    source_explanation_id: Digest
    source_manifest_hash: Digest
    source_generation_status: Literal["validated", "failed", "ambiguous"]
    packet_hash: Digest
    policy_hash: Digest
    response_hash: Digest
    claims_hash: Digest
    input_hash: Digest
    decisions: list[ClaimReviewDecision] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def exact_inventory(self):
        if [d.ordinal for d in self.decisions] != list(range(len(self.decisions))):
            raise ValueError("Review decisions must cover the ordered candidate claim inventory")
        if len({d.claim_id for d in self.decisions}) != len(self.decisions):
            raise ValueError("Duplicate reviewed claim identity")
        inventory = [{"claim_id": d.claim_id, "claim_hash": d.claim_hash} for d in self.decisions]
        if (
            canonical_hash(inventory) != self.claims_hash
            or _input_hash(self.model_dump()) != self.input_hash
        ):
            raise ValueError("Review input or claim inventory binding differs")
        return self


class IndependentReviewSupport(WireModel):
    receipt: IndependentDescriptionReview
    packet: FactPacket
    generation_manifest: dict[str, JsonValue]
    generation_manifest_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    @model_validator(mode="after")
    def exact_generation_binding(self):
        _validate_review_manifest(self.generation_manifest, self.receipt, self.packet)
        if canonical_hash(self.generation_manifest) != "sha256:" + self.generation_manifest_sha256:
            raise ValueError("Reviewed generation manifest digest differs from its content")
        return self


def _input_hash(value):
    return canonical_hash(
        {
            key: value[key]
            for key in (
                "source_explanation_id",
                "source_manifest_hash",
                "source_generation_status",
                "packet_hash",
                "policy_hash",
                "response_hash",
                "claims_hash",
            )
        }
    )


def claim_content(claim):
    raw = claim.model_dump() if hasattr(claim, "model_dump") else claim
    return {key: raw[key] for key in ("text", "fact_ids", "category")}


def _claim_id(source_id, ordinal, claim):
    return canonical_hash({"generation": source_id, "ordinal": ordinal, **claim_content(claim)})


def build_review_request(source_result, raw_response):
    """Expose an existing response for a separately invoked human/agent review.

    This helper assigns no approval. The request is kept offline; rejected draft
    prose is never copied into the served reviewed artifact.
    """
    packet = FactPacket.model_validate(source_result["support_packet"])
    if packet.selection.get("description_contract") != "exact-description/2":
        raise ValueError("Independent description review requires a versioned packet")
    manifest = source_result["manifest"]
    if (
        canonical_hash(packet) != manifest["packet_hash"]
        or packet.policy_hash != manifest["visibility_policy_hash"]
        or canonical_hash(raw_response) != manifest["response_hash"]
    ):
        raise ValueError("Saved review response differs from its source input bindings")
    output, _ = parse_provider_output(raw_response["choices"][0]["message"]["content"])
    if not output.claims:
        raise ValueError("There are no prose claims to review")
    if output.relation not in (None, "unresolved"):
        raise ValueError("Independent descriptions cannot add a pair verdict")
    if any(text not in packet.missingness for text in output.limitations):
        raise ValueError("Review candidates contain unsupported limitation prose")
    ids = {fact["fact_id"] for fact in packet.facts}
    claims = []
    for ordinal, claim in enumerate(output.claims):
        if not claim.fact_ids or not set(claim.fact_ids) <= ids:
            raise ValueError("Review claim lacks cited originals in its packet")
        value = claim_content(claim)
        claims.append(
            {"claim_id": _claim_id(source_result["explanation_id"], ordinal, claim), **value}
        )
    inventory = [
        {"claim_id": c["claim_id"], "claim_hash": canonical_hash(claim_content(c))} for c in claims
    ]
    bindings = {
        "source_explanation_id": source_result["explanation_id"],
        "source_manifest_hash": canonical_hash(manifest),
        "source_generation_status": manifest["status"],
        "packet_hash": canonical_hash(packet),
        "policy_hash": packet.policy_hash,
        "response_hash": canonical_hash(raw_response),
        "claims_hash": canonical_hash(inventory),
    }
    bindings["input_hash"] = _input_hash(bindings)
    return {
        "contract_version": REVIEW_VERSION,
        "bindings": bindings,
        "claims": claims,
        "claim_inventory": inventory,
        "packet": packet.model_dump(mode="json"),
    }


def validate_receipt_claims(receipt, packet, claims):
    """Verify exact reviewed claims; this does not infer their semantic correctness."""
    receipt = IndependentDescriptionReview.model_validate(receipt)
    packet = FactPacket.model_validate(packet)
    if canonical_hash(packet) != receipt.packet_hash or packet.policy_hash != receipt.policy_hash:
        raise ValueError("Review receipt belongs to another packet or policy")
    approved = {d.claim_id: d for d in receipt.decisions if d.decision == "approved"}
    if (
        not approved
        or {c["claim_id"] for c in claims} != set(approved)
        or len(claims) != len(approved)
    ):
        raise ValueError("Served claims must match exactly the independently approved inventory")
    ids = {fact["fact_id"] for fact in packet.facts}
    for claim in claims:
        decision = approved[claim["claim_id"]]
        if (
            canonical_hash(claim_content(claim)) != decision.claim_hash
            or _claim_id(receipt.source_explanation_id, decision.ordinal, claim)
            != decision.claim_id
            or not claim["fact_ids"]
            or not set(claim["fact_ids"]) <= ids
        ):
            raise ValueError("Reviewed claim content or original support differs from its receipt")
    return receipt


def validate_reviewed_artifact(result):
    packet = FactPacket.model_validate(result["support_packet"])
    receipt = validate_receipt_claims(result.get("independent_review"), packet, result["claims"])
    manifest = result["manifest"]
    _validate_review_manifest(manifest, receipt, packet)
    review_hash = canonical_hash(receipt)
    if (
        result["explanation_id"]
        != canonical_hash({"source": receipt.source_explanation_id, "review": review_hash})
        or result.get("preparation_kind") != "grounded_description"
        or result.get("support_status") != "independently_reviewed"
        or result.get("grounding_status") != "validated"
        or result.get("limitations") != list(dict.fromkeys(packet.missingness))
        or result.get("support_reasons") != []
        or result.get("fixture_provenance")
        != "Prepared prose with a separately recorded independent semantic review"
    ):
        raise ValueError("Reviewed artifact differs from its immutable review bindings")
    return receipt


def _validate_review_manifest(manifest, receipt, packet):
    """Bind the served manifest to both the frozen source and reviewed input.

    Import the shared manifest model at validation time: its module also exports
    the response model that contains review receipts.
    """
    from .models import GenerationManifest

    if set(manifest) - (set(GenerationManifest.model_fields) | {"category_alias_normalization"}):
        raise ValueError("Unadmitted field in reviewed generation manifest")
    normalization = manifest.get("category_alias_normalization", {})
    if not isinstance(normalization, dict) or any(
        not isinstance(key, str) or type(count) is not int or count < 0
        for key, count in normalization.items()
    ):
        raise ValueError("Invalid recorded category normalization")
    if GenerationManifest.model_validate(manifest).model_dump(mode="json") != manifest:
        raise ValueError("Reviewed generation manifest is not its canonical model projection")
    original_manifest = {
        key: value for key, value in manifest.items() if key != "description_review_hash"
    }
    original_manifest["status"] = receipt.source_generation_status
    if (
        manifest.get("description_contract") != "exact-description/2"
        or manifest.get("status") != "validated"
        or manifest.get("description_review_hash") != canonical_hash(receipt)
        or canonical_hash(original_manifest) != receipt.source_manifest_hash
        or manifest.get("packet_hash") != receipt.packet_hash
        or canonical_hash(packet) != receipt.packet_hash
        or manifest.get("visibility_policy_hash") != receipt.policy_hash
        or packet.policy_hash != receipt.policy_hash
        or manifest.get("ontology_context_hashes") != packet.context_hashes
        or manifest.get("language") != packet.language
        or manifest.get("response_hash") != receipt.response_hash
    ):
        raise ValueError("Reviewed generation manifest differs from its immutable review bindings")


def admit_independent_review(source_result, raw_response, receipt):
    """Create a new reviewed artifact from a separately supplied offline receipt.

    The receipt is never read from provider output, and the source result, response
    and cache files are never changed. Caller persists the returned artifact under
    its new explanation identity. Original facts are rechecked again at admission.
    """
    request = build_review_request(source_result, raw_response)
    receipt = IndependentDescriptionReview.model_validate(receipt)
    values = receipt.model_dump(mode="json")
    if any(values[key] != value for key, value in request["bindings"].items()):
        raise ValueError("Review receipt does not describe this saved response")
    inventory = request["claim_inventory"]
    if any(
        d.claim_id != item["claim_id"] or d.claim_hash != item["claim_hash"]
        for d, item in zip(receipt.decisions, inventory)
    ) or len(receipt.decisions) != len(inventory):
        raise ValueError("Review decisions differ from the saved candidate claims")
    approved = {d.claim_id for d in receipt.decisions if d.decision == "approved"}
    claims = [c for c in request["claims"] if c["claim_id"] in approved]
    validate_receipt_claims(receipt, request["packet"], claims)
    result = copy.deepcopy(source_result)
    review_hash = canonical_hash(receipt)
    result.update(
        explanation_id=canonical_hash(
            {"source": receipt.source_explanation_id, "review": review_hash}
        ),
        claims=claims,
        grounding_status="validated",
        preparation_kind="grounded_description",
        support_status="independently_reviewed",
        support_reasons=[],
        independent_review=values,
        limitations=list(dict.fromkeys(request["packet"]["missingness"])),
        fixture_provenance="Prepared prose with a separately recorded independent semantic review",
    )
    result["manifest"].update(status="validated", description_review_hash=review_hash)
    validate_reviewed_artifact(result)
    return result
