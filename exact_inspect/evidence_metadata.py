"""Recorded exact-rule and selected-evidence completeness, without inferred bridges."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import DomainError, EntityRef, VisibilityPolicy, canonical_hash


class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ExactBasisInput(MetadataModel):
    entity: EntityRef
    value: str = Field(min_length=1, max_length=4000)
    predicate_iri: str | None = None
    fact_ids: list[str] = Field(default_factory=list)


class ExactMatchBasis(MetadataModel):
    """Only an explicitly saved producer basis may have status recorded."""

    status: Literal["recorded", "not_recorded", "not_applicable", "withheld"]
    predicate: (
        Literal["normalized_label_equality", "candidate_token_key_equality", "recorded_other"]
        | None
    ) = None
    normalization_rule: str | None = None
    comparison_key: str | None = None
    source: ExactBasisInput | None = None
    target: ExactBasisInput | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    reason_code: str

    @model_validator(mode="after")
    def recorded_inputs(self):
        if self.status == "recorded" and not (
            self.predicate
            and self.normalization_rule
            and self.comparison_key
            and self.source
            and self.target
            and self.artifact_refs
        ):
            raise ValueError("Recorded exact basis requires saved rule, inputs and provenance")
        if self.status != "recorded" and any(
            value is not None
            for value in (
                self.predicate,
                self.normalization_rule,
                self.comparison_key,
                self.source,
                self.target,
            )
        ):
            raise ValueError("Unavailable exact basis cannot claim saved matching inputs")
        return self


class EvidenceBridge(MetadataModel):
    """Matcher comparison metadata, never an original ontology relationship."""

    bridge_id: str
    level: Literal["channel_comparison", "feature_correspondence"]
    status: Literal["recorded", "withheld"] = "recorded"
    channel: str
    source_channel: str | None = None
    target_channel: str | None = None
    source_family: str | None = None
    target_family: str | None = None
    comparison_basis: (
        Literal["matrix_support", "attribute_anchor", "difference_endpoint", "lexical_score"] | None
    ) = None
    recorded_score: float | None = Field(
        default=None,
        description="Saved matcher comparison metric; not semantic equivalence or calibrated probability.",
    )
    source: EntityRef
    target: EntityRef
    source_evidence_ids: list[str] = Field(default_factory=list)
    target_evidence_ids: list[str] = Field(default_factory=list)
    source_fact_ids: list[str] = Field(default_factory=list)
    target_fact_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(min_length=1)
    interpretation: Literal["matcher_comparison"] = "matcher_comparison"

    @model_validator(mode="after")
    def feature_identities(self):
        if self.level == "feature_correspondence" and not (
            self.source_evidence_ids and self.target_evidence_ids
        ):
            raise ValueError("Feature correspondence requires recorded evidence on both sides")
        return self


class RecordedComparisonResolution(MetadataModel):
    """A saved comparison's identity resolution, without exposing raw legacy aliases."""

    record_id: str
    channel: Literal["lexical", "hierarchy", "similarity", "attributes", "difference", "unknown"]
    status: Literal["resolved", "unresolved", "ambiguous", "unsupported"]
    reason_code: str
    bridge_id: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)


class ComparisonResolutionCounts(MetadataModel):
    recorded: int = Field(ge=0)
    resolved: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    ambiguous: int = Field(ge=0)
    unsupported: int = Field(ge=0)

    @model_validator(mode="after")
    def conserved(self):
        if self.recorded != self.resolved + self.unresolved + self.ambiguous + self.unsupported:
            raise ValueError("Every saved comparison needs one identity resolution")
        return self


class EvidenceCompleteness(MetadataModel):
    selected: int = Field(ge=0)
    resolved: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    not_exported: int = Field(ge=0)
    withheld: int = Field(ge=0)
    graph_supported: int = Field(ge=0)
    graph_unsupported: int = Field(ge=0)
    graph_not_recorded: int = Field(ge=0)

    @model_validator(mode="after")
    def conserved(self):
        if self.selected != self.resolved + self.unresolved + self.not_exported + self.withheld:
            raise ValueError("Every selected feature needs exactly one origin resolution status")
        if self.selected != self.graph_supported + self.graph_unsupported + self.graph_not_recorded:
            raise ValueError("Every selected feature needs exactly one graph projection status")
        return self


class PairEvidenceMetadata(MetadataModel):
    version: Literal["exact-evidence-metadata/1"] = "exact-evidence-metadata/1"
    status: Literal["selected", "none_selected", "not_exported", "request_failed"]
    counts: EvidenceCompleteness
    bridge_status: Literal["recorded", "not_recorded", "withheld"]
    bridges: list[EvidenceBridge] = Field(default_factory=list)
    comparison_records: list[RecordedComparisonResolution] = Field(default_factory=list)
    comparison_resolution: ComparisonResolutionCounts | None = None
    scope: Literal["saved_run", "prepared_visibility_policy"] = "saved_run"

    @model_validator(mode="after")
    def comparison_inventory(self):
        bridge_ids = [bridge.bridge_id for bridge in self.bridges]
        if len(set(bridge_ids)) != len(bridge_ids):
            raise ValueError("Duplicate recorded bridge identity")
        if self.bridge_status == "not_recorded" and (self.bridges or self.comparison_records):
            raise ValueError("Saved comparisons cannot be reported as not recorded")
        if self.comparison_resolution is None:
            if self.comparison_records:
                raise ValueError("Saved comparison inventory requires resolution counts")
            return self
        counts = Counter(row.status for row in self.comparison_records)
        expected = {
            key: counts[key] for key in ("resolved", "unresolved", "ambiguous", "unsupported")
        }
        expected["recorded"] = len(self.comparison_records)
        if self.comparison_resolution.model_dump() != expected or len(
            {row.record_id for row in self.comparison_records}
        ) != len(self.comparison_records):
            raise ValueError("Saved comparison resolution inventory differs from its counts")
        for row in self.comparison_records:
            if (row.status == "resolved" and row.bridge_id not in bridge_ids) or (
                row.status != "resolved" and row.bridge_id is not None
            ):
                raise ValueError("Comparison resolution differs from its recorded bridge")
        return self


def exact_match_basis(raw, *, source=None, target=None):
    """Keep producer effects distinct from their sometimes-unrecorded lexical basis.

    The current producer merges compact-label and token-key routes without recording
    which input key triggered a particular pair. Its protected flag or score of one
    cannot establish normalized-label equality.
    """
    if raw.get("stage") != "prefilter":
        return None
    reason = raw.get("reason_code")
    if raw.get("status") == "not_run" or reason in {"prefilter_disabled", "prefilter_retained"}:
        return ExactMatchBasis(
            status="not_applicable", reason_code="exact_rule_not_applied"
        ).model_dump(mode="json")
    saved = raw.get("exact_basis")
    if saved is not None:
        try:
            basis = ExactMatchBasis.model_validate(saved)
        except ValidationError as exc:
            raise DomainError(
                "invalid_exact_basis", "Recorded exact-rule metadata is invalid.", 422
            ) from exc
        if basis.status == "recorded" and (
            (source is not None and basis.source.entity.model_dump() != source)
            or (target is not None and basis.target.entity.model_dump() != target)
        ):
            raise DomainError(
                "invalid_exact_basis", "Recorded exact-rule entities differ from the pair.", 422
            )
        return basis.model_dump(mode="json")
    return ExactMatchBasis(
        status="not_recorded", reason_code="exact_rule_inputs_not_recorded"
    ).model_dump(mode="json")


def evidence_completeness(items):
    origins = Counter()
    projection = Counter()
    for item in items:
        origin = item.get("origin_resolution") or (
            "resolved"
            if item.get("status") == "available" and item.get("fact_ids")
            else (
                "withheld"
                if item.get("status") == "filtered"
                else (
                    "unresolved"
                    if item.get("status") in {"unresolved", "unresolved_import"}
                    else "not_exported"
                )
            )
        )
        origins[origin] += 1
        projection[item.get("graph_projection") or "not_recorded"] += 1
    return EvidenceCompleteness(
        selected=len(items),
        resolved=origins["resolved"],
        unresolved=origins["unresolved"],
        not_exported=origins["not_exported"],
        withheld=origins["withheld"],
        graph_supported=projection["supported"],
        graph_unsupported=projection["unsupported"],
        graph_not_recorded=projection["not_recorded"],
    ).model_dump()


def recorded_bridges(record, items, source, target, artifact_refs):
    """Admit only explicit recorded comparisons; shared channels imply no bridge."""
    by_id = {item["evidence_id"]: item for item in items}
    bridges = []
    for raw in record.get("evidence_bridges", []):
        value = dict(raw)
        value.setdefault("artifact_refs", list(artifact_refs))
        value.setdefault("bridge_id", canonical_hash({"pair": [source, target], "record": raw}))
        try:
            bridge = EvidenceBridge.model_validate(value)
        except ValidationError as exc:
            raise DomainError(
                "invalid_evidence_bridge", "Recorded evidence comparison is invalid.", 422
            ) from exc
        if bridge.source.model_dump() != source or bridge.target.model_dump() != target:
            raise DomainError(
                "invalid_evidence_bridge",
                "Recorded evidence comparison has a different typed pair.",
                422,
            )
        for side in ("source", "target"):
            ids = getattr(bridge, side + "_evidence_ids")
            if any(
                eid not in by_id
                or by_id[eid]["side"] != side
                or by_id[eid]["entity"] != (source if side == "source" else target)
                or by_id[eid]["channel"] != (getattr(bridge, side + "_channel") or bridge.channel)
                or (
                    getattr(bridge, side + "_family") is not None
                    and by_id[eid].get("family") != getattr(bridge, side + "_family")
                )
                for eid in ids
            ):
                raise DomainError(
                    "invalid_evidence_bridge",
                    "Recorded comparison evidence is outside its scope.",
                    422,
                )
            linked = {fid for eid in ids for fid in by_id[eid]["fact_ids"]}
            if not set(getattr(bridge, side + "_fact_ids")) <= linked:
                raise DomainError(
                    "invalid_evidence_bridge",
                    "Recorded comparison facts are not linked to its evidence.",
                    422,
                )
        bridges.append(bridge.model_dump(mode="json"))
    return bridges


def legacy_recorded_bridges(record, items, source, target, artifact_refs):
    """Adapt saved producer links only; never rerun scoring or infer correspondence.

    PairAdaptiveEvidenceMixin records matrix links for hierarchy/similarity,
    attribute-to-opposing-bank anchors, and difference-to-opposing-endpoint links.
    An attribute hierarchy anchor omits family: it is admissible only if exactly
    one selected feature across that opposite channel has the recorded alias.
    """
    provenance = record.get("cross_side_provenance")
    if provenance is None:
        return [], []
    scopes = defaultdict(list)
    for item in items:
        scopes[
            (item["side"], item["channel"], item.get("family"), item.get("historical_item_alias"))
        ].append(item)
    bridges, resolutions = [], []
    artifacts = list(artifact_refs)

    def resolve(side, channel, family, alias, *, family_unknown=False):
        if not isinstance(alias, str) or not alias:
            return None, "unresolved", "saved_item_alias_missing"
        alias = "legacy:" + alias
        if family_unknown:
            matches = [
                item
                for (s, c, _, a), rows in scopes.items()
                if (s, c, a) == (side, channel, alias)
                for item in rows
            ]
        else:
            matches = scopes.get((side, channel, family, alias), [])
        if not matches:
            return None, "unresolved", "saved_item_not_selected_in_scope"
        if len(matches) != 1:
            return None, "ambiguous", "saved_item_alias_not_unique_in_scope"
        item = matches[0]
        if item["entity"] != (source if side == "source" else target):
            return None, "unresolved", "saved_item_typed_entity_mismatch"
        return item, "resolved", "saved_item_identity_resolved"

    def adapt(channel, location, raw, *, family=None, side=None, malformed=False):
        record_id = canonical_hash({"pair": [source, target], "location": location, "record": raw})
        result = dict(record_id=record_id, channel=channel, artifact_refs=artifacts)

        def unavailable(status, reason):
            resolutions.append(
                RecordedComparisonResolution(
                    **result, status=status, reason_code=reason
                ).model_dump(mode="json")
            )

        if malformed or not isinstance(raw, dict):
            unavailable("unsupported", "saved_comparison_shape_unsupported")
            return
        score = raw.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            unavailable("unsupported", "saved_comparison_score_invalid")
            return
        if not artifacts:
            unavailable("unresolved", "saved_comparison_artifact_unavailable")
            return
        resolved = {"source": None, "target": None}
        failures = []
        basis = None

        def feature(which, chan, fam, alias, family_unknown=False):
            value, status, reason = resolve(which, chan, fam, alias, family_unknown=family_unknown)
            if status != "resolved":
                failures.append((status, reason))
            resolved[which] = value

        if channel in {"hierarchy", "similarity"}:
            if set(raw) != {"source_item_id", "target_item_id", "score"}:
                unavailable("unsupported", "saved_matrix_link_shape_unsupported")
                return
            feature("source", channel, family, raw["source_item_id"])
            feature("target", channel, family, raw["target_item_id"])
            basis = "matrix_support"
        elif channel == "lexical":
            if set(raw) != {"source_ref", "target_ref", "score"} or (
                raw["source_ref"],
                raw["target_ref"],
            ) != ("__source__", "__target__"):
                unavailable("unsupported", "saved_lexical_endpoints_unsupported")
                return
            basis = "lexical_score"
        elif channel in {"attributes", "difference"}:
            if side not in {"source", "target"} or set(raw) != {
                "item_id",
                "anchor_kind",
                "anchor_ref",
                "score",
            }:
                unavailable("unsupported", "saved_anchor_shape_unsupported")
                return
            other = "target" if side == "source" else "source"
            feature(side, channel, None, raw["item_id"])
            anchor = raw["anchor_kind"]
            if (channel == "difference" and anchor == "endpoint") or (
                channel == "attributes" and anchor == "label"
            ):
                if raw["anchor_ref"] != "__" + other + "__":
                    unavailable("unsupported", "saved_anchor_endpoint_differs")
                    return
            elif channel == "attributes" and anchor in {"hierarchy", "similarity", "attribute"}:
                feature(
                    other,
                    "attributes" if anchor == "attribute" else anchor,
                    None,
                    raw["anchor_ref"],
                    family_unknown=anchor == "hierarchy",
                )
            else:
                unavailable("unsupported", "saved_anchor_kind_unsupported")
                return
            basis = "attribute_anchor" if channel == "attributes" else "difference_endpoint"
        else:
            unavailable("unsupported", "saved_comparison_channel_unsupported")
            return
        if failures:
            status, reason = next((f for f in failures if f[0] == "ambiguous"), failures[0])
            unavailable(status, reason)
            return
        values = {}
        for which, item in resolved.items():
            values.update(
                {
                    which + "_channel": item["channel"] if item else None,
                    which + "_family": item.get("family") if item else None,
                    which + "_evidence_ids": [item["evidence_id"]] if item else [],
                    which + "_fact_ids": list(item["fact_ids"]) if item else [],
                }
            )
        bridge = EvidenceBridge(
            bridge_id=canonical_hash({"saved_comparison": record_id}),
            level="feature_correspondence" if all(resolved.values()) else "channel_comparison",
            channel=channel,
            source=source,
            target=target,
            artifact_refs=artifacts,
            comparison_basis=basis,
            recorded_score=score,
            **values,
        ).model_dump(mode="json")
        bridges.append(bridge)
        resolutions.append(
            RecordedComparisonResolution(
                **result,
                status="resolved",
                reason_code="saved_comparison_identity_resolved",
                bridge_id=bridge["bridge_id"],
            ).model_dump(mode="json")
        )

    def rows(channel, payload, location, **scope):
        if not isinstance(payload, list):
            adapt(channel, location, payload, malformed=True)
            return
        for ordinal, raw in enumerate(payload):
            adapt(channel, [*location, ordinal], raw, **scope)

    if not isinstance(provenance, dict):
        adapt("unknown", ["cross_side_provenance"], provenance, malformed=True)
        return bridges, resolutions
    for channel, payload in provenance.items():
        location = ["cross_side_provenance", channel]
        if channel in {"lexical", "similarity"}:
            rows(channel, payload, location)
        elif channel == "hierarchy" and isinstance(payload, dict):
            for family, entries in payload.items():
                rows(channel, entries, [*location, family], family=family)
        elif channel in {"attributes", "difference"} and isinstance(payload, dict):
            for side, entries in payload.items():
                rows(channel, entries, [*location, side], side=side)
        else:
            adapt(
                channel if channel in {"hierarchy", "attributes", "difference"} else "unknown",
                location,
                payload,
                malformed=True,
            )
    return bridges, resolutions


def pair_evidence_metadata(record, items, source, target, artifact_refs, events=()):
    artifact_refs = list(artifact_refs)
    bridges = recorded_bridges(record, items, source, target, artifact_refs)
    legacy, resolutions = legacy_recorded_bridges(record, items, source, target, artifact_refs)
    bridges.extend(legacy)
    resolved_counts = Counter(item["status"] for item in resolutions)
    explicit_empty = any(key in record for key in ("triple_attributions", "attributes"))
    protected = any(
        e.get("stage") == "prefilter" and e.get("reason_code") == "exact_match_protected"
        for e in events
    )
    status = (
        "selected"
        if items
        else (
            "request_failed"
            if record.get("evidence_status") == "failed"
            else "none_selected" if explicit_empty or protected else "not_exported"
        )
    )
    return PairEvidenceMetadata(
        status=status,
        counts=evidence_completeness(items),
        bridge_status="recorded" if bridges or resolutions else "not_recorded",
        bridges=bridges,
        comparison_records=resolutions,
        comparison_resolution=(
            ComparisonResolutionCounts(
                recorded=len(resolutions),
                **{
                    key: resolved_counts[key]
                    for key in ("resolved", "unresolved", "ambiguous", "unsupported")
                },
            )
            if "cross_side_provenance" in record
            else None
        ),
    ).model_dump(mode="json")


def filter_evidence_items(
    items, contexts: Mapping[str, Any], policy: VisibilityPolicy, *, physically_filtered=False
):
    """Keep all selected identities, redact unresolved/withheld content and original links.

    ``physically_filtered`` uses prepared SQLite indexes and requires their matching
    policy hash. Request handling uses each context's existing policy-aware axiom API.
    Request failures propagate; they must never become an apparently empty selection.
    """
    result = []
    for item in items:
        fact_ids = item.get("fact_ids") or []
        context = contexts.get(item["entity"]["ontology_version_id"])
        visible = bool(fact_ids and context is not None)
        if visible and physically_filtered:
            if context.manifest.get("policy_filter", {}).get("policy_hash") != policy.policy_hash:
                raise ValueError("Portable evidence requires a physically filtered context")
            with context._connection() as facts:
                marks = ",".join("?" for _ in fact_ids)
                available = {
                    r[0]
                    for r in facts.execute(
                        "SELECT id FROM axioms WHERE id IN (" + marks + ")", fact_ids
                    )
                }
            visible = set(fact_ids) == available
        elif visible:
            try:
                for fact_id in fact_ids:
                    context.axiom(fact_id, policy=policy)
            except KeyError:
                # OntologyContext uses the same absence boundary for missing
                # originals and originals excluded by the effective policy.
                visible = False
            except DomainError as exc:
                if exc.status_code != 404:
                    raise
                visible = False
        allowed = {
            key: item[key]
            for key in (
                "evidence_id",
                "channel",
                "family",
                "side",
                "role",
                "entity",
                "feature_id",
                "fact_ids",
                "values",
                "interpretation",
                "provenance_status",
                "status",
                "reason",
                "origin_resolution",
                "graph_projection",
            )
            if key in item
        }
        allowed.update(
            display={},
            semantic_terms={},
            axiom_origins=[],
            source_axiom_refs=[],
            historical_item_alias=None,
        )
        if not visible:
            allowed.update(fact_ids=[], feature_id=None)
            if fact_ids or item.get("origin_resolution") == "withheld":
                allowed.update(
                    status="filtered",
                    origin_resolution="withheld",
                    reason="Original fact links are outside the admitted visibility scope.",
                )
            else:
                allowed.setdefault("origin_resolution", "not_exported")
                allowed["reason"] = (
                    "Original fact mapping was not resolved."
                    if allowed["origin_resolution"] == "unresolved"
                    else "Original fact mapping was not exported."
                )
            # Derivation premises may refer to hidden originals; no copied raw strings.
        result.append(allowed)
    return result


def filter_pair_metadata(metadata, safe_items):
    value = dict(metadata)
    value.update(counts=evidence_completeness(safe_items), scope="prepared_visibility_policy")
    visible = {item["evidence_id"] for item in safe_items if item["status"] == "available"}
    bridges = []
    for bridge in value.get("bridges", []):
        bridge = dict(bridge)
        if not set(bridge["source_evidence_ids"] + bridge["target_evidence_ids"]) <= visible:
            bridge.update(status="withheld", source_fact_ids=[], target_fact_ids=[])
        bridges.append(bridge)
    value["bridges"] = bridges
    if bridges:
        value["bridge_status"] = (
            "withheld" if any(b["status"] == "withheld" for b in bridges) else "recorded"
        )
    return PairEvidenceMetadata.model_validate(value).model_dump(mode="json")


def filter_exact_basis(
    events, safe_items, *, contexts=None, policy=None, physically_filtered=False
):
    """Recorded lexical inputs require retained original fact support on both sides."""
    visible = {
        (item["entity"]["ontology_version_id"], fid)
        for item in safe_items
        if item["status"] == "available"
        for fid in item["fact_ids"]
    }
    result = []
    for event in events:
        event = dict(event)
        basis = event.get("exact_basis")
        if basis and basis["status"] == "recorded":

            def permitted(side):
                item = basis[side]
                if not item["fact_ids"]:
                    return False
                if contexts is not None and policy is not None:
                    probe = filter_evidence_items(
                        [
                            {
                                "entity": item["entity"],
                                "fact_ids": item["fact_ids"],
                                "status": "available",
                            }
                        ],
                        contexts,
                        policy,
                        physically_filtered=physically_filtered,
                    )[0]
                    return probe["status"] == "available"
                return all(
                    (item["entity"]["ontology_version_id"], fid) in visible
                    for fid in item["fact_ids"]
                )

            if not all(permitted(side) for side in ("source", "target")):
                event["exact_basis"] = ExactMatchBasis(
                    status="withheld", reason_code="exact_rule_inputs_outside_visible_evidence"
                ).model_dump(mode="json")
        result.append(event)
    return result
