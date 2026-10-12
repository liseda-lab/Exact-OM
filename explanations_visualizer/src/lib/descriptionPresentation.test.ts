import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { claimSides, comparisonView, coveredSides, profileView, roleSides } from "./descriptionPresentation.ts";

const NCIT = { ontology_version_id: "sha256:ncit", iri: "http://ncit#C121750", kind: "class" as const };
const DOID = { ontology_version_id: "sha256:doid", iri: "http://doid#DOID_0112183", kind: "class" as const };
const artifact = (kind: string, support: string, entities: unknown[], claimSidesValue: Record<string, string[]>, covered: string[]) =>
  ({
    entities,
    claims: Object.keys(claimSidesValue).map((claim_id) => ({ claim_id, text: "t", fact_ids: [], category: "key_fact" })),
    support_status: support,
    presentation: { contract_version: "exact-description-presentation/1", kind, covered_sides: covered, claim_sides: claimSidesValue, relation_asserted: false, scope_note: "Descriptions and excerpts do not determine mapping equivalence or incompatibility.", render_revision: null, review_receipt_sha256: null },
  }) as never;

test("artifact roles resolve by entity identity, not column position (23 C03)", () => {
  // review-12: a target-only comparison whose artifact lists NCIT first; the workspace shows DOID as source.
  const comparison = artifact("target_only_profile", "independently_reviewed", [NCIT, DOID], { c1: ["target"] }, ["target"]);
  const roles = roleSides(comparison, { source: DOID, target: NCIT }, () => null);
  assert.deepEqual(roles, { source: "target", target: "source" });
  assert.deepEqual(claimSides(comparison, roles), { c1: ["source"] });
  const covered = coveredSides(comparison, roles);
  assert.deepEqual(covered, ["source"]);
  const view = comparisonView(comparison, covered);
  assert.equal(view.title, "One-sided information");
  assert.match(view.intro ?? "", /not a difference/);
  assert.equal(view.grouped, false);
});

test("each preparation keeps its own label", () => {
  assert.equal(profileView(artifact("necessary_conditions", "typed_template_support", [DOID], { c: ["source"] }, ["source"])).title, "Necessary conditions");
  assert.deepEqual(profileView(artifact("necessary_conditions", "typed_template_support", [DOID], { c: ["source"] }, ["source"])).tag, { text: "From original axioms", tone: "derived" });
  assert.equal(profileView(artifact("reviewed_profile", "independently_reviewed", [DOID], { c: ["source"] }, ["source"])).tag.text, "Generated · reviewed");
  assert.equal(profileView(artifact("available_facts", "fallback", [DOID], { c: ["source"] }, ["source"])).title, "Available facts");
  assert.equal(comparisonView(artifact("paired_context", "exact_support", [NCIT, DOID], { a: ["source"], b: ["target"] }, ["source", "target"]), ["source", "target"]).title, "Definitions side by side");
  // Typed-readable/3 paired definitions are quoted excerpts (fact_extract), not readings of axioms.
  const paired = Object.assign(artifact("paired_context", "typed_template_support", [NCIT, DOID], { a: ["source"], b: ["target"] }, ["source", "target"]), { preparation_kind: "fact_extract" });
  assert.deepEqual(comparisonView(paired, ["source", "target"]).tag, { text: "Original excerpts", tone: "original" });
  assert.equal(comparisonView(artifact("reviewed_comparison", "independently_reviewed", [NCIT, DOID], { a: ["source", "target"] }, ["source", "target"]), ["source", "target"]).grouped, true);
  assert.equal(profileView(artifact("unavailable", "fallback", [DOID], {}, [])).unavailable, true);
});
