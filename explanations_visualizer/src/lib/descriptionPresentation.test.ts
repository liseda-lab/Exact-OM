import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { citedSubjects, claimSides, comparisonView, coveredSides, profileView, roleSides, textOrigin } from "./descriptionPresentation.ts";

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

test("generated reviewed prose is never presented as a quotation; only declared extracts are (R1)", () => {
  const with_ = (kind: string, support: string, preparation: string, categories: Record<string, string[]>) =>
    Object.assign(artifact(kind, support, [NCIT, DOID], categories, Array.from(new Set(Object.values(categories).flat()))), { preparation_kind: preparation });
  // The four reviewed paired_context artifacts of the Luna publication: generated per-concept prose.
  const pairedReviewed = with_("paired_context", "independently_reviewed", "grounded_description", { a: ["source"], b: ["target"] });
  const pairedView = comparisonView(pairedReviewed, ["source", "target"]);
  assert.equal(textOrigin(pairedReviewed), "generated");
  assert.equal(pairedView.title, "Descriptions side by side");
  assert.equal(pairedView.quote, false);
  assert.doesNotMatch(pairedView.intro ?? "", /Quoted/);
  assert.deepEqual(pairedView.tag, { text: "Generated · reviewed", tone: "generated" });
  // The five reviewed comparisons whose only claim is a meaning summary stay grouped, unquoted.
  const meaningOnly = with_("reviewed_comparison", "independently_reviewed", "grounded_description", { m: ["source", "target"] });
  assert.equal(comparisonView(meaningOnly, ["source", "target"]).quote, false);
  assert.equal(comparisonView(meaningOnly, ["source", "target"]).grouped, true);
  // Genuine extracts keep quotation and "Definitions" wording.
  const extract = with_("paired_context", "typed_template_support", "fact_extract", { a: ["source"], b: ["target"] });
  assert.equal(comparisonView(extract, ["source", "target"]).title, "Definitions side by side");
  assert.equal(comparisonView(extract, ["source", "target"]).quote, true);
  assert.equal(comparisonView(with_("source_only_profile", "typed_template_support", "fact_extract", { a: ["source"] }), ["source"]).quote, true);
  // A fixed reading of axioms is neither generated prose nor a quotation.
  const typed = with_("paired_context", "typed_template_support", "grounded_description", { a: ["source"], b: ["target"] });
  assert.equal(textOrigin(typed), "derived");
  assert.equal(comparisonView(typed, ["source", "target"]).title, "Original facts side by side");
  assert.equal(comparisonView(typed, ["source", "target"]).quote, false);
  assert.equal(profileView(with_("necessary_conditions", "typed_template_support", "grounded_description", { c: ["source"] })).quote, false);
  assert.equal(profileView(with_("available_facts", "fallback", "factual_fallback", { c: ["source"] })).quote, true);
});

test("each cited record keeps its own typed subject from the artifact's support packet (R2)", () => {
  // ileum cancer (DOID_10153) cites the label of small intestine cancer (DOID_10154).
  const ileum = { ontology_version_id: "sha256:doid", iri: "http://purl.obolibrary.org/obo/DOID_10153", kind: "class" as const };
  const smallIntestine = { ...ileum, iri: "http://purl.obolibrary.org/obo/DOID_10154" };
  const punned = { ontology_version_id: "sha256:doid", iri: "http://purl.obolibrary.org/obo/X", kind: "individual" as const };
  const profile = Object.assign(artifact("reviewed_profile", "independently_reviewed", [ileum], { c: ["source"] }, ["source"]), {
    support_packet: { facts: [{ fact_id: "own", subject: ileum }, { fact_id: "pun", subject: punned }], selection: { referenced_labels: [{ fact_id: "label", subject: smallIntestine }] } },
  });
  const subjects = citedSubjects(profile);
  assert.deepEqual(subjects.get("label"), smallIntestine, "a referenced label is about the referenced entity");
  assert.deepEqual(subjects.get("own"), ileum);
  assert.equal(subjects.get("pun")?.kind, "individual", "a punned subject keeps its kind");
  assert.equal(subjects.get("missing"), undefined);
});
