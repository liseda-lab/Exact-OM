import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { basisAvailability, entryStart, navigationFromOntology, offeredBases, unsupportedConstructs } from "./navigation.ts";

const ONT = "sha256:doid";
const partial = {
  status: "partial",
  supported_kinds: ["class"],
  unsupported_constructs: [
    { features: ["OBJECT_EXACT_CARDINALITY"], task: "class_taxonomy" },
    { features: ["OBJECT_UNION_OF_POSITIVE"], task: "class_taxonomy" },
    { features: ["OBJECT_EXACT_CARDINALITY"], task: "consistency" },
  ],
};
const caps = (navigation: ReturnType<typeof navigationFromOntology>) => ({
  search: "available", navigation: "available", bases: ["literal_asserted", "structural_navigation", "reasoner_inferred", "elk_inferred_partial"], reasoner: "per_ontology",
  graphExpansion: true, profiles: "available", comparison: "available", evidence: "available", scopeNote: null, limitations: [], synthetic: false, navigationByOntology: { [ONT]: navigation },
}) as never;

test("the main API's failed complete reasoner and partial ELK are normalized separately (23 C02)", () => {
  const navigation = navigationFromOntology({ navigation: { contract_version: "exact-hierarchy/1", entry_points: "available" }, reasoner_inferred: { status: "failed", reason: "Native hierarchy preparation did not complete" }, elk_inferred_partial: partial });
  assert.equal(navigation.inference?.status, "failed");
  assert.equal(navigation.partial?.status, "partial");
  assert.equal(navigation.entryPoints, "available");
  // The failed complete mode is not offered as a disabled competitor; partial ELK is.
  assert.deepEqual(offeredBases(caps(navigation), ONT), ["literal_asserted", "structural_navigation", "elk_inferred_partial"]);
  assert.deepEqual(basisAvailability(caps(navigation), ONT, "class", "elk_inferred_partial"), { available: true, reason: null });
  assert.equal(basisAvailability(caps(navigation), ONT, "object_property", "elk_inferred_partial").available, false, "classes only");
  assert.equal(basisAvailability(caps(navigation), ONT, "class", "reasoner_inferred").available, false);
  assert.deepEqual(unsupportedConstructs(partial as never), ["exact cardinality restrictions", "unions (OR) in class definitions"]);
});

test("without any inferred result one disabled Inferred option explains why", () => {
  const navigation = navigationFromOntology({ reasoner_inferred: { status: "not_run" } });
  assert.deepEqual(offeredBases(caps(navigation), ONT), ["literal_asserted", "structural_navigation", "reasoner_inferred"]);
  assert.equal(basisAvailability(caps(navigation), ONT, "class", "reasoner_inferred").available, false);
  assert.equal(basisAvailability(caps(navigation), ONT, "class", "elk_inferred_partial").available, false);
});

// HR4: the first entry page as the real DOID/NCIT services return it, and contract variants.
const DOID = "sha256:doid";
const disease = { ontology_version_id: DOID, iri: "http://purl.obolibrary.org/obo/DOID_4", kind: "class" };
const root = (iri: string, ontology = DOID) => ({ entity: { ontology_version_id: ontology, iri, kind: "class" }, entry_kind: "named_root", component_size: 1, preferred_label: null });
const entryPage = (extra: Record<string, unknown>) =>
  ({ items: [root(disease.iri)], returned_count: 1, total_count: 4189, next_cursor: "c1", truncated: true, scope: { ontology_version_id: DOID }, status: "available", reason: null, basis: "literal_asserted", default_entity: disease, ...extra }) as never;
const curated = { kind: "curated_default", revision: "review-navigation/20261013", rationale: "Curated review starting point: disease.", assertion: "navigation_only" };

test("a declared curated default starts there even with more top-level entries to load (HR4)", () => {
  assert.deepEqual(entryStart(entryPage({ selection: curated }), DOID, "class"), { kind: "curated", entity: disease });
});

test("default_entity alone is only the first prepared entry, never a curated choice (HR4)", () => {
  // DOID under partial ELK and NCIT under Asserted: several roots, prepared order, a first default.
  const many = { items: [root(disease.iri), root("urn:other")], returned_count: 2, total_count: 4183 };
  assert.deepEqual(entryStart(entryPage({ ...many, selection: { kind: "prepared_order", assertion: "navigation_only" } }), DOID, "class"), { kind: "list" });
  assert.deepEqual(entryStart(entryPage({ ...many, selection: undefined }), DOID, "class"), { kind: "list" });
});

test("older pages without selection keep the one-root rule (HR4, 21 F06)", () => {
  assert.deepEqual(entryStart(entryPage({ next_cursor: null, total_count: 1 }), DOID, "class"), { kind: "single", entity: disease });
  assert.deepEqual(entryStart(entryPage({}), DOID, "class"), { kind: "list" }, "one item with a cursor is not a single root");
});

test("a missing, foreign, mistyped or non-navigation curated default is not applied (HR4)", () => {
  assert.deepEqual(entryStart(entryPage({ selection: curated, default_entity: null }), DOID, "class"), { kind: "list" });
  assert.deepEqual(entryStart(entryPage({ selection: curated, default_entity: { ...disease, ontology_version_id: "sha256:other" } }), DOID, "class"), { kind: "list" });
  assert.deepEqual(entryStart(entryPage({ selection: curated }), DOID, "individual"), { kind: "list" });
  assert.deepEqual(entryStart(entryPage({ selection: { ...curated, assertion: "subclass_fact" } }), DOID, "class"), { kind: "list" });
  assert.deepEqual(entryStart(entryPage({ selection: curated, scope: { ontology_version_id: "sha256:other" } }), DOID, "class"), { kind: "list" });
  for (const status of ["partial", "not_run", "not_exported", "unsupported"]) assert.deepEqual(entryStart(entryPage({ selection: curated, status }), DOID, "class"), { kind: "list" }, status);
  assert.deepEqual(entryStart(null, DOID, "class"), { kind: "list" });
});
