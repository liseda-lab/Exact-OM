import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { basisAvailability, navigationFromOntology, offeredBases, unsupportedConstructs } from "./navigation.ts";

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
