import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { BUILDER_CATEGORIES, comparisonFor, entityContextFrom, evidenceFor, hierarchyFor, indexResources, profileFor, resolveFact, searchIn } from "./resourceIndex.ts";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { ENTITY, TUTORIAL_RESOURCE } from "../../study/v2/tutorialContent.ts";

const complete = indexResources([TUTORIAL_RESOURCE], { prepared: "all" });
// A bounded case resource like the study builder's: only the source and its five candidates.
const caseKeys = new Set(["a.crpc", "b.c1", "b.c2", "b.c3", "b.crate", "b.c5"].map((key) => ENTITY[key].iri));
const bounded = {
  ...TUTORIAL_RESOURCE,
  entities: TUTORIAL_RESOURCE.entities.filter((entity: { iri: string }) => caseKeys.has(entity.iri)),
  facts: TUTORIAL_RESOURCE.facts.filter((fact: { subject: { iri: string } }) => caseKeys.has(fact.subject.iri)),
  hierarchy: TUTORIAL_RESOURCE.hierarchy.filter((edge: { child: { iri: string } }) => caseKeys.has(edge.child.iri)),
};
const prepared = indexResources([bounded], { prepared: new Set(BUILDER_CATEGORIES) });
const source = ENTITY["a.crpc"];
const crate = ENTITY["b.crate"];
const c2 = ENTITY["b.c2"];

test("a shared original axiom keeps a copy under every typed subject (C18)", () => {
  const id = "practice.b.c2.subclass.b.crate";
  const fact = resolveFact(complete, { factId: id, ontologies: [], subject: crate });
  assert.equal(fact.status, "available");
  assert.deepEqual(fact.subjects.map((subject: { entity: { iri: string } }) => subject.entity.iri).sort(), [c2.iri, crate.iri].sort());
  assert.equal(fact.reconstructed, false, "a stored typed AST is shown as stored");
});

test("citations to labels and hidden duplicate synonyms resolve to their exact record (C03)", () => {
  const profile = profileFor(complete, crate, true);
  assert.equal(profile.status, "available");
  const cited = profile.status === "available" ? profile.explanation.claims.flatMap((claim: { fact_ids: string[] }) => claim.fact_ids) : [];
  const synonym = cited.find((id: string) => id.includes("synonym"));
  assert.ok(synonym, "the practice profile cites the synonym identical to the label");
  const context = entityContextFrom(complete, crate, "scope");
  assert.ok(context.synonyms.items.some((item: { fact_id: string }) => item.fact_id === synonym));
  const record = resolveFact(complete, { factId: synonym!, ontologies: [], subject: crate });
  assert.equal(record.literal?.text, "crate");
  assert.equal(record.reconstructed, true);
  assert.equal(record.axiom?.ast.type, "AnnotationAssertion");
  const label = resolveFact(complete, { factId: "practice.b.c5.label", ontologies: [], subject: ENTITY["b.c5"] });
  assert.equal(label.category, "labels");
});

test("an unknown record states that it is not prepared instead of inventing support", () => {
  const record = resolveFact(complete, { factId: "practice.missing", ontologies: [], subject: source });
  assert.equal(record.status, "not_exported");
  assert.match(record.reason ?? "", /not part of the information prepared/);
  assert.equal(record.axiom, null);
});

test("bounded study resources distinguish absent, unprepared and truncated information (C02, C18)", () => {
  const ctx = entityContextFrom(prepared, ENTITY["b.c5"], "scope");
  assert.equal(ctx.definitions.status, "absent_in_scope", "a prepared category with no fact is absent in scope");
  assert.equal(ctx.categories.types.status, "not_exported", "a category the builder does not prepare is not reported as absent");
  const bounded = indexResources([{ ...TUTORIAL_RESOURCE, limitations: [`Original synonyms for ${crate.iri} are bounded to the prepared page.`] }]);
  const truncated = entityContextFrom(bounded, crate, "scope");
  assert.equal(truncated.synonyms.status, "partial");
  assert.equal(truncated.synonyms.truncated, true);
});

test("prepared hierarchy navigation says where its information ends (C02)", () => {
  const children = hierarchyFor(prepared, crate, "children", "literal_asserted", "prepared");
  assert.equal(children.status, "partial", "prepared children of a focal entity are shown as a bounded page");
  const parentsOfUnprepared = hierarchyFor(prepared, ENTITY["b.container"], "parents", "literal_asserted", "prepared");
  assert.equal(parentsOfUnprepared.status, "not_exported");
  const full = hierarchyFor(complete, crate, "children", "literal_asserted", "complete");
  assert.equal(full.status, "available");
  assert.equal(full.items.length, 4);
  const sourceParents = hierarchyFor(complete, source, "parents", "literal_asserted", "complete");
  assert.equal(sourceParents.items.length, 2, "multiple inheritance is preserved");
  assert.equal(hierarchyFor(complete, source, "parents", "reasoner_inferred", "complete").status, "not_run");
});

test("evidence keeps its recorded interpretation and per-candidate scope (C04)", () => {
  const bundle = evidenceFor(complete, source, c2, "practice-c2");
  assert.ok(bundle.items.length >= 4);
  assert.ok(bundle.items.every((item: { interpretation: string }) => item.interpretation === "projected"));
  assert.ok(bundle.items.every((item: { fact_ids: string[] }) => item.fact_ids.every((id) => id in bundle.axioms)));
  const withValue = indexResources([{ ...TUTORIAL_RESOURCE, evidence: [{ ...TUTORIAL_RESOURCE.evidence[0], saved_value: 0.4, value_meaning: "feature weight" }] }]);
  const valued = evidenceFor(withValue, source, ENTITY["b.c1"], "practice-c1");
  assert.deepEqual(valued.items[0].values, { "feature weight": 0.4 });
});

test("comparisons and profiles are scoped to the inspected pair and entity", () => {
  const comparison = comparisonFor(complete, source, ENTITY["b.c5"], true);
  assert.equal(comparison.status, "available");
  if (comparison.status === "available") {
    assert.ok(comparison.explanation.claims.some((claim: { text: string }) => claim.text.includes("One-sided information does not establish incompatibility")));
    assert.equal(comparison.provenance.synthetic, true);
  }
  assert.equal(comparisonFor(complete, ENTITY["b.c1"], ENTITY["b.c5"]).status, "not_requested");
});

test("search over a bounded resource declares its partial scope", () => {
  const page = searchIn(prepared, source.ontology_version_id, "crate", true);
  assert.equal(page.status, "partial");
  assert.ok(page.items.some((item: { entity: { iri: string } }) => item.entity.iri === source.iri));
});
