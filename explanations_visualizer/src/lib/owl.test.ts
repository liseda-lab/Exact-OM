import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { axiomRelation, functionalSyntax, hasReading } from "./owl.ts";

// Regression fixtures modelled on earlier participant feedback (spec 08/15): a restriction on
// the same attribute with different fillers on each side, and an entity with several parents
// plus a qualified onset restriction. They test faithful presentation only; they assert
// nothing about which biomedical reading is correct.
const iri = (value: string) => ({ type: "IRI", value });
const cls = (value: string) => ({ type: "Class", iri: iri(value) });
const prop = (value: string) => ({ type: "ObjectProperty", iri: iri(value) });
const X = "https://example.org/fixture#";

test("the same attribute with different fillers keeps both qualified readings intact", () => {
  const source = { type: "SubClassOf", sub_class: cls(`${X}SourceDisease`), super_class: { type: "ObjectSomeValuesFrom", property: prop(`${X}has_onset`), filler: cls(`${X}ChildhoodOnset`) } };
  const target = { type: "SubClassOf", sub_class: cls(`${X}TargetDisease`), super_class: { type: "ObjectAllValuesFrom", property: prop(`${X}has_onset`), filler: cls(`${X}AdultOnset`) } };
  assert.equal(functionalSyntax(source), `SubClassOf(<${X}SourceDisease> ObjectSomeValuesFrom(<${X}has_onset> <${X}ChildhoodOnset>))`);
  assert.equal(functionalSyntax(target), `SubClassOf(<${X}TargetDisease> ObjectAllValuesFrom(<${X}has_onset> <${X}AdultOnset>))`);
  assert.equal(axiomRelation(source, `${X}SourceDisease`)?.relation, "Subclass of");
  assert.ok(hasReading(axiomRelation(source, `${X}SourceDisease`)!.other));
  assert.ok(hasReading(axiomRelation(target, `${X}TargetDisease`)!.other));
});

test("mixed inheritance and onset qualifiers are neither dropped nor reordered", () => {
  const equivalence = {
    type: "EquivalentClasses",
    expressions: [
      cls(`${X}Disease`),
      { type: "ObjectIntersectionOf", operands: [cls(`${X}ParentA`), cls(`${X}ParentB`), { type: "ObjectMinCardinality", cardinality: 1, property: prop(`${X}has_onset`), filler: cls(`${X}InfantOnset`) }, { type: "ObjectComplementOf", operand: cls(`${X}AdultOnset`) }] },
    ],
  };
  const text = functionalSyntax(equivalence);
  assert.equal(text, `EquivalentClasses(<${X}Disease> ObjectIntersectionOf(<${X}ParentA> <${X}ParentB> ObjectMinCardinality(1 <${X}has_onset> <${X}InfantOnset>) ObjectComplementOf(<${X}AdultOnset>)))`);
  const relation = axiomRelation(equivalence, `${X}Disease`);
  assert.equal(relation?.relation, "Equivalent to");
  assert.ok(hasReading(relation!.other), "conjunction, cardinality and negation all have conservative readings");
});

test("an unsupported constructor falls back to the original form instead of a reading", () => {
  const chain = { type: "SubObjectPropertyOf", sub_property: { type: "ObjectPropertyChain", properties: [prop(`${X}p`), prop(`${X}q`)] }, super_property: prop(`${X}r`) };
  assert.equal(axiomRelation(chain, `${X}r`), null);
  assert.equal(hasReading({ type: "DatatypeRestriction", datatype: iri("x"), restrictions: [] }), false);
  assert.match(functionalSyntax({ type: "UnknownConstructor", value: 1 }), /UnknownConstructor/);
});
