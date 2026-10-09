import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { buildGraphModel, entityNodeId, layoutGraph, LITERAL_LIMIT } from "./evidenceGraphModel.ts";

const S = "ont:source/1";
const T = "ont:target/1";
const source = { ontology_version_id: S, iri: "https://s#A", kind: "class" as const };
const target = { ontology_version_id: T, iri: "https://t#B", kind: "class" as const };
const named = (iri: string, type = "Class", kind = "class") => ({ type, iri: { type: "IRI", value: iri }, kind });
const sub = (child: string, parent: unknown) => ({ ast: { type: "SubClassOf", sub_class: named(child), super_class: parent } });
const label = (subject: string, text: string) => ({ ast: { type: "AnnotationAssertion", property: { type: "AnnotationProperty", iri: { type: "IRI", value: "http://www.w3.org/2000/01/rdf-schema#label" } }, subject: { type: "IRI", value: subject }, value: { type: "Literal", lexical_form: text } } });
const item = (id: string, side: "source" | "target", channel: string, factIds: string[]) => ({
  evidence_id: id, channel, side, role: side, entity: side === "source" ? source : target, feature_id: null, fact_ids: factIds, source_axiom_refs: [], semantic_terms: {}, historical_item_alias: null, display: {}, values: {}, interpretation: "projected", provenance_status: "recorded", status: "available", reason: null,
});

test("a channel recorded on one side gets no comparison line; both sides get one channel-level line (U15)", () => {
  const axioms = { f1: sub(source.iri, named("https://s#P")), f2: sub(target.iri, named("https://t#Q")), f3: label(target.iri, "B") };
  const model = buildGraphModel(source, target, [item("e1", "source", "hierarchy", ["f1"]), item("e2", "target", "hierarchy", ["f2"]), item("e3", "target", "lexical", ["f3"])] as never, axioms);
  const bridges = model.edges.filter((edge: { kind: string }) => edge.kind === "bridge");
  assert.deepEqual(bridges.map((edge: { channel?: string }) => edge.channel), ["hierarchy"]);
  assert.deepEqual(model.oneSided, [{ channel: "lexical", side: "target" }]);
  assert.ok(bridges.every((edge: { evidenceId?: string; factId?: string }) => !edge.evidenceId && !edge.factId), "a bridge is not an original fact");
});

test("node identity keeps side, ontology, kind and IRI: punned entities and shared IRIs never merge", () => {
  const punned = "https://s#X";
  const axioms = {
    f1: sub(source.iri, named(punned)),
    f2: sub(source.iri, { type: "ObjectSomeValuesFrom", property: named("https://s#p", "ObjectProperty", "object_property"), filler: named(punned) }),
  };
  const model = buildGraphModel(source, target, [item("e1", "source", "hierarchy", ["f1"]), item("e2", "source", "relational", ["f2"])] as never, axioms);
  const ids = model.nodes.map((node: { id: string }) => node.id);
  assert.equal(ids.filter((id: string) => id.endsWith(punned)).length, 1, "the same typed class reached twice is one node");
  assert.ok(ids.includes(entityNodeId("source", { ontology_version_id: S, kind: "class", iri: punned })));
  const sameIri = buildGraphModel(source, { ...target, iri: source.iri, ontology_version_id: S }, [] as never, {});
  assert.equal(sameIri.nodes.length, 2, "an entity compared with itself is still one node per side");
  const property = buildGraphModel(source, target, [item("e3", "source", "relational", ["f3"])] as never, { f3: sub(source.iri, named(punned, "ObjectProperty", "object_property")) });
  assert.ok(property.nodes.some((node: { kind: string; iri?: string }) => node.iri === punned && node.kind === "object_property"));
});

test("every feature line keeps its evidence and record identity; unsupported forms are counted, not dropped", () => {
  const axioms = { f1: sub(source.iri, named("https://s#P")), f2: { ast: { type: "EquivalentClasses", operands: [] } }, f3: null };
  const model = buildGraphModel(source, target, [item("e1", "source", "hierarchy", ["f1", "f2", "f3"])] as never, axioms);
  const features = model.edges.filter((edge: { kind: string }) => edge.kind === "feature");
  assert.deepEqual(features.map((edge: { evidenceId?: string; factId?: string }) => [edge.evidenceId, edge.factId]), [["e1", "f1"]]);
  assert.deepEqual(model.omitted, { source: 2, target: 0 });
});

test("restrictions keep their quantifier; cardinalities are not shortened into another statement", () => {
  const axioms = {
    f1: sub(source.iri, { type: "ObjectAllValuesFrom", property: named("https://s#holds", "ObjectProperty"), filler: named("https://s#Round") }),
    f2: sub(source.iri, { type: "ObjectMinCardinality", cardinality: 2, property: named("https://s#holds", "ObjectProperty"), filler: named("https://s#Item") }),
  };
  const model = buildGraphModel(source, target, [item("e1", "source", "relational", ["f1", "f2"])] as never, axioms);
  assert.deepEqual(model.edges.map((edge: { relation: { quantifier?: string } }) => edge.relation.quantifier), ["only", "at least 2"]);
});

test("many literal records are grouped behind a count and can be expanded; each record keeps its own node", () => {
  const facts = Array.from({ length: 6 }, (_, index) => `l${index}`);
  const axioms = Object.fromEntries(facts.map((id, index) => [id, label(target.iri, `name ${index}`)]));
  const grouped = buildGraphModel(source, target, [item("e1", "target", "lexical", facts)] as never, axioms);
  assert.equal(grouped.nodes.filter((node: { role: string }) => node.role === "literal").length, LITERAL_LIMIT - 1);
  assert.equal(grouped.nodes.find((node: { role: string }) => node.role === "group")?.count, 6 - (LITERAL_LIMIT - 1));
  const expanded = buildGraphModel(source, target, [item("e1", "target", "lexical", facts)] as never, axioms, { expandLiterals: new Set(["target"]) });
  assert.equal(expanded.nodes.filter((node: { role: string }) => node.role === "literal").length, 6);
  assert.ok(expanded.nodes.every((node: { role: string; id: string }) => node.role !== "literal" || node.id.startsWith("lit:target:")));
});

test("the layout separates the two sides and never overlaps nodes in a column", () => {
  const facts = Array.from({ length: 12 }, (_, index) => `p${index}`);
  const axioms = Object.fromEntries(facts.map((id, index) => [id, sub(source.iri, named(`https://s#P${index}`))]));
  const model = buildGraphModel(source, target, [item("e1", "source", "hierarchy", facts)] as never, axioms);
  const size = (node: { id: string }) => ({ width: 220, height: node.id.endsWith("P3") ? 90 : 50 });
  const positions = layoutGraph(model.nodes, size);
  const sourceNodes = model.nodes.filter((node: { side: string }) => node.side === "source");
  assert.ok(sourceNodes.every((node: { id: string }) => positions[node.id].x < 0));
  assert.ok(positions[entityNodeId("target", target)].x > 0);
  const column = sourceNodes.filter((node: { depth: number }) => node.depth === 1).sort((a: { id: string }, b: { id: string }) => positions[a.id].y - positions[b.id].y);
  for (let index = 1; index < column.length; index += 1) {
    const above = column[index - 1];
    const below = column[index];
    assert.ok(positions[below.id].y - size(below).height / 2 >= positions[above.id].y + size(above).height / 2, "no overlap");
  }
});
