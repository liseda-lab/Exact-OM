// Presentation of the backend's typed OWL AST. This never parses ontology source text:
// it prints the already-structured axiom as OWL 2 Functional Syntax for verification, and
// exposes a few conservative constructor readings ("some", "only", "at least n").

import { curie } from "@/lib/iri";
import type { OwlNode } from "@/lib/types";

const ENTITY_TYPES = new Set(["Class", "ObjectProperty", "DataProperty", "AnnotationProperty", "NamedIndividual", "Datatype", "Entity"]);

const ORDER: Record<string, string[]> = {
  SubClassOf: ["sub_class", "super_class"],
  EquivalentClasses: ["expressions"],
  DisjointClasses: ["expressions"],
  DisjointUnion: ["defined_class", "expressions"],
  AnnotationAssertion: ["property", "subject", "value"],
  Annotation: ["property", "value"],
  ClassAssertion: ["class_expression", "individual"],
  ObjectPropertyAssertion: ["property", "source", "target"],
  NegativeObjectPropertyAssertion: ["property", "source", "target"],
  DataPropertyAssertion: ["property", "source", "value"],
  NegativeDataPropertyAssertion: ["property", "source", "value"],
  ObjectPropertyDomain: ["property", "domain"],
  ObjectPropertyRange: ["property", "range"],
  DataPropertyDomain: ["property", "domain"],
  DataPropertyRange: ["property", "range"],
  AnnotationPropertyDomain: ["property", "domain"],
  AnnotationPropertyRange: ["property", "range"],
  SubObjectPropertyOf: ["sub_property", "super_property"],
  SubDataPropertyOf: ["sub_property", "super_property"],
  SubAnnotationPropertyOf: ["sub_property", "super_property"],
  EquivalentObjectProperties: ["properties"],
  DisjointObjectProperties: ["properties"],
  EquivalentDataProperties: ["properties"],
  DisjointDataProperties: ["properties"],
  InverseObjectProperties: ["first", "second"],
  FunctionalObjectProperty: ["property"],
  InverseFunctionalObjectProperty: ["property"],
  ReflexiveObjectProperty: ["property"],
  IrreflexiveObjectProperty: ["property"],
  SymmetricObjectProperty: ["property"],
  AsymmetricObjectProperty: ["property"],
  TransitiveObjectProperty: ["property"],
  FunctionalDataProperty: ["property"],
  SameIndividual: ["individuals"],
  DifferentIndividuals: ["individuals"],
  HasKey: ["class_expression", "object_properties", "data_properties"],
  DatatypeDefinition: ["datatype", "data_range"],
  Declaration: ["entity"],
  ObjectSomeValuesFrom: ["property", "filler"],
  ObjectAllValuesFrom: ["property", "filler"],
  ObjectHasValue: ["property", "value"],
  ObjectHasSelf: ["property"],
  ObjectMinCardinality: ["cardinality", "property", "filler"],
  ObjectMaxCardinality: ["cardinality", "property", "filler"],
  ObjectExactCardinality: ["cardinality", "property", "filler"],
  ObjectIntersectionOf: ["operands"],
  ObjectUnionOf: ["operands"],
  ObjectComplementOf: ["operand"],
  ObjectOneOf: ["individuals"],
  ObjectInverseOf: ["property"],
  ObjectPropertyChain: ["properties"],
  DataSomeValuesFrom: ["properties", "filler"],
  DataAllValuesFrom: ["properties", "filler"],
  DataHasValue: ["property", "value"],
  DataMinCardinality: ["cardinality", "property", "filler"],
  DataMaxCardinality: ["cardinality", "property", "filler"],
  DataExactCardinality: ["cardinality", "property", "filler"],
  DataIntersectionOf: ["operands"],
  DataUnionOf: ["operands"],
  DataComplementOf: ["operand"],
  DataOneOf: ["values"],
  DatatypeRestriction: ["datatype", "restrictions"],
  FacetRestriction: ["facet", "value"],
};

export function iriOf(node: unknown): string | null {
  if (!node || typeof node !== "object") return null;
  const record = node as Record<string, unknown>;
  if (record.type === "IRI" && typeof record.value === "string") return record.value;
  return iriOf(record.iri);
}

export function isNamed(node: unknown): boolean {
  return Boolean(node && typeof node === "object" && ENTITY_TYPES.has((node as OwlNode).type) && iriOf(node));
}

function literal(node: Record<string, unknown>, shorten: (iri: string) => string): string {
  const text = JSON.stringify(String(node.lexical_form ?? ""));
  if (node.language) return `${text}@${node.language}`;
  const datatype = iriOf(node.datatype);
  return datatype ? `${text}^^${shorten(datatype)}` : text;
}

/** Faithful OWL 2 Functional Syntax for a typed AST; unknown shapes fall back to JSON. */
export function functionalSyntax(node: unknown, options: { abbreviate?: boolean } = {}): string {
  const shorten = (iri: string) => {
    if (!options.abbreviate) return `<${iri}>`;
    const short = curie(iri);
    return short === iri ? `<${iri}>` : short;
  };
  const print = (value: unknown, inDeclaration = false): string => {
    if (Array.isArray(value)) return value.map((item) => print(item)).join(" ");
    if (value === null || value === undefined) return "";
    if (typeof value === "number" || typeof value === "string") return String(value);
    const record = value as Record<string, unknown>;
    const type = String(record.type ?? "");
    if (type === "IRI") return shorten(String(record.value));
    if (type === "Literal") return literal(record, shorten);
    if (type === "AnonymousIndividual") {
      const key = record.local_key as { value?: string } | undefined;
      return `_:b${key?.value?.slice(0, 12) ?? ""}`;
    }
    if (ENTITY_TYPES.has(type)) {
      const iri = shorten(iriOf(record) ?? "");
      return inDeclaration ? `${type}(${iri})` : iri;
    }
    const order = ORDER[type];
    if (!order) return JSON.stringify(record);
    const parts: string[] = [];
    const annotations = record.annotations as unknown[] | undefined;
    if (annotations && annotations.length) parts.push(annotations.map((a) => print(a)).join(" "));
    for (const field of order) {
      const fieldValue = record[field];
      if (fieldValue === undefined || fieldValue === null) continue;
      if (type === "HasKey" && field !== "class_expression") parts.push(`(${print(fieldValue)})`);
      else parts.push(print(fieldValue, type === "Declaration"));
    }
    return `${type}(${parts.filter(Boolean).join(" ")})`;
  };
  return print(node);
}

/** Every named IRI mentioned by an AST, for batched label lookup. */
export function collectIris(node: unknown, into: Set<string> = new Set()): Set<string> {
  if (Array.isArray(node)) {
    node.forEach((item) => collectIris(item, into));
    return into;
  }
  if (!node || typeof node !== "object") return into;
  const record = node as Record<string, unknown>;
  if (record.type === "IRI" && typeof record.value === "string") {
    into.add(record.value);
    return into;
  }
  for (const [field, value] of Object.entries(record)) {
    if (field === "annotations") continue;
    collectIris(value, into);
  }
  return into;
}

/** Heading for an axiom relative to the entity it describes; wording adds no verbs. */
export function axiomRelation(ast: OwlNode, subjectIri: string): { relation: string; other: unknown } | null {
  const type = ast.type;
  if (type === "SubClassOf") {
    if (iriOf(ast.sub_class) === subjectIri) return { relation: "Subclass of", other: ast.super_class };
    if (iriOf(ast.super_class) === subjectIri) return { relation: "Has subclass", other: ast.sub_class };
  }
  if (type === "EquivalentClasses") {
    const others = (ast.expressions as unknown[]).filter((item) => iriOf(item) !== subjectIri);
    if (others.length === 1) return { relation: "Equivalent to", other: others[0] };
    return { relation: "Equivalent to", other: { type: "ObjectIntersectionOf", operands: others, _list: true } };
  }
  if (type === "DisjointClasses") {
    const others = (ast.expressions as unknown[]).filter((item) => iriOf(item) !== subjectIri);
    return { relation: "Disjoint with", other: others.length === 1 ? others[0] : { type: "ObjectUnionOf", operands: others, _list: true } };
  }
  if (type === "ClassAssertion") return { relation: "Instance of", other: ast.class_expression };
  if (type === "ObjectPropertyDomain" || type === "DataPropertyDomain") return { relation: "Domain", other: ast.domain };
  if (type === "ObjectPropertyRange" || type === "DataPropertyRange") return { relation: "Range", other: ast.range };
  return null;
}
