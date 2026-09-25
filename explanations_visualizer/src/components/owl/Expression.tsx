"use client";

// Conservative constructor readings of a typed OWL expression. Property names are shown as
// recorded; no verbs are added, so predicates like "may have" keep their own strength.

import { Fragment, useMemo } from "react";

import { curie } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import { collectIris, iriOf, isNamed } from "@/lib/owl";

type Lookup = (iri: string) => { value: string | null; status: string } | undefined;

function Term({ iri, lookup, property = false, onOpen }: { iri: string; lookup: Lookup; property?: boolean; onOpen?: (iri: string) => void }) {
  const label = lookup(iri);
  const text = label?.value ?? curie(iri);
  const missing = !label?.value && label?.status !== "loading";
  const note = label?.status === "not_included" ? " (label not in the prepared view)" : label?.status === "failed" ? " (label could not be loaded)" : " (no label in scope)";
  const content = (
    <>
      <span className={property ? "owl-property" : "owl-term"}>{text}</span>
      {missing && <span className="owl-unlabelled">{note}</span>}
    </>
  );
  if (onOpen && !property) {
    return (
      <button type="button" className="owl-link" title={iri} onClick={() => onOpen(iri)}>
        {content}
      </button>
    );
  }
  return <span title={iri}>{content}</span>;
}

function render(node: unknown, lookup: Lookup, onOpen: ((iri: string) => void) | undefined, depth: number): React.ReactNode {
  if (!node || typeof node !== "object") return String(node ?? "");
  const record = node as Record<string, unknown>;
  const type = String(record.type ?? "");
  if (isNamed(record)) {
    const iri = iriOf(record)!;
    const property = type.endsWith("Property");
    return <Term iri={iri} lookup={lookup} property={property} onOpen={onOpen} />;
  }
  if (type === "IRI") return <Term iri={String(record.value)} lookup={lookup} onOpen={onOpen} />;
  if (type === "Literal") {
    return (
      <span className="owl-literal">
        “{String(record.lexical_form)}”{record.language ? <span className="meta">@{String(record.language)}</span> : null}
      </span>
    );
  }
  const wrap = (content: React.ReactNode) => (depth > 0 ? <span className="owl-group">({content})</span> : content);
  const operands = (record.operands ?? record.individuals ?? record.values ?? record.properties) as unknown[] | undefined;
  switch (type) {
    case "ObjectSomeValuesFrom":
    case "ObjectAllValuesFrom":
    case "DataSomeValuesFrom":
    case "DataAllValuesFrom": {
      const property = record.property ?? (record.properties as unknown[] | undefined)?.[0];
      return wrap(
        <>
          {render(property, lookup, onOpen, depth + 1)} <span className="owl-keyword">{type.includes("Some") ? "some" : "only"}</span>{" "}
          {render(record.filler, lookup, onOpen, depth + 1)}
        </>,
      );
    }
    case "ObjectHasValue":
    case "DataHasValue":
      return wrap(
        <>
          {render(record.property, lookup, onOpen, depth + 1)} <span className="owl-keyword">value</span> {render(record.value, lookup, onOpen, depth + 1)}
        </>,
      );
    case "ObjectHasSelf":
      return wrap(
        <>
          {render(record.property, lookup, onOpen, depth + 1)} <span className="owl-keyword">itself</span>
        </>,
      );
    case "ObjectMinCardinality":
    case "ObjectMaxCardinality":
    case "ObjectExactCardinality":
    case "DataMinCardinality":
    case "DataMaxCardinality":
    case "DataExactCardinality": {
      const word = type.includes("Min") ? "at least" : type.includes("Max") ? "at most" : "exactly";
      return wrap(
        <>
          {render(record.property, lookup, onOpen, depth + 1)} <span className="owl-keyword">{word}</span> {String(record.cardinality)}
          {record.filler ? <> {render(record.filler, lookup, onOpen, depth + 1)}</> : null}
        </>,
      );
    }
    case "ObjectIntersectionOf":
    case "DataIntersectionOf":
    case "ObjectUnionOf":
    case "DataUnionOf": {
      const joiner = type.includes("Intersection") ? "and" : "or";
      const list = record._list === true;
      return (list ? (content: React.ReactNode) => content : wrap)(
        (operands ?? []).map((item, index) => (
          <Fragment key={index}>
            {index > 0 && <> <span className="owl-keyword">{list ? "·" : joiner}</span> </>}
            {render(item, lookup, onOpen, depth + 1)}
          </Fragment>
        )),
      );
    }
    case "ObjectComplementOf":
    case "DataComplementOf":
      return wrap(
        <>
          <span className="owl-keyword">not</span> {render(record.operand, lookup, onOpen, depth + 1)}
        </>,
      );
    case "ObjectInverseOf":
      return wrap(
        <>
          <span className="owl-keyword">inverse of</span> {render(record.property, lookup, onOpen, depth + 1)}
        </>,
      );
    case "ObjectOneOf":
    case "DataOneOf":
      return wrap(
        <>
          <span className="owl-keyword">one of</span>{" "}
          {(operands ?? []).map((item, index) => (
            <Fragment key={index}>
              {index > 0 && ", "}
              {render(item, lookup, onOpen, depth + 1)}
            </Fragment>
          ))}
        </>,
      );
    default:
      return null;
  }
}

/** True when every constructor in the expression has a reading template. */
export function hasReading(node: unknown): boolean {
  if (!node || typeof node !== "object") return true;
  const record = node as Record<string, unknown>;
  const type = String(record.type ?? "");
  const known =
    isNamed(record) ||
    ["IRI", "Literal"].includes(type) ||
    /^(Object|Data)(SomeValuesFrom|AllValuesFrom|HasValue|HasSelf|MinCardinality|MaxCardinality|ExactCardinality|IntersectionOf|UnionOf|ComplementOf|InverseOf|OneOf)$/.test(type);
  if (!known) return false;
  return Object.entries(record).every(([key, value]) => key === "iri" || key === "annotations" || (Array.isArray(value) ? value.every(hasReading) : typeof value === "object" ? hasReading(value) : true));
}

export function Expression({ node, ontology, onOpen }: { node: unknown; ontology: string; onOpen?: (iri: string) => void }) {
  const iris = useMemo(() => Array.from(collectIris(node)), [node]);
  const lookup = useLabelLookup(ontology, iris) as Lookup;
  return <span className="owl-expression">{render(node, lookup, onOpen, 0)}</span>;
}
