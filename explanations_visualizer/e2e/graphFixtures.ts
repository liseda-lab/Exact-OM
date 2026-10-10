import type { Page } from "@playwright/test";

// Synthetic evidence-graph shapes for the main app (S0 follow-up). Only the pair-evidence,
// axiom and label READ responses of an isolated test browser are overridden; the two compared
// entities, their ontologies and the served app are real. Every invented record uses an
// example.org IRI or an obviously synthetic fact ID: these are UI test shapes, not ontology
// findings or matcher results, and nothing here is written anywhere.

export const EX = "https://example.org/review#";
type Node = Record<string, unknown>;
type Side = "source" | "target";

export const cls = (local: string): Node => ({ type: "Class", iri: { type: "IRI", value: EX + local }, kind: "class" });
export const prop = (local: string): Node => ({ type: "ObjectProperty", iri: { type: "IRI", value: EX + local }, kind: "object_property" });
export const some = (property: string, filler: Node): Node => ({ type: "ObjectSomeValuesFrom", property: prop(property), filler });
export const only = (property: string, filler: Node): Node => ({ type: "ObjectAllValuesFrom", property: prop(property), filler });
export const atLeast = (n: number, property: string, filler: Node): Node => ({ type: "ObjectMinCardinality", cardinality: n, property: prop(property), filler });
export const exactly = (n: number, property: string, filler: Node): Node => ({ type: "ObjectExactCardinality", cardinality: n, property: prop(property), filler });

export interface SyntheticFact {
  id: string;
  /** SubClassOf(compared entity, superClass): a named parent or a restriction. */
  superClass?: Node;
  /** An annotation literal on the compared entity. */
  annotation?: { property: string; text: string };
  /** A record with no graph form (an unsupported axiom shape). */
  unsupported?: boolean;
}

export interface SyntheticFeature {
  evidenceId: string;
  side: Side;
  channel: string;
  facts: SyntheticFact[];
  status?: "available" | "unresolved";
}

export const factId = (name: string) => `sha256:${Buffer.from(name).toString("hex").padEnd(64, "0").slice(0, 64)}`;

/**
 * Replace the main app's pair evidence with `features` (for whichever pair is opened first) and
 * serve their axioms and synthetic labels. Returns the compared entities once known.
 */
export async function syntheticGraph(page: Page, features: SyntheticFeature[], labels: Record<string, string> = {}) {
  const records = new Map<string, unknown>();
  const focal: Partial<Record<Side, { ontology_version_id: string; iri: string; kind: string }>> = {};
  await page.route("**/api/v1/axioms/**", async (route) => {
    const id = decodeURIComponent(new URL(route.request().url()).pathname.split("/").at(-1)!);
    if (records.has(id)) return route.fulfill({ json: records.get(id) });
    return route.continue();
  });
  await page.route("**/api/v1/labels?*", async (route) => {
    const url = new URL(route.request().url());
    const wanted = url.searchParams.getAll("iri").filter((iri) => labels[iri]);
    if (!wanted.length) return route.continue();
    const original = await route.fetch();
    const payload = await original.json();
    const ontology = url.searchParams.get("ontology_version_id");
    payload.items.push(...wanted.map((iri) => ({ entity: { ontology_version_id: ontology, iri, kind: "class" }, iri: null, preferred_label: { status: "available", value: labels[iri], language: "en" } })));
    return route.fulfill({ response: original, json: payload });
  });
  await page.route("**/pair-evidence?*", async (route) => {
    const original = await route.fetch();
    const payload = await original.json();
    const template = payload.items[0];
    for (const side of ["source", "target"] as Side[]) focal[side] = payload.items.find((item: { side: Side }) => item.side === side)?.entity ?? focal[side];
    const axiom = await (await page.request.get(`/api/v1/axioms/${encodeURIComponent(template.fact_ids[0])}?${new URLSearchParams({ ontology_version_id: template.entity.ontology_version_id })}`)).json();
    for (const feature of features) {
      const entity = focal[feature.side]!;
      const subject = { type: "Class", iri: { type: "IRI", value: entity.iri }, kind: entity.kind };
      for (const fact of feature.facts) {
        const ast = fact.unsupported
          ? { type: "EquivalentClasses", annotations: [], expressions: [subject, { type: "ObjectIntersectionOf", operands: [cls("A"), cls("B")] }] }
          : fact.annotation
            ? { type: "AnnotationAssertion", annotations: [], property: { type: "AnnotationProperty", iri: { type: "IRI", value: fact.annotation.property } }, subject: { type: "IRI", value: entity.iri }, value: { type: "Literal", lexical_form: fact.annotation.text, language: "en" } }
            : { type: "SubClassOf", annotations: [], sub_class: subject, super_class: fact.superClass };
        records.set(fact.id, { ...axiom, id: fact.id, axiom_id: fact.id, fact_id: fact.id, ontology_version_id: entity.ontology_version_id, category: fact.annotation ? "annotations" : "restrictions", ast });
      }
    }
    payload.items = features.map((feature) => ({
      ...template,
      evidence_id: feature.evidenceId,
      channel: feature.channel,
      side: feature.side,
      entity: focal[feature.side],
      fact_ids: feature.status === "unresolved" ? [] : feature.facts.map((fact) => fact.id),
      status: feature.status ?? "available",
      reason: feature.status === "unresolved" ? "Synthetic test: the original facts could not be resolved." : null,
    }));
    payload.returned_count = payload.items.length;
    payload.total_count = payload.items.length;
    return route.fulfill({ response: original, json: payload });
  });
  return focal;
}

/** Read the live Cytoscape graph: elements with their data and resolved style (what is drawn). */
export async function drawnGraph(page: Page) {
  return page.locator(".graph-canvas").evaluate((element) => {
    type Element = { id: () => string; isNode: () => boolean; data: (key: string) => unknown; style: (key: string) => string; selected: () => boolean; position: () => { x: number; y: number }; hasClass: (name: string) => boolean; source: () => Element; target: () => Element };
    const cy = (element as unknown as { _cyreg: { cy: { elements: () => { map: <T>(fn: (item: Element) => T) => T[] }; zoom: () => number; background?: unknown } } })._cyreg.cy;
    const canvas = getComputedStyle(element).backgroundColor;
    return {
      zoom: cy.zoom(),
      canvas,
      elements: cy.elements().map((item) => ({
        id: item.id(),
        node: item.isNode(),
        source: item.isNode() ? null : item.source().id(),
        target: item.isNode() ? null : item.target().id(),
        role: item.data("role") as string | undefined,
        kind: item.data("kind") as string | undefined,
        label: String(item.data("label") ?? ""),
        drawnLabel: item.style("label"),
        color: item.style("color"),
        background: item.isNode() ? item.style("background-color") : item.style("text-background-color"),
        opacity: Number(item.style("opacity")),
        textOpacity: Number(item.style("text-opacity")),
        fontSize: item.style("font-size"),
        width: item.isNode() ? Number(String(item.style("width")).replace("px", "")) : 0,
        height: item.isNode() ? Number(String(item.style("height")).replace("px", "")) : 0,
        x: item.isNode() ? item.position().x : 0,
        y: item.isNode() ? item.position().y : 0,
        selected: item.selected(),
        faded: item.hasClass("faded"),
      })),
    };
  });
}

/** WCAG contrast of two `rgb(...)` colours. */
export function contrast(a: string, b: string): number {
  const channel = (value: number) => {
    const c = value / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const lum = (rgb: string) => {
    const [r, g, bl] = (rgb.match(/\d+(\.\d+)?/g) ?? ["0", "0", "0"]).map(Number);
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(bl);
  };
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}
