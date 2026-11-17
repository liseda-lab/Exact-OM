// The evidence graph's model and layout (21 F08), separate from rendering so its meaning can be
// tested. Pure module (type-only imports) for Node's test runner.
//
// Semantics:
// - A node's identity is its presentation side, ontology version, entity kind and IRI, so a
//   punned class/property pair never merges and an entity shown on both sides stays two nodes.
//   Literal records are scoped by side and fact identity.
// - Every selected matcher feature becomes one line per original record it rests on, carrying
//   its evidence ID and fact ID. A record with no supported graph form is counted, not dropped.
// - Comparison lines come only from recorded matcher comparisons (exact-evidence-metadata/1,
//   23 C04): a recorded channel comparison joins the two compared entities; a recorded feature
//   correspondence joins the two features it names when both are drawn. Nothing is inferred
//   from coincident channels or labels; withheld or undrawable comparisons stay listed.
// - Ontology context added on request is a different kind of line and never a feature.
//
// - Each feature line carries its own relation (predicate and qualifier). Entity nodes show only
//   the entity, so two relationships that share a target never borrow each other's meaning
//   (S0-R1); literal records keep their predicate in their own fact-scoped node.
// - Every selected record has a status (drawn, grouped behind a count, no graph form, or not
//   available), so a selection from the evidence list can reveal exactly that record or say why
//   it is not drawn (S0-R3).
//
// Layout: source on the left, target on the right, each side's features in a column outside its
// compared entity, ordered by kind, with room between for the relation labels; expansions in
// further columns. Heights come from the label lines, and each column is swept so nothing
// overlaps (no fixed wide rows). Distances scale with the reader's text size.

import type { EntityKind, EntityRef, EvidenceBridge, OwlNode, PairEvidenceMetadata, SelectedEvidence } from "./types";

export type GraphSide = "source" | "target";
export type ModelEdgeKind = "feature" | "asserted" | "structural" | "inferred" | "bridge" | "group";

export interface ModelNode {
  id: string;
  side: GraphSide;
  role: "focal" | "context" | "literal" | "group";
  ontology: string;
  kind: EntityKind;
  iri?: string;
  /** Literal text (records) or a group's hidden count. */
  text?: string;
  count?: number;
  /** Column distance from the compared entity: 0 focal, 1 features, 2+ expansions. */
  depth: number;
  /** Ordering key within a column. */
  order: number;
  anchor?: string;
}

export interface ModelEdge {
  id: string;
  source: string;
  target: string;
  kind: ModelEdgeKind;
  side?: GraphSide;
  channel?: string;
  evidenceId?: string;
  factId?: string;
  /** What the line says: a fixed relation, or a property plus quantifier to label later. */
  relation: { text?: string; property?: string; quantifier?: string };
  /** A recorded matcher comparison (bridge lines only). */
  bridge?: EvidenceBridge;
}

/** Where one recorded comparison ended up in the graph. */
export interface BridgeRecord {
  bridge: EvidenceBridge;
  status: "drawn" | "not_drawn" | "withheld";
  edgeId?: string;
  /** Identical recorded comparisons (everything but their IDs) shown once; at least 1. */
  copies: number;
}

/** Where one selected record ended up in the graph. */
export interface ModelRecord {
  evidenceId: string;
  factId: string | null;
  side: GraphSide;
  channel: string;
  status: "drawn" | "grouped" | "no_graph_form" | "unavailable";
  /** The feature line for a drawn or grouped record (grouped lines appear when the group is shown). */
  edgeId?: string;
}

export interface GraphModel {
  nodes: ModelNode[];
  edges: ModelEdge[];
  /** Every selected record, in evidence order, including those not drawn. */
  records: ModelRecord[];
  /** Records with no supported graph form, per side (they stay in the evidence list). */
  omitted: { source: number; target: number };
  /** Channels with features on one side only. */
  oneSided: { channel: string; side: GraphSide }[];
  /** Every recorded comparison, drawn or not; empty when the pair has no recorded metadata. */
  bridges: BridgeRecord[];
  /** The pair's recorded comparison status; `unknown` when no metadata was supplied. */
  bridgeStatus: PairEvidenceMetadata["bridge_status"] | "unknown";
}

export function entityNodeId(side: GraphSide, entity: { ontology_version_id: string; kind: string; iri: string }) {
  return `n:${side}:${entity.ontology_version_id}:${entity.kind}:${entity.iri}`;
}

const ENTITY_KIND: Record<string, EntityKind> = { Class: "class", ObjectProperty: "object_property", DataProperty: "data_property", NamedIndividual: "individual" };

function iriOf(node: unknown): string | null {
  if (!node || typeof node !== "object") return null;
  const record = node as Record<string, unknown>;
  if (record.type === "IRI" && typeof record.value === "string") return record.value;
  return iriOf(record.iri);
}

/** The recorded kind of a named AST node, or null when it is not a named entity. */
export function namedKind(node: unknown): EntityKind | null {
  if (!node || typeof node !== "object" || !iriOf(node)) return null;
  const record = node as { type?: string; kind?: string };
  const kind = record.kind === "named_individual" ? "individual" : record.kind;
  if (kind && ["class", "object_property", "data_property", "individual"].includes(kind)) return kind as EntityKind;
  return ENTITY_KIND[record.type ?? ""] ?? null;
}

const QUANTIFIERS: Record<string, string> = {
  ObjectSomeValuesFrom: "some",
  ObjectAllValuesFrom: "only",
  DataSomeValuesFrom: "some",
  DataAllValuesFrom: "only",
  ObjectHasValue: "value",
};

function cardinality(node: OwlNode): string | null {
  const n = (node as { cardinality?: number }).cardinality;
  if (typeof n !== "number") return null;
  if (/MinCardinality$/.test(node.type)) return `at least ${n}`;
  if (/MaxCardinality$/.test(node.type)) return `at most ${n}`;
  if (/ExactCardinality$/.test(node.type)) return `exactly ${n}`;
  return null;
}

/** Literal features beyond this many per side are grouped behind an expandable count. */
export const LITERAL_LIMIT = 3;

export function buildGraphModel(
  source: EntityRef,
  target: EntityRef,
  items: SelectedEvidence[],
  axioms: Record<string, { ast: unknown } | null>,
  options: { expandLiterals?: Set<GraphSide>; metadata?: Pick<PairEvidenceMetadata, "bridge_status" | "bridges"> | null } = {},
): GraphModel {
  const nodes = new Map<string, ModelNode>();
  const edges: ModelEdge[] = [];
  const omitted = { source: 0, target: 0 };
  const focal = { source: entityNodeId("source", source), target: entityNodeId("target", target) };
  nodes.set(focal.source, { id: focal.source, side: "source", role: "focal", ontology: source.ontology_version_id, kind: source.kind, iri: source.iri, depth: 0, order: 0 });
  nodes.set(focal.target, { id: focal.target, side: "target", role: "focal", ontology: target.ontology_version_id, kind: target.kind, iri: target.iri, depth: 0, order: 0 });
  const entity = (side: GraphSide, ontology: string, iri: string, kind: EntityKind, order: number) => {
    const id = entityNodeId(side, { ontology_version_id: ontology, kind, iri });
    if (!nodes.has(id)) nodes.set(id, { id, side, role: "context", ontology, kind, iri, depth: 1, order });
    return id;
  };
  const sides = { source: new Set<string>(), target: new Set<string>() };
  const literals: { side: GraphSide; record: ModelRecord; build: () => void }[] = [];
  const records: ModelRecord[] = [];
  for (const item of items) {
    const side = item.side;
    sides[side].add(item.channel);
    const from = focal[side];
    const ontology = item.entity.ontology_version_id;
    if (!item.fact_ids.length) records.push({ evidenceId: item.evidence_id, factId: null, side, channel: item.channel, status: "unavailable" });
    for (const factId of item.fact_ids) {
      const ast = axioms[factId]?.ast as OwlNode | undefined;
      const base = { kind: "feature" as const, side, channel: item.channel, evidenceId: item.evidence_id, factId };
      const id = `e:${item.evidence_id}:${factId}`;
      const record: ModelRecord = { evidenceId: item.evidence_id, factId, side, channel: item.channel, status: "drawn", edgeId: id };
      records.push(record);
      if (!ast) {
        record.status = item.status === "available" ? "no_graph_form" : "unavailable";
        delete record.edgeId;
        omitted[side] += 1;
        continue;
      }
      if (ast.type === "SubClassOf" && iriOf(ast.sub_class) === item.entity.iri) {
        const sup = ast.super_class as OwlNode;
        const supKind = namedKind(sup);
        if (supKind) {
          edges.push({ ...base, id, source: from, target: entity(side, ontology, iriOf(sup)!, supKind, 0), relation: { text: "subclass of" } });
          continue;
        }
        const quantifier = QUANTIFIERS[sup?.type] ?? cardinality(sup);
        const filler = (sup as { filler?: unknown; value?: unknown })?.filler;
        const fillerKind = namedKind(filler);
        const property = iriOf((sup as { property?: unknown })?.property);
        if (quantifier && property && filler && fillerKind) {
          edges.push({ ...base, id, source: from, target: entity(side, ontology, iriOf(filler)!, fillerKind, 1), relation: { property, quantifier } });
          continue;
        }
      }
      if (ast.type === "AnnotationAssertion" && (ast.value as OwlNode)?.type === "Literal") {
        const text = String((ast.value as OwlNode).lexical_form);
        const property = iriOf(ast.property);
        literals.push({
          side,
          record,
          build: () => {
            const node = `lit:${side}:${factId}`;
            nodes.set(node, { id: node, side, role: "literal", ontology, kind: "class", text, depth: 1, order: 2 });
            edges.push({ ...base, id, source: from, target: node, relation: { property: property ?? undefined } });
          },
        });
        continue;
      }
      record.status = "no_graph_form";
      delete record.edgeId;
      omitted[side] += 1;
    }
  }
  for (const side of ["source", "target"] as GraphSide[]) {
    const own = literals.filter((item) => item.side === side);
    const shown = options.expandLiterals?.has(side) || own.length <= LITERAL_LIMIT ? own : own.slice(0, LITERAL_LIMIT - 1);
    shown.forEach((item) => item.build());
    own.filter((item) => !shown.includes(item)).forEach((item) => (item.record.status = "grouped"));
    const hidden = own.length - shown.length;
    if (hidden > 0) {
      const id = `group:${side}`;
      nodes.set(id, { id, side, role: "group", ontology: side === "source" ? source.ontology_version_id : target.ontology_version_id, kind: "class", count: hidden, depth: 1, order: 3 });
      edges.push({ id: `g:${side}`, source: focal[side], target: id, kind: "group", side, relation: { text: `${hidden} more records` } });
    }
  }
  const oneSided: GraphModel["oneSided"] = [];
  for (const channel of new Set([...sides.source, ...sides.target])) {
    if (!(sides.source.has(channel) && sides.target.has(channel))) oneSided.push({ channel, side: sides.source.has(channel) ? "source" : "target" });
  }
  // Recorded comparisons only. A feature correspondence is drawn between the two feature nodes
  // it names, when both features are drawn; otherwise it stays listed with its status.
  const featureNode = (evidenceIds: string[], side: GraphSide) => edges.find((edge) => edge.kind === "feature" && edge.side === side && edge.evidenceId && evidenceIds.includes(edge.evidenceId))?.target ?? null;
  // The same comparison can be recorded more than once with only its ID differing: it is one
  // line and one row, and keeps its multiplicity so every record is still counted.
  const seen = new Map<string, BridgeRecord>();
  const bridges: BridgeRecord[] = [];
  for (const bridge of options.metadata?.bridges ?? []) {
    const { bridge_id: _id, ...content } = bridge;
    // The service serialises bridges in one field order, so equal content gives equal text.
    const key = JSON.stringify(content);
    const earlier = seen.get(key);
    if (earlier) {
      earlier.copies += 1;
      continue;
    }
    const record = ((): BridgeRecord => {
      if (bridge.status === "withheld") return { bridge, status: "withheld", copies: 1 };
      if (bridge.level === "channel_comparison") {
        const id = `bridge:${bridge.bridge_id}`;
        edges.push({ id, source: focal.source, target: focal.target, kind: "bridge", channel: bridge.channel, relation: {}, bridge });
        return { bridge, status: "drawn", edgeId: id, copies: 1 };
      }
      const from = featureNode(bridge.source_evidence_ids, "source");
      const to = featureNode(bridge.target_evidence_ids, "target");
      if (!from || !to) return { bridge, status: "not_drawn", copies: 1 };
      const id = `corr:${bridge.bridge_id}`;
      edges.push({ id, source: from, target: to, kind: "bridge", channel: bridge.channel, relation: {}, bridge });
      return { bridge, status: "drawn", edgeId: id, copies: 1 };
    })();
    seen.set(key, record);
    bridges.push(record);
  }
  return { nodes: Array.from(nodes.values()), edges, records, omitted, oneSided, bridges, bridgeStatus: options.metadata ? options.metadata.bridge_status : "unknown" };
}

export interface Size {
  width: number;
  height: number;
}

/**
 * Column-based layout with per-column overlap sweeps. `size` gives each node's drawn box; `scale`
 * is the reader's text size, so enlarged labels get proportionally more room. The gap between
 * a compared entity and its feature column holds the feature lines' relation labels.
 */
export function layoutGraph(nodes: ModelNode[], size: (node: ModelNode) => Size, gap = 18, scale = 1): Record<string, { x: number; y: number }> {
  const positions: Record<string, { x: number; y: number }> = {};
  const column = (side: GraphSide, depth: number) => (side === "source" ? -1 : 1) * scale * (depth === 0 ? 130 : 475 + 270 * (depth - 1));
  gap *= scale;
  for (const side of ["source", "target"] as GraphSide[]) {
    const focal = nodes.find((node) => node.side === side && node.role === "focal");
    if (focal) positions[focal.id] = { x: column(side, 0), y: 0 };
    const depths = Array.from(new Set(nodes.filter((node) => node.side === side && node.depth > 0).map((node) => node.depth))).sort((a, b) => a - b);
    for (const depth of depths) {
      const members = nodes.filter((node) => node.side === side && node.depth === depth);
      // Desired centre: features stack around the compared entity; expansions beside their anchor.
      const wanted = new Map<string, number>();
      if (depth === 1) {
        const ordered = [...members].sort((a, b) => a.order - b.order);
        const total = ordered.reduce((sum, node) => sum + size(node).height, 0) + gap * Math.max(0, ordered.length - 1);
        let y = -total / 2;
        for (const node of ordered) {
          const h = size(node).height;
          wanted.set(node.id, y + h / 2);
          y += h + gap;
        }
      } else {
        const byAnchor = new Map<string, ModelNode[]>();
        members.forEach((node) => byAnchor.set(node.anchor ?? "", [...(byAnchor.get(node.anchor ?? "") ?? []), node]));
        for (const [anchor, group] of byAnchor) {
          const centre = positions[anchor]?.y ?? 0;
          const total = group.reduce((sum, node) => sum + size(node).height, 0) + gap * (group.length - 1);
          let y = centre - total / 2;
          for (const node of group) {
            const h = size(node).height;
            wanted.set(node.id, y + h / 2);
            y += h + gap;
          }
        }
      }
      // Sweep: keep the order, push down anything that would overlap its predecessor.
      const sorted = [...members].sort((a, b) => wanted.get(a.id)! - wanted.get(b.id)!);
      let previousBottom = -Infinity;
      for (const node of sorted) {
        const h = size(node).height;
        const y = Math.max(wanted.get(node.id)!, previousBottom + gap + h / 2);
        positions[node.id] = { x: column(side, depth), y };
        previousBottom = y + h / 2;
      }
    }
  }
  return positions;
}
