"use client";

// Optional evidence graph (21 F08). The model (lib/evidenceGraphModel) keeps typed identities,
// draws one line per selected original record, and draws a comparison line only for a channel
// recorded on both sides. Source sits left, target right, each side's features in a column.
// Every feature line carries its own relation (predicate and qualifier); entity nodes show only
// the entity, so relationships that share a target keep their own meaning (S0-R1). Below the
// canvas, an inspector explains the current selection and a complete item list gives the same
// items to keyboard and screen-reader users. A selection from the evidence list resolves to the
// exact record, revealing a grouped record or saying why it is not drawn (S0-R3). De-emphasis
// fades lines and borders only; labels keep at least 4.5:1 contrast in both themes (S0-R4).
// Sizes follow the reader's text size. Geometry changes only on explicit actions; the view
// (zoom, pan, expansions, selection) belongs to the compared pair and survives re-renders,
// answer saves, tab switches and theme or text-size changes.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { IconZoomIn, IconZoomOut } from "@/components/common/Icons";
import { channelName, type EvidenceBundle } from "@/components/explore/evidenceData";
import { useFactInspector } from "@/components/workspace/FactInspector";
import { buildGraphModel, entityNodeId, layoutGraph, type GraphSide, type ModelEdge, type ModelNode, type ModelRecord } from "@/lib/evidenceGraphModel";
import { curie, KIND_NAMES, localName, predicateName } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import type { EntityRef } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import type { PairScope } from "@/lib/workspace/types";
import { useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

interface Expansion {
  nodes: ModelNode[];
  edges: ModelEdge[];
}
interface GraphView {
  zoom: number;
  pan: { x: number; y: number };
  /** The text scale the layout had and the canvas size, so a changed one keeps the same centre. */
  scale: number;
  size: { w: number; h: number };
  expansions: Expansion[];
  literals: GraphSide[];
  focusNode: string | null;
  focusEdge: string | null;
  focusEvidence: string | null;
}

const MAX_NODES = 150;
const READABLE_ZOOM = 0.7;
const NODE_WIDTH = 200;
const FOCAL_WIDTH = 210;
const CHARS_PER_LINE = 25;
const LABEL_WIDTH = 132;
// About how many characters of a relation label fit on one line of LABEL_WIDTH, and the longest
// property name drawn on a line (its qualifier is always added in full).
const LABEL_CHARS = 19;
const LABEL_PITCH = 24;
const LINE_LABEL_MAX = 48;
// The view of each compared pair, kept for this page's lifetime (bounded).
const views = new Map<string, GraphView>();
type Point = { x: number; y: number };
type Size = { w: number; h: number };
/** The graph point at the middle of a canvas of `size`. */
const centreOf = (pan: Point, zoom: number, size: Size): Point => ({ x: (size.w / 2 - pan.x) / zoom, y: (size.h / 2 - pan.y) / zoom });
function remember(key: string, view: GraphView) {
  views.delete(key);
  views.set(key, view);
  while (views.size > 40) views.delete(views.keys().next().value as string);
}

export type EdgeKind = "feature" | "asserted" | "structural" | "inferred" | "bridge";

export const EDGE_MEANING: Record<EdgeKind, string> = {
  feature: "A feature Exact selected for this pair, read from the original record.",
  asserted: "Added from the ontology on request: an asserted parent. Not one of Exact's selected features.",
  structural: "Added from the ontology on request: a parent reached by a documented structural rule.",
  inferred: "Added from the ontology on request: a relationship computed by the recorded reasoner.",
  bridge: "A comparison Exact recorded when scoring this pair. It is matcher evidence, not a statement in either ontology, and its score is not a probability of a match.",
};

/** "4 attributes · 2 hierarchy": counts per channel, in first-seen order. */
type Direction = "left" | "right" | "up" | "down";
const DIRECTIONS: Direction[] = ["left", "right", "up", "down"];
const NONE_OUTSIDE: Record<Direction, number> = { left: 0, right: 0, up: 0, down: 0 };
const DIRECTION_WORDS: Record<Direction, string> = { left: "to the left", right: "to the right", up: "above", down: "below" };
const DIRECTION_ARROWS: Record<Direction, string> = { left: "←", right: "→", up: "↑", down: "↓" };

/** Which edge of the canvas a node lies wholly beyond, if any (a partly visible node is in view). */
function outsideDirection(box: { x1: number; x2: number; y1: number; y2: number }, width: number, height: number): Direction | null {
  if (box.x2 < 0) return "left";
  if (box.x1 > width) return "right";
  if (box.y2 < 0) return "up";
  if (box.y1 > height) return "down";
  return null;
}

function countsByChannel(channels: string[]): string {
  const counts = new Map<string, number>();
  for (const channel of channels) counts.set(channel, (counts.get(channel) ?? 0) + 1);
  return Array.from(counts, ([channel, count]) => `${count} ${channelName(channel).toLowerCase()}`).join(", ");
}

const COMPARISON_BASIS: Record<string, string> = {
  lexical_score: "Compared by Exact's lexical scoring",
  matrix_support: "Recorded in Exact's feature-comparison matrix",
  attribute_anchor: "Recorded as an attribute anchor",
  difference_endpoint: "Recorded as a difference endpoint",
};

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

/** The reader's text-size setting (1–2); the graph's fonts and spacing follow it. */
function textScale(): number {
  const value = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--text-scale"));
  return Number.isFinite(value) && value > 0 ? value : 1;
}

function useThemeVersion() {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const bump = () => setVersion((value) => value + 1);
    media.addEventListener("change", bump);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "style"] });
    return () => {
      media.removeEventListener("change", bump);
      observer.disconnect();
    };
  }, []);
  return version;
}

/** Legend glyphs drawn with the same patterns as the rendered lines. */
export function EdgeGlyph({ kind }: { kind: EdgeKind }) {
  if (kind === "bridge")
    return (
      <svg width="36" height="10" aria-hidden="true">
        <line x1="3" y1="5" x2="33" y2="5" stroke="var(--bridge)" strokeWidth="4" strokeLinecap="round" strokeDasharray="0.1 8" />
      </svg>
    );
  const dash = kind === "feature" ? "9 4 2 4" : kind === "structural" ? "7 5" : kind === "inferred" ? "2 4" : undefined;
  return (
    <svg width="36" height="10" aria-hidden="true">
      <line x1="0" y1="5" x2="36" y2="5" stroke="currentColor" strokeWidth="2" strokeDasharray={dash} />
    </svg>
  );
}

const CYTOSCAPE_STYLESHEET = "__________cytoscape_stylesheet";

function wrapLines(text: string): number {
  return Math.max(1, Math.ceil(text.length / CHARS_PER_LINE));
}

function clip(text: string, max = 80): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

const RECORD_STATE: Record<ModelRecord["status"], string> = {
  drawn: "shown in the graph",
  grouped: "grouped in the graph",
  no_graph_form: "has no graph form; read it as the original record",
  unavailable: "its original record is not available in this view",
};

export function EvidenceGraph({
  source,
  target,
  bundle,
  pair,
  viewKey,
  selected,
  onSelect,
  onShowInList,
  canExpand = true,
}: {
  source: EntityRef;
  target: EntityRef;
  bundle: EvidenceBundle;
  /** The pair, for its recorded comparisons (exact-evidence-metadata/1); none are drawn without it. */
  pair?: PairScope | null;
  /** Identity of the compared pair in its product and scope; a new key starts a new view. */
  viewKey: string;
  /** Evidence chosen elsewhere (the evidence list); the graph shows exactly its records. */
  selected: string | null;
  onSelect: (evidenceId: string | null) => void;
  /** Reveal a feature in the evidence list. */
  onShowInList?: (evidenceId: string) => void;
  canExpand?: boolean;
}) {
  const workspace = useWorkspace();
  const report = useWorkspaceAction();
  const inspector = useFactInspector();
  const container = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cyRef = useRef<any>(null);
  const stored = views.get(viewKey);
  const [expansions, setExpansions] = useState<Expansion[]>(() => stored?.expansions ?? []);
  const [literals, setLiterals] = useState<GraphSide[]>(() => stored?.literals ?? []);
  const [focusNodeId, setFocusNodeId] = useState<string | null>(() => stored?.focusNode ?? null);
  const [focusEdgeId, setFocusEdgeId] = useState<string | null>(() => stored?.focusEdge ?? null);
  // A feature chosen in the list that has no drawn line (no graph form, or unavailable).
  const [focusEvidence, setFocusEvidence] = useState<string | null>(() => stored?.focusEvidence ?? null);
  const [revealEdge, setRevealEdge] = useState<string | null>(null);
  const [showUndrawn, setShowUndrawn] = useState<Partial<Record<GraphSide, boolean>>>({});
  const [showComparisons, setShowComparisons] = useState(false);
  const [expandError, setExpandError] = useState<string | null>(null);
  // Bumped once the canvas exists, so restored selections and reveals are applied to it.
  const [ready, setReady] = useState(0);
  // Items wholly outside the visible canvas, by direction: narrow study panels and enlarged
  // text leave context off screen at a readable zoom, so the canvas says where it is.
  const [outside, setOutside] = useState<Record<Direction, number>>(NONE_OUTSIDE);
  const measureFrame = useRef(0);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const measureOutside = useCallback((cy: any) => {
    cancelAnimationFrame(measureFrame.current);
    measureFrame.current = requestAnimationFrame(() => {
      if (cy.destroyed?.()) return;
      const counts = { ...NONE_OUTSIDE };
      cy.nodes().forEach((node: { renderedBoundingBox: () => { x1: number; x2: number; y1: number; y2: number } }) => {
        const direction = outsideDirection(node.renderedBoundingBox(), cy.width(), cy.height());
        if (direction) counts[direction] += 1;
      });
      setOutside((current) => (DIRECTIONS.every((direction) => current[direction] === counts[direction]) ? current : counts));
    });
  }, []);
  useEffect(() => () => cancelAnimationFrame(measureFrame.current), []);
  const themeVersion = useThemeVersion();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const scale = useMemo(() => textScale(), [themeVersion]);
  const scaleRef = useRef(scale);
  scaleRef.current = scale;
  // The text scale of the positions currently drawn.
  const drawnScale = useRef(scale);
  const expansionAllowed = canExpand && workspace.capabilities.graphExpansion;

  // A different pair is a different view: never carry nodes or selections across.
  const lastKey = useRef(viewKey);
  useEffect(() => {
    if (lastKey.current === viewKey) return;
    lastKey.current = viewKey;
    const next = views.get(viewKey);
    setExpansions(next?.expansions ?? []);
    setLiterals(next?.literals ?? []);
    setFocusNodeId(next?.focusNode ?? null);
    setFocusEdgeId(next?.focusEdge ?? null);
    setFocusEvidence(next?.focusEvidence ?? null);
    setExpandError(null);
  }, [viewKey]);

  // Recorded comparisons only (23 C04): no metadata, no comparison lines.
  const metadata = useAsync(pair && workspace.evidenceMetadata ? `${workspace.key}|evidence-metadata|${pair.runId ?? ""}|${pair.pairId ?? ""}|${pair.candidateId ?? ""}` : null, (signal) => workspace.evidenceMetadata!(pair!, signal));
  const base = useMemo(
    () => buildGraphModel(source, target, bundle.items, bundle.axioms as never, { expandLiterals: new Set(literals), metadata: metadata.data ?? null }),
    [source, target, bundle, literals, metadata.data],
  );
  const baseRef = useRef(base);
  baseRef.current = base;
  const model = useMemo(() => {
    const nodes = new Map(base.nodes.map((node) => [node.id, node]));
    const edges = [...base.edges];
    for (const step of expansions) {
      step.nodes.forEach((node) => {
        if (!nodes.has(node.id)) nodes.set(node.id, node);
      });
      step.edges.forEach((edge) => {
        if (!edges.some((existing) => existing.id === edge.id) && nodes.has(edge.source)) edges.push(edge);
      });
    }
    return { nodes: Array.from(nodes.values()), edges };
  }, [base, expansions]);
  const modelRef = useRef(model);
  modelRef.current = model;
  const focusNode = model.nodes.find((node) => node.id === focusNodeId) ?? null;
  const focusEdge = model.edges.find((edge) => edge.id === focusEdgeId) ?? null;
  const focusEdgeRef = useRef(focusEdgeId);
  focusEdgeRef.current = focusEdgeId;

  const save = useCallback(
    (changes: Partial<GraphView>) => {
      const current = views.get(viewKey) ?? { zoom: 0, pan: { x: 0, y: 0 }, scale: 1, size: { w: 0, h: 0 }, expansions: [], literals: [], focusNode: null, focusEdge: null, focusEvidence: null };
      remember(viewKey, { ...current, ...changes });
    },
    [viewKey],
  );
  useEffect(() => save({ expansions, literals, focusNode: focusNodeId, focusEdge: focusEdgeId, focusEvidence }), [expansions, literals, focusNodeId, focusEdgeId, focusEvidence, save]);

  // Labels, per ontology, for entities and the properties named in restrictions.
  const propertyIris = model.edges.map((edge) => edge.relation.property).filter((iri): iri is string => Boolean(iri));
  const sourceLabel = useLabelLookup(source.ontology_version_id, [...model.nodes.filter((node) => node.iri && node.ontology === source.ontology_version_id).map((node) => node.iri!), ...propertyIris]);
  const targetLabel = useLabelLookup(target.ontology_version_id, [...model.nodes.filter((node) => node.iri && node.ontology === target.ontology_version_id).map((node) => node.iri!), ...propertyIris]);
  const lookup = useCallback((ontology: string) => (ontology === source.ontology_version_id ? sourceLabel : targetLabel), [source, sourceLabel, targetLabel]);
  const entityName = useCallback((node: ModelNode) => (node.iri ? lookup(node.ontology)(node.iri)?.value ?? curie(node.iri) : ""), [lookup]);

  /**
   * A relation's text. `drawn` is the short form on a line: an unlabelled property outside a known
   * prefix by its local name, clipped; the list, inspector and original record keep the full IRI.
   */
  const relationText = useCallback(
    (edge: ModelEdge, drawn = false) => {
      if (edge.kind === "bridge") return edge.bridge?.level === "feature_correspondence" ? `${channelName(edge.channel ?? "")} correspondence` : `${channelName(edge.channel ?? "")} compared`;
      if (edge.relation.text) return edge.relation.text;
      const ontology = edge.side === "target" ? target.ontology_version_id : source.ontology_version_id;
      const property = edge.relation.property;
      if (property && edge.relation.quantifier) {
        const name = lookup(ontology)(property)?.value ?? (drawn && curie(property) === property ? localName(property) : curie(property));
        // Only the name is clipped: the qualifier is the meaning and is always drawn (S0-R1).
        return `${drawn ? clip(name, LINE_LABEL_MAX) : name} ${edge.relation.quantifier}`;
      }
      if (property) {
        const name = predicateName(property);
        return drawn && name === property ? localName(property) : name;
      }
      return "";
    },
    [lookup, source, target],
  );
  const nodeById = useCallback((id: string) => model.nodes.find((node) => node.id === id), [model.nodes]);
  /** A feature line into an entity carries its relation as a visible label; a literal record names its predicate itself. */
  const labelledLine = useCallback((edge: ModelEdge) => edge.kind === "feature" && nodeById(edge.target)?.role === "context", [nodeById]);
  const literalPredicate = useMemo(() => {
    const map = new Map<string, string>();
    for (const edge of model.edges) if (edge.kind === "feature" && edge.target.startsWith("lit:")) map.set(edge.target, relationText(edge, true));
    return map;
  }, [model.edges, relationText]);

  /** Drawn node text: the compared entity with its side and kind, an entity's name, or a literal record. */
  const nodeLabel = useCallback(
    (node: ModelNode) => {
      if (node.role === "focal") return `${node.side === "source" ? "Source" : "Target"} ${KIND_NAMES[node.kind]}\n${clip(entityName(node), 70)}`;
      if (node.role === "group") return `+${node.count} more records\nselect to show all`;
      if (node.role === "literal") {
        const predicate = literalPredicate.get(node.id);
        const quote = `“${clip(node.text ?? "", 66)}”`;
        return predicate ? `${predicate}\n${quote}` : quote;
      }
      return clip(entityName(node), 66);
    },
    [entityName, literalPredicate],
  );
  const sizeOf = useCallback(
    (node: ModelNode) => {
      const lines = nodeLabel(node).split("\n").reduce((sum, line) => sum + wrapLines(line), 0);
      return { width: (node.role === "focal" ? FOCAL_WIDTH : NODE_WIDTH) * scale, height: (20 + lines * 19) * scale };
    },
    [nodeLabel, scale],
  );
  // What each line draws. Parallel lines are spread for their wrapped labels, and a node keeps
  // vertical room for the labels at the middle of its incoming lines, so labels never collide.
  const drawn = useMemo(() => {
    const labels = new Map(model.edges.map((edge) => [edge.id, edge.target.startsWith("lit:") || edge.kind === "group" ? "" : relationText(edge, true)]));
    const pair = (edge: ModelEdge) => [edge.source, edge.target].sort().join("|");
    const bundles = new Map<string, { target: string; count: number; lines: number }>();
    for (const edge of model.edges) {
      if (!labelledLine(edge)) continue;
      const bundle = bundles.get(pair(edge)) ?? { target: edge.target, count: 0, lines: 1 };
      bundle.count += 1;
      bundle.lines = Math.max(bundle.lines, Math.ceil(labels.get(edge.id)!.length / LABEL_CHARS));
      bundles.set(pair(edge), bundle);
    }
    // Parallel labels are stacked LABEL_PITCH apart per text line; a label sits halfway along its
    // line, so a node needs twice its bundle's extent.
    const room = new Map<string, number>();
    for (const bundle of bundles.values()) room.set(bundle.target, (room.get(bundle.target) ?? 0) + 2 * ((bundle.count - 1) * LABEL_PITCH * bundle.lines + 16 * bundle.lines + 4) * scale);
    return { labels, room, bundle: (edge: ModelEdge) => bundles.get(pair(edge)) };
  }, [model.edges, relationText, labelledLine, scale]);
  const positions = useMemo(
    () =>
      layoutGraph(
        model.nodes,
        (node) => {
          const size = sizeOf(node);
          return { width: size.width, height: Math.max(size.height, drawn.room.get(node.id) ?? 0) };
        },
        18,
        scale,
      ),
    [model.nodes, sizeOf, scale, drawn],
  );
  const positionsRef = useRef(positions);
  positionsRef.current = positions;
  /**
   * The distance between parallel lines. Their labels are offset across the line, so a steep line
   * needs more distance to stack its labels LABEL_PITCH apart vertically.
   */
  const stepOf = useCallback(
    (edge: ModelEdge) => {
      const bundle = drawn.bundle(edge);
      const [from, to] = [positions[edge.source], positions[edge.target]];
      if (!bundle || !from || !to) return 44 * scale;
      const cos = Math.abs(to.x - from.x) / Math.max(1, Math.hypot(to.x - from.x, to.y - from.y));
      return (2 * LABEL_PITCH * bundle.lines * scale) / Math.max(0.35, cos);
    },
    [drawn, positions, scale],
  );

  const styles = useCallback(() => {
    const ink = cssVar("--ink");
    const ink2 = cssVar("--ink-2");
    const ink3 = cssVar("--ink-3");
    const surface = cssVar("--surface");
    const surface2 = cssVar("--surface-2");
    const lineStrong = cssVar("--line-strong");
    const focus = cssVar("--focus");
    const font = "IBM Plex Sans, Segoe UI, sans-serif";
    return [
      {
        selector: "node",
        style: {
          label: "data(label)",
          shape: "round-rectangle",
          "background-color": surface,
          "border-width": 1.5,
          "border-color": "data(line)",
          color: ink,
          "font-family": font,
          "font-size": 14 * scale,
          "line-height": 1.3,
          "text-wrap": "wrap",
          "text-max-width": "data(textWidth)",
          "text-valign": "center",
          "text-halign": "center",
          width: "data(width)",
          height: "data(height)",
        },
      },
      { selector: "node[side = 'target']", style: { shape: "cut-rectangle" } },
      { selector: "node[role = 'focal']", style: { "background-color": surface2, "border-width": 3, "font-weight": 600, "font-size": 15 * scale } },
      { selector: "node[role = 'literal']", style: { "font-style": "italic", "border-style": "dashed" } },
      { selector: "node[role = 'group']", style: { "border-style": "dotted", "border-width": 2, color: ink3 } },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": ink3,
          "target-arrow-color": ink3,
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.9,
          "curve-style": "bezier",
          "control-point-step-size": "data(step)",
          "font-size": 12.5 * scale,
          "font-family": font,
          color: ink2,
          "text-wrap": "wrap",
          "text-max-width": LABEL_WIDTH * scale,
          "text-background-color": surface,
          "text-background-opacity": 1,
          "text-background-padding": 2 * scale,
          "text-background-shape": "round-rectangle",
        },
      },
      // Each feature line into an entity says its own relation, always (S0-R1).
      { selector: "edge[?labelled]", style: { label: "data(label)" } },
      { selector: "edge[kind = 'feature']", style: { "line-style": "dashed", "line-dash-pattern": [9, 4, 2, 4] } },
      { selector: "edge[kind = 'group']", style: { "line-style": "dotted", "target-arrow-shape": "none" } },
      { selector: "edge[kind = 'structural']", style: { "line-style": "dashed", "line-dash-pattern": [7, 5] } },
      { selector: "edge[kind = 'inferred']", style: { "line-style": "dotted" } },
      {
        // A channel-level comparison: a line of round dots, no arrow.
        selector: "edge[kind = 'bridge']",
        style: { "line-style": "dashed", "line-cap": "round", "line-dash-pattern": [0.1, 8], width: 4, "line-color": cssVar("--bridge"), "target-arrow-shape": "none", "control-point-step-size": 26 * scale },
      },
      {
        // A recorded feature correspondence joins two features, not the compared entities.
        selector: "edge[id ^= 'corr:']",
        style: { "line-dash-pattern": [6, 4], width: 2, "curve-style": "unbundled-bezier", "control-point-distances": [30 * scale], "control-point-weights": [0.5] },
      },
      // De-emphasis fades geometry only; every label keeps at least 4.5:1 (S0-R4).
      { selector: "node.faded", style: { "border-opacity": 0.4, color: ink2 } },
      { selector: "node[role = 'focal'].faded", style: { color: ink, "border-opacity": 0.55 } },
      { selector: "edge.faded", style: { "line-opacity": 0.3, "target-arrow-color": lineStrong, color: ink2 } },
      { selector: "node:selected", style: { "border-width": 4, "border-color": focus, "border-opacity": 1 } },
      { selector: "edge:selected", style: { width: 4, "line-color": focus, "target-arrow-color": focus, "line-opacity": 1, label: "data(label)", color: ink, "font-weight": 600 } },
    ];
  }, [scale]);

  const elements = useCallback(() => {
    const els: unknown[] = [];
    for (const node of model.nodes) {
      const size = sizeOf(node);
      els.push({
        group: "nodes",
        data: { id: node.id, label: nodeLabel(node), side: node.side, role: node.role, line: node.side === "source" ? cssVar("--source") : cssVar("--target"), width: size.width, height: size.height, textWidth: size.width - 22 * scale },
        position: positions[node.id],
      });
    }
    for (const edge of model.edges) {
      els.push({ group: "edges", data: { id: edge.id, source: edge.source, target: edge.target, label: drawn.labels.get(edge.id), labelled: labelledLine(edge), kind: edge.kind, step: stepOf(edge) } });
    }
    return els;
  }, [model, nodeLabel, positions, sizeOf, labelledLine, scale, drawn, stepOf]);
  const elementsRef = useRef(elements);
  elementsRef.current = elements;

  /**
   * The starting view: fit the width (both sides and their feature columns), never shrink
   * labels below a readable zoom, and centre vertically on the compared pair. Tall columns are
   * panned; "Fit all" is an explicit request.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const initialView = useCallback((cy: any) => {
    const box = cy.elements().boundingBox();
    const level = Math.max(READABLE_ZOOM, Math.min(1.1, (cy.width() - 48) / Math.max(1, box.w)));
    cy.zoom(level);
    const focal = cy.nodes("[role = 'focal']").boundingBox();
    // Centre the whole width when it fits (one side may have deeper columns), else the pair.
    const across = box.w * level <= cy.width() - 48 ? box : focal;
    cy.pan({ x: cy.width() / 2 - ((across.x1 + across.x2) / 2) * level, y: cy.height() / 2 - ((focal.y1 + focal.y2) / 2) * level });
  }, []);

  /**
   * Pan (never zoom) so a selected line and both its ends are on the canvas, with both compared
   * concepts too when they fit; otherwise as close to them as the selection allows.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const keepInView = useCallback((cy: any, edgeId: string | null) => {
    const edge = edgeId ? cy.getElementById(edgeId) : null;
    if (!edge || edge.empty()) return;
    const selection = edge.union(edge.connectedNodes());
    const own = selection.renderedBoundingBox();
    const wanted = selection.union(cy.nodes("[role = 'focal']")).renderedBoundingBox();
    const margin = 24;
    const shift = (lo: number, hi: number, wantLo: number, wantHi: number, size: number) => {
      const ideal = wantHi - wantLo <= size - 2 * margin ? (wantLo < margin ? margin - wantLo : wantHi > size - margin ? size - margin - wantHi : 0) : size / 2 - (wantLo + wantHi) / 2;
      const [min, max] = [margin - lo, size - margin - hi];
      return min > max ? min : Math.min(max, Math.max(min, ideal));
    };
    const dx = shift(own.x1, own.x2, wanted.x1, wanted.x2, cy.width());
    const dy = shift(own.y1, own.y2, wanted.y1, wanted.y2, cy.height());
    if (dx || dy) cy.panBy({ x: dx, y: dy });
  }, []);

  /** Select a line or node. Choosing it from a list (`reveal`) pans it into view if needed; a tap never moves the view. */
  const choose = useCallback(
    (edge: ModelEdge | null, node: ModelNode | null, reveal = false) => {
      setFocusEdgeId(edge?.id ?? null);
      setFocusNodeId(node?.id ?? null);
      setFocusEvidence(null);
      const group = edge?.kind === "group" ? edge.side : node?.role === "group" ? node.side : null;
      if (group) {
        setLiterals((current) => Array.from(new Set([...current, group])));
        return;
      }
      if (!edge) return;
      if (reveal) setRevealEdge(edge.id);
      onSelect(edge.evidenceId ?? null);
      // Only a feature line is evidence; bridges and context lines are inspection only (21 F01).
      report({ type: "graph_edge", edgeKind: edge.kind as Exclude<ModelEdge["kind"], "group">, evidenceId: edge.evidenceId ?? null, factId: edge.factId ?? null });
    },
    [onSelect, report],
  );
  const chooseRef = useRef(choose);
  chooseRef.current = choose;

  // A feature chosen elsewhere resolves to exactly its records (S0-R3): reveal a grouped record,
  // focus its line, or clear any unrelated previous focus and say why it is not drawn.
  useEffect(() => {
    if (!selected) return;
    const current = modelRef.current.edges.find((edge) => edge.id === focusEdgeId);
    if (current?.evidenceId === selected || focusEvidence === selected) return;
    const records = baseRef.current.records.filter((record) => record.evidenceId === selected);
    const drawable = records.find((record) => record.status === "drawn" || record.status === "grouped");
    setFocusNodeId(null);
    if (drawable?.edgeId) {
      if (drawable.status === "grouped") setLiterals((sides) => Array.from(new Set([...sides, drawable.side])));
      setFocusEdgeId(drawable.edgeId);
      setFocusEvidence(null);
      setRevealEdge(drawable.edgeId);
    } else {
      setFocusEdgeId(null);
      setFocusEvidence(selected);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  // Create once per compared pair; everything else updates the existing graph in place.
  useEffect(() => {
    let destroyed = false;
    let observer: ResizeObserver | null = null;
    (async () => {
      const cytoscape = (await import("cytoscape")).default;
      if (destroyed || !container.current) return;
      // Cytoscape injects an inline <style> unless its stylesheet id already exists; the study
      // CSP forbids inline styles, so its one rule lives in app.css and the id is reserved here.
      if (!document.getElementById(CYTOSCAPE_STYLESHEET)) {
        const marker = document.createElement("meta");
        marker.id = CYTOSCAPE_STYLESHEET;
        marker.name = "cytoscape-stylesheet";
        marker.content = "app.css";
        document.head.appendChild(marker);
      }
      const cy = cytoscape({
        container: container.current,
        elements: elementsRef.current() as never,
        style: styles() as never,
        layout: { name: "preset" },
        userZoomingEnabled: false,
        userPanningEnabled: true,
        boxSelectionEnabled: false,
        autoungrabify: true,
        minZoom: 0.3,
        maxZoom: 2.5,
      });
      const previous = views.get(viewKey);
      let size: Size = { w: cy.width(), h: cy.height() };
      drawnScale.current = scaleRef.current;
      if (previous && previous.zoom > 0) {
        cy.zoom(previous.zoom);
        const ratio = scaleRef.current / previous.scale;
        if (ratio === 1 && previous.size.w === size.w && previous.size.h === size.h) cy.pan(previous.pan);
        else {
          // The text size or canvas changed while away: same zoom and centre, selection in view.
          const centre = centreOf(previous.pan, previous.zoom, previous.size);
          cy.pan({ x: size.w / 2 - centre.x * ratio * previous.zoom, y: size.h / 2 - centre.y * ratio * previous.zoom });
          keepInView(cy, focusEdgeRef.current);
        }
      } else initialView(cy);
      const view = () => {
        save({ zoom: cy.zoom(), pan: { ...cy.pan() }, scale: drawnScale.current, size: { w: cy.width(), h: cy.height() } });
        measureOutside(cy);
      };
      cy.on("viewport", view);
      view();
      cy.on("tap", "node", (event: { target: { id: () => string } }) => chooseRef.current(null, modelRef.current.nodes.find((node) => node.id === event.target.id()) ?? null));
      cy.on("tap", "edge", (event: { target: { id: () => string } }) => chooseRef.current(modelRef.current.edges.find((edge) => edge.id === event.target.id()) ?? null, null));
      cyRef.current = cy;
      setReady((value) => value + 1);
      // A resized container (layout switch, text size) keeps its centre and the selection in view.
      observer = new ResizeObserver(() => {
        cy.resize();
        const next: Size = { w: cy.width(), h: cy.height() };
        if (next.w === size.w && next.h === size.h) return;
        cy.panBy({ x: (next.w - size.w) / 2, y: (next.h - size.h) / 2 });
        size = next;
        keepInView(cy, focusEdgeRef.current);
        view();
      });
      observer.observe(container.current);
    })();
    return () => {
      destroyed = true;
      observer?.disconnect();
      cyRef.current?.destroy();
      cyRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewKey]);

  // Sync elements (labels, expansions, text size) and theme. A text-size change spreads the
  // layout, so the same zoom and centre are kept and the selection stays in view (21 F01).
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    const ratio = scale / drawnScale.current;
    const centre = centreOf(cy.pan(), cy.zoom(), { w: cy.width(), h: cy.height() });
    cy.batch(() => {
      const wanted = elements() as { group: string; data: { id: string } & Record<string, unknown>; position?: { x: number; y: number } }[];
      const ids = new Set(wanted.map((element) => element.data.id));
      cy.elements().forEach((element: { id: () => string; remove: () => void }) => {
        if (!ids.has(element.id())) element.remove();
      });
      for (const element of wanted) {
        const existing = cy.getElementById(element.data.id);
        if (existing.nonempty()) {
          existing.data(element.data);
          if (element.position && existing.isNode()) existing.position(element.position);
        } else cy.add(element);
      }
      cy.style(styles());
    });
    if (ratio !== 1) {
      drawnScale.current = scale;
      cy.pan({ x: cy.width() / 2 - centre.x * ratio * cy.zoom(), y: cy.height() / 2 - centre.y * ratio * cy.zoom() });
      keepInView(cy, focusEdgeRef.current);
    }
    measureOutside(cy);
  }, [elements, styles, themeVersion, scale, keepInView, measureOutside]);

  // Selection highlights its line or node and de-emphasises the rest; the view never moves.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().unselect().removeClass("faded");
    const edge = focusEdgeId ? cy.getElementById(focusEdgeId) : null;
    const node = focusNodeId ? cy.getElementById(focusNodeId) : null;
    const chosen = edge?.nonempty() ? edge : node?.nonempty() ? node : null;
    if (!chosen) return;
    chosen.select();
    const keep = chosen.union(chosen.connectedNodes()).union(chosen.connectedEdges());
    cy.elements().difference(keep).addClass("faded");
  }, [focusEdgeId, focusNodeId, elements, ready]);

  // An explicit reveal (from the evidence list) brings its line into view, keeping the zoom.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !revealEdge || cy.getElementById(revealEdge).empty()) return;
    keepInView(cy, revealEdge);
    setRevealEdge(null);
  }, [revealEdge, elements, ready, keepInView]);

  const zoom = (factor: number) => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.zoom({ level: Math.max(0.3, Math.min(2.5, cy.zoom() * factor)), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
    report({ type: "graph_zoom" });
  };
  const fit = () => {
    cyRef.current?.fit(undefined, 32);
    report({ type: "graph_fit" });
  };
  /** Bring the items beyond one edge of the canvas into view at the current zoom. */
  const reveal = (direction: Direction) => {
    const cy = cyRef.current;
    if (!cy) return;
    const hidden = cy.nodes().filter((node: { renderedBoundingBox: () => { x1: number; x2: number; y1: number; y2: number } }) => outsideDirection(node.renderedBoundingBox(), cy.width(), cy.height()) === direction);
    if (hidden.empty()) return;
    const box = hidden.renderedBoundingBox();
    const across = direction === "left" || direction === "right";
    cy.panBy({
      x: across ? cy.width() / 2 - (box.x1 + box.x2) / 2 : 0,
      y: !across || box.y1 < 0 || box.y2 > cy.height() ? cy.height() / 2 - (box.y1 + box.y2) / 2 : 0,
    });
    report({ type: "graph_zoom" });
  };
  const hiddenCount = DIRECTIONS.reduce((sum, direction) => sum + outside[direction], 0);
  const reset = () => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().forEach((node: { id: () => string; position: (p: { x: number; y: number }) => void }) => {
      const position = positionsRef.current[node.id()];
      if (position) node.position(position);
    });
    initialView(cy);
    setFocusEdgeId(null);
    setFocusNodeId(null);
    setFocusEvidence(null);
    onSelect(null);
    report({ type: "graph_reset" });
  };

  const expandParents = async (node: ModelNode) => {
    setExpandError(null);
    if (!node.iri) return;
    if (model.nodes.length >= MAX_NODES) {
      setExpandError(`The graph already shows ${MAX_NODES} nodes. Use Hierarchy for further navigation.`);
      return;
    }
    try {
      const page = await workspace.hierarchy({ ontology_version_id: node.ontology, iri: node.iri, kind: node.kind }, "parents", "literal_asserted", null);
      const nodes: ModelNode[] = [];
      const edges: ModelEdge[] = [];
      page.items.forEach((edge, index) => {
        const id = entityNodeId(node.side, { ontology_version_id: node.ontology, kind: edge.parent.kind, iri: edge.parent.iri });
        if (!model.nodes.some((item) => item.id === id)) nodes.push({ id, side: node.side, role: "context", ontology: node.ontology, kind: edge.parent.kind, iri: edge.parent.iri, depth: node.depth + 1, order: index, anchor: node.id });
        const kind = edge.basis === "structural_navigation" ? "structural" : edge.basis === "reasoner_inferred" ? "inferred" : "asserted";
        edges.push({ id: `h:${node.side}:${node.id}:${edge.id}`, source: node.id, target: id, kind, side: node.side, relation: { text: node.kind === "class" ? "subclass of" : node.kind === "individual" ? "instance of" : "subproperty of" } });
      });
      if (!edges.length) {
        setExpandError(page.status === "absent_in_scope" ? "No asserted parent is shown for this node." : page.reason ?? "Parents of this node are not available here.");
        return;
      }
      setExpansions((list) => [...list, { nodes, edges }]);
      // The graph is bounded; never imply that a first page is all of a node's parents.
      if (page.next_cursor) setExpandError(`Added ${edges.length}${page.total_count != null ? ` of ${page.total_count}` : ""} parents; Hierarchy lists them all.`);
    } catch {
      setExpandError("Parents could not be loaded.");
    }
  };

  const features = model.edges.filter((edge) => edge.kind === "feature");
  const added = model.edges.filter((edge) => ["asserted", "structural", "inferred"].includes(edge.kind));
  const itemText = (edge: ModelEdge) => {
    const to = nodeById(edge.target);
    const name = to?.role === "literal" ? `“${clip(to.text ?? "", 60)}”` : to ? entityName(to) : "";
    if (edge.kind === "bridge") return relationText(edge);
    if (edge.kind === "feature") return `${relationText(edge)} ${name}`.trim();
    const from = nodeById(edge.source);
    return `${from ? entityName(from) : ""} ${relationText(edge)} ${name}`.trim();
  };
  /** The two drawn features a recorded correspondence pairs, briefly. */
  const correspondenceEnds = (bridge: NonNullable<ModelEdge["bridge"]>) => {
    const end = (ids: string[], side: GraphSide) => {
      const edge = features.find((item) => item.side === side && item.evidenceId && ids.includes(item.evidenceId));
      const node = edge ? nodeById(edge.target) : null;
      return node ? (node.role === "literal" ? `“${clip(node.text ?? "", 32)}”` : clip(entityName(node), 32)) : "a feature not drawn";
    };
    return `${end(bridge.source_evidence_ids, "source")} ↔ ${end(bridge.target_evidence_ids, "target")}`;
  };
  const legendKinds = (["feature", "bridge", "asserted", "structural", "inferred"] as EdgeKind[]).filter((kind) => kind === "feature" || model.edges.some((edge) => edge.kind === kind));
  const itemOf = (evidenceId: string) => bundle.items.find((item) => item.evidence_id === evidenceId) ?? null;
  const openRecord = (evidenceId: string, factId: string) => {
    const item = itemOf(evidenceId);
    const subject = item?.entity ?? (item?.side === "target" ? target : source);
    inspector?.open({ factId, ontologies: [subject.ontology_version_id], subject }, `${channelName(item?.channel ?? "")} record`, "record");
  };

  /** Every record of one feature, with where it is in the graph; drawn ones can be selected. */
  const recordList = (evidenceId: string, current: string | null) => {
    const records = base.records.filter((record) => record.evidenceId === evidenceId);
    if (records.length < 2 && records.every((record) => record.edgeId === current)) return null;
    return (
      <ul className="graph-records" aria-label="Records of this feature">
        {records.map((record, index) => {
          const edge = record.edgeId ? model.edges.find((item) => item.id === record.edgeId) : null;
          return (
            <li key={`${record.factId ?? "none"}-${index}`}>
              <span>
                {edge && record.edgeId !== current ? (
                  <button type="button" className="link-btn" onClick={() => choose(edge, null, true)}>
                    {itemText(edge)}
                  </button>
                ) : (
                  (edge ? itemText(edge) : `Record ${index + 1}`)
                )}
                <span className="meta"> · {record.edgeId === current ? "selected" : RECORD_STATE[record.status]}</span>
              </span>
              {record.factId && !edge && inspector && (
                <button type="button" className="btn btn-sm" onClick={() => openRecord(evidenceId, record.factId!)}>
                  Original record
                </button>
              )}
            </li>
          );
        })}
      </ul>
    );
  };

  const inspectorBody = () => {
    if (focusEdge) {
      const meaning = EDGE_MEANING[(focusEdge.kind === "group" ? "feature" : focusEdge.kind) as EdgeKind];
      if (focusEdge.kind === "bridge") {
        const recorded = focusEdge.bridge;
        const paired = (ids: string[], side: GraphSide) => features.filter((edge) => edge.side === side && edge.evidenceId && ids.includes(edge.evidenceId)).map((edge) => itemText(edge));
        return (
          <>
            <h4 className="graph-inspector-title">{relationText(focusEdge)}</h4>
            <p>{meaning}</p>
            {recorded?.level === "feature_correspondence" && (
              <p className="meta">
                Pairs source “{paired(recorded.source_evidence_ids, "source").join("; ") || "a feature not drawn"}” with target “{paired(recorded.target_evidence_ids, "target").join("; ") || "a feature not drawn"}”.
              </p>
            )}
            {recorded && (
              <p className="meta">
                {COMPARISON_BASIS[recorded.comparison_basis ?? ""] ?? "Recorded comparison"}
                {recorded.recorded_score != null ? ` · saved matcher value ${recorded.recorded_score.toFixed(2)}` : ""}
              </p>
            )}
          </>
        );
      }
      if (focusEdge.kind === "feature") {
        return (
          <>
            <h4 className="graph-inspector-title">
              {channelName(focusEdge.channel ?? "")} feature · {focusEdge.side === "source" ? "source" : "target"}
            </h4>
            <p className="fact-reading" data-fact-id={focusEdge.factId ?? undefined}>
              {itemText(focusEdge)}
            </p>
            <p className="meta">{meaning}</p>
            {focusEdge.evidenceId && recordList(focusEdge.evidenceId, focusEdge.id)}
            <div className="graph-inspector-actions">
              {inspector && focusEdge.factId && focusEdge.evidenceId && (
                <button type="button" className="btn btn-sm" onClick={() => openRecord(focusEdge.evidenceId!, focusEdge.factId!)}>
                  Original record
                </button>
              )}
              {onShowInList && focusEdge.evidenceId && (
                <button type="button" className="btn btn-sm btn-quiet" onClick={() => onShowInList(focusEdge.evidenceId!)}>
                  Show in evidence list
                </button>
              )}
              {expansionAllowed && nodeById(focusEdge.target)?.role === "context" && (
                <button type="button" className="btn btn-sm btn-quiet" onClick={() => expandParents(nodeById(focusEdge.target)!)}>
                  Add parents of {entityName(nodeById(focusEdge.target)!)}
                </button>
              )}
            </div>
          </>
        );
      }
      return (
        <>
          <h4 className="graph-inspector-title">{itemText(focusEdge)}</h4>
          <p className="meta">{meaning}</p>
        </>
      );
    }
    if (focusEvidence) {
      const item = itemOf(focusEvidence);
      return (
        <>
          <h4 className="graph-inspector-title">
            {channelName(item?.channel ?? "")} feature · {item?.side === "target" ? "target" : "source"}
          </h4>
          <p>This feature is not drawn in the graph.</p>
          {recordList(focusEvidence, null) ?? (
            <ul className="graph-records" aria-label="Records of this feature">
              {base.records
                .filter((record) => record.evidenceId === focusEvidence)
                .map((record, index) => (
                  <li key={index}>
                    <span>Record {index + 1}</span>
                    <span className="meta"> · {RECORD_STATE[record.status]}</span>
                    {record.factId && inspector && (
                      <button type="button" className="btn btn-sm" onClick={() => openRecord(focusEvidence, record.factId!)}>
                        Original record
                      </button>
                    )}
                  </li>
                ))}
            </ul>
          )}
          {onShowInList && (
            <div className="graph-inspector-actions">
              <button type="button" className="btn btn-sm btn-quiet" onClick={() => onShowInList(focusEvidence)}>
                Show in evidence list
              </button>
            </div>
          )}
        </>
      );
    }
    if (focusNode) {
      return (
        <>
          <h4 className="graph-inspector-title">{focusNode.role === "literal" ? `“${focusNode.text}”` : focusNode.role === "group" ? "More records" : entityName(focusNode)}</h4>
          <p className="meta">
            {focusNode.side === "source" ? "Source" : "Target"} · {focusNode.role === "focal" ? `compared ${KIND_NAMES[focusNode.kind]}` : focusNode.role === "literal" ? "recorded text" : KIND_NAMES[focusNode.kind]}
          </p>
          {focusNode.iri && <p className="iri">{focusNode.iri}</p>}
          {expansionAllowed && focusNode.iri && (focusNode.role === "context" || focusNode.role === "focal") && (
            <button type="button" className="btn btn-sm" onClick={() => expandParents(focusNode)}>
              Add its parents
            </button>
          )}
        </>
      );
    }
    return <p className="meta">Select a line, a node or an item below to see what it means.</p>;
  };

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar" role="toolbar" aria-label="Graph view controls">
        <button type="button" className="icon-btn" aria-label="Zoom in" data-tour="graph-zoom" onClick={() => zoom(1.25)}>
          <IconZoomIn />
        </button>
        <button type="button" className="icon-btn" aria-label="Zoom out" onClick={() => zoom(0.8)}>
          <IconZoomOut />
        </button>
        <button type="button" className="btn btn-sm" onClick={fit}>
          Fit all
        </button>
        <button type="button" className="btn btn-sm" onClick={reset}>
          Reset view
        </button>
        <span className="graph-toolbar-spacer" />
        {expansionAllowed && (
          <button type="button" className="btn btn-sm" disabled={!expansions.length} onClick={() => setExpansions((list) => list.slice(0, -1))}>
            Undo last expansion
          </button>
        )}
      </div>
      <div className="graph-body">
        <div className="graph-stage">
          {/* Context beyond the canvas edge is named and one press away; it never covers the drawing. */}
          {hiddenCount > 0 && (
            <div className="graph-outside" role="status">
              <span className="meta">
                {hiddenCount} of {model.nodes.length} items are outside this view; the item list has every item.
              </span>
              {DIRECTIONS.filter((direction) => outside[direction] > 0).map((direction) => (
                <button key={direction} type="button" className="btn btn-sm graph-cue" aria-label={`Show ${outside[direction]} ${outside[direction] === 1 ? "item" : "items"} ${DIRECTION_WORDS[direction]}`} onClick={() => reveal(direction)}>
                  <span aria-hidden="true">{DIRECTION_ARROWS[direction]}</span> {outside[direction]} {DIRECTION_WORDS[direction]}
                </button>
              ))}
            </div>
          )}
          <div className="graph-canvas-wrap">
            <div ref={container} className="graph-canvas" role="img" aria-label={`Evidence graph with ${model.nodes.length} nodes and ${model.edges.length} lines. The item list below it has the same items.`} />
            {DIRECTIONS.filter((direction) => outside[direction] > 0).map((direction) => (
              <div key={direction} className={`graph-edge-shade graph-edge-shade-${direction}`} aria-hidden="true" />
            ))}
          </div>
          <div className="graph-legend" aria-hidden="true">
            {legendKinds.map((kind) => (
              <span key={kind}>
                <EdgeGlyph kind={kind} />
                {kind === "feature" ? "Feature Exact used (its relation is on the line)" : kind === "bridge" ? "Comparison Exact recorded" : kind === "asserted" ? "Added: asserted parent" : kind === "structural" ? "Added: structural" : "Added: inferred"}
              </span>
            ))}
            <span>Rounded = source · cut corners = target</span>
          </div>
        </div>
        <aside className="graph-side" aria-label="Selected item and all graph items">
          <div className="graph-inspector" aria-live="polite" tabIndex={-1}>
            {inspectorBody()}
            {expandError && <p className="note note-warn">{expandError}</p>}
          </div>
          <div className="graph-items">
            {(["source", "target"] as GraphSide[]).map((side) => {
              const own = features.filter((edge) => edge.side === side);
              const hidden = model.nodes.find((node) => node.role === "group" && node.side === side);
              const compared = model.nodes.find((node) => node.role === "focal" && node.side === side);
              const undrawn = base.records.filter((record) => record.side === side && (record.status === "no_graph_form" || record.status === "unavailable"));
              return (
                <section key={side}>
                  <h4>{side === "source" ? "Source features" : "Target features"}</h4>
                  {compared && (
                    <button type="button" className={focusNodeId === compared.id ? "graph-item graph-item-node on" : "graph-item graph-item-node"} aria-pressed={focusNodeId === compared.id} onClick={() => choose(null, compared)}>
                      <span className="graph-item-kind">Compared</span> {entityName(compared)}
                    </button>
                  )}
                  {own.length ? (
                    <ul>
                      {own.map((edge) => (
                        <li key={edge.id}>
                          <button type="button" className={focusEdgeId === edge.id ? "graph-item on" : "graph-item"} aria-pressed={focusEdgeId === edge.id} data-tour="graph-feature" data-evidence-id={edge.evidenceId} data-fact-id={edge.factId ?? undefined} onClick={() => choose(edge, null, true)}>
                            <span className="graph-item-kind">{channelName(edge.channel ?? "")}</span> {itemText(edge)}
                          </button>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    !hidden && !undrawn.length && <p className="meta">None shown.</p>
                  )}
                  {hidden && (
                    <button type="button" className="link-btn" onClick={() => setLiterals((current) => Array.from(new Set([...current, side])))}>
                      Show {hidden.count} more records
                    </button>
                  )}
                  {undrawn.length > 0 && (
                    // Undrawn records stay listed (23 C04) but as one summary until asked for, so
                    // a dense pair does not bury its drawn features (21 F09).
                    <p className="meta graph-undrawn-summary">
                      {undrawn.length} {undrawn.length === 1 ? "record" : "records"} not drawn ({countsByChannel(undrawn.map((record) => record.channel))})
                      {undrawn.some((record) => record.status === "unavailable") ? " · no original fact resolved in this bundle" : " · no graph form"}{" "}
                      <button type="button" className="link-btn" aria-expanded={showUndrawn[side] || undrawn.some((record) => record.evidenceId === focusEvidence)} onClick={() => setShowUndrawn((current) => ({ ...current, [side]: !current[side] }))}>
                        {showUndrawn[side] ? "Hide them" : "List them"}
                      </button>
                    </p>
                  )}
                  {undrawn.length > 0 && (showUndrawn[side] || undrawn.some((record) => record.evidenceId === focusEvidence)) && (
                    <ul>
                      {undrawn.map((record, index) => (
                        <li key={`${record.evidenceId}-${record.factId ?? index}`}>
                          <button
                            type="button"
                            className={focusEvidence === record.evidenceId ? "graph-item on" : "graph-item"}
                            aria-pressed={focusEvidence === record.evidenceId}
                            data-evidence-id={record.evidenceId}
                            onClick={() => {
                              setFocusEdgeId(null);
                              setFocusNodeId(null);
                              setFocusEvidence(record.evidenceId);
                              onSelect(record.evidenceId);
                            }}
                          >
                            <span className="graph-item-kind">{channelName(record.channel)}</span> record not drawn
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              );
            })}
            <section aria-busy={metadata.loading || undefined}>
              <h4>Recorded comparisons</h4>
              {/* A failed read is not an absence (21 F04): features stay drawn; Retry keeps the view (R3). */}
              {metadata.error ? <ErrorNote error={metadata.error} onRetry={metadata.reload} what="Recorded comparisons" /> : null}
              {metadata.loading && !metadata.data && <p className="meta">Loading recorded comparisons…</p>}
              {base.bridges.length > 0 && (
                <ul>
                  {base.bridges
                    .filter((record) => record.status === "drawn")
                    .map((record) => {
                      const edge = model.edges.find((item) => item.id === record.edgeId);
                      if (!edge) return null;
                      return (
                        <li key={record.bridge.bridge_id}>
                          <button type="button" className={focusEdgeId === edge.id ? "graph-item on" : "graph-item"} aria-pressed={focusEdgeId === edge.id} onClick={() => choose(edge, null, true)}>
                            {relationText(edge)}
                            {record.bridge.level === "feature_correspondence" && <span className="meta"> · {correspondenceEnds(record.bridge)}</span>}
                            {record.copies > 1 && <span className="meta"> · recorded {record.copies}×</span>}
                          </button>
                        </li>
                      );
                    })}
                </ul>
              )}
              {base.bridges.some((record) => record.status !== "drawn") && (
                <p className="meta">
                  Also recorded, not drawn:{" "}
                  {(["not_drawn", "withheld"] as const)
                    .map((status) => {
                      // Record counts, so identical repeated records still count individually.
                      const channels = base.bridges.filter((record) => record.status === status).flatMap((record) => Array<string>(record.copies).fill(record.bridge.channel));
                      return channels.length ? `${channels.length} ${status === "withheld" ? "with withheld records" : "pairing a feature that is not drawn"} (${countsByChannel(channels)})` : null;
                    })
                    .filter(Boolean)
                    .join(" · ")}{" "}
                  <button type="button" className="link-btn" aria-expanded={showComparisons} onClick={() => setShowComparisons((value) => !value)}>
                    {showComparisons ? "Hide the list" : "List them"}
                  </button>
                </p>
              )}
              {showComparisons && base.bridges.length ? (
                <ul>
                  {base.bridges.filter((record) => record.status !== "drawn").map((record) => {
                    const edge = record.edgeId ? model.edges.find((item) => item.id === record.edgeId) : null;
                    const name = record.bridge.level === "feature_correspondence" ? `Recorded ${channelName(record.bridge.channel).toLowerCase()} correspondence` : `${channelName(record.bridge.channel)} compared`;
                    return (
                      <li key={record.bridge.bridge_id}>
                        {edge ? (
                          <button type="button" className={focusEdgeId === edge.id ? "graph-item on" : "graph-item"} aria-pressed={focusEdgeId === edge.id} onClick={() => choose(edge, null, true)}>
                            {name}
                          </button>
                        ) : (
                          <span className="graph-item graph-item-static">
                            {name} <span className="meta">· {record.status === "withheld" ? "its records are withheld in this view" : "a feature it pairs is not drawn"}{record.copies > 1 ? ` · recorded ${record.copies}×` : ""}</span>
                          </span>
                        )}
                      </li>
                    );
                  })}
                </ul>
              ) : null}
              {!base.bridges.length && !metadata.error && !metadata.loading && (
                <p className="meta">{base.bridgeStatus === "unknown" ? "No comparison record is available for this view." : "No comparison was recorded for this pair."}</p>
              )}
              {base.oneSided.length > 0 && <p className="meta">Features on one side only: {base.oneSided.map((item) => `${channelName(item.channel)} (${item.side})`).join(" · ")}.</p>}
            </section>
            {added.length > 0 && (
              <section>
                <h4>Added from the ontology</h4>
                <ul>
                  {added.map((edge) => (
                    <li key={edge.id}>
                      <button type="button" className={focusEdgeId === edge.id ? "graph-item on" : "graph-item"} aria-pressed={focusEdgeId === edge.id} onClick={() => choose(edge, null, true)}>
                        {itemText(edge)}
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
