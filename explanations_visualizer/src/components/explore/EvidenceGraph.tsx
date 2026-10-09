"use client";

// Optional evidence graph (21 F08). The model (lib/evidenceGraphModel) keeps typed identities,
// draws one line per selected original record, and draws a comparison line only for a channel
// recorded on both sides. Source sits left, target right, each side's features in a column;
// relations are written inside the nodes, so no text sits on crossing lines. Beside the canvas,
// an inspector explains the current selection and a complete item list gives the same items to
// keyboard and screen-reader users. Geometry changes only on explicit actions; the view (zoom,
// pan, expansions, selection) belongs to the compared pair and survives re-renders, answer
// saves, tab switches and theme or text-size changes.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { IconZoomIn, IconZoomOut } from "@/components/common/Icons";
import { channelName, type EvidenceBundle } from "@/components/explore/evidenceData";
import { useFactInspector } from "@/components/workspace/FactInspector";
import { buildGraphModel, entityNodeId, layoutGraph, type GraphSide, type ModelEdge, type ModelNode } from "@/lib/evidenceGraphModel";
import { curie, KIND_NAMES, predicateName } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import type { EntityRef } from "@/lib/types";
import { useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

interface Expansion {
  nodes: ModelNode[];
  edges: ModelEdge[];
}
interface GraphView {
  zoom: number;
  pan: { x: number; y: number };
  expansions: Expansion[];
  literals: GraphSide[];
  focusNode: string | null;
  focusEdge: string | null;
}

const MAX_NODES = 150;
const READABLE_ZOOM = 0.7;
const NODE_WIDTH = 220;
const FOCAL_WIDTH = 220;
const CHARS_PER_LINE = 26;
// The view of each compared pair, kept for this page's lifetime (bounded).
const views = new Map<string, GraphView>();
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
  bridge: "Exact compared this kind of feature on both sides when scoring. It does not say that any two features correspond.",
};

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
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

export function EvidenceGraph({
  source,
  target,
  bundle,
  viewKey,
  selected,
  onSelect,
  onShowInList,
  canExpand = true,
}: {
  source: EntityRef;
  target: EntityRef;
  bundle: EvidenceBundle;
  /** Identity of the compared pair in its product and scope; a new key starts a new view. */
  viewKey: string;
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
  const [expandError, setExpandError] = useState<string | null>(null);
  const themeVersion = useThemeVersion();
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
    setExpandError(null);
  }, [viewKey]);

  const base = useMemo(() => buildGraphModel(source, target, bundle.items, bundle.axioms as never, { expandLiterals: new Set(literals) }), [source, target, bundle, literals]);
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

  const save = useCallback(
    (changes: Partial<GraphView>) => {
      const current = views.get(viewKey) ?? { zoom: 0, pan: { x: 0, y: 0 }, expansions: [], literals: [], focusNode: null, focusEdge: null };
      remember(viewKey, { ...current, ...changes });
    },
    [viewKey],
  );
  useEffect(() => save({ expansions, literals, focusNode: focusNodeId, focusEdge: focusEdgeId }), [expansions, literals, focusNodeId, focusEdgeId, save]);

  // Labels, per ontology, for entities and the properties named in restrictions.
  const propertyIris = model.edges.map((edge) => edge.relation.property).filter((iri): iri is string => Boolean(iri));
  const sourceLabel = useLabelLookup(source.ontology_version_id, [...model.nodes.filter((node) => node.iri && node.ontology === source.ontology_version_id).map((node) => node.iri!), ...propertyIris]);
  const targetLabel = useLabelLookup(target.ontology_version_id, [...model.nodes.filter((node) => node.iri && node.ontology === target.ontology_version_id).map((node) => node.iri!), ...propertyIris]);
  const lookup = useCallback((ontology: string) => (ontology === source.ontology_version_id ? sourceLabel : targetLabel), [source, sourceLabel, targetLabel]);
  const entityName = useCallback((node: ModelNode) => (node.iri ? lookup(node.ontology)(node.iri)?.value ?? curie(node.iri) : ""), [lookup]);

  const relationText = useCallback(
    (edge: ModelEdge, side: GraphSide | undefined = edge.side) => {
      if (edge.kind === "bridge") return `${channelName(edge.channel ?? "")} compared`;
      if (edge.relation.text) return edge.relation.text;
      const ontology = side === "target" ? target.ontology_version_id : source.ontology_version_id;
      if (edge.relation.property && edge.relation.quantifier) return `${lookup(ontology)(edge.relation.property)?.value ?? curie(edge.relation.property)} ${edge.relation.quantifier}`;
      if (edge.relation.property) return predicateName(edge.relation.property);
      return "";
    },
    [lookup, source, target],
  );
  const incoming = useMemo(() => {
    const map = new Map<string, ModelEdge>();
    for (const edge of model.edges) if (edge.kind === "feature" && !map.has(edge.target)) map.set(edge.target, edge);
    return map;
  }, [model.edges]);

  /** Drawn label: a relation line where the node is a feature's object, then the name. */
  const nodeLabel = useCallback(
    (node: ModelNode) => {
      if (node.role === "focal") return `${node.side === "source" ? "Source" : "Target"} ${KIND_NAMES[node.kind]}\n${clip(entityName(node), 70)}`;
      if (node.role === "group") return `+${node.count} more records\nselect to show all`;
      const edge = incoming.get(node.id);
      const relation = edge ? relationText(edge) : "";
      const name = node.role === "literal" ? `“${clip(node.text ?? "", 66)}”` : clip(entityName(node), 66);
      return relation ? `${relation}\n${name}` : name;
    },
    [entityName, incoming, relationText],
  );
  const sizeOf = useCallback(
    (node: ModelNode) => {
      const lines = nodeLabel(node).split("\n").reduce((sum, line) => sum + wrapLines(line), 0);
      return { width: node.role === "focal" ? FOCAL_WIDTH : NODE_WIDTH, height: 20 + lines * 19 };
    },
    [nodeLabel],
  );
  const positions = useMemo(() => layoutGraph(model.nodes, sizeOf), [model.nodes, sizeOf]);
  const positionsRef = useRef(positions);
  positionsRef.current = positions;

  const styles = useCallback(() => {
    const ink = cssVar("--ink");
    const ink3 = cssVar("--ink-3");
    const surface = cssVar("--surface");
    const surface2 = cssVar("--surface-2");
    const focus = cssVar("--focus");
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
          "font-family": "IBM Plex Sans, Segoe UI, sans-serif",
          "font-size": 14,
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
      { selector: "node[role = 'focal']", style: { "background-color": surface2, "border-width": 3, "font-weight": 600, "font-size": 15 } },
      { selector: "node[role = 'literal']", style: { "font-style": "italic", "border-style": "dashed" } },
      { selector: "node[role = 'group']", style: { "border-style": "dotted", "border-width": 2, color: ink3 } },
      { selector: "node:selected", style: { "border-width": 4, "border-color": focus } },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": ink3,
          "target-arrow-color": ink3,
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.9,
          "curve-style": "bezier",
          "control-point-step-size": 60,
          "font-size": 13,
          "font-family": "IBM Plex Sans, Segoe UI, sans-serif",
          color: ink,
          "text-background-color": surface,
          "text-background-opacity": 1,
          "text-background-padding": 3,
        },
      },
      { selector: "edge[kind = 'feature']", style: { "line-style": "dashed", "line-dash-pattern": [9, 4, 2, 4] } },
      { selector: "edge[kind = 'group']", style: { "line-style": "dotted", "target-arrow-shape": "none" } },
      { selector: "edge[kind = 'structural']", style: { "line-style": "dashed", "line-dash-pattern": [7, 5] } },
      { selector: "edge[kind = 'inferred']", style: { "line-style": "dotted" } },
      {
        // A channel-level comparison: a line of round dots, no arrow, labelled with the channel.
        selector: "edge[kind = 'bridge']",
        style: { "line-style": "dashed", "line-cap": "round", "line-dash-pattern": [0.1, 8], width: 4, "line-color": cssVar("--bridge"), "target-arrow-shape": "none", color: ink, "control-point-step-size": 26 },
      },
      { selector: ".faded", style: { opacity: 0.35 } },
      // Feature relations are written in their nodes; only selected comparison and context lines get a label.
      { selector: "edge:selected", style: { width: 4, "line-color": focus, "target-arrow-color": focus, opacity: 1 } },
      { selector: "edge[kind != 'feature'][kind != 'group']:selected", style: { label: "data(label)" } },
    ];
  }, []);

  const elements = useCallback(() => {
    const els: unknown[] = [];
    for (const node of model.nodes) {
      const size = sizeOf(node);
      els.push({
        group: "nodes",
        data: { id: node.id, label: nodeLabel(node), side: node.side, role: node.role, line: node.side === "source" ? cssVar("--source") : cssVar("--target"), width: size.width, height: size.height, textWidth: size.width - 22 },
        position: positions[node.id],
      });
    }
    for (const edge of model.edges) {
      els.push({ group: "edges", data: { id: edge.id, source: edge.source, target: edge.target, label: relationText(edge), kind: edge.kind } });
    }
    return els;
  }, [model, nodeLabel, relationText, positions, sizeOf]);
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
    cy.pan({ x: cy.width() / 2 - ((box.x1 + box.x2) / 2) * level, y: cy.height() / 2 - ((focal.y1 + focal.y2) / 2) * level });
  }, []);

  const choose = useCallback(
    (edge: ModelEdge | null, node: ModelNode | null) => {
      setFocusEdgeId(edge?.id ?? null);
      setFocusNodeId(node?.id ?? null);
      const group = edge?.kind === "group" ? edge.side : node?.role === "group" ? node.side : null;
      if (group) {
        setLiterals((current) => Array.from(new Set([...current, group])));
        return;
      }
      if (!edge) return;
      onSelect(edge.evidenceId ?? null);
      // Only a feature line is evidence; bridges and context lines are inspection only (21 F01).
      report({ type: "graph_edge", edgeKind: edge.kind as Exclude<ModelEdge["kind"], "group">, evidenceId: edge.evidenceId ?? null, factId: edge.factId ?? null });
    },
    [onSelect, report],
  );
  const chooseRef = useRef(choose);
  chooseRef.current = choose;

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
      if (previous && previous.zoom > 0) {
        cy.zoom(previous.zoom);
        cy.pan(previous.pan);
      } else initialView(cy);
      cy.on("viewport", () => save({ zoom: cy.zoom(), pan: { ...cy.pan() } }));
      save({ zoom: cy.zoom(), pan: { ...cy.pan() } });
      cy.on("tap", "node", (event: { target: { id: () => string } }) => chooseRef.current(null, modelRef.current.nodes.find((node) => node.id === event.target.id()) ?? null));
      cy.on("tap", "edge", (event: { target: { id: () => string } }) => chooseRef.current(modelRef.current.edges.find((edge) => edge.id === event.target.id()) ?? null, null));
      cyRef.current = cy;
      // A resized container (layout switch, text size) keeps the same view.
      observer = new ResizeObserver(() => cy.resize());
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

  // Sync elements (labels, expansions) and theme without moving the view.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
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
  }, [elements, styles, themeVersion]);

  // Selection (here or from the list) highlights its line and fades the rest; the view never moves.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().unselect().removeClass("faded");
    const edges = focusEdgeId ? cy.getElementById(focusEdgeId) : selected ? cy.edges().filter((edge: { data: (key: string) => string }) => edge.data("id").startsWith(`e:${selected}:`)) : null;
    const node = focusNodeId ? cy.getElementById(focusNodeId) : null;
    const chosen = edges?.nonempty() ? edges : node?.nonempty() ? node : null;
    if (!chosen) return;
    chosen.select();
    const keep = chosen.union(chosen.connectedNodes()).union(chosen.connectedEdges());
    cy.elements().difference(keep).addClass("faded");
  }, [selected, focusEdgeId, focusNodeId, elements]);

  // A feature chosen in the evidence list becomes this view's selection.
  useEffect(() => {
    if (!selected) return;
    const current = modelRef.current.edges.find((edge) => edge.id === focusEdgeId);
    if (current?.evidenceId === selected) return;
    const first = modelRef.current.edges.find((edge) => edge.evidenceId === selected);
    if (first) {
      setFocusEdgeId(first.id);
      setFocusNodeId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

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
        edges.push({ id: `h:${node.side}:${edge.id}`, source: node.id, target: id, kind, side: node.side, relation: { text: node.kind === "class" ? "subclass of" : node.kind === "individual" ? "instance of" : "subproperty of" } });
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
  const bridges = model.edges.filter((edge) => edge.kind === "bridge");
  const added = model.edges.filter((edge) => ["asserted", "structural", "inferred"].includes(edge.kind));
  const nodeById = (id: string) => model.nodes.find((node) => node.id === id);
  const itemText = (edge: ModelEdge) => {
    const to = nodeById(edge.target);
    const name = to?.role === "literal" ? `“${clip(to.text ?? "", 60)}”` : to ? entityName(to) : "";
    if (edge.kind === "bridge") return relationText(edge);
    if (edge.kind === "feature") return `${relationText(edge)} ${name}`.trim();
    const from = nodeById(edge.source);
    return `${from ? entityName(from) : ""} ${relationText(edge)} ${name}`.trim();
  };
  const legendKinds = (["feature", "bridge", "asserted", "structural", "inferred"] as EdgeKind[]).filter((kind) => kind === "feature" || model.edges.some((edge) => edge.kind === kind));

  const inspectorBody = () => {
    if (focusEdge) {
      const meaning = EDGE_MEANING[(focusEdge.kind === "group" ? "feature" : focusEdge.kind) as EdgeKind];
      if (focusEdge.kind === "bridge") {
        const count = (side: GraphSide) => features.filter((edge) => edge.side === side && edge.channel === focusEdge.channel).length;
        return (
          <>
            <h4 className="graph-inspector-title">{relationText(focusEdge)}</h4>
            <p>{meaning}</p>
            <p className="meta">
              Source: {count("source")} {count("source") === 1 ? "record" : "records"} · Target: {count("target")} {count("target") === 1 ? "record" : "records"}
            </p>
          </>
        );
      }
      if (focusEdge.kind === "feature") {
        const subject = focusEdge.side === "source" ? source : target;
        return (
          <>
            <h4 className="graph-inspector-title">
              {channelName(focusEdge.channel ?? "")} feature · {focusEdge.side === "source" ? "source" : "target"}
            </h4>
            <p className="fact-reading">{itemText(focusEdge)}</p>
            <p className="meta">{meaning}</p>
            <div className="graph-inspector-actions">
              {inspector && focusEdge.factId && (
                <button type="button" className="btn btn-sm" onClick={() => inspector.open({ factId: focusEdge.factId!, ontologies: [subject.ontology_version_id], subject }, `${channelName(focusEdge.channel ?? "")} record`, "record")}>
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
          <div ref={container} className="graph-canvas" role="img" aria-label={`Evidence graph with ${model.nodes.length} nodes and ${model.edges.length} lines. The item list below it has the same items.`} />
          <div className="graph-legend" aria-hidden="true">
            {legendKinds.map((kind) => (
              <span key={kind}>
                <EdgeGlyph kind={kind} />
                {kind === "feature" ? "Feature Exact used" : kind === "bridge" ? "Kind compared on both sides" : kind === "asserted" ? "Added: asserted parent" : kind === "structural" ? "Added: structural" : "Added: inferred"}
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
                          <button type="button" className={focusEdgeId === edge.id ? "graph-item on" : "graph-item"} aria-pressed={focusEdgeId === edge.id} data-tour="graph-feature" onClick={() => choose(edge, null)}>
                            <span className="graph-item-kind">{channelName(edge.channel ?? "")}</span> {itemText(edge)}
                          </button>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="meta">None shown.</p>
                  )}
                  {hidden && (
                    <button type="button" className="link-btn" onClick={() => setLiterals((current) => Array.from(new Set([...current, side])))}>
                      Show {hidden.count} more records
                    </button>
                  )}
                  {base.omitted[side] > 0 && (
                    <p className="meta">
                      {base.omitted[side]} {base.omitted[side] === 1 ? "record has" : "records have"} no graph form; see the evidence list.
                    </p>
                  )}
                </section>
              );
            })}
            <section>
              <h4>Compared kinds</h4>
              {bridges.length ? (
                <ul>
                  {bridges.map((edge) => (
                    <li key={edge.id}>
                      <button type="button" className={focusEdgeId === edge.id ? "graph-item on" : "graph-item"} aria-pressed={focusEdgeId === edge.id} onClick={() => choose(edge, null)}>
                        {relationText(edge)}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="meta">No kind of feature was recorded on both sides.</p>
              )}
              {base.oneSided.length > 0 && <p className="meta">{base.oneSided.map((item) => `${channelName(item.channel)}: ${item.side} only`).join(" · ")}. No comparison line for these.</p>}
            </section>
            {added.length > 0 && (
              <section>
                <h4>Added from the ontology</h4>
                <ul>
                  {added.map((edge) => (
                    <li key={edge.id}>
                      <button type="button" className={focusEdgeId === edge.id ? "graph-item on" : "graph-item"} aria-pressed={focusEdgeId === edge.id} onClick={() => choose(edge, null)}>
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
