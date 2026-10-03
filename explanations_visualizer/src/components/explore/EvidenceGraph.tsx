"use client";

// Optional evidence graph. Geometry changes only on explicit actions (buttons, drag-to-pan,
// selection by click or tap); hovering never moves, zooms or reflows anything. Line patterns
// and node shapes carry the same meaning as colour. The evidence list is the equivalent
// accessible view. The view (zoom, pan, expansions, selection) belongs to the compared pair:
// it survives re-renders, answer saves, tab switches, theme/text-size changes and layout
// changes, and only resets on request or when the pair itself changes.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { IconZoomIn, IconZoomOut } from "@/components/common/Icons";
import { channelName, type EvidenceBundle } from "@/components/explore/evidenceData";
import { curie, predicateName } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import { iriOf, isNamed } from "@/lib/owl";
import type { EntityKind, EntityRef, OwlNode } from "@/lib/types";
import { useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

type Side = "source" | "target";
interface GNode {
  id: string;
  side: Side;
  role: "focal" | "context" | "literal";
  iri?: string;
  kind: EntityKind;
  ontology: string;
  text?: string;
  group: "up" | "down" | "expanded";
  anchor?: string;
}
type EdgeKind = "feature" | "asserted" | "structural" | "bridge";
interface GEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  kind: EdgeKind;
  evidenceId?: string;
  factId?: string;
}
interface Expansion {
  nodes: GNode[];
  edges: GEdge[];
}
interface GraphView {
  zoom: number;
  pan: { x: number; y: number };
  expansions: Expansion[];
  focusNode: string | null;
  focusEdge: string | null;
}

const MAX_NODES = 150;
const READABLE_ZOOM = 0.7;
// The view of each compared pair, kept for this page's lifetime (bounded).
const views = new Map<string, GraphView>();
function remember(key: string, view: GraphView) {
  views.delete(key);
  views.set(key, view);
  while (views.size > 40) views.delete(views.keys().next().value as string);
}

export const EDGE_MEANING: Record<EdgeKind, string> = {
  asserted: "An asserted hierarchy axiom (subclass or subproperty), added to the graph from the ontology.",
  structural: "A parent reached by a documented structural rule, not asserted directly.",
  feature: "A feature Exact selected for this pair, projected from the original record shown in the evidence list.",
  bridge: "Exact compared features of this kind from both entities when scoring. This does not state that any two features correspond or are equivalent.",
};

function nodeId(ontology: string, iri: string) {
  return `n:${ontology}:${iri}`;
}

function buildModel(source: EntityRef, target: EntityRef, bundle: EvidenceBundle) {
  const nodes = new Map<string, GNode>();
  const edges: GEdge[] = [];
  let omitted = 0;
  const focal = (entity: EntityRef, side: Side) => {
    const id = nodeId(entity.ontology_version_id, entity.iri);
    nodes.set(id, { id, side, role: "focal", iri: entity.iri, kind: entity.kind, ontology: entity.ontology_version_id, group: "up" });
    return id;
  };
  const ids = { source: focal(source, "source"), target: focal(target, "target") };
  const context = (side: Side, ontology: string, iri: string, group: "up" | "down", kind: EntityKind = "class") => {
    const id = nodeId(ontology, iri);
    if (!nodes.has(id)) nodes.set(id, { id, side, role: "context", iri, kind, ontology, group });
    return id;
  };
  const channels = new Set<string>();
  for (const item of bundle.items) {
    channels.add(item.channel);
    const from = ids[item.side];
    for (const factId of item.fact_ids) {
      const axiom = bundle.axioms[factId];
      const ast = axiom?.ast as OwlNode | undefined;
      if (!ast) {
        omitted += 1;
        continue;
      }
      const ontology = item.entity.ontology_version_id;
      if (ast.type === "SubClassOf" && iriOf(ast.sub_class) === item.entity.iri) {
        const sup = ast.super_class as OwlNode;
        if (isNamed(sup)) {
          const to = context(item.side, ontology, iriOf(sup)!, "up");
          edges.push({ id: `e:${item.evidence_id}:${factId}`, source: from, target: to, label: "subclass of", kind: "feature", evidenceId: item.evidence_id, factId });
          continue;
        }
        if (/SomeValuesFrom|AllValuesFrom/.test(sup.type) && isNamed(sup.filler) && isNamed(sup.property)) {
          const to = context(item.side, ontology, iriOf(sup.filler)!, "down");
          edges.push({
            id: `e:${item.evidence_id}:${factId}`,
            source: from,
            target: to,
            label: `${iriOf(sup.property)}\u0000${sup.type.includes("Some") ? "some" : "only"}`,
            kind: "feature",
            evidenceId: item.evidence_id,
            factId,
          });
          continue;
        }
      }
      if (ast.type === "AnnotationAssertion" && (ast.value as OwlNode)?.type === "Literal") {
        const text = String((ast.value as OwlNode).lexical_form);
        const id = `lit:${factId}`;
        nodes.set(id, { id, side: item.side, role: "literal", kind: "class", ontology, text: text.length > 60 ? `${text.slice(0, 57)}…` : text, group: "down" });
        edges.push({ id: `e:${item.evidence_id}:${factId}`, source: from, target: id, label: predicateName(iriOf(ast.property)), kind: "feature", evidenceId: item.evidence_id, factId });
        continue;
      }
      omitted += 1;
    }
  }
  for (const channel of channels) {
    edges.push({ id: `bridge:${channel}`, source: ids.source, target: ids.target, label: `${channelName(channel)} features compared`, kind: "bridge" });
  }
  return { nodes: Array.from(nodes.values()), edges, omitted };
}

function layout(nodes: GNode[]): Record<string, { x: number; y: number }> {
  const positions: Record<string, { x: number; y: number }> = {};
  const spread = (list: GNode[], centre: number, y: number, towardsCentre: number) => {
    const xs = list.map((_, index) => centre + (index - (list.length - 1) / 2) * 250);
    const edge = Math.max(...xs.map((x) => Math.sign(towardsCentre) * x), -Infinity);
    const limit = Math.sign(towardsCentre) * towardsCentre;
    const shift = edge > limit ? (edge - limit) * -Math.sign(towardsCentre) : 0;
    list.forEach((node, index) => {
      positions[node.id] = { x: xs[index] + shift, y };
    });
  };
  for (const side of ["source", "target"] as Side[]) {
    const centre = side === "source" ? -300 : 300;
    const towards = side === "source" ? -150 : 150;
    const focal = nodes.find((node) => node.side === side && node.role === "focal");
    if (focal) positions[focal.id] = { x: centre, y: 0 };
    spread(nodes.filter((node) => node.side === side && node.role !== "focal" && node.group === "up"), centre, -200, towards);
    spread(nodes.filter((node) => node.side === side && node.role !== "focal" && node.group === "down"), centre, 200, towards);
  }
  return positions;
}

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

/** Legend glyphs drawn with the same patterns as the rendered edges. */
const CYTOSCAPE_STYLESHEET = "__________cytoscape_stylesheet";

export function EdgeGlyph({ kind }: { kind: EdgeKind }) {
  if (kind === "bridge")
    return (
      <svg width="40" height="10" aria-hidden="true">
        <line x1="3" y1="5" x2="37" y2="5" stroke="var(--bridge)" strokeWidth="5" strokeLinecap="round" strokeDasharray="0.1 9" />
      </svg>
    );
  const dash = kind === "feature" ? "10 4 2 4" : kind === "structural" ? "8 5" : undefined;
  return (
    <svg width="40" height="10" aria-hidden="true">
      <line x1="0" y1="5" x2="40" y2="5" stroke="currentColor" strokeWidth="2.5" strokeDasharray={dash} />
    </svg>
  );
}

export function EvidenceGraph({
  source,
  target,
  bundle,
  viewKey,
  selected,
  onSelect,
  canExpand = true,
}: {
  source: EntityRef;
  target: EntityRef;
  bundle: EvidenceBundle;
  /** Identity of the compared pair in its product and scope; a new key starts a new view. */
  viewKey: string;
  selected: string | null;
  onSelect: (evidenceId: string | null) => void;
  canExpand?: boolean;
}) {
  const workspace = useWorkspace();
  const report = useWorkspaceAction();
  const container = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cyRef = useRef<any>(null);
  const stored = views.get(viewKey);
  const [expansions, setExpansions] = useState<Expansion[]>(() => stored?.expansions ?? []);
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
    setFocusNodeId(next?.focusNode ?? null);
    setFocusEdgeId(next?.focusEdge ?? null);
    setExpandError(null);
  }, [viewKey]);

  const base = useMemo(() => buildModel(source, target, bundle), [source, target, bundle]);
  const model = useMemo(() => {
    const nodes = new Map(base.nodes.map((node) => [node.id, node]));
    const edges = [...base.edges];
    for (const step of expansions) {
      step.nodes.forEach((node) => {
        if (!nodes.has(node.id)) nodes.set(node.id, node);
      });
      step.edges.forEach((edge) => {
        if (!edges.some((existing) => existing.id === edge.id)) edges.push(edge);
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
      const current = views.get(viewKey) ?? { zoom: 0, pan: { x: 0, y: 0 }, expansions: [], focusNode: null, focusEdge: null };
      remember(viewKey, { ...current, ...changes });
    },
    [viewKey],
  );
  useEffect(() => save({ expansions, focusNode: focusNodeId, focusEdge: focusEdgeId }), [expansions, focusNodeId, focusEdgeId, save]);

  const sourceIris = model.nodes.filter((node) => node.iri && node.ontology === source.ontology_version_id).map((node) => node.iri!);
  const targetIris = model.nodes.filter((node) => node.iri && node.ontology === target.ontology_version_id).map((node) => node.iri!);
  const propertyIris = model.edges.filter((edge) => edge.label.includes("\u0000")).map((edge) => edge.label.split("\u0000")[0]);
  const sourceLabel = useLabelLookup(source.ontology_version_id, [...sourceIris, ...propertyIris]);
  const targetLabel = useLabelLookup(target.ontology_version_id, [...targetIris, ...propertyIris]);

  const labelFor = useCallback(
    (node: GNode) => {
      if (node.role === "literal") return `“${node.text}”`;
      const lookup = node.ontology === source.ontology_version_id ? sourceLabel : targetLabel;
      const text = lookup(node.iri!)?.value ?? curie(node.iri!);
      if (node.role === "focal") return `${node.side === "source" ? "SOURCE" : "TARGET"} ${node.kind === "class" ? "CLASS" : node.kind.replace(/_/g, " ").toUpperCase()}\n${text}`;
      return text;
    },
    [source, sourceLabel, targetLabel],
  );
  const edgeLabel = useCallback(
    (edge: GEdge) => {
      if (!edge.label.includes("\u0000")) return edge.label;
      const [property, quantifier] = edge.label.split("\u0000");
      const node = model.nodes.find((item) => item.id === edge.source);
      const lookup = node?.ontology === source.ontology_version_id ? sourceLabel : targetLabel;
      return `${lookup(property)?.value ?? curie(property)} · ${quantifier}`;
    },
    [model.nodes, source, sourceLabel, targetLabel],
  );

  const positions = useMemo(() => {
    const placed = layout(model.nodes.filter((node) => node.group !== "expanded"));
    // Expanded parents sit above the node they were added from, in insertion order.
    const perAnchor: Record<string, number> = {};
    for (const node of model.nodes.filter((item) => item.group === "expanded")) {
      const anchor = node.anchor ? placed[node.anchor] : undefined;
      const index = (perAnchor[node.anchor ?? ""] = (perAnchor[node.anchor ?? ""] ?? -1) + 1);
      placed[node.id] = anchor ? { x: anchor.x + (index - 0.5) * 240, y: anchor.y - 180 } : { x: 0, y: -400 - index * 90 };
    }
    return placed;
  }, [model.nodes]);
  const positionsRef = useRef(positions);
  positionsRef.current = positions;

  const styles = useCallback(() => {
    const ink = cssVar("--ink");
    const ink2 = cssVar("--ink-2");
    const surface = cssVar("--surface");
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
          "font-size": 15,
          "text-wrap": "wrap",
          "text-max-width": 210,
          "text-valign": "center",
          "text-halign": "center",
          width: "data(width)",
          height: "data(height)",
          padding: 10,
        },
      },
      { selector: "node[side = 'target']", style: { shape: "rectangle" } },
      { selector: "node[role = 'focal']", style: { "background-color": "data(line)", color: cssVar("--on-primary"), "font-weight": 600, "border-width": 0 } },
      { selector: "node[role = 'literal']", style: { "font-style": "italic", "border-style": "dashed" } },
      { selector: "node:selected", style: { "border-width": 4, "border-color": cssVar("--focus") } },
      {
        selector: "edge",
        style: {
          width: 2.5,
          "line-color": ink2,
          "target-arrow-color": ink2,
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "control-point-step-size": 70,
          label: "data(label)",
          "font-size": 13,
          "font-family": "IBM Plex Sans, Segoe UI, sans-serif",
          color: ink,
          "text-background-color": cssVar("--paper"),
          "text-background-opacity": 1,
          "text-background-padding": 3,
          "text-wrap": "wrap",
          "text-max-width": 180,
        },
      },
      { selector: "edge[kind = 'feature']", style: { "line-style": "dashed", "line-dash-pattern": [10, 4, 2, 4] } },
      { selector: "edge[kind = 'structural']", style: { "line-style": "dashed", "line-dash-pattern": [8, 5] } },
      {
        // A channel-level comparison: a thick line of round dots (dash pattern with round caps), no arrow.
        selector: "edge[kind = 'bridge']",
        style: { "line-style": "dashed", "line-cap": "round", "line-dash-pattern": [0.1, 9], width: 5, "line-color": cssVar("--bridge"), "target-arrow-shape": "none", color: cssVar("--bridge") },
      },
      { selector: "edge:selected", style: { width: 5, "line-color": cssVar("--focus"), "target-arrow-color": cssVar("--focus") } },
    ];
  }, []);

  const elements = useCallback(() => {
    const els: unknown[] = [];
    for (const node of model.nodes) {
      const label = labelFor(node);
      const longest = Math.max(...label.split("\n").map((line) => line.length));
      const lines = label.split("\n").length + Math.floor(longest / 26);
      els.push({
        group: "nodes",
        data: {
          id: node.id,
          label,
          side: node.side,
          role: node.role,
          line: node.side === "source" ? cssVar("--source") : cssVar("--target"),
          width: Math.min(230, Math.max(110, longest * 8.4 + 28)),
          height: 26 + lines * 20,
        },
        position: positions[node.id],
      });
    }
    for (const edge of model.edges) {
      els.push({ group: "edges", data: { id: edge.id, source: edge.source, target: edge.target, label: edgeLabel(edge), kind: edge.kind } });
    }
    return els;
  }, [model, labelFor, edgeLabel, positions]);
  const elementsRef = useRef(elements);
  elementsRef.current = elements;

  /** The starting view: readable labels first, centred on the compared pair. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const initialView = useCallback((cy: any) => {
    cy.fit(undefined, 40);
    if (cy.zoom() > 1.15) {
      cy.zoom(1.15);
      cy.center();
    } else if (cy.zoom() < READABLE_ZOOM) {
      cy.zoom(READABLE_ZOOM);
      cy.center(cy.nodes("[role = 'focal']"));
    }
  }, []);

  const bindTaps = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (cy: any) => {
      cy.off("tap");
      cy.on("tap", "node", (event: { target: { id: () => string } }) => {
        setFocusNodeId(event.target.id());
        setFocusEdgeId(null);
      });
      cy.on("tap", "edge", (event: { target: { id: () => string } }) => {
        const edge = modelRef.current.edges.find((item) => item.id === event.target.id()) ?? null;
        setFocusEdgeId(edge?.id ?? null);
        setFocusNodeId(null);
        onSelect(edge?.evidenceId ?? null);
        if (edge) report({ type: "graph_edge", edgeKind: edge.kind, evidenceId: edge.evidenceId ?? null });
      });
    },
    [onSelect, report],
  );

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
        minZoom: 0.35,
        maxZoom: 2.5,
      });
      const previous = views.get(viewKey);
      if (previous && previous.zoom > 0) {
        cy.zoom(previous.zoom);
        cy.pan(previous.pan);
      } else initialView(cy);
      cy.on("viewport", () => save({ zoom: cy.zoom(), pan: { ...cy.pan() } }));
      save({ zoom: cy.zoom(), pan: { ...cy.pan() } });
      bindTaps(cy);
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

  // Sync elements (labels, expansions) and theme without moving existing nodes or the view.
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
        if (existing.nonempty()) existing.data(element.data);
        else cy.add(element);
      }
      cy.style(styles());
    });
    bindTaps(cy);
  }, [elements, styles, themeVersion, bindTaps]);

  // Selection from the list highlights the matching edge; it never moves the view.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().unselect();
    if (selected) cy.edges().filter((edge: { data: (key: string) => string }) => edge.data("id").startsWith(`e:${selected}:`)).select();
  }, [selected, elements]);

  const zoom = (factor: number) => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.zoom({ level: Math.max(0.35, Math.min(2.5, cy.zoom() * factor)), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
    report({ type: "graph_zoom" });
  };
  const fit = () => {
    cyRef.current?.fit(undefined, 40);
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

  const expandParents = async (node: GNode) => {
    setExpandError(null);
    if (!node.iri) return;
    if (model.nodes.length >= MAX_NODES) {
      setExpandError(`The graph already shows ${MAX_NODES} nodes. Use the hierarchy view for further navigation.`);
      return;
    }
    try {
      const page = await workspace.hierarchy({ ontology_version_id: node.ontology, iri: node.iri, kind: node.kind }, "parents", "literal_asserted", null);
      const nodes: GNode[] = [];
      const edges: GEdge[] = [];
      for (const edge of page.items) {
        const id = nodeId(node.ontology, edge.parent.iri);
        if (!model.nodes.some((item) => item.id === id)) nodes.push({ id, side: node.side, role: "context", iri: edge.parent.iri, kind: edge.parent.kind, ontology: node.ontology, group: "expanded", anchor: node.id });
        edges.push({ id: `h:${edge.id}`, source: node.id, target: id, label: node.kind === "class" ? "subclass of" : node.kind === "individual" ? "instance of" : "subproperty of", kind: edge.basis === "structural_navigation" ? "structural" : "asserted" });
      }
      if (!edges.length) {
        setExpandError(page.status === "absent_in_scope" ? "No asserted parent is recorded for this node in the loaded scope." : page.reason ?? "Parents of this node are not available in this view.");
        return;
      }
      setExpansions((list) => [...list, { nodes, edges }]);
    } catch {
      setExpandError("Parents could not be loaded.");
    }
  };

  const nodeCount = model.nodes.length;
  const edgeCount = model.edges.length;
  const kinds = new Set(model.edges.map((edge) => edge.kind));
  return (
    <div className="graph-wrap">
      <div className="graph-toolbar" role="toolbar" aria-label="Graph view controls">
        <button type="button" className="icon-btn" aria-label="Zoom in" onClick={() => zoom(1.25)}>
          <IconZoomIn />
        </button>
        <button type="button" className="icon-btn" aria-label="Zoom out" onClick={() => zoom(0.8)}>
          <IconZoomOut />
        </button>
        <button type="button" className="btn btn-sm" onClick={fit} title="Zoom out until every node is visible">
          Fit all
        </button>
        <button type="button" className="btn btn-sm" onClick={reset} title="Return to the starting layout and view, and clear the selection">
          Reset view
        </button>
        <span className="graph-toolbar-spacer" />
        {expansionAllowed && (
          <button type="button" className="btn btn-sm" disabled={!expansions.length} onClick={() => setExpansions((list) => list.slice(0, -1))}>
            Undo last expansion
          </button>
        )}
      </div>
      <div ref={container} className="graph-canvas" role="img" aria-label={`Evidence graph with ${nodeCount} nodes and ${edgeCount} edges. The evidence list contains the same items.`} />
      <div className="graph-legend" aria-hidden="true">
        {(["asserted", "structural", "feature", "bridge"] as EdgeKind[])
          .filter((kind) => kind === "asserted" || kind === "feature" || kinds.has(kind))
          .map((kind) => (
            <span key={kind}>
              <EdgeGlyph kind={kind} />
              {kind === "asserted" ? "Ontology assertion" : kind === "structural" ? "Structural relation" : kind === "feature" ? "Feature Exact used" : "Feature kind compared across ontologies"}
            </span>
          ))}
        <span>Rounded = source side · square = target side</span>
      </div>
      <p className="meta">
        {nodeCount} nodes · {edgeCount} edges shown{base.omitted ? ` · ${base.omitted} records have no graph form and appear only in the list` : " · nothing omitted"}. “Fit all” shows every node; “Reset view” returns to the starting view.
      </p>
      {(focusNode || focusEdge || expandError) && (
        <div className="graph-detail card-inset" aria-live="polite">
          {focusNode && (
            <>
              <h3>{labelFor(focusNode).replace(/^.*\n/, "")}</h3>
              {focusNode.iri && <p className="iri">{focusNode.iri}</p>}
              <p className="meta">{focusNode.side === "source" ? "Source ontology" : "Target ontology"} · {focusNode.role === "focal" ? "compared entity" : focusNode.role === "literal" ? "recorded literal" : "related entity"}</p>
              {expansionAllowed && focusNode.iri && focusNode.role !== "literal" && (
                <button type="button" className="btn btn-sm" onClick={() => expandParents(focusNode)}>
                  Add its parents to the graph
                </button>
              )}
            </>
          )}
          {focusEdge && (
            <>
              <h3>{edgeLabel(focusEdge)}</h3>
              <p className="muted">{EDGE_MEANING[focusEdge.kind]}</p>
            </>
          )}
          {expandError && <p className="note note-warn">{expandError}</p>}
        </div>
      )}
    </div>
  );
}
