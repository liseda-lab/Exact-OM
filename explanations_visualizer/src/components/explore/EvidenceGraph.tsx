"use client";

// Optional evidence graph. Geometry changes only on explicit actions (buttons, drag-to-pan,
// selection by click or tap); hovering never moves, zooms or reflows anything. Line patterns
// and node shapes carry the same meaning as colour. The evidence list is the equivalent
// accessible view.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { IconZoomIn, IconZoomOut } from "@/components/common/Icons";
import { channelName, type EvidenceBundle } from "@/components/explore/evidenceData";
import { getJson } from "@/lib/api";
import { curie, predicateName } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import { iriOf, isNamed } from "@/lib/owl";
import type { EntityRef, HierarchyPage, OwlNode } from "@/lib/types";

type Side = "source" | "target";
interface GNode {
  id: string;
  side: Side;
  role: "focal" | "context" | "literal";
  iri?: string;
  ontology: string;
  text?: string;
  group: "up" | "down" | "expanded";
  anchor?: string;
}
interface GEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  kind: "feature" | "asserted" | "bridge";
  evidenceId?: string;
  factId?: string;
}

const MAX_NODES = 150;

function nodeId(ontology: string, iri: string) {
  return `n:${ontology}:${iri}`;
}

function buildModel(source: EntityRef, target: EntityRef, bundle: EvidenceBundle) {
  const nodes = new Map<string, GNode>();
  const edges: GEdge[] = [];
  let omitted = 0;
  const focal = (entity: EntityRef, side: Side) => {
    const id = nodeId(entity.ontology_version_id, entity.iri);
    nodes.set(id, { id, side, role: "focal", iri: entity.iri, ontology: entity.ontology_version_id, group: "up" });
    return id;
  };
  const ids = { source: focal(source, "source"), target: focal(target, "target") };
  const context = (side: Side, ontology: string, iri: string, group: "up" | "down") => {
    const id = nodeId(ontology, iri);
    if (!nodes.has(id)) nodes.set(id, { id, side, role: "context", iri, ontology, group });
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
        nodes.set(id, { id, side: item.side, role: "literal", ontology, text: text.length > 60 ? `${text.slice(0, 57)}…` : text, group: "down" });
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

export function EvidenceGraph({
  source,
  target,
  bundle,
  selected,
  onSelect,
  canExpand = true,
  onViewChange,
}: {
  source: EntityRef;
  target: EntityRef;
  bundle: EvidenceBundle;
  selected: string | null;
  onSelect: (evidenceId: string | null) => void;
  canExpand?: boolean;
  onViewChange?: (action: "zoom" | "fit") => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cyRef = useRef<any>(null);
  const [expansions, setExpansions] = useState<{ nodes: GNode[]; edges: GEdge[] }[]>([]);
  const [focusNode, setFocusNode] = useState<GNode | null>(null);
  const [focusEdge, setFocusEdge] = useState<GEdge | null>(null);
  const [expandError, setExpandError] = useState<string | null>(null);
  const themeVersion = useThemeVersion();

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
      if (node.role === "focal") return `${node.side === "source" ? "SOURCE" : "TARGET"} ${source.kind === "class" ? "CLASS" : source.kind.toUpperCase()}\n${text}`;
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
      {
        selector: "edge[kind = 'bridge']",
        style: { "line-style": "double", width: 7, "line-color": cssVar("--bridge"), "target-arrow-shape": "none", color: cssVar("--bridge") },
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

  // Create once per pair; later label/theme changes update in place without moving nodes.
  useEffect(() => {
    let destroyed = false;
    (async () => {
      const cytoscape = (await import("cytoscape")).default;
      if (destroyed || !container.current) return;
      const cy = cytoscape({
        container: container.current,
        elements: elements() as never,
        style: styles() as never,
        layout: { name: "preset" },
        userZoomingEnabled: false,
        userPanningEnabled: true,
        boxSelectionEnabled: false,
        autoungrabify: true,
        minZoom: 0.35,
        maxZoom: 2.5,
      });
      cy.fit(undefined, 40);
      if (cy.zoom() > 1.15) {
        cy.zoom(1.15);
        cy.center();
      }
      cy.on("tap", "node", (event: { target: { id: () => string } }) => {
        const node = model.nodes.find((item) => item.id === event.target.id()) ?? null;
        setFocusNode(node);
        setFocusEdge(null);
      });
      cy.on("tap", "edge", (event: { target: { id: () => string } }) => {
        const edge = model.edges.find((item) => item.id === event.target.id()) ?? null;
        setFocusEdge(edge);
        setFocusNode(null);
        onSelect(edge?.evidenceId ?? null);
      });
      cyRef.current = cy;
    })();
    return () => {
      destroyed = true;
      cyRef.current?.destroy();
      cyRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source.iri, target.iri, bundle]);

  // Sync elements (labels, expansions) and theme without re-running the layout of existing nodes.
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
    cy.off("tap");
    cy.on("tap", "node", (event: { target: { id: () => string } }) => {
      setFocusNode(model.nodes.find((item) => item.id === event.target.id()) ?? null);
      setFocusEdge(null);
    });
    cy.on("tap", "edge", (event: { target: { id: () => string } }) => {
      const edge = model.edges.find((item) => item.id === event.target.id()) ?? null;
      setFocusEdge(edge);
      setFocusNode(null);
      onSelect(edge?.evidenceId ?? null);
    });
  }, [elements, styles, themeVersion, model, onSelect]);

  // Selection from the list highlights the matching edge.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().unselect();
    if (selected) cy.edges().filter((edge: { data: (key: string) => string }) => edge.data("id").startsWith(`e:${selected}:`)).select();
  }, [selected]);

  const zoom = (factor: number) => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.zoom({ level: Math.max(0.35, Math.min(2.5, cy.zoom() * factor)), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } });
    onViewChange?.("zoom");
  };
  const fit = () => {
    cyRef.current?.fit(undefined, 40);
    onViewChange?.("fit");
  };
  const reset = () => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().forEach((node: { id: () => string; position: (p: { x: number; y: number }) => void }) => {
      const position = positions[node.id()];
      if (position) node.position(position);
    });
    cy.fit(undefined, 40);
  };

  const expandParents = async (node: GNode) => {
    setExpandError(null);
    if (!node.iri) return;
    if (model.nodes.length >= MAX_NODES) {
      setExpandError(`The graph already shows ${MAX_NODES} nodes. Use the hierarchy browser for further navigation.`);
      return;
    }
    try {
      const page = await getJson<HierarchyPage>("/api/v1/hierarchy", {
        ontology_version_id: node.ontology,
        iri: node.iri,
        kind: "class",
        direction: "parents",
        limit: 20,
      });
      const nodes: GNode[] = [];
      const edges: GEdge[] = [];
      for (const edge of page.items) {
        const id = nodeId(node.ontology, edge.parent.iri);
        if (!model.nodes.some((item) => item.id === id)) nodes.push({ id, side: node.side, role: "context", iri: edge.parent.iri, ontology: node.ontology, group: "expanded", anchor: node.id });
        edges.push({ id: `h:${edge.id}`, source: node.id, target: id, label: "subclass of", kind: "asserted" });
      }
      if (!edges.length) {
        setExpandError("No asserted parent is recorded for this node in the loaded scope.");
        return;
      }
      setExpansions((list) => [...list, { nodes, edges }]);
    } catch {
      setExpandError("Parents could not be loaded.");
    }
  };

  const nodeCount = model.nodes.length;
  const edgeCount = model.edges.length;
  return (
    <div className="graph-wrap">
      <div className="graph-toolbar" role="toolbar" aria-label="Graph view controls">
        <button type="button" className="icon-btn" aria-label="Zoom in" onClick={() => zoom(1.25)}>
          <IconZoomIn />
        </button>
        <button type="button" className="icon-btn" aria-label="Zoom out" onClick={() => zoom(0.8)}>
          <IconZoomOut />
        </button>
        <button type="button" className="btn btn-sm" onClick={fit}>
          Fit
        </button>
        <button type="button" className="btn btn-sm" onClick={reset}>
          Reset view
        </button>
        <span className="graph-toolbar-spacer" />
        {canExpand && (
          <button type="button" className="btn btn-sm" disabled={!expansions.length} onClick={() => setExpansions((list) => list.slice(0, -1))}>
            Undo last expansion
          </button>
        )}
      </div>
      <div ref={container} className="graph-canvas" role="img" aria-label={`Evidence graph with ${nodeCount} nodes and ${edgeCount} edges. The evidence list contains the same items.`} />
      <div className="graph-legend" aria-hidden="true">
        <span>
          <svg width="36" height="10">
            <line x1="0" y1="5" x2="36" y2="5" stroke="currentColor" strokeWidth="2.5" />
          </svg>
          Ontology assertion
        </span>
        <span>
          <svg width="36" height="10">
            <line x1="0" y1="5" x2="36" y2="5" stroke="currentColor" strokeWidth="2.5" strokeDasharray="10 4 2 4" />
          </svg>
          Feature Exact used
        </span>
        <span>
          <svg width="36" height="12">
            <line x1="0" y1="3" x2="36" y2="3" stroke="var(--bridge)" strokeWidth="2" />
            <line x1="0" y1="9" x2="36" y2="9" stroke="var(--bridge)" strokeWidth="2" />
          </svg>
          Compared across ontologies
        </span>
        <span>Rounded = source side · square = target side</span>
      </div>
      <p className="meta">
        {nodeCount} nodes · {edgeCount} edges shown{base.omitted ? ` · ${base.omitted} facts have no graph form and appear only in the list` : " · nothing omitted"}
      </p>
      {(focusNode || focusEdge) && (
        <div className="graph-detail card-inset" aria-live="polite">
          {focusNode && (
            <>
              <h3>{labelFor(focusNode).replace(/^.*\n/, "")}</h3>
              {focusNode.iri && <p className="iri">{focusNode.iri}</p>}
              <p className="meta">{focusNode.side === "source" ? "Source ontology" : "Target ontology"} · {focusNode.role === "focal" ? "compared entity" : focusNode.role === "literal" ? "recorded literal" : "related entity"}</p>
              {canExpand && focusNode.iri && focusNode.role !== "literal" && (
                <button type="button" className="btn btn-sm" onClick={() => expandParents(focusNode)}>
                  Add its parents to the graph
                </button>
              )}
            </>
          )}
          {focusEdge && (
            <>
              <h3>{edgeLabel(focusEdge)}</h3>
              <p className="muted">
                {focusEdge.kind === "bridge"
                  ? "Exact compared features of this kind from both entities when scoring. This does not state that any two features are equivalent."
                  : focusEdge.kind === "feature"
                    ? "A feature Exact selected for this pair, projected from the asserted fact listed in the evidence list."
                    : "An asserted subclass axiom added to the graph by you."}
              </p>
            </>
          )}
          {expandError && <p className="note note-warn">{expandError}</p>}
        </div>
      )}
    </div>
  );
}
