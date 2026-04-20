"use client";

import React, { useEffect, useMemo, useRef } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";

import graphStyles from "@/app/hooks/graphStyles";
import { StudyEdge, StudyNode } from "@/app/hooks/types";


type StudyGraphProps = {
  nodes: StudyNode[];
  edges: StudyEdge[];
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  expandedNodeId: string | null;
  viewportKey: string;
  onNodeClick: (nodeId: string) => void;
  onEdgeClick: (edgeId: string) => void;
  onBackgroundClick: () => void;
};

const CHANNEL_ORDER = ["hierarchy", "similarity", "difference", "attribute", "other"] as const;

function primaryChannel(node: StudyNode): string {
  const counts = new Map<string, number>();
  for (const detail of node.details || []) {
    const channel = String(detail.channel || "").trim().toLowerCase();
    if (!channel) continue;
    counts.set(channel, (counts.get(channel) || 0) + 1);
  }
  if (!counts.size) return "other";
  return Array.from(counts.entries()).sort((left, right) => {
    if (right[1] !== left[1]) return right[1] - left[1];
    return left[0].localeCompare(right[0]);
  })[0][0];
}

function placeArc(
  positions: Record<string, { x: number; y: number }>,
  items: StudyNode[],
  centerX: number,
  centerY: number,
  startDegrees: number,
  endDegrees: number,
  baseRadius: number,
  ringGap: number,
  itemsPerRing: number,
) {
  items.forEach((item, index) => {
    const ring = Math.floor(index / itemsPerRing);
    const slot = index % itemsPerRing;
    const countInRing = Math.min(itemsPerRing, items.length - ring * itemsPerRing);
    const radius = baseRadius + ring * ringGap;
    const start = (startDegrees * Math.PI) / 180;
    const end = (endDegrees * Math.PI) / 180;
    const theta = (() => {
      if (countInRing === 1) return (start + end) / 2;
      const oddRingOffset = ring % 2 === 1 ? 0.5 : 0;
      const fraction = (slot + 0.5 + oddRingOffset) / countInRing;
      return start + (end - start) * Math.min(Math.max(fraction, 0.04), 0.96);
    })();
    positions[item.id] = {
      x: centerX + Math.cos(theta) * radius,
      y: centerY + Math.sin(theta) * radius,
    };
  });
}

function placeGroupedArc(
  positions: Record<string, { x: number; y: number }>,
  items: StudyNode[],
  centerX: number,
  centerY: number,
  startDegrees: number,
  endDegrees: number,
  baseRadius: number,
  ringGap: number,
  itemsPerRing: number,
  sectorGapDegrees: number,
) {
  const grouped = new Map<string, StudyNode[]>();
  CHANNEL_ORDER.forEach((channel) => grouped.set(channel, []));
  items.forEach((item) => {
    const channel = primaryChannel(item);
    const bucket = grouped.get(channel) || grouped.get("other");
    bucket?.push(item);
  });

  const activeGroups = CHANNEL_ORDER
    .map((channel) => ({
      channel,
      items: (grouped.get(channel) || []).sort((left, right) => left.label.localeCompare(right.label)),
    }))
    .filter((group) => group.items.length > 0);

  if (!activeGroups.length) return;

  if (activeGroups.length === 1) {
    placeArc(
      positions,
      activeGroups[0].items,
      centerX,
      centerY,
      startDegrees,
      endDegrees,
      baseRadius,
      ringGap,
      itemsPerRing,
    );
    return;
  }

  const totalSpan = endDegrees - startDegrees;
  const reservedGap = sectorGapDegrees * (activeGroups.length - 1);
  const usableSpan = Math.max(totalSpan - reservedGap, totalSpan * 0.76);
  const totalWeight = activeGroups.reduce((acc, group) => acc + Math.max(group.items.length, 1.75), 0);
  let cursor = startDegrees;

  activeGroups.forEach((group, index) => {
    const weight = Math.max(group.items.length, 1.75);
    const rawSpan = (usableSpan * weight) / totalWeight;
    const span = index === activeGroups.length - 1 ? endDegrees - cursor : rawSpan;
    placeArc(
      positions,
      group.items,
      centerX,
      centerY,
      cursor,
      cursor + span,
      baseRadius,
      ringGap,
      itemsPerRing,
    );
    cursor += span + sectorGapDegrees;
  });
}

function buildPositions(nodes: StudyNode[], expandedNodeId: string | null): Record<string, { x: number; y: number }> {
  const positions: Record<string, { x: number; y: number }> = {};
  const sourceNodes = nodes.filter((node) => node.type === "Source");
  const targetNodes = nodes.filter((node) => node.type === "Target");
  const sourceContext = nodes.filter((node) => node.type === "source-context");
  const targetContext = nodes.filter((node) => node.type === "target-context");
  const ontologyExtra = nodes
    .filter((node) => node.type === "ontology-extra")
    .sort((left, right) => left.label.localeCompare(right.label));

  const sourceAnchor = { x: 500, y: 760 };
  const targetAnchor = { x: 2480, y: 760 };

  if (sourceNodes[0]) positions[sourceNodes[0].id] = sourceAnchor;
  if (targetNodes[0]) positions[targetNodes[0].id] = targetAnchor;

  placeGroupedArc(positions, sourceContext, sourceAnchor.x, sourceAnchor.y, 48, 312, 390, 118, 4, 10);
  placeGroupedArc(positions, targetContext, targetAnchor.x, targetAnchor.y, -132, 132, 390, 118, 4, 10);

  const expandedNode = expandedNodeId ? nodes.find((node) => node.id === expandedNodeId) : null;
  const fallbackAnchor = expandedNodeId && positions[expandedNodeId]
    ? positions[expandedNodeId]
    : { x: 1490, y: 760 };

  const expansionSide =
    expandedNode?.ontology_side ??
    (expandedNode?.type === "Source" || expandedNode?.type === "source-context"
      ? "source"
      : expandedNode?.type === "Target" || expandedNode?.type === "target-context"
        ? "target"
        : null);

  if (expansionSide === "source") {
    placeArc(positions, ontologyExtra, fallbackAnchor.x, fallbackAnchor.y, 90, 270, 240, 96, 4);
  } else if (expansionSide === "target") {
    placeArc(positions, ontologyExtra, fallbackAnchor.x, fallbackAnchor.y, -90, 90, 240, 96, 4);
  } else {
    placeArc(positions, ontologyExtra, fallbackAnchor.x, fallbackAnchor.y, 0, 330, 240, 96, 5);
  }

  return positions;
}

export default function StudyGraph({
  nodes,
  edges,
  selectedNodeId,
  selectedEdgeId,
  expandedNodeId,
  viewportKey,
  onNodeClick,
  onEdgeClick,
  onBackgroundClick,
}: StudyGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const onNodeClickRef = useRef(onNodeClick);
  const onEdgeClickRef = useRef(onEdgeClick);
  const onBackgroundClickRef = useRef(onBackgroundClick);
  const selectedNodeIdRef = useRef<string | null>(selectedNodeId);
  const selectedEdgeIdRef = useRef<string | null>(selectedEdgeId);
  const viewportKeyRef = useRef<string | null>(null);
  const viewportStateRef = useRef<Record<string, { zoom: number; pan: { x: number; y: number } }>>({});

  useEffect(() => {
    onNodeClickRef.current = onNodeClick;
  }, [onNodeClick]);

  useEffect(() => {
    onEdgeClickRef.current = onEdgeClick;
  }, [onEdgeClick]);

  useEffect(() => {
    onBackgroundClickRef.current = onBackgroundClick;
  }, [onBackgroundClick]);

  useEffect(() => {
    selectedNodeIdRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    selectedEdgeIdRef.current = selectedEdgeId;
  }, [selectedEdgeId]);

  const elements = useMemo(() => {
    const positions = buildPositions(nodes, expandedNodeId);
    const nodeElements: ElementDefinition[] = nodes.map((node) => ({
      data: {
        ...node,
      },
      position: positions[node.id],
    }));
    const edgeElements: ElementDefinition[] = edges.map((edge) => ({
      classes: edge.bridge ? "bridge-edge" : "context-edge",
      data: {
        ...edge,
      },
    }));
    return [...nodeElements, ...edgeElements];
  }, [nodes, edges, expandedNodeId]);

  useEffect(() => {
    if (!containerRef.current || cyRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements: [],
      style: graphStyles() as any,
      layout: { name: "preset", fit: false, padding: 160 },
      minZoom: 0.10,
      maxZoom: 2.4,
      wheelSensitivity: 0.18,
    });
    cyRef.current = cy;

    const clearInteractiveState = () => {
      cy.elements().removeClass("dimmed hovered-node hovered-edge");
    };

    const applySelectionState = (focusNodeId: string | null, focusEdgeId: string | null) => {
      cy.elements().removeClass("selected");
      if (focusNodeId) {
        const node = cy.getElementById(focusNodeId);
        if (node.nonempty()) node.addClass("selected");
      }
      if (focusEdgeId) {
        const edge = cy.getElementById(focusEdgeId);
        if (edge.nonempty()) edge.addClass("selected");
      }
    };

    cy.on("tap", "node", (event) => {
      const nodeId = String(event.target.id());
      applySelectionState(nodeId, null);
      onNodeClickRef.current(nodeId);
    });

    cy.on("tap", "edge", (event) => {
      const edgeId = String(event.target.id());
      applySelectionState(null, edgeId);
      onEdgeClickRef.current(edgeId);
    });

    cy.on("tap", (event) => {
      if (event.target !== cy) return;
      clearInteractiveState();
      applySelectionState(null, null);
      onBackgroundClickRef.current();
    });

    cy.on("mouseover", "node", (event) => {
      const node = event.target;
      clearInteractiveState();
      cy.elements().addClass("dimmed");
      node.removeClass("dimmed").addClass("hovered-node");
      node.connectedEdges().removeClass("dimmed").addClass("hovered-edge");
      node.connectedEdges().connectedNodes().removeClass("dimmed").addClass("hovered-node");
    });

    cy.on("mouseout", "node", () => {
      clearInteractiveState();
      applySelectionState(selectedNodeIdRef.current, selectedEdgeIdRef.current);
    });

    const persistViewport = () => {
      const key = viewportKeyRef.current;
      if (!key) return;
      viewportStateRef.current[key] = {
        zoom: cy.zoom(),
        pan: cy.pan(),
      };
    };

    cy.on("zoom pan dragfree", persistViewport);

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const sameViewport = viewportKeyRef.current === viewportKey;
    const savedZoom = cy.zoom();
    const savedPan = cy.pan();
    const savedForTarget = viewportStateRef.current[viewportKey];

    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: "preset", fit: false, padding: 160 }).run();

    if (sameViewport && Number.isFinite(savedZoom)) {
      cy.zoom(savedZoom);
      cy.pan(savedPan);
    } else if (savedForTarget && Number.isFinite(savedForTarget.zoom)) {
      cy.zoom(savedForTarget.zoom);
      cy.pan(savedForTarget.pan);
    } else {
      try {
        cy.fit(undefined, 210);
        cy.zoom(cy.zoom() * 1.10);
      } catch {}
    }
    viewportKeyRef.current = viewportKey;
  }, [elements, viewportKey]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass("selected");
    if (selectedNodeId) {
      const node = cy.getElementById(selectedNodeId);
      if (node.nonempty()) node.addClass("selected");
    }
    if (selectedEdgeId) {
      const edge = cy.getElementById(selectedEdgeId);
      if (edge.nonempty()) edge.addClass("selected");
    }
  }, [selectedNodeId, selectedEdgeId]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
