"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import cytoscape, { Core, ElementDefinition, NodeSingular } from "cytoscape";

import graphStyles from "@/app/hooks/graphStyles";
import { GraphViewportState, StudyEdge, StudyMode, StudyNode } from "@/app/hooks/types";


type StudyGraphProps = {
  nodes: StudyNode[];
  edges: StudyEdge[];
  mode: StudyMode;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  expandedNodeId: string | null;
  layoutResetKey: string;
  viewportStateKey: string;
  savedViewportState: GraphViewportState | null;
  allowInspection: boolean;
  graphExpanded: boolean;
  canToggleGraphExpanded: boolean;
  onToggleGraphExpanded?: () => void;
  onViewportChange?: (nextViewportState: GraphViewportState) => void;
  onNodeClick: (nodeId: string) => void;
  onEdgeClick: (edgeId: string) => void;
  onBackgroundClick: () => void;
};

type ViewportSize = {
  width: number;
  height: number;
};

type GraphLayoutMetrics = {
  width: number;
  height: number;
  sourceAnchor: { x: number; y: number };
  targetAnchor: { x: number; y: number };
  baseRadius: number;
  ringGap: number;
  itemsPerRing: number;
  sectorGapDegrees: number;
  expansionBaseRadius: number;
  expansionGap: number;
  expansionItemsPerRing: number;
  fitPadding: number;
  fitZoomMultiplier: number;
  radialStagger: number;
};

type GraphFitPadding = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

const CHANNEL_ORDER = ["hierarchy", "similarity", "difference", "attribute", "other"] as const;
const STUDY_MIN_READABLE_AUTO_ZOOM = 0.58;
const APP_MIN_READABLE_AUTO_ZOOM = 0.58;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

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

function buildLayoutMetrics(viewport: ViewportSize, mode: StudyMode): GraphLayoutMetrics {
  const width = Math.max(viewport.width || 1320, 640);
  const height = Math.max(viewport.height || 860, 420);
  const constrained = width < 760 || height < 700;
  const centerX = width / 2;
  const centerY = mode === "study" ? height * 0.43 : height * (constrained ? 0.47 : 0.48);
  const desiredSeparation = width * 0.5;
  const maxSeparation = Math.max(340, width - 140);
  const separation = clamp(desiredSeparation, 360, Math.min(1120, maxSeparation));
  const minDim = Math.min(width, height);
  const baseRadiusMultiplier = constrained ? 0.3 : 0.34;
  const ringGapMultiplier = constrained ? 0.118 : 0.135;

  return {
    width,
    height,
    sourceAnchor: { x: centerX - separation / 2, y: centerY },
    targetAnchor: { x: centerX + separation / 2, y: centerY },
    baseRadius: clamp(minDim * baseRadiusMultiplier, constrained ? 200 : 230, constrained ? 390 : 470),
    ringGap: clamp(minDim * ringGapMultiplier, constrained ? 84 : 96, constrained ? 145 : 180),
    itemsPerRing: Math.round(
      clamp(Math.round(width / 360), 3, constrained ? 4 : 5),
    ),
    sectorGapDegrees: clamp(width / (constrained ? 150 : 132), constrained ? 6 : 8, constrained ? 12 : 16),
    expansionBaseRadius: clamp(minDim * 0.18, 155, 270),
    expansionGap: clamp(minDim * 0.075, 74, 112),
    expansionItemsPerRing: Math.round(clamp(Math.round(width / 340), 3, 6)),
    fitPadding: clamp(minDim * 0.055, 36, 82),
    fitZoomMultiplier: 0.93,
    radialStagger: clamp(minDim * (constrained ? 0.032 : 0.045), constrained ? 24 : 36, constrained ? 48 : 72),
  };
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
  radialStagger = 0,
) {
  items.forEach((item, index) => {
    const ring = Math.floor(index / itemsPerRing);
    const slot = index % itemsPerRing;
    const countInRing = Math.min(itemsPerRing, items.length - ring * itemsPerRing);
    const stagger = radialStagger ? ((slot + ring) % 2 === 0 ? 0 : radialStagger) : 0;
    const radius = baseRadius + ring * ringGap + stagger;
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
  radialStagger = 0,
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
      radialStagger,
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
      radialStagger,
    );
    cursor += span + sectorGapDegrees;
  });
}

function buildPositions(
  nodes: StudyNode[],
  expandedNodeId: string | null,
  layoutMetrics: GraphLayoutMetrics,
): Record<string, { x: number; y: number }> {
  const positions: Record<string, { x: number; y: number }> = {};
  const sourceNodes = nodes.filter((node) => node.type === "Source");
  const targetNodes = nodes.filter((node) => node.type === "Target");
  const sourceContext = nodes.filter((node) => node.type === "source-context");
  const targetContext = nodes.filter((node) => node.type === "target-context");
  const ontologyExtra = nodes
    .filter((node) => node.type === "ontology-extra")
    .sort((left, right) => left.label.localeCompare(right.label));

  if (sourceNodes[0]) positions[sourceNodes[0].id] = layoutMetrics.sourceAnchor;
  if (targetNodes[0]) positions[targetNodes[0].id] = layoutMetrics.targetAnchor;

  placeGroupedArc(
    positions,
    sourceContext,
    layoutMetrics.sourceAnchor.x,
    layoutMetrics.sourceAnchor.y,
    24,
    336,
    layoutMetrics.baseRadius,
    layoutMetrics.ringGap,
    layoutMetrics.itemsPerRing,
    layoutMetrics.sectorGapDegrees,
    layoutMetrics.radialStagger,
  );
  placeGroupedArc(
    positions,
    targetContext,
    layoutMetrics.targetAnchor.x,
    layoutMetrics.targetAnchor.y,
    -156,
    156,
    layoutMetrics.baseRadius,
    layoutMetrics.ringGap,
    layoutMetrics.itemsPerRing,
    layoutMetrics.sectorGapDegrees,
    layoutMetrics.radialStagger,
  );

  const expandedNode = expandedNodeId ? nodes.find((node) => node.id === expandedNodeId) : null;
  const fallbackAnchor = expandedNodeId && positions[expandedNodeId]
    ? positions[expandedNodeId]
    : { x: layoutMetrics.width / 2, y: layoutMetrics.height / 2 };

  const expansionSide =
    expandedNode?.ontology_side ??
    (expandedNode?.type === "Source" || expandedNode?.type === "source-context"
      ? "source"
      : expandedNode?.type === "Target" || expandedNode?.type === "target-context"
        ? "target"
        : null);

  if (expansionSide === "source") {
    placeArc(
      positions,
      ontologyExtra,
      fallbackAnchor.x,
      fallbackAnchor.y,
      90,
      270,
      layoutMetrics.expansionBaseRadius,
      layoutMetrics.expansionGap,
      layoutMetrics.expansionItemsPerRing,
    );
  } else if (expansionSide === "target") {
    placeArc(
      positions,
      ontologyExtra,
      fallbackAnchor.x,
      fallbackAnchor.y,
      -90,
      90,
      layoutMetrics.expansionBaseRadius,
      layoutMetrics.expansionGap,
      layoutMetrics.expansionItemsPerRing,
    );
  } else {
    placeArc(
      positions,
      ontologyExtra,
      fallbackAnchor.x,
      fallbackAnchor.y,
      0,
      330,
      layoutMetrics.expansionBaseRadius,
      layoutMetrics.expansionGap,
      layoutMetrics.expansionItemsPerRing,
    );
  }

  return positions;
}

function graphFitPadding(metrics: GraphLayoutMetrics, mode: StudyMode): GraphFitPadding {
  const constrained = metrics.width < 760 || metrics.height < 700;
  if (mode === "study") {
    if (constrained) {
      return {
        top: clamp(metrics.height * 0.08, 54, 86),
        right: clamp(metrics.width * 0.14, 72, 150),
        bottom: clamp(metrics.height * 0.22, 110, 180),
        left: clamp(metrics.width * 0.14, 72, 150),
      };
    }
    return {
      top: clamp(metrics.height * 0.1, 78, 116),
      right: clamp(metrics.width * 0.21, 132, 260),
      bottom: clamp(metrics.height * 0.3, 170, 290),
      left: clamp(metrics.width * 0.19, 132, 240),
    };
  }
  if (constrained) {
    return {
      top: clamp(metrics.height * 0.08, 46, 82),
      right: clamp(metrics.width * 0.13, 64, 132),
      bottom: clamp(metrics.height * 0.12, 58, 118),
      left: clamp(metrics.width * 0.13, 64, 132),
    };
  }
  return {
    top: clamp(metrics.height * 0.09, 62, 102),
    right: clamp(metrics.width * 0.16, 100, 190),
    bottom: clamp(metrics.height * 0.13, 86, 150),
    left: clamp(metrics.width * 0.16, 100, 190),
  };
}

function defaultFit(cy: Core, metrics: GraphLayoutMetrics, mode: StudyMode) {
  if (!cy.elements().length) return;
  try {
    const padding = graphFitPadding(metrics, mode);
    const boundingBox = cy.elements().boundingBox();
    const width = Math.max(1, boundingBox.w);
    const height = Math.max(1, boundingBox.h);
    const availableWidth = Math.max(80, cy.width() - padding.left - padding.right);
    const availableHeight = Math.max(80, cy.height() - padding.top - padding.bottom);
    const minReadableZoom =
      mode === "study"
        ? Math.min(cy.maxZoom(), Math.max(cy.minZoom(), STUDY_MIN_READABLE_AUTO_ZOOM))
        : Math.min(cy.maxZoom(), Math.max(cy.minZoom(), APP_MIN_READABLE_AUTO_ZOOM));
    const zoom = clamp(
      Math.min(availableWidth / width, availableHeight / height) * metrics.fitZoomMultiplier,
      minReadableZoom,
      cy.maxZoom(),
    );
    const pan = {
      x: padding.left + (availableWidth - width * zoom) / 2 - boundingBox.x1 * zoom,
      y: padding.top + (availableHeight - height * zoom) / 2 - boundingBox.y1 * zoom,
    };
    cy.zoom(zoom);
    cy.pan(pan);
  } catch {}
}

function applyViewportState(cy: Core, nextViewportState: GraphViewportState) {
  cy.zoom(clamp(nextViewportState.zoom, cy.minZoom(), cy.maxZoom()));
  cy.pan(nextViewportState.pan);
}

function zoomAboutCenter(cy: Core, factor: number) {
  const nextZoom = clamp(cy.zoom() * factor, cy.minZoom(), cy.maxZoom());
  cy.zoom({
    level: nextZoom,
    renderedPosition: {
      x: cy.width() / 2,
      y: cy.height() / 2,
    },
  });
}

function centerOnNode(cy: Core, node: NodeSingular) {
  const position = node.position();
  const zoom = cy.zoom();
  const nextPan = {
    x: cy.width() / 2 - position.x * zoom,
    y: cy.height() / 2 - position.y * zoom,
  };
  cy.animate(
    {
      pan: nextPan,
    },
    {
      duration: 240,
      easing: "ease-out-cubic",
    },
  );
}

function GraphControlButton({
  label,
  onClick,
  title,
  compact = false,
}: {
  label: React.ReactNode;
  onClick: () => void;
  title: string;
  compact?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      style={{
        border: "1px solid rgba(40, 58, 72, 0.14)",
        background: "rgba(255, 255, 255, 0.94)",
        color: "#233744",
        borderRadius: "14px",
        minWidth: compact ? "2rem" : "2.1rem",
        minHeight: compact ? "2rem" : "2.1rem",
        padding: compact ? "0.4rem 0.52rem" : "0.46rem 0.66rem",
        boxShadow: "0 10px 24px rgba(32, 49, 61, 0.12)",
        cursor: "pointer",
        fontWeight: 700,
        fontSize: "0.88rem",
        backdropFilter: "blur(10px)",
      }}
    >
      {label}
    </button>
  );
}

function GraphExpandIcon({ expanded }: { expanded: boolean }) {
  const lineStyle = {
    stroke: "currentColor",
    strokeWidth: 2.35,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    fill: "none",
  };

  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="1.08rem" height="1.08rem" focusable="false">
      {expanded ? (
        <>
          <path d="M8 4v5H3" style={lineStyle} />
          <path d="M4 5l5 5" style={lineStyle} />
          <path d="M16 4v5h5" style={lineStyle} />
          <path d="M20 5l-5 5" style={lineStyle} />
          <path d="M8 20v-5H3" style={lineStyle} />
          <path d="M4 19l5-5" style={lineStyle} />
          <path d="M16 20v-5h5" style={lineStyle} />
          <path d="M20 19l-5-5" style={lineStyle} />
        </>
      ) : (
        <>
          <path d="M4 9V4h5" style={lineStyle} />
          <path d="M4 4l6 6" style={lineStyle} />
          <path d="M20 9V4h-5" style={lineStyle} />
          <path d="M20 4l-6 6" style={lineStyle} />
          <path d="M4 15v5h5" style={lineStyle} />
          <path d="M4 20l6-6" style={lineStyle} />
          <path d="M20 15v5h-5" style={lineStyle} />
          <path d="M20 20l-6-6" style={lineStyle} />
        </>
      )}
    </svg>
  );
}

export default function StudyGraph({
  nodes,
  edges,
  mode,
  selectedNodeId,
  selectedEdgeId,
  expandedNodeId,
  layoutResetKey,
  viewportStateKey,
  savedViewportState,
  allowInspection,
  graphExpanded,
  canToggleGraphExpanded,
  onToggleGraphExpanded,
  onViewportChange,
  onNodeClick,
  onEdgeClick,
  onBackgroundClick,
}: StudyGraphProps) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const cyContainerRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  const layoutResetKeyRef = useRef<string | null>(null);
  const viewportStateKeyRef = useRef<string | null>(null);
  const viewportRef = useRef<ViewportSize>({ width: 0, height: 0 });
  const lastReportedViewportRef = useRef<GraphViewportState | null>(null);
  const onNodeClickRef = useRef(onNodeClick);
  const onEdgeClickRef = useRef(onEdgeClick);
  const onBackgroundClickRef = useRef(onBackgroundClick);
  const onViewportChangeRef = useRef(onViewportChange);
  const allowInspectionRef = useRef(allowInspection);
  const selectedNodeIdRef = useRef<string | null>(selectedNodeId);
  const selectedEdgeIdRef = useRef<string | null>(selectedEdgeId);
  const lastTapRef = useRef<{ nodeId: string | null; at: number }>({ nodeId: null, at: 0 });
  const [viewport, setViewport] = useState<ViewportSize>({ width: 0, height: 0 });
  const [graphFontScale, setGraphFontScale] = useState(1);

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
    onViewportChangeRef.current = onViewportChange;
  }, [onViewportChange]);

  useEffect(() => {
    allowInspectionRef.current = allowInspection;
  }, [allowInspection]);

  useEffect(() => {
    selectedNodeIdRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    selectedEdgeIdRef.current = selectedEdgeId;
  }, [selectedEdgeId]);

  useEffect(() => {
    if (!wrapperRef.current) return;
    const target = wrapperRef.current;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const width = Math.round(entry.contentRect.width);
      const height = Math.round(entry.contentRect.height);
      setViewport((prev) => {
        if (prev.width === width && prev.height === height) return prev;
        return { width, height };
      });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  const layoutMetrics = useMemo(
    () => buildLayoutMetrics(viewport, mode),
    [mode, viewport],
  );

  const elements = useMemo(() => {
    const positions = buildPositions(nodes, expandedNodeId, layoutMetrics);
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
  }, [nodes, edges, expandedNodeId, layoutMetrics]);

  useEffect(() => {
    if (!cyContainerRef.current || cyRef.current) return;
    const cy = cytoscape({
      container: cyContainerRef.current,
      elements: [],
      style: graphStyles({
        mode,
        viewportWidth: layoutMetrics.width,
        viewportHeight: layoutMetrics.height,
        fontScale: graphFontScale,
      }) as any,
      layout: { name: "preset", fit: false, padding: layoutMetrics.fitPadding },
      minZoom: 0.1,
      maxZoom: 2.4,
      wheelSensitivity: 0.18,
    });
    cyRef.current = cy;

    const clearInteractiveState = () => {
      cy.elements().removeClass("dimmed hovered-node hovered-edge");
    };

    const applySelectionState = (focusNodeId: string | null, focusEdgeId: string | null) => {
      cy.elements().removeClass("selected");
      if (!allowInspectionRef.current) return;
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
      const now = Date.now();
      const isDoubleTap = lastTapRef.current.nodeId === nodeId && now - lastTapRef.current.at < 280;
      lastTapRef.current = { nodeId, at: now };

      if (allowInspectionRef.current) {
        applySelectionState(nodeId, null);
        onNodeClickRef.current(nodeId);
      }
      if (isDoubleTap) {
        centerOnNode(cy, event.target);
      }
    });

    cy.on("tap", "edge", (event) => {
      if (!allowInspectionRef.current) return;
      const edgeId = String(event.target.id());
      applySelectionState(null, edgeId);
      onEdgeClickRef.current(edgeId);
    });

    cy.on("tap", (event) => {
      if (event.target !== cy) return;
      clearInteractiveState();
      if (!allowInspectionRef.current) return;
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

    const reportViewport = () => {
      if (!onViewportChangeRef.current) return;
      const nextViewportState = {
        zoom: Number(cy.zoom().toFixed(4)),
        pan: {
          x: Number(cy.pan().x.toFixed(2)),
          y: Number(cy.pan().y.toFixed(2)),
        },
      };
      const previousViewportState = lastReportedViewportRef.current;
      if (
        previousViewportState &&
        previousViewportState.zoom === nextViewportState.zoom &&
        previousViewportState.pan.x === nextViewportState.pan.x &&
        previousViewportState.pan.y === nextViewportState.pan.y
      ) {
        return;
      }
      lastReportedViewportRef.current = nextViewportState;
      onViewportChangeRef.current(nextViewportState);
    };

    cy.on("zoom", reportViewport);
    cy.on("pan", reportViewport);

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [layoutMetrics.fitPadding, layoutMetrics.height, layoutMetrics.width, mode]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.style().fromJson(
      graphStyles({
        mode,
        viewportWidth: layoutMetrics.width,
        viewportHeight: layoutMetrics.height,
        fontScale: graphFontScale,
      }) as any,
    ).update();
  }, [graphFontScale, layoutMetrics.height, layoutMetrics.width, mode]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || viewport.width === 0 || viewport.height === 0) return;

    const previousViewport = viewportRef.current;
    const sizeChanged =
      previousViewport.width === 0 ||
      previousViewport.height === 0 ||
      previousViewport.width !== viewport.width ||
      previousViewport.height !== viewport.height;
    const shouldReset = layoutResetKeyRef.current !== layoutResetKey || sizeChanged;
    const targetChanged = viewportStateKeyRef.current !== viewportStateKey;
    const savedZoom = cy.zoom();
    const savedPan = cy.pan();

    cy.resize();
    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: "preset", fit: false, padding: layoutMetrics.fitPadding }).run();

    if (targetChanged && savedViewportState && !sizeChanged) {
      applyViewportState(cy, savedViewportState);
    } else if (shouldReset) {
      defaultFit(cy, layoutMetrics, mode);
    } else if (Number.isFinite(savedZoom)) {
      cy.zoom(savedZoom);
      cy.pan(savedPan);
    }

    layoutResetKeyRef.current = layoutResetKey;
    viewportStateKeyRef.current = viewportStateKey;
    viewportRef.current = viewport;
  }, [elements, layoutMetrics, layoutResetKey, mode, savedViewportState, viewport, viewportStateKey]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.elements().removeClass("selected");
    cy.elements().removeClass("dimmed hovered-node hovered-edge");
    if (!allowInspection) return;
    if (selectedNodeId) {
      const node = cy.getElementById(selectedNodeId);
      if (node.nonempty()) node.addClass("selected");
    }
    if (selectedEdgeId) {
      const edge = cy.getElementById(selectedEdgeId);
      if (edge.nonempty()) edge.addClass("selected");
    }
  }, [allowInspection, selectedEdgeId, selectedNodeId]);

  return (
    <div ref={wrapperRef} style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={cyContainerRef} style={{ position: "absolute", inset: 0 }} />

      <div
        style={{
          position: "absolute",
          right: mode === "study" ? "1rem" : "0.9rem",
          top: "50%",
          transform: "translateY(-50%)",
          zIndex: mode === "study" ? 5 : 4,
          display: "flex",
          flexDirection: "column",
          gap: "0.5rem",
          alignItems: "flex-end",
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            display: "flex",
            gap: "0.45rem",
            pointerEvents: "auto",
            flexDirection: "column",
          }}
        >
          <GraphControlButton
            label="+"
            onClick={() => cyRef.current && zoomAboutCenter(cyRef.current, 1.12)}
            title="Zoom in"
            compact
          />
          <GraphControlButton
            label="-"
            onClick={() => cyRef.current && zoomAboutCenter(cyRef.current, 0.9)}
            title="Zoom out"
            compact
          />
          {canToggleGraphExpanded && onToggleGraphExpanded ? (
            <GraphControlButton
              label={<GraphExpandIcon expanded={graphExpanded} />}
              onClick={onToggleGraphExpanded}
              title={graphExpanded ? "Exit expanded graph view" : "Expand graph view"}
              compact
            />
          ) : null}
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "0.35rem",
            pointerEvents: "auto",
          }}
        >
          <GraphControlButton
            label="A+"
            onClick={() => setGraphFontScale((prev) => clamp(prev + 0.08, 0.84, 1.26))}
            title="Increase graph label size"
            compact
          />
          <GraphControlButton
            label="A-"
            onClick={() => setGraphFontScale((prev) => clamp(prev - 0.08, 0.84, 1.26))}
            title="Decrease graph label size"
            compact
          />
        </div>
      </div>
    </div>
  );
}
