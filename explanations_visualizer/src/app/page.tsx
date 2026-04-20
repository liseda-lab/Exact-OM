"use client";

import React, { useEffect, useMemo, useState } from "react";

import StudyGraph from "@/app/components/study/StudyGraph";
import {
  EDGE_COLORS,
  EDGE_TYPE_LABELS,
  NODE_COLORS,
  NODE_TYPE_LABELS,
} from "@/app/hooks/graphStyles";
import {
  EdgeType,
  ExpandNodeResponse,
  NodeInfoResponse,
  NodeType,
  SourceBundle,
  StudyEdge,
  TargetBundle,
} from "@/app/hooks/types";


const NODE_TYPE_OPTIONS: NodeType[] = [
  "Source",
  "Target",
  "source-context",
  "target-context",
];

const EDGE_TYPE_OPTIONS: EdgeType[] = [
  "hierarchy",
  "similarity",
  "difference",
  "attribute",
  "bridge-support",
  "bridge-contrast",
];

const LEVEL_OPTIONS = [
  { value: 1, label: "Explanation only" },
  { value: 2, label: "Explanation + core links" },
  { value: 3, label: "Explanation + main support links" },
  { value: 4, label: "All explanation links" },
];

function createDefaultFilter<T extends string>(values: T[]): Record<T, boolean> {
  return values.reduce((acc, value) => {
    acc[value] = true;
    return acc;
  }, {} as Record<T, boolean>);
}

function formatScore(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "n/a";
  return value.toFixed(2);
}

function extractDefinitionTexts(info: NodeInfoResponse | null): string[] {
  if (!info) return [];
  const ontologyDefinitions = (info.ontology?.definitions || []).filter(Boolean);
  if (ontologyDefinitions.length) return ontologyDefinitions;
  const attrs = info.explanation?.attributes || [];
  return attrs
    .filter((item) => {
      const property = String(item.property || "").toLowerCase().replace(/[_-]/g, " ");
      const text = String(item.value || item.text || "").trim().toLowerCase();
      return (
        property === "definition" ||
        property === "description" ||
        text.startsWith("definition:") ||
        text.startsWith("description:")
      );
    })
    .map((item) => {
      const text = String(item.value || item.text || "").trim();
      if (text.includes(":")) {
        return text.split(":").slice(1).join(":").trim() || text;
      }
      return text;
    })
    .filter(Boolean);
}

function filterGraph(
  target: TargetBundle | null,
  level: number,
  showOntologyExtra: boolean,
  nodeFilters: Record<NodeType, boolean>,
  edgeFilters: Record<EdgeType, boolean>,
  expansion: ExpandNodeResponse | null,
) {
  if (!target) return { nodes: [], edges: [] };
  const baseNodes = [...target.graph.nodes];
  const baseEdges = target.graph.edges.filter((edge) => {
    if (edge.origin === "ontology-extra") return false;
    if ((edge.level ?? 1) > level) return false;
    return edgeFilters[edge.type];
  });

  const mergedNodes = [...baseNodes];
  const mergedEdges = [...baseEdges];

  if (showOntologyExtra && expansion && expansion.expandable) {
    expansion.nodes.forEach((node) => {
      mergedNodes.push(node);
    });
    expansion.edges.forEach((edge) => {
      if (edgeFilters[edge.type]) mergedEdges.push(edge);
    });
  }

  const visibleNodeIds = new Set(
    mergedNodes
      .filter((node) => nodeFilters[node.type])
      .map((node) => node.id),
  );
  const visibleNodeLookup = new Map(
    mergedNodes
      .filter((node) => visibleNodeIds.has(node.id))
      .map((node) => [node.id, node] as const),
  );

  const candidateEdges = mergedEdges.filter(
    (edge) =>
      edgeFilters[edge.type] &&
      visibleNodeIds.has(edge.source) &&
      visibleNodeIds.has(edge.target),
  );

  const nonBridgeIncidentNodeIds = new Set<string>();
  candidateEdges
    .filter((edge) => !edge.bridge)
    .forEach((edge) => {
      nonBridgeIncidentNodeIds.add(edge.source);
      nonBridgeIncidentNodeIds.add(edge.target);
    });

  const bridgeSupportedEdges = candidateEdges.filter((edge) => {
    if (!edge.bridge) return true;
    const sourceNode = visibleNodeLookup.get(edge.source);
    const targetNode = visibleNodeLookup.get(edge.target);
    const sourceAnchored =
      sourceNode?.type === "Source" ||
      sourceNode?.type === "Target" ||
      nonBridgeIncidentNodeIds.has(edge.source);
    const targetAnchored =
      targetNode?.type === "Source" ||
      targetNode?.type === "Target" ||
      nonBridgeIncidentNodeIds.has(edge.target);
    return sourceAnchored && targetAnchored;
  });

  const incidentNodeIds = new Set<string>();
  bridgeSupportedEdges.forEach((edge) => {
    incidentNodeIds.add(edge.source);
    incidentNodeIds.add(edge.target);
  });

  const prunedNodes = mergedNodes.filter((node) => {
    if (!visibleNodeIds.has(node.id)) return false;
    if (node.type === "Source" || node.type === "Target") return true;
    return incidentNodeIds.has(node.id);
  });

  const prunedNodeIds = new Set(prunedNodes.map((node) => node.id));

  return {
    nodes: prunedNodes,
    edges: bridgeSupportedEdges.filter(
      (edge) =>
        prunedNodeIds.has(edge.source) &&
        prunedNodeIds.has(edge.target),
    ),
  };
}

function LegendNode({
  color,
  label,
}: {
  color: string;
  label: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.55rem", whiteSpace: "nowrap" }}>
      <span
        style={{
          width: "0.95rem",
          height: "0.95rem",
          borderRadius: "0.35rem",
          background: color,
          border: "1px solid rgba(61,79,95,0.22)",
          flexShrink: 0,
        }}
      />
      <span style={{ color: "#516570" }}>{label}</span>
    </div>
  );
}

function LegendEdge({
  color,
  dashed,
  label,
}: {
  color: string;
  dashed?: boolean;
  label: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.55rem", whiteSpace: "nowrap" }}>
      <span
        style={{
          width: "1.45rem",
          borderTop: `3px ${dashed ? "dashed" : "solid"} ${color}`,
          flexShrink: 0,
        }}
      />
      <span style={{ color: "#516570" }}>{label}</span>
    </div>
  );
}

function MetricCard({
  title,
  label,
  description,
}: {
  title: string;
  label?: string;
  description?: string;
}) {
  if (!label && !description) return null;
  return (
    <div
      style={{
        borderRadius: "16px",
        border: "1px solid rgba(74,96,109,0.12)",
        background: "#fafcfd",
        padding: "0.82rem 0.9rem",
      }}
    >
      <div
        style={{
          fontSize: "0.72rem",
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: "#7a8995",
          marginBottom: "0.35rem",
        }}
      >
        {title}
      </div>
      {label ? <div style={{ fontWeight: 800, color: "#304452" }}>{label}</div> : null}
      {description ? (
        <div style={{ marginTop: "0.35rem", color: "#566874", lineHeight: 1.45 }}>{description}</div>
      ) : null}
    </div>
  );
}

function InspectorCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        borderRadius: "16px",
        border: "1px solid rgba(74,96,109,0.12)",
        background: "#fafcfd",
        padding: "0.8rem 0.88rem",
        minWidth: 0,
      }}
    >
      <div style={{ fontWeight: 700, color: "#304452", marginBottom: "0.32rem" }}>{title}</div>
      {children}
    </div>
  );
}

function DetailPill({
  label,
  value,
}: {
  label: string;
  value?: string | null;
}) {
  if (!value) return null;
  return (
    <div
      style={{
        borderRadius: "999px",
        border: "1px solid rgba(74,96,109,0.14)",
        background: "#f7fafc",
        padding: "0.38rem 0.6rem",
      }}
    >
      <span style={{ color: "#73828e", fontSize: "0.78rem", marginRight: "0.35rem" }}>{label}</span>
      <span style={{ color: "#304452", fontWeight: 700 }}>{value}</span>
    </div>
  );
}

function ScrollPanel({
  children,
  minWidth,
}: {
  children: React.ReactNode;
  minWidth?: number;
}) {
  return (
    <section
      style={{
        position: "relative",
        minWidth,
        minHeight: 0,
        borderRadius: "24px",
        border: "1px solid rgba(74,96,109,0.12)",
        background: "rgba(255,255,255,0.94)",
        boxShadow: "0 18px 42px rgba(74,96,109,0.06)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: "100%",
          overflowY: "auto",
          padding: "1rem 1rem 3rem 1rem",
          boxSizing: "border-box",
        }}
      >
        {children}
      </div>
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: 0,
          height: "4.2rem",
          background: "linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.96) 72%, rgba(255,255,255,1) 100%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          left: "50%",
          bottom: "0.62rem",
          transform: "translateX(-50%)",
          fontSize: "0.76rem",
          letterSpacing: "0.04em",
          color: "#7b8a95",
          pointerEvents: "none",
          whiteSpace: "nowrap",
        }}
      >
        Scroll for more
      </div>
    </section>
  );
}

function formatNodeDetail(detail: {
  triple?: string;
  text?: string;
  property?: string;
}): string {
  const triple = String(detail.triple || "").trim();
  if (triple) return triple;
  const property = String(detail.property || "").trim();
  const text = String(detail.text || "").trim();
  if (property && text) return `${property}: ${text}`;
  if (text) return text;
  if (property) return property;
  return "";
}

function SmallLogo() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.65rem" }}>
      <div
        style={{
          width: "1.9rem",
          height: "1.9rem",
          borderRadius: "0.7rem",
          background: "linear-gradient(135deg, #4d6984 0%, #7fa2c0 100%)",
          color: "#ffffff",
          display: "grid",
          placeItems: "center",
          fontWeight: 800,
          fontSize: "0.82rem",
          boxShadow: "0 8px 18px rgba(77,105,132,0.18)",
        }}
      >
        E
      </div>
      <div>
        <div style={{ fontWeight: 800, color: "#304452", lineHeight: 1 }}>Exact-OM</div>
        <div style={{ color: "#6f7f8b", fontSize: "0.88rem", marginTop: "0.15rem" }}>
          Explanation visualization
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const ONTOLOGY_EXPANSION_ENABLED = false;
  const [sourceId, setSourceId] = useState<string>("");
  const [bundle, setBundle] = useState<SourceBundle | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [selectedTargetId, setSelectedTargetId] = useState<string>("");
  const [selectedLevel, setSelectedLevel] = useState<number>(2);
  const [nodeFilters, setNodeFilters] = useState<Record<NodeType, boolean>>(
    createDefaultFilter(NODE_TYPE_OPTIONS),
  );
  const [edgeFilters, setEdgeFilters] = useState<Record<EdgeType, boolean>>(
    createDefaultFilter(EDGE_TYPE_OPTIONS),
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [nodeInfoCache, setNodeInfoCache] = useState<Record<string, NodeInfoResponse>>({});
  const [expandedNodeId, setExpandedNodeId] = useState<string | null>(null);
  const [expansion, setExpansion] = useState<ExpandNodeResponse | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setSourceId(params.get("source") ?? "");
  }, []);

  useEffect(() => {
    if (!sourceId) {
      setLoading(false);
      setError("Missing source query parameter. Use ?source=<exact_source_iri>.");
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(`/api/study/source?source=${encodeURIComponent(sourceId)}`)
      .then(async (response) => {
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || "Failed to load study source.");
        }
        return response.json();
      })
      .then((payload: SourceBundle) => {
        if (cancelled) return;
        setBundle(payload);
        const defaultTarget =
          payload.targets.find((target) => target.rank === payload.default_target_rank) ||
          payload.targets[0];
        setSelectedTargetId(defaultTarget?.target_id ?? "");
        setSelectedNodeId(null);
        setSelectedEdgeId(null);
        setExpandedNodeId(null);
        setExpansion(null);
        setNodeInfoCache({});
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setBundle(null);
        setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sourceId]);

  const selectedTarget = useMemo(() => {
    if (!bundle) return null;
    return bundle.targets.find((target) => target.target_id === selectedTargetId) || bundle.targets[0] || null;
  }, [bundle, selectedTargetId]);

  useEffect(() => {
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setExpandedNodeId(null);
    setExpansion(null);
  }, [selectedTargetId]);

  const cacheKey = (targetId: string, nodeId: string) => `${targetId}::${nodeId}`;

  const ensureNodeInfo = async (nodeId: string): Promise<NodeInfoResponse | null> => {
    if (!bundle || !selectedTarget) return null;
    const key = cacheKey(selectedTarget.target_id, nodeId);
    if (nodeInfoCache[key]) return nodeInfoCache[key];
    const response = await fetch(
      `/api/study/node-info?source=${encodeURIComponent(bundle.source_id)}&target=${encodeURIComponent(
        selectedTarget.target_id,
      )}&node_id=${encodeURIComponent(nodeId)}`,
    );
    if (!response.ok) return null;
    const payload = (await response.json()) as NodeInfoResponse;
    setNodeInfoCache((prev) => ({ ...prev, [key]: payload }));
    return payload;
  };

  const handleNodeClick = async (nodeId: string) => {
    if (!bundle || !selectedTarget) return;
    setSelectedEdgeId(null);
    setSelectedNodeId(nodeId);
    const info = await ensureNodeInfo(nodeId);
    if (!ONTOLOGY_EXPANSION_ENABLED || !info?.expandable) return;
    if (expandedNodeId === nodeId) {
      setExpandedNodeId(null);
      setExpansion(null);
      return;
    }
    const response = await fetch(
      `/api/study/expand-node?source=${encodeURIComponent(bundle.source_id)}&target=${encodeURIComponent(
        selectedTarget.target_id,
      )}&node_id=${encodeURIComponent(nodeId)}`,
    );
    if (!response.ok) return;
    const payload = (await response.json()) as ExpandNodeResponse;
    setExpandedNodeId(nodeId);
    setExpansion(payload);
  };

  const handleEdgeClick = (edgeId: string) => {
    setSelectedNodeId(null);
    setSelectedEdgeId(edgeId);
  };

  const handleBackgroundClick = () => {
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
  };

  const visibleGraph = useMemo(
    () =>
      filterGraph(
        selectedTarget,
        selectedLevel,
        ONTOLOGY_EXPANSION_ENABLED,
        nodeFilters,
        edgeFilters,
        expansion,
      ),
    [selectedTarget, selectedLevel, nodeFilters, edgeFilters, expansion],
  );

  const selectedNodeInfo =
    selectedTarget && selectedNodeId
      ? nodeInfoCache[cacheKey(selectedTarget.target_id, selectedNodeId)]
      : null;
  const selectedNodeDefinitions = extractDefinitionTexts(selectedNodeInfo);
  const selectedNodeDetailItems = Array.from(
    new Set(
      (selectedNodeInfo?.explanation.details || [])
        .map((detail) => formatNodeDetail(detail))
        .filter(Boolean),
    ),
  );
  const selectedNodeSynonyms = Array.from(new Set((selectedNodeInfo?.ontology.synonyms || []).filter(Boolean)));
  const selectedEdge: StudyEdge | null =
    selectedEdgeId ? visibleGraph.edges.find((edge) => edge.id === selectedEdgeId) || null : null;

  return (
    <main
      style={{
        height: "100vh",
        background: "linear-gradient(135deg, #f6f2ec 0%, #f8fbfd 48%, #eef4f8 100%)",
        color: "#273945",
        padding: "1rem",
        boxSizing: "border-box",
        overflow: "hidden",
        fontFamily: "\"Avenir Next\", \"Segoe UI\", sans-serif",
        display: "grid",
        gridTemplateRows: "auto minmax(0, 1fr) auto",
        gap: "1rem",
      }}
    >
      <header
        style={{
          borderRadius: "22px",
          border: "1px solid rgba(74,96,109,0.12)",
          background: "rgba(255,255,255,0.92)",
          boxShadow: "0 14px 32px rgba(74,96,109,0.06)",
          padding: "0.9rem 1rem",
          display: "flex",
          justifyContent: "space-between",
          gap: "1rem",
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        <SmallLogo />
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.95rem 1.25rem", alignItems: "center" }}>
          <LegendNode color={NODE_COLORS.Source} label={NODE_TYPE_LABELS.Source} />
          <LegendNode color={NODE_COLORS.Target} label={NODE_TYPE_LABELS.Target} />
          <LegendNode color={NODE_COLORS["source-context"]} label={NODE_TYPE_LABELS["source-context"]} />
          <LegendNode color={NODE_COLORS["target-context"]} label={NODE_TYPE_LABELS["target-context"]} />
          <LegendEdge color={EDGE_COLORS.hierarchy} label={EDGE_TYPE_LABELS.hierarchy} />
          <LegendEdge color={EDGE_COLORS.similarity} label={EDGE_TYPE_LABELS.similarity} />
          <LegendEdge color={EDGE_COLORS.difference} label={EDGE_TYPE_LABELS.difference} />
          <LegendEdge color={EDGE_COLORS.attribute} label={EDGE_TYPE_LABELS.attribute} />
          <LegendEdge color={EDGE_COLORS["bridge-support"]} dashed label={EDGE_TYPE_LABELS["bridge-support"]} />
          <LegendEdge color={EDGE_COLORS["bridge-contrast"]} dashed label={EDGE_TYPE_LABELS["bridge-contrast"]} />
        </div>
      </header>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "340px minmax(0, 1fr) 360px",
          gap: "1rem",
          minHeight: 0,
        }}
      >
        <ScrollPanel minWidth={340}>
          <div style={{ fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#70808c" }}>
            Selected candidate
          </div>
          {selectedTarget ? (
            <>
              <div
                style={{
                  marginTop: "0.7rem",
                  borderRadius: "18px",
                  border: "1px solid rgba(74,96,109,0.12)",
                  background: "#fafcfd",
                  padding: "0.95rem 1rem",
                }}
              >
                <div style={{ fontWeight: 800, color: "#304452", lineHeight: 1.3 }}>
                  {selectedTarget.target_label}
                </div>
                <div style={{ marginTop: "0.45rem", color: "#61727d", lineHeight: 1.45 }}>
                  Rank #{selectedTarget.rank} • confidence {formatScore(selectedTarget.score)}
                </div>
                <div style={{ marginTop: "0.75rem", display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                  <DetailPill label="Candidate" value={selectedTarget.ground_truth ? "Ground truth" : "Alternative"} />
                  <DetailPill label="LLM" value={selectedTarget.llm.decision || "not used"} />
                </div>
              </div>

              <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>
                <MetricCard
                  title="Decision basis"
                  label={selectedTarget.metrics.decision_basis?.label}
                  description={selectedTarget.metrics.decision_basis?.description}
                />
                <MetricCard
                  title="Evidence strength"
                  label={selectedTarget.metrics.evidence_strength?.label}
                  description={selectedTarget.metrics.evidence_strength?.description}
                />
                <MetricCard
                  title="Evidence agreement"
                  label={selectedTarget.metrics.evidence_agreement?.label}
                  description={selectedTarget.metrics.evidence_agreement?.description}
                />
              </div>

              <div
                style={{
                  marginTop: "1rem",
                  borderRadius: "16px",
                  border: "1px solid rgba(74,96,109,0.12)",
                  background: "#fafcfd",
                  padding: "0.88rem 0.95rem",
                }}
              >
                <div style={{ fontWeight: 700, color: "#304452", marginBottom: "0.45rem" }}>Rationale</div>
                <div style={{ color: "#526570", lineHeight: 1.58 }}>
                  {selectedTarget.llm.rationale || "No rationale available for this candidate."}
                </div>
              </div>
            </>
          ) : (
            <div style={{ marginTop: "0.8rem", color: "#61727d" }}>No candidate selected.</div>
          )}
        </ScrollPanel>

        <section
          style={{
            position: "relative",
            borderRadius: "24px",
            overflow: "hidden",
            border: "1px solid rgba(74,96,109,0.12)",
            background: "rgba(255,255,255,0.9)",
            boxShadow: "0 18px 42px rgba(74,96,109,0.08)",
            minHeight: 0,
          }}
        >
          {loading ? (
            <div style={{ padding: "2rem" }}>Loading study case…</div>
          ) : error ? (
            <div style={{ padding: "2rem", color: "#8a5b4f" }}>{error}</div>
          ) : !bundle || !selectedTarget ? (
            <div style={{ padding: "2rem" }}>No study panel available.</div>
          ) : (
            <div style={{ position: "absolute", inset: 0, padding: "1rem" }}>
              <StudyGraph
                nodes={visibleGraph.nodes}
                edges={visibleGraph.edges}
                selectedNodeId={selectedNodeId}
                selectedEdgeId={selectedEdgeId}
                expandedNodeId={expandedNodeId}
                viewportKey={selectedTarget.target_id}
                onNodeClick={handleNodeClick}
                onEdgeClick={handleEdgeClick}
                onBackgroundClick={handleBackgroundClick}
              />
            </div>
          )}
        </section>

        <ScrollPanel minWidth={360}>
          <div style={{ marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#70808c" }}>
              Source summary
            </div>
            <div style={{ fontSize: "1.12rem", fontWeight: 800, marginTop: "0.35rem" }}>
              {bundle?.source_label || sourceId || "Unknown source"}
            </div>
            {bundle?.source_id ? (
              <div style={{ marginTop: "0.4rem", color: "#61727d", wordBreak: "break-word" }}>
                {bundle.source_id}
              </div>
            ) : null}
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <div style={{ fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#70808c" }}>
              Target selection
            </div>
            <div style={{ display: "grid", gap: "0.55rem", marginTop: "0.65rem" }}>
              {(bundle?.targets || []).map((target) => {
                const active = target.target_id === selectedTargetId;
                return (
                  <button
                    key={target.target_id}
                    type="button"
                    onClick={() => setSelectedTargetId(target.target_id)}
                    style={{
                      textAlign: "left",
                      padding: "0.78rem 0.88rem",
                      borderRadius: "16px",
                      border: active ? "2px solid #617e92" : "1px solid rgba(74,96,109,0.12)",
                      background: active ? "#eef5f8" : "#ffffff",
                      cursor: "pointer",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem" }}>
                      <div style={{ fontWeight: 700, color: "#304452" }}>#{target.rank}</div>
                      <div style={{ color: "#4d6984", fontWeight: 700 }}>{formatScore(target.score)}</div>
                    </div>
                    <div style={{ marginTop: "0.35rem", fontWeight: 700, color: "#304452" }}>{target.target_label}</div>
                    <div style={{ marginTop: "0.3rem", color: "#6d7b86", fontSize: "0.92rem" }}>
                      {target.ground_truth ? "Ground truth candidate" : "Candidate"}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div style={{ fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#70808c" }}>
              Display controls
            </div>

            <label style={{ display: "block", marginTop: "0.7rem", fontWeight: 700 }}>Explanation granularity</label>
            <select
              value={selectedLevel}
              onChange={(event) => setSelectedLevel(Number(event.target.value))}
              style={{
                width: "100%",
                marginTop: "0.4rem",
                padding: "0.65rem 0.75rem",
                borderRadius: "12px",
                border: "1px solid rgba(74,96,109,0.18)",
                background: "#ffffff",
                color: "#304452",
              }}
            >
              {LEVEL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            <div style={{ marginTop: "0.75rem", color: "#687985", lineHeight: 1.45, fontSize: "0.92rem" }}>
              Ontology expansion is temporarily disabled in this viewer revision.
            </div>

            <details open style={{ marginTop: "1rem" }}>
              <summary style={{ fontWeight: 700, cursor: "pointer" }}>Filters</summary>
              <div style={{ display: "grid", gap: "1rem", marginTop: "0.8rem" }}>
                <div>
                  <div style={{ fontWeight: 700, marginBottom: "0.45rem" }}>Node filters</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.35rem 0.6rem" }}>
                    {NODE_TYPE_OPTIONS.map((type) => (
                      <label key={type} style={{ display: "flex", gap: "0.45rem", alignItems: "center", fontSize: "0.94rem" }}>
                        <input
                          type="checkbox"
                          checked={nodeFilters[type]}
                          onChange={() => setNodeFilters((prev) => ({ ...prev, [type]: !prev[type] }))}
                        />
                        <span>{NODE_TYPE_LABELS[type]}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 700, marginBottom: "0.45rem" }}>Edge filters</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.35rem 0.6rem" }}>
                    {EDGE_TYPE_OPTIONS.map((type) => (
                      <label key={type} style={{ display: "flex", gap: "0.45rem", alignItems: "center", fontSize: "0.94rem" }}>
                        <input
                          type="checkbox"
                          checked={edgeFilters[type]}
                          onChange={() => setEdgeFilters((prev) => ({ ...prev, [type]: !prev[type] }))}
                        />
                        <span>{EDGE_TYPE_LABELS[type]}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </details>
          </div>
        </ScrollPanel>
      </div>

      <footer
        style={{
          borderRadius: "22px",
          border: "1px solid rgba(74,96,109,0.12)",
          background: "rgba(255,255,255,0.94)",
          boxShadow: "0 14px 32px rgba(74,96,109,0.06)",
          padding: "0.9rem 1rem",
          maxHeight: "22vh",
          minHeight: "126px",
          overflowY: "auto",
          flexShrink: 0,
        }}
      >
        {selectedNodeInfo ? (
          <>
            <div style={{ fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#70808c", marginBottom: "0.7rem" }}>
              Node inspector
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(260px, 320px) minmax(320px, 1.25fr) minmax(260px, 0.95fr)",
                gap: "0.7rem",
                alignItems: "start",
              }}
            >
              <InspectorCard title="Selected node">
                <div style={{ fontWeight: 800, color: "#304452" }}>{selectedNodeInfo.node.label}</div>
                <div style={{ marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                  <DetailPill label="Type" value={NODE_TYPE_LABELS[selectedNodeInfo.node.type]} />
                  <DetailPill label="Node kind" value={selectedNodeInfo.node.node_kind || "context"} />
                  <DetailPill label="Side" value={selectedNodeInfo.node.ontology_side || "n/a"} />
                  <DetailPill label="Expandable" value={selectedNodeInfo.expandable ? "Yes" : "No"} />
                </div>
              </InspectorCard>

              <InspectorCard title="Explanation details">
                {selectedNodeDetailItems.length ? (
                  <ul style={{ margin: 0, paddingLeft: "1rem", color: "#526570", lineHeight: 1.42 }}>
                    {selectedNodeDetailItems.slice(0, 4).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ color: "#6f7f8b", lineHeight: 1.45 }}>No explanation-local details available for this node.</div>
                )}
              </InspectorCard>

              <div style={{ display: "grid", gap: "0.7rem", minWidth: 0 }}>
                <InspectorCard title="Description">
                  {selectedNodeDefinitions.length ? (
                    <div
                      style={{
                        color: "#526570",
                        lineHeight: 1.42,
                        display: "-webkit-box",
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                      }}
                    >
                      {selectedNodeDefinitions[0]}
                    </div>
                  ) : (
                    <div style={{ color: "#6f7f8b" }}>No description available.</div>
                  )}
                </InspectorCard>
                <InspectorCard title="Synonyms">
                  {selectedNodeSynonyms.length ? (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                      {selectedNodeSynonyms.slice(0, 5).map((item) => (
                        <span
                          key={item}
                          style={{
                            borderRadius: "999px",
                            background: "#eef5f8",
                            color: "#425967",
                            padding: "0.28rem 0.55rem",
                            fontSize: "0.88rem",
                          }}
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div style={{ color: "#6f7f8b" }}>No synonyms available.</div>
                  )}
                </InspectorCard>
              </div>
            </div>
          </>
        ) : selectedEdge ? (
          <>
            <div style={{ fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#70808c", marginBottom: "0.7rem" }}>
              Edge inspector
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(260px, 1fr) minmax(260px, 1fr) minmax(260px, 0.9fr)",
                gap: "0.7rem",
                alignItems: "start",
              }}
            >
              <InspectorCard title="Source">
                <div style={{ color: "#304452", fontWeight: 700, lineHeight: 1.4 }}>
                  {visibleGraph.nodes.find((node) => node.id === selectedEdge.source)?.label || selectedEdge.source}
                </div>
              </InspectorCard>
              <InspectorCard title="Target">
                <div style={{ color: "#304452", fontWeight: 700, lineHeight: 1.4 }}>
                  {visibleGraph.nodes.find((node) => node.id === selectedEdge.target)?.label || selectedEdge.target}
                </div>
              </InspectorCard>
              <InspectorCard title="Edge details">
                <div style={{ fontWeight: 800, color: "#304452", lineHeight: 1.35 }}>{selectedEdge.label}</div>
                <div style={{ marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                  <DetailPill label="Type" value={EDGE_TYPE_LABELS[selectedEdge.type]} />
                  <DetailPill
                    label="Score"
                    value={
                      selectedEdge.score !== undefined && selectedEdge.score !== null && selectedEdge.score !== ""
                        ? String(selectedEdge.score)
                        : "n/a"
                    }
                  />
                  <DetailPill label="Level" value={selectedEdge.level_label || "Context edge"} />
                  <DetailPill label="Bridge" value={selectedEdge.bridge ? "Yes" : "No"} />
                </div>
              </InspectorCard>
            </div>
          </>
        ) : (
          <>
            <div style={{ fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#70808c", marginBottom: "0.7rem" }}>
              Inspector
            </div>
            <div style={{ color: "#61727d", lineHeight: 1.55 }}>
              Click a node or edge in the graph to inspect its label, type, score, and explanation details here.
            </div>
          </>
        )}
      </footer>
    </main>
  );
}
