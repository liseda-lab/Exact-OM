import {
  EdgeType,
  ExpandNodeResponse,
  NodeInfoResponse,
  NodeType,
  StudyEdge,
  StudyMode,
  StudyNode,
  TargetBundle,
} from "@/app/hooks/types";


export const NODE_TYPE_OPTIONS: NodeType[] = [
  "Source",
  "Target",
  "source-context",
  "target-context",
];

export const EDGE_TYPE_OPTIONS: EdgeType[] = [
  "hierarchy",
  "similarity",
  "difference",
  "attribute",
  "bridge-support",
  "bridge-contrast",
];

export const LEVEL_OPTIONS = [
  {
    value: 1,
    label: "Direct evidence only",
    description: "Shows only the local evidence attached directly to the displayed concepts.",
  },
  {
    value: 2,
    label: "Direct evidence + strongest links",
    description: "Adds the main cross-links that most directly connect source and target evidence.",
  },
  {
    value: 3,
    label: "Direct evidence + support links",
    description: "Adds secondary supporting links that help explain why the candidate is plausible.",
  },
  {
    value: 4,
    label: "Full explanation network",
    description: "Shows every explanation link kept for this candidate, including weaker optional links.",
  },
];

export type GraphFilters<T extends string> = Record<T, boolean>;

export function createDefaultFilter<T extends string>(values: T[]): Record<T, boolean> {
  return values.reduce((acc, value) => {
    acc[value] = true;
    return acc;
  }, {} as Record<T, boolean>);
}

export function formatScore(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "n/a";
  return value.toFixed(2);
}

export function buildEndpointNodeId(side: "source" | "target", entityIri: string): string {
  return `${side}-endpoint::${entityIri}`;
}

export function extractDefinitionTexts(info: NodeInfoResponse | null): string[] {
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

export function formatNodeDetail(detail: {
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

function normalizedAttributeText(value: string | undefined | null): string {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[_-]/g, " ");
}

function isExactMatchLabel(value: string | undefined | null): boolean {
  const normalized = normalizedAttributeText(value);
  return normalized === "exactmatch" || normalized === "exact match";
}

function isExactMatchAttributeNode(node: StudyNode): boolean {
  if (node.type !== "source-context" && node.type !== "target-context") return false;
  return (node.details || []).some((detail) => {
    if (isExactMatchLabel(detail.property)) return true;
    const text = normalizedAttributeText(detail.text);
    return text.startsWith("exactmatch:") || text.startsWith("exact match:");
  });
}

export function filterGraph(
  target: TargetBundle | null,
  level: number,
  showOntologyExtra: boolean,
  nodeFilters: GraphFilters<NodeType>,
  edgeFilters: GraphFilters<EdgeType>,
  expansion: ExpandNodeResponse | null,
  mode: StudyMode,
) {
  if (!target) return { nodes: [], edges: [] };
  const hiddenStudyNodeIds =
    mode === "study"
      ? new Set(target.graph.nodes.filter((node) => isExactMatchAttributeNode(node)).map((node) => node.id))
      : new Set<string>();

  const baseNodes = target.graph.nodes.filter((node) => !hiddenStudyNodeIds.has(node.id));
  const baseEdges = target.graph.edges.filter((edge) => {
    if (edge.origin === "ontology-extra") return false;
    if ((edge.level ?? 1) > level) return false;
    if (!edgeFilters[edge.type]) return false;
    if (hiddenStudyNodeIds.has(edge.source) || hiddenStudyNodeIds.has(edge.target)) return false;
    if (mode === "study" && edge.type === "attribute" && isExactMatchLabel(edge.label)) return false;
    return true;
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
      .filter((node) => nodeFilters[node.type] && !hiddenStudyNodeIds.has(node.id))
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
