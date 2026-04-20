export type NodeType =
  | "Source"
  | "Target"
  | "source-context"
  | "target-context"
  | "ontology-extra";

export type EdgeType =
  | "hierarchy"
  | "similarity"
  | "difference"
  | "attribute"
  | "bridge-support"
  | "bridge-contrast"
  | "ontology-extra";

export interface StudyNodeDetail {
  channel?: string;
  role?: string;
  triple?: string;
  property?: string;
  text?: string;
  score?: number;
}

export interface StudyNode {
  id: string;
  label: string;
  type: NodeType;
  origin: "explanation" | "ontology-extra";
  expandable: boolean;
  entity_iri?: string | null;
  ontology_side?: "source" | "target" | null;
  node_kind?: "endpoint" | "entity-context" | "literal-context";
  details?: StudyNodeDetail[];
}

export interface StudyEdge {
  id?: string;
  source: string;
  target: string;
  label: string;
  type: EdgeType;
  score?: number | string | null;
  bridge?: boolean;
  level?: number;
  level_label?: string;
  origin?: "explanation" | "ontology-extra";
}

export interface TargetMetric {
  label?: string;
  description?: string;
  value?: number;
  margin?: number | null;
  channels_present?: number;
  details?: Record<string, number>;
}

export interface TargetBundle {
  target_id: string;
  target_label: string;
  rank: number;
  ground_truth: number;
  score: number;
  metrics: Record<string, TargetMetric>;
  llm: {
    pair_brief: string;
    rationale: string;
    decision: string;
    p_llm: number;
  };
  graph: {
    nodes: StudyNode[];
    edges: StudyEdge[];
  };
  prediction?: {
    threshold_positive?: boolean;
    global_match?: boolean;
  };
}

export interface SourceBundle {
  source_id: string;
  source_label: string;
  targets: TargetBundle[];
  default_target_rank: number;
}

export interface NodeInfoResponse {
  node: Omit<StudyNode, "details">;
  explanation: {
    details?: StudyNodeDetail[];
    attributes?: Array<Record<string, unknown>>;
  };
  ontology: {
    labels?: string[];
    definitions?: string[];
    synonyms?: string[];
    annotations?: Array<{ property: string; value: string }>;
  };
  expandable: boolean;
}

export interface ExpandNodeResponse {
  clicked_node_id: string;
  expandable: boolean;
  nodes: StudyNode[];
  edges: StudyEdge[];
}
