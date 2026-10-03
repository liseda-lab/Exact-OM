// Maps workspace actions to study telemetry. Each event names the action that actually
// happened: opening a tab is not expanding a hierarchy branch, a copied IRI is not external
// use, and navigation has its own type. Actions without an accepted type are not sent.
// Pure module (type-only imports) so it runs under Node's test runner.

import type { WorkspaceAction } from "../lib/workspace/types";
import type { EventType } from "./types";

export interface MappedEvent {
  type: EventType;
  component?: string;
  element?: string;
}

export function workspaceEvent(action: WorkspaceAction): MappedEvent | null {
  switch (action.type) {
    case "tab_open":
      if (action.tab === "evidence") return { type: "table_open", component: "details" };
      if (action.tab === "graph") return { type: "graph_open", component: "details" };
      return { type: "tab_open", component: "details", element: action.tab };
    case "citation_open":
      return { type: "axiom_open", component: "citation", element: action.factId };
    case "axiom_open":
      return { type: "axiom_open", component: "workspace", element: action.factId };
    case "hierarchy_expand":
    case "hierarchy_collapse":
      return { type: action.type, component: `hierarchy_${action.side}`, element: action.iri };
    case "hierarchy_focus":
      return { type: "hierarchy_navigate", component: `hierarchy_${action.side}`, element: `${action.via}:${action.iri}` };
    case "search":
      return { type: "search_select", component: `hierarchy_${action.side}` };
    case "evidence_select":
      return { type: "evidence_open", component: action.from === "graph" ? "graph" : "evidence_list", element: action.evidenceId };
    case "graph_edge":
      return action.evidenceId ? { type: "evidence_open", component: "graph", element: action.evidenceId } : { type: "graph_edge_open", component: "graph", element: action.edgeKind };
    case "graph_zoom":
      return { type: "graph_zoom", component: "graph" };
    case "graph_fit":
      return { type: "graph_fit", component: "graph" };
    case "graph_reset":
      return { type: "graph_reset", component: "graph" };
    case "copy_iri":
      return { type: "copy_iri", component: "workspace", element: action.side };
    case "context_open":
      return { type: "definition_open", component: `context_${action.side}`, element: action.iri };
    default:
      return null;
  }
}
