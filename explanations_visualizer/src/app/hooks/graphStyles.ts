import { StudyMode } from "@/app/hooks/types";


export const NODE_COLORS: Record<string, string> = {
  Source: "#4d6984",
  Target: "#8b6a55",
  "source-context": "#d7e5ef",
  "target-context": "#f0e2d4",
  "ontology-extra": "#e6ebef",
};

export const EDGE_COLORS: Record<string, string> = {
  hierarchy: "#3f7fc0",
  similarity: "#449b69",
  difference: "#d27a45",
  attribute: "#836ac6",
  "bridge-support": "#2e5f8a",
  "bridge-contrast": "#bf5b68",
  "ontology-extra": "#8c9daa",
};

const DARK_NODE_COLORS: Record<string, string> = {
  Source: "#5f83a2",
  Target: "#b48768",
  "source-context": "#223b4f",
  "target-context": "#4a3428",
  "ontology-extra": "#293641",
};

const DARK_EDGE_COLORS: Record<string, string> = {
  hierarchy: "#71a7df",
  similarity: "#61c386",
  difference: "#e59a64",
  attribute: "#a995e5",
  "bridge-support": "#75a7d1",
  "bridge-contrast": "#e07d89",
  "ontology-extra": "#9fb1bf",
};

export const NODE_TYPE_LABELS: Record<string, string> = {
  Source: "Source",
  Target: "Target",
  "source-context": "Source context",
  "target-context": "Target context",
  "ontology-extra": "Additional ontology context",
};

export const EDGE_TYPE_LABELS: Record<string, string> = {
  hierarchy: "Hierarchy",
  similarity: "Similarity",
  difference: "Difference",
  attribute: "Attribute",
  "bridge-support": "Support bridge",
  "bridge-contrast": "Contrast bridge",
  "ontology-extra": "Ontology extra",
};

function wrapText(label: string): string {
  return String(label ?? "").trim();
}

type GraphStyleOptions = {
  mode?: StudyMode;
  viewportWidth?: number;
  viewportHeight?: number;
  fontScale?: number;
  darkMode?: boolean;
};

export default function graphStyles(options: GraphStyleOptions = {}): Array<Record<string, unknown>> {
  const darkMode = options.darkMode ?? false;
  const nodeColors = darkMode ? DARK_NODE_COLORS : NODE_COLORS;
  const edgeColors = darkMode ? DARK_EDGE_COLORS : EDGE_COLORS;
  const width = Math.max(options.viewportWidth ?? 1440, 640);
  const height = Math.max(options.viewportHeight ?? 860, 480);
  const fontScale = Math.max(0.84, Math.min(1.26, options.fontScale ?? 1));
  const minDim = Math.min(width, height);
  const baseScale = Math.max(0.78, Math.min(1.02, minDim / 900));
  const modeBoost = options.mode === "study" ? 0.04 : 0.02;
  const nodeScale = Math.max(0.78, Math.min(1.04, baseScale + modeBoost));
  const endpointScale = Math.max(0.82, Math.min(1.06, nodeScale + 0.04));
  const nodeWidth = Math.round(184 * nodeScale);
  const nodeHeight = Math.round(88 * nodeScale);
  const endpointWidth = Math.round(222 * endpointScale);
  const endpointHeight = Math.round(98 * endpointScale);
  const nodeFontSize = Math.round(14 * nodeScale * fontScale);
  const endpointFontSize = Math.round(15 * endpointScale * fontScale);
  const edgeFontSize = Math.round(11.5 * nodeScale * fontScale);
  const nodeTextWidth = Math.max(124, nodeWidth - Math.round(30 * nodeScale));
  const endpointTextWidth = Math.max(nodeTextWidth, endpointWidth - Math.round(32 * endpointScale));
  const edgeTextWidth = Math.round(158 * nodeScale);
  const minNodeZoomedFontSize = 6;
  const minEdgeZoomedFontSize = 5.5;
  const nodeTextColor = darkMode ? "#dbe8f1" : "#2f4452";
  const endpointTextColor = "#ffffff";
  const nodeBorderColor = darkMode ? "#5e7484" : "#a6b4bf";
  const edgeTextColor = darkMode ? "#d6e3ec" : "#4f616d";
  const edgeTextBackground = darkMode ? "#111a22" : "#ffffff";

  return [
    {
      selector: "node",
      style: {
        label: (ele: any) => wrapText(String(ele.data("label") ?? "")),
        shape: "round-rectangle",
        padding: Math.round(16 * nodeScale),
        width: nodeWidth,
        height: nodeHeight,
        "background-color": (ele: any) => nodeColors[String(ele.data("type"))] ?? (darkMode ? "#25313b" : "#e4e8ec"),
        color: nodeTextColor,
        "font-family": "\"Avenir Next\", \"Segoe UI\", sans-serif",
        "font-size": nodeFontSize,
        "font-weight": 500,
        "min-zoomed-font-size": minNodeZoomedFontSize,
        "text-wrap": "wrap",
        "text-max-width": nodeTextWidth,
        "border-width": 1.5,
        "border-color": nodeBorderColor,
        "text-valign": "center",
        "text-halign": "center",
      },
    },
    {
      selector: 'node[type = "Source"], node[type = "Target"]',
      style: {
        width: endpointWidth,
        height: endpointHeight,
        "font-size": endpointFontSize,
        "text-max-width": endpointTextWidth,
        color: endpointTextColor,
        "border-width": 2.5,
      },
    },
    {
      selector: 'node[type = "Source"]',
      style: {
        "border-color": darkMode ? "#8eb6d4" : "#36536e",
      },
    },
    {
      selector: 'node[type = "Target"]',
      style: {
        "border-color": darkMode ? "#d8a37f" : "#6f4e3b",
      },
    },
    {
      selector: "node[expandable = true]",
      style: {
        "border-width": 2.8,
      },
    },
    {
      selector: 'node[type = "ontology-extra"]',
      style: {
        shape: "round-hexagon",
        "border-style": "dashed",
        "border-color": darkMode ? "#8ea0ad" : "#8fa0ac",
        color: darkMode ? "#cddae3" : "#586873",
      },
    },
    {
      selector: "edge",
      style: {
        width: 3.2,
        "line-color": (ele: any) => edgeColors[String(ele.data("type"))] ?? (darkMode ? "#9db0bd" : "#8596a3"),
        "target-arrow-color": (ele: any) => edgeColors[String(ele.data("type"))] ?? (darkMode ? "#9db0bd" : "#8596a3"),
        "target-arrow-shape": "triangle",
        "arrow-scale": 1.05,
        "curve-style": "straight",
        "line-style": "solid",
        label: (ele: any) => {
          const label = String(ele.data("label") ?? "");
          const score = ele.data("score");
          if (score === null || score === undefined || score === "") return label;
          if (typeof score === "number") {
            return `${label}\n${score.toFixed(2)}`;
          }
          return `${label}\n${String(score)}`;
        },
        color: edgeTextColor,
        "font-family": "\"Avenir Next\", \"Segoe UI\", sans-serif",
        "font-size": edgeFontSize,
        "font-weight": 400,
        "min-zoomed-font-size": minEdgeZoomedFontSize,
        "text-wrap": "wrap",
        "text-max-width": edgeTextWidth,
        "text-background-color": edgeTextBackground,
        "text-background-opacity": darkMode ? 0.82 : 0.9,
        "text-background-padding": 3,
        "text-rotation": "autorotate",
        "text-margin-y": -8,
      },
    },
    {
      selector: "edge.bridge-edge",
      style: {
        "line-style": "dashed",
      },
    },
    {
      selector: 'edge[type = "ontology-extra"]',
      style: {
        width: 2.4,
        "line-style": "dashed",
        opacity: 0.82,
      },
    },
    {
      selector: ".dimmed",
      style: {
        opacity: 0.16,
      },
    },
    {
      selector: ".hovered-node",
      style: {
        opacity: 1,
        "border-width": 3.6,
        "border-color": darkMode ? "#b6d2e5" : "#3e5567",
      },
    },
    {
      selector: ".hovered-edge",
      style: {
        opacity: 1,
        width: 4.2,
      },
    },
    {
      selector: ".selected",
      style: {
        "border-width": 3.8,
        "border-color": darkMode ? "#f0a174" : "#b5614d",
      },
    },
    {
      selector: "edge.selected",
      style: {
        width: 4.8,
        opacity: 1,
        "line-color": darkMode ? "#d7e9f5" : "#293f54",
        "target-arrow-color": darkMode ? "#d7e9f5" : "#293f54",
      },
    },
  ];
}
