"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

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
  GraphViewportState,
  NodeInfoResponse,
  NodeType,
  SourceBundle,
  SourceOption,
  StudyEdge,
  StudyMode,
  TargetBundle,
  TargetMetric,
} from "@/app/hooks/types";
import {
  buildEndpointNodeId,
  createDefaultFilter,
  EDGE_TYPE_OPTIONS,
  extractDefinitionTexts,
  filterGraph,
  formatNodeDetail,
  formatScore,
  LEVEL_OPTIONS,
  NODE_TYPE_OPTIONS,
} from "@/app/components/study/studyUtils";


const ONTOLOGY_EXPANSION_ENABLED = false;
const API_BASE_URL = (process.env.NEXT_PUBLIC_STUDY_API_BASE_URL || "").replace(/\/$/, "");
const FETCH_TIMEOUT_MS = 20000;
const LOCAL_DEV_API_PORTS = ["8000", "8001"] as const;

let discoveredLocalApiBaseUrl: string | null = null;

type ScreenClass = "wide" | "medium" | "narrow";
type PanelKey = "candidate" | "targets" | "controls";
type MobilePanelKey = PanelKey | "source" | "legend" | "inspector";
type PanelSide = "left" | "right";
type FloatingPosition = { x: number; y: number };

const PANEL_LABELS: Record<PanelKey, string> = {
  candidate: "candidate",
  targets: "targets",
  controls: "controls",
};

const PANEL_SHORT_LABELS: Record<PanelKey, string> = {
  candidate: "Candidate",
  targets: "Targets",
  controls: "Controls",
};

const METRIC_META = {
  decision_basis: {
    title: "Decision basis",
    help: "Which kind of evidence most strongly drove the current candidate decision.",
  },
  evidence_strength: {
    title: "Evidence strength",
    help: "How much support the available evidence provides for the current candidate.",
  },
  evidence_agreement: {
    title: "Evidence agreement",
    help: "How consistently the available evidence sources point in the same direction.",
  },
} as const;

function normalizeMode(value: string | null): StudyMode {
  return value === "study" ? "study" : "app";
}

function readQueryState(): { mode: StudyMode; source: string } {
  if (typeof window === "undefined") {
    return { mode: "app", source: "" };
  }
  const params = new URLSearchParams(window.location.search);
  return {
    mode: normalizeMode(params.get("mode")),
    source: params.get("source") ?? "",
  };
}

function localDevApiBaseUrl(): string {
  if (typeof window === "undefined") return "";
  const { hostname, port, protocol } = window.location;
  const isLocalHost = hostname === "localhost" || hostname === "127.0.0.1";
  if (!isLocalHost || LOCAL_DEV_API_PORTS.includes(port as (typeof LOCAL_DEV_API_PORTS)[number])) return "";
  return `${protocol}//${hostname}:${LOCAL_DEV_API_PORTS[0]}`;
}

function configuredApiBaseUrl(): string {
  if (API_BASE_URL) return API_BASE_URL;
  if (discoveredLocalApiBaseUrl !== null) return discoveredLocalApiBaseUrl;
  return localDevApiBaseUrl();
}

function uniqueApiBases(values: string[]): string[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    if (seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

function localDevApiBaseCandidates(): string[] {
  if (API_BASE_URL) return [API_BASE_URL];
  if (typeof window === "undefined") return [""];
  const { hostname, port, protocol } = window.location;
  const isLocalHost = hostname === "localhost" || hostname === "127.0.0.1";
  if (!isLocalHost) return [""];

  const candidates: string[] = [];
  if (discoveredLocalApiBaseUrl !== null) candidates.push(discoveredLocalApiBaseUrl);
  if (LOCAL_DEV_API_PORTS.includes(port as (typeof LOCAL_DEV_API_PORTS)[number])) candidates.push("");
  LOCAL_DEV_API_PORTS.forEach((candidatePort) => {
    if (port !== candidatePort) candidates.push(`${protocol}//${hostname}:${candidatePort}`);
  });
  candidates.push("");
  return uniqueApiBases(candidates);
}

function isLocalHostLocation(): boolean {
  if (typeof window === "undefined") return false;
  const { hostname } = window.location;
  return hostname === "localhost" || hostname === "127.0.0.1";
}

function apiUrl(path: string, baseUrl?: string): string {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  const base = baseUrl ?? configuredApiBaseUrl();
  return `${base}${normalized}`;
}

function formatSourcesLoadError(endpoint: string, response: Response): string {
  const message = `Failed to load sources from ${endpoint} (${response.status} ${response.statusText}).`;
  if (response.status !== 404 || !isLocalHostLocation()) {
    return message;
  }
  return `${message} If this page is being served by a frontend-only dev server, start the Python study runtime on localhost:8000 or localhost:8001, or set NEXT_PUBLIC_STUDY_API_BASE_URL to the study runtime.`;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function fetchWithTimeout(endpoint: string, resourceLabel: string): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(endpoint, { signal: controller.signal });
  } catch (error) {
    if (isAbortError(error)) {
      throw new Error(
        `Timed out loading ${resourceLabel} from ${endpoint} after ${Math.round(FETCH_TIMEOUT_MS / 1000)}s.`,
      );
    }
    if (error instanceof Error && error.message) {
      throw error;
    }
    throw new Error(`Failed to load ${resourceLabel} from ${endpoint}.`);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function fetchSourcesWithApiFallback(): Promise<SourceOption[]> {
  let lastError: Error | null = null;
  for (const baseUrl of localDevApiBaseCandidates()) {
    const endpoint = apiUrl("/api/study/sources", baseUrl);
    try {
      const response = await fetchWithTimeout(endpoint, "sources");
      if (!response.ok) {
        throw new Error(formatSourcesLoadError(endpoint, response));
      }
      const payload = (await response.json()) as SourceOption[];
      discoveredLocalApiBaseUrl = baseUrl;
      return payload;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(`Failed to load sources from ${endpoint}.`);
    }
  }
  throw lastError || new Error("Failed to load sources from the study runtime.");
}

function writeQueryState(mode: StudyMode, source: string) {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (mode === "study") {
    url.searchParams.set("mode", "study");
  } else {
    url.searchParams.delete("mode");
  }
  if (source) {
    url.searchParams.set("source", source);
  } else {
    url.searchParams.delete("source");
  }
  window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
}

function deriveScreenClass(width: number): ScreenClass {
  if (width >= 1540) return "wide";
  if (width >= 1100) return "medium";
  return "narrow";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function clampFloatingPosition(
  position: FloatingPosition,
  boxSize: { width: number; height: number },
  containerSize: { width: number; height: number },
  inset = 16,
): FloatingPosition {
  const maxX = Math.max(inset, containerSize.width - boxSize.width - inset);
  const maxY = Math.max(inset, containerSize.height - boxSize.height - inset);
  return {
    x: clamp(position.x, inset, maxX),
    y: clamp(position.y, inset, maxY),
  };
}

function matchesSourceOption(option: SourceOption, query: string): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return true;
  return (
    option.source_label.toLowerCase().includes(normalized) ||
    option.source_id.toLowerCase().includes(normalized)
  );
}

function shortenMiddle(value: string, maxLength = 72): string {
  if (value.length <= maxLength) return value;
  const keep = Math.max(18, Math.floor((maxLength - 3) / 2));
  return `${value.slice(0, keep)}...${value.slice(-keep)}`;
}

function useScrollCue(containerRef: React.RefObject<HTMLElement | null>): boolean {
  const [showCue, setShowCue] = useState(false);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;

    const updateCue = () => {
      const scrollRemaining = node.scrollHeight - node.clientHeight - node.scrollTop;
      setShowCue(scrollRemaining > 8);
    };

    updateCue();
    node.addEventListener("scroll", updateCue);
    const resizeObserver = new ResizeObserver(updateCue);
    resizeObserver.observe(node);

    return () => {
      node.removeEventListener("scroll", updateCue);
      resizeObserver.disconnect();
    };
  }, [containerRef]);

  return showCue;
}

function LegendNode({
  color,
  label,
  compact = false,
}: {
  color: string;
  label: string;
  compact?: boolean;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: compact ? "0.38rem" : "0.55rem", whiteSpace: "nowrap" }}>
      <span
        style={{
          width: compact ? "0.72rem" : "0.9rem",
          height: compact ? "0.72rem" : "0.9rem",
          borderRadius: "0.32rem",
          background: color,
          border: "1px solid rgba(61,79,95,0.22)",
          flexShrink: 0,
        }}
      />
      <span style={{ color: "#4f6270", fontSize: compact ? "0.78rem" : "0.9rem" }}>{label}</span>
    </div>
  );
}

function LegendEdge({
  color,
  dashed,
  label,
  compact = false,
}: {
  color: string;
  dashed?: boolean;
  label: string;
  compact?: boolean;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: compact ? "0.38rem" : "0.55rem", whiteSpace: "nowrap" }}>
      <span
        style={{
          width: compact ? "1.1rem" : "1.45rem",
          borderTop: `${compact ? 2 : 3}px ${dashed ? "dashed" : "solid"} ${color}`,
          flexShrink: 0,
        }}
      />
      <span style={{ color: "#4f6270", fontSize: compact ? "0.78rem" : "0.9rem" }}>{label}</span>
    </div>
  );
}

function SmallLogo() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.7rem" }}>
      <div
        style={{
          width: "2.15rem",
          height: "2.15rem",
          borderRadius: "0.8rem",
          background: "linear-gradient(140deg, #3f6179 0%, #8fb5d2 100%)",
          color: "#ffffff",
          display: "grid",
          placeItems: "center",
          fontWeight: 800,
          fontSize: "0.9rem",
          boxShadow: "0 12px 28px rgba(58, 89, 114, 0.22)",
        }}
      >
        E
      </div>
      <div>
        <div style={{ fontWeight: 800, color: "#20323d", lineHeight: 1 }}>Exact-OM</div>
        <div style={{ color: "#60717d", fontSize: "0.9rem", marginTop: "0.18rem" }}>
          Explanation visualizer
        </div>
      </div>
    </div>
  );
}

function DetailPill({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value?: string | null;
  tone?: "neutral" | "source" | "target";
}) {
  if (!value) return null;
  const tones = {
    neutral: {
      background: "rgba(242, 246, 248, 0.96)",
      border: "rgba(84, 105, 120, 0.14)",
      label: "#6c7e8a",
      value: "#28404d",
    },
    source: {
      background: "rgba(221, 235, 244, 0.96)",
      border: "rgba(77, 105, 132, 0.2)",
      label: "#5b7488",
      value: "#29475f",
    },
    target: {
      background: "rgba(233, 240, 245, 0.98)",
      border: "rgba(88, 110, 128, 0.18)",
      label: "#6a7d8b",
      value: "#334957",
    },
  }[tone];
  return (
    <div
      style={{
        borderRadius: "999px",
        border: `1px solid ${tones.border}`,
        background: tones.background,
        padding: "0.32rem 0.56rem",
      }}
    >
      <span style={{ color: tones.label, fontSize: "0.78rem", marginRight: "0.35rem" }}>{label}</span>
      <span style={{ color: tones.value, fontWeight: 700 }}>{value}</span>
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
        borderRadius: "18px",
        border: "1px solid rgba(67, 88, 103, 0.12)",
        background: "rgba(250, 252, 253, 0.98)",
        padding: "0.84rem 0.9rem",
        minWidth: 0,
      }}
    >
      <div style={{ fontWeight: 700, color: "#2d4351", marginBottom: "0.36rem" }}>{title}</div>
      {children}
    </div>
  );
}

function PanelShell({
  eyebrow,
  title,
  children,
  onHide,
  collapseSide = "right",
  compact = false,
}: {
  eyebrow: string;
  title?: string;
  children: React.ReactNode;
  onHide?: () => void;
  collapseSide?: PanelSide;
  compact?: boolean;
}) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const showScrollCue = useScrollCue(bodyRef);

  return (
    <section
      style={{
        position: "relative",
        minHeight: 0,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        borderRadius: compact ? "20px" : "26px",
        border: "1px solid rgba(70, 92, 107, 0.12)",
        background: "rgba(255, 255, 255, 0.94)",
        boxShadow: "0 22px 44px rgba(54, 74, 90, 0.08)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: compact ? "0.38rem" : "0.75rem",
          alignItems: "flex-start",
          padding: compact ? "0.46rem 0.54rem 0.04rem" : "0.88rem 0.9rem 0.16rem",
          flexShrink: 0,
        }}
      >
        <div>
          <div style={{ fontSize: compact ? "0.68rem" : "0.76rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#73838f" }}>
            {eyebrow}
          </div>
          {title ? (
            <div style={{ marginTop: compact ? "0.18rem" : "0.28rem", fontWeight: 800, color: "#223441", fontSize: compact ? "0.88rem" : "1.02rem" }}>
              {title}
            </div>
          ) : null}
        </div>
        {onHide ? (
          <button
            type="button"
            onClick={onHide}
            title="Hide panel"
            aria-label="Hide panel"
            style={{
              borderRadius: "999px",
              border: "1px solid rgba(70, 92, 107, 0.12)",
              background: "rgba(255,255,255,0.96)",
              color: "#48606f",
              width: compact ? "1.55rem" : "1.95rem",
              height: compact ? "1.55rem" : "1.95rem",
              padding: 0,
              cursor: "pointer",
              fontWeight: 700,
              flexShrink: 0,
              display: "grid",
              placeItems: "center",
              fontSize: compact ? "0.84rem" : "1rem",
              boxShadow: "0 10px 22px rgba(38, 57, 70, 0.1)",
              backdropFilter: "blur(10px)",
            }}
          >
            {collapseSide === "left" ? "←" : "→"}
          </button>
        ) : null}
      </div>
      <div
        ref={bodyRef}
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          padding: compact ? "0.34rem 0.5rem 1.15rem" : "0.58rem 0.9rem 2.6rem",
          boxSizing: "border-box",
          fontSize: compact ? "0.88rem" : undefined,
        }}
      >
        {children}
      </div>
      {showScrollCue ? (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: 0,
            height: "3.8rem",
            background: "linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.96) 72%, rgba(255,255,255,1) 100%)",
            pointerEvents: "none",
            display: "grid",
            placeItems: "end center",
            paddingBottom: "0.6rem",
            color: "#71818d",
            fontSize: "0.76rem",
            letterSpacing: "0.03em",
          }}
        >
          Scroll for more
        </div>
      ) : null}
    </section>
  );
}

function FloatingPanel({
  title,
  position,
  onClose,
  children,
  compact = false,
}: {
  title: string;
  position: "left" | "right";
  onClose: () => void;
  children: React.ReactNode;
  compact?: boolean;
}) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const showScrollCue = useScrollCue(bodyRef);

  return (
    <div
      style={{
        position: "absolute",
        top: "1rem",
        bottom: compact ? undefined : "1rem",
        ...(position === "left" ? { left: "1rem" } : { right: "1rem" }),
        width: compact ? "min(22rem, calc(100% - 2rem))" : "min(26rem, calc(100% - 4.5rem))",
        maxHeight: compact ? "min(70vh, 28rem)" : undefined,
        borderRadius: compact ? "18px" : "24px",
        border: "1px solid rgba(69, 90, 105, 0.14)",
        background: "rgba(255, 255, 255, 0.96)",
        boxShadow: "0 24px 48px rgba(42, 61, 75, 0.18)",
        overflow: "hidden",
        zIndex: 5,
        backdropFilter: "blur(14px)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "0.75rem",
          padding: compact ? "0.62rem 0.68rem 0.28rem" : "0.9rem 1rem 0.4rem",
        }}
      >
        <div style={{ fontWeight: 800, color: "#243743" }}>{title}</div>
        <button
          type="button"
          onClick={onClose}
          style={{
            borderRadius: "999px",
            border: "1px solid rgba(70, 92, 107, 0.12)",
            background: "#f7fafb",
            color: "#48606f",
            width: compact ? "1.55rem" : "1.75rem",
            height: compact ? "1.55rem" : "1.75rem",
            padding: 0,
            cursor: "pointer",
            fontWeight: 900,
            display: "grid",
            placeItems: "center",
            lineHeight: 1,
          }}
        >
          ×
        </button>
      </div>
      <div
        ref={bodyRef}
        style={{
          flex: compact ? "0 1 auto" : 1,
          minHeight: 0,
          overflowY: "auto",
          padding: compact ? "0.3rem 0.68rem 0.72rem" : "0.3rem 0.9rem 2.2rem",
        }}
      >
        {children}
      </div>
      {showScrollCue ? (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: 0,
            height: "3.5rem",
            background: "linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.96) 72%, rgba(255,255,255,1) 100%)",
            pointerEvents: "none",
            display: "grid",
            placeItems: "end center",
            paddingBottom: "0.55rem",
            color: "#71818d",
            fontSize: "0.76rem",
          }}
        >
          Scroll for more
        </div>
      ) : null}
    </div>
  );
}

function HiddenPanelEdgeButton({
  side,
  title,
  onClick,
}: {
  side: PanelSide;
  title: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      style={{
        width: "2rem",
        height: "2rem",
        borderRadius: "999px",
        border: "1px solid rgba(70, 92, 107, 0.12)",
        background: "rgba(255,255,255,0.96)",
        color: "#48606f",
        cursor: "pointer",
        fontWeight: 700,
        display: "grid",
        placeItems: "center",
        boxShadow: "0 10px 22px rgba(38, 57, 70, 0.1)",
        backdropFilter: "blur(10px)",
      }}
    >
      {side === "left" ? "→" : "←"}
    </button>
  );
}

function ResizableSidebar({
  side,
  width,
  minWidth,
  maxWidth,
  onWidthChange,
  children,
}: {
  side: PanelSide;
  width: number;
  minWidth: number;
  maxWidth: number;
  onWidthChange: (nextWidth: number) => void;
  children: React.ReactNode;
}) {
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!dragRef.current) return;
      const delta = side === "left" ? event.clientX - dragRef.current.startX : dragRef.current.startX - event.clientX;
      onWidthChange(clamp(dragRef.current.startWidth + delta, minWidth, maxWidth));
    };
    const handlePointerUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [maxWidth, minWidth, onWidthChange, side]);

  const handle = (
    <div
      onPointerDown={(event) => {
        dragRef.current = {
          startX: event.clientX,
          startWidth: width,
        };
      }}
      style={{
        position: "absolute",
        top: 0,
        bottom: 0,
        width: "14px",
        cursor: "ew-resize",
        zIndex: 3,
        touchAction: "none",
        ...(side === "left" ? { right: "-7px" } : { left: "-7px" }),
        display: "grid",
        placeItems: "center",
      }}
      title="Resize panel"
      aria-label="Resize panel"
    >
      <div
        style={{
          width: "4px",
          height: "4.8rem",
          borderRadius: "999px",
          background: "rgba(102, 122, 137, 0.18)",
          boxShadow: "0 0 0 1px rgba(255,255,255,0.7)",
        }}
      />
    </div>
  );

  return (
    <div style={{ position: "relative", width, minWidth: 0, minHeight: 0, height: "100%" }}>
      {side === "right" ? handle : null}
      <div style={{ height: "100%" }}>{children}</div>
      {side === "left" ? handle : null}
    </div>
  );
}

function MetricCard({
  title,
  helpText,
  metric,
  onOpen,
  compact = false,
}: {
  title: string;
  helpText: string;
  metric?: TargetMetric;
  onOpen: () => void;
  compact?: boolean;
}) {
  const [showHelp, setShowHelp] = useState(false);
  if (!metric?.label && !metric?.description) return null;
  return (
    <button
      type="button"
      onClick={onOpen}
      onMouseEnter={() => setShowHelp(true)}
      onMouseLeave={() => setShowHelp(false)}
      onFocus={() => setShowHelp(true)}
      onBlur={() => setShowHelp(false)}
      style={{
        textAlign: "left",
        position: "relative",
        borderRadius: compact ? "13px" : "18px",
        border: "1px solid rgba(76, 97, 112, 0.14)",
        background: "linear-gradient(180deg, rgba(245,249,251,0.98) 0%, rgba(255,255,255,0.98) 100%)",
        padding: compact ? "0.52rem 0.58rem" : "0.82rem 0.9rem",
        cursor: "pointer",
        width: "100%",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: compact ? "0.45rem" : "0.75rem", alignItems: "flex-start" }}>
        <div style={{ fontWeight: 800, color: "#28404d", fontSize: compact ? "0.86rem" : "1rem" }}>{title}</div>
        {metric.label ? (
          <span
            style={{
              borderRadius: "999px",
              background: "rgba(76, 111, 139, 0.1)",
              color: "#4a6780",
              padding: compact ? "0.18rem 0.4rem" : "0.24rem 0.55rem",
              fontSize: compact ? "0.72rem" : "0.82rem",
              fontWeight: 800,
              whiteSpace: "nowrap",
            }}
          >
            {metric.label}
          </span>
        ) : null}
      </div>
      {showHelp ? (
        <div
          style={{
            marginTop: compact ? "0.36rem" : "0.58rem",
            borderRadius: compact ? "10px" : "14px",
            border: "1px solid rgba(77, 105, 132, 0.14)",
            background: "rgba(234, 242, 247, 0.8)",
            padding: compact ? "0.4rem 0.46rem" : "0.58rem 0.66rem",
            color: "#4c6271",
            lineHeight: compact ? 1.34 : 1.45,
            fontSize: compact ? "0.76rem" : "0.9rem",
          }}
        >
          <span style={{ fontWeight: 800, color: "#385163" }}>Metric meaning:</span> {helpText}
        </div>
      ) : null}
      {metric.description ? (
        <div style={{ marginTop: compact ? "0.34rem" : "0.55rem", color: "#5e7280", lineHeight: compact ? 1.36 : 1.5, fontSize: compact ? "0.78rem" : undefined }}>{metric.description}</div>
      ) : null}
    </button>
  );
}

function MetricDialog({
  title,
  helpText,
  metric,
  onClose,
}: {
  title: string;
  helpText: string;
  metric?: TargetMetric;
  onClose: () => void;
}) {
  if (!metric) return null;
  return (
    <div
      onMouseDown={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(18, 28, 36, 0.28)",
        display: "grid",
        placeItems: "center",
        zIndex: 40,
        padding: "1.25rem",
      }}
    >
      <div
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        style={{
          width: "min(30rem, 100%)",
          borderRadius: "24px",
          border: "1px solid rgba(70, 92, 107, 0.12)",
          background: "rgba(255,255,255,0.98)",
          boxShadow: "0 28px 60px rgba(27, 42, 51, 0.22)",
          padding: "1.2rem 1.25rem 1.15rem",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "flex-start" }}>
          <div>
            <div style={{ fontSize: "0.76rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#758590" }}>
              Metric detail
            </div>
            <div style={{ marginTop: "0.28rem", fontWeight: 900, color: "#253743", fontSize: "1.15rem" }}>
              {title}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              borderRadius: "999px",
              border: "1px solid rgba(70, 92, 107, 0.12)",
              background: "#f8fbfc",
              color: "#46606f",
              padding: "0.42rem 0.74rem",
              cursor: "pointer",
              fontWeight: 700,
            }}
          >
            Close
          </button>
        </div>

        <div style={{ marginTop: "0.95rem", display: "grid", gap: "0.82rem" }}>
          <InspectorCard title="What it means">
            <div style={{ color: "#526570", lineHeight: 1.5 }}>{helpText}</div>
          </InspectorCard>
          <InspectorCard title="Current label">
            <div style={{ fontWeight: 800, color: "#2d4351" }}>{metric.label || "n/a"}</div>
          </InspectorCard>
          <InspectorCard title="Current description">
            <div style={{ color: "#526570", lineHeight: 1.5 }}>
              {metric.description || "No description available for this metric."}
            </div>
          </InspectorCard>
        </div>
      </div>
    </div>
  );
}

function SourcePicker({
  options,
  selectedSourceId,
  onSelect,
  disabled,
  compact = false,
}: {
  options: SourceOption[];
  selectedSourceId: string;
  onSelect: (sourceId: string) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const pickerRef = useRef<HTMLDivElement | null>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(0);

  const filtered = useMemo(
    () => options.filter((option) => matchesSourceOption(option, query)).slice(0, 18),
    [options, query],
  );
  const selectedOption = options.find((option) => option.source_id === selectedSourceId) || null;

  useEffect(() => {
    if (!open) return;
    const selectedIndex = filtered.findIndex((option) => option.source_id === selectedSourceId);
    setHighlightedIndex(selectedIndex >= 0 ? selectedIndex : 0);
  }, [filtered, open, selectedSourceId]);

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (event: MouseEvent) => {
      if (!pickerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    optionRefs.current[highlightedIndex]?.scrollIntoView({ block: "nearest" });
  }, [highlightedIndex, open]);

  const chooseOption = (option: SourceOption) => {
    onSelect(option.source_id);
    setQuery("");
    setOpen(false);
  };

  return (
    <div ref={pickerRef} style={{ position: "relative", minWidth: compact ? "min(16rem, 100%)" : "min(20rem, 100%)", flex: compact ? "1 1 18rem" : "0.95 1 24rem" }}>
      <div style={{ fontSize: compact ? "0.68rem" : "0.76rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#6f818f" }}>
        Browse sources
      </div>
      <div
        style={{
          marginTop: compact ? "0.28rem" : "0.42rem",
          borderRadius: compact ? "14px" : "18px",
          border: open ? "1px solid rgba(77, 105, 132, 0.28)" : "1px solid rgba(66, 89, 104, 0.14)",
          background: "linear-gradient(180deg, rgba(249,252,253,0.98) 0%, rgba(243,248,250,0.94) 100%)",
          boxShadow: open ? "0 12px 28px rgba(54, 74, 90, 0.1)" : "inset 0 1px 0 rgba(255,255,255,0.6)",
          overflow: "hidden",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: compact ? "0.45rem" : "0.65rem", padding: compact ? "0.46rem 0.56rem 0.34rem" : "0.68rem 0.8rem 0.48rem" }}>
          <span style={{ color: "#6a7d8a", fontSize: compact ? "0.82rem" : "0.92rem" }}>⌕</span>
          <input
            disabled={disabled}
            value={query}
            onFocus={() => setOpen(true)}
            onChange={(event) => {
              setQuery(event.target.value);
              setOpen(true);
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setOpen(true);
                setHighlightedIndex((prev) => Math.min(prev + 1, Math.max(filtered.length - 1, 0)));
                return;
              }
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setOpen(true);
                setHighlightedIndex((prev) => Math.max(prev - 1, 0));
                return;
              }
              if (event.key === "Enter") {
                const option = filtered[highlightedIndex] || filtered[0];
                if (option) {
                  event.preventDefault();
                  chooseOption(option);
                }
                return;
              }
              if (event.key === "Escape") {
                setOpen(false);
              }
            }}
            placeholder="Search labels or IRIs"
            style={{
              width: "100%",
              border: "none",
              outline: "none",
              background: "transparent",
              color: "#223644",
              padding: 0,
              fontSize: compact ? "0.88rem" : undefined,
            }}
          />
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: compact ? "0.55rem" : "0.8rem",
            padding: compact ? "0 0.56rem 0.42rem" : "0 0.8rem 0.62rem",
            color: "#6d808d",
            fontSize: compact ? "0.74rem" : "0.83rem",
          }}
        >
          <span>{selectedOption ? `${compact ? "" : "Current: "}${selectedOption.source_label}` : "Choose a source"}</span>
          <span>{filtered.length} shown</span>
        </div>
      </div>
      {open ? (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: "calc(100% + 0.45rem)",
            zIndex: 12,
            borderRadius: "20px",
            border: "1px solid rgba(69, 91, 106, 0.14)",
            background: "rgba(255,255,255,0.98)",
            boxShadow: "0 20px 44px rgba(39, 58, 72, 0.16)",
            maxHeight: "20rem",
            overflowY: "auto",
            padding: "0.45rem",
          }}
        >
          {filtered.length ? (
            filtered.map((option, index) => {
              const active = option.source_id === selectedSourceId;
              const highlighted = index === highlightedIndex;
              return (
                <button
                  ref={(node) => {
                    optionRefs.current[index] = node;
                  }}
                  key={option.source_id}
                  type="button"
                  onClick={() => chooseOption(option)}
                  onMouseEnter={() => setHighlightedIndex(index)}
                  style={{
                    display: "block",
                    width: "100%",
                    textAlign: "left",
                    borderRadius: "16px",
                    border:
                      active || highlighted
                        ? "1px solid rgba(77, 105, 132, 0.22)"
                        : "1px solid transparent",
                    background:
                      active
                        ? "rgba(230, 239, 246, 0.92)"
                        : highlighted
                          ? "rgba(241, 247, 250, 0.96)"
                          : "transparent",
                    padding: "0.7rem 0.78rem",
                    cursor: "pointer",
                    transition: "background 120ms ease, border-color 120ms ease",
                  }}
                >
                  <div style={{ fontWeight: highlighted ? 700 : 600, color: "#233744" }}>{option.source_label}</div>
                  <div style={{ marginTop: "0.2rem", color: "#647480", fontSize: "0.84rem", wordBreak: "break-word" }}>
                    {option.source_id}
                  </div>
                </button>
              );
            })
          ) : (
            <div style={{ padding: "0.7rem 0.78rem", color: "#667783" }}>No matching sources.</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function CandidateContent({
  target,
  onOpenMetric,
  compact = false,
}: {
  target: TargetBundle | null;
  onOpenMetric: (metricKey: keyof typeof METRIC_META) => void;
  compact?: boolean;
}) {
  if (!target) {
    return <div style={{ color: "#61727d", marginTop: "0.8rem" }}>No candidate selected.</div>;
  }

  return (
    <>
      <div
        style={{
          borderRadius: compact ? "14px" : "20px",
          border: "1px solid rgba(77, 105, 132, 0.14)",
          background: "linear-gradient(180deg, rgba(235,244,249,0.98) 0%, rgba(255,255,255,0.98) 100%)",
          padding: compact ? "0.58rem 0.62rem" : "0.88rem 0.92rem",
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.65)",
        }}
      >
        <div style={{ fontWeight: 900, color: "#284458", lineHeight: 1.24, fontSize: compact ? "0.88rem" : "1.02rem" }}>
          {target.target_label}
        </div>
        <div style={{ marginTop: compact ? "0.24rem" : "0.4rem", color: "#607484", lineHeight: 1.34, fontSize: compact ? "0.78rem" : "0.94rem" }}>
          Rank #{target.rank} • confidence {formatScore(target.score)}
        </div>
        <div style={{ marginTop: compact ? "0.42rem" : "0.68rem", display: "flex", flexWrap: "wrap", gap: compact ? "0.28rem" : "0.45rem" }}>
          <DetailPill label="Candidate" value={target.ground_truth ? "Ground truth" : "Alternative"} tone="target" />
          <DetailPill label="LLM" value={target.llm.decision || "not used"} tone="target" />
        </div>
      </div>

      <div style={{ marginTop: compact ? "0.58rem" : "1rem", display: "grid", gap: compact ? "0.48rem" : "0.76rem" }}>
        <MetricCard
          title={METRIC_META.decision_basis.title}
          helpText={METRIC_META.decision_basis.help}
          metric={target.metrics.decision_basis}
          onOpen={() => onOpenMetric("decision_basis")}
          compact={compact}
        />
        <MetricCard
          title={METRIC_META.evidence_strength.title}
          helpText={METRIC_META.evidence_strength.help}
          metric={target.metrics.evidence_strength}
          onOpen={() => onOpenMetric("evidence_strength")}
          compact={compact}
        />
        <MetricCard
          title={METRIC_META.evidence_agreement.title}
          helpText={METRIC_META.evidence_agreement.help}
          metric={target.metrics.evidence_agreement}
          onOpen={() => onOpenMetric("evidence_agreement")}
          compact={compact}
        />
      </div>

      <div
        style={{
          marginTop: compact ? "0.58rem" : "1rem",
          borderRadius: compact ? "14px" : "18px",
          border: "1px solid rgba(77, 105, 132, 0.14)",
          background: "rgba(244, 249, 252, 0.96)",
          padding: compact ? "0.58rem 0.62rem" : "0.84rem 0.9rem",
        }}
      >
        <div style={{ fontWeight: 800, color: "#2d4759", marginBottom: compact ? "0.28rem" : "0.45rem", fontSize: compact ? "0.86rem" : undefined }}>Textual rationale</div>
        <div style={{ color: "#586c7a", lineHeight: compact ? 1.42 : 1.58, fontSize: compact ? "0.8rem" : undefined }}>
          {target.llm.rationale || "No rationale available for this candidate."}
        </div>
      </div>
    </>
  );
}

export default function Home() {
  const [queryReady, setQueryReady] = useState(false);
  const [mode, setMode] = useState<StudyMode>("app");
  const [sourceId, setSourceId] = useState("");
  const [sourceOptions, setSourceOptions] = useState<SourceOption[]>([]);
  const [sourcesLoading, setSourcesLoading] = useState(true);
  const [sourcesError, setSourcesError] = useState("");
  const [bundle, setBundle] = useState<SourceBundle | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedTargetId, setSelectedTargetId] = useState("");
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
  const [windowWidth, setWindowWidth] = useState(0);
  const [windowHeight, setWindowHeight] = useState(0);
  const [graphExpanded, setGraphExpanded] = useState(false);
  const [panelVisibility, setPanelVisibility] = useState<Record<PanelKey, boolean>>({
    candidate: true,
    targets: true,
    controls: true,
  });
  const [mobilePanel, setMobilePanel] = useState<MobilePanelKey | null>(null);
  const [activeMetricKey, setActiveMetricKey] = useState<keyof typeof METRIC_META | null>(null);
  const [inspectorHeight, setInspectorHeight] = useState(168);
  const [candidatePanelWidth, setCandidatePanelWidth] = useState(292);
  const [rightPanelWidth, setRightPanelWidth] = useState(308);
  const [appLegendPosition, setAppLegendPosition] = useState<FloatingPosition | null>(null);
  const [appLegendVisible, setAppLegendVisible] = useState(true);
  const [appHeaderVisible, setAppHeaderVisible] = useState(true);
  const [studyTargetBoxPosition, setStudyTargetBoxPosition] = useState<FloatingPosition | null>(null);
  const [studyLegendPosition, setStudyLegendPosition] = useState<FloatingPosition | null>(null);
  const [studyTargetBoxVisible, setStudyTargetBoxVisible] = useState(true);
  const [studyLegendVisible, setStudyLegendVisible] = useState(true);
  const [studyBottomPanelHeight, setStudyBottomPanelHeight] = useState(218);
  const [studyBottomPanelVisible, setStudyBottomPanelVisible] = useState(true);
  const [appInspectorVisible, setAppInspectorVisible] = useState(true);
  const inspectorDragRef = useRef<{ startY: number; startHeight: number } | null>(null);
  const inspectorVisibilityUserSetRef = useRef(false);
  const graphShellRef = useRef<HTMLElement | null>(null);
  const appLegendRef = useRef<HTMLDivElement | null>(null);
  const studyTargetBoxRef = useRef<HTMLDivElement | null>(null);
  const studyLegendRef = useRef<HTMLDivElement | null>(null);
  const appLegendDragRef = useRef<{
    startX: number;
    startY: number;
    startLeft: number;
    startTop: number;
    width: number;
    height: number;
  } | null>(null);
  const studyTargetDragRef = useRef<{
    startX: number;
    startY: number;
    startLeft: number;
    startTop: number;
    width: number;
    height: number;
  } | null>(null);
  const studyLegendDragRef = useRef<{
    startX: number;
    startY: number;
    startLeft: number;
    startTop: number;
    width: number;
    height: number;
  } | null>(null);
  const studyBottomPanelDragRef = useRef<{ startY: number; startHeight: number } | null>(null);
  const inspectorUserSizedRef = useRef(false);
  const appLegendVisibilityUserSetRef = useRef(false);
  const appHeaderVisibilityUserSetRef = useRef(false);
  const studyBottomPanelUserSizedRef = useRef(false);
  const studyTargetVisibilityUserSetRef = useRef(false);
  const studyLegendVisibilityUserSetRef = useRef(false);
  const graphViewportStoreRef = useRef<Record<string, GraphViewportState>>({});
  const activeLevel = mode === "study" ? 2 : selectedLevel;
  const screenClass = deriveScreenClass(windowWidth || 1600);
  const constrainedStudyViewport =
    mode === "study" &&
    ((windowWidth > 0 && windowWidth < 760) || (windowHeight > 0 && windowHeight < 700));
  const constrainedAppViewport =
    mode === "app" &&
    ((windowWidth > 0 && windowWidth < 980) || (windowHeight > 0 && windowHeight < 740));
  const veryConstrainedStudyViewport =
    mode === "study" &&
    ((windowWidth > 0 && windowWidth < 560) || (windowHeight > 0 && windowHeight < 560));
  const veryConstrainedAppViewport =
    mode === "app" &&
    ((windowWidth > 0 && windowWidth < 640) || (windowHeight > 0 && windowHeight < 600));
  const compactStudyPanel = mode === "study" && (constrainedStudyViewport || (windowWidth > 0 && windowWidth < 980));
  const compactAppInspector = mode === "app" && (constrainedAppViewport || (windowWidth > 0 && windowWidth < 1120));
  const compactAppSidePanels = mode === "app" && screenClass === "medium";
  const appInspectorAsWindow = mode === "app" && (veryConstrainedAppViewport || screenClass === "medium");
  const compactAppLegend = mode === "app" && (constrainedAppViewport || screenClass === "medium");
  const wideStudyViewport = mode === "study" && windowWidth >= 1280 && windowHeight >= 720;
  const appOverlayInset = constrainedAppViewport ? 10 : 16;
  const studyOverlayInset = constrainedStudyViewport ? 10 : 16;
  const studyAutoPanelHeight = (() => {
    const availableHeight = Math.max(windowHeight || 900, 420);
    if (constrainedStudyViewport) {
      const ideal = availableHeight * (veryConstrainedStudyViewport ? 0.2 : 0.24);
      const minHeight = veryConstrainedStudyViewport ? 126 : 156;
      const maxHeight = veryConstrainedStudyViewport
        ? clamp(availableHeight * 0.3, 150, 210)
        : clamp(availableHeight * 0.34, 190, 260);
      return Math.round(clamp(ideal, minHeight, maxHeight));
    }
    if (wideStudyViewport) {
      const ideal = availableHeight * 0.18;
      return Math.round(clamp(ideal, 156, 230));
    }
    const ideal = compactStudyPanel ? availableHeight * 0.34 : availableHeight * 0.25;
    const minHeight = compactStudyPanel ? 282 : 220;
    const maxHeight = compactStudyPanel
      ? clamp(availableHeight * 0.42, 320, 430)
      : clamp(availableHeight * 0.34, 260, 360);
    return Math.round(clamp(ideal, minHeight, maxHeight));
  })();
  const appAutoInspectorHeight = (() => {
    const availableHeight = Math.max(windowHeight || 900, 420);
    if (constrainedAppViewport) {
      const ideal = availableHeight * (veryConstrainedAppViewport ? 0.18 : 0.22);
      const minHeight = veryConstrainedAppViewport ? 112 : 132;
      const maxHeight = veryConstrainedAppViewport
        ? clamp(availableHeight * 0.28, 130, 190)
        : clamp(availableHeight * 0.32, 170, 250);
      return Math.round(clamp(ideal, minHeight, maxHeight));
    }
    return 168;
  })();

  useEffect(() => {
    const syncFromLocation = () => {
      const next = readQueryState();
      setMode(next.mode);
      setSourceId(next.source);
      setQueryReady(true);
    };
    syncFromLocation();
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
      setWindowHeight(window.innerHeight);
    };
    handleResize();
    window.addEventListener("popstate", syncFromLocation);
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("popstate", syncFromLocation);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    if (mode !== "app") {
      setSourcesLoading(false);
      setSourcesError("");
      setSourceOptions([]);
      return;
    }

    let cancelled = false;
    setSourcesLoading(true);
    setSourcesError("");
    fetchSourcesWithApiFallback()
      .then((payload: SourceOption[]) => {
        if (cancelled) return;
        setSourceOptions(payload);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setSourceOptions([]);
          setSourcesError(err.message || "Failed to load sources from the study runtime.");
        }
      })
      .finally(() => {
        if (!cancelled) setSourcesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  useEffect(() => {
    if (!queryReady || sourcesLoading) return;
    if (mode === "app" && !sourceId && sourceOptions.length) {
      const nextSource = sourceOptions[0].source_id;
      setSourceId(nextSource);
      writeQueryState("app", nextSource);
    }
  }, [mode, queryReady, sourceId, sourceOptions, sourcesLoading]);

  useEffect(() => {
    if (mode === "study") {
      setGraphExpanded(false);
    }
  }, [mode]);

  useEffect(() => {
    if (mode !== "app") {
      setMobilePanel(null);
      return;
    }
    if (constrainedAppViewport) {
      setPanelVisibility({
        candidate: false,
        targets: false,
        controls: false,
      });
      setMobilePanel(null);
      return;
    }
    if (screenClass === "narrow") {
      setPanelVisibility({
        candidate: false,
        targets: false,
        controls: false,
      });
      setMobilePanel(null);
      return;
    }
    setPanelVisibility({
      candidate: true,
      targets: true,
      controls: true,
    });
    setMobilePanel(null);
  }, [constrainedAppViewport, mode, screenClass]);

  useEffect(() => {
    if (graphExpanded) {
      setMobilePanel(null);
    }
  }, [graphExpanded]);

  useEffect(() => {
    if (!windowWidth) return;
    setCandidatePanelWidth((prev) => clamp(prev, 240, clamp(windowWidth * 0.3, 270, 380)));
    setRightPanelWidth((prev) => clamp(prev, 240, clamp(windowWidth * 0.3, 270, 390)));
  }, [windowWidth]);

  useEffect(() => {
    if (mode !== "study" || studyBottomPanelUserSizedRef.current) return;
    setStudyBottomPanelHeight(studyAutoPanelHeight);
  }, [mode, studyAutoPanelHeight]);

  useEffect(() => {
    if (mode !== "app" || inspectorUserSizedRef.current) return;
    setInspectorHeight(appAutoInspectorHeight);
  }, [appAutoInspectorHeight, mode]);

  useEffect(() => {
    if (mode !== "app" || !windowWidth || !windowHeight) return;
    if (!inspectorVisibilityUserSetRef.current && appInspectorAsWindow) {
      setAppInspectorVisible(false);
    }
  }, [appInspectorAsWindow, mode, windowHeight, windowWidth]);

  useEffect(() => {
    if (mode !== "app" || !windowWidth || !windowHeight) return;
    if (!appLegendVisibilityUserSetRef.current) {
      setAppLegendVisible(!constrainedAppViewport);
    }
    if (!appHeaderVisibilityUserSetRef.current) {
      setAppHeaderVisible(!veryConstrainedAppViewport);
    }
  }, [constrainedAppViewport, mode, veryConstrainedAppViewport, windowHeight, windowWidth]);

  useEffect(() => {
    if (mode !== "study" || !windowWidth || !windowHeight) return;
    if (!studyTargetVisibilityUserSetRef.current) {
      setStudyTargetBoxVisible(!constrainedStudyViewport);
    }
    if (!studyLegendVisibilityUserSetRef.current) {
      setStudyLegendVisible(!constrainedStudyViewport);
    }
  }, [constrainedStudyViewport, mode, windowHeight, windowWidth]);

  useEffect(() => {
    if (mode !== "app") return;
    const syncAppOverlayBounds = () => {
      const shellRect = graphShellRef.current?.getBoundingClientRect();
      const legendRect = appLegendRef.current?.getBoundingClientRect();
      if (!shellRect || !legendRect || !appLegendPosition) return;

      const nextLegendPosition = clampFloatingPosition(
        appLegendPosition,
        { width: legendRect.width, height: legendRect.height },
        { width: shellRect.width, height: shellRect.height },
        appOverlayInset,
      );
      if (nextLegendPosition.x !== appLegendPosition.x || nextLegendPosition.y !== appLegendPosition.y) {
        setAppLegendPosition(nextLegendPosition);
      }
    };

    const frameId = window.requestAnimationFrame(syncAppOverlayBounds);
    return () => window.cancelAnimationFrame(frameId);
  }, [appLegendPosition, appOverlayInset, mode, windowWidth]);

  useEffect(() => {
    if (mode !== "study") return;
    const syncStudyOverlayBounds = () => {
      const shellRect = graphShellRef.current?.getBoundingClientRect();
      const boxRect = studyTargetBoxRef.current?.getBoundingClientRect();
      const legendRect = studyLegendRef.current?.getBoundingClientRect();
      if (!shellRect) return;

      if (boxRect) {
        if (!studyTargetBoxPosition) {
          setStudyTargetBoxPosition(
            clampFloatingPosition(
              {
                x: shellRect.width - boxRect.width - studyOverlayInset,
                y: studyOverlayInset,
              },
              { width: boxRect.width, height: boxRect.height },
              { width: shellRect.width, height: shellRect.height },
            ),
          );
        } else {
          const nextTargetPosition = clampFloatingPosition(
            studyTargetBoxPosition,
            { width: boxRect.width, height: boxRect.height },
            { width: shellRect.width, height: shellRect.height },
          );
          if (nextTargetPosition.x !== studyTargetBoxPosition.x || nextTargetPosition.y !== studyTargetBoxPosition.y) {
            setStudyTargetBoxPosition(nextTargetPosition);
          }
        }
      }

      if (legendRect && studyLegendPosition) {
        const nextLegendPosition = clampFloatingPosition(
          studyLegendPosition,
          { width: legendRect.width, height: legendRect.height },
          { width: shellRect.width, height: shellRect.height },
        );
        if (nextLegendPosition.x !== studyLegendPosition.x || nextLegendPosition.y !== studyLegendPosition.y) {
          setStudyLegendPosition(nextLegendPosition);
        }
      }
    };

    const frameId = window.requestAnimationFrame(syncStudyOverlayBounds);
    return () => window.cancelAnimationFrame(frameId);
  }, [mode, studyLegendPosition, studyOverlayInset, studyTargetBoxPosition, windowWidth]);

  useEffect(() => {
    if (!queryReady) return;

    if (mode === "study" && !sourceId) {
      setLoading(false);
      setBundle(null);
      setSelectedTargetId("");
      setError("Missing source query parameter. Use ?mode=study&source=<exact_source_iri>.");
      return;
    }

    if (mode === "app" && !sourceId) {
      if (!sourcesLoading && sourcesError) {
        setLoading(false);
        setBundle(null);
        setSelectedTargetId("");
        setError(sourcesError);
      } else if (!sourcesLoading && sourceOptions.length === 0) {
        setLoading(false);
        setBundle(null);
        setSelectedTargetId("");
        setError("No sources are available for this study.");
      }
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError("");
    const sourceEndpoint = apiUrl(`/api/study/source?source=${encodeURIComponent(sourceId)}`);

    fetchWithTimeout(sourceEndpoint, "study source")
      .then(async (response) => {
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || `Failed to load study source from ${sourceEndpoint}.`);
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
        setSelectedTargetId("");
        setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [mode, queryReady, sourceId, sourceOptions.length, sourcesError, sourcesLoading]);

  const selectedSourceOption = useMemo(
    () => sourceOptions.find((option) => option.source_id === sourceId) || null,
    [sourceId, sourceOptions],
  );

  const selectedTarget = useMemo(() => {
    if (!bundle) return null;
    return bundle.targets.find((target) => target.target_id === selectedTargetId) || bundle.targets[0] || null;
  }, [bundle, selectedTargetId]);

  useEffect(() => {
    setExpandedNodeId(null);
    setExpansion(null);
    setActiveMetricKey(null);
    if (mode === "study") {
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
    }
  }, [mode, selectedTargetId]);

  const cacheKey = (targetId: string, nodeId: string) => `${targetId}::${nodeId}`;

  const ensureNodeInfo = async (nodeId: string): Promise<NodeInfoResponse | null> => {
    if (!bundle || !selectedTarget) return null;
    const key = cacheKey(selectedTarget.target_id, nodeId);
    if (nodeInfoCache[key]) return nodeInfoCache[key];
    const response = await fetch(
      apiUrl(`/api/study/node-info?source=${encodeURIComponent(bundle.source_id)}&target=${encodeURIComponent(
        selectedTarget.target_id,
      )}&node_id=${encodeURIComponent(nodeId)}`),
    );
    if (!response.ok) return null;
    const payload = (await response.json()) as NodeInfoResponse;
    setNodeInfoCache((prev) => ({ ...prev, [key]: payload }));
    return payload;
  };

  useEffect(() => {
    if (mode !== "app" || !selectedTarget || !bundle) return;
    const endpointId = buildEndpointNodeId("target", selectedTarget.target_id);
    setSelectedEdgeId(null);
    setSelectedNodeId(endpointId);
    void ensureNodeInfo(endpointId);
  }, [bundle, mode, selectedTarget]);

  const handleSourceSelect = (nextSourceId: string) => {
    setSourceId(nextSourceId);
    writeQueryState(mode, nextSourceId);
    if (mode === "app" && veryConstrainedAppViewport) {
      setMobilePanel(null);
      setAppHeaderOverlayVisible(false);
    }
  };

  const handleTargetSelect = (nextTargetId: string) => {
    setSelectedTargetId(nextTargetId);
    if (mode === "app" && veryConstrainedAppViewport) {
      setMobilePanel(null);
    }
  };

  const handleNodeClick = async (nodeId: string) => {
    if (mode !== "app" || !bundle || !selectedTarget) return;
    setSelectedEdgeId(null);
    setSelectedNodeId(nodeId);
    if (appInspectorAsWindow) {
      setAppInspectorVisible(false);
      setMobilePanel("inspector");
    }
    const info = await ensureNodeInfo(nodeId);
    if (!ONTOLOGY_EXPANSION_ENABLED || !info?.expandable) return;
    if (expandedNodeId === nodeId) {
      setExpandedNodeId(null);
      setExpansion(null);
      return;
    }
    const response = await fetch(
      apiUrl(`/api/study/expand-node?source=${encodeURIComponent(bundle.source_id)}&target=${encodeURIComponent(
        selectedTarget.target_id,
      )}&node_id=${encodeURIComponent(nodeId)}`),
    );
    if (!response.ok) return;
    const payload = (await response.json()) as ExpandNodeResponse;
    setExpandedNodeId(nodeId);
    setExpansion(payload);
  };

  const handleEdgeClick = (edgeId: string) => {
    if (mode !== "app") return;
    setSelectedNodeId(null);
    setSelectedEdgeId(edgeId);
    if (appInspectorAsWindow) {
      setAppInspectorVisible(false);
      setMobilePanel("inspector");
    }
  };

  const handleBackgroundClick = () => {
    if (mode !== "app") return;
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    if (appInspectorAsWindow && mobilePanel === "inspector") {
      setMobilePanel(null);
    }
  };

  const handleStudyEndpointDefinitionRequest = async (nodeId: string): Promise<string | null> => {
    if (mode !== "study") return null;
    const info = await ensureNodeInfo(nodeId);
    return extractDefinitionTexts(info)[0] || null;
  };

  const visibleGraph = useMemo(
    () =>
      filterGraph(
        selectedTarget,
        activeLevel,
        ONTOLOGY_EXPANSION_ENABLED,
        nodeFilters,
        edgeFilters,
        expansion,
        mode,
      ),
    [activeLevel, edgeFilters, expansion, mode, nodeFilters, selectedTarget],
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

  const activeMetric = activeMetricKey && selectedTarget ? selectedTarget.metrics[activeMetricKey] : undefined;
  const activeLevelOption = LEVEL_OPTIONS.find((option) => option.value === selectedLevel) || LEVEL_OPTIONS[0];
  const graphViewportKey = [mode, sourceId, selectedTarget?.target_id || ""].join("|");
  const savedGraphViewport = graphViewportStoreRef.current[graphViewportKey] || null;
  const layoutResetKey = [
    mode,
    sourceId,
    selectedTarget?.target_id || "",
    String(graphExpanded),
    screenClass,
    String(panelVisibility.candidate),
    String(panelVisibility.targets),
    String(panelVisibility.controls),
  ].join("|");

  const leftHiddenPanels: PanelKey[] = [];
  const rightHiddenPanels: PanelKey[] = [];
  if (!graphExpanded && screenClass !== "narrow") {
    if (screenClass === "wide" && !panelVisibility.candidate) {
      leftHiddenPanels.push("candidate");
    }
    if ((screenClass === "wide" || screenClass === "medium") && !panelVisibility.targets) {
      rightHiddenPanels.push("targets");
    }
    if ((screenClass === "wide" || screenClass === "medium") && !panelVisibility.controls) {
      rightHiddenPanels.push("controls");
    }
    if (screenClass === "medium" && !panelVisibility.candidate) {
      rightHiddenPanels.push("candidate");
    }
  }

  const openMetric = (metricKey: keyof typeof METRIC_META) => setActiveMetricKey(metricKey);

  const togglePanelVisibility = (key: PanelKey, value: boolean) => {
    setPanelVisibility((prev) => ({ ...prev, [key]: value }));
  };

  const setStudyTargetOverlayVisible = (visible: boolean) => {
    studyTargetVisibilityUserSetRef.current = true;
    setStudyTargetBoxVisible(visible);
  };

  const setAppLegendOverlayVisible = (visible: boolean) => {
    appLegendVisibilityUserSetRef.current = true;
    setAppLegendVisible(visible);
  };

  const setAppHeaderOverlayVisible = (visible: boolean) => {
    appHeaderVisibilityUserSetRef.current = true;
    setAppHeaderVisible(visible);
  };

  const setStudyLegendOverlayVisible = (visible: boolean) => {
    studyLegendVisibilityUserSetRef.current = true;
    setStudyLegendVisible(visible);
  };

  const handleAppLegendDragStart = (event: React.PointerEvent<HTMLElement>) => {
    const shellRect = graphShellRef.current?.getBoundingClientRect();
    const legendRect = appLegendRef.current?.getBoundingClientRect();
    if (!shellRect || !legendRect) return;
    event.preventDefault();
    event.stopPropagation();
    appLegendDragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      startLeft: legendRect.left - shellRect.left,
      startTop: legendRect.top - shellRect.top,
      width: legendRect.width,
      height: legendRect.height,
    };
  };

  const handleStudyTargetBoxDragStart = (event: React.PointerEvent<HTMLElement>) => {
    const shellRect = graphShellRef.current?.getBoundingClientRect();
    const boxRect = studyTargetBoxRef.current?.getBoundingClientRect();
    if (!shellRect || !boxRect) return;
    event.preventDefault();
    studyTargetDragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      startLeft: boxRect.left - shellRect.left,
      startTop: boxRect.top - shellRect.top,
      width: boxRect.width,
      height: boxRect.height,
    };
  };

  const handleStudyLegendDragStart = (event: React.PointerEvent<HTMLElement>) => {
    const shellRect = graphShellRef.current?.getBoundingClientRect();
    const legendRect = studyLegendRef.current?.getBoundingClientRect();
    if (!shellRect || !legendRect) return;
    event.preventDefault();
    event.stopPropagation();
    studyLegendDragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      startLeft: legendRect.left - shellRect.left,
      startTop: legendRect.top - shellRect.top,
      width: legendRect.width,
      height: legendRect.height,
    };
  };

  const handleStudyBottomPanelResizeStart = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    studyBottomPanelUserSizedRef.current = true;
    studyBottomPanelDragRef.current = {
      startY: event.clientY,
      startHeight: studyBottomPanelHeight,
    };
  };

  const handleInspectorResizeStart = (event: React.PointerEvent<HTMLDivElement>) => {
    inspectorUserSizedRef.current = true;
    inspectorDragRef.current = {
      startY: event.clientY,
      startHeight: inspectorHeight,
    };
  };

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!graphShellRef.current) return;
      const shellRect = graphShellRef.current.getBoundingClientRect();
      if (appLegendDragRef.current) {
        const drag = appLegendDragRef.current;
        const nextPosition = clampFloatingPosition(
          {
            x: drag.startLeft + (event.clientX - drag.startX),
            y: drag.startTop + (event.clientY - drag.startY),
          },
          { width: drag.width, height: drag.height },
          { width: shellRect.width, height: shellRect.height },
          appOverlayInset,
        );
        setAppLegendPosition(nextPosition);
      }
      if (studyTargetDragRef.current) {
        const drag = studyTargetDragRef.current;
        const nextPosition = clampFloatingPosition(
          {
            x: drag.startLeft + (event.clientX - drag.startX),
            y: drag.startTop + (event.clientY - drag.startY),
          },
          { width: drag.width, height: drag.height },
          { width: shellRect.width, height: shellRect.height },
        );
        setStudyTargetBoxPosition(nextPosition);
      }
      if (studyLegendDragRef.current) {
        const drag = studyLegendDragRef.current;
        const nextPosition = clampFloatingPosition(
          {
            x: drag.startLeft + (event.clientX - drag.startX),
            y: drag.startTop + (event.clientY - drag.startY),
          },
          { width: drag.width, height: drag.height },
          { width: shellRect.width, height: shellRect.height },
        );
        setStudyLegendPosition(nextPosition);
      }
    };
    const handlePointerUp = () => {
      appLegendDragRef.current = null;
      studyTargetDragRef.current = null;
      studyLegendDragRef.current = null;
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [appOverlayInset]);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!studyBottomPanelDragRef.current) return;
      const delta = studyBottomPanelDragRef.current.startY - event.clientY;
      const maxHeight = constrainedStudyViewport
        ? clamp((windowHeight || window.innerHeight) * 0.34, 180, 280)
        : clamp((windowHeight || window.innerHeight) * 0.48, 260, 450);
      const minHeight = constrainedStudyViewport ? 118 : 150;
      const nextHeight = clamp(studyBottomPanelDragRef.current.startHeight + delta, minHeight, maxHeight);
      setStudyBottomPanelHeight(nextHeight);
    };
    const handlePointerUp = () => {
      studyBottomPanelDragRef.current = null;
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [constrainedStudyViewport, studyBottomPanelHeight, windowHeight]);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!inspectorDragRef.current) return;
      const delta = inspectorDragRef.current.startY - event.clientY;
      const maxHeight = constrainedAppViewport
        ? clamp((windowHeight || window.innerHeight) * 0.34, 170, 270)
        : clamp((windowHeight || window.innerHeight) * 0.42, 220, 360);
      const minHeight = constrainedAppViewport ? 104 : 132;
      const nextHeight = clamp(inspectorDragRef.current.startHeight + delta, minHeight, maxHeight);
      setInspectorHeight(nextHeight);
    };
    const handlePointerUp = () => {
      inspectorDragRef.current = null;
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [constrainedAppViewport, inspectorHeight, windowHeight]);

  const candidatePanel = (
    <PanelShell
      eyebrow="Selected candidate"
      collapseSide={screenClass === "wide" ? "left" : "right"}
      onHide={screenClass === "narrow" ? undefined : () => togglePanelVisibility("candidate", false)}
      compact={compactAppSidePanels}
    >
      <CandidateContent target={selectedTarget} onOpenMetric={openMetric} compact={compactAppSidePanels} />
    </PanelShell>
  );

  const targetsPanel = (
    <PanelShell
      eyebrow="Target selection"
      title={selectedTarget ? `${bundle?.targets.length || 0} candidates` : undefined}
      collapseSide="right"
      onHide={screenClass === "narrow" ? undefined : () => togglePanelVisibility("targets", false)}
      compact={compactAppSidePanels}
    >
      <div style={{ display: "grid", gap: compactAppSidePanels ? "0.36rem" : "0.52rem" }}>
        {(bundle?.targets || []).map((target) => {
          const active = target.target_id === selectedTargetId;
          return (
            <button
              key={target.target_id}
              type="button"
              onClick={() => handleTargetSelect(target.target_id)}
              style={{
                textAlign: "left",
                padding: compactAppSidePanels ? "0.46rem 0.54rem" : "0.7rem 0.78rem",
                borderRadius: compactAppSidePanels ? "12px" : "16px",
                border: active ? "2px solid rgba(110, 134, 151, 0.84)" : "1px solid rgba(71, 91, 105, 0.12)",
                background: active ? "rgba(236, 244, 248, 0.94)" : "#ffffff",
                cursor: "pointer",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: compactAppSidePanels ? "0.45rem" : "0.75rem" }}>
                <div style={{ fontWeight: 600, color: "#29404e" }}>#{target.rank}</div>
                <div style={{ color: "#49657b", fontWeight: 600 }}>{formatScore(target.score)}</div>
              </div>
              <div style={{ marginTop: compactAppSidePanels ? "0.2rem" : "0.32rem", fontWeight: 600, color: "#2a404e", lineHeight: 1.28, fontSize: compactAppSidePanels ? "0.84rem" : "0.95rem" }}>
                {target.target_label}
              </div>
              <div style={{ marginTop: compactAppSidePanels ? "0.16rem" : "0.24rem", color: "#6b7b87", fontSize: compactAppSidePanels ? "0.74rem" : "0.86rem" }}>
                {target.ground_truth ? "Ground truth candidate" : "Candidate"}
              </div>
            </button>
          );
        })}
      </div>
    </PanelShell>
  );

  const controlsPanel = (
    <PanelShell
      eyebrow="Display controls"
      collapseSide="right"
      onHide={screenClass === "narrow" ? undefined : () => togglePanelVisibility("controls", false)}
      compact={compactAppSidePanels}
    >
      <label style={{ display: "block", fontWeight: 800, color: "#243643", fontSize: compactAppSidePanels ? "0.86rem" : undefined }}>Explanation granularity</label>
      <select
        value={selectedLevel}
        onChange={(event) => setSelectedLevel(Number(event.target.value))}
        style={{
          width: "100%",
          marginTop: compactAppSidePanels ? "0.28rem" : "0.42rem",
          padding: compactAppSidePanels ? "0.44rem 0.52rem" : "0.62rem 0.72rem",
          borderRadius: compactAppSidePanels ? "11px" : "14px",
          border: "1px solid rgba(70, 92, 107, 0.18)",
          background: "#ffffff",
          color: "#2a404e",
        }}
      >
        {LEVEL_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>

      <div style={{ marginTop: compactAppSidePanels ? "0.42rem" : "0.68rem", color: "#5e7484", lineHeight: 1.36, fontSize: compactAppSidePanels ? "0.78rem" : "0.9rem" }}>
        {activeLevelOption.description}
      </div>

      <div style={{ marginTop: compactAppSidePanels ? "0.28rem" : "0.45rem", color: "#7a8994", lineHeight: 1.36, fontSize: compactAppSidePanels ? "0.74rem" : "0.85rem" }}>
        Ontology expansion is temporarily disabled in this viewer revision.
      </div>

      <details open style={{ marginTop: compactAppSidePanels ? "0.58rem" : "1rem" }}>
        <summary style={{ fontWeight: 800, cursor: "pointer", color: "#2a404e", fontSize: compactAppSidePanels ? "0.86rem" : undefined }}>Filters</summary>
        <div style={{ display: "grid", gap: compactAppSidePanels ? "0.56rem" : "1rem", marginTop: compactAppSidePanels ? "0.48rem" : "0.8rem" }}>
          <div>
            <div style={{ fontWeight: 800, marginBottom: compactAppSidePanels ? "0.28rem" : "0.45rem", color: "#2a404e", fontSize: compactAppSidePanels ? "0.82rem" : undefined }}>Node filters</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: compactAppSidePanels ? "0.26rem 0.42rem" : "0.4rem 0.65rem" }}>
              {NODE_TYPE_OPTIONS.map((type) => (
                <label key={type} style={{ display: "flex", gap: compactAppSidePanels ? "0.32rem" : "0.48rem", alignItems: "center", fontSize: compactAppSidePanels ? "0.78rem" : "0.94rem" }}>
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
            <div style={{ fontWeight: 800, marginBottom: compactAppSidePanels ? "0.28rem" : "0.45rem", color: "#2a404e", fontSize: compactAppSidePanels ? "0.82rem" : undefined }}>Edge filters</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: compactAppSidePanels ? "0.26rem 0.42rem" : "0.4rem 0.65rem" }}>
              {EDGE_TYPE_OPTIONS.map((type) => (
                <label key={type} style={{ display: "flex", gap: compactAppSidePanels ? "0.32rem" : "0.48rem", alignItems: "center", fontSize: compactAppSidePanels ? "0.78rem" : "0.94rem" }}>
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
    </PanelShell>
  );

  const appLegendItems = (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: compactAppLegend ? "0.3rem 0.48rem" : "0.55rem 0.85rem",
        alignItems: "center",
      }}
    >
      <LegendNode color={NODE_COLORS.Source} label={NODE_TYPE_LABELS.Source} compact={compactAppLegend} />
      <LegendNode color={NODE_COLORS.Target} label={NODE_TYPE_LABELS.Target} compact={compactAppLegend} />
      <LegendNode color={NODE_COLORS["source-context"]} label={NODE_TYPE_LABELS["source-context"]} compact={compactAppLegend} />
      <LegendNode color={NODE_COLORS["target-context"]} label={NODE_TYPE_LABELS["target-context"]} compact={compactAppLegend} />
      <LegendEdge color={EDGE_COLORS.hierarchy} label={EDGE_TYPE_LABELS.hierarchy} compact={compactAppLegend} />
      <LegendEdge color={EDGE_COLORS.similarity} label={EDGE_TYPE_LABELS.similarity} compact={compactAppLegend} />
      <LegendEdge color={EDGE_COLORS.difference} label={EDGE_TYPE_LABELS.difference} compact={compactAppLegend} />
      <LegendEdge color={EDGE_COLORS.attribute} label={EDGE_TYPE_LABELS.attribute} compact={compactAppLegend} />
      <LegendEdge color={EDGE_COLORS["bridge-support"]} dashed label={EDGE_TYPE_LABELS["bridge-support"]} compact={compactAppLegend} />
      <LegendEdge color={EDGE_COLORS["bridge-contrast"]} dashed label={EDGE_TYPE_LABELS["bridge-contrast"]} compact={compactAppLegend} />
    </div>
  );

  const graphLegend = (
    <div
      ref={appLegendRef}
      style={{
        position: "absolute",
        left: appLegendPosition ? `${appLegendPosition.x}px` : `${appOverlayInset}px`,
        top: appLegendPosition || !appInspectorAsWindow || veryConstrainedAppViewport ? (appLegendPosition ? `${appLegendPosition.y}px` : `${appOverlayInset}px`) : undefined,
        bottom: !appLegendPosition && appInspectorAsWindow && !veryConstrainedAppViewport ? `${appOverlayInset}px` : undefined,
        right: appLegendPosition ? undefined : `${appOverlayInset}px`,
        zIndex: 4,
        borderRadius: constrainedAppViewport ? "14px" : "18px",
        border: "1px solid rgba(70, 92, 107, 0.12)",
        background: "rgba(255,255,255,0.9)",
        boxShadow: "0 14px 28px rgba(38, 57, 70, 0.1)",
        padding: constrainedAppViewport ? "0.46rem 0.54rem" : "0.72rem 0.84rem",
        display: "flex",
        flexWrap: "wrap",
        gap: compactAppLegend ? "0.36rem 0.48rem" : "0.65rem 0.95rem",
        alignItems: "center",
        backdropFilter: "blur(10px)",
        maxHeight: constrainedAppViewport ? "7.2rem" : undefined,
        maxWidth: appLegendPosition
          ? `calc(100% - ${appLegendPosition.x + appOverlayInset}px)`
          : `calc(100% - ${appOverlayInset * 2}px)`,
        overflowY: constrainedAppViewport ? "auto" : undefined,
      }}
    >
      <button
        type="button"
        onClick={() => setAppLegendOverlayVisible(false)}
        title="Hide legend"
        aria-label="Hide legend"
        style={{
          borderRadius: "999px",
          border: "1px solid rgba(77, 105, 132, 0.14)",
          background: "rgba(255,255,255,0.76)",
          color: "#536b7b",
          width: compactAppLegend ? "1.3rem" : "1.5rem",
          height: compactAppLegend ? "1.3rem" : "1.5rem",
          padding: 0,
          cursor: "pointer",
          fontWeight: 800,
          fontSize: compactAppLegend ? "0.82rem" : "0.92rem",
          flexShrink: 0,
          display: "grid",
          placeItems: "center",
          lineHeight: 1,
        }}
      >
        ×
      </button>
      <button
        type="button"
        onPointerDown={handleAppLegendDragStart}
        title="Drag legend"
        aria-label="Drag legend"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#5d7280",
          fontSize: compactAppLegend ? "0.68rem" : "0.76rem",
          borderRadius: "10px",
          border: "1px solid rgba(77, 105, 132, 0.14)",
          background: "rgba(255,255,255,0.7)",
          width: compactAppLegend ? "2rem" : "2.55rem",
          height: compactAppLegend ? "1.24rem" : "1.48rem",
          padding: 0,
          cursor: "grab",
          userSelect: "none",
          touchAction: "none",
          flexShrink: 0,
        }}
      >
        <span
          aria-hidden="true"
          style={{
            width: compactAppLegend ? "1.2rem" : "1.55rem",
            height: compactAppLegend ? "0.28rem" : "0.34rem",
            borderRadius: "999px",
            background: "rgba(102, 122, 137, 0.24)",
          }}
        />
      </button>
      <div style={{ fontSize: compactAppLegend ? "0.66rem" : "0.72rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#6e8393", flexShrink: 0 }}>
        Legend
      </div>
      {appLegendItems}
    </div>
  );

  const studyLegend = (
    <div
      ref={studyLegendRef}
      style={{
        position: "absolute",
        left: studyLegendPosition ? `${studyLegendPosition.x}px` : `${studyOverlayInset}px`,
        top: studyLegendPosition
          ? `${studyLegendPosition.y}px`
          : windowWidth > 0 && windowWidth < 1180 && studyTargetBoxVisible
            ? constrainedStudyViewport
              ? "4.8rem"
              : "5.8rem"
            : `${studyOverlayInset}px`,
        right: studyLegendPosition
          ? undefined
          : windowWidth > 0 && windowWidth < 1180
            ? `${studyOverlayInset}px`
            : studyTargetBoxVisible && !constrainedStudyViewport
              ? "22rem"
              : `${studyOverlayInset}px`,
        zIndex: 4,
        borderRadius: constrainedStudyViewport ? "14px" : "18px",
        border: "1px solid rgba(70, 92, 107, 0.12)",
        background: "rgba(255,255,255,0.9)",
        boxShadow: "0 14px 28px rgba(38, 57, 70, 0.1)",
        padding: constrainedStudyViewport ? "0.46rem 0.54rem" : "0.72rem 0.84rem",
        display: "flex",
        flexWrap: "wrap",
        gap: constrainedStudyViewport ? "0.42rem 0.62rem" : "0.65rem 0.95rem",
        alignItems: "center",
        backdropFilter: "blur(10px)",
        maxHeight: constrainedStudyViewport ? "7.2rem" : undefined,
        maxWidth: studyLegendPosition
          ? `calc(100% - ${studyLegendPosition.x + studyOverlayInset}px)`
          : windowWidth > 0 && windowWidth < 1180
            ? `calc(100% - ${studyOverlayInset * 2}px)`
            : studyTargetBoxVisible && !constrainedStudyViewport
              ? "calc(100% - 24rem)"
              : `calc(100% - ${studyOverlayInset * 2}px)`,
        overflowY: constrainedStudyViewport ? "auto" : undefined,
      }}
    >
      <button
        type="button"
        onClick={() => setStudyLegendOverlayVisible(false)}
        title="Hide legend"
        aria-label="Hide legend"
        style={{
          borderRadius: "999px",
          border: "1px solid rgba(77, 105, 132, 0.14)",
          background: "rgba(255,255,255,0.76)",
          color: "#536b7b",
          width: constrainedStudyViewport ? "1.36rem" : "1.5rem",
          height: constrainedStudyViewport ? "1.36rem" : "1.5rem",
          padding: 0,
          cursor: "pointer",
          fontWeight: 800,
          fontSize: constrainedStudyViewport ? "0.86rem" : "0.92rem",
          flexShrink: 0,
          display: "grid",
          placeItems: "center",
          lineHeight: 1,
        }}
      >
        ×
      </button>
      <button
        type="button"
        onPointerDown={handleStudyLegendDragStart}
        title="Drag legend"
        aria-label="Drag legend"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#5d7280",
          fontSize: constrainedStudyViewport ? "0.72rem" : "0.76rem",
          borderRadius: "10px",
          border: "1px solid rgba(77, 105, 132, 0.14)",
          background: "rgba(255,255,255,0.7)",
          width: constrainedStudyViewport ? "2.2rem" : "2.55rem",
          height: constrainedStudyViewport ? "1.32rem" : "1.48rem",
          padding: 0,
          cursor: "grab",
          userSelect: "none",
          touchAction: "none",
          flexShrink: 0,
        }}
      >
        <span
          aria-hidden="true"
          style={{
            width: constrainedStudyViewport ? "1.35rem" : "1.55rem",
            height: "0.34rem",
            borderRadius: "999px",
            background: "rgba(102, 122, 137, 0.24)",
          }}
        />
      </button>
      <div
        style={{
          fontSize: "0.72rem",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "#6e8393",
          flexShrink: 0,
        }}
      >
        Legend
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: constrainedStudyViewport ? "0.38rem 0.58rem" : "0.55rem 0.85rem",
          alignItems: "center",
        }}
      >
        <LegendNode color={NODE_COLORS.Source} label={NODE_TYPE_LABELS.Source} compact={constrainedStudyViewport} />
        <LegendNode color={NODE_COLORS.Target} label={NODE_TYPE_LABELS.Target} compact={constrainedStudyViewport} />
        <LegendNode color={NODE_COLORS["source-context"]} label={NODE_TYPE_LABELS["source-context"]} compact={constrainedStudyViewport} />
        <LegendNode color={NODE_COLORS["target-context"]} label={NODE_TYPE_LABELS["target-context"]} compact={constrainedStudyViewport} />
        <LegendEdge color={EDGE_COLORS.hierarchy} label={EDGE_TYPE_LABELS.hierarchy} compact={constrainedStudyViewport} />
        <LegendEdge color={EDGE_COLORS.similarity} label={EDGE_TYPE_LABELS.similarity} compact={constrainedStudyViewport} />
        <LegendEdge color={EDGE_COLORS.difference} label={EDGE_TYPE_LABELS.difference} compact={constrainedStudyViewport} />
        <LegendEdge color={EDGE_COLORS.attribute} label={EDGE_TYPE_LABELS.attribute} compact={constrainedStudyViewport} />
        <LegendEdge color={EDGE_COLORS["bridge-support"]} dashed label={EDGE_TYPE_LABELS["bridge-support"]} compact={constrainedStudyViewport} />
        <LegendEdge color={EDGE_COLORS["bridge-contrast"]} dashed label={EDGE_TYPE_LABELS["bridge-contrast"]} compact={constrainedStudyViewport} />
      </div>
    </div>
  );

  const hiddenPanelButtons = !graphExpanded && screenClass !== "narrow" && (leftHiddenPanels.length || rightHiddenPanels.length) ? (
    <>
      {leftHiddenPanels.length ? (
        <div
          style={{
            position: "absolute",
            left: "-0.95rem",
            top: "50%",
            transform: "translateY(-50%)",
            zIndex: 4,
            display: "grid",
            gap: "0.45rem",
          }}
        >
          {leftHiddenPanels.map((panelKey) => (
            <HiddenPanelEdgeButton
              key={panelKey}
              side="left"
              title={`Show ${PANEL_LABELS[panelKey]}`}
              onClick={() => togglePanelVisibility(panelKey, true)}
            />
          ))}
        </div>
      ) : null}
      {rightHiddenPanels.length ? (
        <div
          style={{
            position: "absolute",
            right: "-0.95rem",
            top: "50%",
            transform: "translateY(-50%)",
            zIndex: 4,
            display: "grid",
            gap: "0.45rem",
          }}
        >
          {rightHiddenPanels.map((panelKey) => (
            <HiddenPanelEdgeButton
              key={panelKey}
              side="right"
              title={`Show ${PANEL_LABELS[panelKey]}`}
              onClick={() => togglePanelVisibility(panelKey, true)}
            />
          ))}
        </div>
      ) : null}
    </>
  ) : null;

  const narrowPanelButtons = mode === "app" && screenClass === "narrow" && !graphExpanded ? (
    <div
      style={{
        position: "absolute",
        left: "1rem",
        bottom: "1rem",
        zIndex: 4,
        display: "flex",
        flexWrap: "wrap",
        gap: constrainedAppViewport ? "0.38rem" : "0.55rem",
        maxWidth: "calc(100% - 2rem)",
      }}
    >
      {(["targets", "controls", "candidate"] as PanelKey[]).map((panelKey) => {
        const active = mobilePanel === panelKey;
        return (
          <button
            key={panelKey}
            type="button"
            onClick={() => setMobilePanel((prev) => (prev === panelKey ? null : panelKey))}
            style={{
              borderRadius: "999px",
              border: "1px solid rgba(70, 92, 107, 0.14)",
              background: active ? "rgba(235,243,247,0.98)" : "rgba(255,255,255,0.94)",
              color: "#29404e",
              padding: constrainedAppViewport ? "0.56rem 0.78rem" : "0.5rem 0.78rem",
              minHeight: constrainedAppViewport ? "2.75rem" : "2.55rem",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              fontWeight: 700,
              fontSize: constrainedAppViewport ? "0.84rem" : "0.9rem",
              lineHeight: 1.1,
              boxShadow: "0 12px 24px rgba(38, 57, 70, 0.12)",
            }}
        >
          {PANEL_SHORT_LABELS[panelKey]}
        </button>
        );
      })}
    </div>
  ) : null;

  const studyRevealButtonStyle: React.CSSProperties = {
    borderRadius: "999px",
    border: "1px solid rgba(70, 92, 107, 0.14)",
    background: "rgba(255,255,255,0.94)",
    color: "#29404e",
    padding: constrainedStudyViewport ? "0.36rem 0.58rem" : "0.46rem 0.72rem",
    cursor: "pointer",
    fontWeight: 800,
    fontSize: constrainedStudyViewport ? "0.78rem" : "0.86rem",
    boxShadow: "0 12px 24px rgba(38, 57, 70, 0.12)",
    backdropFilter: "blur(10px)",
  };

  const studyOverlayRevealButtons = mode === "study" && selectedTarget ? (
    <>
      {!studyLegendVisible ? (
        <button
          type="button"
          onClick={() => setStudyLegendOverlayVisible(true)}
          title="Show legend"
          aria-label="Show legend"
          style={{
            ...studyRevealButtonStyle,
            position: "absolute",
            left: `${studyOverlayInset}px`,
            top: `${studyOverlayInset}px`,
            zIndex: 5,
          }}
        >
          Legend
        </button>
      ) : null}
      {!studyTargetBoxVisible ? (
        <button
          type="button"
          onClick={() => setStudyTargetOverlayVisible(true)}
          title="Show target selector"
          aria-label="Show target selector"
          style={{
            ...studyRevealButtonStyle,
            position: "absolute",
            right: `${studyOverlayInset}px`,
            top: `${studyOverlayInset}px`,
            zIndex: 5,
          }}
        >
          Target
        </button>
      ) : null}
    </>
  ) : null;

  const appInspectorRevealButton = mode === "app" && !appInspectorAsWindow && !appInspectorVisible ? (
    <button
      type="button"
      onClick={() => {
        inspectorVisibilityUserSetRef.current = true;
        setAppInspectorVisible(true);
      }}
      title="Show inspector"
      aria-label="Show inspector"
      style={{
        position: "absolute",
        right: constrainedAppViewport ? "0.6rem" : "1rem",
        bottom: constrainedAppViewport ? "0.6rem" : "1rem",
        zIndex: 5,
        width: constrainedAppViewport ? "1.8rem" : "2rem",
        height: constrainedAppViewport ? "1.8rem" : "2rem",
        borderRadius: "999px",
        border: "1px solid rgba(70, 92, 107, 0.12)",
        background: "rgba(255,255,255,0.96)",
        color: "#48606f",
        cursor: "pointer",
        fontWeight: 800,
        display: "grid",
        placeItems: "center",
        boxShadow: "0 10px 22px rgba(38, 57, 70, 0.1)",
        backdropFilter: "blur(10px)",
      }}
    >
      ↑
    </button>
  ) : null;

  const appOverlayRevealButtons = mode === "app" ? (
    <>
      {!appLegendVisible ? (
        <button
          type="button"
          onClick={() => {
            if (veryConstrainedAppViewport) {
              setMobilePanel("legend");
              return;
            }
            setAppLegendOverlayVisible(true);
          }}
          title="Show legend"
          aria-label="Show legend"
          style={{
            ...studyRevealButtonStyle,
            position: "absolute",
            left: `${appOverlayInset}px`,
            top: `${appOverlayInset}px`,
            zIndex: 5,
          }}
        >
          Legend
        </button>
      ) : null}
      {!appHeaderVisible ? (
        <button
          type="button"
          onClick={() => {
            if (veryConstrainedAppViewport) {
              setMobilePanel("source");
              return;
            }
            setAppHeaderOverlayVisible(true);
          }}
          title="Show source selector"
          aria-label="Show source selector"
          style={{
            ...studyRevealButtonStyle,
            position: "absolute",
            left: `${appOverlayInset}px`,
            top: appLegendVisible ? `calc(${appOverlayInset}px + 8.4rem)` : `calc(${appOverlayInset}px + 2.7rem)`,
            zIndex: appLegendVisible ? 3 : 5,
          }}
        >
          Source
        </button>
      ) : null}
    </>
  ) : null;

  const appInspectorWindowContent = selectedNodeInfo ? (
    <div style={{ display: "grid", gap: "0.72rem" }}>
      <InspectorCard title="Selected node">
        <div style={{ fontWeight: 800, color: "#2d4351" }}>{selectedNodeInfo.node.label}</div>
        <div style={{ marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
          <DetailPill label="Type" value={NODE_TYPE_LABELS[selectedNodeInfo.node.type]} />
          <DetailPill label="Node kind" value={selectedNodeInfo.node.node_kind || "context"} />
          <DetailPill label="Side" value={selectedNodeInfo.node.ontology_side || "n/a"} />
        </div>
      </InspectorCard>
      <InspectorCard title="Explanation details">
        {selectedNodeDetailItems.length ? (
          <ul style={{ margin: 0, paddingLeft: "1rem", color: "#526570", lineHeight: 1.45 }}>
            {selectedNodeDetailItems.slice(0, 8).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : (
          <div style={{ color: "#6f7f8b", lineHeight: 1.45 }}>
            No explanation-local details available for this node.
          </div>
        )}
      </InspectorCard>
      <InspectorCard title="Description">
        <div style={{ color: "#526570", lineHeight: 1.5 }}>
          {selectedNodeDefinitions[0] || "No description available."}
        </div>
      </InspectorCard>
    </div>
  ) : selectedEdge ? (
    <div style={{ display: "grid", gap: "0.72rem" }}>
      <InspectorCard title="Source">
        <div style={{ color: "#2d4351", fontWeight: 700, lineHeight: 1.42 }}>
          {visibleGraph.nodes.find((node) => node.id === selectedEdge.source)?.label || selectedEdge.source}
        </div>
      </InspectorCard>
      <InspectorCard title="Target">
        <div style={{ color: "#2d4351", fontWeight: 700, lineHeight: 1.42 }}>
          {visibleGraph.nodes.find((node) => node.id === selectedEdge.target)?.label || selectedEdge.target}
        </div>
      </InspectorCard>
      <InspectorCard title="Edge details">
        <div style={{ fontWeight: 800, color: "#2d4351", lineHeight: 1.35 }}>{selectedEdge.label}</div>
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
        </div>
      </InspectorCard>
    </div>
  ) : selectedNodeId ? (
    <div style={{ color: "#61727d", lineHeight: 1.56 }}>Loading node details…</div>
  ) : (
    <div style={{ color: "#61727d", lineHeight: 1.56 }}>
      Click a node or edge in the graph to inspect its label, type, score, and explanation details here.
    </div>
  );

  const graphStatus = loading ? (
    <div style={{ padding: "2rem", color: "#425662" }}>Loading study case…</div>
  ) : error ? (
    <div style={{ padding: "2rem", color: "#8a5b4f", lineHeight: 1.5 }}>{error}</div>
  ) : !bundle || !selectedTarget ? (
    <div style={{ padding: "2rem", color: "#516570" }}>No study panel available.</div>
  ) : null;

  const graphShell = (
    <section
      ref={graphShellRef}
      style={{
        position: "relative",
        borderRadius: constrainedStudyViewport || constrainedAppViewport ? "18px" : "30px",
        overflow: "hidden",
        border: "1px solid rgba(70, 92, 107, 0.12)",
        background: "linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(248,251,252,0.96) 100%)",
        boxShadow: "0 22px 50px rgba(44, 63, 77, 0.12)",
        minHeight: 0,
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(circle at 12% 14%, rgba(107, 151, 188, 0.1) 0%, rgba(107, 151, 188, 0) 34%), radial-gradient(circle at 88% 10%, rgba(133, 166, 188, 0.1) 0%, rgba(133, 166, 188, 0) 30%), linear-gradient(180deg, rgba(255,255,255,0.56) 0%, rgba(248,251,252,0.18) 100%)",
          pointerEvents: "none",
        }}
      />

      {graphStatus ? (
        <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", zIndex: 2 }}>
          <div
            style={{
              width: "min(32rem, calc(100% - 2rem))",
              borderRadius: "24px",
              border: "1px solid rgba(70, 92, 107, 0.12)",
              background: "rgba(255,255,255,0.95)",
              boxShadow: "0 18px 36px rgba(46, 64, 76, 0.1)",
            }}
          >
            {graphStatus}
          </div>
        </div>
      ) : (
        <div style={{ position: "absolute", inset: 0 }}>
          <StudyGraph
            nodes={visibleGraph.nodes}
            edges={visibleGraph.edges}
            mode={mode}
            selectedNodeId={selectedNodeId}
            selectedEdgeId={selectedEdgeId}
            expandedNodeId={expandedNodeId}
            layoutResetKey={layoutResetKey}
            viewportStateKey={graphViewportKey}
            savedViewportState={savedGraphViewport}
            allowInspection={mode === "app"}
            graphExpanded={graphExpanded}
            canToggleGraphExpanded={mode === "app" && (!veryConstrainedAppViewport || graphExpanded)}
            onToggleGraphExpanded={() => setGraphExpanded((prev) => !prev)}
            onViewportChange={(nextViewportState) => {
              if (!graphViewportKey) return;
              graphViewportStoreRef.current[graphViewportKey] = nextViewportState;
            }}
            onEndpointDefinitionRequest={mode === "study" ? handleStudyEndpointDefinitionRequest : undefined}
            onNodeClick={handleNodeClick}
            onEdgeClick={handleEdgeClick}
            onBackgroundClick={handleBackgroundClick}
          />
        </div>
      )}

      {mode === "app" && appLegendVisible ? graphLegend : null}
      {appOverlayRevealButtons}
      {mode === "study" && selectedTarget && studyLegendVisible ? studyLegend : null}
      {studyOverlayRevealButtons}
      {appInspectorRevealButton}
      {hiddenPanelButtons}
      {narrowPanelButtons}

      {mode === "app" && mobilePanel === "candidate" ? (
        <FloatingPanel title="Selected candidate" position="left" onClose={() => setMobilePanel(null)}>
          <CandidateContent target={selectedTarget} onOpenMetric={openMetric} />
        </FloatingPanel>
      ) : null}
      {mode === "app" && mobilePanel === "targets" ? (
        <FloatingPanel title="Target selection" position="right" onClose={() => setMobilePanel(null)}>
          {targetsPanel}
        </FloatingPanel>
      ) : null}
      {mode === "app" && mobilePanel === "controls" ? (
        <FloatingPanel title="Display controls" position="right" onClose={() => setMobilePanel(null)}>
          {controlsPanel}
        </FloatingPanel>
      ) : null}
      {mode === "app" && mobilePanel === "source" ? (
        <FloatingPanel title="Source" position="right" onClose={() => setMobilePanel(null)}>
          <SourcePicker
            options={sourceOptions}
            selectedSourceId={sourceId}
            onSelect={handleSourceSelect}
            disabled={sourcesLoading}
          />
        </FloatingPanel>
      ) : null}
      {mode === "app" && mobilePanel === "legend" ? (
        <FloatingPanel title="Legend" position="left" onClose={() => setMobilePanel(null)} compact>
          {appLegendItems}
        </FloatingPanel>
      ) : null}
      {mode === "app" && mobilePanel === "inspector" ? (
        <FloatingPanel title="Inspector" position="left" onClose={() => setMobilePanel(null)}>
          {appInspectorWindowContent}
        </FloatingPanel>
      ) : null}

      {mode === "study" && selectedTarget ? (
        <>
          {studyTargetBoxVisible ? (
            <div
              ref={studyTargetBoxRef}
              style={{
                position: "absolute",
                top: studyTargetBoxPosition ? `${studyTargetBoxPosition.y}px` : `${studyOverlayInset}px`,
                left: studyTargetBoxPosition ? `${studyTargetBoxPosition.x}px` : undefined,
                right: studyTargetBoxPosition ? undefined : `${studyOverlayInset}px`,
                zIndex: 4,
                borderRadius: constrainedStudyViewport ? "14px" : "18px",
                border: "1px solid rgba(70, 92, 107, 0.12)",
                background: "rgba(255,255,255,0.92)",
                boxShadow: "0 14px 28px rgba(40, 58, 72, 0.12)",
                padding: constrainedStudyViewport ? "0.54rem 0.58rem" : "0.82rem 0.88rem",
                backdropFilter: "blur(10px)",
                maxWidth: constrainedStudyViewport ? "min(19rem, calc(100% - 1.25rem))" : "min(25rem, calc(100% - 2rem))",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: constrainedStudyViewport ? "0.45rem" : "0.8rem",
              }}
            >
              <div
                style={{
                  borderRadius: "999px",
                    background: "rgba(77, 105, 132, 0.14)",
                    color: "#28475f",
                    fontSize: constrainedStudyViewport ? "0.7rem" : "0.76rem",
                    fontWeight: 900,
                  letterSpacing: "0.07em",
                  textTransform: "uppercase",
                  padding: constrainedStudyViewport ? "0.24rem 0.46rem" : "0.32rem 0.62rem",
                  minWidth: 0,
                }}
              >
                Select target
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  gap: constrainedStudyViewport ? "0.32rem" : "0.42rem",
                  flexShrink: 0,
                }}
              >
                <button
                  type="button"
                  onPointerDown={handleStudyTargetBoxDragStart}
                  title="Drag target panel"
                  aria-label="Drag target panel"
                  style={{
                    width: constrainedStudyViewport ? "2.2rem" : "2.55rem",
                    height: constrainedStudyViewport ? "1.32rem" : "1.48rem",
                    borderRadius: "10px",
                    border: "1px solid rgba(77, 105, 132, 0.14)",
                    background: "rgba(255,255,255,0.68)",
                    padding: 0,
                    cursor: "grab",
                    display: "grid",
                    placeItems: "center",
                    userSelect: "none",
                    touchAction: "none",
                  }}
                >
                  <span
                    aria-hidden="true"
                    style={{
                      width: constrainedStudyViewport ? "1.35rem" : "1.55rem",
                      height: constrainedStudyViewport ? "0.3rem" : "0.34rem",
                      borderRadius: "999px",
                      background: "rgba(102, 122, 137, 0.24)",
                    }}
                  />
                </button>
                <button
                  type="button"
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation();
                    setStudyTargetOverlayVisible(false);
                  }}
                  title="Hide target selector"
                  aria-label="Hide target selector"
                  style={{
                    borderRadius: "999px",
                    border: "1px solid rgba(77, 105, 132, 0.14)",
                    background: "rgba(255,255,255,0.76)",
                    color: "#536b7b",
                    width: constrainedStudyViewport ? "1.36rem" : "1.5rem",
                    height: constrainedStudyViewport ? "1.36rem" : "1.5rem",
                    padding: 0,
                    cursor: "pointer",
                    fontWeight: 800,
                    fontSize: constrainedStudyViewport ? "0.86rem" : "0.92rem",
                    flexShrink: 0,
                    display: "grid",
                    placeItems: "center",
                    lineHeight: 1,
                  }}
                >
                  ×
                </button>
              </div>
            </div>
              <select
                value={selectedTargetId}
                onChange={(event) => setSelectedTargetId(event.target.value)}
                style={{
                  width: "100%",
                  marginTop: constrainedStudyViewport ? "0.34rem" : "0.42rem",
                  padding: constrainedStudyViewport ? "0.48rem 0.54rem" : "0.68rem 0.72rem",
                  borderRadius: "12px",
                  border: "1px solid rgba(70, 92, 107, 0.16)",
                  background: "#ffffff",
                  color: "#2a404e",
                  fontSize: constrainedStudyViewport ? "0.86rem" : undefined,
                }}
              >
                {(bundle?.targets || []).map((target) => (
                  <option key={target.target_id} value={target.target_id}>
                    #{target.rank} {target.target_label}
                  </option>
                ))}
              </select>
            </div>
          ) : null}

          {studyBottomPanelVisible ? (
            <div
              style={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: 0,
                height: `${studyBottomPanelHeight}px`,
                zIndex: 4,
                background: "linear-gradient(180deg, rgba(241,247,251,0.18) 0%, rgba(248,252,254,0.96) 22%, rgba(255,255,255,0.98) 100%)",
                borderTop: "1px solid rgba(77, 105, 132, 0.12)",
                backdropFilter: "blur(12px)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  right: 0,
                  top: 0,
                  height: constrainedStudyViewport ? "24px" : "28px",
                  zIndex: 2,
                }}
              >
                <div
                  onPointerDown={handleStudyBottomPanelResizeStart}
                  title="Resize bottom panel"
                  aria-label="Resize bottom panel"
                  style={{
                    position: "absolute",
                    left: "50%",
                    top: 0,
                    transform: "translateX(-50%)",
                    width: constrainedStudyViewport ? "4.8rem" : "6rem",
                    height: "100%",
                    cursor: "ns-resize",
                    display: "grid",
                    placeItems: "center",
                    touchAction: "none",
                  }}
                >
                  <div
                    style={{
                      width: constrainedStudyViewport ? "3.2rem" : "4.1rem",
                      height: constrainedStudyViewport ? "0.28rem" : "0.34rem",
                      borderRadius: "999px",
                      background: "rgba(102, 122, 137, 0.26)",
                    }}
                  />
                </div>
                <button
                  type="button"
                  onClick={() => setStudyBottomPanelVisible(false)}
                  title="Hide bottom panel"
                  aria-label="Hide bottom panel"
                  style={{
                    position: "absolute",
                    right: constrainedStudyViewport ? "0.55rem" : "0.9rem",
                    top: constrainedStudyViewport ? "0.12rem" : "0.2rem",
                    width: constrainedStudyViewport ? "1.65rem" : "1.9rem",
                    height: constrainedStudyViewport ? "1.65rem" : "1.9rem",
                    borderRadius: "999px",
                    border: "1px solid rgba(70, 92, 107, 0.12)",
                    background: "rgba(255,255,255,0.96)",
                    color: "#48606f",
                    cursor: "pointer",
                    fontWeight: 800,
                    display: "grid",
                    placeItems: "center",
                    boxShadow: "0 10px 22px rgba(38, 57, 70, 0.1)",
                    backdropFilter: "blur(10px)",
                  }}
                >
                  ↓
                </button>
              </div>

              <div
                style={{
                  height: "100%",
                  overflowY: "auto",
                  padding: constrainedStudyViewport ? "1.28rem 0.58rem 0.62rem" : "1.55rem 1rem 1rem",
                  boxSizing: "border-box",
                }}
              >
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: compactStudyPanel ? "minmax(0, 1fr)" : "minmax(18rem, 24rem) minmax(0, 1fr)",
                    gap: constrainedStudyViewport ? "0.44rem" : compactStudyPanel ? "0.72rem" : "1rem 1.2rem",
                    alignItems: "start",
                  }}
                >
                  {!constrainedStudyViewport ? (
                    <div
                      style={{
                        order: compactStudyPanel ? 2 : 1,
                        minWidth: 0,
                        borderRadius: "18px",
                        border: "1px solid rgba(77, 105, 132, 0.14)",
                        background: "rgba(247, 251, 253, 0.96)",
                        padding: compactStudyPanel ? "0.58rem 0.72rem" : "0.88rem 0.92rem",
                      }}
                    >
                      <div
                        style={{
                          display: compactStudyPanel ? "flex" : "grid",
                          gap: compactStudyPanel ? "0.5rem" : "0.58rem",
                          alignItems: compactStudyPanel ? "center" : undefined,
                          flexWrap: "wrap",
                        }}
                      >
                        <div style={{ fontSize: "0.72rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#6e8393" }}>
                          Target
                        </div>
                        <div style={{ fontWeight: 900, color: "#2d4759", fontSize: compactStudyPanel ? "0.92rem" : "1rem", lineHeight: 1.3 }}>
                          {selectedTarget.target_label}
                        </div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem" }}>
                          <DetailPill label="Rank" value={`#${selectedTarget.rank}`} tone="target" />
                        </div>
                      </div>
                    </div>
                  ) : null}

                  <div
                    style={{
                      order: compactStudyPanel ? 1 : 2,
                      minWidth: 0,
                      borderRadius: constrainedStudyViewport ? "14px" : "18px",
                      border: "1px solid rgba(77, 105, 132, 0.14)",
                      background: "rgba(250, 252, 254, 0.98)",
                      padding: constrainedStudyViewport ? "0.54rem 0.62rem" : compactStudyPanel ? "0.72rem 0.82rem" : "0.88rem 0.98rem",
                    }}
                  >
                    <div
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        borderRadius: "999px",
                        border: "1px solid rgba(77, 105, 132, 0.18)",
                        background: "rgba(220, 233, 243, 0.92)",
                        color: "#29475f",
                        fontSize: constrainedStudyViewport ? "0.7rem" : "0.78rem",
                        fontWeight: 900,
                        letterSpacing: "0.05em",
                        textTransform: "uppercase",
                        padding: constrainedStudyViewport ? "0.24rem 0.48rem" : "0.34rem 0.68rem",
                        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.65)",
                      }}
                    >
                      Textual rationale
                    </div>
                    {constrainedStudyViewport ? (
                      <div
                        style={{
                          marginTop: "0.4rem",
                          color: "#516779",
                          fontSize: "0.78rem",
                          fontWeight: 800,
                          lineHeight: 1.25,
                        }}
                      >
                        Target: {selectedTarget.target_label}
                      </div>
                    ) : null}
                    <div
                      style={{
                        marginTop: constrainedStudyViewport ? "0.42rem" : "0.68rem",
                        color: "#445b6c",
                        lineHeight: constrainedStudyViewport ? 1.45 : 1.62,
                        maxHeight: "none",
                        overflowY: "visible",
                        paddingRight: "0.1rem",
                        fontSize: constrainedStudyViewport ? "0.86rem" : "0.95rem",
                      }}
                    >
                      {selectedTarget.llm.rationale || "No rationale available for this candidate."}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setStudyBottomPanelVisible(true)}
              title="Show bottom panel"
              aria-label="Show bottom panel"
              style={{
                position: "absolute",
                right: constrainedStudyViewport ? "0.6rem" : "1rem",
                bottom: constrainedStudyViewport ? "0.6rem" : "1rem",
                zIndex: 4,
                width: constrainedStudyViewport ? "1.8rem" : "2rem",
                height: constrainedStudyViewport ? "1.8rem" : "2rem",
                borderRadius: "999px",
                border: "1px solid rgba(70, 92, 107, 0.12)",
                background: "rgba(255,255,255,0.96)",
                color: "#48606f",
                cursor: "pointer",
                fontWeight: 800,
                display: "grid",
                placeItems: "center",
                boxShadow: "0 10px 22px rgba(38, 57, 70, 0.1)",
                backdropFilter: "blur(10px)",
              }}
            >
              ↑
            </button>
          )}
        </>
      ) : null}
    </section>
  );

  if (!queryReady) {
    return (
      <main
        className="study-visualizer-root"
        style={{
          height: "100vh",
          display: "grid",
          placeItems: "center",
          background:
            "linear-gradient(135deg, #f4efe8 0%, #f8fbfd 48%, #edf4f8 100%)",
          color: "#2a3d49",
          fontFamily: "\"Avenir Next\", var(--font-geist-sans), sans-serif",
        }}
      >
        Loading visualizer…
      </main>
    );
  }

  const mainColumns = (() => {
    if (mode === "study" || graphExpanded) return "minmax(0, 1fr)";
    if (screenClass === "wide") {
      const columns = [];
      if (panelVisibility.candidate) columns.push(`${candidatePanelWidth}px`);
      columns.push("minmax(0, 1fr)");
      if (panelVisibility.targets || panelVisibility.controls) columns.push(`${rightPanelWidth}px`);
      return columns.join(" ");
    }
    if (screenClass === "medium") {
      const columns = ["minmax(0, 1fr)"];
      if (panelVisibility.candidate || panelVisibility.targets || panelVisibility.controls) {
        columns.push(`${rightPanelWidth}px`);
      }
      return columns.join(" ");
    }
    return "minmax(0, 1fr)";
  })();
  const showAppInspectorFooter = mode === "app" && appInspectorVisible && !appInspectorAsWindow;

  const rightPanels: React.ReactNode[] = [];
  if (mode === "app" && !graphExpanded) {
    if (screenClass === "medium" && panelVisibility.candidate) {
      rightPanels.push(<React.Fragment key="candidate">{candidatePanel}</React.Fragment>);
    }
    if (panelVisibility.targets) {
      rightPanels.push(<React.Fragment key="targets">{targetsPanel}</React.Fragment>);
    }
    if (panelVisibility.controls) {
      rightPanels.push(<React.Fragment key="controls">{controlsPanel}</React.Fragment>);
    }
  }

  return (
    <main
      className="study-visualizer-root"
      style={{
        height: "100vh",
        background:
          "linear-gradient(135deg, #f4efe8 0%, #f8fbfd 48%, #edf4f8 100%)",
        color: "#273945",
        padding:
          mode === "study"
            ? constrainedStudyViewport
              ? "0.35rem"
              : "0.8rem"
            : constrainedAppViewport
              ? "0.45rem"
              : "0.85rem",
        boxSizing: "border-box",
        overflow: "hidden",
        fontFamily: "\"Avenir Next\", var(--font-geist-sans), sans-serif",
        display: "grid",
        gridTemplateRows:
          mode === "study"
            ? "minmax(0, 1fr)"
            : graphExpanded
              ? showAppInspectorFooter
                ? "minmax(0, 1fr) auto"
                : "minmax(0, 1fr)"
              : appHeaderVisible && showAppInspectorFooter
                ? "auto minmax(0, 1fr) auto"
                : appHeaderVisible
                  ? "auto minmax(0, 1fr)"
                  : showAppInspectorFooter
                    ? "minmax(0, 1fr) auto"
                    : "minmax(0, 1fr)",
        gap: constrainedStudyViewport || constrainedAppViewport ? "0.45rem" : "0.85rem",
      }}
    >
      {mode === "app" && !graphExpanded && appHeaderVisible ? (
        <header
          style={{
            position: "relative",
            borderRadius: constrainedAppViewport ? "18px" : "28px",
            border: "1px solid rgba(70, 92, 107, 0.12)",
            background: "rgba(255,255,255,0.9)",
            boxShadow: "0 18px 36px rgba(52, 71, 84, 0.08)",
            padding: constrainedAppViewport ? "0.5rem 0.56rem" : "0.82rem 0.88rem",
            display: "flex",
            justifyContent: "space-between",
            gap: constrainedAppViewport ? "0.5rem" : "0.85rem",
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <button
            type="button"
            onClick={() => setAppHeaderOverlayVisible(false)}
            title="Hide source selector"
            aria-label="Hide source selector"
            style={{
              position: "absolute",
              right: constrainedAppViewport ? "0.45rem" : "0.62rem",
              top: constrainedAppViewport ? "0.42rem" : "0.55rem",
              zIndex: 2,
              width: constrainedAppViewport ? "1.45rem" : "1.65rem",
              height: constrainedAppViewport ? "1.45rem" : "1.65rem",
              borderRadius: "999px",
              border: "1px solid rgba(77, 105, 132, 0.14)",
              background: "rgba(255,255,255,0.78)",
              color: "#536b7b",
              padding: 0,
              cursor: "pointer",
              fontWeight: 800,
              fontSize: constrainedAppViewport ? "0.88rem" : "0.96rem",
              display: "grid",
              placeItems: "center",
              lineHeight: 1,
              boxShadow: "0 8px 18px rgba(38, 57, 70, 0.08)",
            }}
          >
            ×
          </button>
          <div
            style={{
              flex: "1.15 1 34rem",
              minWidth: "min(24rem, 100%)",
              borderRadius: constrainedAppViewport ? "16px" : "20px",
              border: "1px solid rgba(77, 105, 132, 0.18)",
              background: "linear-gradient(135deg, rgba(228,239,247,0.98) 0%, rgba(255,255,255,0.98) 100%)",
              padding: constrainedAppViewport ? "0.48rem 0.56rem" : "0.72rem 0.84rem",
              boxShadow: "inset 0 1px 0 rgba(255,255,255,0.62)",
              display: "flex",
              alignItems: "flex-start",
              gap: constrainedAppViewport ? "0.65rem" : "1rem",
            }}
          >
            {!constrainedAppViewport ? (
              <>
                <SmallLogo />
                <div
                  style={{
                    width: "1px",
                    alignSelf: "stretch",
                    background: "linear-gradient(180deg, rgba(109,130,145,0.08) 0%, rgba(109,130,145,0.22) 50%, rgba(109,130,145,0.08) 100%)",
                    flexShrink: 0,
                  }}
                />
              </>
            ) : null}
            <div style={{ minWidth: 0, marginLeft: "auto", display: "flex", flexDirection: "column", alignItems: "flex-end", textAlign: "right" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "0.55rem", flexWrap: "wrap" }}>
                <div style={{ fontSize: "0.74rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#60798d" }}>
                  Current source
                </div>
                <DetailPill label="Sources" value={String(sourceOptions.length || "0")} tone="source" />
              </div>
              <div
                style={{
                  marginTop: "0.2rem",
                  display: "flex",
                  alignItems: "baseline",
                  justifyContent: "flex-end",
                  gap: "0.55rem",
                  flexWrap: "wrap",
                  minWidth: 0,
                }}
              >
                <span
                  style={{
                    fontWeight: 900,
                    color: "#28475f",
                    fontSize: "1rem",
                    lineHeight: 1.2,
                    minWidth: 0,
                  }}
                >
                  {bundle?.source_label || selectedSourceOption?.source_label || sourceId || "Select a source"}
                </span>
                {!veryConstrainedAppViewport && (bundle?.source_id || selectedSourceOption?.source_id || sourceId) ? (
                  <span
                    style={{
                      color: "#617684",
                      fontSize: "0.84rem",
                      minWidth: 0,
                    }}
                    title={bundle?.source_id || selectedSourceOption?.source_id || sourceId}
                  >
                    {shortenMiddle(bundle?.source_id || selectedSourceOption?.source_id || sourceId, 56)}
                  </span>
                ) : null}
              </div>
            </div>
          </div>

          <SourcePicker
            options={sourceOptions}
            selectedSourceId={sourceId}
            onSelect={handleSourceSelect}
            disabled={sourcesLoading}
            compact={constrainedAppViewport}
          />
        </header>
      ) : null}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: mainColumns,
          gap: constrainedAppViewport ? "0.45rem" : "0.9rem",
          minHeight: 0,
        }}
      >
        {mode === "app" && !graphExpanded && screenClass === "wide" && panelVisibility.candidate ? (
          <ResizableSidebar
            side="left"
            width={candidatePanelWidth}
            minWidth={240}
            maxWidth={clamp(windowWidth * 0.3, 270, 380)}
            onWidthChange={setCandidatePanelWidth}
          >
            {candidatePanel}
          </ResizableSidebar>
        ) : null}

        {graphShell}

        {mode === "app" && !graphExpanded && screenClass !== "narrow" && rightPanels.length ? (
          <ResizableSidebar
            side="right"
            width={rightPanelWidth}
            minWidth={240}
            maxWidth={clamp(windowWidth * 0.3, 270, 390)}
            onWidthChange={setRightPanelWidth}
          >
            <div
              style={{
                display: "grid",
                gap: "0.85rem",
                minHeight: 0,
                alignContent: "start",
                overflowY: "auto",
                paddingRight: "0.1rem",
                height: "100%",
              }}
            >
              {rightPanels}
            </div>
          </ResizableSidebar>
        ) : null}
      </div>

      {showAppInspectorFooter ? (
        <footer
          style={{
            position: "relative",
            borderRadius: constrainedAppViewport ? "18px" : "26px",
            border: "1px solid rgba(70, 92, 107, 0.12)",
            background: "rgba(255,255,255,0.94)",
            boxShadow: "0 18px 36px rgba(52, 71, 84, 0.08)",
            height: inspectorHeight,
            minHeight: constrainedAppViewport ? "104px" : "132px",
            maxHeight: constrainedAppViewport ? "34vh" : "42vh",
            overflow: "hidden",
            flexShrink: 0,
          }}
        >
          <div
            onPointerDown={handleInspectorResizeStart}
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top: 0,
              height: "16px",
              cursor: "ns-resize",
              display: "grid",
              placeItems: "center",
              touchAction: "none",
            }}
          >
            <div
              style={{
                width: "4.1rem",
                height: "0.34rem",
                borderRadius: "999px",
                background: "rgba(102, 122, 137, 0.26)",
              }}
            />
          </div>
          <button
            type="button"
            onClick={() => {
              inspectorVisibilityUserSetRef.current = true;
              setAppInspectorVisible(false);
            }}
            title="Hide inspector"
            aria-label="Hide inspector"
            style={{
              position: "absolute",
              right: constrainedAppViewport ? "0.55rem" : "0.9rem",
              top: constrainedAppViewport ? "0.12rem" : "0.2rem",
              width: constrainedAppViewport ? "1.65rem" : "1.9rem",
              height: constrainedAppViewport ? "1.65rem" : "1.9rem",
              borderRadius: "999px",
              border: "1px solid rgba(70, 92, 107, 0.12)",
              background: "rgba(255,255,255,0.96)",
              color: "#48606f",
              cursor: "pointer",
              fontWeight: 800,
              display: "grid",
              placeItems: "center",
              boxShadow: "0 10px 22px rgba(38, 57, 70, 0.1)",
              backdropFilter: "blur(10px)",
              zIndex: 2,
            }}
          >
            ↓
          </button>

          <div
            style={{
              height: "100%",
              overflowY: "auto",
              padding: constrainedAppViewport ? "0.88rem 0.65rem 0.72rem" : "1rem 1rem 1rem",
              boxSizing: "border-box",
            }}
          >
            {selectedNodeInfo ? (
              <>
                <div style={{ fontSize: "0.76rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#71818d", marginBottom: "0.72rem" }}>
                  Node inspector
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: compactAppInspector
                      ? "minmax(0, 1fr)"
                      : "minmax(240px, 320px) minmax(280px, 1.2fr) minmax(240px, 0.95fr)",
                    gap: constrainedAppViewport ? "0.5rem" : "0.72rem",
                    alignItems: "start",
                  }}
                >
                  <InspectorCard title="Selected node">
                    <div style={{ fontWeight: 800, color: "#2d4351" }}>{selectedNodeInfo.node.label}</div>
                    <div style={{ marginTop: "0.5rem", display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                      <DetailPill label="Type" value={NODE_TYPE_LABELS[selectedNodeInfo.node.type]} />
                      <DetailPill label="Node kind" value={selectedNodeInfo.node.node_kind || "context"} />
                      <DetailPill label="Side" value={selectedNodeInfo.node.ontology_side || "n/a"} />
                      <DetailPill label="Expandable" value={selectedNodeInfo.expandable ? "Yes" : "No"} />
                    </div>
                  </InspectorCard>

                  <InspectorCard title="Explanation details">
                    {selectedNodeDetailItems.length ? (
                      <ul style={{ margin: 0, paddingLeft: "1rem", color: "#526570", lineHeight: 1.45 }}>
                        {selectedNodeDetailItems.slice(0, 5).map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    ) : (
                      <div style={{ color: "#6f7f8b", lineHeight: 1.45 }}>
                        No explanation-local details available for this node.
                      </div>
                    )}
                  </InspectorCard>

                  <div style={{ display: "grid", gap: "0.72rem", minWidth: 0 }}>
                    <InspectorCard title="Description">
                      {selectedNodeDefinitions.length ? (
                        <div
                          style={{
                            color: "#526570",
                            lineHeight: 1.45,
                            display: "-webkit-box",
                            WebkitLineClamp: 4,
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
                          {selectedNodeSynonyms.slice(0, 6).map((item) => (
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
                <div style={{ fontSize: "0.76rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#71818d", marginBottom: "0.72rem" }}>
                  Edge inspector
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: compactAppInspector
                      ? "minmax(0, 1fr)"
                      : "minmax(240px, 1fr) minmax(240px, 1fr) minmax(240px, 0.9fr)",
                    gap: constrainedAppViewport ? "0.5rem" : "0.72rem",
                    alignItems: "start",
                  }}
                >
                  <InspectorCard title="Source">
                    <div style={{ color: "#2d4351", fontWeight: 700, lineHeight: 1.42 }}>
                      {visibleGraph.nodes.find((node) => node.id === selectedEdge.source)?.label || selectedEdge.source}
                    </div>
                  </InspectorCard>
                  <InspectorCard title="Target">
                    <div style={{ color: "#2d4351", fontWeight: 700, lineHeight: 1.42 }}>
                      {visibleGraph.nodes.find((node) => node.id === selectedEdge.target)?.label || selectedEdge.target}
                    </div>
                  </InspectorCard>
                  <InspectorCard title="Edge details">
                    <div style={{ fontWeight: 800, color: "#2d4351", lineHeight: 1.35 }}>{selectedEdge.label}</div>
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
            ) : selectedNodeId ? (
              <>
                <div style={{ fontSize: "0.76rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#71818d", marginBottom: "0.72rem" }}>
                  Inspector
                </div>
                <div style={{ color: "#61727d", lineHeight: 1.56 }}>
                  Loading node details…
                </div>
              </>
            ) : (
              <>
                <div style={{ fontSize: "0.76rem", letterSpacing: "0.08em", textTransform: "uppercase", color: "#71818d", marginBottom: "0.72rem" }}>
                  Inspector
                </div>
                <div style={{ color: "#61727d", lineHeight: 1.56 }}>
                  Click a node or edge in the graph to inspect its label, type, score, and explanation details here.
                </div>
              </>
            )}
          </div>
        </footer>
      ) : null}

      {activeMetricKey ? (
        <MetricDialog
          title={METRIC_META[activeMetricKey].title}
          helpText={METRIC_META[activeMetricKey].help}
          metric={activeMetric}
          onClose={() => setActiveMetricKey(null)}
        />
      ) : null}
    </main>
  );
}
