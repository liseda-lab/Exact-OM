import type { SavedChoice } from "./sourceSelection";
import type { EntityKind, EntityRef, PairTrace, SourceSummary } from "./types";

type ParamsUpdate = Record<string, string | null>;
type LinkedPair = Pick<PairTrace, "run_id" | "pair_id" | "source" | "ranking">;
type Focus = Pick<EntityRef, "iri" | "kind">;
const VIEW_KEYS = ["sk", "q", "list", "details", "hs", "hsk", "ht", "htk"];

function entityKind(value: string | null): EntityKind {
  return value === "object_property" || value === "data_property" || value === "individual" ? value : "class";
}

export function choiceParams(choice: Pick<SavedChoice, "source" | "source_kind" | "query_id" | "pair">): ParamsUpdate {
  return { source: choice.source, sk: choice.source_kind, q: choice.query_id?.toString() ?? null, pair: choice.pair };
}

export function sourceParams(source: SourceSummary): ParamsUpdate {
  return {
    ...choiceParams({ source: source.entity.iri, source_kind: source.entity.kind, query_id: source.query_id, pair: null }),
    hs: null, hsk: null, ht: null, htk: null,
  };
}

export function needsLinkedPair(params: URLSearchParams): boolean {
  return Boolean(params.get("pair") && (!params.get("source") || params.get("q") === null));
}

/** Pair identity resolves incomplete links before any source's candidates are requested. */
export function resolveComparison(params: URLSearchParams, sources: SourceSummary[], runId: string | null, pair?: LinkedPair) {
  const needsPair = needsLinkedPair(params);
  const linked = needsPair && pair?.pair_id === params.get("pair") && pair?.run_id === runId ? pair : undefined;
  const waitingForPair = needsPair && !linked;
  const sourceIri = linked?.source.iri ?? params.get("source");
  const requestedKind = linked?.source.kind ?? params.get("sk");
  const requestedQuery = linked ? linked.ranking?.query_id.toString() ?? null : params.get("q");
  const invalidQuery = requestedQuery !== null && (!/^\d+$/.test(requestedQuery) || !Number.isSafeInteger(Number(requestedQuery)));
  const parsedQuery = requestedQuery !== null && !invalidQuery ? Number(requestedQuery) : null;
  const source = waitingForPair || invalidQuery ? null : sources.find((item) =>
    item.entity.iri === sourceIri &&
    (!requestedKind || item.entity.kind === entityKind(requestedKind)) &&
    (requestedQuery === null || item.query_id === parsedQuery),
  ) ?? null;
  const sourceKind = requestedKind ? entityKind(requestedKind) : source?.entity.kind ?? "class";
  const queryId = parsedQuery ?? (waitingForPair || invalidQuery ? null : source?.query_id ?? null);
  let updates: ParamsUpdate | null = null;
  if (!waitingForPair && !invalidQuery && sourceIri && (linked || source)) {
    const resolved = { source: sourceIri, sk: sourceKind, q: queryId?.toString() ?? null };
    if (Object.entries(resolved).some(([key, value]) => params.get(key) !== value)) updates = resolved;
  }
  return { sourceIri, sourceKind, queryId, source, waitingForPair, invalidQuery, updates };
}

export function sourceIsSelected(source: SourceSummary, iri: string | null, kind: EntityKind, queryId: number | null): boolean {
  return source.entity.iri === iri && source.entity.kind === kind && (source.query_id ?? null) === queryId;
}

/** Keep the comparison's query and view separate from the browser's independent focus. */
export function comparisonBrowseParams(params: URLSearchParams, selection: {
  runId: string | null;
  source: EntityRef | null;
  target: EntityRef | null;
  queryId: number | null;
  sourceFocus: Focus | null;
  targetFocus: Focus | null;
}): URLSearchParams {
  const { runId, source, target, queryId, sourceFocus, targetFocus } = selection;
  const next = new URLSearchParams();
  if (runId) next.set("run", runId);
  if (source) {
    next.set("s", sourceFocus?.iri ?? source.iri);
    next.set("sk", sourceFocus?.kind ?? source.kind);
    next.set("pin_s", source.iri);
    next.set("pin_sk", source.kind);
    next.set("from_source", source.iri);
    next.set("from_sk", source.kind);
  }
  if (target) {
    next.set("t", targetFocus?.iri ?? target.iri);
    next.set("tk", targetFocus?.kind ?? target.kind);
    next.set("pin_t", target.iri);
    next.set("pin_tk", target.kind);
  }
  for (const key of ["source", "pair", ...VIEW_KEYS]) {
    const value = params.get(key);
    if (value) next.set(`from_${key}`, value);
  }
  // The resolved query also covers source-only links before their URL is canonicalized.
  if (queryId !== null) next.set("from_q", String(queryId));
  return next;
}

export function comparisonBackParams(params: URLSearchParams, runId: string | null, source: string, pair: string | null): URLSearchParams {
  const next = new URLSearchParams();
  if (runId) next.set("run", runId);
  next.set("source", source);
  if (pair) next.set("pair", pair);
  for (const key of VIEW_KEYS) {
    const value = params.get(`from_${key}`);
    if (value) next.set(key, value);
  }
  return next;
}
