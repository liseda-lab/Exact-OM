// Participant-facing reading of a recorded decision trace (21 F03). Stage IDs and reason codes
// are the producer's (exact/runs/decisions.py) and are never rewritten; this module only
// chooses words for them. The internal `prefilter` stage is shown as "Exact matching". A route
// is called exact only when the producer recorded `exact_match_protected`; a retained event,
// saved membership or a score of 1 is never read as an exact match. Unknown codes keep a
// generic status with the raw code available as a technical detail. Pure module.

export interface TraceEventLike {
  stage: string;
  status: string;
  outcome: string;
  reason_code: string;
}

export const STAGE_NAMES: Record<string, string> = {
  retrieval: "Retrieval",
  prefilter: "Exact matching",
  pair_scoring: "Pair scoring",
  llm_signal: "Language-model signal",
  selection: "Selection",
  threshold: "Threshold",
  cardinality: "Cardinality",
  extraction: "Extraction",
  relation_typing: "Relation typing",
  repair: "Repair",
};

export function stageName(stage: string): string {
  return STAGE_NAMES[stage] ?? stage.replace(/_/g, " ");
}

const UNRECORDED = "historical_stage_record_not_exported";

/** Short meaning of one recorded event, and whether the code was recognised. */
export function eventMeaning(event: TraceEventLike): { text: string; known: boolean } {
  const { stage, status, outcome, reason_code: code } = event;
  // The bundle's decision log has no record of the stage: never claimed as something the run did not do.
  if (status === "not_recorded" || code === UNRECORDED) return { text: "Not recorded in this bundle", known: true };
  if (stage === "prefilter") {
    if (code === "exact_match_protected") return { text: "Matched by exact matching", known: true };
    if (code === "exact_source_prefiltered") return { text: "Another candidate was chosen by exact matching", known: true };
    if (code === "prefilter_retained") return { text: "Continued to scoring", known: true };
    if (code === "prefilter_disabled" || status === "not_run") return { text: "Exact matching not used", known: true };
  }
  if (code === "exact_match_protected") return { text: "Skipped: matched by exact matching", known: true };
  if (code === "prefilter_removed") return { text: "Not run: another candidate was chosen by exact matching", known: true };
  if (stage === "selection" && status === "completed") {
    if (outcome === "selected") return { text: "Selected among this source's candidates", known: true };
    if (outcome === "not_selected") return { text: "Not selected among this source's candidates", known: true };
    if (outcome === "abstained") return { text: "The selector abstained", known: true };
  }
  const known: Record<string, string> = {
    candidate_pool_generated: "Candidate pool generated for this run",
    provided_candidate_pool: "Taken from the saved candidate pool",
    threshold_passed: "Score met the threshold",
    below_threshold: "Score below the threshold",
    extraction_cardinality: outcome === "retained" ? "Kept by the cardinality rule" : "Not kept by the cardinality rule",
    cardinality_unbounded: "No cardinality limit applied",
    cardinality_retained: "Kept after the cardinality check",
    target_or_source_conflict: "Removed: conflicts with another mapping for the same source or target",
    not_in_cardinality_input: "Not part of the cardinality check",
    selected: "Included in the final mapping",
    extraction_omission: "Not included in the final mapping",
    default_equivalence_convention: "Relation set by the run's default equivalence convention",
    typed_output: "Relation typed by the run",
    not_in_typed_output: "No typed relation recorded",
    production_repair_not_enabled: "Repair not enabled in this run",
  };
  if (known[code]) return { text: known[code], known: true };
  if (status === "failed") return { text: "Failed", known: false };
  const STATUS: Record<string, string> = { completed: "Completed", not_run: "Not run" };
  return { text: STATUS[status] ?? status.replace(/_/g, " "), known: false };
}

export type TraceRoute = "exact" | "exact_other" | "scored" | "exact_not_used" | "unrecorded" | "unknown";

/** One sentence about how this pair travelled through the run, from recorded events only. */
export function routeSummary(events: TraceEventLike[]): { route: TraceRoute; text: string } {
  const prefilter = events.find((event) => event.stage === "prefilter");
  const skipped = events.some((event) => event.stage !== "prefilter" && event.reason_code === "exact_match_protected");
  if (events.some((event) => event.stage === "prefilter" && event.reason_code === "exact_match_protected"))
    return { route: "exact", text: skipped ? "Matched by exact matching. The scoring stages were skipped for this pair, as intended for an exact match." : "Matched by exact matching." };
  if (!prefilter) return { route: "unknown", text: "This bundle has no exact-matching record for this pair." };
  if (prefilter.reason_code === "exact_source_prefiltered") return { route: "exact_other", text: "Not scored: exact matching chose another candidate for this source." };
  if (prefilter.reason_code === "prefilter_retained") return { route: "scored", text: "Scored route: no exact match was used, and this pair continued to scoring." };
  if (prefilter.reason_code === "prefilter_disabled" || prefilter.status === "not_run") return { route: "exact_not_used", text: "Exact matching was not used in this run; the pair was scored." };
  if (prefilter.status === "not_recorded" || prefilter.reason_code === UNRECORDED) return { route: "unrecorded", text: "This bundle has no record of exact matching for this pair." };
  return { route: "unknown", text: "The exact-matching record for this pair has an unrecognised code; see the technical details." };
}

export type TraceItem<E extends TraceEventLike> =
  | { kind: "event"; event: E; index: number }
  | { kind: "group"; reason: "unrecorded" | "skipped_exact" | "removed_exact"; events: { event: E; index: number }[] };

function groupOf(event: TraceEventLike): "unrecorded" | "skipped_exact" | "removed_exact" | null {
  if (event.status === "not_recorded" || event.reason_code === UNRECORDED) return "unrecorded";
  if (event.stage !== "prefilter" && event.reason_code === "exact_match_protected") return "skipped_exact";
  if (event.reason_code === "prefilter_removed") return "removed_exact";
  return null;
}

/** Keep event order; collapse two or more consecutive unrecorded or intentionally skipped stages. */
export function groupTrace<E extends TraceEventLike>(events: E[]): TraceItem<E>[] {
  const items: TraceItem<E>[] = [];
  events.forEach((event, index) => {
    const reason = groupOf(event);
    const last = items.at(-1);
    if (reason && last?.kind === "group" && last.reason === reason) last.events.push({ event, index });
    else if (reason) items.push({ kind: "group", reason, events: [{ event, index }] });
    else items.push({ kind: "event", event, index });
  });
  // A one-event "group" is just that event.
  return items.map((item) => (item.kind === "group" && item.events.length === 1 ? { kind: "event", event: item.events[0].event, index: item.events[0].index } : item));
}

export const GROUP_TEXT: Record<"unrecorded" | "skipped_exact" | "removed_exact", string> = {
  unrecorded: "Not recorded in this bundle",
  skipped_exact: "Skipped: matched by exact matching",
  removed_exact: "Not run: another candidate was chosen by exact matching",
};
