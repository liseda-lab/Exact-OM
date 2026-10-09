import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { eventMeaning, groupTrace, routeSummary, stageName } from "./trace.ts";

const e = (stage: string, status: string, outcome: string, reason_code: string) => ({ stage, status, outcome, reason_code });

test("the prefilter stage reads as Exact matching and its producer codes map one to one (21 F03)", () => {
  assert.equal(stageName("prefilter"), "Exact matching");
  assert.equal(eventMeaning(e("prefilter", "completed", "retained", "exact_match_protected")).text, "Matched by exact matching");
  assert.equal(eventMeaning(e("prefilter", "completed", "rejected", "exact_source_prefiltered")).text, "Another candidate was chosen by exact matching");
  assert.equal(eventMeaning(e("prefilter", "completed", "retained", "prefilter_retained")).text, "Continued to scoring");
  assert.equal(eventMeaning(e("prefilter", "not_run", "retained", "prefilter_disabled")).text, "Exact matching not used");
  assert.equal(eventMeaning(e("prefilter", "not_recorded", "unknown", "historical_stage_record_not_exported")).text, "No record in this run");
});

test("a retained event or a selected pair is never read as an exact match", () => {
  const scored = [e("retrieval", "completed", "retained", "provided_candidate_pool"), e("prefilter", "completed", "retained", "prefilter_retained"), e("selection", "completed", "selected", "selector_returned")];
  assert.equal(routeSummary(scored).route, "scored");
  assert.equal(routeSummary([e("selection", "completed", "selected", "selector_returned")]).route, "unknown");
  assert.equal(routeSummary([e("prefilter", "not_recorded", "unknown", "historical_stage_record_not_exported")]).route, "unrecorded");
});

test("an exact route explains its intentionally skipped scoring stages once", () => {
  const exact = [
    e("retrieval", "completed", "retained", "provided_candidate_pool"),
    e("prefilter", "completed", "retained", "exact_match_protected"),
    e("pair_scoring", "not_run", "not_applicable", "exact_match_protected"),
    e("llm_signal", "not_run", "not_applicable", "exact_match_protected"),
    e("selection", "not_run", "not_applicable", "exact_match_protected"),
    e("cardinality", "completed", "retained", "cardinality_retained"),
  ];
  assert.equal(routeSummary(exact).route, "exact");
  assert.match(routeSummary(exact).text, /skipped/);
  const items = groupTrace(exact);
  assert.deepEqual(items.map((item: { kind: string; reason?: string }) => item.kind === "group" ? `group:${item.reason}` : "event"), ["event", "event", "group:skipped_exact", "event"]);
  const group = items[2] as { events: { index: number }[] };
  assert.deepEqual(group.events.map((item) => item.index), [2, 3, 4], "order and positions are kept");
});

test("consecutive unrecorded historical stages collapse; failures and single stages stay visible", () => {
  const events = [
    e("retrieval", "completed", "retained", "provided_candidate_pool"),
    e("prefilter", "not_recorded", "unknown", "historical_stage_record_not_exported"),
    e("pair_scoring", "not_recorded", "unknown", "historical_stage_record_not_exported"),
    e("selection", "failed", "unknown", "selector_crashed"),
    e("threshold", "not_recorded", "unknown", "historical_stage_record_not_exported"),
  ];
  const items = groupTrace(events);
  assert.deepEqual(items.map((item: { kind: string }) => item.kind), ["event", "group", "event", "event"]);
  assert.deepEqual(eventMeaning(events[3]), { text: "Failed", known: false }, "an unknown failure keeps a generic status, its code stays technical detail");
  assert.deepEqual(eventMeaning(e("selection", "completed", "x", "some_new_code")), { text: "Completed", known: false });
});

test("removal by another exact match is distinct from no record", () => {
  const removed = [e("prefilter", "completed", "rejected", "exact_source_prefiltered"), e("pair_scoring", "not_run", "not_applicable", "prefilter_removed"), e("llm_signal", "not_run", "not_applicable", "prefilter_removed")];
  assert.equal(routeSummary(removed).route, "exact_other");
  assert.deepEqual(groupTrace(removed).map((item: { kind: string; reason?: string }) => item.kind === "group" ? item.reason : "event"), ["event", "removed_exact"]);
});
