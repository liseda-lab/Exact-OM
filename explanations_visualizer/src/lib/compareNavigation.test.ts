import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { choiceParams, comparisonBackParams, comparisonBrowseParams, needsLinkedPair, resolveComparison, sourceIsSelected, sourceParams } from "./compareNavigation.ts";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { savedChoiceFor } from "./sourceSelection.ts";
import type { EntityRef, PairTrace, SourceSummary } from "./types";

const entity: EntityRef = { ontology_version_id: "ontology-source", iri: "urn:repeated-source", kind: "class" };
const sources: SourceSummary[] = [
  { entity_id: "same-entity", entity, query_id: 0, candidate_count: 2 },
  { entity_id: "same-entity", entity, query_id: 9, candidate_count: 1 },
  { entity_id: "same-entity", entity, query_id: 10, candidate_count: 0 },
  { entity_id: "property", entity: { ...entity, kind: "object_property" }, query_id: 11, candidate_count: 1 },
];
const linked: Pick<PairTrace, "run_id" | "pair_id" | "source" | "ranking"> = {
  run_id: "run", pair_id: "pair-nine", source: entity,
  ranking: { mode: "local_ranking", query_id: 9, rank: 1, candidate_count: 1, saved_output_member: true, artifact_ref: "saved-ranking" },
};
function apply(params: URLSearchParams, updates: Record<string, string | null> | null): URLSearchParams {
  const next = new URLSearchParams(params);
  for (const [key, value] of Object.entries(updates ?? {})) {
    if (value === null) next.delete(key);
    else next.set(key, value);
  }
  return next;
}

test("repeated sources remain distinct by query and typed entity, including query zero", () => {
  for (const selected of sources) {
    const params = apply(new URLSearchParams("run=run"), sourceParams(selected));
    const result = resolveComparison(params, sources, "run");
    assert.equal(result.source, selected);
    assert.equal(result.queryId, selected.query_id);
    assert.equal(result.updates, null);
    assert.deepEqual(sources.filter((source) => sourceIsSelected(source, result.sourceIri, result.sourceKind, result.queryId)), [selected]);
  }
});

test("pair-only and source-plus-pair links wait for pair identity instead of choosing the first query", () => {
  for (const search of ["run=run&pair=pair-nine", "run=run&source=urn:repeated-source&pair=pair-nine"]) {
    const params = new URLSearchParams(search);
    assert.equal(needsLinkedPair(params), true);
    const pending = resolveComparison(params, sources, "run");
    assert.equal(pending.waitingForPair, true);
    assert.equal(pending.source, null);
    assert.equal(pending.queryId, null);
    assert.equal(pending.updates, null);
    const resolved = resolveComparison(params, sources, "run", linked);
    assert.equal(resolved.waitingForPair, false);
    assert.equal(resolved.source, sources[1]);
    assert.equal(resolved.queryId, 9);
    const restored = apply(params, resolved.updates);
    assert.equal(restored.get("source"), entity.iri);
    assert.equal(restored.get("sk"), "class");
    assert.equal(restored.get("q"), "9");
    assert.equal(restored.get("pair"), "pair-nine");
    assert.equal(needsLinkedPair(restored), false);
    assert.equal(resolveComparison(restored, sources, "run").source, sources[1]);
  }
});

test("stale linked responses from another pair or run cannot choose a query", () => {
  const params = new URLSearchParams("run=run&source=urn:repeated-source&pair=pair-nine");
  for (const stale of [{ ...linked, pair_id: "old-pair" }, { ...linked, run_id: "old-run" }]) {
    const resolved = resolveComparison(params, sources, "run", stale);
    assert.equal(resolved.waitingForPair, true);
    assert.equal(resolved.updates, null);
  }
});

test("an explicit query is authoritative, including when its source page is not loaded", () => {
  const params = new URLSearchParams("run=run&source=urn:repeated-source&sk=object_property&q=11&pair=explicit-pair");
  const resolved = resolveComparison(params, [], "run", linked);
  assert.equal(resolved.waitingForPair, false);
  assert.equal(resolved.sourceKind, "object_property");
  assert.equal(resolved.queryId, 11);
  assert.equal(resolved.source, null);
  assert.equal(resolved.updates, null);
});

test("source-only links acquire query identity and empty queries stay selected", () => {
  const params = new URLSearchParams("run=run&source=urn:repeated-source");
  const first = resolveComparison(params, sources, "run");
  assert.equal(first.queryId, 0);
  assert.equal(first.updates?.q, "0");
  const previous = apply(params, { pair: "pair-nine", q: "9", hs: "urn:focus", ht: "urn:target-focus" });
  const emptyParams = apply(previous, sourceParams(sources[2]));
  const empty = resolveComparison(emptyParams, sources, "run");
  assert.equal(empty.queryId, 10);
  assert.equal(empty.source?.candidate_count, 0);
  assert.equal(emptyParams.has("pair"), false);
  assert.equal(emptyParams.has("hs"), false);
  assert.equal(emptyParams.has("ht"), false);
  assert.equal(empty.waitingForPair, false);
  assert.equal(empty.updates, null);
});

test("malformed explicit queries never fall back to another occurrence", () => {
  for (const q of ["", "-1", "1.5", "NaN", " 9", "1e2", "9007199254740992"]) {
    const params = new URLSearchParams({ source: entity.iri, q });
    const result = resolveComparison(params, sources, "run");
    assert.equal(result.invalidQuery, true, q);
    assert.equal(result.source, null);
    assert.equal(result.updates, null);
  }
});

test("browse and back restore the full comparison while browser focuses change independently", () => {
  const params = new URLSearchParams({ run: "run", source: entity.iri, sk: "class", q: "0", pair: "pair-zero", list: "all", details: "scores", hs: "urn:source-focus", hsk: "object_property", ht: "urn:target-focus", htk: "individual" });
  const browse = comparisonBrowseParams(params, {
    runId: "run", source: entity, target: { ...entity, iri: "urn:target", ontology_version_id: "ontology-target" }, queryId: 0,
    sourceFocus: { iri: "urn:source-focus", kind: "object_property" }, targetFocus: { iri: "urn:target-focus", kind: "individual" },
  });
  assert.equal(browse.get("from_q"), "0");
  assert.equal(browse.get("s"), "urn:source-focus");
  assert.equal(browse.get("sk"), "object_property");
  assert.equal(browse.get("pin_s"), entity.iri);
  browse.set("s", "urn:other-source");
  browse.set("sk", "data_property");
  browse.set("t", "urn:other-target");
  browse.set("q", "999");
  const back = comparisonBackParams(browse, "run", browse.get("from_source")!, browse.get("from_pair"));
  assert.deepEqual(Object.fromEntries(back), Object.fromEntries(params));
  assert.equal(resolveComparison(back, sources, "run").source, sources[0]);
});

test("browse preserves a resolved query even before it has reached the URL", () => {
  const params = new URLSearchParams({ source: entity.iri });
  const browse = comparisonBrowseParams(params, { runId: "run", source: entity, target: null, queryId: 0, sourceFocus: null, targetFocus: null });
  const back = comparisonBackParams(browse, "run", entity.iri, null);
  assert.equal(back.get("q"), "0");
  assert.equal(back.has("pair"), false);
});

test("persisted query choices restore the same occurrence, even without candidates", () => {
  for (const source of sources) {
    const saved = savedChoiceFor(JSON.stringify({ package_id: "pkg", run_id: "run", source: source.entity.iri, source_kind: source.entity.kind, query_id: source.query_id, pair: null }), "pkg", "run");
    assert.ok(saved);
    const params = apply(new URLSearchParams("run=run"), choiceParams(saved));
    assert.equal(resolveComparison(params, sources, "run").source, source);
  }
});

test("legacy saved pairs without queries resolve via the pair while unranked runs stay query-free", () => {
  const saved = savedChoiceFor(JSON.stringify({ package_id: "pkg", run_id: "run", source: entity.iri, source_kind: "class", pair: "pair-nine" }), "pkg", "run");
  assert.ok(saved);
  const params = apply(new URLSearchParams("run=run"), choiceParams(saved));
  assert.equal(resolveComparison(params, sources, "run").waitingForPair, true);
  assert.equal(resolveComparison(params, sources, "run", linked).queryId, 9);
  const legacy = resolveComparison(params, [{ entity_id: "legacy", entity }], "run", { ...linked, ranking: undefined });
  assert.equal(legacy.waitingForPair, false);
  assert.equal(legacy.queryId, null);
  assert.equal(legacy.updates, null);
});
