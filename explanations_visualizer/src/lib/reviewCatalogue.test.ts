import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { caseForSource, catalogueStateFromError, examplePair, IncompatibleCatalogue, landingChoice, parseCatalogue, savedChoiceFor, sourceListMode } from "./reviewCatalogue.ts";

// Labelled fixture with the live catalogue's shape (exact-review-about/1): two cases, typed
// sources and five typed targets each. The IRIs are synthetic; no React code names real ones.
const NCIT = "sha256:ncit";
const DOID = "sha256:doid";
const src = (iri: string) => ({ ontology_version_id: NCIT, iri, kind: "class" as const });
const tgt = (iri: string) => ({ ontology_version_id: DOID, iri, kind: "class" as const });
const raw = {
  schema_version: "exact-review-about/1",
  purpose: "development_review",
  title: "Provisional development review",
  source_run_id: "sha256:run",
  presented_pairs: 10,
  cases: [
    { case_id: "review-01", source: src("urn:s1"), targets: ["urn:t1", "urn:t2", "urn:t3", "urn:t4", "urn:t5"].map(tgt) },
    { case_id: "review-02", source: src("urn:s2"), targets: ["urn:t6", "urn:t7", "urn:t8", "urn:t9", "urn:t10"].map(tgt) },
  ],
};

test("the catalogue's typed cases are read; incompatible shapes fail visibly, not as empty (HR2)", () => {
  const catalogue = parseCatalogue(raw);
  assert.equal(catalogue.cases.length, 2);
  assert.deepEqual(catalogue.cases[0].source, src("urn:s1"));
  assert.equal(catalogue.source_run_id, "sha256:run");
  for (const broken of [
    { ...raw, schema_version: "exact-review-about/2" },
    { ...raw, source_run_id: undefined },
    { ...raw, cases: undefined },
    { ...raw, cases: [{ ...raw.cases[0], targets: [] }] },
    { ...raw, cases: [{ ...raw.cases[0], source: { iri: "urn:s1", kind: "class" } }] },
    { ...raw, cases: [{ ...raw.cases[0], targets: [{ ...tgt("urn:t1"), kind: "concept" }] }] },
    { ...raw, cases: [raw.cases[0], raw.cases[0]] },
  ])
    assert.throws(() => parseCatalogue(broken), IncompatibleCatalogue);
});

test("only a declared absence means ordinary browsing; other failures stay failures (HR2)", () => {
  assert.deepEqual(catalogueStateFromError({ status: 404, code: "review_bundle_not_available" }), { status: "absent" });
  assert.deepEqual(catalogueStateFromError({ status: 404, code: "not_found" }), { status: "absent" });
  const reasons = [
    [new IncompatibleCatalogue("x"), "incompatible"],
    [{ status: 409, code: "review_bundle_invalid" }, "invalid"],
    [{ status: 403, code: "http_403" }, "denied"],
    [{ status: 401, code: "http_401" }, "denied"],
    [{ status: 0, code: "network_unreachable" }, "unreachable"],
    [{ status: 500, code: "http_500" }, "error"],
    [new Error("boom"), "error"],
  ] as const;
  for (const [error, reason] of reasons) {
    const state = catalogueStateFromError(error);
    assert.equal(state.status, "failed");
    assert.equal((state as { reason: string }).reason, reason);
  }
});

test("an example opens the first candidate in matcher order whose typed target is prepared (HR2)", () => {
  const item = parseCatalogue(raw).cases[0];
  // The candidate list in the app's order: matcher rank 1 is unprepared; catalogue target order is ignored.
  const ordered = [
    { pair_id: "p-a", target: tgt("urn:unprepared") },
    { pair_id: "p-b", target: tgt("urn:t4") },
    { pair_id: "p-c", target: tgt("urn:t1") },
  ];
  assert.equal(examplePair(ordered, item)?.pair_id, "p-b");
  // Same IRI, other ontology or kind: not the prepared target.
  assert.equal(examplePair([{ pair_id: "x", target: { ...tgt("urn:t1"), ontology_version_id: "sha256:other" } }, { pair_id: "y", target: { ...tgt("urn:t1"), kind: "individual" } }], item), null);
  assert.equal(caseForSource(parseCatalogue(raw), src("urn:s2"))?.case_id, "review-02");
  assert.equal(caseForSource(parseCatalogue(raw), { ...src("urn:s2"), kind: "individual" }), null);
});

test("a saved choice belongs to one exact package and run (HR2)", () => {
  const saved = JSON.stringify({ package_id: "pkg", run_id: "sha256:run", source: "urn:s9", source_kind: "class", pair: "p9" });
  assert.deepEqual(savedChoiceFor(saved, "pkg", "sha256:run"), { package_id: "pkg", run_id: "sha256:run", source: "urn:s9", source_kind: "class", pair: "p9" });
  assert.equal(savedChoiceFor(saved, "other-package", "sha256:run"), null);
  assert.equal(savedChoiceFor(saved, "pkg", "sha256:other-run"), null);
  assert.equal(savedChoiceFor("not json", "pkg", "sha256:run"), null);
  assert.equal(savedChoiceFor(JSON.stringify({ package_id: "pkg", run_id: "sha256:run" }), "pkg", "sha256:run"), null);
});

test("landing priority: explicit URL, saved choice, prepared example, then ordinary browsing (HR2)", () => {
  const catalogue = { status: "available" as const, catalogue: parseCatalogue(raw) };
  const base: Parameters<typeof landingChoice>[0] = { explicitSource: false, explicitPair: false, saved: null, catalogue, runIds: ["sha256:run"], listMode: null, waited: false, firstSource: src("urn:first") };
  const saved = savedChoiceFor(JSON.stringify({ package_id: "pkg", run_id: "sha256:run", source: "urn:s9", source_kind: "class", pair: null }), "pkg", "sha256:run");
  assert.deepEqual(landingChoice({ ...base, explicitSource: true }), { kind: "keep" });
  assert.deepEqual(landingChoice({ ...base, explicitPair: true }), { kind: "resolve_pair" });
  assert.equal(landingChoice({ ...base, saved }).kind, "saved");
  const example = landingChoice(base);
  assert.equal(example.kind, "example");
  assert.equal((example as { item: { case_id: string } }).item.case_id, "review-01");
  // A visitor who chose All sources is not moved to an example by a late catalogue.
  assert.deepEqual(landingChoice({ ...base, listMode: "all" }), { kind: "first_source", source: src("urn:first") });
  // Loading: wait, but only for a bounded time; then ordinary browsing.
  assert.deepEqual(landingChoice({ ...base, catalogue: { status: "loading" } }), { kind: "wait" });
  assert.equal(landingChoice({ ...base, catalogue: { status: "loading" }, waited: true }).kind, "first_source");
  // Absent, failed or for another run: ordinary browsing (the failure itself is shown elsewhere).
  assert.equal(landingChoice({ ...base, catalogue: { status: "absent" } }).kind, "first_source");
  assert.equal(landingChoice({ ...base, catalogue: { status: "failed", reason: "unreachable", error: null } }).kind, "first_source");
  assert.equal(landingChoice({ ...base, runIds: ["sha256:another-run"] }).kind, "first_source");
  // Nothing loaded yet: wait rather than invent a choice.
  assert.deepEqual(landingChoice({ ...base, catalogue: { status: "absent" }, firstSource: null }), { kind: "wait" });
});

test("participant and shared workspace code never reads the review catalogue (HR2 isolation)", () => {
  const src = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const files = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((entry) => (entry.isDirectory() ? files(join(dir, entry.name)) : /\.tsx?$/.test(entry.name) && !entry.name.includes(".test.") ? [join(dir, entry.name)] : []));
  const restricted = ["components/study", "components/workspace", "study", "app/participate", "app/admin", "lib/workspace"].flatMap((dir) => files(join(src, dir)));
  assert.ok(restricted.length > 20);
  for (const file of restricted) {
    const text = readFileSync(file, "utf8");
    assert.ok(!/review-bundle|reviewCatalogue|PreparedExamples/.test(text), `${file.slice(src.length)} must not reach the review catalogue`);
  }
});


test("an examples URL keeps ordinary browsing when its catalogue cannot supply examples (HR2-A)", () => {
  const base: Parameters<typeof sourceListMode>[0] = { requested: "examples", catalogue: { status: "loading" }, hasExamples: false, hasSource: true, sourceIsExample: false, waited: false };
  assert.equal(sourceListMode(base), "examples");
  assert.equal(sourceListMode({ ...base, waited: true }), "all");
  for (const catalogue of [{ status: "absent" }, { status: "failed", reason: "unreachable", error: null }, { status: "available", catalogue: parseCatalogue({ ...raw, cases: [] }) }, { status: "available", catalogue: parseCatalogue({ ...raw, source_run_id: "other" }) }] as const)
    assert.equal(sourceListMode({ ...base, catalogue }), "all");
  assert.equal(sourceListMode({ ...base, catalogue: { status: "available", catalogue: parseCatalogue(raw) }, hasExamples: true }), "examples");
  // A deliberate choice of ordinary browsing still wins when discovery recovers.
  assert.equal(sourceListMode({ ...base, requested: "all", catalogue: { status: "available", catalogue: parseCatalogue(raw) }, hasExamples: true }), "all");
});
