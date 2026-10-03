import assert from "node:assert/strict";
import test from "node:test";
import { readFile, mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

async function loadSource() {
  const directory = await mkdtemp(join(tmpdir(), "exact-workspace-source-"));
  const source = await readFile(new URL("./apiSource.ts", import.meta.url), "utf8");
  const api = new URL("../api.ts", import.meta.url).href;
  const output = ts.transpileModule(source.replace('"../api"', JSON.stringify(api)), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  const path = join(directory, "apiSource.mjs");
  await writeFile(path, output);
  return { module: await import(pathToFileURL(path).href), cleanup: () => rm(directory, { recursive: true }) };
}

test("study API workspace uses case candidate evidence and session-scoped requests/cache", async () => {
  const { module, cleanup } = await loadSource();
  const originalFetch = globalThis.fetch;
  const calls: { url: string; session: string | null }[] = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    calls.push({ url, session: new Headers(init?.headers).get("X-Study-Session") });
    const value = url.includes("/evidence") ? { items: [], returned_count: 0, total_count: 0, next_cursor: null, status: "not_exported" } : { ontology_version_id: "o", availability: "available", category: "labels", ast: { type: "AnnotationAssertion" } };
    return new Response(JSON.stringify(value), { headers: { "Content-Type": "application/json" } });
  };
  try {
    const base = "/api/v1/study/workspace/scope";
    const source = module.createApiSource({ key: "a|publication|presentation", base, sessionId: "a", ontologyName: () => null });
    const entity = { ontology_version_id: "o", iri: "urn:a", kind: "class" };
    await source.evidence({ source: entity, target: entity, candidateId: "candidate" });
    assert.match(calls[0].url, /\/evidence\?candidate_id=candidate/);
    assert.equal(calls[0].session, "a");
    const ref = { factId: "fact", ontologies: ["o"], subject: entity };
    await source.fact(ref);
    await source.fact(ref);
    assert.equal(calls.length, 2, "same authorized cache namespace reuses the axiom");
    const other = module.createApiSource({ key: "b|publication|presentation", base, sessionId: "b", ontologyName: () => null });
    await other.fact(ref);
    assert.equal(calls.length, 3, "same scope locator cannot reuse another session's axiom cache");
    assert.equal(calls[2].session, "b");
    assert.deepEqual(source.remoteLabels, { key: "a|publication|presentation", base, sessionId: "a" });
    assert.equal(source.kind, "study_resource");
    assert.deepEqual(source.capabilities.bases, ["literal_asserted", "structural_navigation"]);
  } finally {
    globalThis.fetch = originalFetch;
    await cleanup();
  }
});

test("remote label batches stay inside the active participant scope and session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "exact-workspace-labels-"));
  const source = await readFile(new URL("../labels.ts", import.meta.url), "utf8");
  const output = ts.transpileModule(source.replace('"@/lib/api"', JSON.stringify(new URL("../api.ts", import.meta.url).href)).replace('"react"', JSON.stringify(import.meta.resolve("react"))), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  const path = join(directory, "labels.mjs");
  await writeFile(path, output);
  const labels = await import(pathToFileURL(path).href);
  const originalFetch = globalThis.fetch;
  const calls: string[] = [];
  globalThis.fetch = async (input, init) => {
    const session = new Headers(init?.headers).get("X-Study-Session");
    calls.push(String(input));
    return new Response(JSON.stringify({ items: [{ entity: { ontology_version_id: "o", iri: "urn:a", kind: "class" }, preferred_label: { status: "available", value: `label-${session}` } }] }));
  };
  const a = { key: "a|publication|presentation", base: "/api/v1/study/workspace/opaque", sessionId: "a" };
  const b = { key: "b|publication|presentation", base: a.base, sessionId: "b" };
  try {
    labels.requestLabels("o", ["urn:a"], a);
    labels.requestLabels("o", ["urn:a"], b);
    for (let attempt = 0; attempt < 20 && calls.length < 2; attempt += 1) await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(calls.length, 2);
    assert.ok(calls.every((url) => url.startsWith(`${a.base}/labels?`)));
    assert.equal(labels.peekLabel("o", "urn:a", a).value, "label-a");
    assert.equal(labels.peekLabel("o", "urn:a", b).value, "label-b");
    assert.equal(labels.peekLabel("o", "urn:a"), undefined, "participant labels never enter the exploration cache");
  } finally {
    globalThis.fetch = originalFetch;
    await rm(directory, { recursive: true });
  }
});
