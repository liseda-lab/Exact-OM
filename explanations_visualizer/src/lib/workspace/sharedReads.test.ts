import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import ts from "typescript";

async function load() {
  const directory = await mkdtemp(join(tmpdir(), "exact-shared-reads-"));
  const source = await readFile(new URL("./sharedReads.ts", import.meta.url), "utf8");
  const output = ts.transpileModule(source.replace('"../api"', JSON.stringify(new URL("../api.ts", import.meta.url).href)), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  const path = join(directory, "sharedReads.mjs");
  await writeFile(path, output);
  const api = await import(new URL("../api.ts", import.meta.url).href);
  return { module: await import(pathToFileURL(path).href), api, cleanup: () => rm(directory, { recursive: true }) };
}

const entity = { ontology_version_id: "o", iri: "urn:a", kind: "class" };

function fakeSource(responses: unknown[]) {
  let calls = 0;
  const source = {
    key: "k",
    entityContext: async () => {
      const value = responses[Math.min(calls, responses.length - 1)];
      calls += 1;
      if (value instanceof Error) throw value;
      return value;
    },
    explanation: async () => ({ status: "not_requested" }),
    fact: async () => ({}),
    hierarchy: async () => ({}),
    search: async () => ({}),
    evidence: async () => ({}),
  };
  return { source, calls: () => calls };
}

test("an unusable 200 response is a failure for every consumer and is refetched next time (R08)", async () => {
  const { module, cleanup } = await load();
  try {
    const { source, calls } = fakeSource([{ malformed: true }, { entity, ok: true }]);
    const shared = module.shareReads(source, { onSuspect: () => undefined, validateContext: (_: unknown, value: { ok?: boolean }) => (value.ok ? null : "missing collection") });
    const [first, second] = await Promise.allSettled([shared.entityContext(entity), shared.entityContext(entity)]);
    assert.equal(first.status, "rejected");
    assert.equal(second.status, "rejected");
    assert.equal((first as PromiseRejectedResult).reason.name, "InvalidResponseError");
    assert.equal(calls(), 1, "concurrent consumers share one read");
    const retried = await shared.entityContext(entity);
    assert.equal(retried.ok, true);
    assert.equal(calls(), 2, "the failed result was evicted, so the retry refetched");
    await shared.entityContext(entity);
    assert.equal(calls(), 2, "a validated result is reused");
  } finally {
    await cleanup();
  }
});

test("reset evicts validated results after a render failure of unknown cause (R08)", async () => {
  const { module, cleanup } = await load();
  try {
    const { source, calls } = fakeSource([{ entity, ok: true }]);
    const shared = module.shareReads(source, { onSuspect: () => undefined, validateContext: () => null });
    await shared.entityContext(entity);
    shared.reset();
    await shared.entityContext(entity);
    assert.equal(calls(), 2);
  } finally {
    await cleanup();
  }
});

test("only possible session or policy loss is reported as suspect; cursors and outages are local", async () => {
  const { module, api, cleanup } = await load();
  try {
    const suspects: number[] = [];
    for (const error of [new api.ApiError(403, "http_403", "refused"), new api.ApiError(409, "stale_cursor", "stale"), new api.ApiError(503, "http_503", "down"), new api.ApiError(401, "http_401", "signed out"), new api.ApiError(409, "http_409", "session changed")]) {
      const { source } = fakeSource([error]);
      const shared = module.shareReads(source, { onSuspect: (reason: { status: number }) => suspects.push(reason.status) });
      await shared.entityContext(entity).catch(() => undefined);
    }
    assert.deepEqual(suspects, [403, 401, 409]);
  } finally {
    await cleanup();
  }
});
