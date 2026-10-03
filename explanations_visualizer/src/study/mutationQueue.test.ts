import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { MutationQueue, OUTBOX_ROUTE, type PendingMutation, type QueueState } from "./mutationQueue.ts";

const state = { session_id: "session-a", revision: 0 };
const draft = (value: string) => ({ method: "PUT" as const, path: "/api/v1/study/cases/c/draft", body: { value }, coalesce: "draft:c" });
const tick = () => new Promise((resolve) => setImmediate(resolve));
function fixture(send: (entry: PendingMutation, signal: AbortSignal) => Promise<QueueState>, snapshot?: string | null, initial = state) {
  let saved: string | null = snapshot ?? null;
  let recovery: string | null = null;
  let count = 0;
  const updates: QueueState[] = [];
  const queue = new MutationQueue({ state: initial, snapshot, key: () => `key-${++count}`, send,
    save: (value) => { saved = value; }, recover: (value) => { recovery = value; }, update: (value) => updates.push(value),
    status: () => undefined, retryable: (error) => error === "offline", failed: async () => undefined });
  return { queue, updates, saved: () => saved, recovery: () => recovery };
}

test("reload replays a lost acknowledgement before the newest coalesced draft", async () => {
  const first = fixture(() => new Promise(() => undefined));
  void first.queue.enqueue(draft("first")).catch(() => undefined);
  void first.queue.enqueue(draft("obsolete")).catch(() => undefined);
  void first.queue.enqueue(draft("latest")).catch(() => undefined);
  const stored = JSON.parse(first.saved()!);
  assert.equal(stored.entries.length, 2);
  assert.equal(stored.entries[1].body.value, "latest");
  assert.deepEqual(stored.entries.map((entry: PendingMutation) => entry.expected), [0, 1]);
  first.queue.stop();
  const sent: PendingMutation[] = [];
  const restored = fixture(async (entry) => { sent.push(entry); return { ...state, revision: entry.expected + 1 }; }, first.saved(), { ...state, revision: 1 });
  await restored.queue.start();
  assert.deepEqual(sent.map((entry) => entry.key), stored.entries.map((entry: PendingMutation) => entry.key));
  assert.equal(sent[1].body.value, "latest");
  assert.equal(restored.saved(), null, "restored mutations must also clear the recovery outbox after acknowledgement");
});

test("a lost acknowledgement retries the exact key, revision and body", async () => {
  const sent: PendingMutation[] = [];
  const f = fixture(async (entry) => { sent.push({ ...entry }); if (sent.length === 1) throw "offline"; return { ...state, revision: 1 }; });
  const saved = f.queue.enqueue(draft("answer"));
  await tick();
  f.queue.retry();
  await saved;
  assert.deepEqual(sent[0], sent[1]);
});

test("changing invitation stops retries and ignores a late old-session acknowledgement", async () => {
  let finish!: (value: QueueState) => void;
  let signal!: AbortSignal;
  const f = fixture((_entry, currentSignal) => { signal = currentSignal; return new Promise((resolve) => { finish = resolve; }); });
  const result = f.queue.enqueue(draft("old answer")).catch((error) => error);
  f.queue.stop();
  assert.equal(signal.aborted, true);
  finish({ ...state, revision: 1 });
  await result;
  await tick();
  assert.equal(f.updates.length, 0);
  assert.ok(f.saved(), "the original session retains its own recoverable outbox");
});

test("a conflict never rebases later edits and retains them for explicit recovery", async () => {
  let fail!: (error: unknown) => void;
  const f = fixture(() => new Promise((_resolve, reject) => { fail = reject; }));
  const first = f.queue.enqueue(draft("first")).catch((error) => error);
  const second = f.queue.enqueue(draft("latest")).catch((error) => error);
  fail(new Error("stale revision"));
  await Promise.all([first, second]);
  assert.equal(f.queue.size, 0);
  assert.equal(JSON.parse(f.recovery()!).entries[1].body.value, "latest");
  assert.equal(f.saved(), null);
});

test("the outbox rejects a different session and bounds persisted input", () => {
  const other = fixture(async () => state, JSON.stringify({ session_id: "session-b", entries: [{ ...draft("other") }] }));
  assert.equal(other.queue.size, 0);
  assert.ok(other.recovery());
  const huge = fixture(async () => state, "x".repeat(256 * 1024 + 1));
  assert.equal(huge.queue.size, 0);
});

test("a pending submission cannot be replaced by a later draft", async () => {
  const f = fixture(() => new Promise(() => undefined));
  void f.queue.enqueue(draft("first")).catch(() => undefined);
  void f.queue.enqueue({ method: "POST", path: "/api/v1/study/cases/c/submit", body: { value: "submit" } }).catch(() => undefined);
  void f.queue.enqueue(draft("later")).catch(() => undefined);
  assert.deepEqual(JSON.parse(f.saved()!).entries.map((entry: PendingMutation) => entry.path), [draft("").path, "/api/v1/study/cases/c/submit", draft("").path]);
  f.queue.stop();
});

test("debounced edits are persisted immediately and complete before Pause", async () => {
  const sent: PendingMutation[] = [];
  const f = fixture(async (entry) => { sent.push(entry); return { ...state, revision: entry.expected + 1 }; });
  void f.queue.enqueue({ ...draft("obsolete"), debounceMs: 1000 }).catch(() => undefined);
  const latest = f.queue.enqueue({ ...draft("latest before pause"), debounceMs: 1000 });
  const paused = f.queue.enqueue({ method: "POST", path: "/api/v1/study/pause", body: {} });
  assert.equal(sent.length, 0, "debouncing must not delay persistence");
  const saved = JSON.parse(f.saved()!).entries;
  assert.equal(saved.length, 2);
  assert.equal(saved[0].body.value, "latest before pause");
  f.queue.retry();
  await Promise.all([latest, paused]);
  assert.deepEqual(sent.map((entry) => entry.path), [draft("").path, "/api/v1/study/pause"]);
  assert.deepEqual(sent.map((entry) => entry.expected), [0, 1]);
  assert.equal(f.saved(), null);
});

test("edits during a conflict refresh are archived without retrying against stale state", async () => {
  let rejectWrite!: (error: unknown) => void;
  let finishRefresh!: () => void;
  let writes = 0;
  const archives: string[] = [];
  const queue = new MutationQueue({ state, key: () => `key-${Math.random()}`, send: async () => { writes += 1; return new Promise((_resolve, reject) => { rejectWrite = reject; }); },
    save: () => undefined, recover: (value) => archives.push(value), update: () => undefined, status: () => undefined,
    retryable: () => false, failed: () => new Promise((resolve) => { finishRefresh = resolve; }) });
  const first = queue.enqueue(draft("original conflict")).catch((error) => error);
  rejectWrite(new Error("conflict"));
  await tick();
  const duringRefresh = queue.enqueue(draft("typed during refresh")).catch((error) => error);
  finishRefresh();
  await Promise.all([first, duringRefresh]);
  assert.equal(writes, 1);
  assert.deepEqual(archives.map((value) => JSON.parse(value).entries[0].body.value), ["original conflict", "typed during refresh"]);
  assert.equal(queue.size, 0);
});

test("the outbox accepts the planned v2 participant routes and nothing else (F8)", () => {
  for (const path of ["/api/v1/study/tutorial/progress", "/api/v1/study/tutorial/assessment", "/api/v1/study/tutorial/complete", "/api/v1/study/cases/c-1/consultation/draft", "/api/v1/study/cases/c-1/consultation", "/api/v1/study/setup"]) {
    assert.ok(OUTBOX_ROUTE.test(path), path);
  }
  for (const path of ["/api/v1/study/admin", "/api/v1/admin/studies/x", "/api/v1/study/tutorial/grade", "/api/v1/study/cases/c/consultation/draft/x", "/api/v1/study/resources/r"]) {
    assert.equal(OUTBOX_ROUTE.test(path), false, path);
  }
});

test("a draft never coalesces across an assessment attempt or a final consultation save", async () => {
  const f = fixture(() => new Promise(() => undefined));
  const progress = (value: string) => ({ method: "PUT" as const, path: "/api/v1/study/tutorial/progress", body: { value }, coalesce: "tutorial:item:q1" });
  void f.queue.enqueue(progress("draft one")).catch(() => undefined);
  void f.queue.enqueue({ method: "POST", path: "/api/v1/study/tutorial/assessment", body: { attempt_id: "a1" } }).catch(() => undefined);
  void f.queue.enqueue(progress("draft two")).catch(() => undefined);
  void f.queue.enqueue(progress("draft three")).catch(() => undefined);
  const stored = JSON.parse(f.saved()!);
  assert.deepEqual(stored.entries.map((entry: PendingMutation) => entry.body.value ?? entry.body.attempt_id), ["draft one", "a1", "draft three"]);
  assert.deepEqual(stored.entries.map((entry: PendingMutation) => entry.expected), [0, 1, 2]);
  f.queue.stop();
});
