import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { ObservationStore, TelemetryQueue, type Observation, type StorageLike } from "./telemetryQueue.ts";
class Storage implements StorageLike {
  data = new Map<string, string>();
  get length() { return this.data.size; }
  key(index: number) { return [...this.data.keys()][index] ?? null; }
  getItem(key: string) { return this.data.get(key) ?? null; }
  setItem(key: string, value: string) { this.data.set(key, value); }
  removeItem(key: string) { this.data.delete(key); }
}
const event = (id: string, sequence = 0, type = "tab_open"): Observation => ({ event_id: id, page_instance_id: "page-1", presentation_id: "p1", sequence, type, client_monotonic_ms: sequence * 100 });
const segment = (id: string): Observation => ({ segment_id: id, page_instance_id: "page-1", presentation_id: "p1", stage: "case", monotonic_start_ms: 0, monotonic_end_ms: 1000 });
const failure = (status: number, message = "Rejected") => Object.assign(new Error(message), { status });
const bodies = (body: unknown) => (body as { events: Observation[] }).events;

test("one rejected tab cannot block later interactions or case timing; rejected record is retained", async () => {
  const storage = new Storage(), store = new ObservationStore("s1", storage);
  for (const [i, id] of ["ready", "bad", "candidate", "trace", "scores"].entries()) store.enqueue("event", event(id, i, id === "ready" ? "case_ready" : "tab_open"));
  store.enqueue("timing", segment("time"));
  const saved: string[] = [], timed: string[] = [];
  const queue = new TelemetryQueue(store, async (path, body) => {
    if (path.endsWith("events")) { const records = bodies(body); if (records.some((r) => r.event_id === "bad")) throw failure(422); saved.push(...records.map((r) => r.event_id!)); return { acknowledged_event_ids: records.map((r) => r.event_id) }; }
    timed.push((body as Observation).segment_id!); return { acknowledged_segment_id: (body as Observation).segment_id };
  });
  await queue.flush();
  assert.deepEqual(saved, ["ready", "candidate", "trace", "scores"]); assert.deepEqual(timed, ["time"]);
  assert.equal(store.list().length, 1); assert.equal(store.list()[0].rejected?.status, 422);
  assert.equal(new ObservationStore("s1", storage).health().rejected, 1);
});

test("transient failure and a lost acknowledgement replay identical event IDs after reload", async () => {
  const storage = new Storage(), first = new ObservationStore("s", storage);
  const original = event("original", 7); first.enqueue("event", original);
  await new TelemetryQueue(first, async () => { throw failure(503); }).flush();
  const restored = new ObservationStore("s", storage); assert.deepEqual(restored.list()[0].payload, original);
  let attempts = 0; const persisted = new Set<string>();
  const queue = new TelemetryQueue(restored, async (_, body) => { const e = bodies(body)[0]; persisted.add(e.event_id!); if (++attempts === 1) throw failure(0); return { acknowledged_event_ids: [e.event_id] }; });
  await queue.flush(); assert.equal(restored.list().length, 1);
  await queue.flush(); assert.equal(persisted.size, 1); assert.equal(restored.list().length, 0);
});

test("old page clocks and current ready events stay distinct on reload and case transition", async () => {
  const store = new ObservationStore("s", new Storage());
  store.enqueue("event", event("old-ready", 0, "case_ready"));
  const current = { ...event("new-ready", 0, "case_ready"), page_instance_id: "page-2", presentation_id: "p2" }; store.enqueue("event", current);
  store.enqueue("timing", { ...segment("new-time"), page_instance_id: "page-2", presentation_id: "p2" });
  const calls: string[] = [];
  await new TelemetryQueue(store, async (path, body) => {
    if (path.endsWith("events")) { const es = bodies(body); if (es.some((e) => e.event_id === "old-ready")) throw failure(409); calls.push("ready"); return { acknowledged_event_ids: es.map((e) => e.event_id) }; }
    calls.push("time"); return { acknowledged_segment_id: (body as Observation).segment_id };
  }).flush();
  assert.deepEqual(calls, ["ready", "time"]); assert.equal(store.list()[0].payload.page_instance_id, "page-1"); assert.equal(store.list()[0].rejected?.status, 409);
});

test("pending ready gates only its own timing, not unrelated stages", async () => {
  const store = new ObservationStore("s", new Storage()); store.enqueue("event", event("ready", 0, "case_ready")); store.enqueue("timing", segment("case")); store.enqueue("timing", { ...segment("setup"), stage: "setup", presentation_id: null });
  const sent: string[] = [];
  await new TelemetryQueue(store, async (path, body) => { if (path.endsWith("events")) throw failure(503); sent.push((body as Observation).segment_id!); return { acknowledged_segment_id: (body as Observation).segment_id }; }).flush();
  assert.deepEqual(sent, ["setup"]); assert.equal(store.list().length, 2);
});

test("one rejected timing record cannot block the next valid interval", async () => {
  const store = new ObservationStore("s", new Storage()); store.enqueue("timing", segment("bad")); store.enqueue("timing", segment("good"));
  await new TelemetryQueue(store, async (_, body) => { const id = (body as Observation).segment_id; if (id === "bad") throw failure(422); return { acknowledged_segment_id: id, availability: "observed" }; }).flush();
  assert.equal(store.list().length, 1); assert.equal(store.health().rejected, 1);
});

test("authentication or session switches retain records without quarantine or cross-session delivery", async () => {
  const storage = new Storage(), old = new ObservationStore("old", storage); old.enqueue("event", event("old"));
  for (const error of [failure(401), failure(409, "Study session changed; reopen the original invitation")]) { await new TelemetryQueue(old, async () => { throw error; }).flush(); assert.equal(old.health().rejected, 0); }
  const current = new ObservationStore("new", storage); assert.equal(current.list().length, 0);
  current.enqueue("event", event("new")); assert.deepEqual(current.list().map((r) => r.id), ["new"]);
  assert.deepEqual(old.list().map((r) => r.id), ["old"]);
});

test("storage failure remains visible and memory delivery continues", async () => {
  const storage = new Storage(); storage.setItem = () => { throw Error("quota"); };
  const store = new ObservationStore("s", storage); store.enqueue("event", event("e"));
  assert.equal(store.health().storageUnavailable, true);
  await new TelemetryQueue(store, async (_, body) => ({ acknowledged_event_ids: bodies(body).map((r) => r.event_id) })).flush();
  assert.equal(store.list().length, 0);
});

test("rejected records can be retried after the server is fixed", async () => {
  const store = new ObservationStore("s", new Storage()); store.enqueue("event", event("e")); let fixed = false;
  const queue = new TelemetryQueue(store, async (_, body) => { if (!fixed) throw failure(422); return { acknowledged_event_ids: bodies(body).map((r) => r.event_id) }; });
  await queue.flush(); assert.equal(store.health().rejected, 1); fixed = true; await queue.retry(); assert.equal(store.list().length, 0);
});

test("two tabs cannot overwrite each other's pending records and duplicate receipts are harmless", async () => {
  const storage = new Storage(), a = new ObservationStore("s", storage), b = new ObservationStore("s", storage);
  a.enqueue("event", event("a")); b.enqueue("event", { ...event("b"), page_instance_id: "another-page" });
  let release: (() => void) | undefined; const gate = new Promise<void>((resolve) => { release = resolve; });
  const queue = new TelemetryQueue(a, async (_, body) => { await gate; return { acknowledged_event_ids: bodies(body).map((r) => r.event_id) }; });
  const flush = queue.flush(); b.enqueue("event", event("new")); release!(); await flush;
  assert.deepEqual(b.list().map((r) => r.id), ["new"]);
});

test("invalid receipts cannot delete unrelated records", async () => {
  const store = new ObservationStore("s", new Storage()); store.enqueue("event", event("own"));
  await new TelemetryQueue(store, async () => ({ acknowledged_event_ids: ["other"] })).flush();
  assert.equal(store.list().length, 1); assert.equal(store.health().rejected, 0);
});

test("closure prevents an old session's split retries from continuing", async () => {
  const store = new ObservationStore("s", new Storage()); store.enqueue("event", event("e")); let calls = 0;
  const queue = new TelemetryQueue(store, async () => { calls++; queue.close(); throw failure(422); });
  await queue.flush(); assert.equal(calls, 1); assert.equal(store.health().pending, 1); assert.equal(store.health().rejected, 0);
});

test("pending delay is measured without calling it participant attention", () => {
  const store = new ObservationStore("s", new Storage()); store.enqueue("event", event("e"), 1000);
  assert.equal(store.health(30_999).delayed, false); assert.equal(store.health(31_000).delayed, true);
});

test("bounded storage reports an explicit loss instead of silently truncating the queue", () => {
  const storage = new Storage(), store = new ObservationStore("s", storage);
  assert.equal(store.enqueue("event", { ...event("too-large"), element_id: "x".repeat(2_000_001) }), false);
  assert.equal(store.health().lost, 1); assert.equal(new ObservationStore("s", storage).health().lost, 1);
  assert.equal(store.enqueue("event", event("small")), true);
});

test("a malformed persisted record does not hide valid records", () => {
  const storage = new Storage(); storage.setItem("exact.study.observations.v1.s.record.corrupt", "{");
  const store = new ObservationStore("s", storage); store.enqueue("event", event("good"));
  assert.deepEqual(store.list().map((r) => r.id), ["good"]); assert.equal(store.health().storageUnavailable, true);
});

test("a failed persisted rejection update does not override its memory quarantine", async () => {
  const storage = new Storage(), store = new ObservationStore("s", storage); store.enqueue("event", event("bad"));
  storage.setItem = () => { throw Error("quota"); };
  await new TelemetryQueue(store, async () => { throw failure(422); }).flush();
  assert.equal(store.health().rejected, 1); assert.equal(store.health().pending, 0);
});
