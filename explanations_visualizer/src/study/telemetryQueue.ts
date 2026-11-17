/** Durable, session-scoped optional observations. No answers or invitation secrets belong here. */
export interface Observation {
  [key: string]: unknown;
  event_id?: string;
  segment_id?: string;
  page_instance_id: string;
}
export interface PendingObservation {
  id: string;
  kind: "event" | "timing";
  payload: Observation;
  queuedAt: number;
  rejected?: { status: number; at: number };
}
export interface StorageLike {
  readonly length: number;
  key(index: number): string | null;
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}
export interface DeliveryHealth {
  pending: number;
  rejected: number;
  delayed: boolean;
  storageUnavailable: boolean;
  lost: number;
}
export type DeliveryTransport = (path: string, body: unknown, keepalive: boolean) => Promise<Record<string, unknown>>;
const PREFIX = "exact.study.observations.v1.";
const LIMIT = 2500;
const MAX_BYTES = 2_000_000;

/** One key per immutable record: acknowledgements from another tab cannot erase new records. */
export class ObservationStore {
  private readonly prefix: string;
  private readonly memory = new Map<string, PendingObservation>();
  private unavailable = false;
  private lost = 0;
  private readonly storage: StorageLike | null;
  constructor(sessionId: string, storage: StorageLike | null) {
    this.storage = storage;
    this.prefix = `${PREFIX}${encodeURIComponent(sessionId)}.`;
    this.unavailable = !storage;
    try { this.lost = Number(storage?.getItem(this.prefix + "loss") ?? 0) || 0; } catch { this.unavailable = true; }
  }
  list(): PendingObservation[] {
    const records = new Map<string, PendingObservation>();
    try {
      if (this.storage) for (let i = 0; i < this.storage.length; i++) {
        const key = this.storage.key(i);
        if (!key?.startsWith(this.prefix + "record.")) continue;
        const raw = this.storage.getItem(key);
        if (!raw) continue;
        let record: PendingObservation;
        try { record = JSON.parse(raw) as PendingObservation; } catch { this.unavailable = true; continue; }
        if (!record?.id || !["event", "timing"].includes(record.kind) || !record.payload?.page_instance_id || !Number.isFinite(record.queuedAt)) continue;
        if (key !== this.recordKey(record.id)) continue;
        records.set(record.id, record);
      }
    } catch { this.unavailable = true; }
    for (const [id, record] of this.memory) records.set(id, record);
    return [...records.values()].sort((a, b) => a.queuedAt - b.queuedAt || (a.payload.page_instance_id === b.payload.page_instance_id ? Number(a.payload.sequence ?? 0) - Number(b.payload.sequence ?? 0) : a.id.localeCompare(b.id)));
  }
  private recordKey(id: string) { return this.prefix + "record." + encodeURIComponent(id); }
  private put(record: PendingObservation) {
    try {
      if (!this.storage) throw new Error("Storage unavailable");
      this.storage.setItem(this.recordKey(record.id), JSON.stringify(record));
      this.memory.delete(record.id);
    } catch { this.unavailable = true; this.memory.set(record.id, record); }
  }
  enqueue(kind: PendingObservation["kind"], payload: Observation, now = Date.now()): boolean {
    const id = kind === "event" ? payload.event_id : payload.segment_id;
    if (!id) throw new Error("Observation requires a stable identity");
    const existing = this.list();
    if (existing.some((r) => r.id === id)) return true;
    const record = { id, kind, payload, queuedAt: now };
    if (existing.length >= LIMIT || existing.reduce((sum, r) => sum + JSON.stringify(r).length, 0) + JSON.stringify(record).length > MAX_BYTES) {
      this.lost++;
      try { this.storage?.setItem(this.prefix + "loss", String(this.lost)); } catch { this.unavailable = true; }
      return false;
    }
    this.put(record);
    return true;
  }
  acknowledge(id: string) {
    this.memory.delete(id);
    try { this.storage?.removeItem(this.recordKey(id)); } catch { this.unavailable = true; }
  }
  reject(record: PendingObservation, status: number) { this.put({ ...record, rejected: { status, at: Date.now() } }); }
  retryRejected() {
    for (const record of this.list()) {
      if (record.rejected) { const next = { ...record }; delete next.rejected; this.put(next); }
    }
  }
  health(now = Date.now()): DeliveryHealth {
    const records = this.list();
    const pending = records.filter((r) => !r.rejected);
    return { pending: pending.length, rejected: records.length - pending.length, delayed: pending.some((r) => now - r.queuedAt >= 30_000), storageUnavailable: this.unavailable, lost: this.lost };
  }
}

function statusOf(error: unknown): number { return Number((error as { status?: number })?.status ?? 0); }
function isolates(error: unknown): boolean {
  // Authentication and cookie/session changes are global failures, not bad observations.
  if (/Study session changed/i.test((error as Error)?.message ?? "")) return false;
  return [400, 403, 409, 413, 422].includes(statusOf(error));
}

export class TelemetryQueue {
  private running: Promise<void> | null = null;
  private enabled = true;
  readonly store: ObservationStore;
  private readonly send: DeliveryTransport;
  private readonly changed: () => void;
  constructor(store: ObservationStore, send: DeliveryTransport, changed: () => void = () => {}) {
    this.store = store; this.send = send; this.changed = changed;
  }
  close() { this.enabled = false; }
  enqueue(kind: PendingObservation["kind"], value: Observation) { const saved = this.store.enqueue(kind, value); this.changed(); return saved; }
  retry() { this.store.retryRejected(); this.changed(); return this.flush(); }
  flush(keepalive = false): Promise<void> {
    if (this.running) return this.running;
    if (!this.enabled) return Promise.resolve();
    const work = this.deliver(keepalive).catch(() => { /* Network/contract failures retain the outbox. */ });
    this.running = work;
    void work.finally(() => { if (this.running === work) this.running = null; this.changed(); });
    return work;
  }
  private async deliver(keepalive: boolean) {
    let calls = 0;
    let stopped = false;
    const request = async (path: string, body: unknown) => {
      if (!this.enabled || calls++ >= 50) throw new Error("Delivery budget exhausted");
      return this.send(path, body, keepalive);
    };
    const sendEvents = async (records: PendingObservation[]): Promise<void> => {
      if (!records.length || !this.enabled || stopped || calls >= 50) return;
      // Leave room under the browser's 64 KB keepalive budget; split by bytes as well as count.
      if (records.length > 1 && JSON.stringify({ events: records.map((r) => r.payload) }).length > 40_000) {
        const middle = Math.ceil(records.length / 2);
        await sendEvents(records.slice(0, middle)); await sendEvents(records.slice(middle)); return;
      }
      try {
        const result = await request("/api/v1/study/events", { events: records.map((r) => r.payload) });
        const ids = result.acknowledged_event_ids;
        if (!Array.isArray(ids) || !ids.length || ids.some((id) => typeof id !== "string" || !records.some((r) => r.id === id))) throw new Error("Invalid event receipt");
        for (const id of ids) this.store.acknowledge(id as string);
        if (ids.length !== records.length) stopped = true;
      } catch (error) {
        if (!this.enabled) return;
        if (isolates(error)) {
          if (records.length === 1) this.store.reject(records[0], statusOf(error));
          else { const middle = Math.ceil(records.length / 2); await sendEvents(records.slice(0, middle)); await sendEvents(records.slice(middle)); }
        } else stopped = true;
      }
    };
    // A fixed snapshot keeps ongoing clicks from starving timing uploads.
    const pendingEvents = this.store.list().filter((r) => r.kind === "event" && !r.rejected).slice(0, 200);
    const size = keepalive ? 20 : 100;
    for (let i = 0; i < pendingEvents.length && !stopped && calls < 40; i += size) await sendEvents(pendingEvents.slice(i, i + size));
    if (!this.enabled) return;
    for (const record of this.store.list().filter((r) => r.kind === "timing" && !r.rejected).slice(0, 20)) {
      if (!this.enabled || calls >= 50) break;
      // Only this page/presentation's unacknowledged ready event gates its case timing.
      const waitingForReady = record.payload.stage === "case" && this.store.list().some((r) => r.kind === "event" && !r.rejected && r.payload.type === "case_ready" && r.payload.page_instance_id === record.payload.page_instance_id && r.payload.presentation_id === record.payload.presentation_id && Number(r.payload.client_monotonic_ms) <= Number(record.payload.monotonic_start_ms));
      if (waitingForReady) continue;
      try {
        const result = await request("/api/v1/study/timing", record.payload);
        if (result.acknowledged_segment_id !== record.id) throw new Error("Invalid timing receipt");
        this.store.acknowledge(record.id);
      } catch (error) {
        if (!this.enabled) return;
        if (isolates(error)) this.store.reject(record, statusOf(error));
        else break;
      }
    }
  }
}
