/** A tab-local, session-bound outbox. Retried requests keep their exact key and revision. */
export interface Mutation {
  method: "PUT" | "POST";
  path: string;
  body: Record<string, unknown>;
  coalesce?: string;
  persist?: boolean;
  debounceMs?: number;
}
export interface QueueState { session_id: string; revision: number }
export interface PendingMutation extends Mutation { key: string; expected: number }
interface Entry extends PendingMutation { resolve: (value: QueueState) => void; reject: (error: unknown) => void }
export type QueueStatus = "saving" | "saved" | "offline";
interface Options {
  state: QueueState;
  snapshot?: string | null;
  key: () => string;
  send: (entry: PendingMutation, signal: AbortSignal) => Promise<QueueState>;
  save: (snapshot: string | null) => void;
  recover: (snapshot: string) => void;
  update: (state: QueueState) => void;
  status: (status: QueueStatus, pending: number) => void;
  retryable: (error: unknown) => boolean;
  failed: (error: unknown) => Promise<void>;
}
const MAX_ENTRIES = 32;
const MAX_BYTES = 256 * 1024;

export class MutationQueue {
  private entries: Entry[] = [];
  private running = false;
  private inFlight: Entry | null = null;
  private stopped = false;
  private controller = new AbortController();
  private wake: (() => void) | null = null;
  private revision: number;
  private options: Options;

  constructor(options: Options) {
    this.options = options;
    this.revision = options.state.revision;
    if (!options.snapshot) return;
    try {
      if (options.snapshot.length > MAX_BYTES) throw new Error("Outbox is too large");
      const stored = JSON.parse(options.snapshot);
      if (stored.session_id !== options.state.session_id || !Array.isArray(stored.entries) || stored.entries.length > MAX_ENTRIES) throw new Error("Invalid outbox");
      this.entries = stored.entries.map((entry: PendingMutation) => {
        if (!entry || !["PUT", "POST"].includes(entry.method) || typeof entry.path !== "string" || !/^\/api\/v1\/study\/(?:consent|setup|pause|resume|complete|questionnaires\/(?:background|final)|cases\/[^/]+\/(?:draft|submit|consultation))$/.test(entry.path) || !entry.body || typeof entry.body !== "object" || Array.isArray(entry.body) || typeof entry.key !== "string" || !Number.isSafeInteger(entry.expected) || entry.expected < 0) throw new Error("Invalid outbox entry");
        return { ...entry, resolve: () => undefined, reject: () => undefined };
      });
    } catch {
      this.entries = [];
      options.recover(options.snapshot);
      options.save(null);
    }
  }

  get size() { return this.entries.length; }

  observe(state: QueueState) {
    if (state.session_id === this.options.state.session_id) this.revision = Math.max(this.revision, state.revision);
  }

  private snapshot() {
    return JSON.stringify({ session_id: this.options.state.session_id, entries: this.entries.map(({ resolve: _resolve, reject: _reject, ...entry }) => entry) });
  }

  private persist() { this.options.save(this.entries.length ? this.snapshot() : null); }

  enqueue(mutation: Mutation): Promise<QueueState> {
    if (this.stopped) return Promise.reject(new Error("This study session is no longer active."));
    return new Promise((resolve, reject) => {
      // Only coalesce the last unsent entry: never reorder a draft across a submission.
      const last = this.entries.at(-1);
      const replace = !!mutation.coalesce && !!last && last !== this.inFlight && last.coalesce === mutation.coalesce;
      const expected = replace ? last!.expected : last ? last.expected + 1 : this.revision;
      const entry: Entry = { ...mutation, key: this.options.key(), expected, resolve, reject };
      if (this.entries.length >= MAX_ENTRIES && !replace) return reject(new Error("Too many changes are waiting. Reconnect before making more changes."));
      if (JSON.stringify(entry).length + this.snapshot().length > MAX_BYTES) return reject(new Error("Unsaved changes exceed the local recovery limit. Reconnect to save them."));
      if (replace) {
        this.entries[this.entries.length - 1] = entry;
        last!.reject(new Error("Replaced by a newer unsent draft."));
      } else this.entries.push(entry);
      this.persist();
      void this.start();
    });
  }

  retry() { this.wake?.(); }

  /** Stop before exchanging a different invitation. Keep unsent work in its own outbox. */
  stop() {
    this.stopped = true;
    this.controller.abort();
    this.wake?.();
    this.entries.forEach((entry) => entry.reject(new Error("This study session is no longer active.")));
  }

  async start() {
    if (this.running || this.stopped) return;
    this.running = true;
    try {
      while (this.entries.length && !this.stopped) {
        const debounce = Math.max(0, Math.min(1000, this.entries[0].debounceMs ?? 0));
        if (debounce) {
          this.options.status("saving", this.entries.length);
          await new Promise<void>((resolve) => {
            const timer = setTimeout(resolve, debounce);
            this.wake = () => { clearTimeout(timer); resolve(); };
          });
          this.wake = null;
        }
        if (this.stopped) return;
        const entry = this.entries[0];
        this.inFlight = entry;
        let delay = 1500;
        for (;;) {
          if (this.stopped) return;
          this.options.status("saving", this.entries.length);
          try {
            const next = await this.options.send(entry, this.controller.signal);
            if (this.stopped) return;
            if (next.session_id !== this.options.state.session_id) throw new Error("The active private link changed. Reopen this session's original link.");
            this.entries.shift();
            this.inFlight = null;
            this.persist();
            this.revision = Math.max(this.revision, next.revision);
            this.options.update(next);
            entry.resolve(next);
            this.options.status(this.entries.length ? "saving" : "saved", this.entries.length);
            break;
          } catch (error) {
            if (this.stopped) return;
            if (this.options.retryable(error)) {
              this.options.status("offline", this.entries.length);
              await new Promise<void>((resolve) => {
                const timer = setTimeout(resolve, delay);
                this.wake = () => { clearTimeout(timer); resolve(); };
              });
              this.wake = null;
              delay = Math.min(30_000, delay * 2);
              continue;
            }
            this.options.recover(this.snapshot());
            const rejected = this.entries.splice(0);
            this.persist();
            await this.options.failed(error);
            // Edits entered while the conflict refresh was pending still describe the
            // rejected screen. Preserve them without rebasing or issuing another write.
            if (this.entries.length) {
              this.options.recover(this.snapshot());
              rejected.push(...this.entries.splice(0));
              this.persist();
            }
            rejected.forEach((item) => item.reject(error));
            return;
          }
        }
      }
    } finally {
      this.running = false;
      this.inFlight = null;
      if (this.entries.length && !this.stopped) void this.start();
    }
  }
}
