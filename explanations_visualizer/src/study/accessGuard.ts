// Decides whether a refused optional read means the case's access was really lost (19 F18,
// F20). A 403 from one panel may only be that resource's restriction, so the shell re-checks the
// scope itself. Each check belongs to the attempt current when it is ISSUED: the shared read
// source outlives retries, so the binding cannot be fixed when that source is created, and it
// must not be read when the check resolves (a late answer would then land on a newer attempt).
// Retry and teardown cancel outstanding checks; anything that still resolves for superseded
// work is ignored. Pure module (no imports) for Node's test runner.

export interface AccessGuardOptions {
  /** The attempt now current; read once, when a check is issued. */
  attempt: () => number;
  /** Whether this guard's workspace, presentation and session are still the current ones. */
  live: () => boolean;
  /** Re-reads the scope's capabilities; rejects when the scope itself is refused. */
  check: (signal: AbortSignal) => Promise<unknown>;
  /** Failures that mean access is gone (as opposed to an outage). */
  lost: (error: unknown) => boolean;
  /** Whether a suspicious read failure needs a scope check (403) or is itself a loss (401/409). */
  needsCheck: (error: unknown) => boolean;
  onLost: (attempt: number, error: unknown) => void;
}

export interface AccessGuard {
  suspect: (error: unknown) => void;
  /** Abort and forget every outstanding check (Retry, presentation or session change). */
  cancel: () => void;
  readonly pending: number;
}

export function createAccessGuard(options: AccessGuardOptions): AccessGuard {
  const outstanding = new Set<AbortController>();
  return {
    suspect(error) {
      if (!options.live()) return;
      const attempt = options.attempt();
      if (!options.needsCheck(error)) {
        options.onLost(attempt, error);
        return;
      }
      const controller = new AbortController();
      outstanding.add(controller);
      options
        .check(controller.signal)
        .then(
          () => undefined,
          (failure) => {
            // Superseded by Retry, teardown, another attempt or another workspace: ignore.
            if (controller.signal.aborted || !options.live() || options.attempt() !== attempt) return;
            if (options.lost(failure)) options.onLost(attempt, failure);
          },
        )
        .finally(() => outstanding.delete(controller));
    },
    cancel() {
      outstanding.forEach((controller) => controller.abort());
      outstanding.clear();
    },
    get pending() {
      return outstanding.size;
    },
  };
}
