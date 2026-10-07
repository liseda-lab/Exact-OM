import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error Node's direct TypeScript runner requires the explicit extension.
import { createAccessGuard } from "./accessGuard.ts";

const refused = { status: 403 };
const signedOut = { status: 401 };
const outage = { status: 503 };

/** A check that ignores its abort signal and settles only when released, like a late network reply. */
function heldChecks() {
  const releases: ((error: unknown) => void)[] = [];
  return {
    check: () =>
      new Promise((_, reject) => {
        releases.push(reject);
      }),
    release: (index: number, error: unknown) => releases[index](error),
    count: () => releases.length,
  };
}

function guardFor(state: { attempt: number; live: boolean }, check: (signal: AbortSignal) => Promise<unknown>) {
  const lost: { attempt: number; error: unknown }[] = [];
  const guard = createAccessGuard({
    attempt: () => state.attempt,
    live: () => state.live,
    check,
    needsCheck: (error: unknown) => (error as { status: number }).status === 403,
    lost: (error: unknown) => [401, 403, 409].includes((error as { status: number }).status),
    onLost: (attempt: number, error: unknown) => lost.push({ attempt, error }),
  });
  return { guard, lost };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test("a check issued before a successful retry cannot block the retried attempt (R10)", async () => {
  const state = { attempt: 1, live: true };
  const held = heldChecks();
  const { guard, lost } = guardFor(state, held.check);
  guard.suspect(refused);
  guard.suspect(refused);
  assert.equal(held.count(), 2, "both refused reads re-check the scope");
  held.release(0, refused);
  await settle();
  assert.deepEqual(lost.map((item) => item.attempt), [1], "the first check blocks the attempt that issued it");
  // Retry: outstanding checks are cancelled and the attempt advances.
  guard.cancel();
  state.attempt = 2;
  held.release(1, refused);
  await settle();
  assert.deepEqual(lost.map((item) => item.attempt), [1], "the older check cannot block attempt 2");
});

test("even an uncancelled late check is bound to the attempt that issued it", async () => {
  const state = { attempt: 4, live: true };
  const held = heldChecks();
  const { guard, lost } = guardFor(state, held.check);
  guard.suspect(refused);
  state.attempt = 5;
  held.release(0, refused);
  await settle();
  assert.deepEqual(lost, []);
});

test("a genuine loss in the current attempt still blocks; an outage does not", async () => {
  const state = { attempt: 7, live: true };
  const held = heldChecks();
  const { guard, lost } = guardFor(state, held.check);
  guard.suspect(refused);
  held.release(0, refused);
  guard.suspect(refused);
  held.release(1, outage);
  guard.suspect(signedOut);
  await settle();
  assert.deepEqual(lost.map((item) => [item.attempt, (item.error as { status: number }).status]), [[7, 401], [7, 403]]);
});

test("checks from an obsolete presentation, scope or session are ignored", async () => {
  const state = { attempt: 1, live: true };
  const held = heldChecks();
  const { guard, lost } = guardFor(state, held.check);
  guard.suspect(refused);
  state.live = false; // the shell replaced this workspace (another presentation or session)
  held.release(0, refused);
  guard.suspect(signedOut);
  await settle();
  assert.deepEqual(lost, []);
  assert.equal(held.count(), 1, "no new check starts for a workspace that is no longer current");
});

test("cancel aborts the outstanding requests", async () => {
  const state = { attempt: 1, live: true };
  const signals: AbortSignal[] = [];
  const { guard } = guardFor(state, (signal) => {
    signals.push(signal);
    return new Promise(() => undefined);
  });
  guard.suspect(refused);
  guard.suspect(refused);
  assert.equal(guard.pending, 2);
  guard.cancel();
  assert.ok(signals.every((signal) => signal.aborted));
  assert.equal(guard.pending, 0);
});
