// Continuing a server page from its own cursor (19 F16). Pure helpers (no imports) so the
// merge and count rules run under Node's test runner; the React hook lives in
// components/explore/useContinuation.ts.

/** Append a later page in the service's order, keeping earlier items and dropping repeats. */
export function appendPage<T>(existing: T[], incoming: T[], key: (item: T) => string): T[] {
  const seen = new Set(existing.map(key));
  const added: T[] = [];
  for (const item of incoming) {
    const id = key(item);
    if (seen.has(id)) continue;
    seen.add(id);
    added.push(item);
  }
  return added.length ? [...existing, ...added] : existing;
}

/**
 * How many are shown, stated without overclaiming: a known total gives "N of M"; a cursor
 * without a total gives "N shown · more exist"; only a page with neither is complete.
 */
export function shownText(shown: number, total: number | null | undefined, hasMore: boolean, noun: { one: string; many: string }): string {
  const name = (count: number) => (count === 1 ? noun.one : noun.many);
  if (total != null && (hasMore || shown < total)) return `${shown} of ${total} ${name(total)} shown`;
  if (hasMore) return `${shown} ${name(shown)} shown · more exist`;
  return `${shown} ${name(shown)}`;
}

/** True only when nothing further is recorded for this page. */
export function isComplete(shown: number, total: number | null | undefined, hasMore: boolean): boolean {
  return !hasMore && (total == null || shown >= total);
}
