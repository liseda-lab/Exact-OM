import type { EntityKind } from "./types";
const KINDS: EntityKind[] = ["class", "object_property", "data_property", "individual"];

export interface SavedChoice {
  package_id: string;
  run_id: string;
  source: string;
  source_kind: EntityKind;
  pair: string | null;
}

export function savedChoiceFor(raw: string | null, packageId: string | null, runId: string | null): SavedChoice | null {
  if (!raw || !packageId || !runId) return null;
  try {
    const value = JSON.parse(raw) as Partial<SavedChoice>;
    if (value.package_id !== packageId || value.run_id !== runId) return null;
    if (typeof value.source !== "string" || !value.source) return null;
    const kind = KINDS.includes(value.source_kind as EntityKind) ? (value.source_kind as EntityKind) : "class";
    return { package_id: packageId, run_id: runId, source: value.source, source_kind: kind, pair: typeof value.pair === "string" && value.pair ? value.pair : null };
  } catch {
    return null;
  }
}
