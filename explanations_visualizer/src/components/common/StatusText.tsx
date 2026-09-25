// Availability statuses in plain words. A zero-length list always travels with its reason:
// "absent in scope" is not "false", and "not exported" is not "absent".

import { IconInfo, IconWarning } from "@/components/common/Icons";
import type { Availability } from "@/lib/types";

const PHRASES: Record<string, string> = {
  available: "Available",
  absent_in_scope: "Not stated in the loaded scope",
  not_exported: "Not included in this bundle",
  unavailable_source: "Source file unavailable",
  unresolved_import: "Depends on an import that was not resolved",
  unsupported: "Not supported for this item",
  filtered: "Withheld by this view's information policy",
  partial: "Partly available",
  failed: "Failed",
  not_requested: "Not prepared",
  not_run: "Not run",
  not_recorded: "Not recorded",
};

export function statusPhrase(status: string | null | undefined): string {
  if (!status) return "Unknown";
  return PHRASES[status] ?? status.replace(/_/g, " ");
}

export function statusTone(status: string | null | undefined): "ok" | "warn" | "bad" | "neutral" {
  if (status === "available") return "ok";
  if (status === "failed") return "bad";
  if (status === "partial" || status === "unresolved_import") return "warn";
  return "neutral";
}

/** A category-specific explanation for an empty or unavailable section. */
export function EmptyReason({
  status,
  what,
  scopeNote,
  reason,
}: {
  status: Availability | string | null | undefined;
  what: string;
  scopeNote?: string;
  reason?: string | null;
}) {
  let text: string;
  switch (status) {
    case "absent_in_scope":
      text = `No ${what} is stated for this entity in the loaded scope${scopeNote ? ` (${scopeNote})` : ""}. This does not mean one exists nowhere.`;
      break;
    case "not_exported":
      text = `${capitalize(what)} ${what.endsWith("s") ? "were" : "was"} not included in this bundle.`;
      break;
    case "filtered":
      text = `${capitalize(what)} ${what.endsWith("s") ? "are" : "is"} withheld by this view's information policy.`;
      break;
    case "unsupported":
      text = `${capitalize(what)} ${what.endsWith("s") ? "are" : "is"} not supported for this kind of entity.`;
      break;
    case "not_run":
      text = `${capitalize(what)} ${what.endsWith("s") ? "were" : "was"} not computed for this bundle.`;
      break;
    case "unresolved_import":
      text = `${capitalize(what)} may come from an ontology import that was not resolved.`;
      break;
    case "partial":
      text = `Only part of the ${what} could be shown inline.`;
      break;
    case "failed":
      text = `Loading ${what} failed.`;
      break;
    default:
      text = reason || `No ${what} to show.`;
  }
  const warn = status === "unresolved_import" || status === "partial" || status === "failed";
  return (
    <p className={warn ? "note note-warn" : "note"}>
      {warn ? <IconWarning /> : <IconInfo />}
      <span>{text}</span>
    </p>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
