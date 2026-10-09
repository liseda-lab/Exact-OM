// Availability statuses in plain words. A zero-length list always travels with its reason:
// "absent in scope" is not "false", and "not exported" is not "absent".

import { IconWarning } from "@/components/common/Icons";
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

/**
 * One short sentence for an empty or unavailable section (21 F04). Meanings stay distinct:
 * absent in this ontology, not included in this view (unprepared or a partial scope),
 * withheld by policy, not computed, and failed. Scope, import and basis details belong to the
 * panel's single scope disclosure, not to every empty category; a loading failure keeps its
 * Retry elsewhere. `partial` says the view holds only part of the ontology, so absence here
 * is not absence from the ontology.
 */
export function EmptyReason({
  status,
  what,
  partial = false,
  reason,
  absent,
}: {
  status: Availability | string | null | undefined;
  what: string;
  partial?: boolean;
  reason?: string | null;
  /** Wording for "absent" when the noun phrase does not fit (for example "No parent shown."). */
  absent?: string;
}) {
  const What = capitalize(what);
  const plural = what.endsWith("s");
  let text: string;
  switch (status) {
    case "absent_in_scope":
      text = partial ? `${What} not included in this view.` : absent ?? `No ${what} in this ontology.`;
      break;
    case "not_exported":
    case "not_requested":
      text = `${What} not included in this view.`;
      break;
    case "filtered":
      text = `${What} ${plural ? "are" : "is"} withheld by this view's information policy.`;
      break;
    case "unsupported":
      text = `Not available for this kind of entity.`;
      break;
    case "not_run":
      text = `${What} ${plural ? "were" : "was"} not computed for this bundle.`;
      break;
    case "unresolved_import":
      text = `${What} may come from an import that was not resolved.`;
      break;
    case "partial":
      text = `Only part of the ${what} is shown here.`;
      break;
    case "failed":
      text = `Loading ${what} failed.`;
      break;
    default:
      text = reason || `No ${what} to show.`;
  }
  // A view-specific reason (for example "not prepared for this study case") is more precise
  // than the generic wording; absence keeps its short form.
  if (reason && status && status !== "absent_in_scope" && status !== "available") text = reason;
  const warn = status === "unresolved_import" || status === "partial" || status === "failed";
  if (!warn) return <p className="empty-reason">{text}</p>;
  return (
    <p className="note note-warn">
      <IconWarning />
      <span>{text}</span>
    </p>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
