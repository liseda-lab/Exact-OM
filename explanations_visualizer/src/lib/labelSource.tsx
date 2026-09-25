"use client";

// Where display labels come from. The exploration app asks the read API; the study uses
// only the frozen, policy-filtered facts it was given, so it never calls exploration routes.

import { createContext, useContext } from "react";

import { useLabels, type LabelEntry } from "@/lib/labels";

export type LabelLookup = (iri: string) => LabelEntry | undefined;

export const LabelSourceContext = createContext<((ontology: string) => LabelLookup) | null>(null);

/** Returns a label lookup for one ontology from the active source. */
export function useLabelLookup(ontology: string | null | undefined, iris: string[]): LabelLookup {
  const override = useContext(LabelSourceContext);
  const remote = useLabels(override ? null : ontology, override ? [] : iris);
  if (override && ontology) return override(ontology);
  return remote;
}
