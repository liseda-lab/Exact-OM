"use client";

// Selected matcher evidence for one pair, joined to the original records it rests on,
// read through the active workspace (exploration API, study resource or tutorial).

import { useAsync } from "@/lib/useAsync";
import type { EvidenceBundle, PairScope } from "@/lib/workspace/types";
import { entitySignature, useWorkspace } from "@/lib/workspace/WorkspaceContext";

export type { EvidenceBundle } from "@/lib/workspace/types";

export function usePairEvidence(pair: PairScope | null) {
  const source = useWorkspace();
  const signature = pair ? `${entitySignature(pair.source)}>${entitySignature(pair.target)}|${pair.runId ?? ""}|${pair.pairId ?? ""}|${pair.candidateId ?? ""}` : null;
  return useAsync<EvidenceBundle>(signature ? `${source.key}|evidence|${signature}` : null, (signal) => source.evidence(pair!, signal));
}

export const CHANNEL_NAMES: Record<string, string> = {
  hierarchy: "Hierarchy",
  lexical: "Lexical",
  relational: "Relational",
  similarity: "Relational",
  contrastive: "Contrastive",
  difference: "Contrastive",
  attributes: "Attributes",
  attribute: "Attributes",
};

export function channelName(channel: string): string {
  return CHANNEL_NAMES[channel] ?? channel.replace(/_/g, " ");
}
