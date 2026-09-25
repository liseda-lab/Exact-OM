"use client";

// Selected matcher evidence for one pair, joined to the original axioms it rests on.

import { loadAxiom } from "@/components/owl/AxiomBlock";
import { getJson } from "@/lib/api";
import type { Axiom, Page, SelectedEvidence } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";

export interface EvidenceBundle {
  items: SelectedEvidence[];
  total: number | null;
  status: string;
  reason: string | null;
  axioms: Record<string, Axiom | null>;
}

export function usePairEvidence(runId: string | null, pairId: string | null) {
  return useAsync<EvidenceBundle>(runId && pairId ? `evidence|${runId}|${pairId}` : null, async (signal) => {
    const items: SelectedEvidence[] = [];
    let cursor: string | null = null;
    let first: Page<SelectedEvidence> | null = null;
    for (let guard = 0; guard < 10; guard += 1) {
      const page: Page<SelectedEvidence> = await getJson<Page<SelectedEvidence>>(
        `/api/v1/runs/${encodeURIComponent(runId!)}/pair-evidence`,
        { pair_id: pairId!, limit: 100, cursor },
        signal,
      );
      first = first ?? page;
      items.push(...page.items);
      if (!page.next_cursor) break;
      cursor = page.next_cursor;
    }
    // Reading order: source side first, then by channel; identities stay stable.
    items.sort((a, b) => (a.side === b.side ? a.channel.localeCompare(b.channel) || a.evidence_id.localeCompare(b.evidence_id) : a.side === "source" ? -1 : 1));
    const axioms: Record<string, Axiom | null> = {};
    await Promise.all(
      items.flatMap((item) =>
        item.fact_ids.map(async (factId) => {
          try {
            axioms[factId] = await loadAxiom(item.entity.ontology_version_id, factId, signal);
          } catch {
            axioms[factId] = null;
          }
        }),
      ),
    );
    return { items, total: first?.total_count ?? items.length, status: first?.status ?? "not_exported", reason: first?.reason ?? null, axioms };
  });
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
