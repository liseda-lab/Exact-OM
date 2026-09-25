"use client";

// Prepared generated text, visibly distinct from original ontology statements. Only
// explanations that passed grounding are shown; every claim links to its cited facts.

import { useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { ApiError, getJson } from "@/lib/api";
import type { Claim, ExplanationSummary, GeneratedExplanation, Page } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";

export interface CitedFact {
  label: string;
  side?: "source" | "target";
}

const CATEGORY_NAMES: Record<string, string> = {
  meaning: "Meaning",
  scope: "Scope",
  key_fact: "Recorded fact",
  agreement: "Shared",
  difference: "Difference",
  explicit_incompatibility: "Incompatibility",
  unknown: "Not stated",
  review_question: "Open question",
};

export function claimCategoryName(category: string): string {
  return CATEGORY_NAMES[category] ?? category.replace(/_/g, " ");
}

export async function findExplanations(
  params: Record<string, string | undefined>,
  signal?: AbortSignal,
): Promise<ExplanationSummary[]> {
  const page = await getJson<Page<ExplanationSummary>>("/api/v1/explanations", { ...params, limit: 100 }, signal);
  return page.items;
}

/** Loads the validated explanation for a selection, reporting absence and review state. */
export function useExplanation(key: string | null, params: Record<string, string | undefined> | null) {
  return useAsync(key, async (signal) => {
    const items = await findExplanations(params ?? {}, signal);
    if (!items.length) return { status: "not_requested" as const };
    const chosen = items.find((item) => item.grounding_status === "validated") ?? items[0];
    if (chosen.grounding_status !== "validated") return { status: "unverified" as const, summary: chosen };
    try {
      const explanation = await getJson<GeneratedExplanation>(`/api/v1/explanations/${encodeURIComponent(chosen.explanation_id)}`, undefined, signal);
      return { status: "available" as const, explanation };
    } catch (error) {
      if (error instanceof ApiError && error.code === "explanation_unverified") return { status: "unverified" as const, summary: chosen };
      throw error;
    }
  });
}

export function Citations({ claim, cite }: { claim: Claim; cite: (factId: string) => CitedFact | undefined }) {
  if (!claim.fact_ids.length) return null;
  return (
    <span className="citations">
      {claim.fact_ids.map((factId) => {
        const fact = cite(factId);
        return (
          <a key={factId} href={`#fact-${factId.slice(7, 19)}`} className={`citation ${fact?.side ? `citation-${fact.side}` : ""}`} title={factId}>
            {fact?.label ?? "cited fact"}
          </a>
        );
      })}
    </span>
  );
}

export function ProvenanceDetails({ explanation }: { explanation: GeneratedExplanation }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="provenance">
      <button type="button" className="btn btn-quiet btn-sm" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        {open ? "Hide how this was prepared" : "How this was prepared"}
      </button>
      {open && (
        <dl className="provenance-list">
          <div>
            <dt>Model requested</dt>
            <dd>{explanation.manifest.requested_model}</dd>
          </div>
          <div>
            <dt>Model returned</dt>
            <dd>{explanation.manifest.returned_model ?? "not recorded"}</dd>
          </div>
          <div>
            <dt>Provider</dt>
            <dd>{explanation.manifest.provider ?? "not recorded"}</dd>
          </div>
          <div>
            <dt>Grounding</dt>
            <dd>{explanation.grounding_status}</dd>
          </div>
          <div>
            <dt>Language</dt>
            <dd>{explanation.manifest.language}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}

export function GeneratedProfile({
  state,
  cite,
}: {
  state: ReturnType<typeof useExplanation>;
  cite: (factId: string) => CitedFact | undefined;
}) {
  if (state.error) return <ErrorNote error={state.error} onRetry={state.reload} what="Generated description" />;
  if (!state.data) return <div className="generated generated-loading" aria-busy="true" />;
  const data = state.data;
  if (data.status === "not_requested") {
    return (
      <section className="generated generated-empty" aria-label="Generated description">
        <h3 className="generated-title">Generated description</h3>
        <p className="meta">No generated description was prepared for this entity. The original facts above are complete for the loaded scope.</p>
      </section>
    );
  }
  if (data.status === "unverified") {
    return (
      <section className="generated generated-empty" aria-label="Generated description">
        <h3 className="generated-title">Generated description</h3>
        <p className="meta">A generated description exists but has not passed grounding review, so it is not shown.</p>
      </section>
    );
  }
  const explanation = data.explanation;
  const facts = new Set(explanation.claims.flatMap((claim) => claim.fact_ids));
  return (
    <section className="generated" aria-label="Generated description">
      <div className="generated-head">
        <h3 className="generated-title">Generated description</h3>
        <span className="meta">
          Cites {facts.size} {facts.size === 1 ? "fact" : "facts"} · grounding validated
        </span>
      </div>
      {explanation.claims.length === 0 && <p className="meta">The generator made no claim it could support from the available facts.</p>}
      <ul className="claim-list">
        {explanation.claims.map((claim, index) => (
          <li key={claim.claim_id ?? index} className="claim">
            <span className="claim-category">{claimCategoryName(claim.category)}</span>
            <span className="claim-text">{claim.text}</span>
            <Citations claim={claim} cite={cite} />
          </li>
        ))}
      </ul>
      {explanation.limitations.length > 0 && (
        <ul className="limitations">
          {explanation.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
      <ProvenanceDetails explanation={explanation} />
    </section>
  );
}
