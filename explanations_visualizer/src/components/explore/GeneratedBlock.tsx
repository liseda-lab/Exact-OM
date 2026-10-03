"use client";

// Prepared generated text, visibly distinct from original ontology statements. Only
// explanations that passed grounding are shown; every claim cites the original records it
// rests on, and each citation opens that exact record.

import { useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { useFactInspector } from "@/components/workspace/FactInspector";
import { useSideOf } from "@/lib/workspace/WorkspaceContext";
import { shortHash } from "@/lib/iri";
import type { Claim, EntityRef } from "@/lib/types";
import type { AsyncState } from "@/lib/useAsync";
import type { ExplanationProvenance, ExplanationResult } from "@/lib/workspace/types";

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

export function Citations({
  claim,
  ontologies,
  subject,
  cite,
}: {
  claim: Claim;
  ontologies: string[];
  subject?: EntityRef | null;
  cite: (factId: string) => CitedFact | undefined;
}) {
  const inspector = useFactInspector();
  const sideOf = useSideOf();
  if (!claim.fact_ids.length) return null;
  return (
    <span className="citations">
      {claim.fact_ids.map((factId, index) => {
        const fact = cite(factId);
        const label = fact?.label ?? (claim.fact_ids.length > 1 ? `cited fact ${index + 1}` : "cited fact");
        // Look the record up in the ontology of the side it was cited from first.
        const ordered = fact?.side ? [...ontologies].sort((a, b) => Number(sideOf(b) === fact.side) - Number(sideOf(a) === fact.side)) : ontologies;
        return (
          <button
            key={factId}
            type="button"
            className={`citation ${fact?.side ? `citation-${fact.side}` : ""}`}
            aria-label={`Open the cited original record: ${label}`}
            title={`Reference ${shortHash(factId, 12)}`}
            onClick={() => inspector?.open({ factId, ontologies: ordered, subject: subject ?? null }, label)}
            disabled={!inspector}
          >
            {label}
          </button>
        );
      })}
    </span>
  );
}

export function ProvenanceDetails({ provenance }: { provenance: ExplanationProvenance }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="provenance">
      <button type="button" className="btn btn-quiet btn-sm" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        {open ? "Hide how this was prepared" : "How this was prepared"}
      </button>
      {open && (
        <dl className="provenance-list">
          {provenance.kind === "manifest" ? (
            <>
              <div>
                <dt>Model requested</dt>
                <dd>{provenance.requestedModel || "not recorded"}</dd>
              </div>
              <div>
                <dt>Model returned</dt>
                <dd>{provenance.returnedModel ?? "not recorded"}</dd>
              </div>
              <div>
                <dt>Provider</dt>
                <dd>{provenance.provider ?? "not recorded"}</dd>
              </div>
              <div>
                <dt>Language</dt>
                <dd>{provenance.language || "not recorded"}</dd>
              </div>
            </>
          ) : (
            <>
              <div>
                <dt>Prepared</dt>
                <dd>{provenance.synthetic ? "Authored synthetic practice text; no model or matcher was run" : "Before the study, frozen with its publication; nothing is generated while you take part"}</dd>
              </div>
              {provenance.manifestHashes && provenance.manifestHashes.length > 0 && (
                <div>
                  <dt>Generation record</dt>
                  <dd className="mono">{provenance.manifestHashes.map((hash) => shortHash(hash, 12)).join(", ")}</dd>
                </div>
              )}
              {provenance.claimGrounding && (
                <div>
                  <dt>Claim checks</dt>
                  <dd>
                    {Object.values(provenance.claimGrounding).filter((value) => value === "exact_extract").length} exact quotations ·{" "}
                    {Object.values(provenance.claimGrounding).filter((value) => value === "semantic_template").length} fixed comparison templates
                  </dd>
                </div>
              )}
            </>
          )}
          <div>
            <dt>Grounding</dt>
            <dd>{provenance.grounding}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}

export function GeneratedProfile({
  state,
  subject,
  cite,
}: {
  state: AsyncState<ExplanationResult>;
  subject: EntityRef;
  cite: (factId: string) => CitedFact | undefined;
}) {
  if (state.error) return <ErrorNote error={state.error} onRetry={state.reload} what="Generated description" />;
  if (!state.data) return <div className="generated generated-loading" aria-busy="true" />;
  const data = state.data;
  if (data.status !== "available") {
    const text =
      data.status === "unverified"
        ? "A generated description exists but has not passed grounding review, so it is not shown."
        : data.status === "not_exported"
          ? data.reason
          : "No generated description was prepared for this entity. The original facts above are complete for the loaded scope.";
    return (
      <section className="generated generated-empty" aria-label="Generated description">
        <h3 className="generated-title">Generated description</h3>
        <p className="meta">{text}</p>
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
            <Citations claim={claim} ontologies={data.factOntologies} subject={subject} cite={cite} />
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
      <ProvenanceDetails provenance={data.provenance} />
    </section>
  );
}
