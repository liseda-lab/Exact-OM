"use client";

// Prepared generated text, visibly distinct from original ontology statements. Only
// explanations that passed grounding are shown; every claim cites the original records it
// rests on, and each citation opens that exact record.

import { useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { citedSubjects, profileView, referencedLabelSubjects } from "@/lib/descriptionPresentation";
import { useFactInspector } from "@/components/workspace/FactInspector";
import { useSideOf } from "@/lib/workspace/WorkspaceContext";
import { curie, shortHash } from "@/lib/iri";
import type { Claim, EntityRef, GeneratedExplanation } from "@/lib/types";
import type { AsyncState } from "@/lib/useAsync";
import type { ExplanationProvenance, ExplanationResult } from "@/lib/workspace/types";

export interface CitedFact {
  /** Short name of the record; the claim's own "cited fact" wording is used when absent. */
  label?: string;
  side?: "source" | "target";
  /** The cited record's exact typed subject (ontology, kind, IRI); the inspector opens it about this entity. */
  subject?: EntityRef | null;
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
        // The record's own typed subject wins; the block's subject is only a fallback (R2).
        const about = fact?.subject ?? subject ?? null;
        // Look the record up in its subject's ontology first, then the side it was cited from.
        const ordered = about
          ? [about.ontology_version_id, ...ontologies.filter((ontology) => ontology !== about.ontology_version_id)]
          : fact?.side
            ? [...ontologies].sort((a, b) => Number(sideOf(b) === fact.side) - Number(sideOf(a) === fact.side))
            : ontologies;
        return (
          <button
            key={factId}
            type="button"
            className={`citation ${fact?.side ? `citation-${fact.side}` : ""}`}
            data-tour="citation"
            aria-label={`Open the cited original record: ${label}`}
            title={`Reference ${shortHash(factId, 12)}`}
            onClick={() => inspector?.open({ factId, ontologies: ordered, subject: about }, label, "citation")}
            disabled={!inspector}
          >
            {label}
          </button>
        );
      })}
    </span>
  );
}

/**
 * Citations of one artifact with their exact typed subjects from its support packet, and labels
 * of referenced entities named as such ("Label of DOID:7505", R2).
 */
export function withReferencedLabels(cite: (factId: string) => CitedFact | undefined, explanation: GeneratedExplanation): (factId: string) => CitedFact | undefined {
  const referenced = referencedLabelSubjects(explanation);
  const subjects = citedSubjects(explanation);
  return (factId) => {
    const known = cite(factId);
    const subject = subjects.get(factId) ?? known?.subject ?? null;
    if (known) return { ...known, subject };
    const other = referenced.get(factId);
    if (other) return { label: `Label of ${curie(other.iri)}`, subject: other };
    return subject ? { subject } : undefined;
  };
}

const SUPPORT_NAMES: Record<string, string> = {
  independently_reviewed: "Generated text, independently reviewed sentence by sentence",
  typed_template_support: "A fixed reading of the original axioms (no language model)",
  exact_support: "Exact quotations of original records",
  fallback: "Original facts only; no description could be prepared",
};

/** One "Sources and preparation" disclosure per generated block: model, checks and limits (21 F04). */
export function ProvenanceDetails({ provenance, limitations = [], note }: { provenance: ExplanationProvenance; limitations?: string[]; note?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="provenance">
      <button type="button" className="link-btn" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        {open ? "Hide sources and preparation" : "Sources and preparation"}
      </button>
      {open && (
        <dl className="provenance-list">
          {note && (
            <div>
              <dt>Scope</dt>
              <dd>{note}</dd>
            </div>
          )}
          {limitations.length > 0 && (
            <div>
              <dt>Limits</dt>
              <dd>{limitations.join(" ")}</dd>
            </div>
          )}
          {provenance.supportStatus && (
            <div>
              <dt>Prepared as</dt>
              <dd>{SUPPORT_NAMES[provenance.supportStatus] ?? provenance.supportStatus.replace(/_/g, " ")}</dd>
            </div>
          )}
          {provenance.supportReasons && provenance.supportReasons.length > 0 && (
            <div>
              <dt>Notes</dt>
              <dd>{provenance.supportReasons.join(" ")}</dd>
            </div>
          )}
          {provenance.kind === "manifest" ? (
            // A deterministic rendering names its renderer, not a model it never used.
            provenance.provider ? (
              <>
                <div>
                  <dt>Model</dt>
                  <dd>
                    {provenance.returnedModel ?? provenance.requestedModel ?? "not recorded"} · {provenance.provider}
                  </dd>
                </div>
                {provenance.reviewerKind && (
                  <div>
                    <dt>Review</dt>
                    <dd>Each sentence was checked against its cited original records by an independent reviewer ({provenance.reviewerKind.replace(/_/g, " ")}); rejected sentences are not shown.</dd>
                  </div>
                )}
                <div>
                  <dt>Language</dt>
                  <dd>{provenance.language || "not recorded"}</dd>
                </div>
              </>
            ) : (
              <div>
                <dt>Renderer</dt>
                <dd>{provenance.renderRevision ?? provenance.requestedModel ?? "not recorded"} · no language model</dd>
              </div>
            )
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
  definitionNote = null,
}: {
  state: AsyncState<ExplanationResult>;
  subject: EntityRef;
  cite: (factId: string) => CitedFact | undefined;
  /** No original definition is shown for this entity: this description is the primary meaning area (21 F05). */
  definitionNote?: string | null;
}) {
  if (state.error) return <ErrorNote error={state.error} onRetry={state.reload} what="Generated description" />;
  if (!state.data) return <div className="generated generated-loading" aria-busy="true" />;
  const data = state.data;
  if (data.status !== "available") {
    // Never claim the original facts are complete just because no description was prepared (21 F04).
    const text =
      data.status === "unverified"
        ? "A generated description exists but did not pass grounding review, so it is not shown."
        : data.status === "not_exported"
          ? data.reason
          : "No generated description was prepared.";
    return (
      <section className="generated generated-empty" aria-label="Generated description">
        <h3 className="generated-title">Generated description</h3>
        <p className="meta">{text}</p>
      </section>
    );
  }
  const explanation = data.explanation;
  const view = profileView(explanation);
  const citeFact = withReferencedLabels(cite, explanation);
  return (
    <section className={definitionNote ? "generated generated-primary" : "generated"} aria-label={view.title}>
      <div className="generated-head">
        <h3 className="generated-title">{view.title}</h3>
        <span className={`origin-tag origin-${view.tag.tone}`}>{view.tag.text}</span>
      </div>
      {definitionNote && <p className="meta">{definitionNote}</p>}
      {view.intro && <p className="meta">{view.intro}</p>}
      {view.unavailable && <p className="meta">No description was prepared.{explanation.support_reasons?.length ? ` ${explanation.support_reasons.join(" ")}` : ""}</p>}
      {!view.unavailable && explanation.claims.length === 0 && <p className="meta">No statement could be supported from the available facts.</p>}
      <ul className="claim-list">
        {explanation.claims.map((claim, index) => (
          <li key={claim.claim_id ?? index} className="claim">
            <span className="claim-text">{claim.text}</span>
            <Citations claim={claim} ontologies={data.factOntologies} subject={subject} cite={citeFact} />
          </li>
        ))}
      </ul>
      <ProvenanceDetails provenance={data.provenance} limitations={explanation.limitations} note={explanation.presentation?.scope_note} />
    </section>
  );
}
