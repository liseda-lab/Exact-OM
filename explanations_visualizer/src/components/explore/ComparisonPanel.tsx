"use client";

// The prepared pair comparison, grouped by what each claim asserts. The comparison was
// generated without Exact's score or verdict; shared wording and one-sided information are
// labelled as such and never presented as proof of equivalence or incompatibility.

import { useMemo, useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { IconDiffer, IconQuestion, IconShared, IconWarning } from "@/components/common/Icons";
import { Citations, ProvenanceDetails, type CitedFact } from "@/components/explore/GeneratedBlock";
import type { Claim, EntityRef } from "@/lib/types";
import { useExplanation } from "@/lib/workspace/WorkspaceContext";

const GROUPS: { key: string; title: string; categories: string[]; tone: string; icon: React.ReactNode }[] = [
  { key: "shared", title: "Shared", categories: ["agreement"], tone: "ok", icon: <IconShared /> },
  { key: "differences", title: "Differences and one-sided information", categories: ["difference", "scope"], tone: "warn", icon: <IconDiffer /> },
  { key: "incompatible", title: "Explicit incompatibilities", categories: ["explicit_incompatibility"], tone: "bad", icon: <IconWarning /> },
  { key: "open", title: "Open questions", categories: ["unknown", "review_question"], tone: "neutral", icon: <IconQuestion /> },
];

export function ComparisonPanel({ source, target, cite, headingLevel = 2 }: { source: EntityRef; target: EntityRef; cite: (factId: string) => CitedFact | undefined; headingLevel?: 2 | 3 }) {
  const entities = useMemo(() => [source, target], [source, target]);
  const state = useExplanation("pair_comparison", entities);
  const [showExcerpts, setShowExcerpts] = useState(false);
  const Heading = headingLevel === 2 ? "h2" : "h3";
  const Group = headingLevel === 2 ? "h3" : "h4";
  const data = state.data;
  return (
    <section className="card comparison" aria-labelledby="comparison-h">
      <div className="comparison-head">
        <Heading id="comparison-h">How the two descriptions compare</Heading>
        <span className="origin-tag origin-generated">Generated</span>
      </div>
      {state.error ? <ErrorNote error={state.error} onRetry={state.reload} what="Comparison" /> : null}
      {!data && !state.error ? <div className="comparison-grid skeleton-row" aria-busy="true" /> : null}
      {data?.status === "not_requested" && <p className="empty-reason">No comparison was prepared for this pair.</p>}
      {data?.status === "not_exported" && <p className="empty-reason">{data.reason}</p>}
      {data?.status === "unverified" && <p className="empty-reason">A comparison exists but did not pass grounding review, so it is not shown.</p>}
      {data?.status === "available" &&
        (() => {
          const claims = data.explanation.claims;
          const byGroup = (categories: string[]) => claims.filter((claim) => categories.includes(claim.category));
          const excerpts = claims.filter((claim) => ["meaning", "key_fact"].includes(claim.category));
          return (
            <>
              {/* Only groups with claims take space; the empty ones are named once (21 F04). */}
              <div className="comparison-grid">
                {GROUPS.filter((group) => byGroup(group.categories).length).map((group) => (
                  <div key={group.key} className={`comparison-group tone-${group.tone}`}>
                    <Group>
                      {group.icon}
                      <span>{group.title}</span>
                    </Group>
                    <ul>
                      {byGroup(group.categories).map((claim, index) => (
                        <ClaimItem key={claim.claim_id ?? index} claim={claim} ontologies={data.factOntologies} cite={cite} />
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
              {GROUPS.some((group) => !byGroup(group.categories).length) && (
                <p className="meta">
                  Nothing recorded for: {GROUPS.filter((group) => !byGroup(group.categories).length).map((group) => group.title.toLowerCase()).join(" · ")}.
                </p>
              )}
              {excerpts.length > 0 && (
                <div>
                  <button type="button" className="btn btn-sm" aria-expanded={showExcerpts} onClick={() => setShowExcerpts((value) => !value)}>
                    {showExcerpts ? "Hide the original statements it quotes" : `Original statements it quotes (${excerpts.length})`}
                  </button>
                  {showExcerpts && (
                    <ul className="excerpt-list">
                      {excerpts.map((claim, index) => (
                        <ClaimItem key={claim.claim_id ?? index} claim={claim} ontologies={data.factOntologies} cite={cite} quote />
                      ))}
                    </ul>
                  )}
                </div>
              )}
              <ProvenanceDetails provenance={data.provenance} limitations={data.explanation.limitations} note="Prepared without Exact's score or verdict." />
            </>
          );
        })()}
    </section>
  );
}

function ClaimItem({ claim, ontologies, cite, quote = false }: { claim: Claim; ontologies: string[]; cite: (factId: string) => CitedFact | undefined; quote?: boolean }) {
  return (
    <li className="claim">
      <span className={quote ? "claim-text claim-quote" : "claim-text"}>{quote ? `“${claim.text}”` : claim.text}</span>
      <Citations claim={claim} ontologies={ontologies} cite={cite} />
    </li>
  );
}
