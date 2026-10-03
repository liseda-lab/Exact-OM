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
        <span className="origin-tag origin-generated">Generated · prepared without Exact&apos;s score or verdict</span>
      </div>
      {state.error ? <ErrorNote error={state.error} onRetry={state.reload} what="Comparison" /> : null}
      {!data && !state.error ? <div className="comparison-grid skeleton-row" aria-busy="true" /> : null}
      {data?.status === "not_requested" && (
        <p className="note">No comparison was prepared for this pair. Read the two cards side by side, or open the evidence below.</p>
      )}
      {data?.status === "not_exported" && <p className="note">{data.reason}</p>}
      {data?.status === "unverified" && <p className="note">A comparison exists but has not passed grounding review, so it is not shown.</p>}
      {data?.status === "available" &&
        (() => {
          const claims = data.explanation.claims;
          const byGroup = (categories: string[]) => claims.filter((claim) => categories.includes(claim.category));
          const excerpts = claims.filter((claim) => ["meaning", "key_fact"].includes(claim.category));
          return (
            <>
              <div className="comparison-grid">
                {GROUPS.map((group) => {
                  const items = byGroup(group.categories);
                  return (
                    <div key={group.key} className={items.length ? `comparison-group tone-${group.tone}` : "comparison-group"}>
                      <Group>
                        {group.icon}
                        <span>{group.title}</span>
                      </Group>
                      {items.length ? (
                        <ul>
                          {items.map((claim, index) => (
                            <ClaimItem key={claim.claim_id ?? index} claim={claim} ontologies={data.factOntologies} cite={cite} />
                          ))}
                        </ul>
                      ) : (
                        <p className="meta">
                          {group.key === "incompatible"
                            ? "No opposing statement is recorded in the cited facts."
                            : group.key === "shared"
                              ? "No shared recorded wording was found."
                              : group.key === "open"
                                ? "No open question was recorded."
                                : "No difference was recorded."}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
              {data.explanation.limitations.length > 0 && (
                <div className="note">
                  <span>
                    <strong>Limits: </strong>
                    {data.explanation.limitations.join(" ")}
                  </span>
                </div>
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
              <ProvenanceDetails provenance={data.provenance} />
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
