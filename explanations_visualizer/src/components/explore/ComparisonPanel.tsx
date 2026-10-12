"use client";

// The prepared pair comparison, presented as what it declares itself to be (23 C03, 21 F05).
// A reviewed comparison is grouped by what each claim asserts; paired definition excerpts and
// one-sided information are laid out under the concept they describe, resolved by entity
// identity rather than column position. Nothing here is proof of equivalence or
// incompatibility, and one-sided text is never presented as a difference.

import { useMemo, useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { IconDiffer, IconQuestion, IconShared, IconWarning, SideMarker } from "@/components/common/Icons";
import { Citations, ProvenanceDetails, withReferencedLabels, type CitedFact } from "@/components/explore/GeneratedBlock";
import { claimSides, comparisonView, coveredSides, roleSides, type Side } from "@/lib/descriptionPresentation";
import { curie } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import type { Claim, EntityRef } from "@/lib/types";
import { useExplanation, useSideOf } from "@/lib/workspace/WorkspaceContext";

const GROUPS: { key: string; title: string; categories: string[]; tone: string; icon: React.ReactNode }[] = [
  { key: "shared", title: "Shared", categories: ["agreement"], tone: "ok", icon: <IconShared /> },
  { key: "differences", title: "Differences and one-sided information", categories: ["difference", "scope"], tone: "warn", icon: <IconDiffer /> },
  { key: "incompatible", title: "Explicit incompatibilities", categories: ["explicit_incompatibility"], tone: "bad", icon: <IconWarning /> },
  { key: "open", title: "Open questions", categories: ["unknown", "review_question"], tone: "neutral", icon: <IconQuestion /> },
];

export function ComparisonPanel({ source, target, cite, headingLevel = 2 }: { source: EntityRef; target: EntityRef; cite: (factId: string) => CitedFact | undefined; headingLevel?: 2 | 3 }) {
  const entities = useMemo(() => [source, target], [source, target]);
  const state = useExplanation("pair_comparison", entities);
  const sideOf = useSideOf();
  const sourceLabel = useLabelLookup(source.ontology_version_id, [source.iri])(source.iri)?.value ?? curie(source.iri);
  const targetLabel = useLabelLookup(target.ontology_version_id, [target.iri])(target.iri)?.value ?? curie(target.iri);
  const [showExcerpts, setShowExcerpts] = useState(false);
  const Heading = headingLevel === 2 ? "h2" : "h3";
  const Group = headingLevel === 2 ? "h3" : "h4";
  const data = state.data;
  const explanation = data?.status === "available" ? data.explanation : null;
  const roles = explanation ? roleSides(explanation, { source, target }, (ontology) => sideOf(ontology)) : null;
  const covered = explanation && roles ? coveredSides(explanation, roles) : [];
  const view = explanation ? comparisonView(explanation, covered) : null;
  const names: Record<Side, string> = { source: sourceLabel, target: targetLabel };

  return (
    <section className="card comparison" aria-labelledby="comparison-h">
      <div className="comparison-head">
        <Heading id="comparison-h">{view?.title ?? "How the two descriptions compare"}</Heading>
        {view && <span className={`origin-tag origin-${view.tag.tone}`}>{view.tag.text}</span>}
      </div>
      {state.error ? <ErrorNote error={state.error} onRetry={state.reload} what="Comparison" /> : null}
      {!data && !state.error ? <div className="comparison-grid skeleton-row" aria-busy="true" /> : null}
      {data?.status === "not_requested" && <p className="empty-reason">No comparison was prepared for this pair.</p>}
      {data?.status === "not_exported" && <p className="empty-reason">{data.reason}</p>}
      {data?.status === "unverified" && <p className="empty-reason">A comparison exists but did not pass grounding review, so it is not shown.</p>}
      {data?.status === "available" &&
        explanation &&
        view &&
        roles &&
        (() => {
          const claims = explanation.claims;
          if (view.unavailable || !claims.length)
            return (
              <p className="empty-reason">
                No comparison could be supported for this pair.{explanation.support_reasons?.length ? ` ${explanation.support_reasons.join(" ")}` : ""}
              </p>
            );
          const sidesOf = claimSides(explanation, roles);
          const citeFact = withReferencedLabels(cite, explanation);
          return (
            <>
              {view.intro && <p className="meta">{view.intro}</p>}
              {view.bySide ? (
                <SideClaims claims={claims} sidesOf={sidesOf} covered={covered} names={names} ontologies={data.factOntologies} cite={citeFact} Group={Group} quote={view.tag.tone === "original"} />
              ) : (
                <GroupedClaims claims={claims} ontologies={data.factOntologies} cite={citeFact} Group={Group} showExcerpts={showExcerpts} setShowExcerpts={setShowExcerpts} />
              )}
              {explanation.presentation?.scope_note && <p className="meta comparison-scope">{explanation.presentation.scope_note}</p>}
              <ProvenanceDetails provenance={data.provenance} limitations={explanation.limitations} note="Prepared without Exact's score or verdict." />
            </>
          );
        })()}
    </section>
  );
}

/** Claims under the concept each one describes (paired excerpts, one-sided information). */
function SideClaims({
  claims,
  sidesOf,
  covered,
  names,
  ontologies,
  cite,
  Group,
  quote,
}: {
  claims: Claim[];
  sidesOf: Record<string, Side[]>;
  covered: Side[];
  names: Record<Side, string>;
  ontologies: string[];
  cite: (factId: string) => CitedFact | undefined;
  Group: "h3" | "h4";
  quote: boolean;
}) {
  const sides: Side[] = covered.length ? covered : ["source", "target"];
  const placed = (claim: Claim) => (claim.claim_id ? sidesOf[claim.claim_id] ?? [] : []);
  const of = (side: Side) => claims.filter((claim) => placed(claim).includes(side));
  const unplaced = claims.filter((claim) => !placed(claim).length);
  return (
    <div className={sides.length === 2 ? "side-claims two" : "side-claims"}>
      {sides.map((side) => (
        <div key={side}>
          <Group>
            <SideMarker side={side} />
            <span>{names[side]}</span>
          </Group>
          {of(side).length ? (
            <ul className="claim-list">
              {of(side).map((claim, index) => (
                <ClaimItem key={claim.claim_id ?? index} claim={claim} ontologies={ontologies} cite={cite} quote={quote} />
              ))}
            </ul>
          ) : (
            <p className="meta">Nothing prepared for this concept.</p>
          )}
        </div>
      ))}
      {unplaced.length > 0 && (
        <ul className="claim-list">
          {unplaced.map((claim, index) => (
            <ClaimItem key={claim.claim_id ?? index} claim={claim} ontologies={ontologies} cite={cite} quote={quote} />
          ))}
        </ul>
      )}
    </div>
  );
}

/** A reviewed (or historical) comparison grouped by what each claim asserts. */
function GroupedClaims({
  claims,
  ontologies,
  cite,
  Group,
  showExcerpts,
  setShowExcerpts,
}: {
  claims: Claim[];
  ontologies: string[];
  cite: (factId: string) => CitedFact | undefined;
  Group: "h3" | "h4";
  showExcerpts: boolean;
  setShowExcerpts: (update: (value: boolean) => boolean) => void;
}) {
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
                <ClaimItem key={claim.claim_id ?? index} claim={claim} ontologies={ontologies} cite={cite} />
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
                <ClaimItem key={claim.claim_id ?? index} claim={claim} ontologies={ontologies} cite={cite} quote />
              ))}
            </ul>
          )}
        </div>
      )}
    </>
  );
}

function ClaimItem({ claim, ontologies, cite, quote = false }: { claim: Claim; ontologies: string[]; cite: (factId: string) => CitedFact | undefined; quote?: boolean }) {
  return (
    <li className="claim">
      <span className={quote ? "claim-text claim-quote" : "claim-text"}>{claim.text}</span>
      <Citations claim={claim} ontologies={ontologies} cite={cite} />
    </li>
  );
}
