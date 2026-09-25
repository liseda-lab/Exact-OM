"use client";

// The complete, accessible evidence list. It carries the same items as the graph, so the
// graph is never the only way to reach a piece of evidence.

import { useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { channelName, type EvidenceBundle } from "@/components/explore/evidenceData";
import { Expression, hasReading } from "@/components/owl/Expression";
import { OriginalAxiom } from "@/components/owl/AxiomBlock";
import { axiomRelation } from "@/lib/owl";
import type { AsyncState } from "@/lib/useAsync";

export function EvidenceList({
  state,
  selected,
  onSelect,
  onOpenEntity,
}: {
  state: AsyncState<EvidenceBundle>;
  selected?: string | null;
  onSelect?: (evidenceId: string) => void;
  onOpenEntity?: (side: "source" | "target", iri: string) => void;
}) {
  const [weights, setWeights] = useState(false);
  const [openAxiom, setOpenAxiom] = useState<string | null>(null);
  if (state.error) return <ErrorNote error={state.error} onRetry={state.reload} what="Evidence" />;
  if (!state.data) return <Skeleton lines={4} title={false} />;
  const { items, axioms } = state.data;
  if (!items.length) return <EmptyReason status={state.data.status} what="selected matcher evidence" reason={state.data.reason} />;
  return (
    <div className="evidence-list">
      <div className="evidence-toolbar">
        <p className="muted">
          {items.length} {items.length === 1 ? "feature" : "features"} Exact selected when scoring this pair. Each rests on the original facts shown.
        </p>
        <label className="check">
          <input type="checkbox" checked={weights} onChange={(event) => setWeights(event.target.checked)} />
          Show matching weights
        </label>
      </div>
      <ul className="evidence-items">
        {items.map((item) => (
          <li key={item.evidence_id} className={selected === item.evidence_id ? "evidence-item selected" : "evidence-item"}>
            <div className="evidence-head">
              <SideMarker side={item.side} size="0.875rem" />
              <span className="evidence-kind">
                {channelName(item.channel)} feature · {item.side === "source" ? "source side" : "target side"}
              </span>
              <span className="meta">{item.interpretation === "projected" ? "Projected by Exact from an asserted fact" : item.interpretation.replace(/_/g, " ")}</span>
              {onSelect && (
                <button type="button" className="btn btn-quiet btn-sm" onClick={() => onSelect(item.evidence_id)} aria-pressed={selected === item.evidence_id}>
                  Show in graph
                </button>
              )}
            </div>
            {item.status !== "available" && <p className="note note-warn">{item.reason ?? "Original facts were not recoverable for this feature."}</p>}
            {item.fact_ids.map((factId) => {
              const axiom = axioms[factId];
              if (!axiom) return <p key={factId} className="meta">Original fact could not be loaded.</p>;
              const relation = axiomRelation(axiom.ast, item.entity.iri);
              return (
                <div key={factId} className="evidence-fact">
                  {relation && hasReading(relation.other) ? (
                    <p className="fact-reading">
                      <span className="fact-relation">{relation.relation}</span>{" "}
                      <Expression node={relation.other} ontology={item.entity.ontology_version_id} onOpen={onOpenEntity ? (iri) => onOpenEntity(item.side, iri) : undefined} />
                    </p>
                  ) : (
                    <p className="meta">No plain-language template; original shown.</p>
                  )}
                  <button type="button" className="btn btn-quiet btn-sm" aria-expanded={openAxiom === factId} onClick={() => setOpenAxiom(openAxiom === factId ? null : factId)}>
                    {openAxiom === factId ? "Hide original axiom" : "Show original axiom"}
                  </button>
                  {(openAxiom === factId || !relation) && <OriginalAxiom axiom={axiom} />}
                </div>
              );
            })}
            {weights && (
              <p className="weights">
                {Object.keys(item.values).length
                  ? Object.entries(item.values)
                      .map(([name, value]) => `${name} ${value.toFixed(3)}`)
                      .join(" · ")
                  : "No matching weight was recorded for this feature."}{" "}
                <span className="meta">Weights describe the matcher&apos;s internal use, not correctness.</span>
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
