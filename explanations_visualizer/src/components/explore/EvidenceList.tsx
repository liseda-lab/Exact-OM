"use client";

// The complete, accessible evidence list. It carries the same items as the graph, so the
// graph is never the only way to reach a piece of evidence, and every record it rests on
// can be inspected in its original form even when no reading or graph form exists.

import { useEffect, useRef, useState } from "react";

import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { channelName, type EvidenceBundle } from "@/components/explore/evidenceData";
import { Expression, hasReading } from "@/components/owl/Expression";
import { OriginalAxiom } from "@/components/owl/AxiomBlock";
import { useFactInspector } from "@/components/workspace/FactInspector";
import { axiomRelation } from "@/lib/owl";
import type { AsyncState } from "@/lib/useAsync";
import { interpretationText } from "@/lib/workspace/interpretation";
import { useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

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
  const inspector = useFactInspector();
  const report = useWorkspaceAction();
  const synthetic = useWorkspace().capabilities.synthetic;
  const list = useRef<HTMLUListElement>(null);
  // Selecting an edge in the graph locates the same item here.
  useEffect(() => {
    if (!selected || !list.current) return;
    const node = list.current.querySelector<HTMLElement>(`[data-evidence-id="${CSS.escape(selected)}"]`);
    node?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  if (state.error) return <ErrorNote error={state.error} onRetry={state.reload} what="Evidence" />;
  if (!state.data) return <Skeleton lines={4} title={false} />;
  const { items, axioms } = state.data;
  if (!items.length) return <EmptyReason status={state.data.status} what="selected matcher evidence" reason={state.data.reason} />;
  return (
    <div className="evidence-list">
      <div className="evidence-toolbar">
        <p className="muted">
          {items.length} {items.length === 1 ? "feature" : "features"} Exact selected when scoring this pair{state.data.status === "partial" && state.data.total ? ` (of ${state.data.total} recorded)` : ""}. Each rests on the original records shown.
        </p>
        <label className="check">
          <input type="checkbox" checked={weights} onChange={(event) => setWeights(event.target.checked)} />
          Show matching weights
        </label>
      </div>
      <ul className="evidence-items" ref={list}>
        {items.map((item) => (
          <li key={item.evidence_id} data-evidence-id={item.evidence_id} className={selected === item.evidence_id ? "evidence-item selected" : "evidence-item"} aria-current={selected === item.evidence_id ? "true" : undefined}>
            <div className="evidence-head">
              <SideMarker side={item.side} size="0.875rem" />
              <span className="evidence-kind">
                {channelName(item.channel)} feature · {item.side === "source" ? "source side" : "target side"}
              </span>
              <span className="meta">{synthetic ? "Invented practice example of matcher evidence; no matcher was run" : interpretationText(item.interpretation)}</span>
              {onSelect && (
                <button
                  type="button"
                  className="btn btn-quiet btn-sm"
                  onClick={() => {
                    onSelect(item.evidence_id);
                    report({ type: "evidence_select", evidenceId: item.evidence_id, from: "list" });
                  }}
                  aria-pressed={selected === item.evidence_id}
                >
                  Show in graph
                </button>
              )}
            </div>
            {item.status !== "available" && <p className="note note-warn">{item.reason ?? "Original facts were not recoverable for this feature."}</p>}
            {!item.fact_ids.length && item.status === "available" && <p className="meta">No original record is linked to this feature.</p>}
            {item.fact_ids.map((factId) => {
              const axiom = axioms[factId];
              if (!axiom) {
                return (
                  <div key={factId} className="evidence-fact" data-fact-id={factId}>
                    <p className="meta">No readable or structured form of this record is available here.</p>
                    <button type="button" className="btn btn-quiet btn-sm" onClick={() => inspector?.open({ factId, ontologies: [item.entity.ontology_version_id], subject: item.entity }, `${item.side === "source" ? "Source" : "Target"} ${channelName(item.channel).toLowerCase()} record`)}>
                      Inspect the original record
                    </button>
                  </div>
                );
              }
              const relation = axiomRelation(axiom.ast, item.entity.iri);
              const readable = relation && hasReading(relation.other);
              return (
                <div key={factId} className="evidence-fact" data-fact-id={factId}>
                  {readable && relation ? (
                    <p className="fact-reading">
                      <span className="fact-relation">{relation.relation}</span>{" "}
                      <Expression node={relation.other} ontology={item.entity.ontology_version_id} onOpen={onOpenEntity ? (iri) => onOpenEntity(item.side, iri) : undefined} />
                    </p>
                  ) : (
                    <p className="meta">No plain-language template; the original is shown.</p>
                  )}
                  {readable && (
                    <button
                      type="button"
                      className="btn btn-quiet btn-sm"
                      aria-expanded={openAxiom === factId}
                      onClick={() => {
                        if (openAxiom !== factId) report({ type: "axiom_open", factId });
                        setOpenAxiom(openAxiom === factId ? null : factId);
                      }}
                    >
                      {openAxiom === factId ? "Hide original axiom" : "Show original axiom"}
                    </button>
                  )}
                  {(openAxiom === factId || !readable) && <OriginalAxiom axiom={axiom} />}
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
                <span className="meta">Weights describe the matcher&apos;s internal use, not correctness or semantic strength.</span>
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
