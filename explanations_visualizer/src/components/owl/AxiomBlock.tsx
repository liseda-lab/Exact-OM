"use client";

// One original axiom: a conservative plain reading when a template exists, its recorded
// interpretation, and the full original structure in OWL Functional Syntax on request.
// Records are resolved through the active workspace, so the same block serves exploration,
// a study case and the tutorial.

import { useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { EmptyReason } from "@/components/common/StatusText";
import { Expression, hasReading } from "@/components/owl/Expression";
import { axiomRelation, functionalSyntax } from "@/lib/owl";
import type { Axiom, EntityRef } from "@/lib/types";
import { interpretationText } from "@/lib/workspace/interpretation";
import { useResolvedFact, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

export function OriginalAxiom({ axiom }: { axiom: Axiom }) {
  const [full, setFull] = useState(false);
  return (
    <div className="original-axiom">
      <div className="original-axiom-head">
        <span className="meta">Original · OWL Functional Syntax · {interpretationText(axiom.interpretation?.kind)}</span>
        <button type="button" className="btn-quiet btn btn-sm" onClick={() => setFull((value) => !value)} aria-pressed={full}>
          {full ? "Show short names" : "Show full IRIs"}
        </button>
      </div>
      <code className="fs-code">{functionalSyntax(axiom.ast, { abbreviate: !full })}</code>
      {axiom.original_availability !== "available" && (
        <p className="meta">
          {axiom.original_format === "study-resource"
            ? "The study resource holds this typed structure, not the original file bytes."
            : "The original source bytes are not included in this bundle; this is the stored structure."}
        </p>
      )}
    </div>
  );
}

export function AxiomBlock({ ontology, factId, subject, onOpen }: { ontology: string; factId: string; subject: EntityRef; onOpen?: (entity: EntityRef) => void }) {
  const [open, setOpen] = useState(false);
  const report = useWorkspaceAction();
  const state = useResolvedFact({ factId, ontologies: [ontology], subject });
  if (state.error) return <ErrorNote error={state.error} onRetry={state.reload} what="Axiom" />;
  if (!state.data) return <div className="fact-block skeleton-row" aria-busy="true" />;
  const fact = state.data;
  if (fact.status !== "available" || !fact.axiom) {
    return (
      <div className="fact-block" data-fact-id={factId}>
        <EmptyReason status={fact.status === "available" ? "partial" : fact.status} what="original axiom" reason={fact.reason ?? "This record has no stored structure in this view."} />
      </div>
    );
  }
  const axiom = fact.axiom;
  const relation = axiomRelation(axiom.ast, subject.iri);
  const readable = relation && hasReading(relation.other);
  return (
    <div className="fact-block" data-fact-id={factId}>
      {readable && relation ? (
        <p className="fact-reading">
          <span className="fact-relation">{relation.relation}</span> <Expression node={relation.other} ontology={ontology} onOpen={onOpen ? (iri, kind) => onOpen({ ontology_version_id: ontology, iri, kind }) : undefined} />
        </p>
      ) : (
        <p className="meta">No plain-language template exists for this axiom; its original form is shown in full.</p>
      )}
      <div className="fact-foot">
        <span className="meta">{interpretationText(fact.interpretation)}{fact.derivation ? ` · rule: ${fact.derivation}` : ""}</span>
        {readable && (
          <button
            type="button"
            className="btn btn-quiet btn-sm"
            aria-expanded={open}
            onClick={() => {
              if (!open) report({ type: "axiom_open", factId });
              setOpen((value) => !value);
            }}
          >
            {open ? "Hide original axiom" : "Show original axiom"}
          </button>
        )}
      </div>
      {(open || !readable) && <OriginalAxiom axiom={axiom} />}
    </div>
  );
}
