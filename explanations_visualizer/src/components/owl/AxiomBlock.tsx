"use client";

// One original axiom: a conservative plain reading when a template exists, and the full
// original structure in OWL Functional Syntax on request.

import { useState } from "react";

import { ErrorNote } from "@/components/common/ErrorNote";
import { Expression, hasReading } from "@/components/owl/Expression";
import { getJson } from "@/lib/api";
import { axiomRelation, functionalSyntax } from "@/lib/owl";
import type { Axiom } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";

const axiomCache = new Map<string, Promise<Axiom>>();

export function loadAxiom(ontology: string, axiomId: string, signal?: AbortSignal): Promise<Axiom> {
  const key = `${ontology}|${axiomId}`;
  if (!axiomCache.has(key)) {
    const promise = getJson<Axiom>(`/api/v1/axioms/${encodeURIComponent(axiomId)}`, { ontology_version_id: ontology }, signal);
    promise.catch(() => axiomCache.delete(key));
    axiomCache.set(key, promise);
  }
  return axiomCache.get(key)!;
}

export function OriginalAxiom({ axiom }: { axiom: Axiom }) {
  const [full, setFull] = useState(false);
  return (
    <div className="original-axiom">
      <div className="original-axiom-head">
        <span className="meta">Original · OWL Functional Syntax · {axiom.interpretation.kind}</span>
        <button type="button" className="btn-quiet btn btn-sm" onClick={() => setFull((value) => !value)} aria-pressed={full}>
          {full ? "Show short names" : "Show full IRIs"}
        </button>
      </div>
      <code className="fs-code">{functionalSyntax(axiom.ast, { abbreviate: !full })}</code>
      {axiom.original_availability !== "available" && <p className="meta">The original source bytes are not included in this bundle; this is the stored structure.</p>}
    </div>
  );
}

export function AxiomBlock({ ontology, axiomId, subjectIri, onOpen }: { ontology: string; axiomId: string; subjectIri: string; onOpen?: (iri: string) => void }) {
  const [open, setOpen] = useState(false);
  const state = useAsync(`${ontology}|${axiomId}`, (signal) => loadAxiom(ontology, axiomId, signal));
  if (state.error) return <ErrorNote error={state.error} onRetry={state.reload} what="Axiom" />;
  if (!state.data) return <div className="fact-block skeleton-row" aria-busy="true" />;
  const axiom = state.data;
  const relation = axiomRelation(axiom.ast, subjectIri);
  const readable = relation && hasReading(relation.other);
  return (
    <div className="fact-block" id={`fact-${axiomId.slice(7, 19)}`}>
      {readable && relation ? (
        <p className="fact-reading">
          <span className="fact-relation">{relation.relation}</span> <Expression node={relation.other} ontology={ontology} onOpen={onOpen} />
        </p>
      ) : (
        <p className="meta">No plain-language template exists for this axiom; its original form is shown in full.</p>
      )}
      <div className="fact-foot">
        <span className="meta">{axiom.interpretation.kind === "asserted" ? "Asserted" : axiom.interpretation.kind}</span>
        {readable && (
          <button type="button" className="btn btn-quiet btn-sm" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
            {open ? "Hide original axiom" : "Show original axiom"}
          </button>
        )}
      </div>
      {(open || !readable) && <OriginalAxiom axiom={axiom} />}
    </div>
  );
}
