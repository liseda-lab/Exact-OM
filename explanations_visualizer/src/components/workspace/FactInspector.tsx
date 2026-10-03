"use client";

// Opens any cited or displayed original record by identity and typed subject, whether or
// not it is currently rendered (hidden duplicates, labels, records beyond a loaded page).
// Unavailable records state their specific reason; nothing is invented.

import { createContext, useCallback, useContext, useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { ErrorNote, Skeleton } from "@/components/common/ErrorNote";
import { SideMarker } from "@/components/common/Icons";
import { EmptyReason } from "@/components/common/StatusText";
import { Expression, hasReading } from "@/components/owl/Expression";
import { OriginalAxiom } from "@/components/owl/AxiomBlock";
import { curie, predicateName, shortHash } from "@/lib/iri";
import { useLabelLookup } from "@/lib/labelSource";
import { axiomRelation } from "@/lib/owl";
import { interpretationText } from "@/lib/workspace/interpretation";
import type { FactRef, ResolvedFact } from "@/lib/workspace/types";
import { useResolvedFact, useSideOf, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

interface Inspector {
  open: (ref: FactRef, label?: string) => void;
}

const InspectorContext = createContext<Inspector | null>(null);

export function useFactInspector(): Inspector | null {
  return useContext(InspectorContext);
}

export function FactInspectorProvider({ children }: { children: React.ReactNode }) {
  const [current, setCurrent] = useState<{ ref: FactRef; label?: string } | null>(null);
  const report = useWorkspaceAction();
  const open = useCallback(
    (ref: FactRef, label?: string) => {
      setCurrent({ ref, label });
      report({ type: "citation_open", factId: ref.factId });
    },
    [report],
  );
  return (
    <InspectorContext.Provider value={{ open }}>
      {children}
      {current && <FactDialog reference={current.ref} label={current.label} onClose={() => setCurrent(null)} />}
    </InspectorContext.Provider>
  );
}

function SubjectLine({ fact }: { fact: ResolvedFact }) {
  const sideOf = useSideOf();
  const first = fact.subjects[0]?.entity.ontology_version_id ?? null;
  const label = useLabelLookup(first, fact.subjects.filter((subject) => subject.entity.ontology_version_id === first).map((subject) => subject.entity.iri));
  if (!fact.subjects.length) return <p className="meta">Subject not recorded with this reference.</p>;
  return (
    <div className="inspector-subjects">
      <span className="meta">{fact.subjects.length > 1 ? `Stored under ${fact.subjects.length} entities (one shared original axiom)` : "About"}</span>
      <ul>
        {fact.subjects.map((subject) => {
          const side = sideOf(subject.entity.ontology_version_id) ?? subject.side;
          const name = subject.entity.ontology_version_id === first ? label(subject.entity.iri)?.value : null;
          return (
            <li key={`${subject.entity.ontology_version_id}|${subject.entity.kind}|${subject.entity.iri}`}>
              {side && <SideMarker side={side} size="0.875rem" />}
              <span>{name ?? curie(subject.entity.iri)}</span>
              <span className="iri">{subject.entity.iri}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function locate(factId: string): HTMLElement | null {
  const nodes = Array.from(document.querySelectorAll<HTMLElement>(`[data-fact-id="${CSS.escape(factId)}"]`));
  return nodes.find((node) => !node.closest(".dialog") && node.offsetParent !== null) ?? null;
}

function FactDialog({ reference, label, onClose }: { reference: FactRef; label?: string; onClose: () => void }) {
  const state = useResolvedFact(reference);
  const report = useWorkspaceAction();
  const fact = state.data;
  const subjectIri = fact?.subjects[0]?.entity.iri ?? reference.subject?.iri ?? "";
  const relation = fact?.axiom ? axiomRelation(fact.axiom.ast, subjectIri) : null;
  const readable = relation && hasReading(relation.other);
  const target = typeof document !== "undefined" ? locate(reference.factId) : null;
  return (
    <Dialog title={label ? `Cited fact · ${label}` : "Original record"} onClose={onClose} wide>
      {state.error ? <ErrorNote error={state.error} onRetry={state.reload} what="This record" /> : null}
      {!fact && !state.error ? <Skeleton lines={4} title={false} /> : null}
      {fact && fact.status !== "available" && (
        <>
          <EmptyReason status={fact.status} what="original record" reason={fact.reason} />
          <p className="meta">Reference {shortHash(reference.factId, 16)}. The text that cites it is shown without changes; no substitute record is invented.</p>
        </>
      )}
      {fact && fact.status === "available" && (
        <div className="inspector">
          <SubjectLine fact={fact} />
          <dl className="inspector-facts">
            <div>
              <dt>Kind</dt>
              <dd>{fact.category ? fact.category.replace(/_/g, " ") : "not recorded"}{fact.predicateIri ? ` · ${predicateName(fact.predicateIri)}` : ""}</dd>
            </div>
            <div>
              <dt>Origin</dt>
              <dd>
                {interpretationText(fact.interpretation)}
                {fact.derivation ? ` · rule: ${fact.derivation}` : ""}
              </dd>
            </div>
          </dl>
          {fact.literal ? (
            <blockquote className="inspector-literal" lang={fact.literal.language ?? undefined}>
              “{fact.literal.text}”{fact.literal.language ? <span className="meta"> @{fact.literal.language}</span> : null}
            </blockquote>
          ) : readable && relation ? (
            <p className="fact-reading">
              <span className="fact-relation">{relation.relation}</span> <Expression node={relation.other} ontology={fact.ontology ?? ""} />
            </p>
          ) : (
            <p className="meta">No plain-language reading exists for this record; its original form is shown below.</p>
          )}
          {fact.axiom ? (
            <>
              <OriginalAxiom axiom={fact.axiom} />
              {fact.reconstructed && <p className="meta">Shown in Functional Syntax from the stored subject, predicate and value; the original file bytes are not part of this view.</p>}
            </>
          ) : (
            <p className="note">The original structure of this record is not included in this view.</p>
          )}
          <p className="meta">
            {fact.origins.length
              ? fact.origins.map((origin) => `Document ${shortHash(origin.document, 12)}${origin.span ? ` · bytes ${origin.span}` : ""}`).join(" · ")
              : "No source location was recorded for this record."}{" "}
            · Reference {shortHash(reference.factId, 12)}
          </p>
        </div>
      )}
      <div className="study-actions">
        {target && (
          <button
            type="button"
            className="btn"
            onClick={() => {
              report({ type: "locate_fact", factId: reference.factId, inList: Boolean(target.closest(".evidence-list")) });
              onClose();
              window.setTimeout(() => {
                const node = locate(reference.factId);
                if (!node) return;
                node.setAttribute("tabindex", "-1");
                node.classList.add("located");
                node.scrollIntoView({ block: "center" });
                node.focus({ preventScroll: true });
                window.setTimeout(() => node.classList.remove("located"), 2400);
              }, 30);
            }}
          >
            Show where it appears on this page
          </button>
        )}
        <button type="button" className="btn btn-primary" onClick={onClose}>
          Close
        </button>
      </div>
    </Dialog>
  );
}
