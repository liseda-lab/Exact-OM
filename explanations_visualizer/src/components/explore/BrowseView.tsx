"use client";

// Two independent ontology browsers. Each side keeps its own focus and history in the URL;
// changing one side never resets the other.

import { useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { useEntityContext } from "@/components/explore/CompareView";
import { EntityCard } from "@/components/explore/EntityCard";
import { ontologyName, useExplore } from "@/components/explore/ExploreContext";
import { HierarchyBrowser } from "@/components/explore/HierarchyBrowser";
import type { OntologyMeta } from "@/lib/types";
import { useUrlState } from "@/lib/urlState";

function reasonerStatus(meta: OntologyMeta | undefined): string {
  const value = meta?.capabilities.reasoner_inferred;
  if (!value) return "not_run";
  return typeof value === "string" ? value : value.status;
}

function ScopeNote({ meta }: { meta: OntologyMeta | undefined }) {
  if (!meta) return null;
  const notes: string[] = [];
  if (meta.completeness.scope === "root") notes.push("Root document only: imports were not loaded, so imported labels and axioms may be missing.");
  if (meta.completeness.imports_complete === false) notes.push("Some declared imports were not resolved.");
  if (meta.source_derivation?.status === "declared_derivative") notes.push("This copy is a declared derivative of the original file; its receipt is recorded in the bundle.");
  if (!notes.length) return null;
  return (
    <div className="note note-warn browser-note" role="note">
      <span>{notes.join(" ")}</span>
    </div>
  );
}

export function BrowseView() {
  const state = useExplore();
  const [params, setParams] = useUrlState();
  const runId = params.get("run") ?? state.runs[0]?.run_id ?? null;
  const run = state.runs.find((item) => item.run_id === runId) ?? state.runs[0] ?? null;
  const ontologyIds = Object.keys(state.ontologies);
  const sourceOntology = params.get("so") ?? run?.source_ontology_version_id ?? ontologyIds[0] ?? null;
  const targetOntology = params.get("to") ?? run?.target_ontology_version_id ?? ontologyIds.find((id) => id !== sourceOntology) ?? null;
  const [contextFor, setContextFor] = useState<{ side: "source" | "target"; iri: string } | null>(null);
  const fromSource = params.get("from_source");
  const fromPair = params.get("from_pair");

  const backHref = fromSource ? `/?${new URLSearchParams({ ...(runId ? { run: runId } : {}), source: fromSource, ...(fromPair ? { pair: fromPair } : {}) }).toString()}` : "/";

  const contextOntology = contextFor ? (contextFor.side === "source" ? sourceOntology : targetOntology) : null;
  const contextEntity = contextFor && contextOntology ? { ontology_version_id: contextOntology, iri: contextFor.iri, kind: "class" as const } : null;
  const context = useEntityContext(contextEntity);

  return (
    <div className="browse-page" id="main" tabIndex={-1}>
      <div className="browse-head">
        <h1 className="page-title">Browse both ontologies</h1>
        {fromSource && (
          <a className="btn" href={backHref}>
            Back to the comparison
          </a>
        )}
      </div>
      <div className="browse-grid">
        {[
          ["source", sourceOntology, params.get("s"), "s", "so"] as const,
          ["target", targetOntology, params.get("t"), "t", "to"] as const,
        ].map(([side, ontology, focus, key, ontologyKey]) =>
          ontology ? (
            <div key={side} className="browse-column">
              {ontologyIds.length > 2 && (
                <label className="field">
                  <span className="meta">Ontology shown on this side</span>
                  <select className="select" value={ontology} onChange={(event) => setParams({ [ontologyKey]: event.target.value, [key]: null })}>
                    {ontologyIds.map((id) => (
                      <option key={id} value={id}>
                        {ontologyName(state, id, run)} · {id.slice(7, 15)}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <HierarchyBrowser
                side={side}
                ontology={ontology}
                ontologyLabel={ontologyName(state, ontology, run)}
                focusIri={focus}
                pinnedIri={side === "source" ? params.get("pin_s") ?? null : params.get("pin_t") ?? null}
                onFocus={(iri) => setParams({ [key]: iri })}
                reasonerStatus={reasonerStatus(state.ontologies[ontology])}
                entityCount={state.ontologies[ontology]?.completeness.entity_count}
                scopeNote={<ScopeNote meta={state.ontologies[ontology]} />}
                onOpenContext={(iri) => setContextFor({ side, iri })}
              />
            </div>
          ) : (
            <p key={side} className="note">
              No {side} ontology is available in this bundle.
            </p>
          ),
        )}
      </div>
      {contextFor && contextEntity && (
        <Dialog title="Full context" onClose={() => setContextFor(null)} wide>
          <EntityCard
            side={contextFor.side}
            entity={contextEntity}
            context={context}
            ontologyLabel={ontologyName(state, contextEntity.ontology_version_id, run)}
            cite={() => undefined}
            onOpenEntity={(iri) => {
              setParams({ [contextFor.side === "source" ? "s" : "t"]: iri });
              setContextFor(null);
            }}
          />
        </Dialog>
      )}
    </div>
  );
}
