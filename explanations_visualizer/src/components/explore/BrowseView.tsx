"use client";

// Two independent ontology browsers. Each side keeps its own focus and history in the URL;
// changing one side never resets the other.

import { useEffect, useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { useEntityContext } from "@/components/explore/CompareView";
import { EntityCard } from "@/components/explore/EntityCard";
import { ontologyName, useExplore } from "@/components/explore/ExploreContext";
import { HierarchyBrowser } from "@/components/explore/HierarchyBrowser";
import { FactInspectorProvider } from "@/components/workspace/FactInspector";
import type { EntityKind, EntityRef, OntologyMeta } from "@/lib/types";
import { useUrlState } from "@/lib/urlState";

function entityKind(value: string | null): EntityKind {
  return value === "object_property" || value === "data_property" || value === "individual" ? value : "class";
}

// A deliberate focus is remembered for this exact package, side and ontology (21 F06). An
// explicit URL focus always wins; the backend's entry points apply when neither exists.
const FOCUS_KEY = "exact.browse.focus";
function savedFocus(key: string): { iri: string; kind: EntityKind } | null {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(key) ?? "null");
    return value && typeof value.iri === "string" ? { iri: value.iri, kind: entityKind(value.kind) } : null;
  } catch {
    return null;
  }
}
function rememberFocus(key: string, iri: string, kind: EntityKind) {
  try {
    window.sessionStorage.setItem(key, JSON.stringify({ iri, kind }));
  } catch {
    /* a convenience only */
  }
}

/**
 * One scope indicator per ontology panel (21 F04). Only an unresolved declared import is a
 * visible notice; loading from the root document is not a warning by itself, and provenance
 * details stay one disclosure away.
 */
function ScopeNote({ meta }: { meta: OntologyMeta | undefined }) {
  if (!meta) return null;
  const details: string[] = [];
  if (meta.completeness.scope === "root") details.push("Loaded from its root document.");
  if (meta.completeness.scope === "closure") details.push("Loaded with its resolved import closure.");
  if (meta.source_derivation?.status === "declared_derivative") details.push("This copy is a declared derivative of the original file; its receipt is recorded in the bundle.");
  return (
    <>
      {meta.completeness.imports_complete === false && (
        <p className="note note-warn browser-note" role="note">
          Some declared imports were not resolved, so imported labels and axioms may be missing.
        </p>
      )}
      {details.length > 0 && (
        <details className="scope-details">
          <summary>About this ontology</summary>
          <p className="meta">{details.join(" ")}</p>
        </details>
      )}
    </>
  );
}

/** The comparison exactly as it was left: run, source, pair and its own view (HR2 §5). */
function backParams(params: URLSearchParams, runId: string | null, source: string, pair: string | null): URLSearchParams {
  const next = new URLSearchParams();
  if (runId) next.set("run", runId);
  next.set("source", source);
  if (pair) next.set("pair", pair);
  for (const key of ["sk", "list", "details", "hs", "hsk", "ht", "htk"]) {
    const value = params.get(`from_${key}`);
    if (value) next.set(key, value);
  }
  return next;
}

export function BrowseView() {
  const state = useExplore();
  const [params, setParams] = useUrlState();
  const runId = params.get("run") ?? state.runs[0]?.run_id ?? null;
  const run = state.runs.find((item) => item.run_id === runId) ?? state.runs[0] ?? null;
  const ontologyIds = Object.keys(state.ontologies);
  const sourceOntology = params.get("so") ?? run?.source_ontology_version_id ?? ontologyIds[0] ?? null;
  const targetOntology = params.get("to") ?? run?.target_ontology_version_id ?? ontologyIds.find((id) => id !== sourceOntology) ?? null;
  const [contextFor, setContextFor] = useState<{ side: "source" | "target"; entity: EntityRef } | null>(null);
  const fromSource = params.get("from_source");
  const fromPair = params.get("from_pair");

  const backHref = fromSource ? `/?${backParams(params, runId, fromSource, fromPair).toString()}` : "/";

  const contextEntity = contextFor?.entity ?? null;
  const context = useEntityContext(contextEntity);
  const packageId = state.health?.package_id ?? "none";
  const focusKey = (side: string, ontology: string) => `${FOCUS_KEY}|${packageId}|${side}|${ontology}`;
  // Restore a remembered focus only when the URL names none for that side.
  useEffect(() => {
    const changes: Record<string, string> = {};
    for (const [side, ontology, key, kindKey] of [["source", sourceOntology, "s", "sk"], ["target", targetOntology, "t", "tk"]] as const) {
      if (!ontology || params.get(key)) continue;
      const saved = savedFocus(focusKey(side, ontology));
      if (saved) Object.assign(changes, { [key]: saved.iri, [kindKey]: saved.kind });
    }
    if (Object.keys(changes).length) setParams(changes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceOntology, targetOntology, packageId]);

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
          ["source", sourceOntology, params.get("s"), "s", "so", "sk"] as const,
          ["target", targetOntology, params.get("t"), "t", "to", "tk"] as const,
        ].map(([side, ontology, focus, key, ontologyKey, kindKey]) =>
          ontology ? (
            <div key={side} className="browse-column">
              {ontologyIds.length > 2 && (
                <label className="field">
                  <span className="meta">Ontology shown on this side</span>
                  <select className="select" value={ontology} onChange={(event) => setParams({ [ontologyKey]: event.target.value, [key]: null, [kindKey]: null })}>
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
                kind={entityKind(params.get(kindKey))}
                pinnedIri={side === "source" ? params.get("pin_s") ?? null : params.get("pin_t") ?? null}
                pinnedKind={entityKind(params.get(side === "source" ? "pin_sk" : "pin_tk"))}
                onFocus={(iri, kind) => {
                  rememberFocus(focusKey(side, ontology), iri, kind);
                  setParams({ [key]: iri, [kindKey]: kind });
                }}
                entityCount={state.ontologies[ontology]?.completeness.entity_count}
                scopeNote={<ScopeNote meta={state.ontologies[ontology]} />}
                onOpenContext={(iri, kind) => setContextFor({ side, entity: { ontology_version_id: ontology, iri, kind } })}
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
          <FactInspectorProvider>
          <EntityCard
            side={contextFor.side}
            entity={contextEntity}
            context={context}
            ontologyLabel={ontologyName(state, contextEntity.ontology_version_id, run)}
            cite={() => undefined}
            onOpenEntity={(target) => {
              // The card supplies the recorded typed entity, whichever page it came from (19 F21).
              const sourceSide = contextFor.side === "source";
              setParams({
                [sourceSide ? "s" : "t"]: target.iri,
                [sourceSide ? "sk" : "tk"]: target.kind,
                ...(target.ontology_version_id !== contextEntity.ontology_version_id ? { [sourceSide ? "so" : "to"]: target.ontology_version_id } : {}),
              });
              setContextFor(null);
            }}
          />
          </FactInspectorProvider>
        </Dialog>
      )}
    </div>
  );
}
