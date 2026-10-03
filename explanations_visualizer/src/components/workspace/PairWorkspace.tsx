"use client";

// The shared ontology/explanation workspace for one compared pair: the question, aligned
// source and target meaning cards, the prepared comparison, and detail views (hierarchy
// navigation, evidence list, evidence graph). The exploration app, the explanation study
// condition and the tutorial render this same composition through their own data source.
// State ownership: the shell owns the pair, the open detail view and hierarchy focus; this
// component owns only transient inspection (selected evidence, open fact dialog, the full
// context of a browsed entity). The graph view belongs to `viewKey` and survives re-renders
// and tab switches. "Open full context" shows the shared card for any browsed entity over the
// same source (19 F17): original facts only when `original_context` is admitted, a generated
// description only when `entity_description` is admitted and the shell allows it.

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Dialog } from "@/components/common/Dialog";
import { Tabs } from "@/components/common/Tabs";
import { ComparisonPanel } from "@/components/explore/ComparisonPanel";
import { EntityCard } from "@/components/explore/EntityCard";
import { usePairEvidence } from "@/components/explore/evidenceData";
import { EvidenceList } from "@/components/explore/EvidenceList";
import type { CitedFact } from "@/components/explore/GeneratedBlock";
import { HierarchyBrowser, type FocusVia } from "@/components/explore/HierarchyBrowser";
import { FactInspectorProvider } from "@/components/workspace/FactInspector";
import type { EntityContext, EntityKind, EntityRef } from "@/lib/types";
import type { PairScope, Side } from "@/lib/workspace/types";
import { SidesProvider, useEntityContext, useWorkspace, useWorkspaceAction } from "@/lib/workspace/WorkspaceContext";

const EvidenceGraph = dynamic(() => import("@/components/explore/EvidenceGraph").then((module) => module.EvidenceGraph), {
  ssr: false,
  loading: () => <div className="graph-canvas skeleton" aria-busy="true" />,
});

/** Components a study publication can admit individually (final-form matrix codes). */
export type Component = "original_context" | "entity_description" | "hierarchy" | "evidence_table" | "evidence_graph" | "pair_comparison";
export const ALL_COMPONENTS: Component[] = ["original_context", "entity_description", "hierarchy", "evidence_table", "evidence_graph", "pair_comparison"];

export interface Focus {
  iri: string;
  kind: EntityKind;
}

export interface PairNavigation {
  source: Focus | null;
  target: Focus | null;
  set: (side: Side, focus: Focus | null) => void;
}

export interface ExtraTab {
  key: string;
  label: string;
  render: () => React.ReactNode;
}

/** Map every fact visible in the two contexts to a short citation label. */
export function citationIndex(source: EntityContext | undefined, target: EntityContext | undefined) {
  const index = new Map<string, CitedFact>();
  for (const [side, ctx] of [
    ["source", source],
    ["target", target],
  ] as const) {
    if (!ctx) continue;
    const prefix = side === "source" ? "Source" : "Target";
    const add = (items: { fact_id: string }[], name: string) => items.forEach((fact) => { if (!index.has(fact.fact_id)) index.set(fact.fact_id, { label: `${prefix} ${name}`, side }); });
    add(ctx.definitions.items, "definition");
    add(ctx.synonyms.items, "synonym");
    add(ctx.parents.items, "parent");
    for (const [category, page] of Object.entries(ctx.categories)) {
      add(page.items, category === "labels" ? "label" : category.replace(/_/g, " ").replace(/s$/, ""));
    }
    if (ctx.preferred_label.fact_id) add([{ fact_id: ctx.preferred_label.fact_id }], "label");
  }
  return index;
}

export function PairWorkspace({
  pair,
  viewKey,
  ontologyLabel,
  components = new Set(ALL_COMPONENTS),
  tab,
  onTab,
  order = ["hierarchy", "evidence", "graph"],
  extraTabs = [],
  allowNoTab = true,
  navigation,
  header,
  hierarchyFooter,
  cardTitles,
  compactCards = false,
  profileAllowed = () => true,
}: {
  pair: PairScope;
  viewKey: string;
  ontologyLabel: (ontology: string) => string;
  components?: Set<Component>;
  tab: string | null;
  onTab: (tab: string | null) => void;
  order?: string[];
  extraTabs?: ExtraTab[];
  allowNoTab?: boolean;
  navigation: PairNavigation;
  header?: (names: { source: string | null; target: string | null }) => React.ReactNode;
  hierarchyFooter?: React.ReactNode;
  cardTitles?: { source?: string; target?: string };
  compactCards?: boolean;
  /** Whether a generated description may be read for an entity (the study restricts it to focal entities). */
  profileAllowed?: (entity: EntityRef) => boolean;
}) {
  const workspace = useWorkspace();
  const report = useWorkspaceAction();
  const { source, target } = pair;
  const sourceCtx = useEntityContext(source);
  const targetCtx = useEntityContext(target);
  const cites = useMemo(() => citationIndex(sourceCtx.data, targetCtx.data), [sourceCtx.data, targetCtx.data]);
  const cite = useCallback((factId: string) => cites.get(factId), [cites]);
  const evidenceOpen = tab === "evidence" || tab === "graph";
  const evidence = usePairEvidence(evidenceOpen ? pair : null);
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null);
  const [contextFor, setContextFor] = useState<{ side: Side; entity: EntityRef } | null>(null);
  useEffect(() => setSelectedEvidence(null), [viewKey]);

  const builtIn: Record<string, { label: string; show: boolean }> = {
    hierarchy: { label: "Hierarchy", show: components.has("hierarchy") },
    evidence: { label: "Evidence", show: components.has("evidence_table") },
    graph: { label: "Evidence graph", show: components.has("evidence_graph") },
  };
  const tabs = order
    .map((key) => (builtIn[key] ? (builtIn[key].show ? { key, label: builtIn[key].label } : null) : extraTabs.find((extra) => extra.key === key) ?? null))
    .filter((item): item is { key: string; label: string } => Boolean(item));
  const activeTab = tab && tabs.some((item) => item.key === tab) ? tab : null;

  const focusFrom = (side: Side) => (iri: string, kind: EntityKind, via: FocusVia) => {
    navigation.set(side, { iri, kind });
    report({ type: "hierarchy_focus", side, iri, kind, via });
  };
  const openFromCard = (side: Side) => (iri: string) => {
    const ctx = side === "source" ? sourceCtx.data : targetCtx.data;
    const entity = side === "source" ? source : target;
    const parent = ctx?.parents.items.some((fact) => (fact.hierarchy_projection?.parent.iri ?? (fact.value?.term_type === "iri" ? fact.value.iri : null)) === iri);
    if (!components.has("hierarchy")) return;
    navigation.set(side, { iri, kind: parent ? entity.kind : "class" });
    report({ type: "hierarchy_focus", side, iri, kind: parent ? entity.kind : "class", via: "card" });
    onTab("hierarchy");
    report({ type: "tab_open", tab: "hierarchy" });
  };

  const openContext = (side: Side) => (iri: string, kind: EntityKind) => {
    const entity = { ontology_version_id: (side === "source" ? source : target).ontology_version_id, iri, kind };
    setContextFor({ side, entity });
    report({ type: "context_open", side, iri });
  };

  const names = { source: sourceCtx.data?.preferred_label.value ?? null, target: targetCtx.data?.preferred_label.value ?? null };
  const limitations = workspace.capabilities.limitations;

  return (
    <SidesProvider source={source.ontology_version_id} target={target.ontology_version_id}>
      <FactInspectorProvider>
        <div className="pair-workspace">
          {header?.(names)}
          {(components.has("original_context") || components.has("entity_description")) && (
            <div className="card-pair">
              {(["source", "target"] as const).map((side) => (
                <EntityCard
                  key={side}
                  side={side}
                  entity={side === "source" ? source : target}
                  context={side === "source" ? sourceCtx : targetCtx}
                  ontologyLabel={ontologyLabel((side === "source" ? source : target).ontology_version_id)}
                  cite={cite}
                  onOpenEntity={components.has("hierarchy") ? openFromCard(side) : undefined}
                  compact={compactCards}
                  titleOverride={cardTitles?.[side]}
                  showOriginal={components.has("original_context")}
                  showProfile={components.has("entity_description")}
                />
              ))}
            </div>
          )}

          {components.has("pair_comparison") && <ComparisonPanel source={source} target={target} cite={cite} />}

          {tabs.length > 0 && (
            <section className="card details" aria-label="Optional details">
              <Tabs
                label="Optional details"
                tabs={tabs}
                active={activeTab}
                allowNone={allowNoTab}
                onChange={(next) => {
                  onTab(next);
                  if (next) report({ type: "tab_open", tab: next });
                }}
                emptyNote={<p className="meta">Optional details stay closed until you open one. They never change the comparison above.</p>}
              >
                {(active) => {
                  if (active === "hierarchy")
                    return (
                      <div className="hierarchy-pair">
                        {(["source", "target"] as const).map((side) => {
                          const entity = side === "source" ? source : target;
                          const focus = navigation[side];
                          return (
                            <HierarchyBrowser
                              key={side}
                              side={side}
                              ontology={entity.ontology_version_id}
                              ontologyLabel={ontologyLabel(entity.ontology_version_id)}
                              focusIri={focus?.iri ?? entity.iri}
                              kind={focus?.kind ?? entity.kind}
                              pinnedIri={entity.iri}
                              pinnedKind={entity.kind}
                              onFocus={focusFrom(side)}
                              onOpenContext={components.has("original_context") ? openContext(side) : undefined}
                              reasonerStatus={workspace.capabilities.reasoner}
                              headingLevel={3}
                            />
                          );
                        })}
                        {hierarchyFooter}
                      </div>
                    );
                  if (active === "evidence")
                    return <EvidenceList state={evidence} selected={selectedEvidence} onSelect={components.has("evidence_graph") ? (id) => { setSelectedEvidence(id); onTab("graph"); } : undefined} />;
                  if (active === "graph")
                    return evidence.data ? (
                      evidence.data.items.length ? (
                        <EvidenceGraph key={`${workspace.key}|${viewKey}`} source={source} target={target} bundle={evidence.data} viewKey={`${workspace.key}|${viewKey}`} selected={selectedEvidence} onSelect={setSelectedEvidence} />
                      ) : (
                        <EvidenceList state={evidence} />
                      )
                    ) : (
                      <EvidenceList state={evidence} />
                    );
                  return extraTabs.find((extra) => extra.key === active)?.render() ?? null;
                }}
              </Tabs>
            </section>
          )}

          {contextFor && (
            <FullContextDialog
              side={contextFor.side}
              entity={contextFor.entity}
              ontologyLabel={ontologyLabel(contextFor.entity.ontology_version_id)}
              cite={cite}
              showProfile={components.has("entity_description") && profileAllowed(contextFor.entity)}
              onClose={() => setContextFor(null)}
              onOpenEntity={(iri, kind) => {
                navigation.set(contextFor.side, { iri, kind });
                report({ type: "hierarchy_focus", side: contextFor.side, iri, kind, via: "card" });
                setContextFor(null);
              }}
            />
          )}

          {limitations.length > 0 && (
            <details className="limits card">
              <summary>Limits of the prepared information ({limitations.length})</summary>
              <ul>
                {limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      </FactInspectorProvider>
    </SidesProvider>
  );
}

/** The shared full-context card for a browsed entity, over the workspace's own source. */
function FullContextDialog({
  side,
  entity,
  ontologyLabel,
  cite,
  showProfile,
  onClose,
  onOpenEntity,
}: {
  side: Side;
  entity: EntityRef;
  ontologyLabel: string;
  cite: (factId: string) => CitedFact | undefined;
  showProfile: boolean;
  onClose: () => void;
  onOpenEntity: (iri: string, kind: EntityKind) => void;
}) {
  const context = useEntityContext(entity);
  return (
    <Dialog title="Full context" onClose={onClose} wide>
      <FactInspectorProvider>
        <EntityCard
          side={side}
          entity={entity}
          context={context}
          ontologyLabel={ontologyLabel}
          cite={cite}
          showProfile={showProfile}
          onOpenEntity={(iri) => {
            const parent = context.data?.parents.items.some((fact) => (fact.hierarchy_projection?.parent.iri ?? (fact.value?.term_type === "iri" ? fact.value.iri : null)) === iri);
            onOpenEntity(iri, parent ? entity.kind : "class");
          }}
        />
      </FactInspectorProvider>
    </Dialog>
  );
}
