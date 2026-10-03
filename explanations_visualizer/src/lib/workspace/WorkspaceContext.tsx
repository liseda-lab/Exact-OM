"use client";

// Delivers one workspace data source, its label source and an action observer to every
// shared component. The shell decides which product it is; the components never branch on it.

import { createContext, useCallback, useContext, useMemo } from "react";

import { LabelSourceContext } from "@/lib/labelSource";
import type { EntityRef } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import type { FactRef, Side, WorkspaceAction, WorkspaceSource } from "@/lib/workspace/types";

const SourceContext = createContext<WorkspaceSource | null>(null);
const ActionContext = createContext<(action: WorkspaceAction) => void>(() => undefined);
/** Which ontology is the source side and which the target side of the compared pair. */
const SidesContext = createContext<Record<string, Side>>({});

export function WorkspaceProvider({
  source,
  onAction,
  children,
}: {
  source: WorkspaceSource;
  onAction?: (action: WorkspaceAction) => void;
  children: React.ReactNode;
}) {
  const report = useCallback((action: WorkspaceAction) => onAction?.(action), [onAction]);
  const labels = source.labels;
  const content = (
    <SourceContext.Provider value={source}>
      <ActionContext.Provider value={report}>{children}</ActionContext.Provider>
    </SourceContext.Provider>
  );
  return labels ? <LabelSourceContext.Provider value={labels}>{content}</LabelSourceContext.Provider> : content;
}

/** Declares which ontology is the source side and which the target side of a pair. */
export function SidesProvider({ source, target, children }: { source: string; target: string; children: React.ReactNode }) {
  const sides = useMemo<Record<string, Side>>(() => ({ [source]: "source", [target]: "target" }), [source, target]);
  return <SidesContext.Provider value={sides}>{children}</SidesContext.Provider>;
}

export function useWorkspace(): WorkspaceSource {
  const source = useContext(SourceContext);
  if (!source) throw new Error("Workspace components need a WorkspaceProvider.");
  return source;
}

export function useWorkspaceAction() {
  return useContext(ActionContext);
}

export function useSideOf() {
  const sides = useContext(SidesContext);
  return useCallback((ontology: string | null | undefined): Side | null => (ontology ? sides[ontology] ?? null : null), [sides]);
}

export function entitySignature(entity: EntityRef | null | undefined): string | null {
  return entity ? `${entity.ontology_version_id}|${entity.kind}|${entity.iri}` : null;
}

export function useEntityContext(entity: EntityRef | null) {
  const source = useWorkspace();
  const signature = entitySignature(entity);
  return useAsync(signature ? `${source.key}|ctx|${signature}` : null, (signal) => source.entityContext(entity!, signal));
}

export function useExplanation(task: "entity_profile" | "pair_comparison", entities: EntityRef[] | null) {
  const source = useWorkspace();
  const signature = entities?.map(entitySignature).join(">") ?? null;
  return useAsync(signature ? `${source.key}|expl|${task}|${signature}` : null, (signal) => source.explanation(task, entities!, signal));
}

export function useResolvedFact(ref: FactRef | null) {
  const source = useWorkspace();
  const signature = ref ? `${ref.factId}|${ref.ontologies.join(",")}|${entitySignature(ref.subject) ?? ""}` : null;
  return useAsync(signature ? `${source.key}|fact|${signature}` : null, (signal) => source.fact(ref!, signal));
}
