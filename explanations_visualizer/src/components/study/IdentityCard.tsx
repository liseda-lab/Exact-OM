"use client";

// Identity only: the side, label and IRI of an entity, with an IRI copy action. Used where
// the condition provides no explanation content (the baseline and its practice lesson).

import { SideMarker } from "@/components/common/Icons";
import { CopyIri } from "@/components/explore/EntityCard";
import { sideTitle } from "@/lib/iri";
import type { EntityRef } from "@/lib/types";

export function IdentityCard({ side, entity, label, ontology, title, onCopied }: { side: "source" | "target"; entity: EntityRef; label: string; ontology: string; title?: string; onCopied?: (method: "clipboard" | "selected") => void }) {
  return (
    <article className={`entity-card entity-card-${side} compact identity-card`} aria-label={`${title ?? sideTitle(side, entity.kind)}: ${label}`}>
      <div className="entity-card-head">
        <SideMarker side={side} />
        <span className={`eyebrow text-${side}`}>{title ?? sideTitle(side, entity.kind)}</span>
        <span className="meta">{ontology}</span>
        <span className="entity-card-spacer" />
        <CopyIri iri={entity.iri} label={`Copy IRI of ${label}`} onCopied={onCopied} />
      </div>
      <div className="entity-title-block">
        <h2 className="entity-title">{label}</h2>
        <span className="iri">{entity.iri}</span>
      </div>
    </article>
  );
}
