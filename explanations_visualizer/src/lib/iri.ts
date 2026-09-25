// Display-only identifier helpers. Full IRIs remain the identity everywhere; CURIEs are
// a reading aid and are never parsed back into identities.

const PREFIXES: [string, string][] = [
  ["http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#", "NCIT:"],
  ["http://www.w3.org/2000/01/rdf-schema#", "rdfs:"],
  ["http://www.w3.org/2002/07/owl#", "owl:"],
  ["http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"],
  ["http://www.w3.org/2001/XMLSchema#", "xsd:"],
  ["http://www.geneontology.org/formats/oboInOwl#", "oboInOwl:"],
  ["http://purl.org/sig/ont/fma/", "FMA:"],
];

const OBO = "http://purl.obolibrary.org/obo/";

export function curie(iri: string): string {
  if (iri.startsWith(OBO)) {
    const local = iri.slice(OBO.length);
    const match = /^([A-Za-z][A-Za-z0-9]*)_(.+)$/.exec(local);
    return match ? `${match[1]}:${match[2]}` : `obo:${local}`;
  }
  for (const [namespace, prefix] of PREFIXES) {
    if (iri.startsWith(namespace)) return prefix + iri.slice(namespace.length);
  }
  return iri;
}

export function localName(iri: string): string {
  const trimmed = iri.replace(/[#/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("#"), trimmed.lastIndexOf("/"));
  return index >= 0 ? trimmed.slice(index + 1) : trimmed;
}

const PREDICATE_NAMES: Record<string, string> = {
  "http://www.w3.org/2000/01/rdf-schema#label": "label",
  "http://www.w3.org/2000/01/rdf-schema#comment": "comment",
  "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P108": "NCIT P108 preferred name",
  "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P97": "NCIT P97 definition",
  "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P325": "NCIT P325 alternate definition",
  "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P90": "NCIT P90 full synonym",
  "http://purl.obolibrary.org/obo/IAO_0000115": "IAO_0000115 definition",
  "http://www.geneontology.org/formats/oboInOwl#hasDefinition": "oboInOwl definition",
  "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym": "exact synonym",
  "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym": "related synonym",
  "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym": "broad synonym",
  "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym": "narrow synonym",
  "http://www.geneontology.org/formats/oboInOwl#hasDbXref": "database cross-reference",
};

/** The exact predicate, named for reading; unknown predicates show their CURIE. */
export function predicateName(iri: string | null | undefined): string {
  if (!iri) return "no predicate recorded";
  return PREDICATE_NAMES[iri] ?? curie(iri);
}

export function shortHash(value: string | null | undefined, length = 12): string {
  if (!value) return "";
  const bare = value.replace(/^sha256:/, "");
  return `${bare.slice(0, length)}…`;
}

export function entityKey(entity: { ontology_version_id: string; iri: string; kind: string }): string {
  return `${entity.ontology_version_id}|${entity.kind}|${entity.iri}`;
}

export function sameEntity(
  a: { ontology_version_id: string; iri: string; kind: string } | null | undefined,
  b: { ontology_version_id: string; iri: string; kind: string } | null | undefined,
): boolean {
  return Boolean(a && b && a.ontology_version_id === b.ontology_version_id && a.iri === b.iri && a.kind === b.kind);
}

export const KIND_NAMES: Record<string, string> = {
  class: "class",
  object_property: "object property",
  data_property: "data property",
  individual: "individual",
};

/** "Source class", "Target object property", … — never a generic "Source". */
export function sideTitle(side: "source" | "target", kind: string): string {
  const role = side === "source" ? "Source" : "Target";
  return `${role} ${KIND_NAMES[kind] ?? kind}`;
}
