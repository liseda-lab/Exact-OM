// Proposed exact-study/2.0 tutorial material (14, 16 B3): two tiny invented ontologies, one
// practice case, prepared claims and evidence in the study-resource shape, six lessons and
// the five comprehension items with their answer keys and feedback. Everything is synthetic
// and disjoint from scored cases and the public demo: example.org IRIs, no matcher, no model.
// The backend owner freezes (and may revise) this material in the v2 publication; the
// frontend never certifies a publication with it as a fallback. Grading keys are publication
// data and stay server-side; only `tutorialPublic()` is ever sent to a participant.

import type { EntityRef } from "../../lib/types";
import type { AssessmentItem, AssessmentResponse, ExplanationResource, GroundedClaim, OriginalFact, PublicAsset, StudyCandidate, StudyEvidenceLink, StudyHierarchyEdge, TutorialLesson, TutorialPublic } from "../types";

export const TUTORIAL_VERSION = "tutorial/2.0-draft-1";
export const ONT_A = "synthetic:practice-ontology-a/1";
export const ONT_B = "synthetic:practice-ontology-b/1";
const A = "https://example.org/practice/a#";
const B = "https://example.org/practice/b#";
const LABEL = "http://www.w3.org/2000/01/rdf-schema#label";
const DEFINITION = "http://purl.obolibrary.org/obo/IAO_0000115";
const EXACT = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym";
const MANIFEST = "5".repeat(64);

type Node = Record<string, unknown>;
const ref = (ontology: string, iri: string, kind: EntityRef["kind"] = "class"): EntityRef => ({ ontology_version_id: ontology, iri, kind });
const cls = (iri: string): Node => ({ type: "Class", iri: { type: "IRI", value: iri }, kind: "class" });
const prop = (iri: string): Node => ({ type: "ObjectProperty", iri: { type: "IRI", value: iri }, kind: "object_property" });
const subClass = (child: string, parent: Node): Node => ({ type: "SubClassOf", annotations: [], sub_class: cls(child), super_class: parent });
const some = (property: string, filler: string): Node => ({ type: "ObjectSomeValuesFrom", property: prop(property), filler: cls(filler) });
const only = (property: string, filler: string): Node => ({ type: "ObjectAllValuesFrom", property: prop(property), filler: cls(filler) });
const atLeast = (n: number, property: string, filler: string): Node => ({ type: "ObjectMinCardinality", cardinality: n, property: prop(property), filler: cls(filler) });
const exactly = (n: number, property: string, filler: string): Node => ({ type: "ObjectExactCardinality", cardinality: n, property: prop(property), filler: cls(filler) });

const facts: OriginalFact[] = [];
const hierarchy: StudyHierarchyEdge[] = [];
const ids = new Map<string, string>();

function literal(id: string, subject: EntityRef, predicate: string, category: string, text: string) {
  facts.push({ fact_id: id, subject, predicate_iri: predicate, category, value: { term_type: "literal", lexical_form: text, datatype: null, language: "en" }, qualifiers: [], axiom_ref: id, origins: [], interpretation: "asserted" });
  return id;
}

function axiom(id: string, subjects: EntityRef[], category: string, ast: Node) {
  for (const subject of subjects) facts.push({ fact_id: id, subject, predicate_iri: null, category, value: { term_type: "expression_ref", expression_id: id, ast }, qualifiers: [], axiom_ref: id, origins: [], interpretation: "asserted" });
  return id;
}

interface Spec {
  key: string;
  ontology: string;
  iri: string;
  label: string;
  definition?: string;
  synonyms?: string[];
  parents?: string[];
  restrictions?: Node[];
  kind?: EntityRef["kind"];
}

const SPECS: Spec[] = [
  { key: "a.object", ontology: ONT_A, iri: `${A}Object`, label: "object", definition: "Anything that can be handled as a single thing." },
  { key: "a.container", ontology: ONT_A, iri: `${A}Container`, label: "container", definition: "An object whose purpose is to hold other objects.", parents: ["a.object"] },
  { key: "a.crate", ontology: ONT_A, iri: `${A}Crate`, label: "crate", definition: "A rigid container with a lid.", parents: ["a.container"] },
  { key: "a.lidded", ontology: ONT_A, iri: `${A}LiddedObject`, label: "lidded object", definition: "An object that has a lid.", parents: ["a.object"] },
  { key: "a.piece", ontology: ONT_A, iri: `${A}Piece`, label: "piece", definition: "A small separate object.", parents: ["a.object"] },
  { key: "a.round", ontology: ONT_A, iri: `${A}RoundPiece`, label: "round piece", definition: "A piece whose outline is round.", parents: ["a.piece"] },
  { key: "a.lidstate", ontology: ONT_A, iri: `${A}LidState`, label: "lid state", definition: "Whether a lid is open or closed.", parents: ["a.object"] },
  { key: "a.closedlid", ontology: ONT_A, iri: `${A}ClosedLidState`, label: "closed lid state", definition: "The state of a lid that is shut.", parents: ["a.lidstate"] },
  {
    key: "a.crpc",
    ontology: ONT_A,
    iri: `${A}ClosedRoundPieceCrate`,
    label: "closed crate of round pieces",
    definition: "A crate whose lid is closed, that holds at least two pieces, and all of whose contents are round pieces.",
    synonyms: ["sealed round-piece crate"],
    parents: ["a.crate", "a.lidded"],
    restrictions: [some(`${A}hasLidState`, `${A}ClosedLidState`), only(`${A}contains`, `${A}RoundPiece`), atLeast(2, `${A}contains`, `${A}Piece`)],
  },
  { key: "a.wooden", ontology: ONT_A, iri: `${A}WoodenCrate`, label: "wooden crate", parents: ["a.crate"] },
  { key: "a.open", ontology: ONT_A, iri: `${A}OpenCrate`, label: "open crate", definition: "A crate whose lid is open.", parents: ["a.crate"] },
  { key: "a.contains", ontology: ONT_A, iri: `${A}contains`, label: "contains", kind: "object_property" },
  { key: "a.haslid", ontology: ONT_A, iri: `${A}hasLidState`, label: "has lid state", kind: "object_property" },
  { key: "b.item", ontology: ONT_B, iri: `${B}Item`, label: "item", definition: "A physical thing." },
  { key: "b.container", ontology: ONT_B, iri: `${B}Container`, label: "container", definition: "An item used to hold other items.", parents: ["b.item"] },
  { key: "b.crate", ontology: ONT_B, iri: `${B}Crate`, label: "crate", definition: "A box-like container with a lid.", synonyms: ["crate"], parents: ["b.container"] },
  { key: "b.round", ontology: ONT_B, iri: `${B}RoundItem`, label: "round item", definition: "An item with a round outline.", parents: ["b.item"] },
  { key: "b.lidstate", ontology: ONT_B, iri: `${B}LidState`, label: "lid state", parents: ["b.item"] },
  { key: "b.closed", ontology: ONT_B, iri: `${B}Closed`, label: "closed", definition: "A lid state in which the lid is shut.", parents: ["b.lidstate"] },
  { key: "b.open", ontology: ONT_B, iri: `${B}Open`, label: "open", definition: "A lid state in which the lid is not shut.", parents: ["b.lidstate"] },
  {
    key: "b.c1",
    ontology: ONT_B,
    iri: `${B}CrateWithRoundPiece`,
    label: "crate with a round piece",
    definition: "A crate that holds some round item. Its other contents and lid position are not specified.",
    parents: ["b.crate"],
    restrictions: [some(`${B}holds`, `${B}RoundItem`)],
  },
  {
    key: "b.c2",
    ontology: ONT_B,
    iri: `${B}ClosedRoundPieceCollectionCrate`,
    label: "closed round-piece collection crate",
    definition: "A crate with a closed lid that holds at least two items, every one of which is round.",
    parents: ["b.crate"],
    restrictions: [some(`${B}lidState`, `${B}Closed`), only(`${B}holds`, `${B}RoundItem`), atLeast(2, `${B}holds`, `${B}Item`)],
  },
  {
    key: "b.c3",
    ontology: ONT_B,
    iri: `${B}ClosedTwoPieceCrate`,
    label: "closed two-piece crate",
    definition: "A closed crate that holds exactly two round items.",
    parents: ["b.crate"],
    restrictions: [some(`${B}lidState`, `${B}Closed`), exactly(2, `${B}holds`, `${B}RoundItem`)],
  },
  {
    key: "b.c5",
    ontology: ONT_B,
    iri: `${B}OpenRoundPieceCrate`,
    label: "open round-piece crate",
    parents: ["b.crate"],
    restrictions: [some(`${B}lidState`, `${B}Open`), only(`${B}holds`, `${B}RoundItem`), atLeast(2, `${B}holds`, `${B}Item`)],
  },
  { key: "b.holds", ontology: ONT_B, iri: `${B}holds`, label: "holds", kind: "object_property" },
  { key: "b.lidprop", ontology: ONT_B, iri: `${B}lidState`, label: "lid state", kind: "object_property" },
];

const byKey = new Map(SPECS.map((spec) => [spec.key, spec]));
export const ENTITY: Record<string, EntityRef> = Object.fromEntries(SPECS.map((spec) => [spec.key, ref(spec.ontology, spec.iri, spec.kind ?? "class")]));
const FOCAL = new Set(["a.crpc", "b.c1", "b.c2", "b.c3", "b.crate", "b.c5"]);

for (const spec of SPECS) {
  const subject = ENTITY[spec.key];
  ids.set(`${spec.key}.label`, literal(`practice.${spec.key}.label`, subject, LABEL, "labels", spec.label));
  if (spec.definition) ids.set(`${spec.key}.definition`, literal(`practice.${spec.key}.definition`, subject, DEFINITION, "definitions", spec.definition));
  (spec.synonyms ?? []).forEach((text, index) => ids.set(`${spec.key}.synonym.${index}`, literal(`practice.${spec.key}.synonym.${index}`, subject, EXACT, "synonyms", text)));
  for (const parentKey of spec.parents ?? []) {
    const parent = byKey.get(parentKey)!;
    const id = `practice.${spec.key}.subclass.${parentKey}`;
    // A subclass axiom between two compared entities is one shared original record.
    const subjects = FOCAL.has(parentKey) ? [subject, ENTITY[parentKey]] : [subject];
    ids.set(`${spec.key}.parent.${parentKey}`, axiom(id, subjects, "hierarchy", subClass(spec.iri, cls(parent.iri))));
    hierarchy.push({ child: subject, parent: ENTITY[parentKey], basis: "literal_asserted", fact_ids: [id] });
  }
  (spec.restrictions ?? []).forEach((restriction, index) => ids.set(`${spec.key}.restriction.${index}`, axiom(`practice.${spec.key}.restriction.${index}`, [subject], "restrictions", subClass(spec.iri, restriction))));
}

const fact = (key: string) => {
  const id = ids.get(key);
  if (!id) throw new Error(`Unknown practice fact ${key}`);
  return id;
};

function claim(id: string, text: string, factIds: string[], category: string, scoped: EntityRef[], grounding: GroundedClaim["grounding"] = "exact_extract"): GroundedClaim {
  return { claim_id: `practice.claim.${id}`, text, fact_ids: factIds, category, grounding, scoped_entities: scoped, generation_manifest_sha256: MANIFEST };
}

const SOURCE_KEY = "a.crpc";
const CANDIDATE_KEYS = ["b.c1", "b.c2", "b.c3", "b.crate", "b.c5"];
const SCORES = [0.91, 0.88, 0.84, 0.8, 0.77];

const profiles: GroundedClaim[] = [];
for (const key of [SOURCE_KEY, ...CANDIDATE_KEYS]) {
  const spec = byKey.get(key)!;
  if (spec.definition) profiles.push(claim(`${key}.meaning`, spec.definition, [fact(`${key}.definition`)], "meaning", [ENTITY[key]]));
  else profiles.push(claim(`${key}.label`, spec.label, [fact(`${key}.label`)], "key_fact", [ENTITY[key]]));
  (spec.synonyms ?? []).forEach((text, index) => profiles.push(claim(`${key}.synonym.${index}`, text, [fact(`${key}.synonym.${index}`)], "key_fact", [ENTITY[key]])));
}

// Comparisons use the backend's fixed wording templates (exact_inspect.generation.comparison_templates).
const comparisons: GroundedClaim[] = [];
const source = byKey.get(SOURCE_KEY)!;
for (const key of CANDIDATE_KEYS) {
  const target = byKey.get(key)!;
  const pair = [ENTITY[SOURCE_KEY], ENTITY[key]];
  for (const [name, a, b, fa, fb] of [
    ["definition", source.definition, target.definition, `${SOURCE_KEY}.definition`, `${key}.definition`],
    ["label", source.label, target.label, `${SOURCE_KEY}.label`, `${key}.label`],
  ] as const) {
    if (a && b) {
      const agree = a === b;
      const text = agree
        ? `Both entities have this recorded ${name}: “${a}”. Shared wording alone does not establish equivalence.`
        : `The source ${name} is “${a}”. The target ${name} is “${b}”. Different wording alone does not establish incompatibility.`;
      const c = claim(`${key}.${name}`, text, [fact(fa), fact(fb)], agree ? "agreement" : "difference", pair, "semantic_template");
      comparisons.push({ ...c, packet_fact_ids: [fact(fa), fact(fb)], packet_fact_subjects: pair });
    } else if (a || b) {
      const side = a ? "source" : "target";
      const other = a ? "target" : "source";
      const id = fact(a ? fa : fb);
      const c = claim(`${key}.${name}.onesided`, `The ${side} has a recorded ${name} in this fact packet; the ${other} does not. One-sided information does not establish incompatibility.`, [id], "difference", pair, "semantic_template");
      comparisons.push({ ...c, packet_fact_ids: [id], packet_fact_subjects: [a ? pair[0] : pair[1]] });
    }
  }
}

const evidence: StudyEvidenceLink[] = [];
CANDIDATE_KEYS.forEach((key, index) => {
  const candidateId = `practice-c${index + 1}`;
  const targetParent = key === "b.crate" ? fact("b.crate.parent.b.container") : fact(`${key}.parent.b.crate`);
  evidence.push({ evidence_id: `practice.ev.${index + 1}.h.s`, candidate_id: candidateId, fact_ids: [fact("a.crpc.parent.a.crate")], channel: "hierarchy", role: "source", interpretation: "projected", status: "available" });
  evidence.push({ evidence_id: `practice.ev.${index + 1}.h.t`, candidate_id: candidateId, fact_ids: [targetParent], channel: "hierarchy", role: "target", interpretation: "projected", status: "available" });
  evidence.push({ evidence_id: `practice.ev.${index + 1}.r.s`, candidate_id: candidateId, fact_ids: [fact("a.crpc.restriction.1")], channel: "relational", role: "source", interpretation: "projected", status: "available" });
  if (byKey.get(key)!.restrictions?.length) evidence.push({ evidence_id: `practice.ev.${index + 1}.r.t`, candidate_id: candidateId, fact_ids: [fact(`${key}.restriction.0`)], channel: "relational", role: "target", interpretation: "projected", status: "available" });
  evidence.push({ evidence_id: `practice.ev.${index + 1}.l.t`, candidate_id: candidateId, fact_ids: [fact(`${key}.label`)], channel: "lexical", role: "target", interpretation: "projected", status: "available" });
});

export const TUTORIAL_RESOURCE: ExplanationResource = {
  artifact_type: "study_explanation",
  contract_version: "exact-study/2.0",
  entities: SPECS.map((spec) => ENTITY[spec.key]),
  facts,
  referenced_labels: [],
  referenced_labels_truncated: false,
  entity_profiles: profiles,
  pair_comparison: comparisons,
  hierarchy,
  evidence,
  limitations: ["Synthetic practice material: every name, fact, score, description and feature is invented for training. No matcher or model was run."],
  capabilities: { context: "available", profiles: "available", hierarchy: "available", evidence: "available", comparison: "available" },
};

export const TUTORIAL_RESOURCE_ID = "practice-explanations";

export const TUTORIAL_CANDIDATES: StudyCandidate[] = CANDIDATE_KEYS.map((key, index) => ({
  candidate_id: `practice-c${index + 1}`,
  entity: ENTITY[key],
  label: byKey.get(key)!.label,
  score: SCORES[index],
  score_meaning: "Illustrative practice score. In scored cases a matching score is the matcher's suggestion, never a probability of being correct.",
  display_position: index + 1,
}));

/** Minimal OWL Functional Syntax for a synthetic download. */
export function ontologyDocument(ontology: string): string {
  const base = ontology === ONT_A ? A : B;
  const lines = [`Prefix(:=<${base}>)`, `Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)`, `Ontology(<${base.replace(/#$/, "")}>`];
  const iri = (value: string) => `<${value}>`;
  const term = (node: Node): string => {
    const type = node.type as string;
    if (type === "Class" || type === "ObjectProperty") return iri(((node.iri as Node).value as string));
    if (type === "ObjectSomeValuesFrom" || type === "ObjectAllValuesFrom") return `${type}(${term(node.property as Node)} ${term(node.filler as Node)})`;
    if (type.endsWith("Cardinality")) return `${type}(${node.cardinality} ${term(node.property as Node)} ${term(node.filler as Node)})`;
    return "";
  };
  for (const spec of SPECS.filter((item) => item.ontology === ontology)) {
    lines.push(`  Declaration(${spec.kind === "object_property" ? "ObjectProperty" : "Class"}(${iri(spec.iri)}))`);
  }
  for (const item of facts.filter((value, index, all) => value.subject.ontology_version_id === ontology && all.findIndex((other) => other.fact_id === value.fact_id) === index)) {
    if (item.value.term_type === "literal") lines.push(`  AnnotationAssertion(${iri(item.predicate_iri!)} ${iri(item.subject.iri)} ${JSON.stringify(item.value.lexical_form)}@en)`);
    else if (item.value.ast) lines.push(`  SubClassOf(${term((item.value.ast as Node).sub_class as Node)} ${term((item.value.ast as Node).super_class as Node)})`);
  }
  lines.push(")");
  return `${lines.join("\n")}\n`;
}

export const TUTORIAL_DOWNLOADS: (Omit<PublicAsset, "sha256" | "size_bytes"> & { ontology: string })[] = [
  { asset_id: "practice-ontology-a", kind: "ontology", media_type: "application/owl-functional", title: "Practice ontology A (synthetic)", role: "source", ontology_version_id: ONT_A, version_label: "practice-1", ontology: ONT_A },
  { asset_id: "practice-ontology-b", kind: "ontology", media_type: "application/owl-functional", title: "Practice ontology B (synthetic)", role: "target", ontology_version_id: ONT_B, version_label: "practice-1", ontology: ONT_B },
];

export const LESSONS: TutorialLesson[] = [
  {
    lesson_id: "identity",
    title: "The source and its candidates",
    view: "explanation",
    steps: [
      "This practice case uses two small invented ontologies. The source concept comes from one; the five candidates come from the other.",
      "Candidates are listed in the system’s initial order, with a matching score. The initial position is a suggestion, not an answer: your rank is whatever you decide.",
      "Inspect a different candidate, then go back to one you inspected before. Inspecting a candidate never ranks it.",
    ],
    requirements: [
      { requirement_id: "identity.inspect", action: "inspect_other_candidate", label: "Inspect a candidate other than the first" },
      { requirement_id: "identity.return", action: "return_to_candidate", label: "Return to a candidate you inspected before" },
    ],
  },
  {
    lesson_id: "context",
    title: "Context: parents, children and missing information",
    view: "explanation",
    steps: [
      "Open Hierarchy. The source and the candidate each have their own browser.",
      "Search the source ontology for “crate” and choose a result. Then open one of its parents and one of its children.",
      "The source concept has two parents (multiple inheritance). Some candidates have no definition; the page says so instead of leaving a gap, and missing information is not a contradiction.",
      "Use “Return to …” to come back to the compared entity.",
    ],
    requirements: [
      { requirement_id: "context.search", action: "search_entity", label: "Find an entity with search" },
      { requirement_id: "context.parent", action: "navigate_parent", label: "Open a parent" },
      { requirement_id: "context.child", action: "navigate_child", label: "Open a child" },
      { requirement_id: "context.return", action: "return_to_compared", label: "Return to the compared entity" },
    ],
  },
  {
    lesson_id: "evidence",
    title: "Where each statement comes from",
    view: "explanation",
    steps: [
      "Statements marked “Original” come from the ontology. Generated text sits in a dashed “Generated” frame and cites the records it rests on.",
      "Open a citation under a generated description to see the exact original record.",
      "In a card, use “Show original axiom” to see a restriction in its original form.",
      "Open Evidence and select a feature (“Show in graph”), or use “Show where it appears” from a citation, to find the same record in the list.",
    ],
    requirements: [
      { requirement_id: "evidence.citation", action: "open_citation", label: "Open a citation" },
      { requirement_id: "evidence.axiom", action: "open_original_axiom", label: "Show an original axiom" },
      { requirement_id: "evidence.locate", action: "locate_in_evidence_list", label: "Find a record in the evidence list" },
    ],
  },
  {
    lesson_id: "graph",
    title: "The optional evidence graph",
    view: "explanation",
    optional: true,
    steps: [
      "The evidence graph shows the same items as the evidence list. The list is a complete alternative if you prefer it.",
      "Select a line in the graph to read what it means, or select an item in the list. Try zoom or “Fit all”.",
      "With the graph or list open, add or move a candidate in your practice answer: the view stays where it was.",
    ],
    requirements: [
      { requirement_id: "graph.inspect", action: "inspect_graph_or_list", label: "Inspect a line in the graph or an item in the list" },
      { requirement_id: "graph.view", action: "change_graph_view", label: "Zoom or fit the graph", alternatives: ["inspect_graph_or_list"] },
      { requirement_id: "graph.rank", action: "rank_with_details_open", label: "Change your practice answer with the graph or list open" },
    ],
  },
  {
    lesson_id: "answers",
    title: "Your answer",
    view: "explanation",
    steps: [
      "Add the candidates you consider equivalent, best first. Move one, remove one and use Undo.",
      "Try “Keep initial order”, then make a partial ranking of one to four candidates and check it.",
      "Try “None of these” and “Insufficient information”: they are different answers. An empty answer is never submitted.",
      "This source keeps its qualifiers: a closed lid, only round pieces, at least two pieces. A broader or narrower candidate is not the same meaning.",
    ],
    requirements: [
      { requirement_id: "answers.add", action: "add_rank", label: "Add a candidate" },
      { requirement_id: "answers.move", action: "move_rank", label: "Move a candidate" },
      { requirement_id: "answers.remove", action: "remove_rank", label: "Remove a candidate" },
      { requirement_id: "answers.undo", action: "undo_rank", label: "Undo a change" },
      { requirement_id: "answers.keep", action: "keep_initial_order", label: "Use “Keep initial order”" },
      { requirement_id: "answers.partial", action: "check_partial_ranking", label: "Check a partial ranking" },
      { requirement_id: "answers.none", action: "choose_none", label: "Choose “None of these”" },
      { requirement_id: "answers.insufficient", action: "choose_insufficient", label: "Choose “Insufficient information”" },
    ],
  },
  {
    lesson_id: "baseline",
    title: "The block without explanations, and your methods",
    view: "baseline",
    steps: [
      "In one block of cases the study shows only names, IRIs and scores. You can inspect the ontologies in any way you like, or not at all.",
      "Copy an IRI, and open “Ontology files” to find the two downloads. Downloading is optional.",
      "After each case you will say which outside methods, if any, you used for it. Practise one report with two methods and one report with No.",
    ],
    requirements: [
      { requirement_id: "baseline.copy", action: "copy_iri", label: "Copy an IRI" },
      { requirement_id: "baseline.downloads", action: "locate_downloads", label: "Find the two downloads" },
      { requirement_id: "baseline.multi", action: "report_multiple_methods", label: "Practise a report with two or more methods" },
      { requirement_id: "baseline.no", action: "report_no_methods", label: "Practise a report with No" },
    ],
  },
];

export const ASSESSMENT: AssessmentItem[] = [
  {
    question_id: "score_meaning",
    title: "Matching scores",
    prompt: "A candidate is first in the initial order and has the highest matching score. What does this tell you?",
    kind: "single",
    options: [
      { code: "advice", label: "The matcher suggests it; I should inspect the evidence." },
      { code: "certain", label: "It must be equivalent." },
      { code: "probability", label: "The score is necessarily a probability of being correct." },
    ],
    lesson_id: "identity",
  },
  {
    question_id: "equivalence_scope",
    title: "Same meaning?",
    prompt: "In a practice ontology, the source means red circles. The candidate means circles of any colour and explicitly includes blue circles. Are these the same concept?",
    kind: "single",
    options: [
      { code: "same", label: "Yes." },
      { code: "broader", label: "No, the candidate is broader." },
      { code: "unrelated", label: "No, they are unrelated." },
    ],
    lesson_id: "answers",
  },
  {
    question_id: "response_states",
    title: "Three kinds of answer",
    prompt: "Match each situation to the answer it describes.",
    kind: "match",
    rows: [
      { row_id: "judged_none", label: "You judged that none of the displayed candidates is equivalent." },
      { row_id: "cannot_judge", label: "You cannot make a justified judgment." },
      { row_id: "not_answered", label: "You have not chosen or submitted an answer." },
    ],
    row_options: [
      { code: "none_of_these", label: "None of these" },
      { code: "insufficient_evidence", label: "Insufficient information" },
      { code: "unanswered", label: "An empty, unsubmitted draft" },
    ],
    lesson_id: "answers",
  },
  {
    question_id: "evidence_origin",
    title: "Where information comes from",
    prompt: "Match each kind of information to its origin.",
    kind: "match_and_single",
    rows: [
      { row_id: "statement", label: "A statement in the ontology" },
      { row_id: "feature", label: "A feature selected or projected by Exact" },
      { row_id: "summary", label: "A prepared generated summary" },
    ],
    row_options: [
      { code: "original", label: "Original ontology fact" },
      { code: "matcher", label: "Matcher evidence" },
      { code: "generated", label: "Generated text" },
    ],
    part_b: {
      prompt: "The candidate has no definition in the loaded scope. Does that alone establish a contradiction with the source?",
      options: [
        { code: "yes", label: "Yes" },
        { code: "no", label: "No" },
      ],
    },
    lesson_id: "evidence",
  },
  {
    question_id: "external_methods",
    title: "Outside inspection",
    prompt: "Which statements are true? Select all that apply.",
    kind: "multiple",
    options: [
      { code: "optional", label: "Inspecting the ontologies outside the study pages is optional." },
      { code: "combine_change", label: "I may combine methods and change them between cases." },
      { code: "per_case", label: "After each source-and-candidates case, I report the methods I actually used." },
      { code: "protege_required", label: "I must use Protégé." },
      { code: "locked", label: "I must keep one method for the whole study." },
    ],
    lesson_id: "baseline",
  },
];

/** Answer keys and feedback: tutorial training material, never a research case key. */
export const ASSESSMENT_KEYS: Record<string, { correct: AssessmentResponse; right: string; wrong: string }> = {
  score_meaning: {
    correct: { choice: "advice" },
    right: "Right. The initial position and the matching score are recorded advice from the matcher. Neither proves equivalence, and the score is not a calibrated probability.",
    wrong: "Not quite. A first-place, high-scoring candidate can still be wrong, and the score is not a probability of being correct. Treat it as a suggestion and inspect the evidence.",
  },
  equivalence_scope: {
    correct: { choice: "broader" },
    right: "Right. Circles of any colour include red circles but also blue ones, so the candidate is broader. Overlap or a broader category is not the same meaning.",
    wrong: "Not quite. Compare the two definitions: the candidate also covers blue circles, so it is broader than the source. A broader or overlapping concept is related, but it is not equivalent.",
  },
  response_states: {
    correct: { matches: { judged_none: "none_of_these", cannot_judge: "insufficient_evidence", not_answered: "unanswered" } },
    right: "Right. None of these is an explicit judgment, Insufficient information says you cannot judge, and an empty draft is not a submitted answer.",
    wrong: "Not quite. None of these is a judgment that no candidate is equivalent. Insufficient information means you cannot make a justified judgment. An empty draft has not been answered or submitted at all.",
  },
  evidence_origin: {
    correct: { matches: { statement: "original", feature: "matcher", summary: "generated" }, part_b: "no" },
    right: "Right. Each kind of information keeps its origin, and information missing from the loaded scope does not prove incompatibility.",
    wrong: "Not quite. Statements in the ontology are original facts, features Exact selected are matcher evidence, and prepared summaries are generated text that cites original facts. A missing, filtered or unprepared definition is not a contradiction.",
  },
  external_methods: {
    correct: { choices: ["optional", "combine_change", "per_case"] },
    right: "Right. Outside inspection is optional, you can combine and change methods, and you report what you actually used after each case.",
    wrong: "Not quite. No tool is required, and you can change methods from case to case. After each case you report the methods you actually used for that source and its candidates, not for each candidate separately.",
  },
};

export function gradeResponse(questionId: string, response: AssessmentResponse): boolean {
  const key = ASSESSMENT_KEYS[questionId]?.correct;
  if (!key) return false;
  if (key.choice !== undefined && response.choice !== key.choice) return false;
  if (key.choices && JSON.stringify([...(response.choices ?? [])].sort()) !== JSON.stringify([...key.choices].sort())) return false;
  if (key.matches && Object.entries(key.matches).some(([row, code]) => response.matches?.[row] !== code)) return false;
  if (key.part_b !== undefined && response.part_b !== key.part_b) return false;
  return true;
}

export function tutorialPublic(downloads: PublicAsset[]): TutorialPublic {
  return {
    tutorial_id: "exact-explain-tutorial",
    version: TUTORIAL_VERSION,
    hash: "draft",
    synthetic: true,
    intro: "This tutorial uses an invented practice case in the same tools as the study. Nothing here is scored, and every name, fact, score and description in it is made up. Your progress is saved, so you can pause and come back on any device with your private link.",
    lessons: LESSONS,
    assessment: ASSESSMENT,
    case: {
      source: ENTITY[SOURCE_KEY],
      source_label: source.label,
      candidates: TUTORIAL_CANDIDATES,
      explanation_refs: [TUTORIAL_RESOURCE_ID],
      ontology_resources: downloads,
    },
  };
}
