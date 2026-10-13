import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import ts from "typescript";

// Typed citation identity through the complete path (R2, R2-F1): the artifact's citation wrapper
// → the citation button's action → the inspector reference → the original-axiom resolver. It
// runs the actual GeneratedBlock and API-adapter source on native backend facts
// (fixtures/citation-identity.json: a class and an individual punning one IRI that share one
// label record, and an EquivalentClasses axiom stored under two classes). Only the workspace and
// inspector hooks are replaced; the inspector renders whatever reference reaches it.

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../..");
const requireDependency = createRequire(import.meta.url);
type Entity = { ontology_version_id: string; iri: string; kind: string };
type Ref = { factId: string; ontologies: string[]; subject?: Entity | null; candidates?: Entity[] };

let opened: Ref | null = null;
const mocks: Record<string, unknown> = {
  "@/lib/workspace/WorkspaceContext": { useSideOf: () => () => null },
  "@/components/workspace/FactInspector": { useFactInspector: () => ({ open: (ref: Ref) => (opened = ref) }) },
};
const loaded = new Map<string, { exports: Record<string, any> }>();
function load(file: string): Record<string, any> {
  const cached = loaded.get(file);
  if (cached) return cached.exports;
  const module = { exports: {} as Record<string, any> };
  loaded.set(file, module);
  const output = ts.transpileModule(readFileSync(file, "utf8"), { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText;
  const localRequire = (name: string) => {
    if (name in mocks) return mocks[name];
    if (name.startsWith("@/") || name.startsWith(".")) {
      const base = name.startsWith("@/") ? join(src, name.slice(2)) : resolve(dirname(file), name);
      for (const suffix of ["", ".ts", ".tsx", "/index.ts", "/index.tsx"]) {
        try {
          if (readFileSync(base + suffix) && !(base + suffix).endsWith("/")) return load(base + suffix);
        } catch {
          // try the next suffix
        }
      }
    }
    return requireDependency(name);
  };
  vm.runInThisContext(`(function (require, module, exports) {${output}\n})`, { filename: file })(localRequire, module, module.exports);
  return module.exports;
}

const block = load(join(src, "components/explore/GeneratedBlock.tsx"));
const adapter = load(join(src, "lib/workspace/apiSource.ts"));
const fixtures = JSON.parse(readFileSync(join(here, "fixtures/citation-identity.json"), "utf8"));

interface Case {
  ontology_version_id: string;
  subjects: Entity[];
  fact_id: string;
  facts: { fact_id: string; subject: Entity }[];
  axiom: { ontology_version_id: string; ast: unknown };
}

/** Wrap, click the citation, resolve: what the inspector receives and shows. */
function inspect(item: Case, packetFacts: { fact_id: string; subject: Entity }[], options: { context?: Entity | null; blockSubject?: Entity | null; referenced?: { fact_id: string; subject: Entity }[]; ontologies?: string[] }) {
  opened = null;
  const artifact = { entities: item.subjects, claims: [], support_packet: { facts: packetFacts, selection: { referenced_labels: options.referenced ?? [] } } };
  const known = options.context ? { label: "Target label", side: "target", subject: options.context } : undefined;
  const cite = block.withReferencedLabels(() => known, artifact);
  const tree = block.Citations({ claim: { fact_ids: [item.fact_id] }, ontologies: options.ontologies ?? [item.ontology_version_id], subject: options.blockSubject ?? null, cite });
  const buttons = [tree.props.children].flat();
  buttons[0].props.onClick();
  assert.ok(opened, "the citation opened the inspector");
  const resolved = adapter.resolvedFromAxiom(opened, item.axiom);
  return { ref: opened as Ref, resolved, about: resolved.ambiguous ? null : resolved.subjects[0]?.entity ?? null, all: resolved.subjects.map((subject: { entity: Entity }) => subject.entity) };
}

const key = (entity: Entity) => `${entity.ontology_version_id}|${entity.kind}|${entity.iri}`;
const sorted = (list: Entity[]) => list.map(key).sort();
const orders = (facts: Case["facts"]) => [facts, [...facts].reverse()];

const punned: Case = fixtures.punned_label;
const shared: Case = fixtures.shared_axiom;
const [asClass, asIndividual] = punned.subjects;
const [classA, classB] = shared.subjects;

test("native fixtures: one original record under two typed subjects", () => {
  assert.equal(punned.facts[0].fact_id, punned.facts[1].fact_id);
  assert.deepEqual(punned.facts.map((fact) => fact.subject.kind), ["class", "individual"]);
  assert.equal((punned.axiom.ast as { type: string }).type, "AnnotationAssertion");
  assert.equal(shared.facts[0].fact_id, shared.facts[1].fact_id);
  assert.equal((shared.axiom.ast as { type: string }).type, "EquivalentClasses");
});

test("a punned record keeps the caller's admitted kind in either packet order (R2-F1)", () => {
  for (const facts of orders(punned.facts))
    for (const caller of [asIndividual, asClass]) {
      const viaContext = inspect(punned, facts, { context: caller });
      assert.deepEqual(viaContext.ref.subject, caller, `context ${caller.kind}, packet ${facts.map((fact) => fact.subject.kind)}`);
      assert.deepEqual(viaContext.about, caller);
      assert.deepEqual(sorted(viaContext.all), sorted([asClass, asIndividual]), "the other typed subject is listed, not dropped");
      const viaBlock = inspect(punned, facts, { blockSubject: caller });
      assert.deepEqual(viaBlock.about, caller);
    }
});

test("without an admitted caller several typed subjects stay explicitly ambiguous (R2-F1)", () => {
  const focal = { ...asClass, iri: "urn:focal" };
  const foreign = { ...asIndividual, ontology_version_id: "sha256:foreign" };
  const wrongKind = { ...asClass, kind: "object_property" };
  for (const facts of orders(punned.facts))
    for (const caller of [null, focal, foreign, wrongKind]) {
      const result = inspect(punned, facts, { context: caller, blockSubject: caller });
      assert.equal(result.ref.subject, null, `caller ${caller ? key(caller) : "none"} is not admitted`);
      assert.equal(result.resolved.ambiguous, true);
      assert.deepEqual(sorted(result.all), sorted([asClass, asIndividual]), "both admitted, never the caller, never by order");
    }
});

test("a shared axiom stored under two classes: the matching caller, otherwise both (R2-F1)", () => {
  for (const facts of orders(shared.facts)) {
    for (const caller of [classA, classB]) {
      const result = inspect(shared, facts, { context: caller });
      assert.deepEqual(result.about, caller);
      assert.deepEqual(sorted(result.all), sorted([classA, classB]));
    }
    const open = inspect(shared, facts, { blockSubject: { ...classA, iri: "urn:focal" } });
    assert.equal(open.resolved.ambiguous, true);
    assert.deepEqual(sorted(open.all), sorted([classA, classB]));
  }
});

test("one admitted subject still overrides a wrong focal caller; the axiom still validates (R2)", () => {
  // Only the individual's copy is admitted: a class caller with the same IRI is not that subject.
  const single = inspect(punned, [punned.facts[1]], { blockSubject: asClass });
  assert.deepEqual(single.about, asIndividual);
  // A referenced entity's label cited from another profile (the 33 Luna profiles' case).
  const referenced = { fact_id: punned.fact_id, subject: asIndividual };
  const label = inspect(punned, [], { blockSubject: { ...asClass, iri: "urn:focal" }, referenced: [referenced] });
  assert.deepEqual(label.about, asIndividual);
  // Older artifacts without support subjects: a wrong caller is replaced by the axiom's subject IRI.
  const legacy = inspect(punned, [], { blockSubject: { ...asClass, iri: "urn:focal" } });
  assert.equal(legacy.about?.iri, "urn:punned");
  // A caller from another ontology is never attached to the record.
  const foreign = inspect(shared, [], { blockSubject: { ...classA, ontology_version_id: "sha256:foreign" }, ontologies: [shared.ontology_version_id] });
  assert.deepEqual(foreign.all, []);
});
