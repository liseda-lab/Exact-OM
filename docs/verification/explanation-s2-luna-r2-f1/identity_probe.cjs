// R2-F1 probe: the review's native punned-label fixture through the actual citation wrapper,
// citation click and axiom resolver, with the individual as caller context, in both packet orders.
// Only the workspace and inspector hooks are stubbed. Adapted from the review's identity_probe.cjs
// so that the fixture, frontend source tree and dependencies are arguments.
//
// Usage: node identity_probe.cjs <identity-fixture.json> <explanations_visualizer/src> <dir with node_modules/typescript>
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const {createRequire} = require('node:module');
const [fixturePath, src, depsRoot] = process.argv.slice(2);
const deps = createRequire(path.join(path.resolve(depsRoot), 'package.json'));
const ts = deps('typescript');
let opened;
const cache = new Map();
const mocks = {
  '@/lib/workspace/WorkspaceContext': {useSideOf: () => () => null},
  '@/components/workspace/FactInspector': {useFactInspector: () => ({open: ref => opened = ref})},
};
function load(file) {
  if (cache.has(file)) return cache.get(file).exports;
  const module = {exports: {}};
  cache.set(file, module);
  const js = ts.transpileModule(fs.readFileSync(file, 'utf8'), {compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX,
    target: ts.ScriptTarget.ES2022, esModuleInterop: true,
  }}).outputText;
  const req = name => {
    if (mocks[name]) return mocks[name];
    if (name.startsWith('@/') || name.startsWith('.')) {
      const base = name.startsWith('@/') ? path.join(src, name.slice(2)) : path.resolve(path.dirname(file), name);
      const found = ['', '.ts', '.tsx', '/index.ts', '/index.tsx'].map(suffix => base + suffix)
        .find(p => fs.existsSync(p) && fs.statSync(p).isFile());
      if (found) return load(found);
    }
    return deps(name);
  };
  vm.runInThisContext('(function(require,module,exports){' + js + '\n})', {filename: file})(req, module, module.exports);
  return module.exports;
}
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
const gp = load(path.join(src, 'components/explore/GeneratedBlock.tsx'));
const adapter = load(path.join(src, 'lib/workspace/apiSource.ts'));
const factId = fixture.packet.facts[0].fact_id;
const known = {label: 'Target label', side: 'target', subject: fixture.expected_context};
function run(facts) {
  opened = null;
  const artifact = {entities: fixture.packet.entities, support_packet: {...fixture.packet, facts}};
  const cite = gp.withReferencedLabels(() => known, artifact);
  const tree = gp.Citations({claim: {fact_ids: [factId]},
    ontologies: [fixture.expected_context.ontology_version_id], subject: fixture.expected_context, cite});
  [tree.props.children].flat()[0].props.onClick();
  const resolved = adapter.resolvedFromAxiom(opened, fixture.axiom);
  return {packet_order: facts.map(f => f.subject.kind),
    wrapped_subject: cite(factId).subject ?? null,
    inspector_subject: resolved.ambiguous ? null : resolved.subjects[0]?.entity ?? null,
    inspector_subjects: resolved.subjects.map(s => s.entity.kind),
    ambiguous: Boolean(resolved.ambiguous)};
}
const result = {
  fixture: 'Native pyowl_core ontology with a class and an individual using the same IRI; valid FactPacket',
  fact_id: factId,
  caller_subject: known.subject,
  runs: [run(fixture.packet.facts), run([...fixture.packet.facts].reverse())],
};
const kinds = result.runs.map(r => r.inspector_subject ? r.inspector_subject.kind : 'ambiguous');
result.order_dependent = kinds[0] !== kinds[1];
result.caller_kept = result.runs.every(r => r.inspector_subject && r.inspector_subject.kind === known.subject.kind);
console.log(JSON.stringify(result, null, 2));
