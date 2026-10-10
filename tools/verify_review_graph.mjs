// Exercise the actual frontend projection function with admitted real originals.
// Feature projection is checked here; S0 channel-derived bridges are NOT accepted
// as recorded comparisons, and this is not a rendered-browser test.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

const modulePath = process.argv[2];
const { buildGraphModel } = await import(pathToFileURL(modulePath).href);
const input = JSON.parse(readFileSync(0, 'utf8'));
const counts = values => Object.fromEntries([...new Set(values)].sort().map(v => [v, values.filter(x => x === v).length]));
const pairs = input.pairs.map(pair => {
  const model = buildGraphModel(pair.source, pair.target, pair.items, pair.axioms, { expandLiterals: new Set(['source', 'target']) });
  const channels = {};
  for (const record of model.records) {
    if (record.status !== 'drawn') continue;
    const edge = model.edges.find(e => e.id === record.edgeId && e.kind === 'feature');
    if (!edge || edge.factId !== record.factId || edge.evidenceId !== record.evidenceId) throw new Error('Projection lost recorded identity');
    (channels[record.channel] ??= new Set()).add(record.side);
  }
  return { case_id: pair.case_id, candidate_id: pair.candidate_id,
    selected_features: pair.items.length, original_records: model.records.length,
    projection_records: counts(model.records.map(r => r.status)),
    projected_channel_sides: Object.fromEntries(Object.entries(channels).map(([k, v]) => [k, [...v].sort()])),
    source_only_channels: Object.keys(channels).filter(k => channels[k].size === 1 && channels[k].has('source')),
    both_sides_channels: Object.keys(channels).filter(k => channels[k].size === 2),
    max_literal_length: Math.max(0, ...model.nodes.filter(n => n.role === 'literal').map(n => n.text.length)),
    expression_edges: model.edges.filter(e => e.kind === 'feature' && e.relation.quantifier).map(e => ({ fact_id: e.factId, evidence_id: e.evidenceId, quantifier: e.relation.quantifier })) };
});
console.log(JSON.stringify({ schema: 'exact-review-frontend-projection-check/1',
  module_sha256: createHash('sha256').update(readFileSync(modulePath)).digest('hex'),
  publication_sha256: input.publication_sha256, pairs,
  source_only_pairs: pairs.filter(p => p.source_only_channels.length).length,
  multiple_both_sides_channel_pairs: pairs.filter(p => p.both_sides_channels.length > 1).length,
  limitation: 'Pure S0 feature projection only; automatic channel bridges excluded from acceptance. No DOM, visual layout or S2 verification.' }, null, 2));
