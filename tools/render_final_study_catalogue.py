"""Render a self-contained PRIVATE owner catalogue, with local notes and consent preview."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.final_study_copy import OWNER_DETAILS, participant_information

TEMPLATE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exact Explain · Final study review</title>
<style>
:root{color-scheme:light dark;--paper:#f5f2ec;--card:#fffdf9;--ink:#243038;--muted:#647078;--line:#d8d9d5;--blue:#2c5878;--gold:#755719;font:16px/1.5 system-ui,sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink)}button,select,input,textarea{font:inherit}button,select{border:1px solid var(--line);border-radius:6px;padding:.45rem .8rem;background:var(--card);color:var(--ink);cursor:pointer}button:hover,button[aria-pressed=true]{border-color:var(--blue);background:#e7eff4}button:focus-visible,input:focus-visible,textarea:focus-visible,select:focus-visible{outline:3px solid #6690b4;outline-offset:2px}header{padding:1.2rem 2rem;border-bottom:1px solid var(--line);display:flex;gap:1rem;align-items:center;flex-wrap:wrap}header h1{font:600 1.5rem Georgia,serif;margin:0;flex:1}.private{font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;color:var(--gold)}.tabs{display:flex;gap:.5rem}.summary{padding:1rem 2rem;background:var(--card);border-bottom:1px solid var(--line)}.summary p{margin:.25rem 0}.muted,small{color:var(--muted)}.layout{display:grid;grid-template-columns:290px minmax(0,1fr)}aside{border-right:1px solid var(--line);padding:1rem;height:calc(100vh - 215px);position:sticky;top:0;overflow:auto}.filters{display:grid;gap:.7rem}input,textarea{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:.55rem;max-width:100%}aside input{width:100%}#case-list{display:grid;gap:.5rem;margin-top:1rem}.case-link{text-align:left;font-size:.9rem}.case-link span{display:block}.rank{font-size:.78rem;color:var(--muted)}main{min-width:0;padding:1.5rem 2rem;max-width:1300px}h2,h3{line-height:1.25}h2{font:600 1.9rem Georgia,serif;margin:.2rem 0 .6rem}h3{font-size:1.15rem}.iri{font:.77rem/1.6 ui-monospace,monospace;overflow-wrap:anywhere;color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1.15rem;margin:1rem 0}.card.reference{border-left:4px solid var(--gold)}.candidate-head{display:flex;align-items:baseline;gap:.8rem;flex-wrap:wrap}.candidate-head h3{flex:1;margin:0}.pill{border:1px solid var(--line);border-radius:20px;padding:.1rem .5rem;font-size:.76rem;white-space:nowrap}.definition{margin:.8rem 0;max-width:80ch}.facts{display:flex;gap:.5rem;flex-wrap:wrap;font-size:.8rem}.facts span{padding:.1rem .4rem;border-radius:4px;background:var(--paper)}details{margin:.7rem 0}summary{cursor:pointer;font-weight:500}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:430px;overflow:auto;font:12px/1.5 ui-monospace,monospace;background:var(--paper);padding:1rem}textarea{width:100%;min-height:90px}.notes{display:grid;gap:.5rem}.nav{display:flex;justify-content:space-between;gap:1rem;margin:1rem 0}.warning{padding:.8rem 1rem;border-left:3px solid var(--gold);background:var(--card)}#welcome{max-width:900px;margin:auto;padding:2rem}.prose p{margin:1rem 0}.prose h2{font-size:1.6rem}#welcome[hidden],#cases[hidden]{display:none}.empty{padding:1rem}.status{min-height:1.5rem}.toolbar{display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
@media(max-width:800px){header{padding:1rem}.summary{padding:1rem}.layout{grid-template-columns:1fr}aside{height:auto;position:static;border-right:0;border-bottom:1px solid var(--line)}#case-list{max-height:220px;overflow:auto}main,#welcome{padding:1rem}}
@media(prefers-color-scheme:dark){:root{--paper:#202629;--card:#293034;--ink:#e3e7e7;--muted:#b0bdc4;--line:#465258;--blue:#9ab9cc;--gold:#dbc396}button:hover,button[aria-pressed=true]{background:#354650}}
</style>
<header><div><div class="private">Private owner review · not participant material</div><h1>Final NCIT–DOID study</h1></div><nav class="tabs" aria-label="Review sections"><button id="tab-cases" aria-pressed="true">Sources and candidates</button><button id="tab-welcome" aria-pressed="false">Welcome and consent</button></nav><button id="export">Export my review notes</button></header>
<section class="summary"><p id="pool-summary"></p><p class="muted">Original Exact 2.0 ranks and scores. Reference answers are shown here for your review only.</p><p id="save-status" class="status muted" role="status"></p></section>
<section id="cases"><div class="layout"><aside><div class="filters"><label>Pool option <select id="pool"></select></label><label>Reference rank <select id="rank"><option value="all">All ranks</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label><label>Find a concept <input id="search" type="search" placeholder="Name or IRI"></label><label><input type="checkbox" id="show-reference" checked> Highlight reference answers</label></div><nav id="case-list" aria-label="Study sources"></nav></aside><main id="detail"></main></div></section>
<section id="welcome" hidden><div class="warning" id="draft-status"></div><h2>Welcome, and thank you for considering this study</h2><p>You will compare concepts from two ontologies and decide which ones have the same meaning.</p><section class="card prose"><h3>Information and consent</h3><div id="information"></div><p id="consent"></p></section><section class="card prose"><h3>Setup · 1 · The task</h3><div id="task"></div><div id="setup"></div></section></section>
<script type="application/json" id="review-data">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent), byId=new Map(data.cases.map(c=>[c.case_id,c]));
const $=id=>document.getElementById(id), key='exact.final-study.review.'+data.catalogue_sha256;let notes={},current=null;
try{const saved=localStorage.getItem(key);if(saved!==null){notes=JSON.parse(saved)}else if(data.review_predecessor){const old=JSON.parse(localStorage.getItem('exact.final-study.review.'+data.review_predecessor.catalogue_sha256)||'{}');for(const id of data.review_predecessor.unchanged_case_ids)if(old[id])notes[id]=old[id];if(Object.keys(notes).length)localStorage.setItem(key,JSON.stringify(notes))}}catch{}
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function paragraphs(id,text){$(id).replaceChildren(...text.split(/\n\s*\n/).map(t=>node('p',t)))}
function persist(){try{localStorage.setItem(key,JSON.stringify(notes));$('save-status').textContent='Review notes saved in this browser. Export them to keep or share a copy.'}catch{$('save-status').textContent='Browser storage is unavailable. Export your notes before closing this page.'}}
function visible(){const q=$('search').value.trim().toLowerCase(),rank=$('rank').value;return data.options[$('pool').value].map(id=>byId.get(id)).filter(c=>(rank==='all'||c.reference_rank===Number(rank))&&(!q||JSON.stringify([c.source,c.source_summary.label,...c.candidates.map(t=>[t.target,t.context_summary.label])]).toLowerCase().includes(q)))}
function renderList(){const rows=visible();if(!rows.some(c=>c.case_id===current))current=rows[0]?.case_id??null;$('case-list').replaceChildren(...rows.map(c=>{const b=node('button',undefined,'case-link');b.setAttribute('aria-pressed',String(c.case_id===current));b.append(node('span',c.source_summary.label),node('span','Reference rank '+c.reference_rank+' · definitions '+(c.quality?.[0]??c.challenge_screen?.entities_with_original_definitions??'?')+'/6','rank'));b.onclick=()=>{current=c.case_id;renderList();$('detail').scrollIntoView({block:'start'})};return b}));const quota=data.option_quotas[$('pool').value];$('pool-summary').textContent=data.options[$('pool').value].length+' sources · pool counts at ranks 1–5: '+[1,2,3,4,5].map(r=>quota[r]||0).join(', ')+'. '+(data.review_summary||'Each participant receives five cases per condition, one at each rank.');renderCase()}
function textValue(f){const v=f.value||{};return v.lexical_form??v.value??JSON.stringify(v)}
function expression(v){if(v==null)return '';if(Array.isArray(v))return v.map(expression).join(' ');if(typeof v!=='object')return String(v);if(v.type==='IRI')return data.labels?.[v.value]||v.value.split(/[#/]/).pop();if(v.iri)return expression(v.iri);if(v.type==='Literal')return JSON.stringify(v.lexical_form);const order={SubClassOf:['sub_class','super_class'],EquivalentClasses:['expressions'],ObjectSomeValuesFrom:['property','filler'],ObjectAllValuesFrom:['property','filler'],ObjectIntersectionOf:['operands'],ObjectUnionOf:['operands'],ObjectMinCardinality:['cardinality','property','filler'],ObjectMaxCardinality:['cardinality','property','filler'],ObjectExactCardinality:['cardinality','property','filler']};if(order[v.type])return v.type+'('+order[v.type].map(k=>expression(v[k])).join(' ')+')';return JSON.stringify(v,null,2)}
function preparedText(parent,result,title){if(!result)return;const d=node('details');d.append(node('summary',title));d.append(node('p',result.support_status==='independently_reviewed'?'Generated':'Prepared from original facts · typed rendering','muted'));for(const claim of result.claims){d.append(node('p',claim.text));const refs=node('details');refs.append(node('summary','Original supporting facts'));for(const f of result.citations||[])if(claim.fact_ids.includes(f.fact_id))refs.append(node('pre',f.value.ast?expression(f.value.ast):textValue(f)));d.append(refs)}parent.append(d)}
function factsDetails(parent,iri){const info=data.entities[iri];for(const cat of ['hierarchy','restrictions','synonyms']){const page=info.facts[cat];if(!page.items.length)continue;const d=node('details');d.append(node('summary',cat[0].toUpperCase()+cat.slice(1)+' · '+page.items.length+(page.has_more?' shown':'')));for(const f of page.items){const wrap=node('div');wrap.append(node('pre',f.value.ast?expression(f.value.ast):textValue(f)));d.append(wrap)}parent.append(d)}}
function entityCard(iri,summary,title,rank,score,reference){const card=node('section',undefined,'card'+(reference&&$('show-reference').checked?' reference':''));const head=node('div',undefined,'candidate-head');head.append(node('h3',(rank?rank+' · ':'')+title));if(score!==undefined){const s=node('span','Score '+Number(score).toPrecision(6),'pill');s.title='Exact recorded value: '+score;head.append(s)}if(reference&&$('show-reference').checked)head.append(node('span','Reference match','pill'));card.append(head,node('p',iri,'iri'));const info=data.entities[iri],defs=info.facts.definitions.items;if(defs.length)for(const f of defs)card.append(node('p',String(textValue(f)),'definition'));else card.append(node('p',info.prepared_profile?'No original definition. See the prepared description and original hierarchy below.':'No original definition in these excerpts.','definition muted'));const counts=node('div',undefined,'facts');for(const [label,n] of Object.entries(summary.counts))counts.append(node('span',n+' '+label));card.append(counts);factsDetails(card,iri);preparedText(card,info.prepared_profile,'Prepared description');return card}
function renderCase(){const main=$('detail');main.replaceChildren();if(!current){main.append(node('p','No cases match these filters.','empty'));return}const c=byId.get(current),list=visible(),index=list.findIndex(x=>x.case_id===current);const nav=node('div',undefined,'nav');for(const [label,delta] of [['Previous source',-1],['Next source',1]]){const b=node('button',label);b.disabled=index+delta<0||index+delta>=list.length;b.onclick=()=>{current=list[index+delta].case_id;renderList();main.scrollIntoView({block:'start'})};nav.append(b)}main.append(nav,node('div',c.case_id+' · original query '+c.query_id,'private'),node('h2',c.source_summary.label));if(c.selection_review){const review=node('section',undefined,'card');review.append(node('h3',c.selection_review.status));for(const text of c.selection_review.notes)review.append(node('p',text));main.append(review)}main.append(entityCard(c.source,c.source_summary,'Source concept'));if(c.other_reference_targets_outside_five.length)main.append(node('p','This source has '+c.other_reference_targets_outside_five.length+' additional reference match outside the displayed five. The displayed set contains exactly one known reference match.','warning'));const section=node('section',undefined,'card notes');section.append(node('h3','Your review'));const label=node('label','Suitability '),select=node('select');for(const [value,text]of[['unreviewed','Not reviewed'],['keep','Keep'],['consider','Needs discussion'],['exclude','Exclude']]){const o=node('option',text);o.value=value;select.append(o)}select.value=notes[current]?.decision||'unreviewed';label.append(select);const text=node('textarea');text.placeholder='Relevance, missing information, misleading candidates, related cases…';text.setAttribute('aria-label','Notes for '+c.source_summary.label);text.value=notes[current]?.text||'';const save=()=>{notes[c.case_id]={decision:select.value,text:text.value};persist()};select.onchange=save;text.oninput=save;section.append(label,text);main.append(section,node('h3','The five original candidates'));for(const t of c.candidates){const card=entityCard(t.target,t.context_summary,t.context_summary.label,t.original_rank,t.score,t.target===c.reference_target);const repeated=data.options[$('pool').value].map(id=>byId.get(id)).filter(other=>other.case_id!==c.case_id&&other.candidates.some(x=>x.target===t.target));if(repeated.length)card.append(node('p','Also appears in '+repeated.length+' other selected source cases: '+repeated.map(x=>x.source_summary.label).join('; '),'muted'));const d=node('details');d.append(node('summary',t.evidence_status==='not_extracted_for_shortlist'?'Matcher evidence · not yet extracted for this proposal':t.evidence_status==='exact_match'?'Matcher record · exact matching':t.evidence_status==='recorded_selected_evidence'?'Recorded matcher evidence':'Matcher evidence not recorded'));const record=t.matcher_record;if(record?.llm_evidence_packet)d.append(node('pre',record.llm_evidence_packet));else d.append(node('p',t.evidence_status==='not_extracted_for_shortlist'?'This shortlist contains original ontology facts. Evidence preparation has not been run for this proposal.':t.evidence_status==='exact_match'?'This candidate was retained as a protected exact match.':'No selected evidence packet is available for this pair.'));if(record){const raw=node('details');raw.append(node('summary','Structured record and original-fact references'),node('pre',JSON.stringify(record,null,2)));d.append(raw)}card.append(d);preparedText(card,t.prepared_comparison,'Prepared comparison');if(t.prepared_evidence){const admitted=node('details');admitted.append(node('summary','Admitted evidence · '+t.prepared_evidence.filter(e=>e.fact_ids.length).length+' of '+t.prepared_evidence.length+' features linked to original facts'));for(const e of t.prepared_evidence)admitted.append(node('p',e.side+' · '+e.channel+' · '+(e.display||e.origin_resolution)));card.append(admitted)}main.append(card)}const limits=node('details');limits.append(node('summary','Selection and preparation limits'));const ul=node('ul');for(const text of data.limitations)ul.append(node('li',text));limits.append(ul);main.append(limits)}
for(const [value,label] of Object.entries(data.review_options||{forty_unequal:'40 sources · unequal pool strata'})){const option=node('option',label);option.value=value;$('pool').append(option)}
for(const id of ['pool','rank','search','show-reference'])$(id).addEventListener(id==='search'?'input':'change',renderList);
for(const name of ['cases','welcome'])$('tab-'+name).onclick=()=>{for(const other of ['cases','welcome']){$(other).hidden=other!==name;$('tab-'+other).setAttribute('aria-pressed',String(other===name))}};
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'exact-final-study-owner-notes/1',catalogue_sha256:data.catalogue_sha256,pool_option:$('pool').value,notes},null,2)],{type:'application/json'});const a=node('a');a.href=URL.createObjectURL(blob);a.download='exact-final-study-review-notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
const copy=data.participant_copy;paragraphs('information',copy.information_text);$('consent').textContent=copy.consent_text;paragraphs('task',copy.instructions);paragraphs('setup',copy.setup_instructions);$('draft-status').textContent=copy.ready_for_participant_publication?'Participant wording for owner review. No final study has been published.':'Owner review draft. Still to confirm: '+copy.pending_review_items.join('; ')+'.';renderList();
</script></html>"""


def render(catalogue, details=None, reviewed_results=None):
    raw = Path(catalogue).read_bytes()
    data = json.loads(raw)
    data["catalogue_sha256"] = hashlib.sha256(raw).hexdigest()
    reviewed = json.loads(Path(reviewed_results).read_bytes()) if reviewed_results else {}
    overrides = reviewed.get("results", {})
    for info in data["entities"].values():
        original = info.get("prepared_profile")
        if original:
            info["prepared_profile"] = overrides.get(original["explanation_id"], original)
    for case in data["cases"]:
        for row in case["candidates"]:
            original = row.get("prepared_comparison")
            if original:
                row["prepared_comparison"] = overrides.get(original["explanation_id"], original)
    if reviewed:
        data["prose_review"] = {k: v for k, v in reviewed.items() if k != "results"}
        data["limitations"] = [
            "Private owner review, not a research publication. Original ranks and scores are unchanged.",
            "Forty sources with pool counts 9, 9, 9, 8 and 5. Each participant receives one source from every rank in each condition; rare ranks repeat more across participants.",
            "One known reference match among each displayed five; two rank-5 sources also have reference matches outside those five.",
            "Related sources and repeated candidates can create learning effects; review the displayed overlap notices.",
            "435 Luna claims approved and 26 rejected after independent original-fact review. Unavailable or fully rejected drafts retain typed original-fact descriptions.",
            "Original ontology terminology and classifications can themselves be outdated or inconsistent. Generated text is grounded in those records, not independently validated biomedical advice.",
            "199 of 200 pairs have original-fact-linked matcher evidence. The protected exact-match pair has no selected-evidence record. No cross-ontology graph bridges were recorded in this run.",
            "Existing partial ELK results are reused; absent inferred edges remain unknown.",
        ]
    labels = {}
    for info in data["entities"].values():
        packet = (info.get("prepared_profile") or {}).get("support_packet") or {}
        for fact in packet.get("facts", []) + packet.get("selection", {}).get(
            "referenced_labels", []
        ):
            if fact.get("category") == "labels":
                labels[fact["subject"]["iri"]] = fact["value"].get(
                    "lexical_form", fact["subject"]["iri"]
                )
    data["labels"] = labels

    # Keep only display and citation data in the portable owner page. Full source,
    # request, response, manifest and review receipts remain in the private archive.
    def presentation(result):
        if not result:
            return result
        packet = result.get("support_packet", {})
        ids = {fid for c in result["claims"] for fid in c["fact_ids"]}
        return {
            "claims": result["claims"],
            "support_status": result["support_status"],
            "citations": [
                f
                for f in packet.get("facts", [])
                + packet.get("selection", {}).get("referenced_labels", [])
                if f["fact_id"] in ids
            ],
        }

    for info in data["entities"].values():
        info["prepared_profile"] = presentation(info.get("prepared_profile"))
    for case in data["cases"]:
        for row in case["candidates"]:
            row["prepared_comparison"] = presentation(row.get("prepared_comparison"))
    data["participant_copy"] = participant_information(
        OWNER_DETAILS if details is None else details
    )
    # Raw ontology content never becomes executable markup or JavaScript source.
    payload = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return TEMPLATE.replace("__DATA__", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--consent-details", type=Path)
    parser.add_argument("--reviewed-results", type=Path)
    args = parser.parse_args()
    details = json.loads(args.consent_details.read_text()) if args.consent_details else None
    with args.output.open("x") as stream:
        stream.write(render(args.catalogue, details, args.reviewed_results))


if __name__ == "__main__":
    main()
