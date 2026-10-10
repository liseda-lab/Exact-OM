"""Complete a fresh HTTPS workflow with separately preserved exhaustive inventory evidence.

No provider calls, no database reset, no acceptance assertion replacement. The full
ordered inventory check already passed on the exact frozen publication in the
preserved attempt; only its subsequent empty-class probe failed. This run repeats
all workflow and corrected bounded navigation checks and reports the scopes apart.
"""
import json
import runpy
import sys
import time
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

REPO=Path('/Users/pedrocotovio/.codex/worktrees/study-timing-integration/Exact-OM')
ROOT=Path('/Users/pedrocotovio/Library/Caches/exact-om-review/s1-delivered-final-20261010')
SOURCE=Path('/Users/pedrocotovio/Library/Caches/exact-om-review/s1-interim-20261010')
sys.path.insert(0,str(REPO))
from tools import review_environment as env
from tools.verify_review_environment import Journey, fingerprint, require
helpers=runpy.run_path('/tmp/exact-delivered-verification.py')
PRIOR_VERIFIER='d5e923b82b68dfc0feec9a03519854ecba6ce7cc43096e4fedb2ada24d4c11f9'

class ResumedJourney(Journey):
    def complete_read_surfaces(self,prefix,capabilities):
        prior_path=ROOT/'verification/api-journey-attempt-1.json'
        prior=json.loads(prior_path.read_text())
        require(prior['status']=='failed' and prior['failure']=='nonempty_class_hierarchy_exercised','specific_prior_probe_failure')
        require(prior['verifier_sha256']==PRIOR_VERIFIER,'known_prior_inventory_control_flow')
        require(prior['publication_sha256']==env.file_hash(self.config['publication']),'same_immutable_publication')
        require(prior['check_counts']['study_full_index_page']==2285 and prior['check_counts']['main_full_index_page']==2285,'prior_complete_inventory_counts')
        require(prior['check_counts']['study_exact_iri_search']==2 and prior['check_counts']['main_exact_iri_search']==2,'both_prior_ordered_inventory_assertions_passed_before_search')
        record=env.processes(self.config).get('exploration')
        require(record and env.owned(self.config,'exploration',record),'owned_exploration_process')
        main=f"https://127.0.0.1:{self.config['ports']['exploration']}/api/v1"
        candidates=[e for case in self.definition['cases'] for e in [case['source'],*[c['entity'] for c in case['candidates']]]]
        summaries=[]
        for version in capabilities['ontology_versions']:
            # Real exact-IRI searches are repeated on both services; exhaustive walk
            # evidence is referenced, never synthesized as if repeated by this run.
            selected=next(e for e in candidates if e['ontology_version_id']==version)
            query=dict(ontology_version_id=version,term=selected['iri'],kind=selected['kind'],limit=100)
            for endpoint,label in ((prefix+'/entities','study_exact_iri_search'),(main+'/entities','main_exact_iri_search')):
                page=self.request('GET',endpoint,params=query,label=label).json()
                require(page['returned_count']==len(page['items']),'exact_iri_page_count')
                require(selected in [r['entity'] for r in page['items']],'exact_iri_search_identity')
            probe,counts,attempted=self.hierarchy_probe(prefix,main,version,candidates)
            graph=self.request('GET',main+'/hierarchy',params={**probe,'direction':'ancestors','basis':'literal_asserted','limit':100},label='main_bounded_ancestor_graph').json()
            require(len(graph['nodes'])<=100 and probe in graph['nodes'],'ancestor_graph_node_bound')
            summaries.append(dict(ontology_sha256=fingerprint(version),hierarchy=counts,hierarchy_probes_attempted=attempted,hierarchy_empty_probes=attempted-1,ancestor_graph_nodes=len(graph['nodes']),ancestor_graph_edges=len(graph['items'])))
        require(env.owned(self.config,'exploration',record),'exploration_identity_preserved')
        self.report['complete_read_surfaces']=dict(scope='This run repeats real search, nonempty hierarchy/continuation parity and bounded ancestor graph. Full ordered inventories passed in the separately preserved prior attempt; they were not repeated.',prior_inventory_receipt_sha256=env.file_hash(prior_path),prior_inventory_verifier_sha256=PRIOR_VERIFIER,prior_inventory_elapsed_seconds=prior['elapsed_seconds'],prior_inventory_pages=dict(study=2285,exploration=2285),same_publication_sha256=prior['publication_sha256'],ontologies=summaries)
        self.report['resume_orchestration_sha256']=env.file_hash(__file__)


def main():
    config=env.load(ROOT)
    require(hasattr(Journey,'hierarchy_probe'),'corrected_committed_verifier_required')
    current=ROOT/'verification/relocation.json'
    previous=ROOT/'verification/relocation-attempt-1.json'
    api=ROOT/'verification/api-journey.json'
    prior_api=ROOT/'verification/api-journey-attempt-1.json'
    require(not previous.exists() and not prior_api.exists(),'preserved_attempt_names_must_be_new')
    report=json.loads(current.read_text());old=json.loads(api.read_text())
    require(report['status']=='failed' and report['source_restored'],'previous_cleanup_completed')
    require(old.get('failure')=='nonempty_class_hierarchy_exercised','specific_failed_probe_only')
    env.verify_backend(config)
    require(all(env.port_available(p) for p in config['ports'].values()),'owned_ports_are_clear')
    previous.write_bytes(current.read_bytes());prior_api.write_bytes(api.read_bytes());api.unlink()
    report.update(status='running',source_hidden=False,source_restored=False,previous_relocation_sha256=env.file_hash(previous),previous_api_attempt=dict(receipt_sha256=env.file_hash(prior_api),status='failed',failure=old['failure'],elapsed_seconds=old['elapsed_seconds'],scope='Full ordered inventories passed; later first-focal-class hierarchy probe was empty. All original bytes preserved.'),resume_orchestration_sha256=env.file_hash(__file__))
    report.pop('failed_phase',None);report.pop('exception_type',None)
    (ROOT/'verification/resume_workflow.py').write_bytes(Path(__file__).read_bytes())
    hidden=SOURCE.with_name(SOURCE.name+'.verification-hidden-'+uuid4().hex[:8])
    phase='resume_start'
    with env.environment_lock(ROOT):
        try:
            require(SOURCE.exists() and not hidden.exists(),'source_restore_precondition')
            SOURCE.rename(hidden);report['source_hidden']=True
            env.private_json(current,report)
            print(json.dumps(dict(phase=phase)),flush=True)
            start=time.monotonic()
            with helpers['Samples'](env,config) as sample:
                result=env.start(config)
                start_seconds=round(time.monotonic()-start,3)
                frontends=helpers['probe_frontend'](config)
            report['phases'][phase]=dict(start_seconds=start_seconds,seconds=round(time.monotonic()-start,3),frontend_probes=frontends,processes=sample.values,services=result['services'])
            env.private_json(current,report)
            phase='api_journey';print(json.dumps(dict(phase=phase,full_inventory='separately_preserved_prior_pass')),flush=True)
            start=time.monotonic()
            with helpers['Samples'](env,config) as sample:
                ResumedJourney(config,receipt=api).run()
            report['phases'][phase]=dict(seconds=round(time.monotonic()-start,3),processes=sample.values)
            report['api_receipt_sha256']=env.file_hash(api)
            phase='frozen_copy_unchanged';start=time.monotonic()
            for item in report['clone']['entries']:
                require(env.file_hash(ROOT/item['path'])==item['sha256'],'frozen_input_unchanged')
            report['phases'][phase]=dict(seconds=round(time.monotonic()-start,3),files=len(report['clone']['entries']),unchanged=True)
            report['status']='passed'
        except BaseException as error:
            report.update(status='failed',failed_phase=phase,exception_type=type(error).__name__)
            for role in reversed(env.ROLES):
                try:env.stop_role(config,role)
                except Exception:report.setdefault('stop_failures',[]).append(role)
            raise
        finally:
            if hidden.exists() and not SOURCE.exists():hidden.rename(SOURCE)
            report['source_restored']=SOURCE.exists() and not hidden.exists()
            if not report['source_restored']:report.update(status='failed',failed_phase='source_restoration')
            env.private_json(current,report)
            print(json.dumps(dict(status=report['status'],phase=phase,source_restored=report['source_restored'])),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(json.dumps(dict(status='failed',exception_type=type(error).__name__)),flush=True)
        raise SystemExit(1)
