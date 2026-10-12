"""An additive package revision reuses immutable data without modifying source inodes."""
import json
import os
import zipfile

import pytest

from exact_inspect.artifacts import atomic_json, export_archive, publish_bundle, validate_bundle
from exact_inspect.contracts import canonical_hash, file_hash
from tools.publish_luna_review import revised_manifest
from tools.review_luna_bundle import freeze, snapshot


def test_overlay_and_portable_export_preserve_predecessor_and_shared_inode(tmp_path):
    root=tmp_path/'package';root.mkdir()
    old='sha256:'+'1'*64
    new='sha256:'+'2'*64
    atomic_json(root/'original.json',{'explanation_id':old,'claims':[]})
    (root/'immutable.dat').write_bytes(b'immutable ontology stand-in')
    os.link(root/'immutable.dat',tmp_path/'retained-original.dat')
    original=publish_bundle(root,explanations={old:'original.json'})
    predecessor=validate_bundle(original)
    before={p.name:(snapshot(p),file_hash(p)) for p in root.iterdir() if p.is_file()}
    changed={'explanation_id':new,'claims':[{'text':'Separately reviewed prose'}]}
    manifest=revised_manifest(predecessor,root,{old:changed},{'schema':'synthetic-test'})
    path=root/'package-luna.json';freeze(path,manifest.model_dump(mode='json'))
    assert validate_bundle(original)==predecessor
    assert validate_bundle(path).explanations=={new:'luna-20261012/explanations/'+new[7:]+'.json'}
    for name,identity in before.items():
        assert (snapshot(root/name),file_hash(root/name))==identity
    assert os.path.samefile(root/'immutable.dat',tmp_path/'retained-original.dat')
    export_archive(path,tmp_path/'luna.zip')
    with zipfile.ZipFile(tmp_path/'luna.zip') as archive:
        archive.extractall(tmp_path/'relocated')
    assert validate_bundle(tmp_path/'relocated/package.json').package_id==manifest.package_id
    assert (tmp_path/'relocated/immutable.dat').read_bytes()==b'immutable ontology stand-in'
    # Exact resume is free and cannot change an already-written reviewed explanation.
    assert revised_manifest(predecessor,root,{old:changed},{'schema':'synthetic-test'})==manifest
    with pytest.raises(ValueError,match='Frozen revision artifact changed'):
        revised_manifest(predecessor,root,{old:{**changed,'claims':[]}},{'schema':'synthetic-test'})


def test_retaining_predecessor_allocates_no_description_copy(tmp_path):
    root=tmp_path/'package';root.mkdir()
    old=canonical_hash('old')
    artifact={'explanation_id':old,'claims':[]}
    atomic_json(root/'original.json',artifact)
    original=publish_bundle(root,explanations={old:'original.json'})
    predecessor=validate_bundle(original)
    files={str(p) for p in root.rglob('*')}
    revised=revised_manifest(predecessor,root,{old:artifact},{'schema':'synthetic-test'})
    assert revised.explanations==predecessor.explanations
    assert files=={str(p) for p in root.rglob('*')}
