from pathlib import Path
import importlib.util


def test_publication_whitelist_contains_only_aggregate_scope(tmp_path):
    path=Path(__file__).with_name('publish_completed.py')
    spec=importlib.util.spec_from_file_location('cluster_publisher',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names=module.publication_paths(tmp_path)
    assert names and len(names)==len(set(names))
    assert all(p.is_relative_to(tmp_path) for p in names)
    assert not any('private' in p.parts or p.suffix in ['.npy','.npz','.joblib','.pth'] for p in names)
    assert not any('predictions' in p.name or 'pipeline.log'==p.name for p in names)
    assert tmp_path/'results/report.md' in names
    assert tmp_path/'results/paired_comparisons.csv' in names
    assert not any('wsi_maps' in p.parts or 'prototype_images' in p.parts for p in names)


def test_completed_publication_commits_only_whitelist(tmp_path,monkeypatch):
    import json
    path=Path(__file__).with_name('publish_completed.py');spec=importlib.util.spec_from_file_location('cluster_publisher_mock',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'repo';base=root/'tests/cluster';base.mkdir(parents=True)
    monkeypatch.setattr(module,'ROOT',root);monkeypatch.setattr(module,'BASE',base)
    for p in module.publication_paths(base):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fold,value\n0,1\n' if p.suffix=='.csv' else 'test')
    (base/'validation/publish_authorization.json').write_text(json.dumps(dict(user_requested_push_after_completion=True)))
    (base/'validation/source_sha256.json').write_text('{}')
    (base/'status.json').write_text(json.dumps(dict(status='completed')))
    (base/'results/validation.json').write_text(json.dumps(dict(folds=5,arms=3,passed=True,frozen_parameters_exact=True,branch_pairing_verified=True,selection_verified=True,final_cindices_recomputed=True)))
    (base/'private').mkdir();(base/'private/patients.csv').write_text('private')
    staged=[];calls=[];committed=[False]
    def fake_git(*args):
        calls.append(args)
        if args==('diff','--cached','--name-only'):return '\n'.join(staged)
        if args==('branch','--show-current'):return 'main'
        if args==('config','--get','remote.origin.url'):return 'https://github.com/CEHN9140/pa-ct-surv.git'
        if args[0]=='rev-parse':return 'new' if args[1]=='HEAD' and committed[0] else 'old'
        if args[0]=='add':staged.extend(args[3:])
        if args[0]=='commit':committed[0]=True
        if args[0]=='diff-tree':return '\n'.join(staged)
        return ''
    monkeypatch.setattr(module,'git',fake_git);monkeypatch.setattr(module,'network',fake_git)
    assert module.publish()=='new'
    assert set(staged)=={str(p.relative_to(root)) for p in module.publication_paths(base)}
    assert not any('/private/' in p for p in staged)
    assert ('push','origin','new:refs/heads/main') in calls
