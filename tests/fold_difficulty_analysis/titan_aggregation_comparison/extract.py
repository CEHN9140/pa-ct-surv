"""Same CONCH bags -> raw MeanPool and official frozen TITAN slide embeddings."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
# Keep downloaded runtime code/tokenizer caches inside this experiment, offline.
os.environ['HF_HOME']=str(HERE/'private/hf')
os.environ['HF_HUB_CACHE']=str(HERE/'private/hf/hub')
os.environ['HF_MODULES_CACHE']=str(HERE/'private/hf/modules')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(HERE/'private/deps'))

import h5py
import numpy as np
import pandas as pd
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,obj):
    p=Path(path);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)


def read_bag(path):
    st=Path(path).stat()
    with h5py.File(path,'r') as f:
        x=np.asarray(f['features'][:]);coords=f['coords'][:];attrs=dict(f['coords'].attrs)
    if x.ndim!=2 or x.shape[1]!=768 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid/nonfinite CONCH features')
    if coords.shape!=(len(x),2) or not np.issubdtype(coords.dtype,np.integer) or (coords<0).any():raise ValueError('Invalid level0 coordinates')
    step=float(attrs.get('patch_size_level0',np.nan))
    if not np.isfinite(step) or step<=0 or abs(step-round(step))>1e-6:raise ValueError('Noninteger/invalid patch_size_level0')
    if int(attrs.get('patch_size',0))!=512 or abs(float(attrs.get('target_magnification',0))-20)>1e-4:
        raise ValueError('TITAN requires the declared 20x512 patch recipe')
    step=int(round(step));grid=(coords-coords.min(axis=0))//step
    if len(np.unique(grid,axis=0))!=len(coords):raise ValueError('Duplicate grid cells would merge patch tokens')
    now=Path(path).stat()
    if (now.st_size,now.st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise ValueError('CONCH input changed during read')
    mean=x.mean(axis=0,dtype=np.float64)
    meta=dict(patches=len(x),size=st.st_size,mtime_ns=st.st_mtime_ns,patch_size_lv0=step,
              input_dtype=str(x.dtype),feature_values_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
              coords_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),zero_vectors=int((x==0).all(axis=1).sum()))
    # Preserve numerical feature values; TITAN's official grid accumulator uses float32.
    return x.astype(np.float32),coords.astype(np.int64),step,mean,meta


def encode(model,x,coords,step,device):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval model required')
    context=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    with torch.inference_mode(),context:
        z=model.encode_slide_from_patch_features(torch.from_numpy(x).unsqueeze(0).to(device),
                    torch.from_numpy(coords).unsqueeze(0).to(device),step)
    if tuple(z.shape) not in [(768,),(1,768)] or not torch.isfinite(z).all():raise ValueError('Invalid TITAN slide embedding')
    return z.float().cpu().numpy().reshape(768)


def stage_python_modules(weights,module_cache):
    # Transformers local-dir loader copies direct imports only; stage all official
    # flat repository modules so recursive imports can be hashed/loaded offline.
    target=Path(module_cache)/'transformers_modules'/Path(weights).name
    target.mkdir(parents=True,exist_ok=True)
    for p in Path(weights).glob('*.py'):shutil.copyfile(p,target/p.name)
    return target


def load_model(config,device):
    weights=Path(config['weights'])
    for filename in ['config.json','model.safetensors','modeling_titan.py','vision_transformer.py']:
        if not (weights/filename).exists():raise FileNotFoundError(f'Missing local TITAN file: {filename}')
    cache=Path(os.environ['HF_HUB_CACHE'])/'models--MahmoodLab--TITAN'
    snapshot=cache/'snapshots'/config['revision'];snapshot.mkdir(parents=True,exist_ok=True)
    (cache/'refs').mkdir(exist_ok=True);(cache/'refs/main').write_text(config['revision'])
    for filename in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
        target=snapshot/filename
        if not target.exists():target.symlink_to((weights/filename).resolve())
    stage_python_modules(weights,os.environ['HF_MODULES_CACHE'])
    from transformers import AutoModel
    model,info=AutoModel.from_pretrained(weights,trust_remote_code=True,local_files_only=True,
                                        use_safetensors=True,output_loading_info=True)
    if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'TITAN checkpoint loading differs: {info}')
    model.requires_grad_(False);model.eval().to(device)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    return model


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--smoke',action='store_true');group.add_argument('--mean-only',action='store_true')
    args=parser.parse_args();config=json.loads((HERE/'config.json').read_text())
    private=HERE/'private';private.mkdir(exist_ok=True);out=HERE/'results';out.mkdir(exist_ok=True)
    c=pd.read_csv(ROOT/'data/trident/features_uni_v1/all_label_roi64.csv',dtype={'pa_id':str})
    if len(c)!=721 or c.event.sum()!=251 or not c.pa_id.is_unique:raise ValueError('Original cohort required')
    source=Path(config['source']);feature_root=source/'features_conch_v15'
    for case in c.pa_id:
        if not (feature_root/f'{case}.h5').exists():raise FileNotFoundError('Incomplete CONCH coverage; no patient dropping')
    model=None
    if not args.mean_only:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for real TITAN inference')
        model=load_model(config,'cuda:0')
    means=[];slides=[];records={};repeat_error=None
    for i,case in enumerate(c.pa_id):
        x,coords,step,mean,record=read_bag(feature_root/f'{case}.h5')
        with h5py.File(source/'patches'/f'{case}_patches.h5','r') as f:
            if not np.array_equal(coords,f['coords'][:]):raise ValueError('CONCH feature/coordinate source differs')
        means.append(mean);records[case]=record
        if model is not None:
            z=encode(model,x,coords,step,'cuda:0');slides.append(z)
            if i==0:
                other=encode(model,x,coords,step,'cuda:0');repeat_error=float(np.max(abs(z-other)))
                np.testing.assert_allclose(z,other,rtol=0,atol=2e-4)
        if args.smoke:
            dump(out/'smoke_validation.json',dict(real_checkpoint=True,feature_dim=768,slide_dim=len(z),
                 repeat_max_abs_error=repeat_error,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 checkpoint_sha256=sha(Path(config['weights'])/'model.safetensors'),revision=config['revision']))
            print('Real TITAN smoke passed; repeated embedding error',repeat_error,flush=True)
            return
        if (i+1)%50==0 or i==720:
            dump(out/'extraction_status.json',dict(stage='extracting',completed_cases=i+1,titan=model is not None))
            print(f'Same-bag mean'+('/TITAN' if model is not None else '')+f' embeddings {i+1}/721',flush=True)
    ids=c.pa_id.to_numpy(dtype=str)
    np.savez_compressed(private/'conch_mean.npz',features=np.stack(means),case_ids=ids)
    if slides:np.savez_compressed(private/'titan.npz',features=np.stack(slides),case_ids=ids)
    for case,record in records.items():
        st=(feature_root/f'{case}.h5').stat()
        if (st.st_size,st.st_mtime_ns)!=(record['size'],record['mtime_ns']):raise RuntimeError('Source changed during extraction')
    dump(private/'input_provenance.json',records)
    files=list(Path(config['weights']).glob('*.py'))+[Path(config['weights'])/'config.json',Path(config['weights'])/'model.safetensors'] if slides else []
    dump(out/'extraction_validation.json',dict(stage='completed',patients=721,events=251,patches=sum(r['patches'] for r in records.values()),
         mean_matrix_sha256=sha(private/'conch_mean.npz'),titan_matrix_sha256=sha(private/'titan.npz') if slides else None,
         revision=config['revision'],frozen=model is not None,repeat_max_abs_error=repeat_error,
         weights_and_code_sha256={p.name:sha(p) for p in files},feature_bundle_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()))
    dump(out/'extraction_status.json',dict(stage='completed',completed_cases=721,titan=bool(slides)))


if __name__=='__main__':main()
