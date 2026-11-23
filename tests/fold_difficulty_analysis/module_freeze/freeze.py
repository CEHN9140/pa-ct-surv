"""Parameter-only freezing with all train-mode dropout and full gradient paths retained."""
import hashlib,random
import numpy as np
import torch


def modules(model):
    return dict(A=[model.mil.compression],B=[model.mil.attention],C=[model.mil.post_compression,model.mil.risk_head])


def configure(model,arm):
    frozen={'baseline':[],'freeze_ab':['A','B'],'freeze_c':['C']}[arm]
    for p in model.parameters():p.requires_grad_(True)
    for name in frozen:
        for layer in modules(model)[name]:
            for p in layer.parameters():p.requires_grad_(False)


def weight_hash(model,scope='ABC'):
    selected=[layer for key,values in modules(model).items() if key in scope for layer in values]
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for layer in selected for p in layer.parameters())).hexdigest()


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def rng_hash(state):
    return hashlib.sha256(repr(state['python']).encode()+repr((state['numpy'][0],*state['numpy'][2:])).encode()+state['numpy'][1].tobytes()+state['cpu'].numpy().tobytes()+b''.join(x.cpu().numpy().tobytes() for x in state['cuda'])).hexdigest()


def choose_budget(candidates):
    best=max(score for score,epoch in candidates)
    return min((item for item in candidates if item[0]>=best-1e-12),key=lambda item:item[1])
