"""The existing ABMIL with narrower projections; no change to input or pooling."""

import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from model.abmil_cox import ABMIL
from utils import seed_everything


def make_model(seed=123,device='cpu',attention_dim=32):
    if attention_dim not in (32,128): raise ValueError('attention_dim must be 32 or 128')
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return ABMIL(in_dim=1024,hidden_dim=128,attention_dim=attention_dim,
                 post_hidden_dim=64,dropout=.25).to(device)


def parameter_counts(model):
    # Construct the reference without perturbing the caller's initialization RNG.
    with torch.random.fork_rng(devices=[]):
        full=sum(p.numel() for p in ABMIL().parameters())
    small=sum(p.numel() for p in model.parameters())
    return dict(full_parameters=full,light_parameters=small,reduction_fraction=1-small/full)
