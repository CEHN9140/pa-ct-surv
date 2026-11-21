"""Per-patient attention penalties, applied only to the joint training Cox batch."""
import math
import torch
from loss.survival_loss import cox_loss


def attention_penalty(weights,mode,q_min=.1):
    p=weights.reshape(-1);n=p.numel()
    if n<1:raise ValueError('Empty attention bag')
    if mode=='baseline' or n==1:return p.sum()*0.
    if mode=='entropy':
        entropy=-(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum()
        return (1-entropy/math.log(n)).clamp_min(0.)
    if mode=='neff':
        if not 0<q_min<=1:raise ValueError('q_min must lie in(0,1]')
        q=1/(n*p.square().sum())
        return (q_min-q).clamp_min(0.).square()
    raise ValueError('Unknown attention penalty')


class AttentionObjective:
    def __init__(self,model,mode='baseline',weight=0.,q_min=.1):
        if mode not in ['baseline','entropy','neff'] or not math.isfinite(weight) or weight<0:raise ValueError('Invalid regularization')
        self.model=model;self.mode=mode;self.weight=float(weight);self.q_min=q_min
        self.pending=[];self.components=[];self.patient_exposures=0;self.active_exposures=0
        self.handle=model.register_forward_hook(self.capture)
    def capture(self,module,inputs,output):
        if not module.training or self.weight==0:return
        weights=output[2]
        if weights.ndim!=3 or weights.shape[0]!=1 or weights.shape[2]!=1:raise ValueError('One WSI/one attention branch per loader batch required')
        self.pending.append(attention_penalty(weights,self.mode,self.q_min))
    def __call__(self,risk,time,event):
        cox=cox_loss(risk,time,event)
        if not self.model.training:
            if self.pending:raise RuntimeError('Unconsumed training attention before evaluation')
            return cox
        self.patient_exposures+=risk.numel()
        if self.weight==0:
            self.components.append(dict(cox=float(cox.detach()),penalty=0.,total=float(cox.detach()),patients=risk.numel(),active_patients=0))
            return cox
        if len(self.pending)!=risk.numel():raise RuntimeError('Attention penalties do not match Cox patients')
        values=torch.stack(self.pending);penalty=values.mean();self.pending.clear()
        active=int((values.detach()>0).sum());self.active_exposures+=active
        total=cox+self.weight*penalty
        self.components.append(dict(cox=float(cox.detach()),penalty=float(penalty.detach()),total=float(total.detach()),patients=risk.numel(),active_patients=active))
        return total
    def close(self):
        self.handle.remove()
        if self.pending:raise RuntimeError('Incomplete Cox batch left unconsumed attention')
