"""
CRD (Contrastive Representation Distillation) for survival analysis.
Adapted from pa_ge/CL_utils/CRD_loss.py — simplified API:
no option object needed, all hyperparams passed directly.
"""

import torch
import torch.nn as nn

from .memory_new import ContrastMemory_v3

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD loss: contrastive learning between teacher and student features.

    Uses a memory bank of all training samples for efficient negative sampling.
    """

    def __init__(
        self,
        s_dim: int = 512,
        t_dim: int = 512,
        feat_dim: int = 128,
        n_data: int = 1024,
        nce_k: int = 20,
        nce_p: int = 15,
        nce_p2: int = 10,
        nce_k2: int = 512,
        nce_t: float = 0.07,
        nce_m: float = 0.5,
        select_pos_pairs: bool = True,
        select_neg_pairs: bool = True,
        select_pos_mode: str = "mid",
    ):
        super().__init__()
        self.P2 = nce_p2
        self.select_pos_mode = select_pos_mode

        self.embed_s = nn.Sequential(
            nn.Linear(s_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.embed_t = nn.Sequential(
            nn.Linear(t_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        self.contrast = ContrastMemory_v3(
            inputSize=feat_dim,
            outputSize=n_data,
            P=nce_p,
            K=nce_k,
            T=nce_t,
            momentum=nce_m,
            select_pos_pairs=select_pos_pairs,
            P2=nce_p2,
            select_neg_pairs=select_neg_pairs,
            K2=nce_k2,
        )

    def forward(self, epoch, f_s, f_t, idx):
        """
        Args:
            epoch: current epoch (controls positive pair curriculum)
            f_s: student features [B, s_dim]
            f_t: teacher features [B, t_dim] (detached internally)
            idx: global sample indices [B] for memory bank lookup

        Returns:
            scalar contrastive loss
        """
        f_t = f_t.clone().detach()
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)

        out_s, out_t = self.contrast(
            epoch, f_s, f_t, idx, idx=None, select_pos_mode=self.select_pos_mode
        )

        s_loss = self._contrast_loss(out_s, self.P2)
        t_loss = self._contrast_loss(out_t, self.P2)
        return s_loss + t_loss

    @staticmethod
    def _contrast_loss(x, P):
        """Contrastive loss with P positive samples."""
        bsz = x.shape[0]
        N = x.size(1) - P
        m = N
        Pn = 1.0 / (bsz + eps)

        P_pos = x.narrow(1, 0, P) + eps
        log_D1 = torch.log(P_pos / (P_pos + m * Pn + eps))

        P_neg = x.narrow(1, P, N) + eps
        log_D0 = torch.log((m * Pn) / (P_neg + m * Pn + eps))

        loss = -(
            (log_D1.squeeze().sum(0) + log_D0.view(-1, 1).repeat(1, P).sum(0))
            / (bsz + eps)
        ).sum(0) / (P + eps)
        return loss
