"""
Memory bank for CRD contrastive distillation.
Directly copied from pa_ge/CL_utils/memory_new.py.
"""

import math

import numpy as np
import torch
from torch import nn


class ContrastMemory(nn.Module):
    """Memory buffer that supplies large amount of negative samples."""

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer("params", torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer(
            "memory_v1",
            torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv),
        )
        self.register_buffer(
            "memory_v2",
            torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv),
        )

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))

        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class ContrastMemory_v3(nn.Module):
    """Select positive and negative pairs simultaneously."""

    def __init__(
        self,
        inputSize,
        outputSize,
        P,
        K,
        T=0.07,
        momentum=0.5,
        select_pos_pairs=True,
        P2=10,
        select_neg_pairs=True,
        K2=512,
    ):
        super(ContrastMemory_v3, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.P = P
        self.K = K
        self.P2 = P2
        self.K2 = K2
        self.select_pos_pairs = select_pos_pairs
        self.select_neg_pairs = select_neg_pairs

        self.register_buffer("params", torch.tensor([K, T, -1, -1, momentum, P]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer("memory_v1", torch.randn(outputSize, inputSize).mul_(stdv))
        self.register_buffer("memory_v2", torch.randn(outputSize, inputSize).mul_(stdv))

    def forward(
        self, epoch, v1, v2, y, idx=None, select_pos_mode="mid", dynamic_p2=None
    ):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()
        P = int(self.params[5].item())

        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        P2 = max(1, min(dynamic_p2 if dynamic_p2 is not None else self.P2, P))

        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + P)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        idx = torch.clamp(idx, 0, outputSize - 1)
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + P, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.clamp(out_v2 / T, min=-20, max=20)
        out_v2 = torch.exp(out_v2)

        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + P, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.clamp(out_v1 / T, min=-20, max=20)
        out_v1 = torch.exp(out_v1)

        if self.select_pos_pairs:
            v1_norm = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
            v2_norm = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-8)
            weight_v1_norm = weight_v1 / (
                torch.norm(weight_v1, dim=2, keepdim=True) + 1e-8
            )
            weight_v2_norm = weight_v2 / (
                torch.norm(weight_v2, dim=2, keepdim=True) + 1e-8
            )

            t_relation = torch.bmm(
                weight_v1_norm, v1_norm.view(batchSize, inputSize, 1)
            )
            s_relation = torch.bmm(
                weight_v2_norm, v2_norm.view(batchSize, inputSize, 1)
            )

            t_relation_pos, s_relation_pos = (
                t_relation.narrow(1, 0, P),
                s_relation.narrow(1, 0, P),
            )
            indices = torch.sort(
                t_relation_pos - s_relation_pos, dim=1, descending=True
            )[1]

            if select_pos_mode == "hard":
                selected_indices = indices[:, :P2, :].squeeze(-1)
            elif select_pos_mode == "mid":
                index = torch.tensor(
                    np.random.choice(np.arange(0, P, 1), min(P2, P), replace=False)
                ).cuda()
                selected_indices = indices.index_select(1, index).squeeze(-1)
            elif select_pos_mode == "random":
                index = torch.tensor(np.random.randint(0, P, P2)).cuda()
                selected_indices = indices.index_select(1, index).squeeze(-1)
            elif select_pos_mode == "curriculum":
                interval = 4 - np.ceil(3 * epoch)
                index = torch.tensor(
                    np.random.randint(50 * (interval - 1), 50 * interval, P2)
                ).cuda()
                selected_indices = indices.index_select(1, index).squeeze(-1)

            selected_indices[:, 0] = 0
            sample_index = (
                torch.arange(0, out_v2.shape[0], 1).view(-1, 1).repeat(1, P2).cuda()
            )
            selected_indices = (sample_index * (K + P) + selected_indices).view(-1)
            selected_indices = torch.clamp(selected_indices, 0, batchSize * (K + P) - 1)

            out_v2_pos = (
                out_v2.view(-1, 1).index_select(0, selected_indices).view(-1, P2, 1)
            )
            out_v1_pos = (
                out_v1.view(-1, 1).index_select(0, selected_indices).view(-1, P2, 1)
            )
        else:
            out_v2_pos = out_v2.narrow(1, 0, P2)
            out_v1_pos = out_v1.narrow(1, 0, P2)

        if self.select_neg_pairs:
            t_relation_neg, s_relation_neg = (
                t_relation.narrow(1, P, K),
                s_relation.narrow(1, P, K),
            )
            indices = torch.sort(
                t_relation_neg - s_relation_neg, dim=1, descending=False
            )[1]
            K2 = min(self.K2, K)
            selected_indices_neg = P + indices[:, :K2, :].squeeze(-1)
            sample_index = (
                torch.arange(0, out_v2.shape[0], 1).view(-1, 1).repeat(1, K2).cuda()
            )
            selected_indices_neg = (sample_index * (K + P) + selected_indices_neg).view(
                -1
            )
            selected_indices_neg = torch.clamp(
                selected_indices_neg, 0, batchSize * (K + P) - 1
            )
            out_v2_neg = (
                out_v2.view(-1, 1).index_select(0, selected_indices_neg).view(-1, K2, 1)
            )
            out_v1_neg = (
                out_v1.view(-1, 1).index_select(0, selected_indices_neg).view(-1, K2, 1)
            )
        else:
            out_v2_neg = out_v2.narrow(1, P, K)
            out_v1_neg = out_v1.narrow(1, P, K)

        out_v2 = torch.cat((out_v2_pos, out_v2_neg), 1)
        out_v1 = torch.cat((out_v1_pos, out_v1_neg), 1)

        if Z_v1 < 0 or epoch == 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0 or epoch == 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj
