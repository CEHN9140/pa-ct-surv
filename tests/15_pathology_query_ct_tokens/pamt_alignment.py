"""PAMT training-entry Top-K token alignment, without cross-patient negatives."""

import torch.nn.functional as F


def pamt_alignment_loss(pa_tokens, ct_tokens, topk=2):
    if (pa_tokens.ndim != 3 or ct_tokens.ndim != 3
            or pa_tokens.shape[0] != ct_tokens.shape[0]
            or pa_tokens.shape[-1] != ct_tokens.shape[-1]
            or pa_tokens.size(1) == 0 or not isinstance(topk, int)
            or not 1 <= topk <= ct_tokens.size(1)):
        raise ValueError('Expected matching [B,Q,D]/[B,K,D] tokens and 1 <= topk <= K')
    # PAMT uses unscaled dot products before cross-attention, with no extra L2.
    scores = pa_tokens @ ct_tokens.transpose(-2, -1)
    # Equivalent to sorting keys and CE with target=1 at each of the top-K keys.
    # Deliberately sum, not average, over K: original soft targets sum to K.
    return -F.log_softmax(scores, dim=-1).topk(topk, dim=-1).values.sum(dim=-1).mean()
