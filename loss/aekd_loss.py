"""
AEKD (Adaptive Ensemble Knowledge Distillation) loss weighting.
Computes gradient similarity between each KD loss and the main (Cox) loss
w.r.t. the student feature backbone, then filters out harmful KD signals.

Uses torch.autograd.grad instead of .backward() to avoid polluting
parameter gradients (the outer training loop handles the real backward).

Based on pa_ge/train_test_ct_multi_distill.py AEKD_loss_v2.
"""

import torch


def AEKD_loss_v2(main_loss, feat_s, kd_loss_list):
    """
    Gradient-guided automatic KD loss weighting.

    Computes the gradient of each KD loss and the main loss w.r.t. the
    student feature `feat_s`. KD losses with negative cosine similarity
    to the main loss gradient are discarded (scale=0).

    Uses torch.autograd.grad — does NOT call .backward(), so parameter
    gradients are unaffected. The caller is responsible for the final backward.

    Args:
        main_loss: scalar Cox loss tensor
        feat_s: student feature tensor [B, D]
        kd_loss_list: list of scalar KD loss tensors

    Returns:
        total_kd_loss: weighted sum of KD losses, scalar
    """
    # Gradient of each KD loss w.r.t. student feature
    kd_grads = []
    for loss_t in kd_loss_list:
        grad = torch.autograd.grad(
            loss_t, feat_s, retain_graph=True, create_graph=False
        )[0]
        kd_grads.append(grad)

    # Gradient of main (Cox) loss w.r.t. student feature
    main_grad = torch.autograd.grad(
        main_loss, feat_s, retain_graph=True, create_graph=False
    )[0]

    # Flatten to [N, B*D] and [1, B*D]
    kd_all = torch.stack(kd_grads).view(len(kd_grads), -1)  # [N, B*D]
    main_all = main_grad.view(1, -1)  # [1, B*D]

    kd_norm = torch.norm(kd_all, p=2, dim=1, keepdim=True) + 1e-8
    main_norm = torch.norm(main_all, p=2, dim=1, keepdim=True) + 1e-8

    # Cosine similarity [N, 1] → [N]
    similarity = torch.matmul(kd_all, main_all.T) / torch.matmul(kd_norm, main_norm)
    similarity = similarity.squeeze()

    # Keep only losses with positive gradient similarity
    scale = torch.where(similarity > 0, 1.0, 0.0).to(similarity.device)

    losses_tensor = torch.stack(kd_loss_list)
    total_kd = torch.sum(scale * losses_tensor)
    return total_kd
