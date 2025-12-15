"""Commonly used loss helpers for PyTorch models in FuxiCTR.

Each loss exposes a function-style API so it can be referenced
via the ``loss`` field in config files (resolved in ``get_loss``).
"""
import torch
import torch.nn.functional as F


def _apply_reduction(loss_tensor, reduction="mean"):
    """Apply a Torch-style reduction to the given tensor."""
    if reduction == "sum":
        return loss_tensor.sum()
    if reduction == "none":
        return loss_tensor
    # Default to mean to stay consistent with torch losses
    return loss_tensor.mean()


def FocalLoss(y_pred, y_true, gamma=2.0, alpha=0.25, reduction="mean", eps=1e-8):
    """Binary focal loss operating on sigmoid probabilities.

    Args:
        y_pred (Tensor): Predicted probabilities in range [0, 1].
        y_true (Tensor): Binary targets with the same shape as y_pred.
        gamma (float): Focusing parameter that down-weights easy samples.
        alpha (float): Class balancing factor (applied to positive class).
        reduction (str): "mean", "sum", or "none".
        eps (float): Numerical stability constant for log operations.
    """
    y_true = y_true.to(y_pred.dtype)
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    p_t = torch.where(y_true > 0, y_pred, 1.0 - y_pred)
    alpha_t = torch.where(y_true > 0, torch.full_like(y_pred, alpha), torch.full_like(y_pred, 1.0 - alpha))
    focal_weight = alpha_t * torch.pow(1.0 - p_t, gamma)
    loss = -focal_weight * torch.log(p_t)
    return _apply_reduction(loss, reduction)


def HingeLoss(y_pred, y_true, margin=1.0, reduction="mean"):
    """Standard hinge loss for binary classification.

    Args:
        y_pred (Tensor): Raw model scores.
        y_true (Tensor): Binary targets {0,1} or {-1,1}.
        margin (float): Desired margin between positive and negative scores.
        reduction (str): "mean", "sum", or "none".
    """
    y_true = torch.where(y_true > 0, torch.ones_like(y_true), -torch.ones_like(y_true))
    loss = torch.clamp(margin - y_true * y_pred, min=0.0)
    return _apply_reduction(loss, reduction)


def BPRLoss(y_pred, y_true=None, reduction="mean"):
    """Bayesian Personalized Ranking loss.

    Expects the prediction tensor to contain positive and negative scores
    stacked along the last dimension: ``[..., 0]`` -> positive, ``[..., 1]`` -> negative.
    The ``y_true`` argument is accepted for API compatibility but ignored.
    """
    if y_pred.ndim < 2 or y_pred.size(-1) < 2:
        raise ValueError("BPRLoss expects y_pred[..., 0] as positive and y_pred[..., 1] as negative scores.")
    pos_scores = y_pred[..., 0]
    neg_scores = y_pred[..., 1]
    loss = -torch.logsigmoid(pos_scores - neg_scores)
    return _apply_reduction(loss, reduction)
