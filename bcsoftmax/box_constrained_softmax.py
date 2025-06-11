# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from typing import Any

import torch

from .lower_bounded_softmax import lbsoftmax_batch
from .upper_bounded_softmax import EPS, ubsoftmax_batch, ubsoftmax_batch_cond


def _bcsoftmax_vector(
    x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor
) -> torch.Tensor:
    """Box-Constrained Softmax function for vector.

    Args:
        x (torch.Tensor): input vector. shape: (num_classes, )
        lb (torch.Tensor): lower bound (constraint) vector. shape: (num_classes, )
        ub (torch.Tensor): upper bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (torch.Tensor): output probability vector. shape: (num_classes, )
            Satisfying the constraints lb[i] <= y[i] <= ub[i] for all i and
            torch.sum(y) = 1.

    """
    lb = torch.clip(lb, min=0.0)
    if torch.any(lb > ub):
        raise ValueError("Not lb <= ub.")

    # sorting O(n \log n)
    num_classes = len(x)
    _, indices = torch.sort(torch.log(lb) - x, descending=True)
    x = x[indices]
    lb = lb[indices]
    ub = ub[indices]
    X = torch.tile(x, (num_classes, 1))  # (num_classes, num_classes)
    # B[i] = concat(lb[:i], ub[i:])
    eq_cond = torch.tril(torch.ones_like(X, dtype=torch.bool), diagonal=-1)
    B = torch.where(
        eq_cond,
        torch.tile(lb, (num_classes, 1)),
        torch.tile(ub, (num_classes, 1)),
    )
    is_feasible_constraints = torch.sum(B, dim=1) >= 1.0
    Y = ubsoftmax_batch_cond(
        X[is_feasible_constraints],
        B[is_feasible_constraints],
        eq_cond[is_feasible_constraints],
    )
    rho_a = torch.all(lb.view(1, -1) <= Y, dim=1).nonzero()[0]
    y = Y[rho_a].view(-1)
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]


class BCSoftmaxVector(torch.autograd.Function):
    """Autograd implementation of Box-Constrained Softmax function for vector."""

    @staticmethod
    def forward(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return _bcsoftmax_vector(x, lb, ub)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        _, lb, ub = inputs
        is_in_V_l = lb == output
        is_in_V_u = ub == output
        s = (
            1
            - torch.sum(torch.where(is_in_V_l, lb, 0.0))
            - torch.sum(torch.where(is_in_V_u, ub, 0.0))
        )
        ctx.save_for_backward(output, is_in_V_l, is_in_V_u, s)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, is_in_V_l, is_in_V_u, s = ctx.saved_tensors
        q = output * (~is_in_V_l) * (~is_in_V_u)
        vq = grad_y * q
        vq_sum = torch.sum(vq)
        vJx = torch.where(s < EPS, 0.0, vq - vq_sum * q / s)
        vJlb = is_in_V_l * grad_y - torch.where(s < EPS, 0.0, vq_sum * is_in_V_l / s)
        vJub = is_in_V_u * grad_y - torch.where(s < EPS, 0.0, vq_sum * is_in_V_u / s)
        return vJx, vJlb, vJub


def bcsoftmax_vector(
    x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor
) -> torch.Tensor:
    """Box-Constrained Softmax function for vector.

    Args:
        x (torch.Tensor): input vector. shape: (num_classes, )
        lb (torch.Tensor): lower bound (constraint) vector. shape: (num_classes, )
        ub (torch.Tensor): upper bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (torch.Tensor): output probability vector. shape: (num_classes, )
            Satisfying the constraints lb[i] <= y[i] <= ub[i] for all i and
            torch.sum(y) = 1.

    """
    return BCSoftmaxVector.apply(x, lb, ub)


def _bcsoftmax_batch(
    x: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """Box-Constrained Softmax function for batch.

    Args:
        x (torch.Tensor): input matrix. shape: (batch_size, num_classes)
        lb (torch.Tensor): lower bound (constraint) matrix.
            shape: (batch_size, num_classes)
        ub (torch.Tensor): upper bound (constraint) matrix.
            shape: (batch_size, num_classes)

    Returns:
        y (torch.Tensor): output probability matrix.
            Satisfying the constraints lb[i, j] <= y[i,j] <= ub[i,j] for all i,j and
            torch.sum(y, dim=1) = all-ones vector.

    """

    batch_size, num_classes = x.shape
    # sorting O(n \log n)
    _, indices = torch.sort(torch.log(lb) - x, descending=True, dim=1)
    x = torch.take_along_dim(x, indices, dim=1)
    lb = torch.take_along_dim(lb, indices, dim=1)
    ub = torch.take_along_dim(ub, indices, dim=1)

    X = torch.tile(x, (num_classes - 1, 1))
    Ub = torch.tile(ub, (num_classes - 1, 1))
    Lb = torch.tile(lb, (num_classes - 1, 1))
    eq_cond = torch.repeat_interleave(
        torch.tril(
            torch.ones(
                (num_classes - 1, num_classes), dtype=torch.bool, device=lb.device
            ),
            diagonal=0,
        ),
        repeats=batch_size,
        dim=0,
    )
    B = torch.where(eq_cond, Lb, Ub)
    is_valid_constraints = torch.sum(B, dim=1) >= 1.0
    Y = torch.zeros_like(X)

    X = X[is_valid_constraints]
    B = B[is_valid_constraints]
    eq_cond = eq_cond[is_valid_constraints]

    Y[is_valid_constraints] = ubsoftmax_batch_cond(X, B, eq_cond)
    is_feasible = torch.all(Y >= Lb, dim=1).view(num_classes - 1, batch_size)
    rho_a = torch.argmin(
        torch.where(
            is_feasible,
            torch.tile(
                torch.arange(num_classes - 1, device=x.device), (batch_size, 1)
            ).transpose(1, 0),
            num_classes - 1,
        ),
        dim=0,
    )
    rho_a_indices = torch.arange(batch_size, device=x.device) + rho_a * batch_size
    y = Y[rho_a_indices]
    # undo sorting O(n \log n)
    _, inv_indices = torch.sort(indices, descending=False, dim=1)
    return torch.take_along_dim(y, inv_indices, dim=1)


class BCSoftmaxBatch(torch.autograd.Function):
    """Autograd implementation of Box-Constrained softmax function for batch."""

    @staticmethod
    def forward(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return _bcsoftmax_batch(x, lb, ub)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        _, lb, ub = inputs
        is_in_V_l = lb == output
        is_in_V_u = ub == output
        s = (
            1
            - torch.sum(torch.where(is_in_V_l, lb, 0.0), dim=1, keepdim=True)
            - torch.sum(torch.where(is_in_V_u, ub, 0.0), dim=1, keepdim=True)
        )  # (batch_size, 1)
        ctx.save_for_backward(output, is_in_V_l, is_in_V_u, s)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, is_in_V_l, is_in_V_u, s = ctx.saved_tensors
        q = output * (~is_in_V_l) * (~is_in_V_u)
        vq = grad_y * q
        vq_sum = torch.sum(vq, dim=1, keepdim=True)
        vJx = torch.where(
            s < EPS,
            0.0,
            vq - vq_sum * q / s,
        )
        vJlb = is_in_V_l * grad_y - torch.where(s < EPS, 0.0, vq_sum * is_in_V_l / s)
        vJub = is_in_V_u * grad_y - torch.where(s < EPS, 0.0, vq_sum * is_in_V_u / s)
        return vJx, vJlb, vJub


def bcsoftmax_batch(
    x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor
) -> torch.Tensor:
    """Box-Constrained Softmax function for batch.

    Args:
        x (torch.Tensor): input matrix. shape: (batch_size, num_classes)
        lb (torch.Tensor): lower bound (constraint) matrix.
            shape: (batch_size, num_classes)
        ub (torch.Tensor): upper bound (constraint) matrix.
            shape: (batch_size, num_classes)

    Returns:
        y (torch.Tensor): output probability matrix. shape: (batch_size, num_classes)
            Satisfying the constraints lb[i, j] <= y[i,j] <= ub[i,j] for all i,j and
            torch.sum(y, dim=1) = all-ones vector.

    """
    lb = torch.clip(lb, min=0.0)
    if torch.any(lb > ub):
        raise ValueError("Not lb <= ub.")

    # First, computes ub/lbsoftmax_batch. If the output vector fortunately satisfies
    # the lower/upper bound constraints, then simply outputs it
    ubsoftmax = ubsoftmax_batch(x, ub)
    lbsoftmax = lbsoftmax_batch(x, lb)
    is_ub = torch.all(ubsoftmax >= lb, dim=1)
    is_lb = torch.all(lbsoftmax <= ub, dim=1)
    y = torch.zeros_like(x)
    y[is_ub] = ubsoftmax[is_ub]
    y[is_lb] = lbsoftmax[is_lb]
    # For neither is_ub nor is_lb instances,
    # computes the box-constrained vectors by the proposed algorithm
    is_bc = ~(is_ub | is_lb)
    if torch.any(is_bc):
        y[is_bc] = BCSoftmaxBatch.apply(x[is_bc], lb[is_bc], ub[is_bc])
    return y


def bcsoftmax(
    x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, dim=None
) -> torch.Tensor:
    """Box-Constrained Softmax function for tensor.

    Args:
        x (torch.Tensor): input tensor.
        lb (torch.Tensor): lower bound (constraint) tensor.
        ub (torch.Tensor): upper bound (constraint) tensor.
        dim (int): the dimension to reduce.

    Returns:
        y (torch.Tensor): output probability tensor.
            Satisfying the constraints lb <= y <= ub and
            torch.sum(y, dim=1) = all-ones tensor.

    """
    if x.ndim == 1:
        return bcsoftmax_vector(x, lb, ub)
    if dim is None:
        dim = -1
    num_classes = x.shape[dim]
    x = x.swapaxes(dim, -1)
    swapped_shape = x.shape
    x = x.view(-1, num_classes)
    lb = lb.swapaxes(dim, -1).view(-1, num_classes)
    ub = ub.swapaxes(dim, -1).view(-1, num_classes)
    y = bcsoftmax_batch(x, lb, ub)
    return y.view(swapped_shape).swapaxes(dim, -1)
