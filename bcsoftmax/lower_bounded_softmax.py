# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from typing import Any

import torch


def _lbsoftmax_vector_naive(x: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """A naive implementation of Lower-Bounded Softmax function for vector.
    This is a naive implementation for presentation and explanation.
    This implementation is numerically unstable and does not consider edge cases.

    Args:
        x (Tensor): input vector. shape: (num_classes, )
        lb (Tensor): lower bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (Tensor): output probability vector. shape: (num_classes, ).
            Satisfying the constraints y[i] <= lb[i] for all i and
            torch.sum(y) = 1.

    """
    exp_x = torch.exp(x)
    # sorting
    _, indices = torch.sort(lb / exp_x, descending=True)
    exp_x = exp_x[indices]
    lb = lb[indices]
    # find V_l
    r = torch.flip(torch.cumsum(torch.flip(exp_x, dims=(0,)), dim=0), dims=(0,))
    s = 1.0 - (torch.cumsum(lb, dim=0) - lb)
    z = r / s
    is_in_V_l = exp_x / z < lb
    # compute outputs
    s = 1 - torch.sum(torch.where(is_in_V_l, lb, 0.0))
    r = torch.sum(torch.where(is_in_V_l, 0.0, exp_x))
    y = torch.where(is_in_V_l, lb, exp_x * s / r)
    # undo sorting
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]


def _lbsoftmax_vector(x: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """Lower-Bounded Softmax function for vector.
    This function is more numerically stable than `_lbsoftmax_vector_naive`.

    Args:
        x (torch.Tensor): input vector. shape: (num_classes, )
        lb (torch.Tensor): lower bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (torch.Tensor): output probability vector. shape: (num_classes, )
            Satisfying the constraints y[i] >= lb[i] for all i and
            torch.sum(y) = 1.

    """
    lb = torch.clip(lb, min=0.0)
    if torch.any(lb > 1.0):
        raise ValueError("lb has element/elements greater than 1.")
    if torch.any(torch.sum(lb) > 1.0):
        raise ValueError("sum of lb is greather than 1.")

    # sorting O(n \log n)
    _, indices = torch.sort(torch.log(lb) - x, descending=True)
    x = x[indices]
    lb = lb[indices]
    # find V_l O(n)
    log_r = torch.flip(torch.logcumsumexp(torch.flip(x, dims=(0,)), dim=0), dims=(0,))
    s = 1.0 - (torch.cumsum(lb, dim=0) - lb)
    is_in_V_l = (lb == 1.0) | (x - log_r + torch.log(s) < torch.log(lb))
    # compute outputs O(n)
    max_x_not_in_V_l = torch.max(torch.where(~is_in_V_l, x, -torch.inf))
    exp_x_not_in_V_l = torch.where(~is_in_V_l, torch.exp(x - max_x_not_in_V_l), 0.0)
    s = 1 - torch.sum(torch.where(is_in_V_l, lb, 0.0))
    r = torch.sum(exp_x_not_in_V_l)
    y = torch.where(is_in_V_l, lb, exp_x_not_in_V_l * s / r)
    # undo sorting O(n \log n)
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]


class LBSoftmaxVector(torch.autograd.Function):
    """Autograd implementation of Lower-Bounded Softmax function for vector."""

    @staticmethod
    def forward(x: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        return _lbsoftmax_vector(x, lb)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        _, lb = inputs
        is_in_V_l = lb == output
        s = 1 - torch.sum(torch.where(is_in_V_l, lb, 0.0))
        ctx.save_for_backward(output, is_in_V_l, s)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        output, is_in_V_l, s = ctx.saved_tensors
        q = output * (~is_in_V_l)
        vq = grad_y * q
        vq_sum = torch.sum(vq)
        vJx = vq - vq_sum * q / s
        vJlb = is_in_V_l * grad_y - vq_sum * is_in_V_l / s
        return vJx, vJlb


def lbsoftmax_vector(x: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """Lower Bounded Softmax function for vector.

    Args:
        x (torch.Tensor): input vector. shape: (num_classes, )
        lb (torch.Tensor): lower bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (torch.Tensor): output probability vector.
            Satisfying the constraints y[i] >= lb[i] for all i and
            torch.sum(y) = 1.

    """
    return LBSoftmaxVector.apply(x, lb)


def _lbsoftmax_batch(x: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """Lower-Bounded Softmax function for batch.

    Args:
        x (torch.Tensor): input matrix. shape: (batch_size, num_classes)
        lb (torch.Tensor): lower bound (constraint) matrix.
            shape: (batch_size, num_classes)

    Returns:
        y (torch.Tensor): output probability matrix. shape: (batch_size, num_classes)
            Satisfying the constraints y[i, j] >= lb[i, j] for all i,j and
            torch.sum(y, dim=1) = all-ones vector.

    """
    lb = torch.clip(lb, min=0.0)
    if torch.any(lb > 1.0):
        raise ValueError("lb has element/elements greater than 1.")

    if torch.any(torch.sum(lb, dim=1) > 1.0):
        raise ValueError("lb has rows whose sum is/are greater than 1.")

    # sorting O(n \log n)
    _, indices = torch.sort(torch.log(lb) - x, descending=True, dim=1)
    x = torch.take_along_dim(x, indices, dim=1)
    lb = torch.take_along_dim(lb, indices, dim=1)
    # find V_l O(n)
    log_r = torch.flip(torch.logcumsumexp(torch.flip(x, dims=(1,)), dim=1), dims=(1,))
    s = 1.0 - (torch.cumsum(lb, dim=1) - lb)
    is_in_V_l = (lb == 1.0) | (x - log_r + torch.log(s) < torch.log(lb))
    # compute outputs O(n)
    max_x_not_in_V_l, _ = torch.max(
        torch.where(~is_in_V_l, x, -torch.inf), dim=1, keepdim=True
    )
    exp_x_not_in_V_l = torch.where(~is_in_V_l, torch.exp(x - max_x_not_in_V_l), 0.0)
    s = 1 - torch.sum(torch.where(is_in_V_l, lb, 0.0), dim=1, keepdim=True)
    r = torch.sum(exp_x_not_in_V_l, dim=1, keepdim=True)
    y = torch.where(is_in_V_l, lb, exp_x_not_in_V_l * s / r)
    # undo sorting O(n \log n)
    _, inv_indices = torch.sort(indices, descending=False, dim=1)
    return torch.take_along_dim(y, inv_indices, dim=1)


class LBSoftmaxBatch(torch.autograd.Function):
    """Autograd implementation of Lower-Bounded Softmax function for batch."""

    @staticmethod
    def forward(x: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        return _lbsoftmax_batch(x, lb)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, int],
        output: torch.Tensor,
    ):
        _, lb = inputs
        is_in_V_l = lb == output
        s = 1 - torch.sum(torch.where(is_in_V_l, lb, 0.0), dim=1, keepdim=True)
        ctx.save_for_backward(output, is_in_V_l, s)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        output, is_in_V_l, s = ctx.saved_tensors
        q = output * (~is_in_V_l)
        vq = grad_y * q
        vq_sum = torch.sum(vq, dim=1, keepdim=True)
        vJx = vq - vq_sum * q / s
        vJlb = is_in_V_l * grad_y - vq_sum * is_in_V_l / s
        return vJx, vJlb, None


def lbsoftmax_batch(x: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """Lower-Bounded Softmax function for batch.

    Args:
        x (torch.Tensor): input matrix. shape: (batch_size, num_classes)
        lb (torch.Tensor): lower bound (constraint) matrix.
            shape: (batch_size, num_classes)

    Returns:
        y (torch.Tensor): output probability matrix. shape: (batch_size, num_classes)
            Satisfying the constraints y[i, j] >= lb[i, j] for all i,j and
            torch.sum(y, dim=1) = all-ones vector.

    """
    return LBSoftmaxBatch.apply(x, lb)


def lbsoftmax(x: torch.Tensor, lb: torch.Tensor, dim=None) -> torch.Tensor:
    """Lower-Bounded Softmax function for tensor.

    Args:
        x (torch.Tensor): input tensor.
        lb (torch.Tensor): lower bound (constraint) tensor.
        dim (int): the dimension to reduce.

    Returns:
        y (torch.Tensor): output probability tensor.
            Satisfying the constraints y >= lb and
            torch.sum(y, dim=dim) = all-ones tensor.

    """
    if x.ndim == 1:
        return lbsoftmax_vector(x, lb)
    if dim is None:
        dim = -1
    num_classes = x.shape[dim]
    x = x.swapaxes(dim, -1)
    swapped_shape = x.shape
    x = x.view(-1, num_classes)
    lb = lb.swapaxes(dim, -1).view(-1, num_classes)
    y = lbsoftmax_batch(x, lb)
    return y.view(swapped_shape).swapaxes(dim, -1)
