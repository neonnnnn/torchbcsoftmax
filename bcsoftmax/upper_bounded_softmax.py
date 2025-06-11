# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

from typing import Any

import torch

EPS = 1e-6


def _ubsoftmax_vector_naive(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """A naive implementation of Upper-Bounded Softmax function for vector.
    This is a naive implementation for presentation and explanation.
    This implementation is numerically unstable and does not consider edge cases.

    Args:
        x (Tensor): input vector. shape: (num_classes, )
        ub (Tensor): upper bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (Tensor): output probability vector. shape: (num_classes, )
            Satisfying the constraints y[i] <= ub[i] for all i and
            torch.sum(y) = 1.

    """
    exp_x = torch.exp(x)
    # sorting
    _, indices = torch.sort(ub / exp_x, descending=False)
    exp_x = exp_x[indices]
    ub = ub[indices]
    # find V_u
    r = torch.flip(torch.cumsum(torch.flip(exp_x, dims=(0,)), dim=0), dims=(0,))
    s = 1.0 - (torch.cumsum(ub, dim=0) - ub)
    z = r / s
    is_in_V_u = exp_x / z > ub
    # compute outputs
    s = 1 - torch.sum(torch.where(is_in_V_u, ub, 0.0))
    r = torch.sum(torch.where(is_in_V_u, 0.0, exp_x))
    y = torch.where(is_in_V_u, ub, exp_x * s / r)
    # undo sorting
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]


def _ubsoftmax_vector(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Upper-Bounded Softmax function for vector.
    This function is more numerically stable than `_ubsoftmax_vector_naive`.

    Args:
        x (torch.Tensor): input vector. shape: (num_classes, )
        ub (torch.Tensor): upper bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (torch.Tensor): output probability vector. shape: (num_classes, )
            Satisfying the constraints y[i] <= ub[i] for all i and
            torch.sum(y) = 1.

    """
    if torch.any(ub < 0):
        raise ValueError("ub has negative element/elements.")

    if torch.any(torch.sum(ub) < 1.0):
        raise ValueError("sum of ub is less than 1.")

    # sorting O(n \log n)
    _, indices = torch.sort(torch.log(ub) - x, descending=False)
    x = x[indices]
    ub = ub[indices]
    # find V_u O(n)
    log_r = torch.flip(torch.logcumsumexp(torch.flip(x, dims=(0,)), dim=0), dims=(0,))
    s = 1.0 - (torch.cumsum(ub, dim=0) - ub)
    is_in_V_u = (s - ub > 0) & (x - log_r + torch.log(s) > torch.log(ub))
    # compute outputs O(n)
    max_x_not_in_V_u = torch.max(torch.where(is_in_V_u, -torch.inf, x))
    exp_x = torch.where(~is_in_V_u, torch.exp(x - max_x_not_in_V_u), 0.0)
    s = 1 - torch.sum(torch.where(is_in_V_u, ub, 0.0))
    r = torch.sum(exp_x)
    y = torch.where(is_in_V_u, ub, exp_x * s / r)
    # undo sorting O(n \log n)
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]


class UBSoftmaxVector(torch.autograd.Function):
    """Autograd implementation of Upper-Bounded Softmax function for vector."""

    @staticmethod
    def forward(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return _ubsoftmax_vector(x, ub)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        _, ub = inputs
        is_in_V_u = ub == output
        s = 1 - torch.sum(torch.where(is_in_V_u, ub, 0.0))
        ctx.save_for_backward(output, is_in_V_u, s)

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, is_in_V_u, s = ctx.saved_tensors
        q = output * (~is_in_V_u)
        vq = grad_y * q
        vq_sum = torch.sum(vq)
        vJx = torch.where(s < EPS, 0.0, vq - vq_sum * q / s)
        vJub = torch.where(
            s < EPS,
            grad_y,
            is_in_V_u * grad_y - vq_sum * is_in_V_u / s,
        )
        return vJx, vJub


def ubsoftmax_vector(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Upper-Bounded Softmax function for vector.

    Args:
        x (torch.Tensor): input vector. shape: (num_classes, )
        ub (torch.Tensor): upper bound (constraint) vector. shape: (num_classes, )

    Returns:
        y (torch.Tensor): output probability vector. shape: (num_classes, )
            Satisfying the constraints y[i] <= ub[i] for all i and
            torch.sum(y) = 1.

    """
    return UBSoftmaxVector.apply(x, ub)


def _ubsoftmax_batch(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Upper-Bounded Softmax function for batch.

    Args:
        x (torch.Tensor): input matrix. shape: (batch_size, num_classes)
        ub (torch.Tensor): upper bound (constraint) matrix.
            shape: (batch_size, num_classes)

    Returns:
        y (torch.Tensor): output probability matrix. shape: (batch_size, num_classes)
            Satisfying the constraints y[i, j] <= ub[i, j] for all i,j and
            torch.sum(y, dim=1) = all-ones vector.

    """
    if torch.any(ub < 0):
        raise ValueError("ub has negative element/elements.")

    if torch.any(torch.sum(ub, dim=1) < 1.0):
        raise ValueError("ub has row/rows whose sum is/are less than 1.")

    # sorting O(n \log n)
    _, indices = torch.sort(torch.log(ub) - x, descending=False, dim=1)
    x = torch.take_along_dim(x, indices, dim=1)
    ub = torch.take_along_dim(ub, indices, dim=1)
    # find V_u O(n)
    log_r = torch.flip(torch.logcumsumexp(torch.flip(x, dims=(1,)), dim=1), dims=(1,))
    s = 1.0 - (torch.cumsum(ub, dim=1) - ub)
    is_in_V_u = (s - ub > 0) & (x - log_r + torch.log(s) > torch.log(ub))
    # compute outputs O(n)
    max_x_not_in_V_u, _ = torch.max(
        torch.where(~is_in_V_u, x, -torch.inf), dim=1, keepdim=True
    )
    exp_x = torch.where(~is_in_V_u, torch.exp(x - max_x_not_in_V_u), 0.0)
    s = 1 - torch.sum(torch.where(is_in_V_u, ub, 0.0), dim=1, keepdim=True)
    r = torch.sum(exp_x, dim=1, keepdim=True)
    y = torch.where(is_in_V_u, ub, exp_x * s / r)
    # undo sorting O(n \log n)
    _, inv_indices = torch.sort(indices, descending=False, dim=1)
    return torch.take_along_dim(y, inv_indices, dim=1)


class UBSoftmaxBatch(torch.autograd.Function):
    """Autograd implementation of Upper-Bounded Softmax function for batch."""

    @staticmethod
    def forward(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return _ubsoftmax_batch(x, ub)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, int],
        output: torch.Tensor,
    ):
        _, ub = inputs
        is_in_V_u = ub == output
        s = 1 - torch.sum(torch.where(is_in_V_u, ub, 0.0), dim=1, keepdim=True)
        ctx.save_for_backward(output, is_in_V_u, s)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        output, is_in_V_u, s = ctx.saved_tensors
        q = output * (~is_in_V_u)
        vq = grad_y * q
        vq_sum = torch.sum(vq, dim=1, keepdim=True)
        vJx = torch.where(s < EPS, 0.0, vq - vq_sum * q / s)
        vJub = torch.where(
            s < EPS,
            grad_y,
            torch.where(is_in_V_u, grad_y, 0.0) - vq_sum * is_in_V_u / s,
        )
        return vJx, vJub, None


def ubsoftmax_batch(x: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Upper-Bounded Softmax function for batch.

    Args:
        x (torch.Tensor): input matrix. shape: (batch_size, num_classes)
        ub (torch.Tensor): upper bound (constraint) matrix.
            shape: (batch_size, num_classes)

    Returns:
        y (torch.Tensor): output probability matrix. shape: (batch_size, num_classes)
            Satisfying the constraints y[i, j] <= ub[i, j] for all i,j and
            torch.sum(y, dim=1) = all-ones vector.

    """
    return UBSoftmaxBatch.apply(x, ub)


def ubsoftmax(x: torch.Tensor, ub: torch.Tensor, dim=None) -> torch.Tensor:
    """Upper-Bounded Softmax function for tensor.

    Args:
        x (torch.Tensor): input tensor.
        ub (torch.Tensor): upper bound (constraint) tensor.
        dim (int): the dimension to reduce.

    Returns:
        y (torch.Tensor): output probability tensor.
            Satisfying the constraints y <= ub and
            torch.sum(y, dim=dim) = all-ones tensor.

    """
    if x.ndim == 1:
        return ubsoftmax_vector(x, ub)
    if dim is None:
        dim = -1
    num_classes = x.shape[dim]
    x = x.swapaxes(dim, -1)
    swapped_shape = x.shape
    x = x.view(-1, num_classes)
    ub = ub.swapaxes(dim, -1).view(-1, num_classes)
    y = ubsoftmax_batch(x, ub)
    return y.view(swapped_shape).swapaxes(dim, -1)


def ubsoftmax_vector_cond(
    x: torch.Tensor, ub: torch.Tensor, eq_cond: torch.Tensor
) -> torch.Tensor:
    """Conditional Upper-Bounded Softmax function for vector.

    Args:
        x (torch.Tensor): input vector. shape: (num_classes, )
        ub (torch.Tensor): upper bound (constraint) vector. shape: (num_classes, )
        eq_cond (torch.Tensor): equality condition boolean vector. If eq_cond[i] = True,
            output[i] = ub[i]. shape: (num_classes, )
    Returns:
        y (torch.Tensor): output probability vector. shape: (num_classes, )
            Satisfying the constraints y[i] = ub[i] if eq_cond[i]
            else y[i] <= ub[i] for all i, and torch.sum(y, dim=1) = 1.

    """
    if torch.any(ub < 0):
        raise ValueError("ub has negative element/elements.")

    if torch.any(torch.sum(ub * eq_cond) > 1.0):
        raise ValueError("sum of ub[eq_cond] is greater than 1.")

    if torch.any(torch.sum(ub) < 1.0):
        raise ValueError("sum of ub is less than 1.")

    # sorting O(n \log n)
    _, indices = torch.sort(
        torch.where(eq_cond, -torch.inf, torch.log(ub) - x), descending=False
    )
    x = x[indices]
    ub = ub[indices]
    eq_cond = eq_cond[indices]

    # find V_u O(n)
    log_r = torch.flip(torch.logcumsumexp(torch.flip(x, dims=(0,)), dim=0), dims=(0,))
    s = 1.0 - (torch.cumsum(ub, dim=0) - ub)
    is_in_V_u = eq_cond | ((s - ub > 0) & (x - log_r + torch.log(s) > torch.log(ub)))
    # compute outputs O(n)
    max_x_not_in_V_u = torch.max(torch.where(is_in_V_u, -torch.inf, x))
    exp_x = torch.where(~is_in_V_u, torch.exp(x - max_x_not_in_V_u), 0.0)
    s = 1 - torch.sum(torch.where(is_in_V_u, ub, 0.0))
    r = torch.sum(exp_x)
    y = torch.where(is_in_V_u, ub, exp_x * s / r)
    # undo sorting O(n \log n)
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]


def ubsoftmax_batch_cond(
    x: torch.Tensor, ub: torch.Tensor, eq_cond: torch.Tensor
) -> torch.Tensor:
    """Conditional Upper-Bounded Softmax function for batch.

    Args:
        x (torch.Tensor): input matrix. shape: (batch_size, num_classes)
        ub (torch.Tensor): upper bound (constraint) matrix.
            shape: (batch_size, num_classes)
        eq_cond (torch.Tensor): equality condition boolean matrix.
            shape: (batch_size, num_classes)
            If eq_cond[i, j] = True, output[i, j] = ub[i, j].

    Returns:
        y (torch.Tensor): output probability matrix. shape: (batch_size, num_classes)
            Satisfying the constraints y[i, j] = ub[i, j] if eq_cond[i, j]
            else y[i, j] <= ub[i, j] for all i, j, and
            torch.sum(y, dim=1) = all-ones vector.

    """
    if torch.any(ub < 0):
        raise ValueError("ub has negative element/elements.")

    if torch.any(torch.sum(ub * eq_cond, dim=1) > 1.0):
        raise ValueError("ub[eq_cond] has row/rows whose sum is/are greater than 1.")

    if torch.any(torch.sum(ub, dim=1) < 1.0):
        raise ValueError("ub has row/rows whose sum is/are less than 1.")

    # sorting O(n \log n)
    _, indices = torch.sort(
        torch.where(eq_cond, -torch.inf, torch.log(ub) - x), descending=False, dim=1
    )
    x = torch.take_along_dim(x, indices, dim=1)
    ub = torch.take_along_dim(ub, indices, dim=1)
    eq_cond = torch.take_along_dim(eq_cond, indices, dim=1)

    # find V_u O(n)
    log_r = torch.flip(torch.logcumsumexp(torch.flip(x, dims=(1,)), dim=1), dims=(1,))
    s = 1.0 - (torch.cumsum(ub, dim=1) - ub)
    is_in_V_u = eq_cond | ((s - ub > 0) & (x - log_r + torch.log(s) > torch.log(ub)))
    # compute outputs O(n)
    max_x_not_in_V_u, _ = torch.max(
        torch.where(~is_in_V_u, x, -torch.inf), dim=1, keepdim=True
    )
    exp_x = torch.where(~is_in_V_u, torch.exp(x - max_x_not_in_V_u), 0.0)
    s = 1 - torch.sum(torch.where(is_in_V_u, ub, 0.0), dim=1, keepdim=True)
    r = torch.sum(exp_x, dim=1, keepdim=True)
    y = torch.where(is_in_V_u, ub, exp_x * s / r)

    # undo sorting O(n \log n)
    _, inv_indices = torch.sort(indices, descending=False, dim=1)
    return torch.take_along_dim(y, inv_indices, dim=1)
