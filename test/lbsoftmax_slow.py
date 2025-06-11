# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import torch


def lbsoftmax_vector_naive(x, lb):
    """A naive implementation of lbsoftmax_vector for testing."""
    x = x - torch.max(x, dim=0)[0]  # normalization to avoid numerical errors
    exp_x = torch.exp(x)
    y = exp_x / torch.sum(exp_x)
    is_in_V_l = torch.zeros_like(x, dtype=torch.bool)
    for _ in range(len(x)):
        is_in_V_l |= y < lb
        s = 1 - torch.sum(lb[is_in_V_l])
        r = torch.sum(exp_x[~is_in_V_l])
        y = torch.where(is_in_V_l, lb, s * exp_x / r)
    return y
