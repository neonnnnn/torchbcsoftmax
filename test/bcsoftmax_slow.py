# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import math

import torch

from .ubsoftmax_slow import ubsoftmax_vector_cond_naive, ubsoftmax_vector_naive


def bcsoftmax_vector_loglinear(x, lb, ub):
    num_classes = len(x)
    x = x - torch.max(x)
    _, indices = torch.sort(lb / torch.exp(x), descending=True)
    x = x[indices]
    lb = lb[indices]
    ub = ub[indices]
    y = ubsoftmax_vector_naive(x, ub)
    L = 0
    R = num_classes - 1
    while L < R:
        rho = math.floor((L + R) / 2)
        cond = torch.zeros(num_classes + 1, dtype=bool)
        cond[1 : rho + 1] = True
        y = ubsoftmax_vector_cond_naive(
            x,
            torch.where(cond[1:], lb, ub),
            cond[1:],
        )
        if torch.all(y >= lb):
            R = rho
        else:
            L = rho + 1
    rho = math.floor((L + R) / 2)
    cond = torch.zeros(num_classes + 1, dtype=bool)
    cond[1 : rho + 1] = True
    y = ubsoftmax_vector_cond_naive(
        x,
        torch.where(cond[1:], lb, ub),
        cond[1:],
    )
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]


def bcsoftmax_vector_naive(x, lb, ub):
    num_classes = len(x)
    x = x - torch.max(x)
    _, indices = torch.sort(lb / torch.exp(x), descending=True)
    x = x[indices]
    lb = lb[indices]
    ub = ub[indices]
    y = ubsoftmax_vector_naive(x, ub)
    cond = torch.zeros_like(x, dtype=bool)
    if not torch.all(y >= lb):
        for i in range(num_classes - 1):
            cond[i] = True
            y = ubsoftmax_vector_cond_naive(
                x,
                torch.where(cond, lb, ub),
                cond,
            )
            if torch.all(y >= lb):
                break
    _, inv_indices = torch.sort(indices, descending=False)
    return y[inv_indices]
