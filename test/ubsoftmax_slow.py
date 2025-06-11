# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import torch
import numpy as np


def ubsoftmax_vector_linear(x_, ub_):
    K = len(x_)
    x = x_.detach().numpy()
    # x -= np.max(x)
    b = ub_.detach().numpy()
    x = np.append(0, x)
    exp_x = np.exp(x)
    exp_x[0] = 0
    b = np.append(0, b)
    t = b / np.exp(x)
    t[0] = -np.inf
    s = 1
    r = np.sum(exp_x[1:])
    rho = 0
    i_prime = np.argmax(t)
    C = list(range(K + 1))
    C.remove(i_prime)
    while C:
        ii = np.random.randint(len(C))
        i = C.pop(ii)
        L = [k for k in C if t[k] <= t[i]]
        L.append(i)
        R = [k for k in C if t[k] > t[i]]
        s_prime = s - np.sum(b[L])
        r_prime = r - np.sum(exp_x[L])
        j = i_prime
        for jj in R:
            if t[j] > t[jj]:
                j = jj
        if s_prime <= 0:
            C = [k for k in C if t[k] < t[i]]
            i_prime = i
        else:
            if s_prime <= r_prime * t[j]:  # 1/z <= t[i]
                C = [k for k in C if t[k] < t[i]]
                rho = i
                i_prime = i
            else:
                C = R
                s = s_prime
                r = r_prime
    ret = np.zeros(K + 1)
    s = 1.0
    r = 0.0
    for k in range(1, K + 1):
        if t[k] <= t[rho]:
            ret[k] = b[k]
            s -= b[k]
        else:
            ret[k] = np.exp(x[k])
            r += ret[k]
    for k in range(1, K + 1):
        if t[k] > t[rho]:
            ret[k] *= s
            ret[k] /= r
    return torch.tensor(ret[1:], dtype=torch.float64)


def ubsoftmax_vector_naive(x, ub):
    """A naive implementation of ubsoftmax_vector for testing."""
    x = x - torch.max(x, dim=0)[0]  # normalization to avoid numerical errors
    exp_x = torch.exp(x)
    y = exp_x / torch.sum(exp_x)
    is_in_V_u = torch.zeros_like(x, dtype=torch.bool)
    for _ in range(len(x)):
        is_in_V_u |= y > ub
        s = 1 - torch.sum(ub[is_in_V_u])
        r = torch.sum(exp_x[~is_in_V_u])
        y = torch.where(is_in_V_u, ub, s * exp_x / r)
    return y


def ubsoftmax_vector_cond_naive(x, ub, eq_cond):
    """A naive implementation of ubsoftmax_vector for testing."""
    s = 1 - torch.sum(ub * eq_cond)
    if torch.any(eq_cond):
        y = ub.clone().detach()
        y[~eq_cond] = s * ubsoftmax_vector_naive(x[~eq_cond], ub[~eq_cond] / s)
        return y
    else:
        return ubsoftmax_vector_naive(x, ub)
