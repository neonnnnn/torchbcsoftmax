# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import numpy as np
import torch

from bcsoftmax.upper_bounded_softmax import (
    ubsoftmax_vector,
    ubsoftmax_vector_cond,
    ubsoftmax_batch,
    ubsoftmax_batch_cond,
    ubsoftmax,
)

from .ubsoftmax_slow import (
    ubsoftmax_vector_cond_naive,
    ubsoftmax_vector_naive,
    ubsoftmax_vector_linear,
)

rng = np.random.RandomState(1)
batch_size = 128


def test_2d(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    for tau in [1e-1, 1, 10]:
        output_2d = ubsoftmax_batch(X / tau, B)
        output_1d = torch.vstack([ubsoftmax_vector(x, b) for x, b in zip(X / tau, B)])
        torch.testing.assert_close(
            output_2d,
            output_1d,
        )


def test_3d(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes // 2, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes // 2, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, num_classes // 2, 1),
        torch.sum(B, dim=2, keepdim=True) - 1e-6,
    )
    for tau in [1e-1, 1, 10]:
        output_3d = ubsoftmax(X / tau, B, dim=-1)
        output_1d = torch.stack(
            [
                torch.vstack([ubsoftmax_vector(x, b) for x, b in zip(Xmat, bmat)])
                for Xmat, bmat in zip(X / tau, B)
            ]
        )
        torch.testing.assert_close(
            output_3d,
            output_1d,
        )

    output_3d = ubsoftmax(X, B, dim=-1)
    output_3dT = ubsoftmax(X.swapaxes(1, 2), B.swapaxes(1, 2), dim=1)
    torch.testing.assert_close(output_3d, output_3dT.swapaxes(2, 1))
    output_3dT = ubsoftmax(X.swapaxes(0, 2), B.swapaxes(0, 2), dim=0)
    torch.testing.assert_close(output_3d, output_3dT.swapaxes(0, 2))


def test_softmax(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    expected = torch.nn.functional.softmax(X, dim=1)
    for coef in [1.0, 1.5, 2.0, 2.5, 3.0]:
        B = torch.ones_like(X) * coef
        actual = ubsoftmax_batch(X, B)
        torch.testing.assert_close(actual, expected)


def test_satisfying_constraints(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    for inv_coef in range(1, num_classes - 1):
        B = torch.ones_like(X) / inv_coef
        output = ubsoftmax_batch(X, B)
        assert torch.all(output <= B), "Box Constraint Error"
        row_sum = torch.sum(output, dim=1)
        torch.testing.assert_close(row_sum, torch.ones_like(row_sum))


def test_satisfying_constraints_with_zero(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    for inv_coef in range(1, num_classes - (num_classes // 4) - 1):
        B = torch.ones_like(X) / inv_coef
        B[:, 0 : num_classes // 4] = 0.0
        output = ubsoftmax_batch(X, B)
        assert torch.all(output <= B), "Box Constraint Error"
        row_sum = torch.sum(output, dim=1)
        torch.testing.assert_close(row_sum, torch.ones_like(row_sum))


def test_grad(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes))).requires_grad_()
    B = torch.tensor(rng.rand(batch_size, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    for tau in [1e-1, 1, 10]:
        # 1d
        for i in range(5):
            torch.autograd.gradcheck(
                ubsoftmax_vector, (X[i] / tau, B[i].requires_grad_())
            )
        # 2d
        torch.autograd.gradcheck(ubsoftmax_batch, (X / tau, B.requires_grad_()))
        torch.autograd.gradcheck(ubsoftmax, (X / tau, B))
        torch.autograd.gradcheck(
            ubsoftmax, (X.transpose(1, 0) / tau, B.transpose(1, 0), 0)
        )


def test_same_as_slow(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    for tau in [1e-1, 1, 10]:
        output = ubsoftmax_batch(X / tau, B)
        output_naive = torch.vstack(
            [ubsoftmax_vector_naive(x, b) for x, b in zip(X / tau, B)]
        )
        torch.testing.assert_close(
            output,
            output_naive,
        )
    for tau in [5e-1, 1.0, 5.0]:  # ubsoftmax_vector_linear is numerically unstable
        output = ubsoftmax_batch(X / tau, B)
        output_naive = torch.vstack(
            [ubsoftmax_vector_linear(x, b) for x, b in zip(X / tau, B)]
        )
        torch.testing.assert_close(
            output,
            output_naive,
        )


def test_cond_2d(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1)) * 0.5
    Cond = torch.tensor(rng.rand(batch_size, num_classes) > 0.5)
    B = torch.where(Cond, A, B)
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    for tau in [1e-1, 1, 10]:
        output_2d = ubsoftmax_batch_cond(X / tau, B, Cond)
        output_1d = torch.vstack(
            [ubsoftmax_vector_cond(x, b, c) for x, b, c in zip(X / tau, B, Cond)]
        )
        torch.testing.assert_close(
            output_2d,
            output_1d,
        )


def test_cond_softmax(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    expected = torch.nn.functional.softmax(X, dim=1)
    # no equality constraints and no meaningful upper bound constraints
    actual = ubsoftmax_batch_cond(
        X, expected + 1e-2, torch.zeros_like(expected, dtype=torch.bool)
    )
    torch.testing.assert_close(actual, expected)

    # no meaningful eqaulity and upper bound constraints
    cond = torch.zeros_like(expected, dtype=torch.bool)
    for i in range(num_classes - 1):
        cond[:, i] = True
        actual = ubsoftmax_batch_cond(X, expected + 1e-7 * (~cond), cond)
        torch.testing.assert_close(actual, expected)


def test_cond_satisfying_constraints(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1))
    Cond = torch.tensor(rng.rand(batch_size, num_classes) > 0.5)
    B = torch.where(Cond, A, B)
    output = ubsoftmax_batch_cond(X, B, Cond)
    assert torch.all(output <= B)
    assert torch.all(output * Cond == B * Cond)


def test_cond_same_as_slow(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1))

    Cond = torch.tensor(rng.rand(batch_size, num_classes) > 0.5)
    while torch.any(torch.all(Cond, dim=1)):
        Cond = torch.tensor(rng.rand(batch_size, num_classes) > 0.5)
    B = torch.where(Cond, A, B)
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    for tau in [1e-1, 1, 10]:
        output = ubsoftmax_batch_cond(X / tau, B, Cond)
        output_naive = torch.vstack(
            [ubsoftmax_vector_cond_naive(x, b, c) for x, b, c in zip(X / tau, B, Cond)]
        )
        torch.testing.assert_close(
            output,
            output_naive,
        )
