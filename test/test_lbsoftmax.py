# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import numpy as np
import torch

from bcsoftmax.lower_bounded_softmax import lbsoftmax_vector, lbsoftmax_batch, lbsoftmax

from .lbsoftmax_slow import lbsoftmax_vector_naive

rng = np.random.RandomState(1)
batch_size = 128


def test_2d(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1))
    for tau in [1e-1, 1, 10]:
        output_2d = lbsoftmax_batch(X / tau, A)
        output_1d = torch.vstack([lbsoftmax_vector(x, a) for x, a in zip(X / tau, A)])
        torch.testing.assert_close(
            output_2d,
            output_1d,
        )


def test_3d(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes // 2, num_classes)))
    A = torch.tensor(rng.rand(batch_size, num_classes // 2, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=2)
    A *= torch.tensor(rng.rand(batch_size, num_classes // 2, 1)) * 0.5
    for tau in [1e-1, 1, 10]:
        output_3d = lbsoftmax(X / tau, A, dim=-1)
        output_1d = torch.stack(
            [
                torch.vstack([lbsoftmax_vector(x, a) for x, a in zip(Xmat, amat)])
                for Xmat, amat in zip(X / tau, A)
            ]
        )
        torch.testing.assert_close(
            output_3d,
            output_1d,
        )

    output_3d = lbsoftmax(X, A, dim=-1)
    output_3dT = lbsoftmax(X.swapaxes(1, 2), A.swapaxes(1, 2), dim=1)
    torch.testing.assert_close(output_3d, output_3dT.swapaxes(2, 1))
    output_3dT = lbsoftmax(X.swapaxes(0, 2), A.swapaxes(0, 2), dim=0)
    torch.testing.assert_close(output_3d, output_3dT.swapaxes(0, 2))


def test_softmax(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    expected = torch.nn.functional.softmax(X, dim=1)
    A = torch.zeros_like(X)
    actual = lbsoftmax_batch(X, A)
    torch.testing.assert_close(actual, expected)


def test_satisfying_constraints(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    for inv_coef in range(num_classes + 1, 4 * num_classes, 4):
        A = torch.ones_like(X) / inv_coef
        output = lbsoftmax_batch(X, A)
        assert torch.all(output >= A), "Constraint Error"
        row_sum = torch.sum(output, dim=1)
        torch.testing.assert_close(row_sum, torch.ones_like(row_sum))


def test_satisfying_constraints_random(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1))
    output = lbsoftmax_batch(X, A)
    assert torch.all(output >= A), "Constraint Error"


def test_grad(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes))).requires_grad_()
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1)) * 0.5
    A += 1e-6
    for tau in [1e-1, 1, 10]:
        # 1d
        for i in range(5):
            torch.autograd.gradcheck(lbsoftmax_vector, (X[i] / tau, A[i]))
        # 2d
        torch.autograd.gradcheck(lbsoftmax_batch, (X / tau, A))
        torch.autograd.gradcheck(lbsoftmax, (X / tau, A))
        torch.autograd.gradcheck(
            lbsoftmax, (X.transpose(1, 0) / tau, A.transpose(1, 0), 0)
        )


def test_same_as_slow(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1))
    for tau in [1e-1, 1, 10]:
        output = lbsoftmax_batch(X / tau, A)
        output_naive = torch.vstack(
            [lbsoftmax_vector_naive(x, a) for x, a in zip(X / tau, A)]
        )
        torch.testing.assert_close(
            output,
            output_naive,
        )
