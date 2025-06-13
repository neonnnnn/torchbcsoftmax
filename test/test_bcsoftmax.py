# encoding: utf-8
# Author: Kyohei Atarashi
# License: MIT

import numpy as np
import torch

from bcsoftmax.box_constrained_softmax import (
    bcsoftmax,
    bcsoftmax_batch,
    bcsoftmax_vector,
)

from .bcsoftmax_slow import bcsoftmax_vector_loglinear, bcsoftmax_vector_naive

rng = np.random.RandomState(1)
batch_size = 64


def test_2d(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1)) * 0.5
    B = torch.maximum(A, B) + 1e-6
    for tau in [1e-1, 1, 10]:
        output_2d = bcsoftmax(X / tau, A, B, dim=1)
        output_1d = torch.vstack(
            [bcsoftmax_vector(x, a, b) for x, a, b in zip(X / tau, A, B)]
        )
        torch.testing.assert_close(
            output_2d,
            output_1d,
        )
    output_2d = bcsoftmax(X, A, B, dim=1)
    output_2dT = bcsoftmax(
        X.transpose(1, 0),
        A.transpose(1, 0),
        B.transpose(1, 0),
        dim=0,
    ).transpose(1, 0)
    torch.testing.assert_close(output_2d, output_2dT)


def test_3d(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes // 2, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes // 2, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, num_classes // 2, 1),
        torch.sum(B, dim=2, keepdim=True) - 1e-6,
    )
    A = torch.tensor(rng.rand(batch_size, num_classes // 2, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=2)
    A *= torch.tensor(rng.rand(batch_size, num_classes // 2, 1)) * 0.5
    B = torch.maximum(A, B) + 1e-6
    for tau in [1e-1, 1, 10]:
        output_3d = bcsoftmax(X / tau, A, B, dim=-1)
        output_1d = torch.stack(
            [
                torch.vstack(
                    [bcsoftmax_vector(x, a, b) for x, a, b in zip(Xmat, amat, bmat)]
                )
                for Xmat, amat, bmat in zip(X / tau, A, B)
            ]
        )
        torch.testing.assert_close(
            output_3d,
            output_1d,
        )

    output_3d = bcsoftmax(X, A, B, dim=-1)
    output_3dT = bcsoftmax(X.swapaxes(1, 2), A.swapaxes(1, 2), B.swapaxes(1, 2), dim=1)
    torch.testing.assert_close(output_3d, output_3dT.swapaxes(2, 1))
    output_3dT = bcsoftmax(X.swapaxes(0, 2), A.swapaxes(0, 2), B.swapaxes(0, 2), dim=0)
    torch.testing.assert_close(output_3d, output_3dT.swapaxes(0, 2))


def test_softmax(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))

    expected = torch.nn.functional.softmax(X, dim=1)
    actual = bcsoftmax_batch(
        X,
        torch.zeros_like(expected),
        torch.ones_like(expected),
    )
    torch.testing.assert_close(actual, expected)


def test_satisfying_constraints_random(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1)) * 0.5
    B = torch.maximum(A, B) + 1e-6

    for tau in [1e-1, 1, 10]:
        output = bcsoftmax_batch(X / tau, A, B)
        assert torch.all(output <= B)
        assert torch.all(output >= A)
        row_sum = torch.sum(output, dim=1)
        torch.testing.assert_close(row_sum, torch.ones_like(row_sum))


def test_grad(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes))).requires_grad_()
    B = torch.tensor(rng.rand(batch_size, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1)) * 0.5
    A += 1e-6
    B = torch.maximum(A, B) + 1e-6
    for tau in [1e-1, 1, 10]:
        # 1d
        for i in range(5):
            torch.autograd.gradcheck(
                bcsoftmax_vector,
                (X[i] / tau, A[i].requires_grad_(), B[i].requires_grad_()),
            )
        # 2d
        torch.autograd.gradcheck(
            bcsoftmax_batch, (X / tau, A.requires_grad_(), B.requires_grad_())
        )
        torch.autograd.gradcheck(
            bcsoftmax, (X / tau, A.requires_grad_(), B.requires_grad_())
        )
        torch.autograd.gradcheck(
            bcsoftmax,
            (
                X.transpose(1, 0) / tau,
                A.transpose(1, 0).requires_grad_(),
                B.transpose(1, 0).requires_grad_(),
                0,
            ),
        )


def test_same_as_slow(num_classes=32):
    X = torch.tensor(rng.normal(0, 3.0, (batch_size, num_classes)))
    B = torch.tensor(rng.rand(batch_size, num_classes))
    B /= torch.minimum(
        torch.ones(batch_size, 1), torch.sum(B, dim=1, keepdim=True) - 1e-6
    )
    A = torch.tensor(rng.rand(batch_size, num_classes))
    A = torch.nn.functional.normalize(A, p=1, dim=1)
    A *= torch.tensor(rng.rand(batch_size, 1)) * 0.5
    B = torch.maximum(A, B) + 1e-6

    for tau in [1e-1, 1, 10]:
        output = bcsoftmax_batch(X / tau, A, B)
        output_naive = torch.vstack(
            [bcsoftmax_vector_naive(x, a, b) for x, a, b in zip(X / tau, A, B)]
        )
        torch.testing.assert_close(
            output,
            output_naive,
        )
        output_loglinear = torch.vstack(
            [bcsoftmax_vector_loglinear(x, a, b) for x, a, b in zip(X / tau, A, B)]
        )
        torch.testing.assert_close(
            output,
            output_loglinear,
        )
