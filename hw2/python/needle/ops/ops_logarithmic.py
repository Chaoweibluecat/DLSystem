from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        shape = list(Z.shape)  # Z.shape
        shape[1] = 1
        self.lse = logsumexp(Tensor(Z), axes=(1,))
        return Z - numpy.reshape(self.lse.numpy(), tuple(shape))

    def gradient(self, out_grad, node):
        z = node.inputs[0]
        lse_out_grad = summation(out_grad, axes=(1,))
        lse_grad = self.lse.op.gradient(lse_out_grad, z)
        z_grad = out_grad
        return z_grad - lse_grad


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        M = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_ = Z - M
        E = array_api.exp(Z_)
        self.exp = Tensor(E)  # save for backwardM
        S = array_api.sum(E, axis=self.axes, keepdims=True)
        self.sum = Tensor(S)
        return array_api.log(array_api.squeeze(S, axis=self.axes)) + array_api.squeeze(
            M, axis=self.axes
        )

    def gradient(self, out_grad, node):
        s = list(node.inputs[0].shape)
        if self.axes is None:
            s = [1] * len(s)
        else:
            for axis in self.axes:
                s[axis] = 1
        G = broadcast_to(reshape(out_grad, s), self.exp.shape)
        return G * self.exp / self.sum


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
