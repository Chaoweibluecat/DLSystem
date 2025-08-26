"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power(a, b)

    # 输入都是相同shape的tensor,全程逐元素操作
    def gradient(self, out_grad, node):
        base: Tensor = node.inputs[0]
        exponent: Tensor = node.inputs[1]
        da = base ** (exponent - 1) * exponent
        db = base**exponent * log(base)
        return out_grad * da, out_grad * db


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    # input_size : batch * feature
    # grad_size : batch * feature in && out_grad_size : batch * feature
    # 全都是numpy的逐元素运算
    def gradient(self, out_grad, node):
        a: Tensor = node.inputs[0]
        da = a ** (self.scalar - 1) * self.scalar
        return out_grad * da


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        lhs: Tensor = node.inputs[0]
        rhs: Tensor = node.inputs[1]
        da = rhs**-1
        db = rhs**-2 * (-lhs)
        return out_grad * da, out_grad * db


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        # operator: Tensor = node.inputs[0]
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray):
        dim = list(range(len(a.shape)))
        if self.axes is not None:
            dim[self.axes[0]], dim[self.axes[1]] = dim[self.axes[1]], dim[self.axes[0]]
        else:
            dim[-1], dim[-2] = dim[-2], dim[-1]
        return array_api.transpose(a, axes=dim)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    # todo takeaway boardcasting
    def gradient(self, out_grad, node):
        input_shape = list(node.inputs[0].shape)
        out_dim = len(self.shape)
        while len(input_shape) < out_dim:
            input_shape = [1] + input_shape
        axis = []
        for i in range(out_dim):
            if self.shape[i] != input_shape[i]:
                axis.append(i)
        sum = summation(out_grad, axes=tuple(axis))
        return sum.reshape(node.inputs[0].shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    # keep dim,
    def compute(self, a):
        if self.axes is None:
            return array_api.sum(a)
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        # 支持sum降维, 也可以compute时直接keepDim
        if self.axes is None:
            output_shape = [1] * len(node.inputs[0].shape)
        else:
            output_shape = list(node.inputs[0].shape)
            for dim in self.axes:
                output_shape[dim] = 1
        return broadcast_to(
            reshape(out_grad, tuple(output_shape)), node.inputs[0].shape
        )


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        # 2 * 2 * 3 @ 3 * 4 -> 2 * 2 * 4
        lhs: Tensor = node.inputs[0]
        rhs: Tensor = node.inputs[1]
        # 2 * 2 * 4 @ 4 * 3 -> 2 * 2 * 3
        da = matmul(out_grad, transpose(rhs))
        # 2 * 3 * 2 @ 2 * 2 * 4 -> 2 * 3 * 4 => 1 * 3 * 4
        db = matmul(transpose(lhs), out_grad)
        # 对于被广播的矩阵来说, 需要sum掉多余的维度
        db = summation(db, axes=tuple(range(len(db.shape) - len(rhs.shape))))
        da = summation(da, axes=tuple(range(len(da.shape) - len(lhs.shape))))
        return da, db


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        out_grad: Tensor = out_grad
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


# 在您的 ops.py 文件中，添加一个新的 Op


class GreaterThanScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        # 前向传播：执行比较，并将布尔结果转换为浮点数 (True->1.0, False->0.0)
        return (a > self.scalar).astype(a.dtype)

    def gradient(self, out_grad, node):
        return array_api.zeros_like(node.inputs[0])


def greater_than_scalar(a, scalar):
    return GreaterThanScalar(scalar)(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return out_grad * greater_than_scalar(node.inputs[0], 0)


def relu(a):
    return ReLU()(a)
