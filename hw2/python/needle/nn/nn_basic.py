"""The module."""

import math
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

# from needle.ops.ops_logarithmic import *


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.kaiming_uniform(self.in_features, self.out_features)
        if bias:
            self.bias = ops.ops_mathematic.transpose(
                init.kaiming_uniform(self.out_features, 1)
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        h = ops.ops_mathematic.matmul(X, self.weight)
        if self.bias is not None:
            h = h + ops.ops_mathematic.broadcast_to(self.bias, h.shape)
        return h


class Flatten(Module):
    def forward(self, X):
        shape = X.shape
        s = math.prod(shape[1:])
        return ops.ops_mathematic.reshape(X, (shape[0], s))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.ops_mathematic.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        dim = len(logits.shape) - 1
        # 需要用tensor操作保留计算图
        # require grad = False
        y_onehot = init.one_hot(logits.shape[dim], y, device=logits.device)
        zy = ops.ops_mathematic.summation(y_onehot * logits, axes=(dim,))
        loss = ops.ops_logarithmic.logsumexp(logits, axes=(dim,)) - zy
        # average through batch
        if dim == 1:
            size = logits.shape[0]
        else:
            size = math.prod(logits.shape[:dim])
        return ops.ops_mathematic.summation(loss) / size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.e = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.var = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.dim
        if not self.training:
            
        axes = list(range(2, len(x.shape)))
        axes.insert(0, 0)
        size = math.prod(x.shape[2::])
        size *= x.shape[0]
        e = ops.ops_mathematic.summation(x, axes=axes) / size


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        shape_fold = x.shape[1::]
        size = math.prod(shape_fold)
        axes = tuple(range(1, len(x.shape)))
        e = ops.ops_mathematic.summation(x, axes=axes) / size
        dim = [1] * len(x.shape)
        dim[0] = x.shape[0]
        avg_B = ops.ops_mathematic.broadcast_to(e.reshape(dim), x.shape)
        var = ops.ops_mathematic.summation((x - avg_B) ** 2, axes=axes) / size
        std = (var + self.eps) ** 0.5
        weight = ops.ops_mathematic.broadcast_to(self.weight, x.shape)
        bias = ops.ops_mathematic.broadcast_to(self.bias, x.shape)
        x = (x - avg_B) / ops.ops_mathematic.broadcast_to(
            ops.ops_mathematic.reshape(std, dim), x.shape
        ) * weight + bias
        return x


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
