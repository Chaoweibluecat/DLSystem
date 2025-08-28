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
        self.weight = Parameter(
            init.kaiming_uniform(
                self.in_features,
                self.out_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    self.out_features, 1, device=device, dtype=dtype, requires_grad=True
                ).transpose()
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        h = X.matmul(self.weight)
        if self.bias is not None:
            h = h + self.bias.broadcast_to(h.shape)
        return h


class Flatten(Module):
    def forward(self, X):
        shape = X.shape
        s = math.prod(shape[1:])
        # return ops.ops_mathematic.reshape(X, (shape[0], s))
        return X.reshape((shape[0], s))


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
        log_softmax = ops.ops_logarithmic.logsoftmax(logits)
        ce = log_softmax * y_onehot
        loss = -ops.ops_mathematic.summation(ce, axes=(dim,))
        # average through batch
        if dim == 1:
            size = logits.shape[0]
        else:
            size = math.prod(logits.shape[:dim])
        return ops.ops_mathematic.summation(loss) / size

    # def forward(self, logits: Tensor, y: Tensor):
    #     dim = len(logits.shape) - 1
    #     # 需要用tensor操作保留计算图
    #     # require grad = False
    #     y_onehot = init.one_hot(logits.shape[dim], y, device=logits.device)
    #     zy = ops.ops_mathematic.summation(y_onehot * logits, axes=(dim,))
    #     loss = ops.ops_logarithmic.logsumexp(logits, axes=(dim,)) - zy
    #     # average through batch
    #     if dim == 1:
    #         size = logits.shape[0]
    #     else:
    #         size = math.prod(logits.shape[:dim])
    #     return ops.ops_mathematic.summation(loss) / size


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.running_mean = init.zeros(
            dim, device=device, dtype=dtype, requires_grad=False
        )
        self.running_var = init.ones(
            dim, device=device, dtype=dtype, requires_grad=False
        )
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.dim
        weight_shape = [1] * len(x.shape)
        weight_shape[1] = self.dim
        if not self.training:
            e = self.running_mean.reshape(weight_shape).broadcast_to(x.shape)
            var = self.running_var.reshape(weight_shape).broadcast_to(x.shape)
            std = (var + self.eps) ** 0.5
            weight = self.weight.reshape(weight_shape).broadcast_to(x.shape)
            bias = self.bias.reshape(weight_shape).broadcast_to(x.shape)
            x = (x - e) / std * weight + bias
            return x
        else:
            axes = list(range(2, len(x.shape)))
            axes.insert(0, 0)
            axes = tuple(axes)
            size = math.prod(x.shape[2::])
            size *= x.shape[0]
            e = x.sum(axes=axes) / size
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * e
            dim = [1] * len(x.shape)
            dim[1] = x.shape[1]
            avg_B = e.reshape(dim).broadcast_to(x.shape)
            var = ((x - avg_B) ** 2).sum(axes=axes) / size
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
            std = (var + self.eps) ** 0.5
            weight = self.weight.reshape(dim).broadcast_to(x.shape)
            bias = self.bias.reshape(dim).broadcast_to(x.shape)
            x = ((x - avg_B) / std.reshape(dim).broadcast_to(x.shape)) * weight + bias
            return x


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
        if not self.training or self.p == 0:
            return x

        keep_prob = 1.0 - self.p
        # 每个批次的dropOut的activation也是随机的
        mask = init.rand(*x.shape, device=x.device)
        # 如果随机数 < 保留概率，则 mask=1
        # 惯例实现p = 0.2时, 保留0-0.8（而不是保留0.2-1)
        mask = ops.ops_mathematic.less_than_scalar(mask, keep_prob)

        return (mask * x) / keep_prob


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
