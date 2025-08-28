"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for i, param in enumerate(self.params):
            param: ndl.Tensor = param
            # 惯例做法,把weight decay加到梯度上, 并累计到momentum
            grad = (param.grad + self.weight_decay * param).detach()
            if not i in self.u:
                self.u[i] = ((1 - self.momentum) * grad).detach()
            else:
                self.u[i] = (
                    self.momentum * self.u[i] + (1 - self.momentum) * grad
                ).detach()
            p_new = param - self.lr * self.u[i]
            param.data = p_new

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        self.u = {}
        self.tt = []

    def step(self):
        # self.tt.append(np.array(ndl.autograd.TENSOR_COUNTER))
        for i, param in enumerate(self.params):
            grad = (param.grad + self.weight_decay * param).detach()
            if not i in self.u:
                self.u[i] = ((1 - self.beta1) * grad).detach()
                self.v[i] = ((1 - self.beta2) * grad**2).detach()
            else:
                self.u[i] = (self.beta1 * self.u[i] + (1 - self.beta1) * grad).detach()
                self.v[i] = (
                    self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
                ).detach()
            # 校正后到值仅用于本次梯度下降计算,重要！,如果校正后进u会直接爆炸
            u_hat = self.u[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p_new = param - self.lr * u_hat / (v_hat**0.5 + self.eps)
            param.data = p_new
