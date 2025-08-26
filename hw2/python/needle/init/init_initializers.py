import math


from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * (6 / (fan_in + fan_out)) ** 0.5
    return rand(fan_in, fan_out, low=-a, high=a)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * (2 / (fan_in + fan_out)) ** 0.5
    return randn(fan_in, fan_out, mean=0, std=a)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    a = math.sqrt(2) * (3 / (fan_in)) ** 0.5
    return rand(fan_in, fan_out, low=-a, high=a)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    a = math.sqrt(2) * fan_in**-0.5
    return randn(fan_in, fan_out, mean=0, std=a)
