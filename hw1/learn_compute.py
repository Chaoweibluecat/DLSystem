import sys

sys.path.append("python/")
import needle as ndl

x1 = ndl.Tensor([1.0, 2.0, 3.0], dtype="float32")
x2 = ndl.Tensor([4.0, 5.0, 6.0], dtype="float32")
y = x1 + x2
print(y.cached_data)
# LAZY
ndl.autograd.LAZY_MODE = True
y = x1 + x2
print(y.cached_data)
y = y + 1
print(y.cached_data)
# trigger 递归求解
# @property
# def data(self):
#     return self.detach()
# pytorch 默认使用ego mode
# Lazy mode在计算量巨大,构建计算图成本较小的情况下优势较大，我们可以建立完图后一次性计算更大的数据(方便优化)
# see Pytorch TPU backend XOA
y.data
print(y.cached_data)
