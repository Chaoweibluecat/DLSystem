import numpy as np
import pytest
import mugrade
import time
import needle as ndl
from needle import backend_ndarray as nd
import torch  # 唯一增加的 import

# --- 您的原始代码，一字不改 ---
n = 4096
m = 4096
p = 4096
device = nd.cuda()
_A = np.random.randn(m, n).astype("float32")
_B = np.random.randn(n, p).astype("float32")
A = nd.array(_A, device=device)
B = nd.array(_B, device=device)

# 预热
C = A @ B
C.numpy()
C = A @ B
C.numpy()

# --- 您的原始计时逻辑，现在是正确的，因为您的C++是同步的 ---
start = time.perf_counter()
C = A @ B
start1 = time.perf_counter()
print("shared memory:")
print(start1 - start)

C_naive = A.matmul_naive(B)
print("naive:")
print(time.perf_counter() - start1)

start1 = time.perf_counter()
_C = _A @ _B
start2 = time.perf_counter()
print("numpy cpu")
print(start2 - start1)

# --- 【新增】您要求的 PyTorch GPU 版本 ---
A_torch = torch.from_numpy(_A).cuda()
B_torch = torch.from_numpy(_B).cuda()
# PyTorch 的操作是异步的，所以它需要预热和同步
_ = A_torch @ B_torch
torch.cuda.synchronize()

torch_start = time.perf_counter()
_ = A_torch @ B_torch
torch.cuda.synchronize()  # PyTorch 必须加同步才能准确计时
torch_end = time.perf_counter()
print("torch gpu:")
print(torch_end - torch_start)


# --- 您的原始正确性检查 ---
np.testing.assert_allclose(C.numpy(), _C, rtol=1e-5, atol=1e-5)
print("success")
