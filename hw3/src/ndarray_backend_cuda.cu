#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle
{
  namespace cuda
  {

#define BASE_THREAD_NUM 256

#define TWO_DIM_THREAD_NUM 16

#define STRIDE 16
#define L (STRIDE * TILE)

#define TILE 4
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    struct CudaArray
    {
      CudaArray(const size_t size)
      {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess)
          throw std::runtime_error(cudaGetErrorString(err));
        this->size = size;
      }
      ~CudaArray() { cudaFree(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }

      scalar_t *ptr;
      size_t size;
    };

    struct CudaDims
    {
      dim3 block, grid;
    };

    CudaDims CudaOneDim(size_t size)
    {
      /**
       * Utility function to get cuda dimensions for 1D call
       */
      CudaDims dim;
      size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
      dim.block = dim3(BASE_THREAD_NUM, 1, 1);
      dim.grid = dim3(num_blocks, 1, 1);
      return dim;
    }

    CudaDims CudaTwoDim(size_t size_x, size_t size_y) {
      CudaDims dim;
      size_t num_blocks_x = (size_x + TWO_DIM_THREAD_NUM - 1) / TWO_DIM_THREAD_NUM;
      size_t num_blocks_y = (size_y + TWO_DIM_THREAD_NUM - 1) / TWO_DIM_THREAD_NUM;
      dim.block = dim3(TWO_DIM_THREAD_NUM, TWO_DIM_THREAD_NUM, 1);
      dim.grid = dim3(num_blocks_x, num_blocks_y, 1);
      return dim;
    }

#define MAX_VEC_SIZE 8
    struct CudaVec
    {
      uint32_t size;
      int32_t data[MAX_VEC_SIZE];
    };

    CudaVec VecToCuda(const std::vector<int32_t> &x)
    {
      CudaVec shape;
      if (x.size() > MAX_VEC_SIZE)
        throw std::runtime_error("Exceeded CUDA supported max dimesions");
      shape.size = x.size();
      for (size_t i = 0; i < x.size(); i++)
      {
        shape.data[i] = x[i];
      }
      return shape;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Fill call
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void FillKernel(scalar_t *out, scalar_t val, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = val;
    }

    void Fill(CudaArray *out, scalar_t val)
    {
      CudaDims dim = CudaOneDim(out->size);
      FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Compact and setitem cals
    ////////////////////////////////////////////////////////////////////////////////

    // Untility function to convert contiguous index i to memory location from strides

    __global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset)
    {
      /**
       * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
       * non-compact input a, to the corresponding item (at location gid) in the compact array out.
       *
       * Args:
       *   a: CUDA pointer to a array
       *   out: CUDA point to out array
       *   size: size of out array
       *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
       *   strides: vector of strides of out array
       *   offset: offset of out array
       */
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= size)
      {
        return;
      }
      size_t source_address = offset;
      size_t temp = gid;
      for (int i = shape.size - 1; i >= 0; i--)
      {
        int32_t idx = temp % shape.data[i];
        source_address += idx * strides.data[i];
        temp /= shape.data[i];
      }
      out[gid] = a[source_address];
    }

    void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                 std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
       * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
       * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
       * the functions after this, however, you'll need to define these kernels as you see fit to
       * execute the underlying function.
       *
       * Args:
       *   a: non-compact represntation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       */

      // Nothing needs to be added here
      CudaDims dim = CudaOneDim(out->size);
      CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
    }

        __global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= size)
      {
        return;
      }
      size_t dst_address = offset;
      size_t temp = gid;
      for (int i = shape.size - 1; i >= 0; i--)
      {
        int32_t idx = temp % shape.data[i];
        dst_address += idx * strides.data[i];
        temp /= shape.data[i];
      }
      out[dst_address] = a[gid];
    }

    void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                      std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
       * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
       *
       * Args:
       *   a: _compact_ array whose items will be written to out
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *out* array (not a, which has compact strides)
       *   offset: offset of the *out* array (not a, which has zero offset, being compact)
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(a.size);
      // 注意: 一定记得要用compact数组的size, out size不准
      EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                                  VecToCuda(strides), offset);
      /// END SOLUTION
    }



    __global__ void ScalarSetitemKernel(scalar_t val, scalar_t *out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= size)
      {
        return;
      }
      size_t dst_address = offset;
      size_t temp = gid;
      for (int i = shape.size - 1; i >= 0; i--)
      {
        int32_t idx = temp % shape.data[i];
        dst_address += idx * strides.data[i];
        temp /= shape.data[i];
      }
      out[dst_address] = val;
    }


    void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<int32_t> shape,
                       std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items is a (non-compact) array
       *
       * Args:
       *   size: number of elements to write in out array (note that this will note be the same as
       *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
       *         product of items in shape, but covenient to just pass it here.
       *   val: scalar value to write to
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension of out
       *   strides: strides of the out array
       *   offset: offset of the out array
       */
      /// BEGIN SOLUTION
      CudaDims cudaDim = CudaOneDim(size);
      ScalarSetitemKernel<<<cudaDim.grid, cudaDim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
      /// END SOLUTION
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + b[gid];
    }


    void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      /**
       * Add together two CUDA arrays.
       * Args:
       *   a: Input array 'a' to be added
       *   b: Input array 'b' to be added
       *   out: Output array to store the result of 'a + b'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + val;
    }

    void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      /**
       * Add a scalar value to every element of a CUDA array.
       * Args:
       *   a: Input array 'a'
       *   val: Scalar value to be added
       *   out: Output array to store the result of 'a + val'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a',
      // and store the result in array 'out'.
      ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    template <typename Op>
    __global__ void EwiseBinaryKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= size) {
        return;
      }
      Op op;
      out[gid] = op(a[gid], b[gid]);
    }    


    template <typename Op>
    __global__ void UnaryKernel(const scalar_t *a, scalar_t *out, uint32_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= size) {
        return;
      }
      Op op;
      out[gid] = op(a[gid]);
    }     
    
    template <typename Op>
    __global__ void ScalarCalKernel(const scalar_t *a, const scalar_t val, scalar_t *out, uint32_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= size) {
        return;
      }
      Op op;
      out[gid] = op(a[gid], val);
    }     


    struct MulOp
    {
      __device__ scalar_t operator()(scalar_t a, scalar_t b)
      {
        return a * b;
      }
    };

    void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out) {
      CudaDims dim = CudaOneDim(out->size);
      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      EwiseBinaryKernel<MulOp><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out) {
      CudaDims dim = CudaOneDim(out->size);
      ScalarCalKernel<MulOp><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }


    struct DivOp{
          __device__ scalar_t operator()(scalar_t a, scalar_t b)
      {
        return a / b;
      }
    };

   void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out) {
      CudaDims dim = CudaOneDim(out->size);
      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      EwiseBinaryKernel<DivOp><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out) {
      CudaDims dim = CudaOneDim(out->size);
      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      ScalarCalKernel<DivOp><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    struct PowerOp{
          __device__ scalar_t operator()(scalar_t a, scalar_t b)
      {
        return powf(a,b);
      }
    };

    void ScalarPower(const CudaArray &a, const scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      ScalarCalKernel<PowerOp><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    struct EqOp{
          __device__ scalar_t operator()(scalar_t a, scalar_t b)
      {
        return a == b;
      }
    };

    void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseBinaryKernel<EqOp><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    void ScalarEq(const CudaArray &a, const scalar_t val, CudaArray *out) {
      CudaDims dim = CudaOneDim(out->size);
      ScalarCalKernel<EqOp><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    struct GeOp {
           __device__ scalar_t operator()(scalar_t a, scalar_t b)
      {
        return a >= b;
      }
    };

  void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseBinaryKernel<GeOp><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    void ScalarGe(const CudaArray &a, const scalar_t val, CudaArray *out) {
      CudaDims dim = CudaOneDim(out->size);
      ScalarCalKernel<GeOp><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    struct LogOp {
        __device__ scalar_t operator()(scalar_t a)
    {
      return log(a);
    }
    };

      void EwiseLog(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      UnaryKernel<LogOp><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    } 

   struct ExpOp {
        __device__ scalar_t operator()(scalar_t a)
    {
      return expf(a);
    }
    };

    void EwiseExp(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      UnaryKernel<ExpOp><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    } 

    struct TanhOp {
             __device__ scalar_t operator()(scalar_t a)
    {
      return tanhf(a);
    }
    };

    void EwiseTanh(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      UnaryKernel<TanhOp><<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    } 

    struct MaximumOp {
             __device__ scalar_t operator()(scalar_t a, scalar_t b)
    {
      return max(a, b);
    }
    };
    
    void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseBinaryKernel<MaximumOp><<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    void ScalarMaximum(const CudaArray &a, const scalar_t val, CudaArray *out) {
      CudaDims dim = CudaOneDim(out->size);
      ScalarCalKernel<MaximumOp><<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }


    __global__ void MatmulNaiveKernel(const scalar_t *a, scalar_t *b, scalar_t *out, uint32_t M, uint32_t N,
                uint32_t P) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= M || j >= P) {
          return ;
        }
        double sum = 0;
        for (size_t k = 0; k< N ; k++) {
            sum += (double)a[i * N + k] * (double)b[k * P + j];
        }
        out[i * P + j] = (scalar_t)sum;
    }

    void MatmulNaive(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P) 
    {
      CudaDims dim = CudaTwoDim(M, P);
      MatmulNaiveKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
      cudaDeviceSynchronize();
    }
    
        double c[TILE][TILE] = {{0.0}};
    scalar_t a_temp[TILE], b_temp[TILE];
    __shared__ scalar_t x[L][STRIDE], y[STRIDE][L];

    // 用 clock64() 记录时间戳
    unsigned long long start_load, end_load, start_compute, end_compute, start_write, end_write;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        start_load = clock64();
    }

    for (size_t k = 0; k < N; k += STRIDE) {
        size_t block_x_base = blockIdx.x * L;
        size_t block_y_base = blockIdx.y * L;
        size_t tid = blockDim.x * threadIdx.y + threadIdx.x;
        size_t threads_count = blockDim.x * blockDim.y;

        // ---------------- 加载阶段 ----------------
        for (int i=0 ; i< L *STRIDE / threads_count; i ++) {
            size_t y_i = (tid + threads_count * i) / STRIDE;
            size_t x_i = (tid + threads_count * i) % STRIDE;
            x[y_i][x_i] = a[(block_y_base + y_i) * N + k + x_i];
        }
        for (int i=0 ; i< L *STRIDE / threads_count; i ++) {
            size_t y_i = (tid + threads_count * i) / L;
            size_t x_i = (tid + threads_count * i) % L;
            y[y_i][x_i] = b[(k + y_i) * P + block_x_base + x_i];
        }
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            end_load = clock64();
            start_compute = clock64();
        }

        // ---------------- 计算阶段 ----------------
        for (size_t kk = k; kk < k + STRIDE; kk++) {
            for (size_t i = 0; i< TILE; i++) { 
                a_temp[i] = x[threadIdx.y * TILE + i][kk - k];
                b_temp[i] = y[kk-k][threadIdx.x * TILE + i];
            }
            for (size_t i = 0; i< TILE ; i++) {
                for (size_t j = 0; j < TILE; j++) {
                    c[i][j] += a_temp[i] * b_temp[j];
                }
            }
        }

        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            end_compute = clock64();
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        start_write = clock64();
    }

    // ---------------- 写回阶段 ----------------
    size_t y_base = TILE * (blockIdx.y * blockDim.y + threadIdx.y);
    size_t x_base = TILE * (blockIdx.x * blockDim.x + threadIdx.x);
    for (size_t i = 0; i< TILE ; i++) {
        for (size_t j = 0; j < TILE; j++) {
            out[(y_base + i) * P + x_base + j] = (scalar_t)c[i][j];
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        end_write = clock64();
        printf("Block(%d,%d): Load=%llu, Compute=%llu, Write=%llu cycles\n",
               blockIdx.x, blockIdx.y,
               (end_load - start_load),
               (end_compute - start_compute),
               (end_write - start_write));
    }
__global__ void MatmulSharedMemoryKernelTimed(const scalar_t *a, scalar_t *b, scalar_t *out,
                                              uint32_t M, uint32_t N, uint32_t P) {
    double c[TILE][TILE] = {{0.0}};
    scalar_t a_temp[TILE], b_temp[TILE];
    __shared__ scalar_t x[L][STRIDE], y[STRIDE][L];

    // 用 clock64() 记录时间戳
    unsigned long long start_load, end_load, start_compute, end_compute, start_write, end_write;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        start_load = clock64();
    }

    for (size_t k = 0; k < N; k += STRIDE) {
        size_t block_x_base = blockIdx.x * L;
        size_t block_y_base = blockIdx.y * L;
        size_t tid = blockDim.x * threadIdx.y + threadIdx.x;
        size_t threads_count = blockDim.x * blockDim.y;

        // ---------------- 加载阶段 ----------------
        for (int i=0 ; i< L *STRIDE / threads_count; i ++) {
            size_t y_i = (tid + threads_count * i) / STRIDE;
            size_t x_i = (tid + threads_count * i) % STRIDE;
            x[y_i][x_i] = a[(block_y_base + y_i) * N + k + x_i];
        }
        for (int i=0 ; i< L *STRIDE / threads_count; i ++) {
            size_t y_i = (tid + threads_count * i) / L;
            size_t x_i = (tid + threads_count * i) % L;
            y[y_i][x_i] = b[(k + y_i) * P + block_x_base + x_i];
        }
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            end_load = clock64();
            start_compute = clock64();
        }

        // ---------------- 计算阶段 ----------------
        for (size_t kk = k; kk < k + STRIDE; kk++) {
            for (size_t i = 0; i< TILE; i++) { 
                a_temp[i] = x[threadIdx.y * TILE + i][kk - k];
                b_temp[i] = y[kk-k][threadIdx.x * TILE + i];
            }
            for (size_t i = 0; i< TILE ; i++) {
                for (size_t j = 0; j < TILE; j++) {
                    c[i][j] += a_temp[i] * b_temp[j];
                }
            }
        }

        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            end_compute = clock64();
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        start_write = clock64();
    }

    // ---------------- 写回阶段 ----------------
    size_t y_base = TILE * (blockIdx.y * blockDim.y + threadIdx.y);
    size_t x_base = TILE * (blockIdx.x * blockDim.x + threadIdx.x);
    for (size_t i = 0; i< TILE ; i++) {
        for (size_t j = 0; j < TILE; j++) {
            out[(y_base + i) * P + x_base + j] = (scalar_t)c[i][j];
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        end_write = clock64();
        printf("Block(%d,%d): Load=%llu, Compute=%llu, Write=%llu cycles\n",
               blockIdx.x, blockIdx.y,
               (end_load - start_load),
               (end_compute - start_compute),
               (end_write - start_write));
    }
}

      __global__ void MatmulSharedMemoryKernel(const scalar_t *a, scalar_t *b, scalar_t *out, uint32_t M, uint32_t N,
                uint32_t P) {
        double c[TILE][TILE] = {{0.0}};
        scalar_t a_temp[TILE], b_temp[TILE];

        __shared__ scalar_t x[L][STRIDE], y[STRIDE][L];
        for (size_t k = 0; k < N; k += STRIDE) {
          size_t block_x_base = blockIdx.x * L;
          size_t block_y_base = blockIdx.y * L;
          size_t tid = blockDim.x * threadIdx.y + threadIdx.x;
          size_t threads_count = blockDim.x * blockDim.y;
        // co-fetch
          for (int i=0 ; i< L *STRIDE / threads_count; i ++) {
              size_t y_i = (tid + threads_count * i) / STRIDE;
              size_t x_i = (tid + threads_count * i) % STRIDE;
              x[y_i][x_i] = a[(block_y_base + y_i) * N + k + x_i];
          }
          for (int i=0 ; i< L *STRIDE / threads_count; i ++) {
              size_t y_i = (tid + threads_count * i) / L;
              size_t x_i = (tid + threads_count * i) % L;
              y[y_i][x_i] = b[(k + y_i) * P + block_x_base + x_i];
          }
          __syncthreads();
        // load from global -> register
        for (size_t kk = k; kk < k + STRIDE; kk++) {
          for (size_t i = 0; i< TILE; i++) { 
            a_temp[i] = x[threadIdx.y * TILE + i][kk - k];
            b_temp[i] = y[kk-k][threadIdx.x * TILE + i];
          }
          // 外积accumulate
          for (size_t i = 0; i< TILE ; i++) {
            for (size_t j = 0; j < TILE; j++) {
              c[i][j] += a_temp[i] * b_temp[j];
            }
          }
        }
        }
        // writeback
        size_t y_base = TILE * (blockIdx.y * blockDim.y + threadIdx.y);
        size_t x_base = TILE * (blockIdx.x * blockDim.x + threadIdx.x);
        for (size_t i = 0; i< TILE ; i++) {
            for (size_t j = 0; j < TILE; j++) {
              out[(y_base + i) * P + x_base + j] = (scalar_t)c[i][j];
            }
          }
      }


    void MatmulSharedMemory(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P) {
      CudaDims dim = CudaTwoDim( (P + TILE - 1) / TILE, (M + TILE - 1) / TILE);
      MatmulSharedMemoryKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
      cudaDeviceSynchronize();
        }
        __global__ void MatmulRegsiterTiledKernel(const scalar_t *a, scalar_t *b, scalar_t *out, uint32_t M, uint32_t N,
                uint32_t P) {
        size_t x_base = TILE * (blockIdx.x * blockDim.x + threadIdx.x);
        size_t y_base = TILE * (blockIdx.y * blockDim.y + threadIdx.y);
        double c[TILE][TILE];
        for (int i = 0; i< TILE ; i++) {
            for (int j = 0; j < TILE; j++) {
              c[i][j] = 0; 
            }
        }
        scalar_t a_temp[TILE], b_temp[TILE];
          // load from global -> register
        for (size_t k = 0; k < N; k++) {
          for (size_t i = 0; i< TILE; i++) { 
            if (x_base + i >= M) {
              a_temp[i] = 0;
            } else {
              a_temp[i] = a[N * (x_base + i) + k];
            }
            if (y_base + i >= P) {
              b_temp[i] = 0;
            } else {
              b_temp[i] = b[k * P + y_base + i];
            }
          }
          // 外积accumulate
          for (size_t i = 0; i< TILE ; i++) {
            for (size_t j = 0; j < TILE; j++) {
              c[i][j] += a_temp[i] * b_temp[j];
            }
          }
        }
        // write_back register -> memory
        for (size_t i = 0; i<  TILE ; i++) {
            for (size_t j = 0; j < TILE; j++) {  
              if (i + x_base < M && j + y_base < P) {
                out[P * (i+x_base) + j + y_base] = (scalar_t)c[i][j];
              }    
            }
          }
      }

    void MatmulRegisterTiled(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P) {
      CudaDims dim = CudaTwoDim((M + TILE - 1) / TILE, (P + TILE - 1) / TILE);
      MatmulRegsiterTiledKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
      cudaDeviceSynchronize();
      }
    /**
     * In the code the follows, use the above template to create analogous elementise
     * and and scalar operators for the following functions.  See the numpy backend for
     * examples of how they should work.
     *   - EwiseMul, ScalarMul
     *   - EwiseDiv, ScalarDiv
     *   - ScalarPower
     *   - EwiseMaximum, ScalarMaximum
     *   - EwiseEq, ScalarEq
     *   - EwiseGe, ScalarGe
     *   - EwiseLog
     *   - EwiseExp
     *   - EwiseTanh
     *
     * If you implement all these naively, there will be a lot of repeated code, so
     * you are welcome (but not required), to use macros or templates to define these
     * functions (however you want to do so, as long as the functions match the proper)
     * signatures above.
     */

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

    void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P)
    {
      /**
       * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
       * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
       * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
       * over (i,j) entries in the output array.  However, to really get the full benefit of this
       * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
       * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
       * the CPU backend, here you should implement a single function that works across all size
       * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
       * implementations, this function here will largely just set up the kernel call, and you should
       * implement the logic in a separate MatmulKernel() call.
       *
       *
       * Args:
       *   a: compact 2D array of size m x n
       *   b: comapct 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   M: rows of a / out
       *   N: columns of a / rows of b
       *   P: columns of b / out
       */

      
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Max and sum reductions
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, uint32_t reduce_size, uint32_t size) {
       size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
       if (gid >= size) {
         return;
       }
       size_t start = gid * reduce_size;
       scalar_t ret = a[start];
       for (size_t i = 0; i<reduce_size; i++) {
          ret = max(ret, a[i + start]);
       }
       out[gid] = ret;
    }

    void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
       * for simplicity you can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      /// BEGIN SOLUTION
      
      CudaDims dim = CudaOneDim(out->size);
      ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
      /// END SOLUTION
    }


    __global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, uint32_t reduce_size, uint32_t size) {
       size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
       if (gid >= size) {
         return;
       }
       size_t start = gid * reduce_size;
       scalar_t sum = 0;
       for (size_t i = 0; i < reduce_size; i++) {
          sum += a[i + start];
       }
       out[gid] = sum;
    }

    void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
       * can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      CudaDims dim = CudaOneDim(out->size);
      ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
    }

  } // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer); });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out)
        {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", MatmulSharedMemory);
  m.def("matmul_naive", MatmulNaive);
  m.def("matmul_register_tiled", MatmulRegisterTiled);
  m.def("matmul_shared_memory", MatmulSharedMemory);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
