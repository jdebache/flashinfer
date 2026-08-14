/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

constexpr int kTokens = 96;
constexpr int kQFeatures = 1536;
constexpr int kKvFeatures = 512;
constexpr int kKPeFeatures = 64;
constexpr int kQkvFeatures = kQFeatures + kKvFeatures + kKPeFeatures;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
  }
  return value;
}

__device__ __forceinline__ float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

__device__ __forceinline__ float ToFloat(float value) { return value; }

template <typename CacheType>
__global__ __launch_bounds__(kThreads, 1) void kernel_qkv_a_proj_norm_rope_post(
    const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ kv_weight, const int64_t* __restrict__ positions,
    const CacheType* __restrict__ cos_sin_cache, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ kv_out, __nv_bfloat16* __restrict__ k_pe_out, float eps) {
  const int token = blockIdx.x;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread % 32;
  const int64_t qkv_row = static_cast<int64_t>(token) * kQkvFeatures;

  float q_values[6];
  float kv_values[2];
  float q_sum_squares = 0.0f;
  float kv_sum_squares = 0.0f;
#pragma unroll
  for (int item = 0; item < 6; ++item) {
    const int feature = thread + item * kThreads;
    const float value = __bfloat162float(qkv[qkv_row + feature]);
    q_values[item] = value;
    q_sum_squares += value * value;
  }
#pragma unroll
  for (int item = 0; item < 2; ++item) {
    const int feature = thread + item * kThreads;
    const float value = __bfloat162float(qkv[qkv_row + kQFeatures + feature]);
    kv_values[item] = value;
    kv_sum_squares += value * value;
  }

  q_sum_squares = WarpSum(q_sum_squares);
  kv_sum_squares = WarpSum(kv_sum_squares);
  __shared__ float reduction[2 * kWarps];
  if (lane == 0) {
    reduction[warp] = q_sum_squares;
    reduction[kWarps + warp] = kv_sum_squares;
  }
  __syncthreads();

  if (warp == 0) {
    float q_block_sum = lane < kWarps ? reduction[lane] : 0.0f;
    float kv_block_sum = lane < kWarps ? reduction[kWarps + lane] : 0.0f;
    q_block_sum = WarpSum(q_block_sum);
    kv_block_sum = WarpSum(kv_block_sum);
    if (lane == 0) {
      reduction[0] = rsqrtf(q_block_sum / static_cast<float>(kQFeatures) + eps);
      reduction[1] = rsqrtf(kv_block_sum / static_cast<float>(kKvFeatures) + eps);
    }
  }
  __syncthreads();

  const float q_inverse_rms = reduction[0];
  const float kv_inverse_rms = reduction[1];
  const int64_t q_row = static_cast<int64_t>(token) * kQFeatures;
  const int64_t kv_row = static_cast<int64_t>(token) * kKvFeatures;
#pragma unroll
  for (int item = 0; item < 6; ++item) {
    const int feature = thread + item * kThreads;
    const float scale = q_inverse_rms * __bfloat162float(q_weight[feature]);
    q_out[q_row + feature] = __float2bfloat16_rn(q_values[item] * scale);
  }
#pragma unroll
  for (int item = 0; item < 2; ++item) {
    const int feature = thread + item * kThreads;
    const float scale = kv_inverse_rms * __bfloat162float(kv_weight[feature]);
    kv_out[kv_row + feature] = __float2bfloat16_rn(kv_values[item] * scale);
  }

  if (thread < kKPeFeatures / 2) {
    const int64_t cache_row = positions[token] * kKPeFeatures;
    const float cosine = ToFloat(cos_sin_cache[cache_row + thread]);
    const float sine = ToFloat(cos_sin_cache[cache_row + kKPeFeatures / 2 + thread]);
    const float even = __bfloat162float(qkv[qkv_row + kQFeatures + kKvFeatures + 2 * thread]);
    const float odd = __bfloat162float(qkv[qkv_row + kQFeatures + kKvFeatures + 2 * thread + 1]);
    const int64_t output_offset = static_cast<int64_t>(token) * kKPeFeatures + 2 * thread;
    k_pe_out[output_offset] = __float2bfloat16_rn(even * cosine - odd * sine);
    k_pe_out[output_offset + 1] = __float2bfloat16_rn(odd * cosine + even * sine);
  }
}

namespace flashinfer {
namespace fused_qkv_a_proj_norm_rope_sm100 {

using tvm::ffi::TensorView;

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckContiguousCuda(const TensorView& tensor, int ndim, const char* name) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.ndim() == ndim) << name << " must be " << ndim << "D";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void CheckBf16(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  TVM_FFI_ICHECK(dtype.code == kDLBfloat && dtype.bits == 16 && dtype.lanes == 1)
      << name << " must be bfloat16";
}

inline bool IsFloat32(const TensorView& tensor) {
  const DLDataType dtype = tensor.dtype();
  return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
}

inline bool IsInt64(const TensorView& tensor) {
  const DLDataType dtype = tensor.dtype();
  return dtype.code == kDLInt && dtype.bits == 64 && dtype.lanes == 1;
}

inline void CheckShape2D(const TensorView& tensor, int64_t rows, int64_t columns,
                         const char* name) {
  TVM_FFI_ICHECK(tensor.size(0) == rows && tensor.size(1) == columns)
      << name << " must have shape (" << rows << ", " << columns << ")";
}

void Run(TensorView qkv, TensorView q_weight, TensorView kv_weight, TensorView positions,
         TensorView cos_sin_cache, TensorView q_out, TensorView kv_out, TensorView k_pe_out,
         double eps) {
  CheckContiguousCuda(qkv, 2, "qkv");
  CheckContiguousCuda(q_weight, 1, "q_weight");
  CheckContiguousCuda(kv_weight, 1, "kv_weight");
  CheckContiguousCuda(positions, 1, "positions");
  CheckContiguousCuda(cos_sin_cache, 2, "cos_sin_cache");
  CheckContiguousCuda(q_out, 2, "q_out");
  CheckContiguousCuda(kv_out, 2, "kv_out");
  CheckContiguousCuda(k_pe_out, 3, "k_pe_out");
  CheckBf16(qkv, "qkv");
  CheckBf16(q_weight, "q_weight");
  CheckBf16(kv_weight, "kv_weight");
  CheckBf16(q_out, "q_out");
  CheckBf16(kv_out, "kv_out");
  CheckBf16(k_pe_out, "k_pe_out");
  TVM_FFI_ICHECK(IsInt64(positions)) << "positions must be int64";
  const DLDataType cache_dtype = cos_sin_cache.dtype();
  const bool cache_is_bf16 =
      cache_dtype.code == kDLBfloat && cache_dtype.bits == 16 && cache_dtype.lanes == 1;
  TVM_FFI_ICHECK(cache_is_bf16 || IsFloat32(cos_sin_cache))
      << "cos_sin_cache must be bfloat16 or float32";

  CheckShape2D(qkv, kTokens, kQkvFeatures, "qkv");
  TVM_FFI_ICHECK(q_weight.size(0) == kQFeatures) << "q_weight must have shape (1536,)";
  TVM_FFI_ICHECK(kv_weight.size(0) == kKvFeatures) << "kv_weight must have shape (512,)";
  TVM_FFI_ICHECK(positions.size(0) == kTokens) << "positions must have shape (96,)";
  TVM_FFI_ICHECK(cos_sin_cache.size(0) > 0 && cos_sin_cache.size(1) == kKPeFeatures)
      << "cos_sin_cache must have shape (max_position, 64)";
  CheckShape2D(q_out, kTokens, kQFeatures, "q_out");
  CheckShape2D(kv_out, kTokens, kKvFeatures, "kv_out");
  TVM_FFI_ICHECK(k_pe_out.size(0) == kTokens && k_pe_out.size(1) == 1 &&
                 k_pe_out.size(2) == kKPeFeatures)
      << "k_pe_out must have shape (96, 1, 64)";

  const int device_id = qkv.device().device_id;
  TVM_FFI_ICHECK(
      q_weight.device().device_id == device_id && kv_weight.device().device_id == device_id &&
      positions.device().device_id == device_id && cos_sin_cache.device().device_id == device_id &&
      q_out.device().device_id == device_id && kv_out.device().device_id == device_id &&
      k_pe_out.device().device_id == device_id)
      << "all tensors must be on the same CUDA device";

  ffi::CUDADeviceGuard device_guard(device_id);
  const cudaStream_t stream = get_stream(qkv.device());
  if (cache_is_bf16) {
    kernel_qkv_a_proj_norm_rope_post<<<kTokens, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kv_weight.data_ptr()),
        reinterpret_cast<const int64_t*>(positions.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(kv_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_pe_out.data_ptr()), static_cast<float>(eps));
  } else {
    kernel_qkv_a_proj_norm_rope_post<<<kTokens, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kv_weight.data_ptr()),
        reinterpret_cast<const int64_t*>(positions.data_ptr()),
        reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(kv_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_pe_out.data_ptr()), static_cast<float>(eps));
  }
  CheckCuda(cudaGetLastError(), "fused_qkv_a_proj_norm_rope post kernel launch");
}

}  // namespace fused_qkv_a_proj_norm_rope_sm100
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_qkv_a_proj_norm_rope_post_sm100,
                              flashinfer::fused_qkv_a_proj_norm_rope_sm100::Run);
