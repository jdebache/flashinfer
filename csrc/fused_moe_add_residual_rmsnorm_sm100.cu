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

// Fused MoE output addition, residual accumulation, and RMSNorm for BF16
// inputs with hidden size 7168. One 896-thread CTA processes each token row.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

constexpr int kFusedMoeRmsNormThreads = 896;
constexpr int kFusedMoeRmsNormNumWarps = kFusedMoeRmsNormThreads / 32;
constexpr int kFusedMoeRmsNormHiddenSize = 7168;
constexpr int kFusedMoeRmsNormValuesPerThread = 8;
constexpr int kFusedMoeRmsNormSmemBytes = 128;
constexpr unsigned long long kEvictFirstCachePolicy = 0x12F0000000000000ULL;

__device__ __forceinline__ int MakeWarpUniform(int value) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(value));
  return result;
}

__device__ __forceinline__ void UnpackBf16x8(const uint4& packed, float* values) {
  const uint32_t* pairs = reinterpret_cast<const uint32_t*>(&packed);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    asm volatile(
        "{\n\t"
        "shl.b32 %0, %2, 16;\n\t"
        "and.b32 %1, %2, 0xffff0000;\n\t"
        "}\n"
        : "=f"(values[pair * 2]), "=f"(values[pair * 2 + 1])
        : "r"(pairs[pair]));
  }
}

__device__ __forceinline__ void LoadBf16x8(const __nv_bfloat16* input, float* values) {
  const uint4 packed = *reinterpret_cast<const uint4*>(input);
  UnpackBf16x8(packed, values);
}

__device__ __forceinline__ void LoadBf16x8EvictFirst(const __nv_bfloat16* input, float* values) {
  uint4 packed;
  asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w)
               : "l"(input), "l"(kEvictFirstCachePolicy)
               : "memory");
  UnpackBf16x8(packed, values);
}

__device__ __forceinline__ void StoreBf16x8(const float* values, __nv_bfloat16* output) {
  __nv_bfloat162 packed[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    packed[pair] = __floats2bfloat162_rn(values[pair * 2], values[pair * 2 + 1]);
  }
  *reinterpret_cast<uint4*>(output) = *reinterpret_cast<const uint4*>(packed);
}

extern "C" {

__global__
__launch_bounds__(kFusedMoeRmsNormThreads, 1) void kernel_fused_moe_add_residual_rmsnorm_h7168(
    const __nv_bfloat16* __restrict__ routed_output,
    const __nv_bfloat16* __restrict__ shared_output, const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ hidden_states,
    __nv_bfloat16* __restrict__ residual_out, float eps) {
  const int thread_id = threadIdx.x;
  const int warp_id = MakeWarpUniform(thread_id / 32);
  const int lane_id = thread_id % 32;
  const int element_offset =
      blockIdx.x * kFusedMoeRmsNormHiddenSize + thread_id * kFusedMoeRmsNormValuesPerThread;

  extern __shared__ __align__(1024) char shared_memory[];
  float* reduction = reinterpret_cast<float*>(shared_memory);

  float weight_values[kFusedMoeRmsNormValuesPerThread];
  float routed_values[kFusedMoeRmsNormValuesPerThread];
  float shared_values[kFusedMoeRmsNormValuesPerThread];
  float residual_input[kFusedMoeRmsNormValuesPerThread];
  LoadBf16x8(weight + thread_id * kFusedMoeRmsNormValuesPerThread, weight_values);
  LoadBf16x8EvictFirst(routed_output + element_offset, routed_values);
  LoadBf16x8EvictFirst(shared_output + element_offset, shared_values);
  LoadBf16x8EvictFirst(residual + element_offset, residual_input);

  float residual_values[kFusedMoeRmsNormValuesPerThread];
  float sum_squares = 0.0f;
#pragma unroll
  for (int item = 0; item < kFusedMoeRmsNormValuesPerThread; ++item) {
    const __nv_bfloat16 moe_value = __float2bfloat16(routed_values[item] + shared_values[item]);
    const float value = __bfloat162float(moe_value) + residual_input[item];
    residual_values[item] = value;
    sum_squares += value * value;
  }
  StoreBf16x8(residual_values, residual_out + element_offset);

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_squares += __shfl_xor_sync(0xFFFFFFFF, sum_squares, offset);
  }
  if (lane_id == 0) {
    reduction[warp_id] = sum_squares;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane_id < kFusedMoeRmsNormNumWarps ? reduction[lane_id] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_sum += __shfl_xor_sync(0xFFFFFFFF, block_sum, offset);
    }
    if (lane_id == 0) {
      reduction[0] = rsqrtf(block_sum / static_cast<float>(kFusedMoeRmsNormHiddenSize) + eps);
    }
  }
  __syncthreads();

  const float inverse_rms = reduction[0];
#pragma unroll
  for (int item = 0; item < kFusedMoeRmsNormValuesPerThread; ++item) {
    residual_values[item] *= inverse_rms * weight_values[item];
  }
  StoreBf16x8(residual_values, hidden_states + element_offset);
}

}  // extern "C"

namespace flashinfer {
namespace fused_moe_add_residual_rmsnorm_sm100 {

using tvm::ffi::TensorView;

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckBf16(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  TVM_FFI_ICHECK(dtype.code == kDLBfloat && dtype.bits == 16 && dtype.lanes == 1)
      << name << " must be bfloat16";
}

inline void CheckTensor(const TensorView& tensor, int ndim, const char* name) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.ndim() == ndim) << name << " must be " << ndim << "D";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0)
      << name << " must be 16-byte aligned";
  CheckBf16(tensor, name);
}

inline void CheckSm100Family(int device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(compute capability major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(compute capability minor)");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "fused_moe_add_residual_rmsnorm_sm100 requires SM100 or SM103, got sm_" << major << minor;
}

void Run(TensorView routed_output, TensorView shared_output, TensorView residual, TensorView weight,
         TensorView hidden_states, TensorView residual_out, double eps) {
  CheckTensor(routed_output, 2, "routed_output");
  CheckTensor(shared_output, 2, "shared_output");
  CheckTensor(residual, 2, "residual");
  CheckTensor(weight, 1, "weight");
  CheckTensor(hidden_states, 2, "hidden_states");
  CheckTensor(residual_out, 2, "residual_out");

  const int device_id = routed_output.device().device_id;
  TVM_FFI_ICHECK(
      shared_output.device().device_id == device_id && residual.device().device_id == device_id &&
      weight.device().device_id == device_id && hidden_states.device().device_id == device_id &&
      residual_out.device().device_id == device_id)
      << "all tensors must be on the same CUDA device";

  const int64_t num_tokens = routed_output.size(0);
  TVM_FFI_ICHECK(routed_output.size(1) == kFusedMoeRmsNormHiddenSize)
      << "hidden size must be " << kFusedMoeRmsNormHiddenSize;
  TVM_FFI_ICHECK(
      shared_output.size(0) == num_tokens && shared_output.size(1) == kFusedMoeRmsNormHiddenSize &&
      residual.size(0) == num_tokens && residual.size(1) == kFusedMoeRmsNormHiddenSize &&
      hidden_states.size(0) == num_tokens && hidden_states.size(1) == kFusedMoeRmsNormHiddenSize &&
      residual_out.size(0) == num_tokens && residual_out.size(1) == kFusedMoeRmsNormHiddenSize)
      << "routed_output, shared_output, residual, hidden_states, and residual_out shapes must "
         "match";
  TVM_FFI_ICHECK(weight.size(0) == kFusedMoeRmsNormHiddenSize)
      << "weight must have shape (" << kFusedMoeRmsNormHiddenSize << ",)";
  TVM_FFI_ICHECK(num_tokens <= std::numeric_limits<int>::max() / kFusedMoeRmsNormHiddenSize)
      << "num_tokens exceeds the kernel's i32 indexing range";
  const void* hidden_states_ptr = hidden_states.data_ptr();
  const void* residual_out_ptr = residual_out.data_ptr();
  TVM_FFI_ICHECK(
      num_tokens == 0 ||
      (hidden_states_ptr != residual_out_ptr && hidden_states_ptr != routed_output.data_ptr() &&
       hidden_states_ptr != shared_output.data_ptr() && hidden_states_ptr != residual.data_ptr() &&
       hidden_states_ptr != weight.data_ptr() && residual_out_ptr != routed_output.data_ptr() &&
       residual_out_ptr != shared_output.data_ptr() && residual_out_ptr != residual.data_ptr() &&
       residual_out_ptr != weight.data_ptr()))
      << "output tensors must not alias inputs or each other";

  ffi::CUDADeviceGuard device_guard(device_id);
  CheckSm100Family(device_id);
  if (num_tokens == 0) {
    return;
  }

  const cudaStream_t stream = get_stream(routed_output.device());
  kernel_fused_moe_add_residual_rmsnorm_h7168<<<dim3(static_cast<unsigned int>(num_tokens)),
                                                dim3(kFusedMoeRmsNormThreads),
                                                kFusedMoeRmsNormSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(routed_output.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(shared_output.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(hidden_states.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()), static_cast<float>(eps));
  CheckCuda(cudaGetLastError(), "fused_moe_add_residual_rmsnorm_sm100 kernel launch");
}

}  // namespace fused_moe_add_residual_rmsnorm_sm100
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_moe_add_residual_rmsnorm_sm100,
                              flashinfer::fused_moe_add_residual_rmsnorm_sm100::Run);
