/*
 * Copyright (c) 2026 by FlashInfer team.
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
#ifndef FLASHINFER_GEMM_GEMM_GATED_ACT_SM90_CUH_
#define FLASHINFER_GEMM_GEMM_GATED_ACT_SM90_CUH_

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "cutlass_gated/gated_builder.hpp"

namespace flashinfer {

namespace gemm {

using namespace cute;

/*!
 * \brief Fused gated-activation GEMM on SM90a (CUTLASS example 113 lineage).
 *
 * Computes, in CUTLASS terms with A = packed weight and B = activations,
 *   D(i, t) = scale * (beta * C(i) + alpha * acc_val(i, t))
 *                   * ActFn(beta * C(i + m_fused/2) + alpha * acc_gate(i, t))
 * where acc_val comes from weight rows [0, m_fused/2) (the up/value half) and
 * acc_gate from rows [m_fused/2, m_fused). The weight stays plainly
 * concatenated in memory; a CuTe layout trick interleaves 8 value rows with
 * their 8 matching gate rows per 16-row WGMMA block, so the whole thing is a
 * single standard TMA warp-specialized mainloop with a custom EVT epilogue.
 *
 * \param w_ptr   weight, [m_fused, k], K-major (row-major), m_fused % 16 == 0
 * \param x_ptr   activations, [n_tokens, k], K-major
 * \param bias_ptr optional bias, [m_fused] (broadcast over tokens); nullptr disables
 * \param d_ptr   output, [m_fused / 2, n_tokens], M-major
 *                (i.e. a row-major [n_tokens, m_fused / 2] tensor)
 * \param alpha / alpha_ptr  accumulator scale applied before bias + activation;
 *                the device pointer takes precedence when non-null
 * \param scale_ptr per-tensor output scale (device); only read when DTypeOut is fp8
 */
template <typename DTypeIn, typename DTypeOut, template <class> class ActivationFn, bool Pingpong>
cudaError_t GemmGatedActSM90Run(void* workspace, size_t workspace_size_in_bytes, void* w_ptr,
                                void* x_ptr, void* bias_ptr, void* d_ptr, float alpha,
                                float const* alpha_ptr, float beta, float const* scale_ptr,
                                int64_t m_fused, int64_t n_tokens, int64_t k, int64_t sm_count,
                                cudaStream_t stream) {
  // A operand: the packed [W_up ; W_gate] weight.
  using ElementA = DTypeIn;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  // B operand: the token activations.
  using ElementB = DTypeIn;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // Fp8 output goes through the epilogue's per-tensor quantize node; its bias
  // (C operand) stays 16-bit.
  constexpr bool Quantize = cutlass::sizeof_bits<DTypeOut>::value == 8;
  using ElementC = cute::conditional_t<Quantize, cutlass::half_t, DTypeOut>;
  using LayoutC = cutlass::layout::ColumnMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementD = DTypeOut;
  using LayoutD = cutlass::layout::ColumnMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpiTileShape = cutlass::epilogue::collective::EpilogueTileAuto;
  using ClusterShape = Shape<_1, _2, _1>;
  using TileShapeK = Int<128 * 8 / cutlass::sizeof_bits<ElementA>::value>;

  using KernelSchedule = cute::conditional_t<
      cutlass::gemm::collective::detail::is_input_fp8<ElementA, ElementB>(),
      cute::conditional_t<Pingpong, cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
                          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>,
      cute::conditional_t<Pingpong, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                          cutlass::gemm::KernelTmaWarpSpecializedCooperative>>;
  using EpilogueSchedule = cute::conditional_t<Pingpong, cutlass::epilogue::TmaWarpSpecialized,
                                               cutlass::epilogue::TmaWarpSpecializedCooperative>;
  using TileShape =
      cute::conditional_t<Pingpong, Shape<_128, _128, TileShapeK>, Shape<_128, _256, TileShapeK>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::Sm90CollectiveBuilderGated<
      OperatorClass, TileShape, ClusterShape, EpiTileShape, ElementAccumulator, ElementCompute,
      ElementScalar,
      ElementCompute,  // ElementIntermediate: fp32 (ExactMode round-trip disabled)
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, EpilogueSchedule, ActivationFn,
      Quantize>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::Sm90CollectiveBuilderGated<
      OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using ProblemShape = Shape<int, int, int, int>;
  using GatedProblemShape = decltype(cutlass::sm90_make_gated_shape<0>(ProblemShape{}));

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<GatedProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  // D has shape (m_fused/2, n_tokens), so it is stored through the epilogue's
  // aux TMA store; its stride type lives on the fusion operation.
  using StrideD =
      typename GemmKernel::CollectiveEpilogue::FusionCallbacks::Operation::GmemLayoutTagAux;

  int m = static_cast<int>(m_fused);
  int n = static_cast<int>(n_tokens);
  int kk = static_cast<int>(k);
  constexpr int L = 1;
  constexpr int NC = 1;  // bias is broadcast along tokens

  auto stride_A = cutlass::sm90_make_gated_packed_stride(StrideA{}, {m, kk, L});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, kk, L});
  auto stride_C = cutlass::sm90_make_gated_packed_stride(StrideC{}, {m, NC, L});
  auto stride_D = cutlass::sm90_make_gated_packed_stride(StrideD{}, {m / 2, n, L});
  get<1>(stride_C) = 0;  // broadcast bias along the token mode

  cutlass::KernelHardwareInfo hw_info;
  cudaError_t err = cudaGetDevice(&hw_info.device_id);
  if (err != cudaSuccess) {
    return err;
  }
  hw_info.sm_count = static_cast<int>(sm_count);

  auto problem_shape = cutlass::sm90_make_gated_shape<0>(make_shape(m, n, kk, L));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {static_cast<ElementA const*>(w_ptr), stride_A, static_cast<ElementB const*>(x_ptr),
       stride_B},
      {{}, static_cast<ElementC const*>(bias_ptr), stride_C, nullptr, {}},
      hw_info,
      {}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = alpha;
  fusion_args.alpha_ptr = alpha_ptr;
  fusion_args.beta = beta;
  fusion_args.scale_ptr = scale_ptr;
  fusion_args.ptr_D = static_cast<ElementD*>(d_ptr);
  fusion_args.dD = stride_D;
  fusion_args.sm_count = hw_info.sm_count;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator allocator(workspace, workspace_size_in_bytes);
  auto workspace_ptr =
      allocator.aligned_alloc<void>(workspace_size, 16, "gemm_gated_act_sm90_workspace");

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr, stream));
  CUTLASS_CHECK(gemm.run(stream));
  return cudaSuccess;
}

}  // namespace gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GEMM_GATED_ACT_SM90_CUH_
