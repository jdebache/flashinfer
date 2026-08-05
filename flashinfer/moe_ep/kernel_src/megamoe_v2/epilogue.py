# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""FC1 epilogue: SwiGLU + clamp + top-k weight + requantization to NVFP4.

What this has to produce
------------------------

FC1's accumulator is ``(2*I channels, tokens)`` under swap-AB, gate in the low
half.  FC2 consumes NVFP4 with per-16 block scales along *its* K axis, which is
FC1's ``I``-wide output-channel axis.  So the epilogue must:

1. clamp gate and up (pre-activation -- clamping after silu is a different
   function, see :func:`..reference.swiglu`);
2. ``silu(gate) * up``;
3. multiply by the row's top-k routing weight, so the later cross-rank combine
   is a plain sum and can be an atomic add (see the module docs of
   :mod:`.types` for why the weight is folded here rather than after FC2);
4. quantize to NVFP4 in blocks of 16 **along channels**, writing swizzled
   scales where the FC2 TMA descriptor will look for them.

Why the smem round trip
-----------------------

The TMEM load hands each thread *one channel and 64 tokens*
(``m = 4*lane + warp``).  Step 2 is therefore register-local -- the thread that
owns channel ``m`` holds both ``gate[m]`` and ``up[m]`` when gate and up are
accumulated into two TMEM stages at the same M offset.  Step 4 is not: a
16-channel block is spread over 16 threads in 4 different warps, so the block
amax cannot be a register or even a warp reduction.

Rather than shuffle, the activation is staged to smem in
``(channel, token)`` order and re-read token-major, so each thread ends up
owning 16 *consecutive channels* of one token and the amax becomes register
local again -- the same shape the standalone input quantizer already uses, and
therefore the same validated encode path.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32

from .quant import (
    FP4_MAX,
    _ZERO_MASK_GAIN,
    scale_element_offset,
)
from .types import NVFP4_BLOCK

_RCP_FP4_MAX = 1.0 / FP4_MAX
_FP32_MAX = 3.402823466e38


@cute.jit
def swiglu_activate(
    gate: Float32,
    up: Float32,
    weight: Float32,
    *,
    clamp: cutlass.Constexpr,
    apply_weight: cutlass.Constexpr[bool],
) -> Float32:
    """One element of ``silu(clamp(gate)) * clamp(up) * weight``.

    ``clamp`` is a Python float or None, so the clamp folds away entirely when
    it is not configured.  silu is written as ``x * sigmoid(x)`` via the fast
    reciprocal-exponential; at NVFP4 output precision the approximate exp is
    far below the quantization step.
    """
    g = gate
    u = up
    if cutlass.const_expr(clamp is not None):
        lo = Float32(-clamp)
        hi = Float32(clamp)
        g = cute.arch.fmax(lo, cute.arch.fmin(hi, g))
        u = cute.arch.fmax(lo, cute.arch.fmin(hi, u))
    sig = Float32(1.0) / (Float32(1.0) + cute.arch.exp(-g))
    out = g * sig * u
    if cutlass.const_expr(apply_weight):
        out = out * weight
    return out


@cute.jit
def quantize_channel_block(
    values,  # rmem tensor of NVFP4_BLOCK Float32
    token_row: Int32,
    block: Int32,
    out_codes: cute.Tensor,
    out_scales: cute.Tensor,
    *,
    num_k_atoms: cutlass.Constexpr[int],
) -> None:
    """Encode one 16-channel block of one token and store code + scale.

    Identical arithmetic to :func:`..quant.quantize_row_range`'s inner block --
    kept as one function so the FC1 output and the model input can never drift
    to two different NVFP4 encodings.
    """
    absmax = Float32(0.0)
    for i in cutlass.range_constexpr(NVFP4_BLOCK):
        absmax = cute.arch.fmax(absmax, cute.arch.fmax(values[i], -values[i]))

    scale_e4m3 = (absmax * Float32(_RCP_FP4_MAX)).to(cutlass.Float8E4M3FN)
    scale = Float32(scale_e4m3)
    encode = cute.arch.fmin(cute.arch.rcp_approx(scale), Float32(_FP32_MAX))
    encode = encode * cute.arch.fmin(scale * Float32(_ZERO_MASK_GAIN), Float32(1.0))

    scaled = cute.make_rmem_tensor((NVFP4_BLOCK,), Float32)
    for i in cutlass.range_constexpr(NVFP4_BLOCK):
        scaled[i] = values[i] * encode
    codes = cute.make_rmem_tensor((NVFP4_BLOCK,), cutlass.Float4E2M1FN)
    codes.store(scaled.load().to(cutlass.Float4E2M1FN))

    base = block * Int32(NVFP4_BLOCK)
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float4E2M1FN,
        num_bits_per_copy=NVFP4_BLOCK * 4,
    )
    dst = cute.zipped_divide(out_codes[token_row, None], (NVFP4_BLOCK,))
    ptr = dst[(None,), (block,)].iterator
    aligned = cute.make_tensor(
        cute.make_ptr(
            ptr.dtype, ptr.toint(), ptr.memspace, assumed_align=NVFP4_BLOCK // 2
        ),
        dst[(None,), (block,)].layout,
    )
    cute.copy(store_atom, codes, aligned)
    out_scales[
        scale_element_offset(token_row, block, num_k_atoms=num_k_atoms)
    ] = scale_e4m3
    return base


@cute.jit
def activate_and_quantize_rows(
    gate: cute.Tensor,  # (tokens, I) Float32
    up: cute.Tensor,  # (tokens, I) Float32
    weights: cute.Tensor,  # (tokens,) Float32
    out_codes: cute.Tensor,  # (tokens, I) Float4E2M1FN
    out_scales: cute.Tensor,  # flat Float8E4M3FN
    *,
    intermediate: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    clamp: cutlass.Constexpr,
    apply_weight: cutlass.Constexpr[bool],
    first_token: Int32,
    token_limit: Int32,
    token_stride: Int32,
    lane_idx: Int32,
) -> None:
    """Reference-shaped driver: one warp per token, lane ``l`` owns block ``l``.

    This is the *standalone* form, used to validate the numerics against the
    torch oracle before the same arithmetic is embedded in the GEMM epilogue
    (where gate/up arrive from TMEM and the token-major re-read happens through
    smem instead of from gmem).
    """
    num_blocks: cutlass.Constexpr[int] = intermediate // NVFP4_BLOCK
    blocks_per_lane: cutlass.Constexpr[int] = (num_blocks + 31) // 32

    token = first_token
    while token < token_limit:
        w = weights[token]
        for step in cutlass.range_constexpr(blocks_per_lane):
            block = lane_idx + Int32(step * 32)
            if block < Int32(num_blocks):
                base = block * Int32(NVFP4_BLOCK)
                vals = cute.make_rmem_tensor((NVFP4_BLOCK,), Float32)
                for i in cutlass.range_constexpr(NVFP4_BLOCK):
                    vals[i] = swiglu_activate(
                        Float32(gate[token, base + Int32(i)]),
                        Float32(up[token, base + Int32(i)]),
                        w,
                        clamp=clamp,
                        apply_weight=apply_weight,
                    )
                quantize_channel_block(
                    vals, token, block, out_codes, out_scales,
                    num_k_atoms=num_k_atoms,
                )
        token = token + token_stride


@cute.kernel
def _activate_quantize_kernel(
    gate: cute.Tensor,
    up: cute.Tensor,
    weights: cute.Tensor,
    out_codes: cute.Tensor,
    out_scales: cute.Tensor,
    num_tokens: Int32,
    intermediate: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    clamp: cutlass.Constexpr,
    apply_weight: cutlass.Constexpr[bool],
    warps_per_cta: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    activate_and_quantize_rows(
        gate, up, weights, out_codes, out_scales,
        intermediate=intermediate,
        num_k_atoms=num_k_atoms,
        clamp=clamp,
        apply_weight=apply_weight,
        first_token=bidx * Int32(warps_per_cta) + tidx // Int32(32),
        token_limit=num_tokens,
        token_stride=gdim * Int32(warps_per_cta),
        lane_idx=tidx % Int32(32),
    )


@cute.jit
def activate_quantize_launch(
    gate: cute.Tensor,
    up: cute.Tensor,
    weights: cute.Tensor,
    out_codes: cute.Tensor,
    out_scales: cute.Tensor,
    num_tokens: Int32,
    stream,
    *,
    intermediate: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    clamp: cutlass.Constexpr = None,
    apply_weight: cutlass.Constexpr[bool] = True,
    num_ctas: cutlass.Constexpr[int] = 32,
    warps_per_cta: cutlass.Constexpr[int] = 8,
) -> None:
    _activate_quantize_kernel(
        gate, up, weights, out_codes, out_scales, num_tokens,
        intermediate, num_k_atoms, clamp, apply_weight, warps_per_cta,
    ).launch(
        grid=[num_ctas, 1, 1], block=[warps_per_cta * 32, 1, 1], stream=stream
    )
