# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused bf16 -> NVFP4 quantization, callable from inside a larger kernel.

Placement rationale
-------------------

The v2 plan proposed folding quantization into the *pull* path, converting
between the dispatch bounce buffer and the token pool.  Measurement says do it
on the **source** side instead, before the dispatch barrier:

* pulling bf16 would put 4x the bytes on the wire (NVFP4 is 0.5 B/elem vs
  2 B/elem), and
* every token is pulled by ~``top_k * (world-1)/world`` distinct peers, so
  quantizing on the consumer side redoes the same work ~3.3x at the reference
  shape.

Quantizing locally costs one pass over ``num_tokens x hidden`` bf16 (1.4 MB at
the reference shape, ~0.3 us) and lands inside the dispatch count-exchange
window, which is otherwise dead time.  So the fused version is strictly cheaper
than both the separate staging launch and the plan's pull-side variant.

Parallel decomposition
----------------------

One warp per token row; lane ``l`` owns scale blocks ``l, l+32, ...``.  A block
is 16 bf16 = 32 contiguous bytes, so a warp's 32 lanes cover 1024 contiguous
bytes per step -- fully coalesced loads, and the 8-byte packed FP4 results are
likewise contiguous across lanes.  Block amax is therefore register-local: no
cross-lane reduction anywhere in the hot loop.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32

from .sf_layout import SF_ATOM_BLOCKS
from .types import NVFP4_BLOCK

# Largest magnitude representable in FP4 E2M1; the block scale normalizes to it.
FP4_MAX = 6.0
_RCP_FP4_MAX = 1.0 / FP4_MAX
# Multiplying by this then min-ing against 1.0 turns "scale > 0" into a branch-
# free 0/1 mask: any positive normal E4M3 (min ~1.95e-3) saturates it to 1.0,
# while an exact zero stays 0.  Used to force all-zero blocks to encode as zero
# instead of NaN from a reciprocal of 0.
_ZERO_MASK_GAIN = 1.0e30
_FP32_MAX = 3.402823466e38


@cute.jit
def quantize_row_range(
    activation: cute.Tensor,  # (tokens, hidden) bf16
    out_codes: cute.Tensor,  # (tokens, hidden) Float4E2M1FN
    out_scales: cute.Tensor,  # flat Float8E4M3FN, indexed via sf_layout
    norm_const: Float32,
    *,
    hidden: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    first_token: Int32,
    token_limit: Int32,
    token_stride: Int32,
    lane_idx: Int32,
) -> None:
    """Quantize ``activation[t]`` for ``t`` in a strided token range.

    The caller supplies the stride so this composes with whatever warp
    decomposition the enclosing kernel already has: kernel A hands it the
    dispatch warps' grid-wide stride, the standalone test kernel hands it one
    warp per row.

    ``out_scales`` is a flat E4M3 view; the swizzled position of each block
    scale is computed here so callers never have to know the atom layout.
    """
    num_blocks: cutlass.Constexpr[int] = hidden // NVFP4_BLOCK
    # Blocks per lane, rounded up: the tail iteration is predicated below.
    blocks_per_lane: cutlass.Constexpr[int] = (num_blocks + 31) // 32

    # FP4 has no scalar conversion in the DSL -- it only exists as a vector
    # (TensorSSA) cast -- so the whole block moves through register tensors
    # rather than element assignments.  That is also what we want for codegen:
    # one 32 B load and one 8 B store per block instead of 16 scalar accesses.
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float4E2M1FN,
        num_bits_per_copy=NVFP4_BLOCK * 4,
    )

    token = first_token
    while token < token_limit:
        src_blocks = cute.zipped_divide(activation[token, None], (NVFP4_BLOCK,))
        dst_blocks = cute.zipped_divide(out_codes[token, None], (NVFP4_BLOCK,))

        for step in cutlass.range_constexpr(blocks_per_lane):
            block = lane_idx + Int32(step * 32)
            if block < Int32(num_blocks):
                raw = cute.make_rmem_tensor((NVFP4_BLOCK,), cutlass.BFloat16)
                cute.copy(load_atom, src_blocks[(None,), (block,)], raw)
                vals = cute.make_rmem_tensor((NVFP4_BLOCK,), Float32)
                vals.store(raw.load().to(Float32))

                absmax = Float32(0.0)
                for i in cutlass.range_constexpr(NVFP4_BLOCK):
                    absmax = cute.arch.fmax(
                        absmax, cute.arch.fmax(vals[i], -vals[i])
                    )

                # Stored scale: amax / FP4_MAX, repositioned by norm_const and
                # rounded to E4M3.  Reading it back as f32 is what the encode
                # step divides by, so the encoding stays self-consistent with
                # what a dequantizing consumer reconstructs.
                scale_e4m3 = (
                    absmax * Float32(_RCP_FP4_MAX) * norm_const
                ).to(cutlass.Float8E4M3FN)
                scale = Float32(scale_e4m3)

                encode = cute.arch.fmin(
                    norm_const * cute.arch.rcp_approx(scale), Float32(_FP32_MAX)
                )
                encode = encode * cute.arch.fmin(
                    scale * Float32(_ZERO_MASK_GAIN), Float32(1.0)
                )

                scaled = cute.make_rmem_tensor((NVFP4_BLOCK,), Float32)
                for i in cutlass.range_constexpr(NVFP4_BLOCK):
                    scaled[i] = vals[i] * encode
                codes = cute.make_rmem_tensor((NVFP4_BLOCK,), cutlass.Float4E2M1FN)
                codes.store(scaled.load().to(cutlass.Float4E2M1FN))
                cute.copy(
                    store_atom,
                    codes,
                    _assume_align(dst_blocks[(None,), (block,)], NVFP4_BLOCK // 2),
                )

                out_scales[
                    scale_element_offset(token, block, num_k_atoms=num_k_atoms)
                ] = scale_e4m3
        token = token + token_stride


@cute.jit
def _assume_align(tensor: cute.Tensor, align_bytes: cutlass.Constexpr[int]):
    """Re-tag a tensor's pointer with a stronger alignment.

    Each FP4 block starts at ``block * 8`` bytes into a row whose own start is
    at least 16 B aligned, so the 64-bit vector store below is legal -- but
    that only follows from the block index, which cute cannot see.
    """
    ptr = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(
            ptr.dtype, ptr.toint(), ptr.memspace, assumed_align=align_bytes
        ),
        tensor.layout,
    )


@cute.jit
def scale_element_offset(
    token_row: Int32,
    block: Int32,
    *,
    num_k_atoms: cutlass.Constexpr[int],
) -> Int32:
    """Flat E4M3 index of one block scale, in the swizzled atom layout.

    Mirrors :func:`..sf_layout.byte_in_atom` exactly -- see that module for the
    ABI.  Kept in byte (E4M3 element) units rather than the int32 units the
    host helper uses, because a single lane here owns one block, not four.
    """
    from .types import SF_ATOM_ROWS

    row_block = token_row // Int32(SF_ATOM_ROWS)
    token_in_atom = token_row % Int32(SF_ATOM_ROWS)
    k_atom = block // Int32(SF_ATOM_BLOCKS)
    k_bank = block % Int32(SF_ATOM_BLOCKS)
    atom_index = row_block * Int32(num_k_atoms) + k_atom
    return (
        atom_index * Int32(SF_ATOM_ROWS * SF_ATOM_BLOCKS)
        + (token_in_atom % Int32(32)) * Int32(16)
        + (token_in_atom // Int32(32)) * Int32(4)
        + k_bank
    )


@cute.kernel
def _quantize_kernel(
    activation: cute.Tensor,
    out_codes: cute.Tensor,
    out_scales: cute.Tensor,
    norm_const: Float32,
    num_tokens: Int32,
    hidden: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    warps_per_cta: cutlass.Constexpr[int],
) -> None:
    """Standalone launcher shape: one warp per token, grid-strided.

    Exists so the encoding can be tested against the torch oracle in isolation,
    before it is embedded in kernel A where a failure would be indistinguishable
    from a dispatch or GEMM bug.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    warp_in_cta = tidx // Int32(32)
    lane_idx = tidx % Int32(32)
    first = bidx * Int32(warps_per_cta) + warp_in_cta
    stride = gdim * Int32(warps_per_cta)

    quantize_row_range(
        activation,
        out_codes,
        out_scales,
        norm_const,
        hidden=hidden,
        num_k_atoms=num_k_atoms,
        first_token=first,
        token_limit=num_tokens,
        token_stride=stride,
        lane_idx=lane_idx,
    )


@cute.jit
def quantize_launch(
    activation: cute.Tensor,
    out_codes: cute.Tensor,
    out_scales: cute.Tensor,
    norm_const: Float32,
    num_tokens: Int32,
    stream,
    *,
    hidden: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int] = 64,
    warps_per_cta: cutlass.Constexpr[int] = 8,
) -> None:
    _quantize_kernel(
        activation,
        out_codes,
        out_scales,
        norm_const,
        num_tokens,
        hidden,
        num_k_atoms,
        warps_per_cta,
    ).launch(
        grid=[num_ctas, 1, 1],
        block=[warps_per_cta * 32, 1, 1],
        stream=stream,
    )
