# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Combine: return each expert's rows to the rank the token came from.

Atomic-free, by the same ownership rule as dispatch
---------------------------------------------------

A token is processed by ``top_k`` experts, so ``top_k`` partial results must
meet somewhere and be summed.  The obvious way is a remote atomic add into the
source's output; instead each result is written to its *own* slot, because the
pair ``(token, slot)`` already identifies one contribution uniquely and the
source rank owns exactly one such slot per pair.  So the push is a plain store
-- no remote read-modify-write, and no contention between the ranks pushing
back to the same token.

Two consequences worth having:

* the sum is **deterministic**.  It runs in slot order on the source rank, so
  the output does not depend on which rank happened to answer first;
* the routing weight is already folded in (FC1), so this really is a sum and
  nothing needs to be rescaled here.

The cost is the landing buffer: ``top_k * max_tokens * hidden`` bf16.  The
atomic variant would need only ``max_tokens * hidden`` but in fp32, zeroed
every launch, with ``top_k`` remote read-modify-writes per element.

Fused into FC2, not a pass of its own
-------------------------------------

The push is the FC2 epilogue's store: :func:`epilogue_fc2_combine` reuses the
transpose that FC2 needs anyway and sends the bytes straight to the peer,
which removes an entire local output buffer and the read-back that a separate
combine kernel would cost.  Only the final sum is its own launch, and it runs
on the source rank after the barrier.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32, Int64

from .dispatch import peer_view
from .fc2 import (
    FC2_EPI_N,
    stage_floats,
    store_vector,
    transpose_tile_to_rows,
)
from .gemm_kernel import launch_grouped_gemm

_STORE_VEC = 8


@cute.jit
def epilogue_fc2_combine(
    tTR_accs,
    tiled_t2r,
    thr_t2r,
    ch_block: Int32,
    abs_tb: Int32,
    mma_v: Int32,
    tidx: Int32,
    epi_args,  # (combine_buf, pool_src, peer_offset)
    stage_smem,
    *,
    acc_stages: cutlass.Constexpr[int],
    channel_blocks: cutlass.Constexpr[int],
    cta_tile_m: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
    epi_n: cutlass.Constexpr[int],
    cta_per_mma: cutlass.Constexpr[int],
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
) -> None:
    """FC2's epilogue, scattering each row into its source rank's slot.

    ``channel_blocks`` is unused; see :func:`..fc1.epilogue_fc1` for why the
    shared epilogue signature carries it.
    """
    combine_buf, pool_src, peer_offset = epi_args[0], epi_args[1], epi_args[2]
    ch_base = (ch_block * Int32(cta_per_mma) + mma_v) * Int32(cta_tile_m)
    pairs: cutlass.Constexpr[int] = max_tokens * top_k

    def store(token_row, chunk, vals):
        packed = pool_src[token_row]
        # Padding rows exist only to round a segment up to a whole tile; they
        # have no destination, and writing them would corrupt a real slot.
        if packed >= Int64(0):
            src_rank = Int32(packed // Int64(pairs))
            rest = Int32(packed % Int64(pairs))
            token = rest // Int32(top_k)
            slot = rest % Int32(top_k)
            remote = peer_view(
                combine_buf,
                peer_offset[src_rank],
                combine_buf.layout,
                cutlass.BFloat16,
            )
            store_vector(
                remote[slot * Int32(max_tokens) + token, None],
                ch_base // Int32(_STORE_VEC) + chunk,
                vals,
            )

    transpose_tile_to_rows(
        tTR_accs,
        tiled_t2r,
        thr_t2r,
        tidx,
        stage_smem,
        abs_tb,
        store,
        cta_tile_m=cta_tile_m,
        tile_tokens=tile_tokens,
        epi_n=epi_n,
    )


@cute.jit
def launch_fc2_combine(
    w2: cute.Tensor,
    fc1_out: cute.Tensor,
    sf_w2: cute.Tensor,
    sf_fc1_out: cute.Tensor,
    combine_buf: cute.Tensor,  # (top_k * max_tokens, hidden) bf16, symmetric
    pool_src: cute.Tensor,  # (pool_rows,) int64 provenance from dispatch
    peer_offset: cute.Tensor,  # (world,) int64
    prefix: cute.Tensor,
    stream,
    *,
    num_experts: cutlass.Constexpr[int],
    intermediate: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    pool_rows: cutlass.Constexpr[int],
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    mma_m: cutlass.Constexpr[int] = 256,
    mma_n: cutlass.Constexpr[int] = 128,
    cluster_m: cutlass.Constexpr[int] = 2,
    two_cta: cutlass.Constexpr[bool] = True,
    num_a_stages: cutlass.Constexpr[int] = 4,
    num_b_stages: cutlass.Constexpr[int] = 3,
    num_clusters: cutlass.Constexpr[int] = 8,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    cta_tile_m: cutlass.Constexpr[int] = mma_m // (2 if two_cta else 1)
    launch_grouped_gemm(
        w2,
        fc1_out,
        sf_w2,
        sf_fc1_out,
        (combine_buf, pool_src, peer_offset),
        prefix,
        stream,
        num_experts=num_experts,
        out_channels=hidden,
        pool_rows=pool_rows,
        k=intermediate,
        mma_m=mma_m,
        mma_n=mma_n,
        cluster_m=cluster_m,
        two_cta=two_cta,
        num_a_stages=num_a_stages,
        num_b_stages=num_b_stages,
        num_clusters=num_clusters,
        acc_stages=1,
        epilogue=functools.partial(
            epilogue_fc2_combine, max_tokens=max_tokens, top_k=top_k
        ),
        epi_n=FC2_EPI_N,
        epi_smem_floats=stage_floats(cta_tile_m, FC2_EPI_N),
        use_pdl=use_pdl,
    )


# --------------------------------------------------------------------------
# reduce: sum the slots a rank's own tokens came back into
# --------------------------------------------------------------------------


@cute.kernel
def _reduce_kernel(
    combine_buf: cute.Tensor,  # (top_k * max_tokens, hidden) bf16
    topk_ids: cute.Tensor,  # (num_tokens, top_k) int32
    out: cute.Tensor,  # (num_tokens, hidden) bf16
    num_tokens: Int32,
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    chunks: cutlass.Constexpr[int] = hidden // _STORE_VEC
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=_STORE_VEC * 16,
    )

    unit = bidx * Int32(threads) + tidx
    stride = gdim * Int32(threads)
    limit = num_tokens * Int32(chunks)
    while unit < limit:
        token = unit // Int32(chunks)
        chunk = unit % Int32(chunks)

        acc = cute.make_rmem_tensor((_STORE_VEC,), Float32)
        for i in cutlass.range_constexpr(_STORE_VEC):
            acc[i] = Float32(0.0)

        # Fixed slot order, so the sum is bit-reproducible regardless of which
        # rank answered first.  Slots whose expert was invalid were never
        # written and must not be read -- the buffer is not cleared between
        # launches, so stale bytes would otherwise be summed in.
        for slot in cutlass.range_constexpr(top_k):
            expert = topk_ids[token, slot]
            if expert >= Int32(0) and expert < Int32(num_experts):
                row = Int32(slot * max_tokens) + token
                part = cute.make_rmem_tensor((_STORE_VEC,), cutlass.BFloat16)
                src = cute.zipped_divide(combine_buf[row, None], (_STORE_VEC,))
                cute.copy(load_atom, src[(None,), (chunk,)], part)
                for i in cutlass.range_constexpr(_STORE_VEC):
                    acc[i] += Float32(part[i])

        vals = cute.make_rmem_tensor((_STORE_VEC,), cutlass.BFloat16)
        for i in cutlass.range_constexpr(_STORE_VEC):
            vals[i] = acc[i].to(cutlass.BFloat16)
        store_vector(out[token, None], chunk, vals)
        unit += stride


@cute.jit
def combine_reduce(
    combine_buf: cute.Tensor,
    topk_ids: cute.Tensor,
    out: cute.Tensor,
    num_tokens: Int32,
    stream,
    *,
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int] = 64,
    threads: cutlass.Constexpr[int] = 256,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(hidden % _STORE_VEC != 0):
        raise ValueError(
            f"hidden ({hidden}) must be a multiple of {_STORE_VEC} for the "
            "vectorized combine reduction"
        )
    _reduce_kernel(
        combine_buf,
        topk_ids,
        out,
        num_tokens,
        max_tokens,
        top_k,
        hidden,
        num_experts,
        threads,
    ).launch(
        grid=[num_ctas, 1, 1],
        block=[threads, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )
