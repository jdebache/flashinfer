# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""FC2: the down projection, emitting token-major rows ready for combine.

Nothing here is fused arithmetic -- the routing weight was already folded into
FC1, which is exactly the payoff of putting it there: FC2 is a plain grouped
GEMM and the combine that follows is a sum.  So this is
:mod:`.gemm_kernel` with ``acc_stages=1`` and one epilogue whose only job is a
*layout* change.

Why the epilogue is not `epilogue_plain`
----------------------------------------

Under swap-AB the accumulator is ``(channel, token)`` and TMEM hands each
thread one channel across many tokens.  Writing that out directly gives a
``(hidden, pool_rows)`` matrix, which is the wrong way round for everything
downstream: combine sends whole token *rows* back to their source rank, and a
row of that matrix is a column in memory -- one 4-byte element per token,
strided by the whole pool.

So the accumulator goes through the same smem transpose the FC1 epilogue uses,
and each thread comes back owning one token and a contiguous run of channels,
which it stores as one 16 B bf16 vector.  bf16 rather than fp32 because these
rows go on the wire next; the destination accumulates them in fp32.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Float32, Int32

from .gemm_kernel import EPI_STAGE_BARRIER, launch_grouped_gemm

# Token width of one epilogue pass; matches FC1 so the two kernels size their
# staging buffer identically.
FC2_EPI_N = 32
# bf16 channels one thread stores at once: 8 * 2 B = one 16 B vector store.
_STORE_VEC = 8
_EPI_THREADS = 128


def stage_floats(cta_tile_m: int, epi_n: int = FC2_EPI_N) -> int:
    """fp32 words of staging for one epilogue pass; see :func:`..fc1.stage_floats`."""
    return cta_tile_m * (epi_n + 1)


def transpose_tile_to_rows(
    tTR_accs,
    tiled_t2r,
    thr_t2r,
    tidx,
    stage_smem,
    abs_tb,
    store,
    *,
    cta_tile_m: int,
    tile_tokens: int,
    epi_n: int,
):
    """Turn one accumulator tile into token-major bf16 vectors and hand them off.

    ``store(token_row, chunk, vals)`` receives one thread's 8 bf16 channels
    for one pool row; where they go is the caller's business.  That split is
    what lets FC2 write a local matrix and the combine variant scatter the
    same bytes straight into a peer's buffer without a second transpose.

    A plain function, not ``@cute.jit``: it inlines into the caller's region at
    trace time, so ``store`` can close over dynamic values.
    """
    epi_n_sub = tile_tokens // epi_n
    chunks = cta_tile_m // _STORE_VEC
    steps = (epi_n * chunks) // _EPI_THREADS
    tok_base = abs_tb * Int32(tile_tokens)

    stage = cute.make_tensor(
        stage_smem.iterator,
        cute.make_layout((cta_tile_m, epi_n), stride=(epi_n + 1, 1)),
    )
    tSt = thr_t2r.partition_D(cute.flat_divide(stage, (cta_tile_m, epi_n)))[
        (None, None, None, 0, 0)
    ]
    frag = cute.make_rmem_tensor(tSt.shape, Float32)
    bar = pipeline.NamedBarrier(barrier_id=EPI_STAGE_BARRIER, num_threads=_EPI_THREADS)

    for sub in range(epi_n_sub):
        cute.copy(tiled_t2r, tTR_accs[0][(None, None, None, 0, sub)], frag)
        cute.arch.fence_view_async_tmem_load()
        # Brackets the staging buffer, which is reused across subtiles.
        bar.arrive_and_wait()
        cute.autovec_copy(frag, tSt)
        bar.arrive_and_wait()

        for step in range(steps):
            unit = tidx + Int32(step * _EPI_THREADS)
            # Consecutive lanes take consecutive tokens, so the strided reads
            # from the staging buffer stay conflict-free.
            chunk = unit // Int32(epi_n)
            tok_local = unit % Int32(epi_n)
            vals = cute.make_rmem_tensor((_STORE_VEC,), cutlass.BFloat16)
            base = chunk * Int32(_STORE_VEC)
            for i in range(_STORE_VEC):
                vals[i] = stage[(base + Int32(i), tok_local)].to(cutlass.BFloat16)
            store(tok_base + Int32(sub * epi_n) + tok_local, chunk, vals)


def store_vector(row_tensor, slot: Int32, vals) -> None:
    """Store 8 bf16 channels at ``slot`` of a token row, as one 16 B write."""
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=_STORE_VEC * 16,
    )
    dst = cute.zipped_divide(row_tensor, (_STORE_VEC,))
    ptr = dst[(None,), (slot,)].iterator
    aligned = cute.make_tensor(
        cute.make_ptr(ptr.dtype, ptr.toint(), ptr.memspace, assumed_align=16),
        dst[(None,), (slot,)].layout,
    )
    cute.copy(atom, vals, aligned)


@cute.jit
def epilogue_fc2(
    tTR_accs,
    tiled_t2r,
    thr_t2r,
    ch_block: Int32,
    abs_tb: Int32,
    mma_v: Int32,
    tidx: Int32,
    epi_args,  # (out_rows,)
    stage_smem,
    *,
    acc_stages: cutlass.Constexpr[int],
    channel_blocks: cutlass.Constexpr[int],
    cta_tile_m: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
    epi_n: cutlass.Constexpr[int],
    cta_per_mma: cutlass.Constexpr[int],
) -> None:
    """Transpose one tile's accumulator into bf16 token rows.

    ``channel_blocks`` is unused: it exists in the shared epilogue signature
    for the plain epilogue's benefit (see :func:`..fc1.epilogue_fc1`).
    """
    out_rows = epi_args[0]
    ch_base = (ch_block * Int32(cta_per_mma) + mma_v) * Int32(cta_tile_m)

    def store(token_row, chunk, vals):
        store_vector(
            out_rows[token_row, None], ch_base // Int32(_STORE_VEC) + chunk, vals
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
def launch_fc2(
    w2: cute.Tensor,  # (experts, hidden, I) Float4E2M1FN
    fc1_out: cute.Tensor,  # (pool_rows, I) Float4E2M1FN
    sf_w2: cute.Tensor,
    sf_fc1_out: cute.Tensor,
    out_rows: cute.Tensor,  # (pool_rows, hidden) bf16
    prefix: cute.Tensor,
    stream,
    *,
    num_experts: cutlass.Constexpr[int],
    intermediate: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    pool_rows: cutlass.Constexpr[int],
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
        (out_rows,),
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
        epilogue=epilogue_fc2,
        epi_n=FC2_EPI_N,
        epi_smem_floats=stage_floats(cta_tile_m, FC2_EPI_N),
        use_pdl=use_pdl,
    )
