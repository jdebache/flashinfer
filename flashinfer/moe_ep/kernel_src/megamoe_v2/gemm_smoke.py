# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Smallest complete block-scaled tcgen05 GEMM, as a learning/validation vehicle.

One CTA, one output tile, no cluster, no 2-CTA MMA, no grouping, no fusion::

    C[m, n] = sum_k  A[m, k] * SFA[m, k/16] * B[n, k] * SFB[n, k/16]

That is deliberately the *whole* mainloop and nothing else, so when it matches
torch we know the five moving parts are right -- smem staging, the A/B
pipelines, the smem->TMEM scale copy, the UMMA issue sequence, and the TMEM
readback -- before any of them are entangled with dispatch, grouping, SwiGLU or
the epilogue quantizer.

Everything here is exercised by ``tests/moe_ep/test_megamoe_v2_gemm.py``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Int32

from .gemm import MMA_TILE_K
from .types import NVFP4_BLOCK

# One warp each for the two producers and the MMA; one warp group for the
# epilogue.  Same role numbering as the real kernels so the mental model
# carries over.
_EPI_WARPS = (0, 1, 2, 3)
_MMA_WARP = 4
_TMA_A_WARP = 5
_TMA_B_WARP = 6
_THREADS = 32 * 8

_TMEM_ALLOC_BARRIER = 1

# SM100 tensor-memory capacity, in 32-bit columns.
_TMEM_CAPACITY_COLS = 512
# UMMA instructions per k-tile (see gemm.MMA_TILE_K).
_MMA_INST_TILE_K = 4


@cute.jit
def smoke_gemm(
    a: cute.Tensor,  # (M, K) Float4E2M1FN, K-major
    b: cute.Tensor,  # (N, K) Float4E2M1FN, K-major
    sfa: cute.Tensor,  # (M, K/16) Float8E4M3FN
    sfb: cute.Tensor,  # (N, K/16) Float8E4M3FN
    c: cute.Tensor,  # (M, N) Float32
    stream,
    *,
    m: cutlass.Constexpr[int],
    n: cutlass.Constexpr[int],
    k: cutlass.Constexpr[int],
    num_stages: cutlass.Constexpr[int] = 3,
):
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        sm100_utils.OperandMajorMode.K,
        sm100_utils.OperandMajorMode.K,
        cutlass.Float8E4M3FN,
        NVFP4_BLOCK,
        tcgen05.CtaGroup.ONE,
        (m, n),
    )
    # One k-tile is MMA_TILE_K wide; the full reduction is k_tiles of them.
    mma_tiler = (m, n, MMA_TILE_K)
    k_tiles: cutlass.Constexpr[int] = k // MMA_TILE_K

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler, cutlass.Float4E2M1FN, num_stages
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler, cutlass.Float4E2M1FN, num_stages
    )
    sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, mma_tiler, NVFP4_BLOCK, num_stages
    )
    sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, mma_tiler, NVFP4_BLOCK, num_stages
    )

    # TMA views are (M, K, L): the trailing batch mode is where a grouped GEMM
    # puts the expert index.  This kernel has one group, so L = 1 with stride 0
    # -- but the rank has to be there or the tiler/coord profiles will not
    # match in local_tile below.
    a3 = cute.make_tensor(a.iterator, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
    b3 = cute.make_tensor(b.iterator, cute.make_layout((n, k, 1), stride=(k, 1, 0)))

    # Scale-factor gmem tensors must be presented in the atom-tiled shape the
    # TMA descriptor expects, not as the flat (rows, blocks) matrix they are in
    # memory.  This is the host-side counterpart of the swizzle in sf_layout.
    sfa_tiled = cute.make_tensor(
        sfa.iterator,
        blockscaled_utils.tile_atom_to_shape_SF((m, k, 1), NVFP4_BLOCK),
    )
    sfb_tiled = cute.make_tensor(
        sfb.iterator,
        blockscaled_utils.tile_atom_to_shape_SF((n, k, 1), NVFP4_BLOCK),
    )

    a_atom, a_view = cute.nvgpu.make_tiled_tma_atom_A(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        a3,
        cute.slice_(a_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        (1, 1, 1),
    )
    b_atom, b_view = cute.nvgpu.make_tiled_tma_atom_B(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        b3,
        cute.slice_(b_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        (1, 1, 1),
    )
    sfa_atom, sfa_view = cute.nvgpu.make_tiled_tma_atom_A(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        sfa_tiled,
        cute.slice_(sfa_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        (1, 1, 1),
    )
    sfb_atom, sfb_view = cute.nvgpu.make_tiled_tma_atom_B(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        sfb_tiled,
        cute.slice_(sfb_smem_layout, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        (1, 1, 1),
    )

    # Bytes one stage's TMA must deliver before its barrier flips.  Getting
    # this wrong deadlocks rather than corrupts, which is at least loud.
    ab_bytes = (
        cute.size_in_bytes(
            cutlass.Float4E2M1FN, cute.slice_(a_smem_layout, (None, None, None, 0))
        )
        + cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.slice_(sfa_smem_layout, (None, None, None, 0))
        )
    )
    b_bytes = (
        cute.size_in_bytes(
            cutlass.Float4E2M1FN, cute.slice_(b_smem_layout, (None, None, None, 0))
        )
        + cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.slice_(sfb_smem_layout, (None, None, None, 0))
        )
    )

    _smoke_kernel(
        tiled_mma,
        a_atom, a_view, b_atom, b_view,
        sfa_atom, sfa_view, sfb_atom, sfb_view,
        c,
        a_smem_layout, b_smem_layout, sfa_smem_layout, sfb_smem_layout,
        mma_tiler,
        ab_bytes,
        b_bytes,
        num_stages,
        k_tiles,
    ).launch(grid=[1, 1, 1], block=[_THREADS, 1, 1], stream=stream)


@cute.kernel
def _smoke_kernel(
    tiled_mma: cute.TiledMma,
    a_atom, a_view, b_atom, b_view,
    sfa_atom, sfa_view, sfb_atom, sfb_view,
    c: cute.Tensor,
    a_smem_layout, b_smem_layout, sfa_smem_layout, sfb_smem_layout,
    mma_tiler: cutlass.Constexpr,
    a_tx_bytes: cutlass.Constexpr[int],
    b_tx_bytes: cutlass.Constexpr[int],
    num_stages: cutlass.Constexpr[int],
    k_tiles: cutlass.Constexpr[int],
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    @cute.struct
    class Shared:
        a_mbar: cute.struct.MemRange[cutlass.Int64, num_stages * 2]
        b_mbar: cute.struct.MemRange[cutlass.Int64, num_stages * 2]
        acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_holding: cutlass.Int32

    smem = utils.SmemAllocator()
    storage = smem.allocate(Shared)

    sA = smem.allocate_tensor(
        cutlass.Float4E2M1FN, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner
    )
    sB = smem.allocate_tensor(
        cutlass.Float4E2M1FN, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
    )
    sSFA = smem.allocate_tensor(cutlass.Float8E4M3FN, sfa_smem_layout, 128)
    sSFB = smem.allocate_tensor(cutlass.Float8E4M3FN, sfb_smem_layout, 128)

    # --- pipelines -------------------------------------------------------
    a_pipe = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.a_mbar.data_ptr(),
        num_stages=num_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=a_tx_bytes,
    )
    b_pipe = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.b_mbar.data_ptr(),
        num_stages=num_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=b_tx_bytes,
    )
    acc_pipe = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar.data_ptr(),
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32 * len(_EPI_WARPS)
        ),
    )

    tmem_barrier = pipeline.NamedBarrier(
        barrier_id=_TMEM_ALLOC_BARRIER,
        num_threads=32 * len((_MMA_WARP, *_EPI_WARPS)),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding.ptr,
        barrier_for_retrieve=tmem_barrier,
        allocator_warp_id=_EPI_WARPS[0],
    )

    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    tAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    # TMEM holds the accumulator plus both scale-factor planes.
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma, mma_tiler, NVFP4_BLOCK,
        cute.slice_(sfa_smem_layout, (None, None, None, 0)),
    )
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma, mma_tiler, NVFP4_BLOCK,
        cute.slice_(sfb_smem_layout, (None, None, None, 0)),
    )
    # TMEM column budget.  A scale-factor plane occupies
    # (tile_mn / 32) * mma_inst_tile_k columns; the accumulator occupies one
    # column per output column.  The allocator only accepts a power-of-two
    # column count, and this kernel is one CTA with one accumulator stage, so
    # taking the whole 512-column capacity and carving it by hand is both
    # simplest and always legal (this is what the upstream kernel does too).
    num_acc_cols: cutlass.Constexpr[int] = mma_tiler[1]
    num_sfa_cols: cutlass.Constexpr[int] = (mma_tiler[0] // 32) * _MMA_INST_TILE_K
    num_sfb_cols: cutlass.Constexpr[int] = (
        cute.round_up(mma_tiler[1], 128) // 32
    ) * _MMA_INST_TILE_K
    num_tmem_cols: cutlass.Constexpr[int] = _TMEM_CAPACITY_COLS

    # allocate()/free() already restrict themselves to the allocator warp;
    # wrapping them in another traced `if` hides the allocator's own bookkeeping
    # from the tracer and its column count comes back as 0.
    tmem.allocate(num_tmem_cols)
    cute.arch.barrier()

    # --- producers -------------------------------------------------------
    if warp_idx == _TMA_A_WARP:
        gA = cute.local_tile(a_view, cute.slice_(mma_tiler, (None, 0, None)),
                             (None, None, None))
        gSFA = cute.local_tile(sfa_view, cute.slice_(mma_tiler, (None, 0, None)),
                               (None, None, None))
        thr = tiled_mma.get_slice(0)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            a_atom, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(thr.partition_A(gA), 0, 3),
        )
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            sfa_atom, 0, cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr.partition_A(gSFA), 0, 3),
        )
        # The scale tensors carry stride-0 broadcast modes on BOTH sides; both
        # must be filtered before the copy or the atom's rest-rank check fails.
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        producer = a_pipe.make_participants()[0]
        _stream_operand(
            a_atom, sfa_atom,
            tAgA[(None, 0, None, 0)], tAgSFA[(None, 0, None, 0)],
            tAsA, tAsSFA, producer, k_tiles=k_tiles,
        )

    if warp_idx == _TMA_B_WARP:
        gB = cute.local_tile(b_view, cute.slice_(mma_tiler, (0, None, None)),
                             (None, None, None))
        gSFB = cute.local_tile(sfb_view, cute.slice_(mma_tiler, (0, None, None)),
                               (None, None, None))
        thr = tiled_mma.get_slice(0)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            b_atom, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(thr.partition_B(gB), 0, 3),
        )
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            sfb_atom, 0, cute.make_layout(1),
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(thr.partition_B(gSFB), 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        producer = b_pipe.make_participants()[0]
        _stream_operand(
            b_atom, sfb_atom,
            tBgB[(None, 0, None, 0)], tBgSFB[(None, 0, None, 0)],
            tBsB, tBsSFB, producer, k_tiles=k_tiles,
        )

    # --- MMA -------------------------------------------------------------
    if warp_idx == _MMA_WARP:
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tAcc = cute.make_tensor(acc_ptr, tAcc_fake.layout)

        sfa_ptr = cute.recast_ptr(acc_ptr + num_acc_cols, dtype=cutlass.Float8E4M3FN)
        tSFA = cute.make_tensor(sfa_ptr, tCtSFA_layout)
        sfb_ptr = cute.recast_ptr(
            acc_ptr + num_acc_cols + num_sfa_cols,
            dtype=cutlass.Float8E4M3FN,
        )
        tSFB = cute.make_tensor(sfb_ptr, tCtSFB_layout)

        s2t_sfa, sSFA_s2t, tSFA_s2t = _make_s2t(sSFA, tSFA)
        s2t_sfb, sSFB_s2t, tSFB_s2t = _make_s2t(sSFB, tSFB)

        thr = tiled_mma.get_slice(0)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        a_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_stages
        )
        b_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_stages
        )
        acc_prod = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for _ in cutlass.range(k_tiles, unroll=1):
            a_pipe.consumer_wait(a_cons)
            b_pipe.consumer_wait(b_cons)

            cute.copy(s2t_sfa, sSFA_s2t[(None, None, None, None, a_cons.index)],
                      tSFA_s2t)
            cute.copy(s2t_sfb, sSFB_s2t[(None, None, None, None, b_cons.index)],
                      tSFB_s2t)

            for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                tiled_mma.set(tcgen05.Field.SFA, tSFA[(None, None, kb)].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tSFB[(None, None, kb)].iterator)
                cute.gemm(
                    tiled_mma, tAcc,
                    tCrA[(None, None, kb, a_cons.index)],
                    tCrB[(None, None, kb, b_cons.index)],
                    tAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            a_pipe.consumer_release(a_cons)
            b_pipe.consumer_release(b_cons)
            a_cons.advance()
            b_cons.advance()

        acc_pipe.producer_commit(acc_prod)

    # --- epilogue --------------------------------------------------------
    if warp_idx < len(_EPI_WARPS):
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tAcc = cute.make_tensor(acc_ptr, tAcc_fake.layout)

        acc_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        acc_pipe.consumer_wait(acc_cons)

        tidx, _, _ = cute.arch.thread_idx()

        # The epilogue reads TMEM in subtiles.  The whole 128-wide tile is too
        # much for one TMEM_LOAD, so pick a 64-column subtile and iterate: the
        # load op is selected *for* that subtile shape, and both the TMEM
        # source and the gmem destination are re-tiled by it so one thread
        # slice indexes both consistently.
        epi_tile = (mma_tiler[0], 64)
        epi_n: cutlass.Constexpr[int] = mma_tiler[1] // 64

        copy_t2r = sm100_utils.get_tmem_load_op(
            (mma_tiler[0], mma_tiler[1], mma_tiler[2]),
            utils.LayoutEnum.ROW_MAJOR,
            cutlass.Float32,
            cutlass.Float32,
            epi_tile,
            False,
        )
        # (EPI_M, EPI_N, M_SUB, N_SUB)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0)], epi_tile)
        tiled_t2r = tcgen05.make_tmem_copy(copy_t2r, tAcc_epi[(None, None, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)

        gC_epi = cute.flat_divide(c, epi_tile)
        tTR_acc = thr_t2r.partition_S(tAcc_epi)
        tTR_gc = thr_t2r.partition_D(gC_epi)

        frag = cute.make_rmem_tensor(
            tTR_gc[(None, None, None, 0, 0)].shape, cutlass.Float32
        )
        for sub in cutlass.range_constexpr(epi_n):
            cute.copy(tiled_t2r, tTR_acc[(None, None, None, 0, sub)], frag)
            # TMEM reads are async with respect to the register file; this
            # fence is what makes the loaded values observable before they are
            # stored back out.
            cute.arch.fence_view_async_tmem_load()
            cute.autovec_copy(frag, tTR_gc[(None, None, None, 0, sub)])

        acc_pipe.consumer_release(acc_cons)
        cute.arch.barrier()
        tmem.free(acc_ptr, num_tmem_cols)


@cute.jit
def _stream_operand(
    data_atom, sf_atom, g_data, g_sf, s_data, s_sf, producer,
    *, k_tiles: cutlass.Constexpr[int],
) -> None:
    """Acquire a stage, fire both TMAs into it, repeat.  No wait: the TMA
    engine signals the stage barrier itself, which is what lets this warp run
    ahead of the MMA by up to ``num_stages`` k-tiles."""
    producer.reset()
    peek = producer.try_acquire()
    for _ in cutlass.range(0, k_tiles, 1, unroll=1):
        handle = producer.acquire_and_advance(peek)
        peek = cutlass.Boolean(1)
        if handle.count + 1 < k_tiles:
            peek = producer.try_acquire()
        cute.copy(data_atom, g_data[(None, handle.count)],
                  s_data[(None, handle.index)], tma_bar_ptr=handle.barrier)
        cute.copy(sf_atom, g_sf[(None, handle.count)],
                  s_sf[(None, handle.index)], tma_bar_ptr=handle.barrier)
    producer.tail()


@cute.jit
def _make_s2t(s_sf: cute.Tensor, t_sf: cute.Tensor):
    """Build the smem->TMEM copy for one scale plane.

    Three steps that are all load-bearing:

    * ``filter_zeros`` drops the stride-0 broadcast modes the scale layouts
      carry (a block scale is shared by 16 elements), leaving the compact
      one-entry-per-block form the copy actually moves;
    * ``make_s2t_copy`` builds the tiled copy from the *TMEM* side, since that
      is what fixes the partitioning;
    * the smem source must then be turned into an S2T **descriptor** tensor --
      the copy reads an smem descriptor, not a plain address, and skipping this
      is a shape mismatch at trace time rather than a silent wrong answer.
    """
    s_compact = cute.filter_zeros(s_sf)
    t_compact = cute.filter_zeros(t_sf)
    s2t_atom = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E4M3FN
    )
    tiled = tcgen05.make_s2t_copy(s2t_atom, t_compact)
    thr = tiled.get_slice(0)
    src = tcgen05.get_s2t_smem_desc_tensor(tiled, thr.partition_S(s_compact))
    return tiled, src, thr.partition_D(t_compact)
