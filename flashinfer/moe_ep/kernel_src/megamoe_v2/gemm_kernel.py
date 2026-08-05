# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Grouped, persistent, 2-CTA block-scaled GEMM -- the real v2 mainloop.

Generalizes ``gemm_smoke`` three ways at once, each of which changes one thing:

**2-CTA MMA.**  ``mma_m`` spans a *pair* of CTAs (128 rows each).  A is split
across the pair along M; B is replicated to both, so both operands are TMA
multicast within the cluster.  Only the leader CTA (``v == 0``) issues the UMMA
and commits the accumulator -- but both CTAs' TMAs must land first, which is
what the cluster-scoped pipeline barriers arrange.

**Grouping.**  Weights are ``(expert, out_channels, K)`` and the expert index
rides the TMA view's trailing ``L`` mode, so switching expert is a coordinate
change, not a descriptor rewrite.  Tokens live in one shared pool addressed by
an absolute row, so B needs no L mode at all.

**Persistent tiles, and no scheduler warp.**  v1 needed a dedicated warp to
walk its stateful schedule and broadcast decoded tiles through an smem
pipeline.  Here ``schedule.decode_tile`` is a pure function of the tile index
and a small per-expert prefix array, so *every* warp recomputes it
independently for a handful of instructions.  That deletes a warp, an smem work
queue, and an entire pipeline -- and it is the direct payoff of making the
schedule order-free.

Warp roles (7 warps, 224 threads)::

    0-3  epilogue      4  MMA      5  TMA-A (weights)      6  TMA-B (tokens)

Two parameters make this the FC1 kernel as well as a plain GEMM
-----------------------------------------------------------------

``acc_stages`` is how many *output channel ranges* a tile computes at once,
into that many TMEM accumulators.  FC1 needs two, because SwiGLU pairs channel
``i`` of the gate half with channel ``i`` of the up half and those live
``intermediate / mma_m`` tiles apart -- a tile that owned only one of them
could not activate anything.  The two ranges share the token operand, so
TMA-B still streams each token tile exactly once: the k-loop consumes
``acc_stages`` A stages per B stage.

``epilogue`` is injected rather than switched on.  With ``acc_stages=1`` and
:func:`epilogue_plain` this is an ordinary grouped GEMM, which is what the
mainloop tests pin; FC1 supplies ``acc_stages=2`` and its own epilogue and
changes nothing else.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Int32

from .gemm import MMA_TILE_K
from .types import NVFP4_BLOCK

_EPI_WARPS = (0, 1, 2, 3)
_MMA_WARP = 4
_TMA_A_WARP = 5
_TMA_B_WARP = 6
_NUM_WARPS = 7
_THREADS = 32 * _NUM_WARPS

_TMEM_ALLOC_BARRIER = 1
_EPI_DONE_BARRIER = 2
EPI_STAGE_BARRIER = 3
_PREFIX_BARRIER = 6
_TMEM_CAPACITY_COLS = 512
_MMA_INST_TILE_K = 4
# Epilogue TMEM readback subtile width; 128x64 is the widest a single
# TMEM_LOAD covers for fp32.
_EPI_N = 64


@cute.jit
def decode_tile_device(
    tile_index: Int32,
    prefix: cute.Tensor,  # smem int32, (num_experts + 1,) exclusive prefix
    *,
    num_experts: cutlass.Constexpr[int],
    channel_blocks: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
):
    """Device twin of :func:`..schedule.decode_tile`.

    Returns ``(expert, channel_block, token_block, first_token_block)``.  The
    channel axis factors out before the expert search, so this is one integer
    division plus a branch-free scan over at most ``num_experts`` smem words --
    cheap enough that every warp can afford to do it rather than being handed
    the answer.

    The scan is a linear count of ``prefix[e+1] <= global_token_block`` rather
    than a binary search: at 16-64 experts it is fewer instructions, it is
    perfectly warp-uniform, and it has no data-dependent branches.
    """
    global_tb = tile_index // Int32(channel_blocks)
    channel_block = tile_index % Int32(channel_blocks)

    expert = Int32(0)
    for e in cutlass.range_constexpr(num_experts):
        expert += Int32(1) if prefix[e + 1] <= global_tb else Int32(0)

    first_tb = prefix[expert]
    return expert, channel_block, global_tb - first_tb, first_tb


@cute.jit
def epilogue_plain(
    tTR_accs,  # one TMEM-partitioned accumulator per acc stage
    tiled_t2r,
    thr_t2r,
    ch_block: Int32,
    abs_tb: Int32,
    mma_v: Int32,
    tidx: Int32,
    epi_args,  # (c,)
    stage_smem,  # unused
    *,
    acc_stages: cutlass.Constexpr[int],
    channel_blocks: cutlass.Constexpr[int],
    cta_tile_m: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
    epi_n: cutlass.Constexpr[int],
    cta_per_mma: cutlass.Constexpr[int],
) -> None:
    """Write the raw accumulators out -- the identity epilogue.

    Kept as the default so that ``acc_stages=1`` reproduces a plain grouped
    GEMM exactly, which is what makes the mainloop independently testable.
    """
    c = epi_args[0]
    epi_n_sub: cutlass.Constexpr[int] = tile_tokens // epi_n
    for j in cutlass.range_constexpr(acc_stages):
        gC = cute.local_tile(
            c,
            (cta_tile_m, tile_tokens),
            (
                (ch_block + Int32(j * channel_blocks)) * Int32(cta_per_mma) + mma_v,
                abs_tb,
            ),
        )
        tTR_gc = thr_t2r.partition_D(cute.flat_divide(gC, (cta_tile_m, epi_n)))
        frag = cute.make_rmem_tensor(
            tTR_gc[(None, None, None, 0, 0)].shape, cutlass.Float32
        )
        for sub in cutlass.range_constexpr(epi_n_sub):
            cute.copy(tiled_t2r, tTR_accs[j][(None, None, None, 0, sub)], frag)
            cute.arch.fence_view_async_tmem_load()
            cute.autovec_copy(frag, tTR_gc[(None, None, None, 0, sub)])


@cute.kernel
def _grouped_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tiled_mma_b: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    a_atom,
    a_view,
    b_atom,
    b_view,
    sfa_atom,
    sfa_view,
    sfb_atom,
    sfb_view,
    epi_args,  # tuple of output tensors, consumed by `epilogue`
    prefix_gmem: cute.Tensor,  # (num_experts + 1,) int32
    a_smem_layout,
    b_smem_layout,
    sfa_smem_layout,
    sfb_smem_layout,
    cluster_vmnk: cute.Layout,
    cluster_sfb_vmnk: cute.Layout,
    mma_tiler: cutlass.Constexpr,
    a_tx_bytes: cutlass.Constexpr[int],
    b_tx_bytes: cutlass.Constexpr[int],
    num_a_stages: cutlass.Constexpr[int],
    num_b_stages: cutlass.Constexpr[int],
    k_tiles: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    channel_blocks: cutlass.Constexpr[int],
    two_cta: cutlass.Constexpr[bool],
    cluster_m: cutlass.Constexpr[int],
    acc_stages: cutlass.Constexpr[int],
    epilogue: cutlass.Constexpr,
    epi_n: cutlass.Constexpr[int],
    epi_smem_floats: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    dispatch_warps: cutlass.Constexpr[int],
    prologue: cutlass.Constexpr,
    wait_schedule: cutlass.Constexpr,
    wait_tokens: cutlass.Constexpr,
    coop_args,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, bidz = cute.arch.block_idx()
    gdim_x, _, gdim_z = cute.arch.grid_dim()

    cta_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    coord_vmnk = cluster_vmnk.get_flat_coord(cta_in_cluster)
    coord_sfb_vmnk = cluster_sfb_vmnk.get_flat_coord(cta_in_cluster)
    # Under a 2-CTA MMA the pair's leader owns the instruction issue.
    mma_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader = mma_v == 0

    @cute.struct
    class Shared:
        a_mbar: cute.struct.MemRange[cutlass.Int64, num_a_stages * 2]
        b_mbar: cute.struct.MemRange[cutlass.Int64, num_b_stages * 2]
        acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_dealloc_mbar: cutlass.Int64
        tmem_holding: cutlass.Int32
        prefix: cute.struct.MemRange[cutlass.Int32, num_experts + 1]

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
    stage_smem = None
    if cutlass.const_expr(epi_smem_floats > 0):
        stage_smem = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(epi_smem_floats), 16
        )

    prefix = cute.make_tensor(
        storage.prefix.data_ptr(), cute.make_layout(num_experts + 1)
    )

    a_pipe = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.a_mbar.data_ptr(),
        num_stages=num_a_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=a_tx_bytes,
        cta_layout_vmnk=cluster_vmnk,
        defer_sync=True,
    )
    b_pipe = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.b_mbar.data_ptr(),
        num_stages=num_b_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=b_tx_bytes,
        cta_layout_vmnk=cluster_vmnk,
        defer_sync=True,
    )
    acc_pipe = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar.data_ptr(),
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            32 * len(_EPI_WARPS) * (2 if two_cta else 1),
        ),
        cta_layout_vmnk=cluster_vmnk,
        defer_sync=True,
    )

    tmem_barrier = pipeline.NamedBarrier(
        barrier_id=_TMEM_ALLOC_BARRIER,
        num_threads=32 * len((_MMA_WARP, *_EPI_WARPS)),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding.ptr,
        barrier_for_retrieve=tmem_barrier,
        allocator_warp_id=_EPI_WARPS[0],
        is_two_cta=two_cta,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    tAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler,
        NVFP4_BLOCK,
        cute.slice_(sfa_smem_layout, (None, None, None, 0)),
    )
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler,
        NVFP4_BLOCK,
        cute.slice_(sfb_smem_layout, (None, None, None, 0)),
    )
    # One accumulator per channel range; SFA/SFB sit above all of them.  A
    # single set of SF columns is enough even with two accumulators: tcgen05
    # operations issued by one warp execute in order, so the S2T copy feeding
    # range j+1 cannot overtake range j's UMMA reads.
    num_acc_cols: cutlass.Constexpr[int] = mma_tiler[1] * acc_stages
    num_sfa_cols: cutlass.Constexpr[int] = (
        (mma_tiler[0] // (2 if two_cta else 1)) // 32
    ) * _MMA_INST_TILE_K

    # cluster_shape_mn is NOT optional here: without it the helper skips the
    # cluster arrive and downgrades the wait to a threadblock sync, so the
    # cluster-scoped mbarriers above are never jointly initialized and the
    # first cross-CTA TMA signal deadlocks.
    cluster_mn: cutlass.Constexpr = (cluster_m, 1)
    pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_mn, is_relaxed=True)
    tmem.allocate(_TMEM_CAPACITY_COLS)
    pipeline.pipeline_init_wait(cluster_shape_mn=cluster_mn)

    # Dispatch warps leave for the prologue here.  Everything above this point
    # is block-wide (the pipeline init is a barrier all threads must reach), so
    # nothing before it may block on work the dispatch warps have not done yet
    # -- that ordering is exactly what deadlocked the first version.
    if cutlass.const_expr(dispatch_warps > 0):
        if warp_idx >= Int32(_NUM_WARPS):
            prologue(coop_args, tidx - Int32(_NUM_WARPS * 32), bidx, bidz, gdim_z)

    # The prefix is staged by the GEMM warps only, and after the split, because
    # when dispatch is fused it does not exist until the prologue's plan step.
    if warp_idx < Int32(_NUM_WARPS):
        if cutlass.const_expr(dispatch_warps > 0):
            wait_schedule(coop_args)
        if tidx < Int32(num_experts + 1):
            prefix[tidx] = prefix_gmem[tidx]
        pipeline.NamedBarrier(
            barrier_id=_PREFIX_BARRIER, num_threads=32 * _NUM_WARPS
        ).arrive_and_wait()

    # The tile count is derived, not passed: the per-expert counts only
    # exist on device once dispatch has run, and `prefix[num_experts]` is
    # already the total token-block count.  Deriving it here keeps one
    # source of truth -- a host-side copy could disagree with the prefix
    # the warps actually walk.
    total_tiles = prefix[num_experts] * Int32(channel_blocks)

    tile_tokens: cutlass.Constexpr[int] = mma_tiler[1]
    # One cluster = one persistent worker.  Static striding needs no atomics
    # and no broadcast; see schedule.persistent_tile_indices for why that is
    # the right default at this shape.
    cluster_id = bidz
    num_clusters = gdim_z

    thr_mma = tiled_mma.get_slice(mma_v)

    # ---------------- TMA-A: weights ----------------
    if warp_idx == _TMA_A_WARP:
        if cutlass.const_expr(dispatch_warps == 0):
            cute.arch.warpgroup_reg_dealloc(40)
        a_mask = None
        sfa_mask = None
        if cutlass.const_expr(two_cta or cute.size(cluster_vmnk.shape[2]) > 1):
            a_mask = cpasync.create_tma_multicast_mask(
                cluster_vmnk, coord_vmnk, mcast_mode=2
            )
            sfa_mask = a_mask
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_vmnk, (0, 0, None, 0)).shape
        )
        producer = a_pipe.make_participants()[0]

        tile = cluster_id
        while tile < total_tiles:
            expert, ch_block, _tb, _first = decode_tile_device(
                tile,
                prefix,
                num_experts=num_experts,
                channel_blocks=channel_blocks,
                tile_tokens=tile_tokens,
            )
            gA = cute.local_tile(
                a_view, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
            )
            gSFA = cute.local_tile(
                sfa_view, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
            )
            tAsA, tAgA = cpasync.tma_partition(
                a_atom,
                coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tAsSFA, tAgSFA = cpasync.tma_partition(
                sfa_atom,
                coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sSFA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gSFA), 0, 3),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
            producer = _stream_a(
                a_atom,
                sfa_atom,
                tAgA,
                tAgSFA,
                tAsA,
                tAsSFA,
                producer,
                ch_block,
                expert,
                k_tiles=k_tiles,
                acc_stages=acc_stages,
                channel_blocks=channel_blocks,
                data_mask=a_mask,
                sf_mask=sfa_mask,
            )
            tile += num_clusters
        producer.tail()

    # ---------------- TMA-B: tokens ----------------
    if warp_idx == _TMA_B_WARP:
        if cutlass.const_expr(dispatch_warps == 0):
            cute.arch.warpgroup_reg_dealloc(40)
        b_mask = None
        sfb_mask = None
        if cutlass.const_expr(two_cta or cute.size(cluster_vmnk.shape[1]) > 1):
            b_mask = cpasync.create_tma_multicast_mask(
                cluster_vmnk, coord_vmnk, mcast_mode=1
            )
            sfb_mask = cpasync.create_tma_multicast_mask(
                cluster_sfb_vmnk, coord_sfb_vmnk, mcast_mode=1
            )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_vmnk, (0, None, 0, 0)).shape
        )
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # The whole point of PDL here: only the *token* operand comes from the
        # predecessor kernel, so only this warp waits on it.  TMA-A is already
        # streaming weights -- which is where the bandwidth goes -- while the
        # previous kernel drains.  The wait sits after the pipeline init so it
        # cannot hold the other warps at that block-wide rendezvous.
        if cutlass.const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()
        # SFB is partitioned by its OWN atom (CtaGroup.ONE, N rounded to 128).
        # Using the main atom here silently produces a different N split and
        # the tma_partition shape check fails.
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_v)
        producer = b_pipe.make_participants()[0]

        tile = cluster_id
        while tile < total_tiles:
            expert_b, _ch, token_block, first_tb = decode_tile_device(
                tile,
                prefix,
                num_experts=num_experts,
                channel_blocks=channel_blocks,
                tile_tokens=tile_tokens,
            )
            # Per expert, not per kernel: expert 0's rows land long before the
            # last expert's, and waiting on the whole pool would idle the MMA
            # -- which would in turn backpressure the weight stream after four
            # smem stages and undo the overlap entirely.
            if cutlass.const_expr(dispatch_warps > 0):
                wait_tokens(coop_args, expert_b)
            # The pool is one flat matrix; an expert's segment start is just
            # its prefix, so the absolute token tile is prefix + local index.
            abs_tb = first_tb + token_block
            gB = cute.local_tile(
                b_view, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
            )
            gSFB = cute.local_tile(
                sfb_view, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
            )
            tBsB, tBgB = cpasync.tma_partition(
                b_atom,
                coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            tBsSFB, tBgSFB = cpasync.tma_partition(
                sfb_atom,
                coord_sfb_vmnk[1],
                sfb_cta_layout,
                cute.group_modes(sSFB, 0, 3),
                cute.group_modes(thr_mma_sfb.partition_B(gSFB), 0, 3),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)
            producer = _stream(
                b_atom,
                sfb_atom,
                tBgB[(None, abs_tb, None, 0)],
                tBgSFB[(None, abs_tb, None, 0)],
                tBsB,
                tBsSFB,
                producer,
                k_tiles=k_tiles,
                data_mask=b_mask,
                sf_mask=sfb_mask,
            )
            tile += num_clusters
        producer.tail()

    # ---------------- MMA ----------------
    if warp_idx == _MMA_WARP:
        if cutlass.const_expr(dispatch_warps == 0):
            cute.arch.warpgroup_reg_dealloc(40)
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tAccs = tuple(
            cute.make_tensor(acc_ptr + j * mma_tiler[1], tAcc_fake.layout)
            for j in range(acc_stages)
        )
        # One TiledMma per channel range: `ACCUMULATE` is a field of the atom's
        # instruction descriptor, so ranges sharing an object would share the
        # first-k-tile "overwrite, do not accumulate" transition and the second
        # accumulator would start from whatever TMEM held.
        mma_a = tiled_mma
        mma_b = tiled_mma_b
        tSFA = cute.make_tensor(
            cute.recast_ptr(acc_ptr + num_acc_cols, dtype=cutlass.Float8E4M3FN),
            tCtSFA_layout,
        )
        tSFB = cute.make_tensor(
            cute.recast_ptr(
                acc_ptr + num_acc_cols + num_sfa_cols, dtype=cutlass.Float8E4M3FN
            ),
            tCtSFB_layout,
        )
        s2t_sfa, sSFA_s2t, tSFA_s2t = _make_s2t(sSFA, tSFA, two_cta)
        s2t_sfb, sSFB_s2t, tSFB_s2t = _make_s2t(sSFB, tSFB, two_cta)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        a_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_a_stages
        )
        b_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_b_stages
        )
        acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)

        # Under a 2-CTA MMA the pair has ONE consumer of the AB pipelines and
        # ONE producer of the accumulator: the leader.  The follower's MMA warp
        # must not wait on, release, or commit any of them -- its TMAs already
        # signalled the (cluster-scoped) stage barriers, and that is its whole
        # contribution.  Having the follower also wait was a hard deadlock.
        if is_leader:
            tile = cluster_id
            while tile < total_tiles:
                # One barrier guards every accumulator of the tile: the
                # epilogue needs them together (SwiGLU pairs them), so a
                # per-range stage would buy nothing and cost a second mbarrier.
                acc_pipe.producer_acquire(acc_prod)
                mma_a.set(tcgen05.Field.ACCUMULATE, False)
                if cutlass.const_expr(acc_stages > 1):
                    mma_b.set(tcgen05.Field.ACCUMULATE, False)
                for _ in cutlass.range(k_tiles, unroll=1):
                    b_pipe.consumer_wait(b_cons)
                    cute.copy(
                        s2t_sfb,
                        sSFB_s2t[(None, None, None, None, b_cons.index)],
                        tSFB_s2t,
                    )
                    # Ranges consume back to back against this one B stage,
                    # which is why TMA-B fetches each token tile once even
                    # though the tile computes `acc_stages` of them.
                    for j in cutlass.range_constexpr(acc_stages):
                        a_pipe.consumer_wait(a_cons)
                        cute.copy(
                            s2t_sfa,
                            sSFA_s2t[(None, None, None, None, a_cons.index)],
                            tSFA_s2t,
                        )
                        # Each range's descriptor must be reached through a
                        # plain named local, not a tuple element: the DSL's
                        # loop-carry analysis follows assignments to names, and
                        # a value mutated only via `mmas[j]` is yielded from a
                        # region it was not defined in.
                        if cutlass.const_expr(j == 0):
                            mma_a = _issue_range(
                                mma_a,
                                tAccs[0],
                                tCrA,
                                tCrB,
                                tSFA,
                                tSFB,
                                a_cons.index,
                                b_cons.index,
                            )
                        else:
                            mma_b = _issue_range(
                                mma_b,
                                tAccs[1],
                                tCrA,
                                tCrB,
                                tSFA,
                                tSFB,
                                a_cons.index,
                                b_cons.index,
                            )
                        a_pipe.consumer_release(a_cons)
                        a_cons.advance()
                    b_pipe.consumer_release(b_cons)
                    b_cons.advance()
                acc_pipe.producer_commit(acc_prod)
                acc_prod.advance()
                tile += num_clusters

    # ---------------- epilogue ----------------
    if warp_idx < len(_EPI_WARPS):
        if cutlass.const_expr(dispatch_warps == 0):
            cute.arch.warpgroup_reg_alloc(232)
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(cutlass.Float32)

        cta_per_mma: cutlass.Constexpr[int] = 2 if two_cta else 1
        cta_tile_m: cutlass.Constexpr[int] = mma_tiler[0] // cta_per_mma
        epi_tile = (cta_tile_m, epi_n)

        copy_t2r = sm100_utils.get_tmem_load_op(
            (cta_tile_m, mma_tiler[1], mma_tiler[2]),
            utils.LayoutEnum.ROW_MAJOR,
            cutlass.Float32,
            cutlass.Float32,
            epi_tile,
            two_cta,
        )
        tAcc_epis = tuple(
            cute.flat_divide(
                cute.make_tensor(acc_ptr + j * mma_tiler[1], tAcc_fake.layout)[
                    ((None, None), 0, 0)
                ],
                epi_tile,
            )
            for j in range(acc_stages)
        )
        tiled_t2r = tcgen05.make_tmem_copy(copy_t2r, tAcc_epis[0][(None, None, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_accs = tuple(thr_t2r.partition_S(t) for t in tAcc_epis)

        acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        tile = cluster_id
        while tile < total_tiles:
            _e, ch_block, token_block, first_tb = decode_tile_device(
                tile,
                prefix,
                num_experts=num_experts,
                channel_blocks=channel_blocks,
                tile_tokens=tile_tokens,
            )
            abs_tb = first_tb + token_block
            acc_pipe.consumer_wait(acc_cons)
            epilogue(
                tTR_accs,
                tiled_t2r,
                thr_t2r,
                ch_block,
                abs_tb,
                mma_v,
                tidx,
                epi_args,
                stage_smem,
                acc_stages=acc_stages,
                channel_blocks=channel_blocks,
                cta_tile_m=cta_tile_m,
                tile_tokens=tile_tokens,
                epi_n=epi_n,
                cta_per_mma=cta_per_mma,
            )
            acc_pipe.consumer_release(acc_cons)
            acc_cons.advance()
            tile += num_clusters

        # Rendezvous the epilogue warps only -- a plain block barrier here
        # would wait on the producer/MMA warps, which have already exited.
        pipeline.NamedBarrier(
            barrier_id=_EPI_DONE_BARRIER, num_threads=32 * len(_EPI_WARPS)
        ).arrive_and_wait()
        tmem.free(acc_ptr, _TMEM_CAPACITY_COLS)

        # Release this block's output before telling the dependent grid it may
        # start.  The fence is not optional: the trigger orders *launch*, and
        # the successor only sees these stores if they are released first.
        if cutlass.const_expr(use_pdl):
            cute.arch.fence_acq_rel_gpu()
            cute.arch.griddepcontrol_launch_dependents()


def _issue_range(mma, tAcc, tCrA, tCrB, tSFA, tSFB, a_index, b_index):
    """Issue one channel range's UMMAs for one k-tile, returning the descriptor.

    Deliberately a *plain* function, not ``@cute.jit``: it is inlined at trace
    time into the caller's region, so the descriptor mutations stay where the
    enclosing loop's carry analysis can see them.  A jit helper would put them
    in a child region and the loop's yield would not dominate its own operand.
    """
    for kb in range(cute.size(tCrA, mode=[2])):
        mma.set(tcgen05.Field.SFA, tSFA[(None, None, kb)].iterator)
        mma.set(tcgen05.Field.SFB, tSFB[(None, None, kb)].iterator)
        cute.gemm(
            mma,
            tAcc,
            tCrA[(None, None, kb, a_index)],
            tCrB[(None, None, kb, b_index)],
            tAcc,
        )
        mma.set(tcgen05.Field.ACCUMULATE, True)
    return mma


@cute.jit
def _stream_a(
    data_atom,
    sf_atom,
    g_data,
    g_sf,
    s_data,
    s_sf,
    producer,
    ch_block: Int32,
    expert: Int32,
    *,
    k_tiles: cutlass.Constexpr[int],
    acc_stages: cutlass.Constexpr[int],
    channel_blocks: cutlass.Constexpr[int],
    data_mask,
    sf_mask,
):
    """Stream the weight tiles for one work tile: k-tile major, range minor.

    The interleaving is what lets the token operand be fetched once.  Range
    ``j`` of k-tile ``kt`` is channel block ``ch_block + j * channel_blocks``:
    for FC1 that is the gate tile followed by its partner up tile, which the
    MMA warp consumes back to back against a single resident B stage.

    See :func:`_stream` for why the producer is returned rather than mutated.
    """
    producer.reset()
    total: cutlass.Constexpr[int] = k_tiles * acc_stages
    peek = producer.try_acquire()
    for kt in cutlass.range(0, k_tiles, 1, unroll=1):
        for j in cutlass.range_constexpr(acc_stages):
            handle = producer.acquire_and_advance(peek)
            peek = cutlass.Boolean(1)
            if handle.count + 1 < total:
                peek = producer.try_acquire()
            cb = ch_block + Int32(j * channel_blocks)
            cute.copy(
                data_atom,
                g_data[(None, cb, kt, expert)],
                s_data[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=data_mask,
            )
            cute.copy(
                sf_atom,
                g_sf[(None, cb, kt, expert)],
                s_sf[(None, handle.index)],
                tma_bar_ptr=handle.barrier,
                mcast_mask=sf_mask,
            )
    return producer


@cute.jit
def _stream(
    data_atom,
    sf_atom,
    g_data,
    g_sf,
    s_data,
    s_sf,
    producer,
    *,
    k_tiles: cutlass.Constexpr[int],
    data_mask,
    sf_mask,
):
    """Stream one operand's k-tiles for one tile, and RETURN the producer.

    The producer carries the live stage index and barrier phase.  A @cute.jit
    call takes it by value, so those advances are lost unless the caller
    rebinds the returned object -- across a persistent tile loop that silently
    restarts the producer at stage 0 while the consumer keeps advancing, which
    deadlocks on the first wrap.  ``reset()`` deliberately clears only the
    per-tile *count*, not index/phase.
    """
    producer.reset()
    peek = producer.try_acquire()
    for _ in cutlass.range(0, k_tiles, 1, unroll=1):
        handle = producer.acquire_and_advance(peek)
        peek = cutlass.Boolean(1)
        if handle.count + 1 < k_tiles:
            peek = producer.try_acquire()
        cute.copy(
            data_atom,
            g_data[(None, handle.count)],
            s_data[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=data_mask,
        )
        cute.copy(
            sf_atom,
            g_sf[(None, handle.count)],
            s_sf[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=sf_mask,
        )
    return producer


@cute.jit
def _make_s2t(s_sf: cute.Tensor, t_sf: cute.Tensor, two_cta: cutlass.Constexpr[bool]):
    s_compact = cute.filter_zeros(s_sf)
    t_compact = cute.filter_zeros(t_sf)
    atom = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(
            tcgen05.CtaGroup.TWO if two_cta else tcgen05.CtaGroup.ONE
        ),
        cutlass.Float8E4M3FN,
    )
    tiled = tcgen05.make_s2t_copy(atom, t_compact)
    thr = tiled.get_slice(0)
    src = tcgen05.get_s2t_smem_desc_tensor(tiled, thr.partition_S(s_compact))
    return tiled, src, thr.partition_D(t_compact)


@cute.jit
def launch_grouped_gemm(
    a: cute.Tensor,  # (experts, out_channels, K) Float4E2M1FN
    b: cute.Tensor,  # (pool_rows, K) Float4E2M1FN
    sfa: cute.Tensor,  # flat Float8E4M3FN, atom-swizzled
    sfb: cute.Tensor,  # flat Float8E4M3FN, atom-swizzled
    epi_args,  # tuple of output tensors for `epilogue`
    prefix: cute.Tensor,  # (experts + 1,) Int32 exclusive prefix of token tiles
    stream,
    *,
    num_experts: cutlass.Constexpr[int],
    out_channels: cutlass.Constexpr[int],
    pool_rows: cutlass.Constexpr[int],
    k: cutlass.Constexpr[int],
    mma_m: cutlass.Constexpr[int] = 256,
    mma_n: cutlass.Constexpr[int] = 128,
    cluster_m: cutlass.Constexpr[int] = 2,
    two_cta: cutlass.Constexpr[bool] = True,
    num_a_stages: cutlass.Constexpr[int] = 4,
    num_b_stages: cutlass.Constexpr[int] = 3,
    num_clusters: cutlass.Constexpr[int] = 8,
    acc_stages: cutlass.Constexpr[int] = 1,
    epilogue: cutlass.Constexpr = epilogue_plain,
    epi_n: cutlass.Constexpr[int] = _EPI_N,
    epi_smem_floats: cutlass.Constexpr[int] = 0,
    use_pdl: cutlass.Constexpr[bool] = False,
    dispatch_warps: cutlass.Constexpr[int] = 0,
    prologue: cutlass.Constexpr = None,
    wait_schedule: cutlass.Constexpr = None,
    wait_tokens: cutlass.Constexpr = None,
    coop_args=(),
):
    cta_group = tcgen05.CtaGroup.TWO if two_cta else tcgen05.CtaGroup.ONE
    make_mma = lambda: sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        sm100_utils.OperandMajorMode.K,
        sm100_utils.OperandMajorMode.K,
        cutlass.Float8E4M3FN,
        NVFP4_BLOCK,
        cta_group,
        (mma_m, mma_n),
    )
    tiled_mma = make_mma()
    tiled_mma_b = make_mma()
    tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        sm100_utils.OperandMajorMode.K,
        sm100_utils.OperandMajorMode.K,
        cutlass.Float8E4M3FN,
        NVFP4_BLOCK,
        tcgen05.CtaGroup.ONE,
        (mma_m // (2 if two_cta else 1), cute.round_up(mma_n, 128)),
    )
    mma_tiler = (mma_m, mma_n, MMA_TILE_K)
    k_tiles: cutlass.Constexpr[int] = k // MMA_TILE_K
    # Channel blocks the *schedule* sees: a tile covers `acc_stages` of them,
    # so the tile space shrinks by that factor and the ranges are found at
    # `ch_block + j * channel_blocks`.
    channel_blocks: cutlass.Constexpr[int] = out_channels // (mma_m * acc_stages)

    cluster_shape = (cluster_m, 1, 1)
    cluster_vmnk = cute.tiled_divide(
        cute.make_layout(cluster_shape), (tiled_mma.thr_id.shape,)
    )
    cluster_sfb_vmnk = cute.tiled_divide(
        cute.make_layout(cluster_shape), (tiled_mma_sfb.thr_id.shape,)
    )

    a_smem = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler, cutlass.Float4E2M1FN, num_a_stages
    )
    b_smem = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler, cutlass.Float4E2M1FN, num_b_stages
    )
    sfa_smem = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, mma_tiler, NVFP4_BLOCK, num_a_stages
    )
    sfb_smem = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, mma_tiler, NVFP4_BLOCK, num_b_stages
    )

    # Weights carry the expert on the trailing L mode; the token pool has one
    # group, so its L is 1.
    a3 = cute.make_tensor(
        a.iterator,
        cute.make_layout(
            (out_channels, k, num_experts), stride=(k, 1, out_channels * k)
        ),
    )
    b3 = cute.make_tensor(
        b.iterator, cute.make_layout((pool_rows, k, 1), stride=(k, 1, 0))
    )
    sfa3 = cute.make_tensor(
        sfa.iterator,
        blockscaled_utils.tile_atom_to_shape_SF(
            (out_channels, k, num_experts), NVFP4_BLOCK
        ),
    )
    sfb3 = cute.make_tensor(
        sfb.iterator,
        blockscaled_utils.tile_atom_to_shape_SF((pool_rows, k, 1), NVFP4_BLOCK),
    )

    a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape, tiled_mma.thr_id)
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape, tiled_mma.thr_id)
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_shape, tiled_mma.thr_id)

    a_atom, a_view = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        a3,
        cute.slice_(a_smem, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cluster_vmnk.shape,
    )
    b_atom, b_view = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b3,
        cute.slice_(b_smem, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cluster_vmnk.shape,
    )
    sfa_atom, sfa_view = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        sfa3,
        cute.slice_(sfa_smem, (None, None, None, 0)),
        mma_tiler,
        tiled_mma,
        cluster_vmnk.shape,
    )
    sfb_atom, sfb_view = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op,
        sfb3,
        cute.slice_(sfb_smem, (None, None, None, 0)),
        mma_tiler,
        tiled_mma_sfb,
        cluster_sfb_vmnk.shape,
    )

    # Transaction counts are CLUSTER-scoped: under a 2-CTA MMA both CTAs' TMAs
    # signal the leader's stage barrier, so the expected byte count is the
    # per-CTA tile times the CTA-pair size.  Counting only one CTA's bytes
    # makes the leader wait forever for a transaction that already completed.
    atom_thr = cute.size(tiled_mma.thr_id.shape)
    a_tx = (
        cute.size_in_bytes(
            cutlass.Float4E2M1FN, cute.slice_(a_smem, (None, None, None, 0))
        )
        + cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.slice_(sfa_smem, (None, None, None, 0))
        )
    ) * atom_thr
    b_tx = (
        cute.size_in_bytes(
            cutlass.Float4E2M1FN, cute.slice_(b_smem, (None, None, None, 0))
        )
        + cute.size_in_bytes(
            cutlass.Float8E4M3FN, cute.slice_(sfb_smem, (None, None, None, 0))
        )
    ) * atom_thr

    _grouped_gemm_kernel(
        tiled_mma,
        tiled_mma_b,
        tiled_mma_sfb,
        a_atom,
        a_view,
        b_atom,
        b_view,
        sfa_atom,
        sfa_view,
        sfb_atom,
        sfb_view,
        epi_args,
        prefix,
        a_smem,
        b_smem,
        sfa_smem,
        sfb_smem,
        cluster_vmnk,
        cluster_sfb_vmnk,
        mma_tiler,
        a_tx,
        b_tx,
        num_a_stages,
        num_b_stages,
        k_tiles,
        num_experts,
        channel_blocks,
        two_cta,
        cluster_m,
        acc_stages,
        epilogue,
        epi_n,
        epi_smem_floats,
        use_pdl,
        dispatch_warps,
        prologue,
        wait_schedule,
        wait_tokens,
        coop_args,
    ).launch(
        grid=[cluster_m, 1, num_clusters],
        block=[32 * (_NUM_WARPS + dispatch_warps), 1, 1],
        cluster=(cluster_m, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
