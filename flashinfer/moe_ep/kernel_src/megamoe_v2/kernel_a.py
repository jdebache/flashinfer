# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Kernel A: quantize + dispatch + FC1, in one launch.

Why fusing is not just "concatenate the phases"
-----------------------------------------------

Running the phases back to back behind grid syncs would save five launch gaps
and nothing else, because FC1's weight stream -- the actual bottleneck, 793 MB
per launch -- still could not start until the last token had landed.

Letting the weight TMA run early does not fix it either: it fills the four smem
stages (~128 KB) and then backpressures, because the MMA cannot consume a
weight tile without a token tile to multiply it by.

So the mechanism has to be **per-expert readiness**.  Each source publishes one
expert's metadata at a time.  Its owner plans and pulls that expert immediately,
then releases the matching GEMM work while later experts are still in flight.

Warp layout (11 warps, 352 threads)::

    0-3  epilogue   4  MMA   5  TMA-A   6  TMA-B   7-10  dispatch

Warps 7-10 run concurrently with 0-6 -- not before them.  The weight side walks
a static ``(expert, channel block)`` tile space, so it starts without a token
prefix.  Token-block counts are runtime inner-loop bounds; every expert owns at
least one padded block, letting TMA-A issue that first block unconditionally.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Float32, Int32, Int64

from .dispatch import peer_view, sf_word_of
from .fc1 import FC1_EPI_N, epilogue_fc1, stage_floats
from .gemm_kernel import launch_grouped_gemm
from .quant import quantize_row_range

DISPATCH_WARPS = 4
_DISPATCH_BARRIER = 5
_WARP = 32
_ROW_VEC_WORDS = 4

# Index into the coop_args tuple.  A tuple rather than a struct because it
# crosses the @cute.kernel boundary, where only tensors and values survive.
(
    _ACT,
    _IDS,
    _TW,
    _SEND_FP4,
    _SEND_SF,
    _SCOUNT,
    _SSLOT,
    _SWEIGHT,
    _PCOUNT,
    _PSLOT,
    _PWEIGHT,
    _ECOUNT,
    _RANKOFF,
    _PREFIX,
    _SEND_I32,
    _SENDSF_I32,
    _POOL_I32,
    _POOLSF_I32,
    _POOLW,
    _POOLSRC,
    _PEER,
    _GSYNC,
    _READY,
    _NTOK,
    _RANK,
) = range(25)


@cute.jit
def poll_i32(ptr) -> Int32:
    """Read a flag another *block on this GPU* publishes.

    A relaxed load, not ``atomic_add(x, 0)``.  The atomic form is a
    read-modify-write: it cannot be served from a cached copy, so every poll is
    a serialized round trip to the one L2 slice owning the line, and the
    spinners contend with the very traffic they are waiting for.  A relaxed
    load is cacheable and still ordered by the ``fence_acq_rel_gpu`` that
    follows a successful wait.
    """
    return cute.arch.load(ptr, Int32, sem="relaxed", scope="gpu")


@cute.jit
def grid_barrier(gsync: cute.Tensor, gen: Int32, tid: Int32, bar, *, num_blocks: Int32):
    """Device-wide barrier across every block's dispatch group.

    Sound only if the grid is co-resident, which the launcher enforces: at this
    kernel's smem footprint one block occupies an SM outright, so a grid no
    larger than the SM count is resident by construction.

    ``gen`` is a monotonically increasing generation rather than a flag, so the
    barrier is reusable without a clearing pass -- every block runs the same
    number of barriers, so their generations agree.
    """
    bar.arrive_and_wait()
    if tid == Int32(0):
        cute.arch.fence_acq_rel_gpu()
        old = cute.arch.atomic_add(gsync.iterator + 0, Int32(1))
        if old == num_blocks - Int32(1):
            # Last in resets the counter *before* releasing, so a block that
            # wakes and races to the next barrier finds it already at zero.
            gsync[0] = Int32(0)
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_exch(gsync.iterator + 1, gen)
        else:
            while poll_i32(gsync.iterator + 1) != gen:
                pass
        cute.arch.fence_acq_rel_gpu()
    bar.arrive_and_wait()
    return gen + Int32(1)


@cute.jit
def wait_token_blocks(args, expert: Int32):
    """Wait until ``expert`` has a published pool segment, then return it."""
    prefix = args[_PREFIX]
    while poll_i32(prefix.iterator + expert + Int32(1)) == Int32(0):
        pass
    cute.arch.fence_acq_rel_gpu()
    first = prefix[expert]
    return first, prefix[expert + Int32(1)] - first


@cute.jit
def wait_tokens(args, expert: Int32, *, tile_tokens: cutlass.Constexpr[int]):
    """TMA-B: block until every row of ``expert``'s pool segment has landed."""
    first, blocks = wait_token_blocks(args, expert)
    ready = args[_READY]
    needed = blocks * Int32(tile_tokens)
    while poll_i32(ready.iterator + expert) < needed:
        pass
    cute.arch.fence_acq_rel_gpu()
    return first, blocks


@cute.jit
def dispatch_prologue(
    args,
    tid: Int32,
    bidx: Int32,
    bidz: Int32,
    gdim_z: Int32,
    *,
    hidden: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    max_pairs: cutlass.Constexpr[int],
    max_tokens: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
    cluster_m: cutlass.Constexpr[int],
    norm_const: cutlass.Constexpr[float],
) -> None:
    """Everything upstream of FC1, run by warps 7-10 of every block."""
    threads: cutlass.Constexpr[int] = DISPATCH_WARPS * _WARP
    warp = tid // Int32(_WARP)
    lane = tid % Int32(_WARP)
    flat_block = bidz * Int32(cluster_m) + bidx
    num_blocks = gdim_z * Int32(cluster_m)
    bar = pipeline.NamedBarrier(barrier_id=_DISPATCH_BARRIER, num_threads=threads)
    gen = Int32(1)
    num_tokens = args[_NTOK][0]
    # Read from the args rather than captured: a @cute.kernel is an
    # isolated region, so a dynamic value closed over at launch time is
    # not visible inside it.  Constexpr would work too but would compile
    # one artifact per rank, which KernelConfig.name() deliberately avoids.
    my_rank = args[_RANK][0]

    # --- quantize local tokens into the peer-readable buffer, and route ---
    quantize_row_range(
        args[_ACT],
        args[_SEND_FP4],
        args[_SEND_SF],
        Float32(norm_const),
        hidden=hidden,
        num_k_atoms=num_k_atoms,
        first_token=flat_block * Int32(DISPATCH_WARPS) + warp,
        token_limit=num_tokens,
        token_stride=num_blocks * Int32(DISPATCH_WARPS),
        lane_idx=lane,
    )
    pair = flat_block * Int32(threads) + tid
    stride = num_blocks * Int32(threads)
    limit = num_tokens * Int32(top_k)
    while pair < limit:
        token = pair // Int32(top_k)
        slot = pair % Int32(top_k)
        expert = args[_IDS][token, slot]
        if expert >= Int32(0) and expert < Int32(num_experts):
            idx = cute.arch.atomic_add(args[_SCOUNT].iterator + expert, Int32(1))
            args[_SSLOT][expert, idx] = pair
            args[_SWEIGHT][expert, idx] = args[_TW][token, slot]
        pair += stride
    gen = grid_barrier(args[_GSYNC], gen, tid, bar, num_blocks=num_blocks)

    # One warp publishes every global expert in order.  Owner CTAs consume
    # independent local experts as soon as their counts arrive, so publication
    # never waits for a pull and needs no cross-CTA ticket.
    if flat_block == num_blocks - Int32(1) and warp == Int32(0):
        expert = Int32(0)
        while expert < Int32(num_experts):
            dst = expert // Int32(local_experts)
            le = expert % Int32(local_experts)
            off = args[_PEER][dst]
            rc = peer_view(
                args[_PCOUNT], off, args[_PCOUNT].layout, cutlass.Int64, align=8
            )
            rs = peer_view(args[_PSLOT], off, args[_PSLOT].layout, cutlass.Int32)
            rw = peer_view(
                args[_PWEIGHT], off, args[_PWEIGHT].layout, cutlass.Float32
            )
            count = args[_SCOUNT][expert]
            base = (le * Int32(world) + my_rank) * Int32(max_pairs)
            i = lane
            while i < count:
                rs[base + i] = args[_SSLOT][expert, i]
                rw[base + i] = args[_SWEIGHT][expert, i]
                i += Int32(_WARP)
            cute.arch.sync_warp()
            if lane == Int32(0):
                cute.arch.store(
                    rc.iterator + my_rank * Int32(local_experts) + le,
                    Int64(count) + Int64(1),
                    sem="release",
                    scope="sys",
                )
            expert += Int32(1)

    row_words: cutlass.Constexpr[int] = hidden // 8
    owner_workers = num_blocks - Int32(1)
    if owner_workers < Int32(1):
        owner_workers = Int32(1)
    if flat_block < owner_workers:
        le = flat_block
        while le < Int32(local_experts):
            if tid == Int32(0):
                total = Int32(0)
                for r in cutlass.range_constexpr(world):
                    slot = r * local_experts + le
                    count_ptr = args[_PCOUNT].iterator + slot
                    seen = cute.arch.load(
                        count_ptr, Int64, sem="acquire", scope="sys"
                    )
                    while seen == Int64(0):
                        seen = cute.arch.load(
                            count_ptr, Int64, sem="acquire", scope="sys"
                        )
                    args[_RANKOFF][le * world + r] = total
                    total += Int32(seen - Int64(1))
                    args[_PCOUNT][slot] = Int64(0)

                if le > Int32(0):
                    while poll_i32(args[_PREFIX].iterator + le) == Int32(0):
                        pass
                    cute.arch.fence_acq_rel_gpu()
                args[_ECOUNT][le] = Int64(total)
                blocks = (total + Int32(tile_tokens - 1)) // Int32(tile_tokens)
                if blocks < Int32(1):
                    blocks = Int32(1)
                end = args[_PREFIX][le] + blocks
                cute.arch.fence_acq_rel_gpu()
                cute.arch.atomic_exch(args[_PREFIX].iterator + le + Int32(1), end)
            bar.arrive_and_wait()

            pool_base = args[_PREFIX][le] * Int32(tile_tokens)
            span = (args[_PREFIX][le + Int32(1)] - args[_PREFIX][le]) * Int32(
                tile_tokens
            )
            live = Int32(args[_ECOUNT][le])
            row = warp
            while row < span:
                if row < live:
                    src = Int32(0)
                    for r in cutlass.range_constexpr(world - 1):
                        nxt = args[_RANKOFF][le * world + r + 1]
                        src += Int32(1) if nxt <= row else Int32(0)
                    src_row = row - args[_RANKOFF][le * world + src]
                    meta = (le * Int32(world) + src) * Int32(max_pairs) + src_row
                    packed = args[_PSLOT][meta]
                    token = packed // Int32(top_k)
                    src_off = args[_PEER][src]
                    rt = peer_view(
                        args[_SEND_I32],
                        src_off,
                        args[_SEND_I32].layout,
                        cutlass.Int32,
                    )
                    rsf = peer_view(
                        args[_SENDSF_I32],
                        src_off,
                        args[_SENDSF_I32].layout,
                        cutlass.Int32,
                    )
                    _copy_row(
                        rt[token, None],
                        args[_POOL_I32][pool_base + row, None],
                        lane,
                        words=row_words,
                    )
                    for st in cutlass.range_constexpr(
                        (num_k_atoms + _WARP - 1) // _WARP
                    ):
                        ka = lane + Int32(st * _WARP)
                        if ka < Int32(num_k_atoms):
                            args[_POOLSF_I32][
                                sf_word_of(
                                    pool_base + row, ka, num_k_atoms=num_k_atoms
                                )
                            ] = rsf[sf_word_of(token, ka, num_k_atoms=num_k_atoms)]
                    if lane == Int32(0):
                        args[_POOLW][pool_base + row] = args[_PWEIGHT][meta]
                        args[_POOLSRC][pool_base + row] = Int64(src) * Int64(
                            max_tokens * top_k
                        ) + Int64(packed)
                else:
                    if lane == Int32(0):
                        args[_POOLW][pool_base + row] = Float32(0.0)
                        args[_POOLSRC][pool_base + row] = Int64(-1)
                    for st2 in cutlass.range_constexpr(
                        (num_k_atoms + _WARP - 1) // _WARP
                    ):
                        ka2 = lane + Int32(st2 * _WARP)
                        if ka2 < Int32(num_k_atoms):
                            args[_POOLSF_I32][
                                sf_word_of(
                                    pool_base + row, ka2, num_k_atoms=num_k_atoms
                                )
                            ] = Int32(0)
                row += Int32(DISPATCH_WARPS)

            bar.arrive_and_wait()
            if tid == Int32(0):
                cute.arch.fence_acq_rel_gpu()
                cute.arch.atomic_exch(args[_READY].iterator + le, span)
            le += owner_workers


@cute.jit
def _copy_row(src_row, dst_row, lane: Int32, *, words: cutlass.Constexpr[int]):
    chunks: cutlass.Constexpr[int] = words // _ROW_VEC_WORDS
    steps: cutlass.Constexpr[int] = (chunks + _WARP - 1) // _WARP
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Int32,
        num_bits_per_copy=_ROW_VEC_WORDS * 32,
    )
    src4 = cute.zipped_divide(src_row, (_ROW_VEC_WORDS,))
    dst4 = cute.zipped_divide(dst_row, (_ROW_VEC_WORDS,))
    frag = cute.make_rmem_tensor((_ROW_VEC_WORDS,), cutlass.Int32)
    for step in cutlass.range_constexpr(steps):
        chunk = lane + Int32(step * _WARP)
        if chunk < Int32(chunks):
            cute.copy(atom, src4[(None,), (chunk,)], frag)
            cute.copy(atom, frag, dst4[(None,), (chunk,)])


@cute.jit
def launch_kernel_a(
    w1: cute.Tensor,
    sf_w1: cute.Tensor,
    fc1_out: cute.Tensor,
    fc1_out_sf: cute.Tensor,
    coop_args,
    stream,
    *,
    num_experts: cutlass.Constexpr[int],
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    intermediate: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    pool_rows: cutlass.Constexpr[int],
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    hidden_atoms: cutlass.Constexpr[int],
    inter_atoms: cutlass.Constexpr[int],
    norm_const: cutlass.Constexpr[float] = 1.0,
    clamp: cutlass.Constexpr = None,
    apply_weight: cutlass.Constexpr[bool] = True,
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
    tile_tokens: cutlass.Constexpr[int] = mma_n
    pool_tokens = cute.recast_tensor(coop_args[_POOL_I32], cutlass.Float4E2M1FN)
    pool_scales = cute.recast_tensor(coop_args[_POOLSF_I32], cutlass.Float8E4M3FN)
    launch_grouped_gemm(
        w1,
        pool_tokens,
        sf_w1,
        pool_scales,
        (fc1_out, fc1_out_sf, coop_args[_POOLW]),
        coop_args[_PREFIX],
        stream,
        num_experts=local_experts,
        out_channels=2 * intermediate,
        pool_rows=pool_rows,
        k=hidden,
        mma_m=mma_m,
        mma_n=mma_n,
        cluster_m=cluster_m,
        two_cta=two_cta,
        num_a_stages=num_a_stages,
        num_b_stages=num_b_stages,
        num_clusters=num_clusters,
        acc_stages=2,
        epilogue=functools.partial(
            epilogue_fc1,
            num_k_atoms=inter_atoms,
            clamp=clamp,
            apply_weight=apply_weight,
        ),
        epi_n=FC1_EPI_N,
        epi_smem_floats=stage_floats(cta_tile_m, FC1_EPI_N),
        use_pdl=use_pdl,
        dispatch_warps=DISPATCH_WARPS,
        prologue=functools.partial(
            dispatch_prologue,
            hidden=hidden,
            num_k_atoms=hidden_atoms,
            top_k=top_k,
            num_experts=num_experts,
            local_experts=local_experts,
            world=world,
            max_pairs=max_tokens * top_k,
            max_tokens=max_tokens,
            tile_tokens=tile_tokens,
            cluster_m=cluster_m,
            norm_const=norm_const,
        ),
        wait_token_blocks=wait_token_blocks,
        wait_tokens=functools.partial(wait_tokens, tile_tokens=tile_tokens),
        coop_args=coop_args,
    )
