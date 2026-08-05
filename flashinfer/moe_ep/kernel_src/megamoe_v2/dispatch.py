# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Cross-rank dispatch: push the metadata, pull the tokens.

Shape of the exchange
---------------------

Every rank quantizes its own tokens once into a peer-readable buffer, then
tells each destination *what* is coming; the destination pulls the rows it
owns.  Push-metadata / pull-data rather than push-data because a token chosen
by ``top_k`` experts on several ranks would otherwise be sent several times,
and because the destination is the only party that knows where in its pool a
row belongs.

The write-ownership rule that removes all remote atomics: the count and slot
arrays on a destination are indexed ``[expert][source rank][slot]``, so each
source rank owns one row of them outright.  A source picks its slot indices
with a *local* atomic while routing, then pushes one contiguous run per
expert.  No rank ever contends with another for a slot, and no rank needs a
remote read-modify-write.

Four phases, four launches
--------------------------

``prepare``   quantize local tokens; bucket ``(token, slot)`` pairs by
              destination expert into local staging.
``push``      copy each expert's staged run into its owner's arrays.
``plan``      (destination side, after the barrier) sum the per-source counts
              into per-expert totals, and turn those into the pool offsets and
              the tile prefix the GEMM schedule consumes.
``pull``      copy each owed row -- codes, scale factors, weight, provenance --
              from its source's send buffer into the local pool.

They are separate launches because the launch boundary already gives the
ordering each phase needs; only ``push`` -> ``plan`` additionally needs the
cross-rank barrier, since it orders *other* ranks' writes.

Peers are reached the way v1 reaches them: a table of ``peer_base -
local_base`` byte offsets, added to a local address.  At ``world_size == 1``
every offset is zero and the same code path runs unchanged, which is what
makes the single-rank test cover the real addressing rather than a special
case.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32, Int64

from .quant import quantize_row_range
from .sf_layout import SF_ATOM_ROWS, SF_ATOM_WORDS

# int32 words moved per lane per step by the row copy.  16 B is the widest
# universal copy, and every row start is 16 B aligned because `hidden` is a
# multiple of 32 (checked in the launchers).
_ROW_VEC_WORDS = 4
_WARP = 32


@cute.jit
def sf_word_of(row: Int32, k_atom: Int32, *, num_k_atoms: cutlass.Constexpr[int]):
    """Device twin of :func:`..sf_layout.word_offset`.

    One token's four K-banks share an int32 word, which is why the whole scale
    plane can be moved as words rather than bytes.
    """
    row_block = row // Int32(SF_ATOM_ROWS)
    t = row % Int32(SF_ATOM_ROWS)
    atom = row_block * Int32(num_k_atoms) + k_atom
    return atom * Int32(SF_ATOM_WORDS) + (t % Int32(32)) * Int32(4) + t // Int32(32)


def peer_view(local: cute.Tensor, byte_offset, layout, dtype, align: int = 16):
    """A peer rank's copy of a symmetric tensor, as a local tensor.

    Plain Python: this only builds a pointer at trace time.  ``byte_offset``
    is ``peer_base - local_base`` for the target rank, so rank ``self`` maps to
    the identity and needs no special case.
    """
    ptr = cute.make_ptr(
        dtype,
        local.iterator.toint() + byte_offset,
        cute.AddressSpace.gmem,
        assumed_align=align,
    )
    return cute.make_tensor(ptr, layout)


# --------------------------------------------------------------------------
# prepare: quantize + route
# --------------------------------------------------------------------------


@cute.kernel
def _prepare_kernel(
    activation: cute.Tensor,  # (num_tokens, hidden) bf16
    topk_ids: cute.Tensor,  # (num_tokens, top_k) int32
    topk_weights: cute.Tensor,  # (num_tokens, top_k) f32
    send_tokens: cute.Tensor,  # (max_tokens, hidden) Float4E2M1FN
    send_sf: cute.Tensor,  # flat Float8E4M3FN
    send_count: cute.Tensor,  # (num_experts,) int32
    send_slot: cute.Tensor,  # (num_experts, max_pairs) int32
    send_weight: cute.Tensor,  # (num_experts, max_pairs) f32
    num_tokens: Int32,
    norm_const: Float32,
    hidden: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    warps_per_cta: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    quantize_row_range(
        activation,
        send_tokens,
        send_sf,
        norm_const,
        hidden=hidden,
        num_k_atoms=num_k_atoms,
        first_token=bidx * Int32(warps_per_cta) + tidx // Int32(_WARP),
        token_limit=num_tokens,
        token_stride=gdim * Int32(warps_per_cta),
        lane_idx=tidx % Int32(_WARP),
    )

    # Routing is thread-per-pair.  The slot a pair lands in is whatever the
    # local atomic hands out, so the order within an expert is unspecified --
    # deliberately: nothing downstream depends on it (the pool is summed, not
    # matched positionally), and pinning it would cost a sort.
    threads = Int32(warps_per_cta * _WARP)
    pair = bidx * threads + tidx
    stride = gdim * threads
    limit = num_tokens * Int32(top_k)
    while pair < limit:
        token = pair // Int32(top_k)
        slot = pair % Int32(top_k)
        expert = topk_ids[token, slot]
        if expert >= Int32(0) and expert < Int32(num_experts):
            index = cute.arch.atomic_add(send_count.iterator + expert, Int32(1))
            send_slot[expert, index] = pair
            send_weight[expert, index] = topk_weights[token, slot]
        pair += stride


@cute.jit
def dispatch_prepare(
    activation: cute.Tensor,
    topk_ids: cute.Tensor,
    topk_weights: cute.Tensor,
    send_tokens: cute.Tensor,
    send_sf: cute.Tensor,
    send_count: cute.Tensor,
    send_slot: cute.Tensor,
    send_weight: cute.Tensor,
    num_tokens: Int32,
    norm_const: Float32,
    stream,
    *,
    hidden: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int] = 64,
    warps_per_cta: cutlass.Constexpr[int] = 8,
):
    _prepare_kernel(
        activation,
        topk_ids,
        topk_weights,
        send_tokens,
        send_sf,
        send_count,
        send_slot,
        send_weight,
        num_tokens,
        norm_const,
        hidden,
        num_k_atoms,
        top_k,
        num_experts,
        warps_per_cta,
    ).launch(grid=[num_ctas, 1, 1], block=[warps_per_cta * _WARP, 1, 1], stream=stream)


# --------------------------------------------------------------------------
# push: hand each destination its metadata
# --------------------------------------------------------------------------


@cute.kernel
def _push_kernel(
    send_count: cute.Tensor,  # (num_experts,) int32
    send_slot: cute.Tensor,  # (num_experts, max_pairs) int32
    send_weight: cute.Tensor,  # (num_experts, max_pairs) f32
    peer_count: cute.Tensor,  # local view of (world * local_experts,) int64
    peer_slot: cute.Tensor,  # local view of (local_experts*world*max_pairs,) i32
    peer_weight: cute.Tensor,  # local view of the same shape, f32
    peer_offset: cute.Tensor,  # (world,) int64 byte offsets
    my_rank: Int32,
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    max_pairs: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    expert, _, _ = cute.arch.block_idx()

    dst_rank = expert // Int32(local_experts)
    local_expert = expert % Int32(local_experts)
    offset = peer_offset[dst_rank]

    rc = peer_view(peer_count, offset, peer_count.layout, cutlass.Int64, align=8)
    rs = peer_view(peer_slot, offset, peer_slot.layout, cutlass.Int32)
    rw = peer_view(peer_weight, offset, peer_weight.layout, cutlass.Float32)

    count = send_count[expert]
    if tidx == Int32(0):
        rc[my_rank * Int32(local_experts) + local_expert] = Int64(count)

    # This source rank owns row (local_expert, my_rank) outright, so the run is
    # contiguous and needs no coordination with the other sources.
    base = (local_expert * Int32(world) + my_rank) * Int32(max_pairs)
    i = tidx
    while i < count:
        rs[base + i] = send_slot[expert, i]
        rw[base + i] = send_weight[expert, i]
        i += Int32(threads)


@cute.jit
def dispatch_push(
    send_count: cute.Tensor,
    send_slot: cute.Tensor,
    send_weight: cute.Tensor,
    peer_count: cute.Tensor,
    peer_slot: cute.Tensor,
    peer_weight: cute.Tensor,
    peer_offset: cute.Tensor,
    my_rank: Int32,
    stream,
    *,
    num_experts: cutlass.Constexpr[int],
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    max_pairs: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int] = 256,
):
    _push_kernel(
        send_count,
        send_slot,
        send_weight,
        peer_count,
        peer_slot,
        peer_weight,
        peer_offset,
        my_rank,
        local_experts,
        world,
        max_pairs,
        threads,
    ).launch(grid=[num_experts, 1, 1], block=[threads, 1, 1], stream=stream)


# --------------------------------------------------------------------------
# barrier: the one point where a rank must wait for the others
# --------------------------------------------------------------------------


@cute.kernel
def _barrier_kernel(
    signal: cute.Tensor,  # (world,) int64, symmetric: slot r is rank r's
    phase_store: cute.Tensor,  # (1,) int32, rank-private, survives launches
    peer_offset: cute.Tensor,
    my_rank: Int32,
    world: cutlass.Constexpr[int],
):
    """Flag-based, sense-carrying barrier across the EP group.

    Each rank publishes a monotonically increasing phase into *its own* slot on
    every peer, then waits for every slot at home to reach that phase.  Slot
    ownership is what makes it atomic-free, exactly as in the dispatch
    metadata push; the phase is monotonic rather than a reset counter so the
    barrier can be reused without a clearing pass between launches.

    Single-threaded: ``world`` is at most a handful, and the kernel boundary
    already guarantees this rank's prior work is complete.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx == Int32(0) and tidx == Int32(0):
        phase = Int64(phase_store[0]) + Int64(1)
        phase_store[0] = Int32(phase)
        # Release everything this rank wrote before anyone can observe the
        # flag; system scope because the observer is another GPU.
        cute.arch.fence_acq_rel_sys()
        for r in cutlass.range_constexpr(world):
            remote = peer_view(
                signal, peer_offset[r], signal.layout, cutlass.Int64, align=8
            )
            cute.arch.atomic_exch(remote.iterator + my_rank, phase)
        for r in cutlass.range_constexpr(world):
            # atomic rather than a plain load: a load in a spin loop is free to
            # be hoisted, and this must re-read memory every iteration.
            seen = cute.arch.atomic_add(signal.iterator + r, Int64(0))
            while seen < phase:
                seen = cute.arch.atomic_add(signal.iterator + r, Int64(0))
        cute.arch.fence_acq_rel_sys()


@cute.jit
def dispatch_barrier(
    signal: cute.Tensor,
    phase_store: cute.Tensor,
    peer_offset: cute.Tensor,
    my_rank: Int32,
    stream,
    *,
    world: cutlass.Constexpr[int],
):
    _barrier_kernel(signal, phase_store, peer_offset, my_rank, world).launch(
        grid=[1, 1, 1], block=[32, 1, 1], stream=stream
    )


# --------------------------------------------------------------------------
# plan: counts -> pool offsets and the tile prefix
# --------------------------------------------------------------------------


@cute.kernel
def _plan_kernel(
    peer_count: cute.Tensor,  # (world * local_experts,) int64
    expert_count: cute.Tensor,  # (local_experts,) int64
    rank_pool_offset: cute.Tensor,  # (local_experts * world,) int32
    token_block_prefix: cute.Tensor,  # (local_experts + 1,) int32
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    # Single-threaded on purpose: this is `local_experts * world` adds (a few
    # hundred at the reference shape), and a serial scan is both the simplest
    # correct prefix and far cheaper than the sync a parallel one would need.
    if tidx == Int32(0):
        blocks = Int32(0)
        token_block_prefix[0] = Int32(0)
        for le in cutlass.range_constexpr(local_experts):
            total = Int32(0)
            for r in cutlass.range_constexpr(world):
                rank_pool_offset[le * world + r] = total
                total += Int32(peer_count[r * local_experts + le])
            expert_count[le] = Int64(total)
            blocks += (total + Int32(tile_tokens - 1)) // Int32(tile_tokens)
            token_block_prefix[le + 1] = blocks


@cute.jit
def dispatch_plan(
    peer_count: cute.Tensor,
    expert_count: cute.Tensor,
    rank_pool_offset: cute.Tensor,
    token_block_prefix: cute.Tensor,
    stream,
    *,
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
):
    _plan_kernel(
        peer_count,
        expert_count,
        rank_pool_offset,
        token_block_prefix,
        local_experts,
        world,
        tile_tokens,
    ).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)


# --------------------------------------------------------------------------
# pull: fetch the owed rows
# --------------------------------------------------------------------------


@cute.jit
def _copy_row_words(src_row, dst_row, lane: Int32, *, words: cutlass.Constexpr[int]):
    """Move one token's NVFP4 codes, 16 B per lane per step."""
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


@cute.kernel
def _pull_kernel(
    send_tokens: cute.Tensor,  # local view: (max_tokens, hidden/8) int32
    send_sf: cute.Tensor,  # local view: flat int32
    peer_slot: cute.Tensor,  # (local_experts*world*max_pairs,) int32
    peer_weight: cute.Tensor,  # same shape, f32
    peer_count: cute.Tensor,  # (world*local_experts,) int64
    peer_offset: cute.Tensor,  # (world,) int64
    expert_count: cute.Tensor,  # (local_experts,) int64
    rank_pool_offset: cute.Tensor,
    token_block_prefix: cute.Tensor,
    pool_tokens: cute.Tensor,  # (pool_rows, hidden/8) int32
    pool_sf: cute.Tensor,  # flat int32
    pool_weight: cute.Tensor,  # (pool_rows,) f32
    pool_src: cute.Tensor,  # (pool_rows,) int64
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    max_pairs: cutlass.Constexpr[int],
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    row_words: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
    warps_per_cta: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    warp = tidx // Int32(_WARP)
    lane = tidx % Int32(_WARP)

    # One (expert, source rank) run per block-step; warps within a block take
    # one row each.  Runs are independent, so no block needs to know the total.
    group = bidx
    while group < Int32(local_experts * world):
        local_expert = group // Int32(world)
        src_rank = group % Int32(world)
        count = Int32(peer_count[src_rank * Int32(local_experts) + local_expert])
        pool_base = (
            token_block_prefix[local_expert] * Int32(tile_tokens)
            + rank_pool_offset[group]
        )
        meta_base = group * Int32(max_pairs)

        offset = peer_offset[src_rank]
        rt = peer_view(send_tokens, offset, send_tokens.layout, cutlass.Int32)
        rsf = peer_view(send_sf, offset, send_sf.layout, cutlass.Int32)

        i = warp
        while i < count:
            packed = peer_slot[meta_base + i]
            token = packed // Int32(top_k)
            row = pool_base + i
            _copy_row_words(
                rt[token, None], pool_tokens[row, None], lane, words=row_words
            )
            for step in cutlass.range_constexpr((num_k_atoms + _WARP - 1) // _WARP):
                k_atom = lane + Int32(step * _WARP)
                if k_atom < Int32(num_k_atoms):
                    pool_sf[sf_word_of(row, k_atom, num_k_atoms=num_k_atoms)] = rsf[
                        sf_word_of(token, k_atom, num_k_atoms=num_k_atoms)
                    ]
            if lane == Int32(0):
                pool_weight[row] = peer_weight[meta_base + i]
                pool_src[row] = Int64(src_rank) * Int64(max_tokens * top_k) + Int64(
                    packed
                )
            i += Int32(warps_per_cta)
        group += gdim

    # Tail rows of each expert's segment exist only to pad it to a whole tile.
    # Their weight is zeroed so the FC1 epilogue emits zeros there, and their
    # scales are zeroed so an uninitialized E4M3 byte cannot read back as NaN
    # and poison a whole 16-channel block.
    expert = bidx
    while expert < Int32(local_experts):
        total = Int32(expert_count[expert])
        base = token_block_prefix[expert] * Int32(tile_tokens)
        limit = token_block_prefix[expert + 1] * Int32(tile_tokens)
        row = base + total + tidx
        while row < limit:
            pool_weight[row] = Float32(0.0)
            pool_src[row] = Int64(-1)
            for k_atom in cutlass.range_constexpr(num_k_atoms):
                pool_sf[sf_word_of(row, Int32(k_atom), num_k_atoms=num_k_atoms)] = (
                    Int32(0)
                )
            row += Int32(warps_per_cta * _WARP)
        expert += gdim


@cute.jit
def dispatch_pull(
    send_tokens: cute.Tensor,
    send_sf: cute.Tensor,
    peer_slot: cute.Tensor,
    peer_weight: cute.Tensor,
    peer_count: cute.Tensor,
    peer_offset: cute.Tensor,
    expert_count: cute.Tensor,
    rank_pool_offset: cute.Tensor,
    token_block_prefix: cute.Tensor,
    pool_tokens: cute.Tensor,
    pool_sf: cute.Tensor,
    pool_weight: cute.Tensor,
    pool_src: cute.Tensor,
    stream,
    *,
    local_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    max_pairs: cutlass.Constexpr[int],
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int] = 64,
    warps_per_cta: cutlass.Constexpr[int] = 8,
):
    if cutlass.const_expr(hidden % 32 != 0):
        raise ValueError(
            f"hidden ({hidden}) must be a multiple of 32 so NVFP4 rows are "
            "16 B aligned and can be moved by vector copies"
        )
    _pull_kernel(
        send_tokens,
        send_sf,
        peer_slot,
        peer_weight,
        peer_count,
        peer_offset,
        expert_count,
        rank_pool_offset,
        token_block_prefix,
        pool_tokens,
        pool_sf,
        pool_weight,
        pool_src,
        local_experts,
        world,
        max_pairs,
        max_tokens,
        top_k,
        hidden // 8,
        num_k_atoms,
        tile_tokens,
        warps_per_cta,
    ).launch(grid=[num_ctas, 1, 1], block=[warps_per_cta * _WARP, 1, 1], stream=stream)
