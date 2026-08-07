# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Workspace layout: which bytes live where, as plain data.

Two workspaces, split by *reachability* rather than by lifetime:

``shared``
    Must be addressable on a peer rank through the symmetric heap, because
    some other rank writes it (the per-expert counts each rank pushes to the
    expert's owner) or reads it (the quantized token rows peers pull).

``local``
    Rank-private scratch: the received-token pool, the FC1 output that kernel
    A hands to kernel B, and the counters.

The A/B contract lives entirely in ``local``: kernel A writes
``expert_token_count`` + ``pool_*`` + ``fc1_out*``, kernel B reads them.  There
is no in-kernel handshake because the launch boundary already orders the two.

Counter regions are grouped into a zeroed prefix so a launch can reset them
with one contiguous fill, *except* ``barrier_phase``, which deliberately sits
outside it: the sense-reversing cross-rank barrier carries its phase across
launches, so zeroing it would desynchronize ranks.
"""

from __future__ import annotations

import dataclasses

from .types import (
    KernelConfig,
    NVFP4_ELEMS_PER_BYTE,
    SF_ATOM_BLOCKS,
    SF_ATOM_ROWS,
    ceil_div,
    round_up,
)

# Element byte widths, named so the region table reads as a layout rather than
# as arithmetic.
_BYTES_I32 = 4
_BYTES_I64 = 8
_BYTES_F32 = 4
_BYTES_BF16 = 2
_BYTES_E4M3 = 1

# TMA store/load destinations want 128 B alignment; counters only need natural
# alignment, but 16 B keeps every region start cache-line friendly.
_ALIGN_TMA = 128
_ALIGN_COUNTER = 16


@dataclasses.dataclass(frozen=True)
class Region:
    """One named byte range inside a workspace."""

    name: str
    nbytes: int
    align: int
    # Zeroed by the per-launch counter reset.  Data planes are fully
    # overwritten before they are read, so they stay out of the reset.
    resettable: bool = False


@dataclasses.dataclass(frozen=True)
class WorkspaceLayout:
    """Resolved byte offsets for one workspace."""

    regions: tuple[Region, ...]
    offsets: tuple[int, ...]
    total_bytes: int
    reset_prefix_bytes: int

    def offset_of(self, name: str) -> int:
        for region, offset in zip(self.regions, self.offsets, strict=True):
            if region.name == name:
                return offset
        raise KeyError(f"no region named {name!r}")

    def nbytes_of(self, name: str) -> int:
        for region in self.regions:
            if region.name == name:
                return region.nbytes
        raise KeyError(f"no region named {name!r}")


def _resolve(regions: tuple[Region, ...]) -> WorkspaceLayout:
    """Pack regions in order, honouring each one's alignment.

    Resettable regions are emitted first by the callers below, so the reset
    prefix is a single contiguous range and the per-launch clear is one fill.
    """
    offsets: list[int] = []
    cursor = 0
    reset_end = 0
    seen_non_resettable = False
    for region in regions:
        if region.resettable and seen_non_resettable:
            raise ValueError(
                f"resettable region {region.name!r} follows a non-resettable "
                "one; the reset prefix must be contiguous"
            )
        seen_non_resettable = seen_non_resettable or not region.resettable
        cursor = round_up(cursor, region.align)
        offsets.append(cursor)
        cursor += region.nbytes
        if region.resettable:
            reset_end = cursor
    return WorkspaceLayout(
        regions=regions,
        offsets=tuple(offsets),
        total_bytes=round_up(cursor, _ALIGN_TMA),
        reset_prefix_bytes=reset_end,
    )


def sf_row_capacity(rows: int) -> int:
    return round_up(rows, SF_ATOM_ROWS)


def sf_cols_for(k_elements: int, block: int) -> int:
    """Padded scale-factor columns for a K extent, in E4M3 entries."""
    return round_up(ceil_div(k_elements, block), SF_ATOM_BLOCKS)


def shared_layout(config: KernelConfig) -> WorkspaceLayout:
    """Peer-reachable regions."""
    shape = config.shape
    world = config.topology.world_size
    local_experts = config.experts_per_rank
    max_pairs = shape.max_tokens_per_rank * shape.top_k

    return _resolve(
        (
            # Per (source rank, local expert) token count, pushed by the
            # *source* rank into its own row of the destination's array.  Each
            # source owns one row, so no rank ever needs a remote atomic to
            # find where to write.  Staged dispatch stores the raw count behind
            # a global barrier; fused dispatch stores count + 1 so zero is an
            # unpublished sentinel for its per-expert rendezvous.  The fused
            # owner clears each slot after consuming it, so a faster rank
            # cannot race a host-side reset on the next iteration.
            Region(
                "peer_expert_count",
                world * local_experts * _BYTES_I64,
                _ALIGN_COUNTER,
            ),
            # Summed per local expert; this is what both kernels' schedules
            # read to size the tile space.
            Region(
                "expert_token_count",
                local_experts * _BYTES_I64,
                _ALIGN_COUNTER,
            ),
            # (local expert, source rank, slot) -> packed (token, topk slot) on
            # the source rank; tells a puller which row to fetch.
            Region(
                "src_token_slot",
                local_experts * world * max_pairs * _BYTES_I32,
                _ALIGN_COUNTER,
            ),
            # Per pushed pair, the routing weight that travels with it.  Sent
            # rather than read back later: the same token row goes to `top_k`
            # experts with a different weight each, so the weight belongs to
            # the pair, not to the row.
            Region(
                "src_topk_weight",
                local_experts * world * max_pairs * _BYTES_F32,
                _ALIGN_COUNTER,
            ),
            # One slot per source rank: rank r publishes its phase into slot r
            # of every peer, so the barrier is flag-based and needs no remote
            # read-modify-write.  NOT resettable -- the phase rides across
            # launches, and zeroing it would desynchronize the ranks.
            Region("barrier_signal", world * _BYTES_I64, _ALIGN_COUNTER),
            # --- peer-readable data: quantized once here, pulled by owners ---
            # Quantizing on the source side rather than the pull side keeps
            # NVFP4 (not bf16) on the wire and does the work once per token
            # instead of once per (token, destination); see :mod:`.quant`.
            Region(
                "send_tokens",
                shape.max_tokens_per_rank * shape.hidden_bytes,
                _ALIGN_TMA,
            ),
            Region(
                "send_token_sf",
                sf_row_capacity(shape.max_tokens_per_rank)
                * sf_cols_for(shape.hidden, 16),
                _ALIGN_TMA,
            ),
            # Where peers land this rank's FC2 results, one slot per
            # (top-k slot, token).  Slot ownership is what makes the combine
            # push a plain store instead of a remote atomic add; the price is
            # this buffer being `top_k` times the output.
            Region(
                "combine_buf",
                shape.top_k * shape.max_tokens_per_rank * shape.hidden * _BYTES_BF16,
                _ALIGN_TMA,
            ),
        )
    )


def local_layout(config: KernelConfig) -> WorkspaceLayout:
    """Rank-private regions, including the whole A -> B handoff."""
    shape = config.shape
    local_experts = config.experts_per_rank
    pool_rows = config.pool_token_capacity
    pool_sf_rows = sf_row_capacity(pool_rows)

    # FC1 writes gate_up channels but SwiGLU halves them, so the pool kernel B
    # reads is `intermediate` wide, not `gate_up`.
    fc1_out_bytes = pool_rows * shape.intermediate // NVFP4_ELEMS_PER_BYTE
    fc1_out_sf_cols = sf_cols_for(shape.intermediate, 16)

    return _resolve(
        (
            # Padded rows that have landed, per local expert.  Kernel A's owner
            # CTA publishes the full segment; FC1's TMA-B warp waits on it.
            Region(
                "token_ready_count",
                local_experts * _BYTES_I32,
                _ALIGN_COUNTER,
                resettable=True,
            ),
            # Device-wide software barrier tickets: arrival counter + release
            # generation.  Kernel A and kernel B get their own so neither has
            # to reason about the other's leftover generation number.
            Region("grid_sync", 2 * _BYTES_I32, _ALIGN_COUNTER, resettable=True),
            Region("grid_sync_b", 2 * _BYTES_I32, _ALIGN_COUNTER, resettable=True),
            # Routing staging, indexed by *global* expert: how many local pairs
            # go to each, and which.  Staged rather than written straight to
            # the destination because the slot index comes from a local atomic,
            # and one contiguous push per expert beats a remote store per pair.
            Region(
                "send_count",
                shape.num_experts * _BYTES_I32,
                _ALIGN_COUNTER,
                resettable=True,
            ),
            # Where each expert's segment starts within the pool, relative to
            # the expert base, per source rank.  Filled by the dispatch plan.
            Region(
                "rank_pool_offset",
                local_experts * config.topology.world_size * _BYTES_I32,
                _ALIGN_COUNTER,
                resettable=True,
            ),
            # Exclusive prefix of per-expert token tiles: the schedule itself,
            # computed on device because the counts are only known there.
            # Fused dispatch advances it by at least one tile per expert.
            Region(
                "token_block_prefix",
                (local_experts + 1) * _BYTES_I32,
                _ALIGN_COUNTER,
                resettable=True,
            ),
            # Sense-reversing barrier phase.  Outside the reset prefix on
            # purpose (see module docstring), hence placed after every
            # resettable region.
            Region("barrier_phase", _BYTES_I32, _ALIGN_COUNTER),
            # --- data planes: fully rewritten each launch, never reset ---
            # Packed (token * top_k + slot) per staged pair, and its weight.
            Region(
                "send_slot",
                shape.num_experts
                * shape.max_tokens_per_rank
                * shape.top_k
                * _BYTES_I32,
                _ALIGN_COUNTER,
            ),
            Region(
                "send_weight",
                shape.num_experts
                * shape.max_tokens_per_rank
                * shape.top_k
                * _BYTES_F32,
                _ALIGN_COUNTER,
            ),
            # Received tokens, NVFP4, padded per expert to a whole token tile.
            Region("pool_tokens", pool_rows * shape.hidden_bytes, _ALIGN_TMA),
            Region(
                "pool_token_sf",
                pool_sf_rows * sf_cols_for(shape.hidden, 16),
                _ALIGN_TMA,
            ),
            # Per pool row: the routing weight and the provenance needed to
            # send the result back.
            Region("pool_topk_weight", pool_rows * _BYTES_F32, _ALIGN_COUNTER),
            Region("pool_src", pool_rows * _BYTES_I64, _ALIGN_COUNTER),
            # --- the A -> B handoff ---
            Region("fc1_out", fc1_out_bytes, _ALIGN_TMA),
            Region("fc1_out_sf", pool_sf_rows * fc1_out_sf_cols, _ALIGN_TMA),
        )
    )


def workspace_sizes(config: KernelConfig) -> tuple[int, int]:
    """``(local_bytes, shared_bytes)`` for a config.

    Both halves of the split must agree exactly on this, since they address
    one shared allocation; the launcher asserts it rather than trusting it.
    """
    return local_layout(config).total_bytes, shared_layout(config).total_bytes
