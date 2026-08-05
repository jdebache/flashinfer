# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Work-tile enumeration for one GEMM phase.

Design note -- why this is not a state machine
----------------------------------------------

v1 walked the tile space with a register-resident state machine (group ->
phase -> expert) whose cumulative pool offsets were *derived* by the walk.
That made expert order load-bearing: offsets were only correct if experts were
visited in strictly ascending order, so work could never be reordered by
readiness, and every consumer had to be handed a bundle of running cumulative
fields.

Here the per-expert prefix is computed **once**, up front, into a small array,
and tile decode is a pure function of that array.  Three consequences:

* decode is O(log E) with no carried state, so any CTA can decode any tile
  index at any time, in any order;
* the expert visiting order is free -- permute ``token_counts`` and the whole
  schedule follows, which is what a readiness-ordered or load-balanced variant
  needs (v1 could not express this without a new offset representation);
* the channel axis factors out of the search entirely (see :func:`decode_tile`),
  so the hot path is one integer division and a branch-free binary search over
  at most ``experts_per_rank`` entries.

The padded pool row for an expert falls out of the *same* prefix that gives its
tile range, because each expert's pool segment is padded to a whole token tile:
``pool_row[e] == cluster_tile_tokens * token_block_prefix[e]``.  One array, two
uses, no second bookkeeping path to get out of sync.
"""

from __future__ import annotations

import bisect
import dataclasses

from .types import (
    Phase,
    ProblemShape,
    SF_ATOM_ROWS,
    TileConfig,
    ceil_div,
)


@dataclasses.dataclass(frozen=True)
class WorkTile:
    """One unit of GEMM work: a (channel block, token block) of one expert."""

    expert: int
    # Cluster tile index along the output-channel axis (GEMM-M under swap-AB).
    channel_block: int
    # Token tile index *within* this expert.
    token_block: int
    # First padded pool row of this expert's segment.
    pool_row: int
    # First scale-factor row of this expert's segment.
    sf_row: int
    # Live tokens in this tile; the tail tile of an expert is partial.
    valid_tokens: int


@dataclasses.dataclass(frozen=True)
class ExpertLayout:
    """Per-expert prefix arrays -- the whole schedule, precomputed.

    ``token_block_prefix`` is an exclusive prefix sum of each expert's token
    tile count, with a final total, so it has ``num_experts + 1`` entries.
    ``sf_block_prefix`` is the same over scale-factor row atoms.
    """

    token_counts: tuple[int, ...]
    token_block_prefix: tuple[int, ...]
    sf_block_prefix: tuple[int, ...]
    channel_blocks: int
    cluster_tile_tokens: int

    @property
    def num_experts(self) -> int:
        return len(self.token_counts)

    @property
    def total_token_blocks(self) -> int:
        return self.token_block_prefix[-1]

    @property
    def total_tiles(self) -> int:
        return self.total_token_blocks * self.channel_blocks

    @property
    def pool_rows(self) -> int:
        """Padded pool rows the token buffer must hold for this layout."""
        return self.total_token_blocks * self.cluster_tile_tokens

    @property
    def sf_rows(self) -> int:
        return self.sf_block_prefix[-1] * SF_ATOM_ROWS


def build_expert_layout(
    token_counts: tuple[int, ...],
    *,
    shape: ProblemShape,
    tile: TileConfig,
    phase: Phase,
    channel_ranges: int = 1,
) -> ExpertLayout:
    """Precompute the prefix arrays for one phase.

    ``token_counts`` is per *local* expert, in the order work will be visited.
    Passing a permuted order is legal and yields a permuted schedule; nothing
    downstream assumes the identity order.

    ``channel_ranges`` is how many output-channel blocks a single tile covers.
    A fused FC1 tile covers two -- the gate block and its partner up block,
    ``intermediate / mma_m`` apart -- so the channel axis it walks is
    ``intermediate`` wide, not ``2 * intermediate``.
    """
    if channel_ranges < 1:
        raise ValueError(f"channel_ranges ({channel_ranges}) must be >= 1")
    out_channels = shape.out_channels_for(phase)
    if out_channels % channel_ranges:
        raise ValueError(
            f"out_channels ({out_channels}) must divide evenly into "
            f"{channel_ranges} channel ranges"
        )
    if any(c < 0 for c in token_counts):
        raise ValueError(f"token_counts must be non-negative; got {token_counts}")

    tile_tokens = tile.cluster_tile_tokens
    token_blocks = 0
    sf_blocks = 0
    token_prefix = [0]
    sf_prefix = [0]
    for count in token_counts:
        token_blocks += ceil_div(count, tile_tokens)
        sf_blocks += ceil_div(count, SF_ATOM_ROWS)
        token_prefix.append(token_blocks)
        sf_prefix.append(sf_blocks)

    return ExpertLayout(
        token_counts=tuple(token_counts),
        token_block_prefix=tuple(token_prefix),
        sf_block_prefix=tuple(sf_prefix),
        channel_blocks=tile.channel_blocks(out_channels // channel_ranges),
        cluster_tile_tokens=tile_tokens,
    )


def decode_tile(layout: ExpertLayout, tile_index: int) -> WorkTile | None:
    """Decode a flat tile index; ``None`` past the end of the tile space.

    The channel axis factors out before the expert search: every expert has the
    same ``channel_blocks``, so ``tile_index`` splits cleanly into a *global*
    token-block index and a channel block, and only the former needs the
    per-expert search.  Keeping the token block as the slow axis means the
    ``channel_blocks`` tiles that share a token tile are consecutive, so they
    reuse that tile's B-side smem across the whole channel sweep.
    """
    if tile_index < 0:
        raise ValueError(f"tile_index must be non-negative; got {tile_index}")
    if tile_index >= layout.total_tiles:
        return None

    global_token_block, channel_block = divmod(tile_index, layout.channel_blocks)

    # Rightmost expert whose prefix start is <= global_token_block.  On device
    # this is an unrolled branch-free binary search over <= 64 entries held in
    # smem; here bisect states the same intent.
    expert = bisect.bisect_right(layout.token_block_prefix, global_token_block) - 1
    token_block = global_token_block - layout.token_block_prefix[expert]

    tile_tokens = layout.cluster_tile_tokens
    remaining = layout.token_counts[expert] - token_block * tile_tokens
    return WorkTile(
        expert=expert,
        channel_block=channel_block,
        token_block=token_block,
        pool_row=layout.token_block_prefix[expert] * tile_tokens,
        sf_row=layout.sf_block_prefix[expert] * SF_ATOM_ROWS,
        valid_tokens=max(0, min(remaining, tile_tokens)),
    )


def enumerate_tiles(layout: ExpertLayout) -> tuple[WorkTile, ...]:
    """Every tile in the space, in flat index order.  Test/oracle helper."""
    tiles = []
    for idx in range(layout.total_tiles):
        tile = decode_tile(layout, idx)
        assert tile is not None
        tiles.append(tile)
    return tuple(tiles)


def persistent_tile_indices(
    layout: ExpertLayout, *, cluster_id: int, num_clusters: int
) -> tuple[int, ...]:
    """Tile indices a persistent cluster claims under static round-robin.

    Static striding is deliberate: it needs no atomics and no cross-CTA
    broadcast, and at this shape the tile count per expert is tiny (one token
    tile), so a dynamic work queue would spend more on coordination than the
    imbalance it removes.  A readiness-ordered variant replaces this function
    alone -- the decode above is already order-agnostic.
    """
    if num_clusters <= 0:
        raise ValueError(f"num_clusters ({num_clusters}) must be positive")
    if not 0 <= cluster_id < num_clusters:
        raise ValueError(f"cluster_id ({cluster_id}) must be in [0, {num_clusters})")
    return tuple(range(cluster_id, layout.total_tiles, num_clusters))
