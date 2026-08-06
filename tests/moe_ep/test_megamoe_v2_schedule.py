# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only tests for the v2 schedule and workspace layout.

These cover the invariants the device code will *assume* rather than check:
tile-space partitioning, pool-row disjointness, and the reset-prefix contract.
No CUDA, no compile -- so they run in milliseconds and catch layout mistakes
before they turn into a silent memory-corruption bug on device.
"""

from __future__ import annotations

import itertools

import pytest

from flashinfer.moe_ep.kernel_src.megamoe_v2.layout import (
    local_layout,
    shared_layout,
    workspace_sizes,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.schedule import (
    build_expert_layout,
    decode_tile,
    enumerate_tiles,
    persistent_tile_indices,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (
    CommConfig,
    EpTopology,
    EpilogueConfig,
    KernelConfig,
    Phase,
    ProblemShape,
    SF_ATOM_ROWS,
    TileConfig,
    ceil_div,
)

# The reference deployment shape, plus small/degenerate ones.
_SHAPES = (
    ProblemShape(
        hidden=7168, intermediate=4096, num_experts=64, top_k=4,
        max_tokens_per_rank=96,
    ),
    ProblemShape(
        hidden=2048, intermediate=1024, num_experts=8, top_k=4,
        max_tokens_per_rank=64,
    ),
    ProblemShape(
        hidden=2880, intermediate=2880, num_experts=8, top_k=2,
        max_tokens_per_rank=16,
    ),
)

_COUNT_PATTERNS = (
    (24,) * 16,                       # the reference: uniform, one tile each
    (0,) * 16,                        # every expert empty
    (0, 0, 1, 0, 127, 128, 129, 0),   # tile-boundary neighbourhood
    (300, 1, 0, 512, 7, 0, 0, 65),    # skewed
    (1,),                             # single expert
)


def _tile_cfg(**kw) -> TileConfig:
    return TileConfig(**kw)


@pytest.mark.parametrize("phase", list(Phase))
@pytest.mark.parametrize("counts", _COUNT_PATTERNS)
def test_tile_space_is_an_exact_partition(phase, counts):
    """Every (expert, channel, token block) appears exactly once."""
    shape = _SHAPES[0]
    tile = _tile_cfg()
    layout = build_expert_layout(counts, shape=shape, tile=tile, phase=phase)

    tiles = enumerate_tiles(layout)
    keys = [(t.expert, t.channel_block, t.token_block) for t in tiles]
    assert len(keys) == len(set(keys)), "tile space contains duplicates"

    expected = {
        (e, c, b)
        for e, count in enumerate(counts)
        for b in range(ceil_div(count, tile.cluster_tile_tokens))
        for c in range(layout.channel_blocks)
    }
    assert set(keys) == expected


@pytest.mark.parametrize("counts", _COUNT_PATTERNS)
def test_valid_tokens_sum_to_the_real_token_count(counts):
    """Summing a tile column's valid tokens recovers each expert's count."""
    shape = _SHAPES[0]
    tile = _tile_cfg()
    layout = build_expert_layout(counts, shape=shape, tile=tile, phase=Phase.FC1)

    for expert, count in enumerate(counts):
        # Fix one channel block so each token tile is counted once.
        total = sum(
            t.valid_tokens
            for t in enumerate_tiles(layout)
            if t.expert == expert and t.channel_block == 0
        )
        assert total == count, f"expert {expert}: {total} != {count}"


@pytest.mark.parametrize("counts", _COUNT_PATTERNS)
def test_pool_segments_are_disjoint_and_ordered(counts):
    """Padded per-expert pool segments never overlap."""
    shape = _SHAPES[0]
    tile = _tile_cfg()
    layout = build_expert_layout(counts, shape=shape, tile=tile, phase=Phase.FC1)

    end = 0
    for expert, count in enumerate(counts):
        start = layout.token_block_prefix[expert] * layout.cluster_tile_tokens
        assert start >= end, f"expert {expert} pool segment overlaps its predecessor"
        span = ceil_div(count, layout.cluster_tile_tokens) * layout.cluster_tile_tokens
        assert start + span <= layout.pool_rows
        end = start + span

    sf_end = 0
    for expert, count in enumerate(counts):
        start = layout.sf_block_prefix[expert] * SF_ATOM_ROWS
        assert start >= sf_end
        sf_end = start + ceil_div(count, SF_ATOM_ROWS) * SF_ATOM_ROWS


@pytest.mark.parametrize("counts", _COUNT_PATTERNS)
def test_tile_pool_row_matches_its_expert_segment(counts):
    shape = _SHAPES[0]
    tile = _tile_cfg()
    layout = build_expert_layout(counts, shape=shape, tile=tile, phase=Phase.FC1)
    for t in enumerate_tiles(layout):
        assert t.pool_row == layout.token_block_prefix[t.expert] * tile.cluster_tile_tokens
        assert t.sf_row == layout.sf_block_prefix[t.expert] * SF_ATOM_ROWS


def test_decode_past_the_end_returns_none():
    layout = build_expert_layout(
        (24,) * 4, shape=_SHAPES[0], tile=_tile_cfg(), phase=Phase.FC1
    )
    assert decode_tile(layout, layout.total_tiles) is None
    assert decode_tile(layout, layout.total_tiles + 7) is None
    with pytest.raises(ValueError):
        decode_tile(layout, -1)


@pytest.mark.parametrize("num_clusters", (1, 3, 8, 148))
@pytest.mark.parametrize("counts", _COUNT_PATTERNS)
def test_persistent_striding_covers_every_tile_once(counts, num_clusters):
    """Static round-robin across clusters is an exact cover."""
    layout = build_expert_layout(
        counts, shape=_SHAPES[0], tile=_tile_cfg(), phase=Phase.FC2
    )
    claimed = [
        idx
        for cluster in range(num_clusters)
        for idx in persistent_tile_indices(
            layout, cluster_id=cluster, num_clusters=num_clusters
        )
    ]
    assert sorted(claimed) == list(range(layout.total_tiles))


def test_expert_order_is_free():
    """Permuting the expert order permutes the schedule and nothing else.

    This is the property v1's cumulative-offset walk could not provide, and it
    is what a readiness-ordered variant would need.
    """
    counts = (7, 0, 260, 33)
    shape, tile = _SHAPES[0], _tile_cfg()
    identity = build_expert_layout(counts, shape=shape, tile=tile, phase=Phase.FC1)
    permuted_counts = (counts[2], counts[0], counts[3], counts[1])
    permuted = build_expert_layout(
        permuted_counts, shape=shape, tile=tile, phase=Phase.FC1
    )

    assert permuted.total_tiles == identity.total_tiles
    assert permuted.pool_rows == identity.pool_rows
    # Same multiset of (valid_tokens, channel_block) work, just relabelled.
    as_work = lambda lay, order: sorted(
        (order[t.expert], t.channel_block, t.token_block, t.valid_tokens)
        for t in enumerate_tiles(lay)
    )
    assert as_work(identity, (0, 1, 2, 3)) == as_work(permuted, (2, 0, 3, 1))


# --------------------------------------------------------------------------
# Workspace layout
# --------------------------------------------------------------------------


def _config(shape: ProblemShape, phase: Phase, world_size: int = 4) -> KernelConfig:
    return KernelConfig(
        shape=shape,
        topology=EpTopology(rank=0, world_size=world_size),
        phase=phase,
        tile=_tile_cfg(),
        comm=CommConfig(),
        epilogue=EpilogueConfig(),
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_both_phases_agree_on_workspace_sizes(shape):
    """A and B address one allocation, so their layouts must be identical."""
    a = workspace_sizes(_config(shape, Phase.FC1))
    b = workspace_sizes(_config(shape, Phase.FC2))
    assert a == b


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("which", (local_layout, shared_layout))
def test_regions_are_aligned_and_non_overlapping(shape, which):
    layout = which(_config(shape, Phase.FC1))
    prev_end = 0
    for region, offset in zip(layout.regions, layout.offsets, strict=True):
        assert offset % region.align == 0, f"{region.name} misaligned"
        assert offset >= prev_end, f"{region.name} overlaps its predecessor"
        prev_end = offset + region.nbytes
    assert layout.total_bytes >= prev_end


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("which", (local_layout, shared_layout))
def test_reset_prefix_is_exactly_the_resettable_regions(shape, which):
    layout = which(_config(shape, Phase.FC1))
    for region, offset in zip(layout.regions, layout.offsets, strict=True):
        inside = offset + region.nbytes <= layout.reset_prefix_bytes
        assert inside == region.resettable, (
            f"{region.name}: resettable={region.resettable} but "
            f"{'inside' if inside else 'outside'} the reset prefix"
        )


@pytest.mark.parametrize("shape", _SHAPES)
def test_barrier_phase_survives_the_reset(shape):
    """The sense-reversing phase must not be zeroed between launches."""
    layout = local_layout(_config(shape, Phase.FC1))
    assert layout.offset_of("barrier_phase") >= layout.reset_prefix_bytes
    shared = shared_layout(_config(shape, Phase.FC1))
    assert shared.offset_of("barrier_signal") >= shared.reset_prefix_bytes


@pytest.mark.parametrize("shape", _SHAPES)
def test_pool_holds_the_worst_case_dispatch(shape):
    """Every (token, expert) pair every peer could send must fit."""
    for world_size in (1, 2, 4, 8):
        if shape.num_experts % world_size:
            continue
        cfg = _config(shape, Phase.FC1, world_size=world_size)
        worst_pairs = min(
            shape.max_tokens_per_rank * world_size * shape.top_k,
            shape.max_tokens_per_rank * world_size * cfg.experts_per_rank,
        )
        counts = _worst_case_counts(worst_pairs, cfg.experts_per_rank)
        layout = build_expert_layout(
            counts, shape=shape, tile=cfg.tile, phase=Phase.FC1
        )
        assert layout.pool_rows <= cfg.pool_token_capacity, (
            f"world_size={world_size}: pool needs {layout.pool_rows} rows but "
            f"capacity is {cfg.pool_token_capacity}"
        )


def test_pool_holds_minimum_tiles_for_empty_experts():
    shape = ProblemShape(
        hidden=512,
        intermediate=256,
        num_experts=4,
        top_k=1,
        max_tokens_per_rank=129,
    )
    cfg = _config(shape, Phase.FC1, world_size=1)
    counts = (129, 0, 0, 0)
    needed = sum(
        max(1, ceil_div(count, cfg.tile.cluster_tile_tokens))
        * cfg.tile.cluster_tile_tokens
        for count in counts
    )
    assert needed <= cfg.pool_token_capacity


def _worst_case_counts(total_pairs: int, num_experts: int) -> tuple[int, ...]:
    """Spread that maximises padding waste: each expert one token over a tile."""
    per = total_pairs // num_experts
    counts = [per] * num_experts
    counts[0] += total_pairs - per * num_experts
    return tuple(counts)


def test_config_name_separates_the_two_phases():
    shape = _SHAPES[0]
    a, b = _config(shape, Phase.FC1), _config(shape, Phase.FC2)
    assert a.name() != b.name(), "A and B would collide in the compile cache"
    # Rank is deployment state, not codegen: all ranks share one artifact.
    other_rank = KernelConfig(
        shape=shape,
        topology=EpTopology(rank=3, world_size=4),
        phase=Phase.FC1,
        tile=_tile_cfg(),
    )
    assert other_rank.name() == a.name()


def test_shape_rejects_inconsistent_geometry():
    with pytest.raises(ValueError):
        ProblemShape(hidden=100, intermediate=1024, num_experts=8, top_k=2,
                     max_tokens_per_rank=16)  # hidden % 16 != 0
    with pytest.raises(ValueError):
        ProblemShape(hidden=2048, intermediate=1024, num_experts=8, top_k=9,
                     max_tokens_per_rank=16)  # top_k > num_experts


def test_tile_config_rejects_unsupported_geometry():
    with pytest.raises(ValueError):
        TileConfig(mma_m=128, two_cta=True)   # per-CTA M would be 64
    with pytest.raises(ValueError):
        TileConfig(mma_k=96)                  # not a multiple of the 64-elem SF atom
    with pytest.raises(ValueError):
        TileConfig(mma_m=256, cluster_m=3, two_cta=True)


def test_gate_up_is_twice_intermediate():
    """The rename exists because v1 overloaded one field with both meanings."""
    for shape in _SHAPES:
        assert shape.gate_up == 2 * shape.intermediate
        assert shape.out_channels_for(Phase.FC1) == shape.gate_up
        assert shape.out_channels_for(Phase.FC2) == shape.hidden
        assert shape.k_for(Phase.FC1) == shape.hidden
        assert shape.k_for(Phase.FC2) == shape.intermediate


# --------------------------------------------------------------------------
# Scale-factor atom layout (hardware ABI -- exhaustively checkable)
# --------------------------------------------------------------------------


def test_byte_in_atom_is_a_bijection_onto_the_atom():
    """The swizzle must cover all 512 bytes exactly once, or scales collide."""
    from flashinfer.moe_ep.kernel_src.megamoe_v2.sf_layout import (
        SF_ATOM_BYTES,
        byte_in_atom,
    )
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (
        SF_ATOM_BLOCKS,
        SF_ATOM_ROWS,
    )

    seen = [
        byte_in_atom(t, k)
        for t in range(SF_ATOM_ROWS)
        for k in range(SF_ATOM_BLOCKS)
    ]
    assert sorted(seen) == list(range(SF_ATOM_BYTES))


def test_word_offset_is_injective_across_rows_and_k_atoms():
    from flashinfer.moe_ep.kernel_src.megamoe_v2.sf_layout import (
        buffer_words,
        word_offset,
    )
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import SF_ATOM_ROWS

    num_k_atoms = 7
    rows = 3 * SF_ATOM_ROWS
    seen = {
        word_offset(row, k, num_k_atoms=num_k_atoms)
        for row in range(rows)
        for k in range(num_k_atoms)
    }
    assert len(seen) == rows * num_k_atoms
    assert max(seen) < buffer_words(rows, num_k_atoms=num_k_atoms)


def test_word_offset_rejects_out_of_range_k_atom():
    from flashinfer.moe_ep.kernel_src.megamoe_v2.sf_layout import word_offset

    with pytest.raises(ValueError):
        word_offset(0, 4, num_k_atoms=4)
    with pytest.raises(ValueError):
        word_offset(-1, 0, num_k_atoms=4)


def test_num_k_atoms_matches_the_reference_shape():
    from flashinfer.moe_ep.kernel_src.megamoe_v2.sf_layout import num_k_atoms_for
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import NVFP4_BLOCK

    # hidden 7168 -> 448 blocks of 16 -> 112 atoms of 4 banks.
    assert num_k_atoms_for(7168, NVFP4_BLOCK) == 112
    # intermediate 4096 -> 256 blocks -> 64 atoms.
    assert num_k_atoms_for(4096, NVFP4_BLOCK) == 64
    # Ragged K rounds up rather than truncating.
    assert num_k_atoms_for(NVFP4_BLOCK * 5, NVFP4_BLOCK) == 2
