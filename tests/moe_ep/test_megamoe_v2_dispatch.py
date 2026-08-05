# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: cross-rank dispatch, with the peer ranks emulated in-process.

Every rank's symmetric regions are allocated as one tensor with a leading
``world`` axis, so a peer's copy is a slice of the same allocation and the
peer offset table is a difference of two real addresses.  The kernels then run
the *same* address arithmetic they would across NVLink -- only the value of
the offset differs -- so this covers multi-rank routing, per-source pool
placement and the pull permutation without NVSHMEM or multiple processes.

What it does not cover: the cross-rank barrier (here the launch order supplies
it) and the actual interconnect.  Those need the multi-process harness.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from flashinfer.moe_ep.kernel_src.megamoe_v2 import sf_layout  # noqa: E402
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import NVFP4_BLOCK  # noqa: E402


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("needs sm_100a / sm_103a")


def _ceil(a, b):
    return -(-a // b)


_HEAP_ALIGN = 128


def _heap(world: int, regions):
    """Allocate ``world`` identical symmetric heaps and return a region carver."""
    offsets = {}
    cursor = 0
    for name, nbytes in regions:
        cursor = _ceil(cursor, _HEAP_ALIGN) * _HEAP_ALIGN
        offsets[name] = cursor
        cursor += nbytes
    total = _ceil(cursor, _HEAP_ALIGN) * _HEAP_ALIGN
    buf = torch.zeros(world, total, dtype=torch.uint8, device="cuda")
    sizes = dict(regions)

    def carve(name, dtype):
        off = offsets[name]
        raw = buf[:, off : off + sizes[name]]
        return raw.view(dtype) if dtype is not torch.uint8 else raw

    return buf, carve


def _private(world: int, n: int, dtype: torch.dtype) -> torch.Tensor:
    """Per-rank array whose every rank slice starts 16 B aligned."""
    per = _ceil(n * dtype.itemsize, 16) * 16 // dtype.itemsize
    return torch.zeros(world, per, dtype=dtype, device="cuda")[:, :n]


class _Fixture:
    """One in-process EP group: per-rank inputs, symmetric heap, and results."""

    def __init__(
        self,
        *,
        world,
        local_experts,
        hidden,
        top_k,
        num_tokens,
        tile_tokens=128,
        seed=3,
    ):
        self.world = world
        self.local_experts = local_experts
        self.num_experts = world * local_experts
        self.hidden = hidden
        self.top_k = top_k
        self.num_tokens = num_tokens
        self.tile_tokens = tile_tokens
        self.max_tokens = num_tokens
        self.max_pairs = num_tokens * top_k
        self.num_k_atoms = sf_layout.num_k_atoms_for(hidden, NVFP4_BLOCK)
        self.sf_words = sf_layout.buffer_words(
            max(num_tokens, sf_layout.SF_ATOM_ROWS), num_k_atoms=self.num_k_atoms
        )

        g = torch.Generator(device="cuda").manual_seed(seed)
        dev = "cuda"
        self.activation = torch.randn(
            world, num_tokens, hidden, dtype=torch.float32, device=dev, generator=g
        ).bfloat16()
        # Skewed routing, plus some invalid slots, so the per-expert counts are
        # uneven and the "expert with no tokens" path is exercised.
        logits = (
            torch.rand(world, num_tokens, self.num_experts, device=dev, generator=g)
            ** 3
        )
        self.topk_ids = logits.topk(top_k, dim=-1).indices.to(torch.int32)
        self.topk_ids[:, ::7, -1] = -1
        self.topk_weights = torch.rand(
            world, num_tokens, top_k, dtype=torch.float32, device=dev, generator=g
        )

        # One symmetric heap per rank, regions carved out of it -- as in the
        # real workspace.  This is load-bearing, not tidiness: a peer is
        # reached by a *single* byte offset, so regions must sit at the same
        # displacement in every rank's heap.  Allocating them as separate
        # tensors gives each one its own delta and silently corrupts every
        # region but the one the offset was computed from.
        pairs = local_experts * world * self.max_pairs
        self.heap, carve = _heap(
            world,
            (
                ("peer_count", world * local_experts * 8),
                ("peer_slot", pairs * 4),
                ("peer_weight", pairs * 4),
                ("send", num_tokens * hidden // 2),
                ("send_sf", self.sf_words * 4),
            ),
        )
        self.peer_count = carve("peer_count", torch.int64)
        self.peer_slot = carve("peer_slot", torch.int32)
        self.peer_weight = carve("peer_weight", torch.float32)
        self.send_bytes = carve("send", torch.uint8).view(
            world, num_tokens, hidden // 2
        )
        self.send_sf_bytes = carve("send_sf", torch.uint8)

        # Rank-private staging and results.
        self.send_count = _private(world, self.num_experts, torch.int32)
        self.send_slot = _private(
            world, self.num_experts * self.max_pairs, torch.int32
        ).view(world, self.num_experts, self.max_pairs)
        self.send_weight = _private(
            world, self.num_experts * self.max_pairs, torch.float32
        ).view(world, self.num_experts, self.max_pairs)
        self.expert_count = _private(world, local_experts, torch.int64)
        self.rank_pool_offset = _private(world, local_experts * world, torch.int32)
        self.prefix = _private(world, local_experts + 1, torch.int32)

        self.pool_rows = _ceil(num_tokens * top_k * world, tile_tokens) * tile_tokens
        self.pool_rows += local_experts * tile_tokens
        pool_sf_rows = _ceil(self.pool_rows, sf_layout.SF_ATOM_ROWS)
        self.pool_sf_words = sf_layout.buffer_words(
            pool_sf_rows * sf_layout.SF_ATOM_ROWS, num_k_atoms=self.num_k_atoms
        )
        self.pool_bytes = torch.zeros(
            world, self.pool_rows, hidden // 2, dtype=torch.uint8, device=dev
        )
        self.pool_sf_bytes = torch.zeros(
            world, self.pool_sf_words * 4, dtype=torch.uint8, device=dev
        )
        self.pool_weight = torch.full(
            (world, self.pool_rows), -1.0, dtype=torch.float32, device=dev
        )
        self.pool_src = torch.full(
            (world, self.pool_rows), -2, dtype=torch.int64, device=dev
        )

    def peer_offsets(self, me: int) -> torch.Tensor:
        """``peer_base - local_base``, one delta covering the whole heap."""
        base = self.heap[me].data_ptr()
        return torch.tensor(
            [self.heap[r].data_ptr() - base for r in range(self.world)],
            dtype=torch.int64,
            device="cuda",
        )

    def run(self):
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as ct
        import cuda.bindings.driver as cuda

        from flashinfer.moe_ep.kernel_src.megamoe_v2 import dispatch

        mk = lambda t: ct.from_dlpack(t, assumed_align=16)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        send_fp4 = self.send_bytes.view(torch.float4_e2m1fn_x2)
        send_e4m3 = self.send_sf_bytes.view(torch.float8_e4m3fn)
        send_i32 = self.send_bytes.view(torch.int32)
        send_sf_i32 = self.send_sf_bytes.view(torch.int32)
        pool_i32 = self.pool_bytes.view(torch.int32)
        pool_sf_i32 = self.pool_sf_bytes.view(torch.int32)

        prep_kw = dict(
            hidden=self.hidden,
            num_k_atoms=self.num_k_atoms,
            top_k=self.top_k,
            num_experts=self.num_experts,
            num_ctas=8,
            warps_per_cta=4,
        )
        push_kw = dict(
            num_experts=self.num_experts,
            local_experts=self.local_experts,
            world=self.world,
            max_pairs=self.max_pairs,
            threads=128,
        )
        plan_kw = dict(
            local_experts=self.local_experts,
            world=self.world,
            tile_tokens=self.tile_tokens,
        )
        pull_kw = dict(
            local_experts=self.local_experts,
            world=self.world,
            max_pairs=self.max_pairs,
            max_tokens=self.max_tokens,
            top_k=self.top_k,
            hidden=self.hidden,
            num_k_atoms=self.num_k_atoms,
            tile_tokens=self.tile_tokens,
            num_ctas=8,
            warps_per_cta=4,
        )

        for me in range(self.world):
            args = (
                mk(self.activation[me]),
                mk(self.topk_ids[me]),
                mk(self.topk_weights[me]),
                mk(send_fp4[me]),
                mk(send_e4m3[me]),
                mk(self.send_count[me]),
                mk(self.send_slot[me]),
                mk(self.send_weight[me]),
                cutlass.Int32(self.num_tokens),
                cutlass.Float32(1.0),
                stream,
            )
            cute.compile(dispatch.dispatch_prepare, *args, **prep_kw)(*args)

        for me in range(self.world):
            args = (
                mk(self.send_count[me]),
                mk(self.send_slot[me]),
                mk(self.send_weight[me]),
                mk(self.peer_count[me]),
                mk(self.peer_slot[me]),
                mk(self.peer_weight[me]),
                mk(self.peer_offsets(me)),
                cutlass.Int32(me),
                stream,
            )
            cute.compile(dispatch.dispatch_push, *args, **push_kw)(*args)

        # Stands in for the cross-rank barrier: every rank's pushes are complete
        # before any rank plans or pulls.
        torch.cuda.synchronize()

        for me in range(self.world):
            args = (
                mk(self.peer_count[me]),
                mk(self.expert_count[me]),
                mk(self.rank_pool_offset[me]),
                mk(self.prefix[me]),
                stream,
            )
            cute.compile(dispatch.dispatch_plan, *args, **plan_kw)(*args)

        for me in range(self.world):
            args = (
                mk(send_i32[me]),
                mk(send_sf_i32[me]),
                mk(self.peer_slot[me]),
                mk(self.peer_weight[me]),
                mk(self.peer_count[me]),
                mk(self.peer_offsets(me)),
                mk(self.expert_count[me]),
                mk(self.rank_pool_offset[me]),
                mk(self.prefix[me]),
                mk(pool_i32[me]),
                mk(pool_sf_i32[me]),
                mk(self.pool_weight[me]),
                mk(self.pool_src[me]),
                stream,
            )
            cute.compile(dispatch.dispatch_pull, *args, **pull_kw)(*args)
        torch.cuda.synchronize()

    # ---- reference ----

    def expected_pairs(self, dst_rank: int, local_expert: int, src_rank: int):
        """The (token, slot) pairs one source owes one expert, as a set."""
        expert = dst_rank * self.local_experts + local_expert
        ids = self.topk_ids[src_rank]
        hit = (ids == expert).nonzero()
        return {(int(t), int(s)) for t, s in hit}


def _check(fx: _Fixture):
    """Every live pool row must be self-consistent with its provenance."""
    send_bytes = fx.send_bytes.cpu()
    send_sf = fx.send_sf_bytes.view(torch.int32).cpu()
    pool_bytes = fx.pool_bytes.cpu()
    pool_sf = fx.pool_sf_bytes.view(torch.int32).cpu()
    pool_weight = fx.pool_weight.cpu()
    pool_src = fx.pool_src.cpu()
    prefix = fx.prefix.cpu()
    expert_count = fx.expert_count.cpu()
    rank_pool_offset = fx.rank_pool_offset.cpu()
    topk_weights = fx.topk_weights.cpu()

    assert send_bytes.any(), "source quantization never ran"

    for me in range(fx.world):
        # Counts and the tile prefix are exact, not order-dependent.
        blocks = 0
        for le in range(fx.local_experts):
            total = sum(len(fx.expected_pairs(me, le, src)) for src in range(fx.world))
            assert int(expert_count[me, le]) == total, (me, le)
            assert int(prefix[me, le]) == blocks
            blocks += _ceil(total, fx.tile_tokens)
        assert int(prefix[me, fx.local_experts]) == blocks

        for le in range(fx.local_experts):
            base = int(prefix[me, le]) * fx.tile_tokens
            for src in range(fx.world):
                expected = fx.expected_pairs(me, le, src)
                start = base + int(rank_pool_offset[me, le * fx.world + src])
                rows = range(start, start + len(expected))

                seen = set()
                for row in rows:
                    packed = int(pool_src[me, row])
                    r_src, rest = divmod(packed, fx.max_tokens * fx.top_k)
                    token, slot = divmod(rest, fx.top_k)
                    assert r_src == src, (me, le, row, r_src, src)
                    seen.add((token, slot))
                    # Codes moved verbatim from the source's send buffer.
                    assert torch.equal(pool_bytes[me, row], send_bytes[src, token]), (
                        f"codes differ at rank {me} row {row}"
                    )
                    for ka in range(fx.num_k_atoms):
                        dst_w = sf_layout.word_offset(
                            row, ka, num_k_atoms=fx.num_k_atoms
                        )
                        src_w = sf_layout.word_offset(
                            token, ka, num_k_atoms=fx.num_k_atoms
                        )
                        assert pool_sf[me, dst_w] == send_sf[src, src_w], (
                            f"scale differs at rank {me} row {row} atom {ka}"
                        )
                    assert pool_weight[me, row] == topk_weights[src, token, slot]
                assert seen == expected, (me, le, src)

            # Padding rows must be inert, not merely unread.
            total = int(expert_count[me, le])
            end = int(prefix[me, le + 1]) * fx.tile_tokens
            for row in range(base + total, end):
                assert pool_weight[me, row] == 0.0
                assert int(pool_src[me, row]) == -1
                for ka in range(fx.num_k_atoms):
                    w = sf_layout.word_offset(row, ka, num_k_atoms=fx.num_k_atoms)
                    assert int(pool_sf[me, w]) == 0


@pytest.mark.parametrize("world", [1, 4])
def test_dispatch_places_every_token(world):
    _require_blackwell()
    fx = _Fixture(world=world, local_experts=2, hidden=512, top_k=2, num_tokens=192)
    fx.run()
    _check(fx)


def test_dispatch_handles_empty_expert():
    """An expert nobody routes to must produce a zero-width, zero-tile segment."""
    _require_blackwell()
    fx = _Fixture(
        world=2, local_experts=3, hidden=512, top_k=1, num_tokens=128, seed=11
    )
    # Force every pair away from local expert 0 of rank 0 (global expert 0).
    fx.topk_ids[fx.topk_ids == 0] = 1
    fx.run()
    _check(fx)
    assert int(fx.expert_count[0, 0]) == 0
    assert int(fx.prefix[0, 0]) == int(fx.prefix[0, 1])


def test_dispatch_all_slots_invalid():
    """With no live pairs at all, every segment is empty and nothing is written."""
    _require_blackwell()
    fx = _Fixture(world=2, local_experts=2, hidden=512, top_k=2, num_tokens=128)
    fx.topk_ids.fill_(-1)
    fx.run()
    _check(fx)
    assert int(fx.expert_count.sum()) == 0
