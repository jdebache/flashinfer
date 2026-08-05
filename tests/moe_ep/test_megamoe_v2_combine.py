# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: FC2 with the fused combine push, then the source-side reduce.

Peer ranks are emulated in-process the same way ``test_megamoe_v2_dispatch.py``
does it -- one symmetric heap per rank, peers reached by a real address delta --
so the scatter runs the address arithmetic it would run over NVLink.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from flashinfer.moe_ep.kernel_src.megamoe_v2.reference import (  # noqa: E402
    quantize_nvfp4,
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_v2_{name}", pathlib.Path(__file__).with_name(f"test_megamoe_v2_{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gemm_test = _load("gemm")
_fc2_test = _load("fc2")
_pack_fp4 = _gemm_test._pack_fp4
_require_blackwell = _gemm_test._require_blackwell
_scatter_scales = _gemm_test._scatter_scales


def _plan(world, counts, intermediate, hidden, top_k, seed, slack=0):
    """Assign every live pool row a distinct ``(src rank, token, slot)``.

    ``slack`` adds tokens nothing routes to, so some ``(token, slot)`` slots
    stay unwritten -- which is what the skip logic has to survive.
    """
    tile, shape, layout, pool_rows = _fc2_test._make(counts, intermediate, hidden, seed)
    live = []  # (pool_row, expert)
    for e, cnt in enumerate(counts):
        base = layout.token_block_prefix[e] * tile.cluster_tile_tokens
        live.extend((base + i, e) for i in range(cnt))

    num_tokens = (-(-len(live) // top_k) if live else 1) + slack
    pool_src = torch.full((pool_rows,), -1, dtype=torch.int64, device="cuda")
    topk_ids = torch.full(
        (world, num_tokens, top_k), -1, dtype=torch.int32, device="cuda"
    )
    routing = []  # (src, token, slot, pool_row)
    for idx, (row, expert) in enumerate(live):
        token, slot = divmod(idx, top_k)
        src = idx % world
        pool_src[row] = (src * num_tokens * top_k) + token * top_k + slot
        topk_ids[src, token, slot] = expert
        routing.append((src, token, slot, row))
    return tile, layout, pool_rows, num_tokens, pool_src, topk_ids, routing


def _run(
    world, counts, intermediate, hidden, top_k, *, num_clusters=2, seed=31, slack=0
):
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.combine import (
        combine_reduce,
        launch_fc2_combine,
    )

    num_experts = len(counts)
    (tile, layout, pool_rows, num_tokens, pool_src, topk_ids, routing) = _plan(
        world, counts, intermediate, hidden, top_k, seed, slack
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    w2 = (
        torch.randn(
            num_experts,
            hidden,
            intermediate,
            dtype=torch.float32,
            device="cuda",
            generator=g,
        )
        * 0.3
    )
    # Each rank owns the same expert weights here; only the routing differs,
    # which is all the combine path depends on.
    h = (
        torch.randn(
            pool_rows, intermediate, dtype=torch.float32, device="cuda", generator=g
        )
        * 0.3
    )
    qw = quantize_nvfp4(w2.reshape(-1, intermediate))
    qh = quantize_nvfp4(h)

    # Symmetric landing buffer, one heap per rank.
    rows = top_k * num_tokens
    combine = torch.zeros(world, rows, hidden, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(world, num_tokens, hidden, dtype=torch.bfloat16, device="cuda")

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    prefix = torch.tensor(layout.token_block_prefix, dtype=torch.int32, device="cuda")

    def offsets(me):
        base = combine[me].data_ptr()
        return torch.tensor(
            [combine[r].data_ptr() - base for r in range(world)],
            dtype=torch.int64,
            device="cuda",
        )

    # Every rank runs the same expert pool; each pushes its rows to the source
    # rank recorded in pool_src.  One rank suffices to cover the scatter, but
    # running all of them also covers many writers landing in one buffer.
    args = (
        mk(_pack_fp4(qw.codes).view(torch.float4_e2m1fn_x2)),
        mk(_pack_fp4(qh.codes).view(torch.float4_e2m1fn_x2)),
        mk(_scatter_scales(qw.scales)),
        mk(_scatter_scales(qh.scales)),
        mk(combine[0]),
        mk(pool_src),
        mk(offsets(0)),
        mk(prefix),
        stream,
    )
    kw = dict(
        num_experts=num_experts,
        intermediate=intermediate,
        hidden=hidden,
        pool_rows=pool_rows,
        max_tokens=num_tokens,
        top_k=top_k,
        num_clusters=num_clusters,
    )
    cute.compile(launch_fc2_combine, *args, **kw)(*args)
    torch.cuda.synchronize()

    rkw = dict(
        max_tokens=num_tokens,
        top_k=top_k,
        hidden=hidden,
        num_experts=num_experts,
        num_ctas=8,
        threads=128,
    )
    for me in range(world):
        rargs = (
            mk(combine[me]),
            mk(topk_ids[me]),
            mk(out[me]),
            cutlass.Int32(num_tokens),
            stream,
        )
        cute.compile(combine_reduce, *rargs, **rkw)(*rargs)
    torch.cuda.synchronize()

    # Reference: dequantized FC2 per live row, summed into its (rank, token).
    wd = qw.dequantize().reshape(num_experts, hidden, intermediate)
    hd = qh.dequantize()
    expected = torch.zeros(
        world, num_tokens, hidden, dtype=torch.float32, device="cuda"
    )
    for src, token, _slot, row in routing:
        expert = int(topk_ids[src, token, _slot])
        expected[src, token] += (wd[expert] @ hd[row]).bfloat16().float()
    return out.float(), expected, combine, routing, topk_ids, num_tokens


@pytest.mark.parametrize("world,top_k", [(1, 1), (1, 4), (4, 2)])
def test_combine_sums_every_contribution(world, top_k):
    _require_blackwell()
    got, expected, *_ = _run(world, (128, 128), 256, 512, top_k)
    assert expected.abs().sum() > 0
    torch.testing.assert_close(got, expected, atol=8e-2, rtol=8e-3)


def test_combine_leaves_invalid_slots_out():
    """Stale bytes in an unrouted slot must not reach the sum.

    The landing buffer is deliberately not cleared between launches, so the
    reduce skips slots by their routing id.  Poisoning an unused slot is the
    only way to tell a correct skip from a buffer that merely happened to be
    zero.
    """
    _require_blackwell()
    got, expected, combine, routing, topk_ids, num_tokens = _run(
        1, (128,), 256, 512, top_k=4, slack=3
    )
    torch.testing.assert_close(got, expected, atol=8e-2, rtol=8e-3)

    used = {(t, s) for _src, t, s, _row in routing}
    poisoned = 0
    for slot in range(4):
        for token in range(num_tokens):
            if (token, slot) not in used:
                combine[0, slot * num_tokens + token].fill_(1e4)
                poisoned += 1
    assert poisoned > 0, "test shape leaves no unrouted slot to poison"

    again = _reduce_only(
        combine[0], topk_ids[0], num_tokens, hidden=512, top_k=4, num_experts=1
    )
    torch.testing.assert_close(again, got[0], atol=0, rtol=0)


def _reduce_only(combine_buf, topk_ids, num_tokens, *, hidden, top_k, num_experts):
    """Re-run just the reduce over an existing landing buffer."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.combine import combine_reduce

    out = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (mk(combine_buf), mk(topk_ids), mk(out), cutlass.Int32(num_tokens), stream)
    cute.compile(
        combine_reduce,
        *args,
        max_tokens=num_tokens,
        top_k=top_k,
        hidden=hidden,
        num_experts=num_experts,
        num_ctas=8,
        threads=128,
    )(*args)
    torch.cuda.synchronize()
    return out.float()
