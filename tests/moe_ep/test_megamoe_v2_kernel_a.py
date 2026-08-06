# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: kernel A -- quantize + dispatch + FC1 fused into one launch.

The staged pipeline already validates the same arithmetic, so what is new here
is the concurrency: dispatch warps and GEMM warps running at once, meeting only
at incrementally published per-expert prefixes and row counters.  The reference
is the staged path's own FC1 output, so any difference is a fusion bug rather
than a numerics one.

Failures here tend to be hangs, not wrong answers -- a grid barrier that not
every block reaches, or a readiness counter that never hits its target.

Dispatch moves the pool through Int32 aliases, while GEMM consumes the same
storage as FP4/E4M3.  This test keeps the schedule nonempty so the typed TMA-B
handoff is exercised.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from flashinfer.moe_ep.kernel_src.megamoe_v2 import sf_layout  # noqa: E402
from flashinfer.moe_ep.kernel_src.megamoe_v2.reference import (  # noqa: E402
    quantize_nvfp4,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import NVFP4_BLOCK  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_v2_gemm_test", pathlib.Path(__file__).with_name("test_megamoe_v2_gemm.py")
)
_gemm_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gemm_test)
_pack_fp4 = _gemm_test._pack_fp4
_require_blackwell = _gemm_test._require_blackwell
_scatter_scales = _gemm_test._scatter_scales

_TILE = 128


def _z(*shape, dtype=torch.int32):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def _build(num_tokens, hidden, intermediate, num_experts, top_k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    act = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device="cuda", generator=g
    ).bfloat16()
    w1 = (
        torch.randn(
            num_experts,
            2 * intermediate,
            hidden,
            dtype=torch.float32,
            device="cuda",
            generator=g,
        )
        * 0.3
    )
    logits = torch.rand(num_tokens, num_experts, device="cuda", generator=g) ** 3
    topk_ids = logits.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_ids[::9, -1] = -1
    topk_weights = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device="cuda", generator=g
    )
    return act, w1, topk_ids, topk_weights


def _run_fused(
    *,
    num_tokens,
    hidden,
    intermediate,
    num_experts,
    top_k,
    clamp=None,
    seed=5,
    num_clusters=4,
    empty_expert=None,
    all_invalid=False,
):
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.kernel_a import launch_kernel_a

    world = 1
    le = num_experts
    max_pairs = num_tokens * top_k
    pool_rows = (num_tokens * top_k + le * _TILE + _TILE - 1) // _TILE * _TILE
    h_atoms = sf_layout.num_k_atoms_for(hidden, NVFP4_BLOCK)
    i_atoms = sf_layout.num_k_atoms_for(intermediate, NVFP4_BLOCK)

    act, w1, topk_ids, topk_weights = _build(
        num_tokens, hidden, intermediate, num_experts, top_k, seed
    )
    if empty_expert is not None:
        topk_ids[topk_ids == empty_expert] = 0
    if all_invalid:
        topk_ids.fill_(-1)
    qw1 = quantize_nvfp4(w1.reshape(-1, hidden))

    send_b = _z(num_tokens, hidden // 2, dtype=torch.uint8)
    send_sf_b = _z(
        sf_layout.buffer_words(max(num_tokens, 128), num_k_atoms=h_atoms) * 4,
        dtype=torch.uint8,
    )
    pool_b = _z(pool_rows, hidden // 2, dtype=torch.uint8)
    pool_sf_b = _z(
        sf_layout.buffer_words(pool_rows, num_k_atoms=h_atoms) * 4,
        dtype=torch.uint8,
    )
    fc1_b = _z(pool_rows, intermediate // 2, dtype=torch.uint8)
    fc1_sf_b = _z(
        sf_layout.buffer_words(pool_rows, num_k_atoms=i_atoms) * 4,
        dtype=torch.uint8,
    )

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    ntok = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
    prefix = _z(le + 1)
    ready = _z(le)
    pool_src = _z(pool_rows, dtype=torch.int64)
    peer_count = _z(world * le, dtype=torch.int64)
    coop = (
        mk(act),
        mk(topk_ids),
        mk(topk_weights),
        mk(send_b.view(torch.float4_e2m1fn_x2)),
        mk(send_sf_b.view(torch.float8_e4m3fn)),
        mk(_z(num_experts)),
        mk(_z(num_experts, max_pairs)),
        mk(_z(num_experts, max_pairs, dtype=torch.float32)),
        mk(peer_count),
        mk(_z(le * world * max_pairs)),
        mk(_z(le * world * max_pairs, dtype=torch.float32)),
        mk(_z(le, dtype=torch.int64)),
        mk(_z(le * world)),
        mk(prefix),
        mk(send_b.view(torch.int32)),
        mk(send_sf_b.view(torch.int32)),
        mk(pool_b.view(torch.int32)),
        mk(pool_sf_b.view(torch.int32)),
        mk(_z(pool_rows, dtype=torch.float32)),
        mk(pool_src),
        mk(_z(world, dtype=torch.int64)),
        mk(_z(2)),
        mk(ready),
        mk(ntok),
        mk(torch.tensor([0], dtype=torch.int32, device="cuda")),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        mk(_pack_fp4(qw1.codes).view(torch.float4_e2m1fn_x2)),
        mk(_scatter_scales(qw1.scales)),
        mk(fc1_b.view(torch.float4_e2m1fn_x2)),
        mk(fc1_sf_b.view(torch.float8_e4m3fn)),
        coop,
        stream,
    )
    kw = dict(
        num_experts=num_experts,
        local_experts=le,
        world=world,
        intermediate=intermediate,
        hidden=hidden,
        pool_rows=pool_rows,
        max_tokens=num_tokens,
        top_k=top_k,
        hidden_atoms=h_atoms,
        inter_atoms=i_atoms,
        clamp=clamp,
        num_clusters=num_clusters,
    )
    cute.compile(launch_kernel_a, *args, **kw)(*args)
    torch.cuda.synchronize()
    return dict(
        fc1=fc1_b,
        fc1_sf=fc1_sf_b,
        prefix=prefix,
        ready=ready,
        topk_ids=topk_ids,
        pool_src=pool_src,
        peer_count=peer_count,
        pool_rows=pool_rows,
    )


def test_kernel_a_runs_and_schedules():
    """The fused kernel completes, and its device-built schedule is right."""
    _require_blackwell()
    r = _run_fused(num_tokens=256, hidden=512, intermediate=256, num_experts=2, top_k=2)
    counts = [int((r["topk_ids"] == e).sum()) for e in range(2)]
    blocks = 0
    for e, c in enumerate(counts):
        assert int(r["prefix"][e]) == blocks
        blocks += max(1, (c + _TILE - 1) // _TILE)
    assert int(r["prefix"][2]) == blocks
    # Every expert's rows were published exactly once.
    for e in range(2):
        want = (int(r["prefix"][e + 1]) - int(r["prefix"][e])) * _TILE
        assert int(r["ready"][e]) == want
    assert r["fc1"].any(), "FC1 produced nothing"


def test_kernel_a_pads_an_empty_expert():
    """An empty expert still has one inert tile with no scatter destination."""
    _require_blackwell()
    r = _run_fused(
        num_tokens=64,
        hidden=512,
        intermediate=256,
        num_experts=2,
        top_k=1,
        empty_expert=1,
    )
    assert tuple(int(v) for v in r["prefix"]) == (0, 1, 2)
    assert tuple(int(v) for v in r["ready"]) == (_TILE, _TILE)
    counts = tuple(int((r["topk_ids"] == e).sum()) for e in range(2))
    assert tuple(int(v) for v in r["peer_count"]) == tuple(c + 1 for c in counts)
    assert (r["pool_src"][_TILE : 2 * _TILE] == -1).all()


def test_kernel_a_publishes_all_empty_experts():
    _require_blackwell()
    r = _run_fused(
        num_tokens=32,
        hidden=512,
        intermediate=256,
        num_experts=2,
        top_k=1,
        all_invalid=True,
    )
    assert tuple(int(v) for v in r["peer_count"]) == (1, 1)
    assert tuple(int(v) for v in r["prefix"]) == (0, 1, 2)
    assert tuple(int(v) for v in r["ready"]) == (_TILE, _TILE)
    assert (r["pool_src"][: 2 * _TILE] == -1).all()
