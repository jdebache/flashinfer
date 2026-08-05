# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the v2 torch oracle.

The oracle is the executable spec for the NVFP4 encoding and the EP dataflow,
so it needs its own tests: a bug here would be silently "confirmed" by the
kernel it is meant to check.  Runs on CPU where torch allows it; the E4M3 cast
needs CUDA, so those tests skip without a GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flashinfer.moe_ep.kernel_src.megamoe_v2.reference import (  # noqa: E402
    _FP4_LEVELS,
    Nvfp4Tensor,
    moe_reference,
    plan_dispatch,
    quantize_nvfp4,
    roundtrip_nvfp4,
    swiglu,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (  # noqa: E402
    NVFP4_BLOCK,
    EpTopology,
    EpilogueConfig,
    ProblemShape,
)


def _device():
    if not torch.cuda.is_available():
        pytest.skip("float8_e4m3fn cast needs CUDA")
    return torch.device("cuda")


# --------------------------------------------------------------------------
# NVFP4 encoding
# --------------------------------------------------------------------------


def test_quantize_reproduces_exactly_representable_values():
    """A block whose values are already E2M1 * an E4M3 scale survives exactly."""
    dev = _device()
    levels = torch.tensor(_FP4_LEVELS, dtype=torch.float32, device=dev)
    # 16 values drawn from the level table, scaled by an exact power of two
    # (exactly representable in E4M3), so the round trip must be lossless.
    block = torch.cat([levels, -levels])[:NVFP4_BLOCK] * 2.0
    out = roundtrip_nvfp4(block.reshape(1, NVFP4_BLOCK))
    torch.testing.assert_close(out.reshape(-1), block, atol=0, rtol=0)


def test_all_zero_block_stays_zero():
    """The zero-scale guard: no NaN from dividing by a zero block scale."""
    dev = _device()
    block = torch.zeros(1, NVFP4_BLOCK, dtype=torch.float32, device=dev)
    q = quantize_nvfp4(block)
    assert float(q.scales.abs().max()) == 0.0
    out = q.dequantize()
    assert torch.isfinite(out).all(), "zero block produced non-finite output"
    assert float(out.abs().max()) == 0.0


def test_codes_are_always_representable_magnitudes():
    dev = _device()
    x = torch.randn(64, NVFP4_BLOCK * 4, dtype=torch.float32, device=dev) * 17.0
    codes = quantize_nvfp4(x).codes.abs().unique()
    allowed = torch.tensor(_FP4_LEVELS, dtype=torch.float32, device=dev)
    for value in codes.tolist():
        assert any(abs(value - a) < 1e-6 for a in allowed.tolist()), (
            f"{value} is not an E2M1 magnitude"
        )


def test_roundtrip_error_is_bounded_by_the_block_scale():
    """Every element lands within half a quantization step of its block."""
    dev = _device()
    x = torch.randn(128, NVFP4_BLOCK * 8, dtype=torch.float32, device=dev)
    q = quantize_nvfp4(x)
    out = q.dequantize()
    # Coarsest gap in the level table is 6 - 4 = 2, so half-step is 1.0 in
    # code space; convert to value space with the per-block scale.
    step = q.scales.repeat_interleave(NVFP4_BLOCK, dim=-1) * 1.0
    assert torch.all((out - x).abs() <= step + 1e-5)


def test_scaling_is_norm_const_invariant():
    """norm_const cancels in the round trip (it only repositions the E4M3)."""
    dev = _device()
    x = torch.randn(32, NVFP4_BLOCK * 3, dtype=torch.float32, device=dev)
    a = roundtrip_nvfp4(x, norm_const=1.0)
    b = roundtrip_nvfp4(x, norm_const=4.0)
    # Both are exact power-of-two rescalings of the same E4M3 grid, so they
    # agree wherever the scale did not saturate.
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_dequantize_matches_manual_expansion():
    dev = _device()
    codes = torch.tensor([[1.0, -2.0] + [0.0] * (NVFP4_BLOCK - 2)], device=dev)
    scales = torch.tensor([[0.5]], device=dev)
    t = Nvfp4Tensor(codes=codes, scales=scales, norm_const=2.0)
    out = t.dequantize()
    assert float(out[0, 0]) == pytest.approx(1.0 * 0.5 / 2.0)
    assert float(out[0, 1]) == pytest.approx(-2.0 * 0.5 / 2.0)


def test_quantize_rejects_ragged_last_dim():
    dev = _device()
    with pytest.raises(ValueError):
        quantize_nvfp4(torch.zeros(4, NVFP4_BLOCK + 1, device=dev))


# --------------------------------------------------------------------------
# SwiGLU
# --------------------------------------------------------------------------


def test_swiglu_splits_gate_first():
    dev = _device()
    gate = torch.tensor([[1.0, 2.0]], device=dev)
    up = torch.tensor([[3.0, 4.0]], device=dev)
    out = swiglu(torch.cat([gate, up], dim=-1), clamp=None)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(out, expected)


def test_swiglu_clamps_before_the_activation():
    """Clamping after silu would be a different function; pin the order."""
    dev = _device()
    gate_up = torch.tensor([[100.0, 100.0]], device=dev)
    out = swiglu(gate_up, clamp=1.0)
    expected = torch.nn.functional.silu(torch.tensor([[1.0]], device=dev)) * 1.0
    torch.testing.assert_close(out, expected)


# --------------------------------------------------------------------------
# Dispatch planning
# --------------------------------------------------------------------------


def test_plan_dispatch_routes_every_live_pair_exactly_once():
    dev = _device()
    shape = ProblemShape(
        hidden=256, intermediate=128, num_experts=8, top_k=2, max_tokens_per_rank=4
    )
    world = 2
    ids = (
        torch.tensor([[0, 5], [3, 7], [1, 1], [4, 6]], device=dev),
        torch.tensor([[2, 2], [6, 0], [7, 3], [5, 1]], device=dev),
    )
    plan = plan_dispatch(
        ids, shape=shape, world_size=world, invalid_expert_id=-1
    )
    total = sum(len(p) for p in plan.pairs)
    assert total == 2 * 4 * 2, "every (token, slot) should be routed"

    # Each pair goes to the rank that owns its expert.
    experts_per_rank = shape.num_experts // world
    for dst, pairs in enumerate(plan.pairs):
        for src, token, slot, local_expert in pairs:
            expert = int(ids[src][token, slot])
            assert expert // experts_per_rank == dst
            assert expert % experts_per_rank == local_expert


def test_plan_dispatch_drops_the_invalid_sentinel():
    dev = _device()
    shape = ProblemShape(
        hidden=256, intermediate=128, num_experts=8, top_k=2, max_tokens_per_rank=4
    )
    ids = (torch.tensor([[0, -1], [-1, -1], [3, 4], [-1, 7]], device=dev),)
    plan = plan_dispatch(ids, shape=shape, world_size=1, invalid_expert_id=-1)
    assert sum(len(p) for p in plan.pairs) == 4
    counts = plan.counts(0, experts_per_rank=8)
    assert sum(counts) == 4


def test_plan_dispatch_counts_match_the_pairs():
    dev = _device()
    shape = ProblemShape(
        hidden=256, intermediate=128, num_experts=4, top_k=2, max_tokens_per_rank=8
    )
    g = torch.Generator(device=dev).manual_seed(3)
    ids = tuple(
        torch.randint(0, 4, (8, 2), device=dev, generator=g) for _ in range(2)
    )
    plan = plan_dispatch(ids, shape=shape, world_size=2, invalid_expert_id=-1)
    for rank in range(2):
        counts = plan.counts(rank, experts_per_rank=2)
        assert sum(counts) == len(plan.pairs[rank])


# --------------------------------------------------------------------------
# End-to-end oracle
# --------------------------------------------------------------------------


def _tiny_problem(dev, world_size=2, num_tokens=6):
    shape = ProblemShape(
        hidden=128,
        intermediate=64,
        num_experts=4,
        top_k=2,
        max_tokens_per_rank=num_tokens,
    )
    experts_per_rank = shape.num_experts // world_size
    g = torch.Generator(device=dev).manual_seed(11)
    hidden_states = tuple(
        torch.randn(num_tokens, shape.hidden, dtype=torch.bfloat16,
                    device=dev, generator=g)
        for _ in range(world_size)
    )
    scores = tuple(
        torch.randn(num_tokens, shape.num_experts, device=dev, generator=g)
        for _ in range(world_size)
    )
    topk = tuple(torch.topk(s, shape.top_k, dim=-1) for s in scores)
    topk_weights = tuple(torch.softmax(t.values, dim=-1) for t in topk)
    topk_ids = tuple(t.indices for t in topk)
    w13 = tuple(
        torch.randn(experts_per_rank, shape.gate_up, shape.hidden,
                    dtype=torch.float32, device=dev, generator=g) * 0.05
        for _ in range(world_size)
    )
    w2 = tuple(
        torch.randn(experts_per_rank, shape.hidden, shape.intermediate,
                    dtype=torch.float32, device=dev, generator=g) * 0.05
        for _ in range(world_size)
    )
    return shape, hidden_states, topk_ids, topk_weights, w13, w2


def test_moe_reference_runs_and_is_finite():
    dev = _device()
    world = 2
    shape, hs, ids, wts, w13, w2 = _tiny_problem(dev, world_size=world)
    out = moe_reference(
        hidden_states=hs, topk_ids=ids, topk_weights=wts, w13=w13, w2=w2,
        shape=shape, topology=EpTopology(rank=0, world_size=world),
        epilogue=EpilogueConfig(gate_up_clamp=10.0),
    )
    assert len(out) == world
    for o, h in zip(out, hs, strict=True):
        assert o.shape == (h.shape[0], shape.hidden)
        assert torch.isfinite(o.float()).all()


def test_topk_weight_placement_is_equivalent_without_requantization():
    """Applying the routing weight in FC1 vs after FC2 agrees up to requant.

    The kernel folds the weight into FC1 so combine stays a plain sum; this
    pins that the choice is a scheduling decision, not a numerics change.  The
    two differ only because the FC1-output requantization sees a rescaled
    tensor, so the comparison is a loose relative one on purpose.
    """
    dev = _device()
    world = 1
    shape, hs, ids, wts, w13, w2 = _tiny_problem(dev, world_size=world)
    common = dict(
        hidden_states=hs, topk_ids=ids, topk_weights=wts, w13=w13, w2=w2,
        shape=shape, topology=EpTopology(rank=0, world_size=world),
    )
    in_fc1 = moe_reference(
        **common, epilogue=EpilogueConfig(apply_topk_in_fc1=True)
    )[0].float()
    post = moe_reference(
        **common, epilogue=EpilogueConfig(apply_topk_in_fc1=False)
    )[0].float()
    denom = post.abs().mean().clamp(min=1e-6)
    assert float((in_fc1 - post).abs().mean() / denom) < 0.1


def test_zero_weights_produce_zero_output():
    dev = _device()
    world = 1
    shape, hs, ids, _wts, w13, w2 = _tiny_problem(dev, world_size=world)
    zeros = tuple(torch.zeros_like(w) for w in (ids,)[0:0]) or (
        torch.zeros(hs[0].shape[0], shape.top_k, device=dev),
    )
    out = moe_reference(
        hidden_states=hs, topk_ids=ids, topk_weights=zeros, w13=w13, w2=w2,
        shape=shape, topology=EpTopology(rank=0, world_size=world),
        epilogue=EpilogueConfig(apply_topk_in_fc1=True),
    )
    assert float(out[0].float().abs().max()) == 0.0


def test_unrouted_tokens_produce_zero_output():
    """A token whose slots are all the sentinel contributes nothing."""
    dev = _device()
    world = 1
    shape, hs, _ids, wts, w13, w2 = _tiny_problem(dev, world_size=world)
    ids = (torch.full((hs[0].shape[0], shape.top_k), -1, device=dev),)
    out = moe_reference(
        hidden_states=hs, topk_ids=ids, topk_weights=wts, w13=w13, w2=w2,
        shape=shape, topology=EpTopology(rank=0, world_size=world),
        epilogue=EpilogueConfig(),
    )
    assert float(out[0].float().abs().max()) == 0.0
