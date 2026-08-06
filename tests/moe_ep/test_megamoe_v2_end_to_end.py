# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end: the whole v2 pipeline against the torch oracle.

Every stage is already pinned in isolation; this is the first test where they
have to agree with each other -- the pool layout dispatch produces must be the
one FC1's schedule walks, the provenance it records must be the one combine
scatters by, and the numerics must match ``moe_reference`` all the way through.

Ranks are emulated in-process, so the barrier is exercised as a real
rendezvous only at ``world_size == 1``; a multi-rank group needs the stages
interleaved across concurrent ranks, which is the multi-process harness's job.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from flashinfer.moe_ep.kernel_src.megamoe_v2 import launcher  # noqa: E402
from flashinfer.moe_ep.kernel_src.megamoe_v2.reference import (  # noqa: E402
    moe_reference,
    quantize_nvfp4,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (  # noqa: E402
    CommConfig,
    EpilogueConfig,
    EpTopology,
    KernelConfig,
    Phase,
    ProblemShape,
    TileConfig,
)

_spec = importlib.util.spec_from_file_location(
    "_v2_gemm_test", pathlib.Path(__file__).with_name("test_megamoe_v2_gemm.py")
)
_gemm_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gemm_test)
_pack_fp4 = _gemm_test._pack_fp4
_require_blackwell = _gemm_test._require_blackwell
_scatter_scales = _gemm_test._scatter_scales


def _local_allocator(nbytes: int):
    """Single-rank symmetric heap: the only peer is self, at offset zero."""
    buf = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    return buf, (buf.data_ptr(),)


def _quantize_weights(w: torch.Tensor):
    """(experts, out, k) fp32 -> packed NVFP4 codes + swizzled scales."""
    experts, out, k = w.shape
    q = quantize_nvfp4(w.reshape(-1, k))
    codes = _pack_fp4(q.codes).view(torch.float4_e2m1fn_x2)
    return codes, _scatter_scales(q.scales), q.dequantize().reshape(experts, out, k)


def _run_pipeline(
    *,
    num_tokens,
    hidden,
    intermediate,
    num_experts,
    top_k,
    clamp=None,
    seed=5,
    num_clusters=8,
):
    import cuda.bindings.driver as cuda

    config = KernelConfig(
        shape=ProblemShape(
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            top_k=top_k,
            max_tokens_per_rank=num_tokens,
        ),
        topology=EpTopology(world_size=1, rank=0),
        phase=Phase.FC1,
        tile=TileConfig(mma_m=256, mma_n=128, mma_k=256, cluster_m=2, two_cta=True),
        comm=CommConfig(invalid_expert_id=-1),
        epilogue=EpilogueConfig(gate_up_clamp=clamp, apply_topk_in_fc1=True),
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    act = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device="cuda", generator=g
    ).bfloat16()
    w13 = (
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

    logits = torch.rand(num_tokens, num_experts, device="cuda", generator=g) ** 3
    topk_ids = logits.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_ids[::9, -1] = -1  # some slots carry no token
    topk_weights = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device="cuda", generator=g
    )

    w1_codes, w1_sf, w13_dq = _quantize_weights(w13)
    w2_codes, w2_sf, w2_dq = _quantize_weights(w2)
    weights = launcher.Weights(w1=w1_codes, w1_sf=w1_sf, w2=w2_codes, w2_sf=w2_sf)

    ws = launcher.allocate_workspaces(config, rank=0, alloc_shared=_local_allocator)
    views = launcher.build_views(ws, config)
    out = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    pipe = launcher.compile_pipeline(
        config,
        rank=0,
        ws=ws,
        views=views,
        weights=weights,
        activation=act,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        out=out,
        stream=stream,
        num_clusters=num_clusters,
    )
    launcher.run(pipe)
    torch.cuda.synchronize()

    expected = moe_reference(
        hidden_states=(act,),
        topk_ids=(topk_ids,),
        topk_weights=(topk_weights,),
        w13=(w13_dq,),
        w2=(w2_dq,),
        shape=config.shape,
        topology=config.topology,
        epilogue=config.epilogue,
        invalid_expert_id=-1,
    )[0]
    return out.float(), expected.float(), pipe, topk_ids


def _assert_matches(got, expected):
    assert expected.abs().sum() > 0, "reference is degenerate"
    rel = (got - expected).norm().item() / expected.norm().item()
    assert rel < 3e-2, f"relative error {rel:.4g}"


@pytest.mark.parametrize("top_k", [1, 2])
def test_pipeline_matches_reference(top_k):
    _require_blackwell()
    got, expected, _, _ = _run_pipeline(
        num_tokens=256, hidden=512, intermediate=256, num_experts=4, top_k=top_k
    )
    _assert_matches(got, expected)


def test_pipeline_with_clamp():
    _require_blackwell()
    got, expected, _, _ = _run_pipeline(
        num_tokens=256,
        hidden=512,
        intermediate=256,
        num_experts=4,
        top_k=2,
        clamp=2.0,
    )
    _assert_matches(got, expected)


def test_unrouted_tokens_produce_zero():
    """A token with every slot invalid must come back exactly zero.

    It never enters the pool, so nothing writes its output; this checks the
    reduce leaves it alone rather than summing whatever the buffer held.
    """
    _require_blackwell()
    got, expected, pipe, topk_ids = _run_pipeline(
        num_tokens=256, hidden=512, intermediate=256, num_experts=4, top_k=1
    )
    _assert_matches(got, expected)
    dead = (topk_ids < 0).all(dim=-1).nonzero().flatten()
    assert dead.numel() > 0, "test routing leaves no fully-unrouted token"
    assert got[dead].abs().max() == 0.0


def test_expert_counts_match_routing():
    """Dispatch's device-side counts must equal the routing they came from."""
    _require_blackwell()
    _got, _exp, pipe, topk_ids = _run_pipeline(
        num_tokens=256, hidden=512, intermediate=256, num_experts=4, top_k=2
    )
    counts = pipe.views.expert_token_count.cpu()
    for e in range(pipe.config.shape.num_experts):
        assert int(counts[e]) == int((topk_ids == e).sum())


def test_rerun_is_idempotent():
    """A second launch on the same workspace must reproduce the first.

    Catches state that survives a launch when it should not -- a counter left
    non-zero, or a barrier phase that got reset and desynchronized.
    """
    _require_blackwell()
    got, expected, pipe, _ = _run_pipeline(
        num_tokens=256, hidden=512, intermediate=256, num_experts=4, top_k=2
    )
    _assert_matches(got, expected)
    launcher.run(pipe)
    torch.cuda.synchronize()
    torch.testing.assert_close(pipe.out.float(), got, atol=0, rtol=0)


# --------------------------------------------------------------------------
# the two-kernel form
# --------------------------------------------------------------------------


def _run_fused_pipeline(
    *,
    num_tokens,
    hidden,
    intermediate,
    num_experts,
    top_k,
    clamp=None,
    seed=5,
    num_clusters=8,
):
    """Same problem, but as the two fused launches instead of nine staged ones."""
    import cuda.bindings.driver as cuda

    config = KernelConfig(
        shape=ProblemShape(
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            top_k=top_k,
            max_tokens_per_rank=num_tokens,
        ),
        topology=EpTopology(world_size=1, rank=0),
        phase=Phase.FC1,
        tile=TileConfig(mma_m=256, mma_n=128, mma_k=256, cluster_m=2, two_cta=True),
        comm=CommConfig(invalid_expert_id=-1),
        epilogue=EpilogueConfig(gate_up_clamp=clamp, apply_topk_in_fc1=True),
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    act = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device="cuda", generator=g
    ).bfloat16()
    w13 = (
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
    logits = torch.rand(num_tokens, num_experts, device="cuda", generator=g) ** 3
    topk_ids = logits.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_ids[::9, -1] = -1
    topk_weights = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device="cuda", generator=g
    )

    w1_codes, w1_sf, w13_dq = _quantize_weights(w13)
    w2_codes, w2_sf, w2_dq = _quantize_weights(w2)
    weights = launcher.Weights(w1=w1_codes, w1_sf=w1_sf, w2=w2_codes, w2_sf=w2_sf)

    ws = launcher.allocate_workspaces(config, rank=0, alloc_shared=_local_allocator)
    views = launcher.build_views(ws, config)
    out = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    pipe = launcher.compile_fused(
        config,
        rank=0,
        ws=ws,
        views=views,
        weights=weights,
        activation=act,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        out=out,
        stream=stream,
        num_clusters=num_clusters,
    )
    launcher.run_fused(pipe)
    torch.cuda.synchronize()

    expected = moe_reference(
        hidden_states=(act,),
        topk_ids=(topk_ids,),
        topk_weights=(topk_weights,),
        w13=(w13_dq,),
        w2=(w2_dq,),
        shape=config.shape,
        topology=config.topology,
        epilogue=config.epilogue,
        invalid_expert_id=-1,
    )[0]
    return out.float(), expected.float(), pipe, topk_ids


@pytest.mark.parametrize("top_k", [1, 2])
def test_fused_pipeline_matches_reference(top_k):
    """Two launches must produce what nine did, and what the oracle says."""
    _require_blackwell()
    got, expected, _, _ = _run_fused_pipeline(
        num_tokens=256, hidden=512, intermediate=256, num_experts=4, top_k=top_k
    )
    _assert_matches(got, expected)


def test_fused_pipeline_with_clamp():
    _require_blackwell()
    got, expected, _, _ = _run_fused_pipeline(
        num_tokens=256,
        hidden=512,
        intermediate=256,
        num_experts=4,
        top_k=2,
        clamp=2.0,
    )
    _assert_matches(got, expected)


def test_fused_rejects_non_resident_grid():
    """A grid larger than the SM count would hang in the device-wide barrier.

    Checked on the host because the failure mode is a deadlock, which gives no
    diagnostic at all.
    """
    _require_blackwell()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    with pytest.raises(ValueError, match="co-resident"):
        _run_fused_pipeline(
            num_tokens=128,
            hidden=512,
            intermediate=256,
            num_experts=2,
            top_k=1,
            num_clusters=sms,
        )
