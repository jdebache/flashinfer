# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: FC2, and the FC1 -> FC2 chain under programmatic dependent launch.

FC2 shares the mainloop that ``test_megamoe_v2_gemm.py`` already pins, so what
is new here is the transposing epilogue.  The chain test then runs both kernels
back to back with PDL enabled: the answer must not change, which is the only
thing PDL is allowed to affect.
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


def _layout_for(counts, shape, tile, phase, channel_ranges=1):
    from flashinfer.moe_ep.kernel_src.megamoe_v2.schedule import build_expert_layout

    return build_expert_layout(
        counts, shape=shape, tile=tile, phase=phase, channel_ranges=channel_ranges
    )


def _make(counts, intermediate, hidden, seed=17):
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (
        Phase,
        ProblemShape,
        TileConfig,
    )

    tile = TileConfig(mma_m=256, mma_n=128, mma_k=256, cluster_m=2, two_cta=True)
    shape = ProblemShape(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=len(counts),
        top_k=1,
        max_tokens_per_rank=max(1, sum(counts)),
    )
    layout = _layout_for(counts, shape, tile, Phase.FC2)
    pool_rows = max(layout.pool_rows, tile.cluster_tile_tokens)
    return tile, shape, layout, pool_rows


def _run_fc2(counts, intermediate, hidden, *, num_clusters, seed=17):
    """FC2 alone, from a synthetic NVFP4 pool."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.fc2 import launch_fc2

    num_experts = len(counts)
    tile, _shape, layout, pool_rows = _make(counts, intermediate, hidden, seed)

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
    h = (
        torch.randn(
            pool_rows, intermediate, dtype=torch.float32, device="cuda", generator=g
        )
        * 0.3
    )

    qw = quantize_nvfp4(w2.reshape(-1, intermediate))
    qh = quantize_nvfp4(h)
    out = torch.zeros(pool_rows, hidden, dtype=torch.bfloat16, device="cuda")
    prefix = torch.tensor(layout.token_block_prefix, dtype=torch.int32, device="cuda")

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        mk(_pack_fp4(qw.codes).view(torch.float4_e2m1fn_x2)),
        mk(_pack_fp4(qh.codes).view(torch.float4_e2m1fn_x2)),
        mk(_scatter_scales(qw.scales)),
        mk(_scatter_scales(qh.scales)),
        mk(out),
        mk(prefix),
        stream,
    )
    kw = dict(
        num_experts=num_experts,
        intermediate=intermediate,
        hidden=hidden,
        pool_rows=pool_rows,
        num_clusters=num_clusters,
    )
    cute.compile(launch_fc2, *args, **kw)(*args)
    torch.cuda.synchronize()

    wd = qw.dequantize().reshape(num_experts, hidden, intermediate)
    hd = qh.dequantize()
    expected = torch.zeros(pool_rows, hidden, dtype=torch.float32, device="cuda")
    for e in range(num_experts):
        start = layout.token_block_prefix[e] * tile.cluster_tile_tokens
        span = (
            layout.token_block_prefix[e + 1] - layout.token_block_prefix[e]
        ) * tile.cluster_tile_tokens
        if span == 0:
            continue
        rows = slice(start, start + span)
        expected[rows] = (wd[e] @ hd[rows].T).T
    return out.float(), expected


@pytest.mark.parametrize(
    "counts,num_clusters",
    [
        ((128,), 1),
        ((128, 128), 2),
        ((300, 0, 128, 40), 3),
    ],
)
def test_fc2_matches_torch(counts, num_clusters):
    _require_blackwell()
    got, expected = _run_fc2(
        counts, intermediate=256, hidden=512, num_clusters=num_clusters
    )
    # bf16 output: one rounding step on top of an fp32-accurate accumulation.
    torch.testing.assert_close(got, expected, atol=6e-2, rtol=6e-3)


def test_fc2_output_is_token_major():
    """The transpose is the epilogue's whole job, so pin it directly.

    A kernel that wrote the accumulator out untransposed would still produce
    the right *values*, just at (channel, token).  Comparing against an
    explicitly transposed reference is what distinguishes the two.
    """
    _require_blackwell()
    got, expected = _run_fc2((128,), intermediate=256, hidden=512, num_clusters=1)
    assert got.shape == (128, 512)
    torch.testing.assert_close(got, expected, atol=6e-2, rtol=6e-3)
    # And it is genuinely not symmetric, so the check above has teeth.
    assert not torch.allclose(expected[:128, :128], expected[:128, :128].T, atol=1e-2)


def _run_chain(counts, intermediate, hidden, *, use_pdl, num_clusters=4, seed=23):
    """FC1 then FC2 on one stream, optionally with PDL on both launches."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.fc1 import launch_fc1
    from flashinfer.moe_ep.kernel_src.megamoe_v2.fc2 import launch_fc2
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import Phase

    num_experts = len(counts)
    tile, shape, layout2, pool_rows = _make(counts, intermediate, hidden, seed)
    layout1 = _layout_for(counts, shape, tile, Phase.FC1, channel_ranges=2)

    g = torch.Generator(device="cuda").manual_seed(seed)
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
    x = (
        torch.randn(pool_rows, hidden, dtype=torch.float32, device="cuda", generator=g)
        * 0.3
    )
    weights = (
        torch.rand(pool_rows, dtype=torch.float32, device="cuda", generator=g) * 0.9
        + 0.1
    )

    qw1 = quantize_nvfp4(w1.reshape(-1, hidden))
    qw2 = quantize_nvfp4(w2.reshape(-1, intermediate))
    qx = quantize_nvfp4(x)

    num_k_atoms = sf_layout.num_k_atoms_for(intermediate, NVFP4_BLOCK)
    fc1_bytes = torch.zeros(
        pool_rows, intermediate // 2, dtype=torch.uint8, device="cuda"
    )
    fc1_sf = torch.zeros(
        sf_layout.buffer_words(pool_rows, num_k_atoms=num_k_atoms) * 4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    out = torch.zeros(pool_rows, hidden, dtype=torch.bfloat16, device="cuda")

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    p1 = torch.tensor(layout1.token_block_prefix, dtype=torch.int32, device="cuda")
    p2 = torch.tensor(layout2.token_block_prefix, dtype=torch.int32, device="cuda")

    a1 = (
        mk(_pack_fp4(qw1.codes).view(torch.float4_e2m1fn_x2)),
        mk(_pack_fp4(qx.codes).view(torch.float4_e2m1fn_x2)),
        mk(_scatter_scales(qw1.scales)),
        mk(_scatter_scales(qx.scales)),
        mk(fc1_bytes.view(torch.float4_e2m1fn_x2)),
        mk(fc1_sf),
        mk(weights),
        mk(p1),
        stream,
    )
    k1 = dict(
        num_experts=num_experts,
        intermediate=intermediate,
        hidden=hidden,
        pool_rows=pool_rows,
        num_k_atoms=num_k_atoms,
        clamp=None,
        apply_weight=True,
        num_clusters=num_clusters,
        use_pdl=use_pdl,
    )
    a2 = (
        mk(_pack_fp4(qw2.codes).view(torch.float4_e2m1fn_x2)),
        mk(fc1_bytes.view(torch.float4_e2m1fn_x2)),
        mk(_scatter_scales(qw2.scales)),
        mk(fc1_sf),
        mk(out),
        mk(p2),
        stream,
    )
    k2 = dict(
        num_experts=num_experts,
        intermediate=intermediate,
        hidden=hidden,
        pool_rows=pool_rows,
        num_clusters=num_clusters,
        use_pdl=use_pdl,
    )

    c1 = cute.compile(launch_fc1, *a1, **k1)
    c2 = cute.compile(launch_fc2, *a2, **k2)
    c1(*a1)
    c2(*a2)
    torch.cuda.synchronize()
    return out.float()


@pytest.mark.parametrize("counts", [((128,)), ((300, 0, 128, 40))])
def test_pdl_does_not_change_the_answer(counts):
    """PDL is a scheduling hint; enabling it must be bit-identical.

    If FC2's token-side wait were missing or misplaced, it would read fc1_out
    before FC1 finished writing it -- a race that shows up here as a mismatch
    against the same chain run without PDL.
    """
    _require_blackwell()
    base = _run_chain(counts, 256, 512, use_pdl=False)
    pdl = _run_chain(counts, 256, 512, use_pdl=True)
    torch.testing.assert_close(pdl, base, atol=0, rtol=0)
    assert base.abs().sum() > 0
