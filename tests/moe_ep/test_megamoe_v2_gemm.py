# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: the v2 block-scaled tcgen05 mainloop vs torch.

Validates the smallest complete GEMM -- one CTA, one tile, no fusion -- so a
failure points at exactly one of: smem staging, the A/B pipelines, the
smem->TMEM scale copy, the UMMA issue sequence, or the TMEM readback.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from flashinfer.moe_ep.kernel_src.megamoe_v2 import sf_layout  # noqa: E402
from flashinfer.moe_ep.kernel_src.megamoe_v2.reference import (  # noqa: E402
    quantize_nvfp4,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import NVFP4_BLOCK  # noqa: E402


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs sm_100a / sm_103a")


def _pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """E2M1 magnitudes -> packed uint8 pairs (low nibble first)."""
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=codes.device,
    )
    mag = codes.abs().unsqueeze(-1)
    idx = (mag - levels).abs().argmin(dim=-1).to(torch.uint8)
    nib = torch.where(codes < 0, idx | 0x8, idx).to(torch.uint8)
    lo, hi = nib[..., 0::2], nib[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def _scatter_scales(scales: torch.Tensor) -> torch.Tensor:
    """(rows, blocks) -> the swizzled flat E4M3 buffer the TMA descriptor reads."""
    rows, blocks = scales.shape
    num_k_atoms = sf_layout.num_k_atoms_for(blocks * NVFP4_BLOCK, NVFP4_BLOCK)
    flat = torch.zeros(
        sf_layout.buffer_words(rows, num_k_atoms=num_k_atoms) * 4,
        dtype=torch.float32,
        device=scales.device,
    )
    for r in range(rows):
        row_block, t_in = divmod(r, 128)
        for b in range(blocks):
            k_atom, k_bank = divmod(b, 4)
            atom = row_block * num_k_atoms + k_atom
            flat[atom * 512 + (t_in % 32) * 16 + (t_in // 32) * 4 + k_bank] = scales[
                r, b
            ]
    return flat.to(torch.float8_e4m3fn)


def _run(a_f32, b_f32, m, n, k):
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.gemm_smoke import smoke_gemm

    qa = quantize_nvfp4(a_f32)
    qb = quantize_nvfp4(b_f32)

    a_pk = _pack_fp4(qa.codes).view(torch.float4_e2m1fn_x2)
    b_pk = _pack_fp4(qb.codes).view(torch.float4_e2m1fn_x2)
    sfa = _scatter_scales(qa.scales)
    sfb = _scatter_scales(qb.scales)
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (mk(a_pk), mk(b_pk), mk(sfa), mk(sfb), mk(c), stream)
    cute.compile(smoke_gemm, *args, m=m, n=n, k=k)(*args)
    torch.cuda.synchronize()

    expected = qa.dequantize() @ qb.dequantize().T
    return c, expected


@pytest.mark.parametrize("k", [256, 512])
def test_smoke_gemm_matches_torch(k):
    _require_blackwell()
    m = n = 128
    g = torch.Generator(device="cuda").manual_seed(4)
    a = torch.randn(m, k, dtype=torch.float32, device="cuda", generator=g)
    b = torch.randn(n, k, dtype=torch.float32, device="cuda", generator=g)

    got, expected = _run(a, b, m, n, k)
    # Both sides consume the *same* dequantized operands, so the only error is
    # fp32 accumulation order -- a tight tolerance is correct here.
    torch.testing.assert_close(got, expected, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------
# Grouped / persistent / 2-CTA mainloop
# --------------------------------------------------------------------------


def _run_grouped(
    counts,
    out_channels,
    k,
    *,
    num_clusters,
    cluster_m=2,
    two_cta=True,
    mma_m=256,
    acc_stages=1,
):
    """Grouped GEMM over a shared token pool; returns (got, expected)."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.gemm_kernel import (
        launch_grouped_gemm,
    )
    from flashinfer.moe_ep.kernel_src.megamoe_v2.schedule import build_expert_layout
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (
        Phase,
        ProblemShape,
        TileConfig,
    )

    num_experts = len(counts)
    tile = TileConfig(
        mma_m=mma_m, mma_n=128, mma_k=256, cluster_m=cluster_m, two_cta=two_cta
    )
    shape = ProblemShape(
        hidden=k,
        intermediate=out_channels // 2,
        num_experts=num_experts,
        top_k=1,
        max_tokens_per_rank=max(1, sum(counts)),
    )
    layout = build_expert_layout(
        counts, shape=shape, tile=tile, phase=Phase.FC1, channel_ranges=acc_stages
    )
    pool_rows = max(layout.pool_rows, tile.cluster_tile_tokens)

    g = torch.Generator(device="cuda").manual_seed(21)
    w = (
        torch.randn(
            num_experts,
            out_channels,
            k,
            dtype=torch.float32,
            device="cuda",
            generator=g,
        )
        * 0.3
    )
    x = torch.randn(pool_rows, k, dtype=torch.float32, device="cuda", generator=g) * 0.3

    qw = quantize_nvfp4(w.reshape(-1, k))
    qx = quantize_nvfp4(x)

    w_pk = _pack_fp4(qw.codes).view(torch.float4_e2m1fn_x2)
    x_pk = _pack_fp4(qx.codes).view(torch.float4_e2m1fn_x2)
    # Scales are swizzled per (rows, blocks) plane; weights are one plane of
    # experts*out_channels rows because the atom layout tiles rows the same way
    # regardless of the L split.
    sfw = _scatter_scales(qw.scales)
    sfx = _scatter_scales(qx.scales)

    c = torch.zeros(out_channels, pool_rows, dtype=torch.float32, device="cuda")
    prefix = torch.tensor(layout.token_block_prefix, dtype=torch.int32, device="cuda")

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        mk(w_pk),
        mk(x_pk),
        mk(sfw),
        mk(sfx),
        (mk(c),),
        mk(prefix),
        cutlass.Int32(layout.total_tiles),
        stream,
    )
    kw = dict(
        num_experts=num_experts,
        out_channels=out_channels,
        pool_rows=pool_rows,
        k=k,
        cluster_m=cluster_m,
        two_cta=two_cta,
        mma_m=mma_m,
        num_clusters=num_clusters,
        acc_stages=acc_stages,
    )
    cute.compile(launch_grouped_gemm, *args, **kw)(*args)
    torch.cuda.synchronize()

    # Reference: for each expert's padded pool segment, W[e] @ X[rows].T
    wd = qw.dequantize().reshape(num_experts, out_channels, k)
    xd = qx.dequantize()
    expected = torch.zeros_like(c)
    for e in range(len(counts)):
        start = layout.token_block_prefix[e] * tile.cluster_tile_tokens
        span = (
            layout.token_block_prefix[e + 1] - layout.token_block_prefix[e]
        ) * tile.cluster_tile_tokens
        if span == 0:
            continue
        rows = slice(start, start + span)
        expected[:, rows] = wd[e] @ xd[rows].T
    return c, expected, layout


@pytest.mark.parametrize(
    "counts,num_clusters",
    [
        ((128,), 1),  # one expert, one tile, one cluster: isolates 2-CTA
        ((128,), 4),  # more clusters than tiles: exercises the exit path
        ((128, 128), 2),  # two experts: exercises the L-mode expert switch
        ((300, 0, 128, 40), 3),  # skew + an empty expert + partial tail tiles
    ],
)
def test_grouped_gemm_matches_torch(counts, num_clusters):
    _require_blackwell()
    got, expected, layout = _run_grouped(
        counts, out_channels=256, k=512, num_clusters=num_clusters
    )
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("counts,num_clusters", [((128,), 1), ((300, 0, 128, 40), 3)])
def test_grouped_gemm_single_cta_matches_torch(counts, num_clusters):
    """Grouping + persistent loop with a 1-CTA MMA: isolates those two from 2-CTA."""
    _require_blackwell()
    got, expected, _ = _run_grouped(
        counts,
        out_channels=256,
        k=512,
        num_clusters=num_clusters,
        cluster_m=1,
        two_cta=False,
        mma_m=128,
    )
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("counts,num_clusters", [((128,), 1), ((300, 0, 128, 40), 3)])
def test_paired_accumulators_match_torch(counts, num_clusters):
    """``acc_stages=2`` with the plain epilogue: the FC1 mainloop, unfused.

    Each tile now runs two channel ranges against one resident token tile, so
    this pins the interleaved A stream, the two independent ACCUMULATE fields
    and the doubled TMEM carve *without* any activation or requantization in
    the way.  A failure here is a mainloop bug; a failure only in the FC1 test
    is an epilogue bug.
    """
    _require_blackwell()
    got, expected, _ = _run_grouped(
        counts,
        out_channels=1024,
        k=512,
        num_clusters=num_clusters,
        acc_stages=2,
    )
    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)
