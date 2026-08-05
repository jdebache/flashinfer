# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: FC1 with the fused SwiGLU + requantization epilogue.

The mainloop underneath is already pinned by ``test_megamoe_v2_gemm.py``
(including the ``acc_stages=2`` paired-accumulator case with a plain
epilogue), so a failure here is an epilogue failure: the gate/up pairing, the
smem transpose, the routing weight, or the NVFP4 encode.
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

import importlib.util  # noqa: E402
import pathlib  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_v2_gemm_test", pathlib.Path(__file__).with_name("test_megamoe_v2_gemm.py")
)
_gemm_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gemm_test)
_pack_fp4 = _gemm_test._pack_fp4
_require_blackwell = _gemm_test._require_blackwell
_scatter_scales = _gemm_test._scatter_scales

_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """(rows, cols/2) uint8 -> (rows, cols) float32 magnitudes with sign."""
    levels = torch.tensor(_FP4_LEVELS, dtype=torch.float32, device=packed.device)
    nib = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(1)
    val = levels[(nib & 0x7).long()]
    return torch.where(nib & 0x8 != 0, -val, val)


def _gather_scales(flat: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
    """Inverse of the swizzled-atom scatter used to build kernel inputs."""
    num_k_atoms = sf_layout.num_k_atoms_for(blocks * NVFP4_BLOCK, NVFP4_BLOCK)
    f = flat.to(torch.float32)
    out = torch.zeros(rows, blocks, dtype=torch.float32, device=flat.device)
    for r in range(rows):
        row_block, t_in = divmod(r, 128)
        for b in range(blocks):
            k_atom, k_bank = divmod(b, 4)
            atom = row_block * num_k_atoms + k_atom
            out[r, b] = f[atom * 512 + (t_in % 32) * 16 + (t_in // 32) * 4 + k_bank]
    return out


def _swiglu(gate, up, clamp):
    if clamp is not None:
        gate = gate.clamp(-clamp, clamp)
        up = up.clamp(-clamp, clamp)
    return torch.nn.functional.silu(gate) * up


def _run_fc1(
    counts, intermediate, hidden, *, num_clusters, clamp, apply_weight, num_a_stages=4
):
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.fc1 import launch_fc1
    from flashinfer.moe_ep.kernel_src.megamoe_v2.schedule import build_expert_layout
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (
        Phase,
        ProblemShape,
        TileConfig,
    )

    num_experts = len(counts)
    tile = TileConfig(mma_m=256, mma_n=128, mma_k=256, cluster_m=2, two_cta=True)
    shape = ProblemShape(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        top_k=1,
        max_tokens_per_rank=max(1, sum(counts)),
    )
    layout = build_expert_layout(
        counts, shape=shape, tile=tile, phase=Phase.FC1, channel_ranges=2
    )
    pool_rows = max(layout.pool_rows, tile.cluster_tile_tokens)

    g = torch.Generator(device="cuda").manual_seed(7)
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
    x = (
        torch.randn(pool_rows, hidden, dtype=torch.float32, device="cuda", generator=g)
        * 0.3
    )
    weights = (
        torch.rand(pool_rows, dtype=torch.float32, device="cuda", generator=g) * 0.9
        + 0.1
    )

    qw = quantize_nvfp4(w1.reshape(-1, hidden))
    qx = quantize_nvfp4(x)

    num_k_atoms = sf_layout.num_k_atoms_for(intermediate, NVFP4_BLOCK)
    out_bytes = torch.zeros(
        pool_rows, intermediate // 2, dtype=torch.uint8, device="cuda"
    )
    out_scales = torch.zeros(
        sf_layout.buffer_words(pool_rows, num_k_atoms=num_k_atoms) * 4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    prefix = torch.tensor(layout.token_block_prefix, dtype=torch.int32, device="cuda")

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        mk(_pack_fp4(qw.codes).view(torch.float4_e2m1fn_x2)),
        mk(_pack_fp4(qx.codes).view(torch.float4_e2m1fn_x2)),
        mk(_scatter_scales(qw.scales)),
        mk(_scatter_scales(qx.scales)),
        mk(out_bytes.view(torch.float4_e2m1fn_x2)),
        mk(out_scales),
        mk(weights),
        mk(prefix),
        stream,
    )
    kw = dict(
        num_experts=num_experts,
        intermediate=intermediate,
        hidden=hidden,
        pool_rows=pool_rows,
        num_k_atoms=num_k_atoms,
        clamp=clamp,
        apply_weight=apply_weight,
        num_clusters=num_clusters,
        num_a_stages=num_a_stages,
    )
    cute.compile(launch_fc1, *args, **kw)(*args)
    torch.cuda.synchronize()

    got_scales = _gather_scales(out_scales, pool_rows, intermediate // NVFP4_BLOCK)
    got = _unpack_fp4(out_bytes) * got_scales.repeat_interleave(NVFP4_BLOCK, dim=1)

    # Reference: same dequantized operands, activation, then the same encoding.
    wd = qw.dequantize().reshape(num_experts, 2 * intermediate, hidden)
    xd = qx.dequantize()
    ref = torch.zeros(pool_rows, intermediate, dtype=torch.float32, device="cuda")
    for e in range(num_experts):
        start = layout.token_block_prefix[e] * tile.cluster_tile_tokens
        span = (
            layout.token_block_prefix[e + 1] - layout.token_block_prefix[e]
        ) * tile.cluster_tile_tokens
        if span == 0:
            continue
        rows = slice(start, start + span)
        acc = wd[e] @ xd[rows].T
        act = _swiglu(acc[:intermediate], acc[intermediate:], clamp).T
        if apply_weight:
            act = act * weights[rows].unsqueeze(1)
        ref[rows] = act
    qref = quantize_nvfp4(ref)
    return got, qref.dequantize(), layout


@pytest.mark.parametrize(
    "counts,num_clusters",
    [
        ((128,), 1),
        ((128, 128), 2),
        ((300, 0, 128, 40), 3),
    ],
)
def test_fc1_matches_reference(counts, num_clusters):
    _require_blackwell()
    got, ref, _ = _run_fc1(
        counts,
        intermediate=256,
        hidden=512,
        num_clusters=num_clusters,
        clamp=None,
        apply_weight=True,
    )
    _assert_fp4_close(got, ref)


@pytest.mark.parametrize("clamp", [None, 1.0])
def test_fc1_clamp(clamp):
    """A tight clamp saturates most pre-activations, so it must be visible."""
    _require_blackwell()
    got, ref, _ = _run_fc1(
        (128,),
        intermediate=256,
        hidden=512,
        num_clusters=1,
        clamp=clamp,
        apply_weight=True,
    )
    _assert_fp4_close(got, ref)


def test_fc1_without_routing_weight():
    _require_blackwell()
    got, ref, _ = _run_fc1(
        (128,),
        intermediate=256,
        hidden=512,
        num_clusters=1,
        clamp=None,
        apply_weight=False,
    )
    _assert_fp4_close(got, ref)


def test_fc1_routing_weight_scales_output():
    """The weight is folded into FC1, so doubling it doubles the activation.

    Not bit-exact: the requantization in between sees a different block amax,
    so this checks the property with a tolerance rather than equality.
    """
    _require_blackwell()
    got, _, _ = _run_fc1(
        (128,),
        intermediate=256,
        hidden=512,
        num_clusters=1,
        clamp=None,
        apply_weight=True,
    )
    unweighted, _, _ = _run_fc1(
        (128,),
        intermediate=256,
        hidden=512,
        num_clusters=1,
        clamp=None,
        apply_weight=False,
    )
    # Same generator seed, so the same routing weights are regenerated.
    g = torch.Generator(device="cuda").manual_seed(7)
    torch.randn(1, 2 * 256, 512, dtype=torch.float32, device="cuda", generator=g)
    torch.randn(128, 512, dtype=torch.float32, device="cuda", generator=g)
    w = torch.rand(128, dtype=torch.float32, device="cuda", generator=g) * 0.9 + 0.1

    expected = unweighted * w.unsqueeze(1)
    denom = expected.abs().max().clamp_min(1e-6)
    assert (got - expected).abs().max() / denom < 0.15


def _assert_fp4_close(got: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides are NVFP4; they should agree except at rounding boundaries.

    The device silu uses a fast approximate exponential, so a handful of values
    land on the other side of an fp4 level boundary.  Requiring exact equality
    would make this test a detector of that approximation rather than of the
    epilogue, so it is bounded instead: the overwhelming majority must match
    exactly, and the residual energy must be negligible.
    """
    # Guard against a vacuous pass: an all-zero output would match an all-zero
    # reference perfectly.
    assert (ref != 0).float().mean().item() > 0.2, "reference is degenerate"
    exact = (got == ref).float().mean().item()
    rel = (got - ref).norm().item() / max(ref.norm().item(), 1e-6)
    assert exact > 0.99, f"only {exact:.4f} of elements match exactly"
    assert rel < 1e-2, f"relative error {rel:.4g}"
