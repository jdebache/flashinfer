# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: FC1 epilogue numerics (SwiGLU + clamp + top-k weight + requant).

Validated in isolation, before being embedded in the GEMM epilogue where a
numerics bug would be indistinguishable from a TMEM-layout or pipeline bug.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")

from flashinfer.moe_ep.kernel_src.megamoe_v2 import sf_layout  # noqa: E402
from flashinfer.moe_ep.kernel_src.megamoe_v2.reference import (  # noqa: E402
    quantize_nvfp4,
    swiglu,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import NVFP4_BLOCK  # noqa: E402
from .test_megamoe_v2_quant import (  # noqa: E402
    _gather_scales,
    _require_blackwell,
    _unpack_fp4,
)


def _run(gate, up, weights, *, clamp, apply_weight):
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.epilogue import (
        activate_quantize_launch,
    )

    num_tokens, inter = gate.shape
    num_k_atoms = sf_layout.num_k_atoms_for(inter, NVFP4_BLOCK)
    scale_elems = sf_layout.buffer_words(num_tokens, num_k_atoms=num_k_atoms) * 4

    codes = torch.zeros(num_tokens, inter // 2, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(scale_elems, dtype=torch.float8_e4m3fn, device="cuda")

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    args = (
        mk(gate), mk(up), mk(weights),
        mk(codes.view(torch.float4_e2m1fn_x2)), mk(scales),
        cutlass.Int32(num_tokens),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    kw = dict(intermediate=inter, num_k_atoms=num_k_atoms, clamp=clamp,
              apply_weight=apply_weight)
    cute.compile(activate_quantize_launch, *args, **kw)(*args)
    torch.cuda.synchronize()
    return codes, scales


def _expected(gate, up, weights, *, clamp, apply_weight):
    act = swiglu(torch.cat([gate, up], dim=-1), clamp=clamp)
    if apply_weight:
        act = act * weights.unsqueeze(-1)
    return quantize_nvfp4(act)


@pytest.mark.parametrize("clamp", [None, 2.0])
@pytest.mark.parametrize("apply_weight", [True, False])
def test_fc1_epilogue_matches_oracle(clamp, apply_weight):
    _require_blackwell()
    n, inter = 96, 512
    g = torch.Generator(device="cuda").manual_seed(31)
    gate = torch.randn(n, inter, dtype=torch.float32, device="cuda", generator=g) * 3
    up = torch.randn(n, inter, dtype=torch.float32, device="cuda", generator=g) * 3
    w = torch.rand(n, dtype=torch.float32, device="cuda", generator=g)

    codes, scales = _run(gate, up, w, clamp=clamp, apply_weight=apply_weight)
    exp = _expected(gate, up, w, clamp=clamp, apply_weight=apply_weight)

    got_scales = _gather_scales(scales, num_tokens=n, hidden=inter)
    # silu uses a fast approximate exp on device, so the activation differs in
    # the last fp32 bits; that can flip a value across an fp4 rounding boundary.
    # Compare the DEQUANTIZED result with a tolerance rather than codes exactly.
    got = torch.zeros(n, inter, dtype=torch.float32, device="cuda")
    got_codes = _unpack_fp4(codes)
    blocks = inter // NVFP4_BLOCK
    got = (got_codes.reshape(n, blocks, NVFP4_BLOCK)
           * got_scales.unsqueeze(-1)).reshape(n, inter)
    ref = exp.dequantize()
    denom = ref.abs().max().clamp(min=1e-6)
    assert float((got - ref).abs().max() / denom) < 0.05


def test_fc1_epilogue_zero_weight_gives_zero():
    """A zero routing weight must produce an all-zero (not NaN) block."""
    _require_blackwell()
    n, inter = 32, 256
    g = torch.Generator(device="cuda").manual_seed(5)
    gate = torch.randn(n, inter, dtype=torch.float32, device="cuda", generator=g)
    up = torch.randn(n, inter, dtype=torch.float32, device="cuda", generator=g)
    w = torch.zeros(n, dtype=torch.float32, device="cuda")

    codes, scales = _run(gate, up, w, clamp=None, apply_weight=True)
    assert torch.isfinite(scales.to(torch.float32)).all()
    assert float(scales.to(torch.float32).abs().max()) == 0.0
    # The codes are fp4 *negative* zero (nibble 0x8) wherever the pre-scaled
    # activation was negative -- sign is preserved through a zero scale, and
    # the oracle does the same.  The contract is the dequantized value, not the
    # raw nibble, so assert on that.
    got = _unpack_fp4(codes)
    assert torch.isfinite(got).all()
    assert float(got.abs().max()) == 0.0


def test_fc1_epilogue_clamp_bounds_the_activation():
    _require_blackwell()
    n, inter = 16, 256
    gate = torch.full((n, inter), 50.0, dtype=torch.float32, device="cuda")
    up = torch.full((n, inter), 50.0, dtype=torch.float32, device="cuda")
    w = torch.ones(n, dtype=torch.float32, device="cuda")

    codes, scales = _run(gate, up, w, clamp=1.0, apply_weight=True)
    got_scales = _gather_scales(scales, num_tokens=n, hidden=inter)
    got = _unpack_fp4(codes).reshape(n, inter // NVFP4_BLOCK, NVFP4_BLOCK)
    got = (got * got_scales.unsqueeze(-1)).reshape(n, inter)
    # silu(1) * 1 = 0.7311
    assert float((got - 0.7311).abs().max()) < 0.02
