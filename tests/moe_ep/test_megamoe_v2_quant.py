# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device test: the v2 fused quantizer must match the torch oracle.

Validating the encoding in isolation matters because once it is embedded in
kernel A, a scale-placement bug is indistinguishable from a dispatch or GEMM
bug -- it just shows up as a slightly wrong output.  Here the scales are read
back through the same swizzle the MMA uses, so both the values and their
placement are checked.

    pytest tests/moe_ep/test_megamoe_v2_quant.py -q
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
    major, _minor = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("NVFP4 conversion needs sm_100a / sm_103a")


def _run_device_quantize(activation: torch.Tensor, norm_const: float):
    """Launch the v2 quantizer; returns (codes float32, scales float32)."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.quant import quantize_launch

    num_tokens, hidden = activation.shape
    num_k_atoms = sf_layout.num_k_atoms_for(hidden, NVFP4_BLOCK)
    scale_elems = sf_layout.buffer_words(num_tokens, num_k_atoms=num_k_atoms) * 4

    # FP4 codes are packed 2-per-byte; cute sees the unpacked element count.
    codes_bytes = torch.zeros(
        num_tokens, hidden // 2, dtype=torch.uint8, device="cuda"
    )
    scales = torch.zeros(scale_elems, dtype=torch.float8_e4m3fn, device="cuda")

    act_cute = cutlass_torch.from_dlpack(activation, assumed_align=16)
    # A uint8 buffer viewed as float4_e2m1fn_x2 presents to cute as
    # Float4E2M1FN with the *logical* element count, so no in-trace recast is
    # needed (cute.recast_tensor requires an MLIR context and is trace-only).
    codes_cute = cutlass_torch.from_dlpack(
        codes_bytes.view(torch.float4_e2m1fn_x2), assumed_align=16
    )
    scales_cute = cutlass_torch.from_dlpack(scales, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        quantize_launch,
        act_cute,
        codes_cute,
        scales_cute,
        cutlass.Float32(norm_const),
        cutlass.Int32(num_tokens),
        stream,
        hidden=hidden,
        num_k_atoms=num_k_atoms,
    )
    compiled(
        act_cute,
        codes_cute,
        scales_cute,
        cutlass.Float32(norm_const),
        cutlass.Int32(num_tokens),
        stream,
    )
    torch.cuda.synchronize()
    return codes_bytes, scales


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 pairs -> float32 E2M1 magnitudes, low nibble first."""
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    def decode(nib):
        sign = torch.where(nib >= 8, -1.0, 1.0)
        return sign * levels[(nib & 0x7).long()]
    out = torch.stack([decode(lo), decode(hi)], dim=-1)
    return out.reshape(packed.shape[0], -1)


def _gather_scales(
    flat: torch.Tensor, *, num_tokens: int, hidden: int
) -> torch.Tensor:
    """Read the swizzled scale buffer back into (tokens, blocks) order."""
    num_blocks = hidden // NVFP4_BLOCK
    num_k_atoms = sf_layout.num_k_atoms_for(hidden, NVFP4_BLOCK)
    as_f32 = flat.to(torch.float32)
    out = torch.zeros(num_tokens, num_blocks, dtype=torch.float32, device=flat.device)
    for token in range(num_tokens):
        row_block, t_in = divmod(token, 128)
        for block in range(num_blocks):
            k_atom, k_bank = divmod(block, 4)
            atom = row_block * num_k_atoms + k_atom
            idx = atom * 512 + (t_in % 32) * 16 + (t_in // 32) * 4 + k_bank
            out[token, block] = as_f32[idx]
    return out


@pytest.mark.parametrize("hidden", [2048, 7168])
@pytest.mark.parametrize("num_tokens", [1, 96, 130])
def test_device_quantize_matches_oracle(hidden, num_tokens):
    """Codes and scales must match the reference bit for bit."""
    _require_blackwell()
    g = torch.Generator(device="cuda").manual_seed(5)
    act = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )

    packed, scales_flat = _run_device_quantize(act, 1.0)

    expected = quantize_nvfp4(act.float(), 1.0)
    got_codes = _unpack_fp4(packed)
    got_scales = _gather_scales(
        scales_flat, num_tokens=num_tokens, hidden=hidden
    )

    torch.testing.assert_close(got_scales, expected.scales, atol=0, rtol=0)
    torch.testing.assert_close(got_codes, expected.codes, atol=0, rtol=0)


def test_device_quantize_handles_zero_rows():
    """All-zero blocks must encode as zero, not NaN (the reciprocal guard)."""
    _require_blackwell()
    hidden = 2048
    act = torch.zeros(64, hidden, dtype=torch.bfloat16, device="cuda")
    act[1, :NVFP4_BLOCK] = 1.0  # one live block among zeros

    packed, scales_flat = _run_device_quantize(act, 1.0)
    codes = _unpack_fp4(packed)
    scales = _gather_scales(scales_flat, num_tokens=64, hidden=hidden)

    assert torch.isfinite(codes).all(), "zero block produced non-finite codes"
    assert torch.isfinite(scales).all()
    assert float(codes[0].abs().max()) == 0.0
    assert float(scales[0].abs().max()) == 0.0
    assert float(codes[1, :NVFP4_BLOCK].abs().max()) > 0.0


def test_device_quantize_respects_norm_const():
    _require_blackwell()
    hidden = 2048
    g = torch.Generator(device="cuda").manual_seed(9)
    act = torch.randn(32, hidden, dtype=torch.bfloat16, device="cuda", generator=g)

    packed, scales_flat = _run_device_quantize(act, 4.0)
    expected = quantize_nvfp4(act.float(), 4.0)

    got_scales = _gather_scales(scales_flat, num_tokens=32, hidden=hidden)
    torch.testing.assert_close(got_scales, expected.scales, atol=0, rtol=0)
    torch.testing.assert_close(
        _unpack_fp4(packed), expected.codes, atol=0, rtol=0
    )


def test_device_quantize_leaves_padding_rows_untouched():
    """Rows past num_tokens must not be written (they belong to other experts)."""
    _require_blackwell()
    hidden = 2048
    live = 40
    g = torch.Generator(device="cuda").manual_seed(17)
    act = torch.randn(128, hidden, dtype=torch.bfloat16, device="cuda", generator=g)

    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cuda.bindings.driver as cuda

    from flashinfer.moe_ep.kernel_src.megamoe_v2.quant import quantize_launch

    num_k_atoms = sf_layout.num_k_atoms_for(hidden, NVFP4_BLOCK)
    scale_elems = sf_layout.buffer_words(128, num_k_atoms=num_k_atoms) * 4
    codes_bytes = torch.full(
        (128, hidden // 2), 0xEE, dtype=torch.uint8, device="cuda"
    )
    scales = torch.zeros(scale_elems, dtype=torch.float8_e4m3fn, device="cuda")

    act_cute = cutlass_torch.from_dlpack(act, assumed_align=16)
    codes_cute = cutlass_torch.from_dlpack(
        codes_bytes.view(torch.float4_e2m1fn_x2), assumed_align=16
    )
    scales_cute = cutlass_torch.from_dlpack(scales, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cute.compile(
        quantize_launch, act_cute, codes_cute, scales_cute,
        cutlass.Float32(1.0), cutlass.Int32(live), stream,
        hidden=hidden, num_k_atoms=num_k_atoms,
    )(
        act_cute, codes_cute, scales_cute,
        cutlass.Float32(1.0), cutlass.Int32(live), stream,
    )
    torch.cuda.synchronize()

    assert bool((codes_bytes[live:] == 0xEE).all()), (
        "quantizer wrote past num_tokens"
    )
