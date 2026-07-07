"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.jit.gemm import gen_gemm_gated_act_sm90_module
from flashinfer.utils import has_flashinfer_jit_cache, is_sm90a_supported

# (dtype_a, out_dtype, activation) configs exercised by this file. Keep in
# sync with the parametrizations below so the warmup fixture pre-builds
# every JIT module the tests need.
JIT_CONFIGS = [
    (torch.bfloat16, torch.bfloat16, "silu"),
    (torch.bfloat16, torch.bfloat16, "gelu"),
    (torch.bfloat16, torch.bfloat16, "relu"),
    (torch.float16, torch.float16, "silu"),
    (torch.float8_e4m3fn, torch.bfloat16, "silu"),
    (torch.float8_e4m3fn, torch.float16, "silu"),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, "silu"),
]


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    if is_sm90a_supported(torch.device("cuda:0")):
        jit_specs = [
            gen_gemm_gated_act_sm90_module(dtype_a, out_dtype, activation)
            for dtype_a, out_dtype, activation in JIT_CONFIGS
        ]
        flashinfer.jit.build_jit_specs(jit_specs, verbose=False)
    yield


def _skip_if_no_sm90a():
    if not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("gemm_gated_act requires SM90a")


_ACT_FNS = {
    "silu": F.silu,
    # CUTLASS's GELU functor is the exact (erf-based) GELU.
    "gelu": F.gelu,
    "relu": F.relu,
}


def reference_gemm_gated_act(a, weight, activation="silu", bias=None, alpha=1.0):
    """Unfused fp32 reference of the kernel contract:
    out = (alpha * a @ w_up.T + b_up) * act(alpha * a @ w_gate.T + b_gate)
    with weight packed [W_up ; W_gate].
    """
    intermediate = weight.shape[0] // 2
    a32 = a.to(torch.float32)
    w32 = weight.to(torch.float32)
    up = alpha * (a32 @ w32[:intermediate].T)
    gate = alpha * (a32 @ w32[intermediate:].T)
    if bias is not None:
        up = up + bias[:intermediate].to(torch.float32)
        gate = gate + bias[intermediate:].to(torch.float32)
    return up * _ACT_FNS[activation](gate)


def to_float8_e4m3(x: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax
    x_q = (x.to(torch.float32) * scale).clamp(finfo.min, finfo.max)
    return x_q.to(torch.float8_e4m3fn), scale.float().reciprocal()


def assert_close_with_cos_sim(res, reference, rtol, atol):
    cos_sim = F.cosine_similarity(
        reference.float().reshape(-1), res.float().reshape(-1), dim=0
    )
    assert cos_sim > 0.99, f"cosine similarity too low: {cos_sim}"
    torch.testing.assert_close(res.float(), reference.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("m", [1, 7, 128, 501])
@pytest.mark.parametrize("intermediate,k", [(512, 512), (4096, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_gemm_gated_act_16bit(m, intermediate, k, dtype, use_bias):
    _skip_if_no_sm90a()
    torch.manual_seed(42)
    a = torch.randn(m, k, device="cuda", dtype=dtype) / 8
    weight = torch.randn(2 * intermediate, k, device="cuda", dtype=dtype) / 8
    bias = (
        torch.randn(2 * intermediate, device="cuda", dtype=dtype) if use_bias else None
    )

    out = flashinfer.gemm_gated_act(a, weight, activation="silu", bias=bias)
    assert out.shape == (m, intermediate)
    assert out.dtype == dtype

    reference = reference_gemm_gated_act(a, weight, "silu", bias=bias)
    assert_close_with_cos_sim(out, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("activation", ["silu", "gelu", "relu"])
def test_gemm_gated_act_activations(activation):
    _skip_if_no_sm90a()
    torch.manual_seed(0)
    m, intermediate, k = 64, 1024, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 8
    weight = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) / 8

    out = flashinfer.gemm_gated_act(a, weight, activation=activation)
    reference = reference_gemm_gated_act(a, weight, activation)
    assert_close_with_cos_sim(out, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("m", [7, 128, 501])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("alpha_as_tensor", [False, True])
def test_gemm_gated_act_fp8_in(m, out_dtype, alpha_as_tensor):
    _skip_if_no_sm90a()
    torch.manual_seed(1)
    intermediate, k = 2048, 1024
    a_ref = torch.randn(m, k, device="cuda", dtype=torch.float32) / 8
    w_ref = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.float32) / 8
    a_fp8, a_scale = to_float8_e4m3(a_ref)
    w_fp8, w_scale = to_float8_e4m3(w_ref)
    # alpha multiplies the accumulator BEFORE the nonlinear activation, so it
    # must carry the fp8 dequant scales.
    alpha = float(a_scale * w_scale)
    alpha_arg = (
        torch.tensor(alpha, device="cuda", dtype=torch.float32)
        if alpha_as_tensor
        else alpha
    )

    out = flashinfer.gemm_gated_act(
        a_fp8, w_fp8, activation="silu", alpha=alpha_arg, out_dtype=out_dtype
    )
    assert out.dtype == out_dtype

    reference = reference_gemm_gated_act(a_fp8, w_fp8, "silu", alpha=alpha)
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.float().reshape(-1), dim=0)
    assert cos_sim > 0.99


def test_gemm_gated_act_fp8_out():
    _skip_if_no_sm90a()
    torch.manual_seed(2)
    m, intermediate, k = 128, 2048, 1024
    a_ref = torch.randn(m, k, device="cuda", dtype=torch.float32) / 8
    w_ref = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.float32) / 8
    a_fp8, a_scale = to_float8_e4m3(a_ref)
    w_fp8, w_scale = to_float8_e4m3(w_ref)
    alpha = float(a_scale * w_scale)

    reference = reference_gemm_gated_act(a_fp8, w_fp8, "silu", alpha=alpha)
    out_amax = reference.abs().amax().clamp(min=1e-12)
    out_quant_scale = torch.finfo(torch.float8_e4m3fn).max / out_amax
    out_scale = out_quant_scale.reshape(1).float()  # multiplied in the epilogue

    out = flashinfer.gemm_gated_act(
        a_fp8,
        w_fp8,
        activation="silu",
        alpha=alpha,
        out_scale=out_scale,
        out_dtype=torch.float8_e4m3fn,
    )
    assert out.dtype == torch.float8_e4m3fn

    dequant = out.float() / out_quant_scale
    cos_sim = F.cosine_similarity(reference.reshape(-1), dequant.reshape(-1), dim=0)
    assert cos_sim > 0.99


def test_gemm_gated_act_out_inplace():
    _skip_if_no_sm90a()
    torch.manual_seed(3)
    m, intermediate, k = 32, 512, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 8
    weight = torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) / 8
    out = torch.empty(m, intermediate, device="cuda", dtype=torch.bfloat16)

    ret = flashinfer.gemm_gated_act(a, weight, out=out)
    assert ret.data_ptr() == out.data_ptr()
    reference = reference_gemm_gated_act(a, weight)
    assert_close_with_cos_sim(out, reference, rtol=2e-2, atol=2e-2)


def test_prepare_gated_act_gemm_weights_matches_silu_and_mul():
    """Cross-check the packing helper against the silu_and_mul contract:
    silu_and_mul consumes [gate | up] (gate FIRST), the fused kernel consumes
    [W_up ; W_gate] (up first)."""
    _skip_if_no_sm90a()
    torch.manual_seed(4)
    m, intermediate, k = 64, 1024, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / 8
    # HF convention: gate rows first.
    hf_weight = (
        torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) / 8
    )

    # Unfused flow: gate_up GEMM -> silu_and_mul.
    y = a @ hf_weight.T  # [m, 2*intermediate] = [gate | up]
    unfused = flashinfer.activation.silu_and_mul(y)

    fused = flashinfer.gemm_gated_act(
        a, flashinfer.prepare_gated_act_gemm_weights(hf_weight)
    )
    assert_close_with_cos_sim(fused, unfused, rtol=2e-2, atol=2e-2)


def test_gemm_gated_act_validation_errors():
    _skip_if_no_sm90a()
    a = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="activation"):
        flashinfer.gemm_gated_act(a, weight, activation="swiglu_limit")
    with pytest.raises(ValueError, match="divisible by 16"):
        flashinfer.gemm_gated_act(a, weight[:1000])
    with pytest.raises(ValueError, match="dtype"):
        flashinfer.gemm_gated_act(a, weight.to(torch.float16))
    with pytest.raises(ValueError, match="out_scale"):
        flashinfer.gemm_gated_act(
            a.to(torch.float8_e4m3fn),
            weight.to(torch.float8_e4m3fn),
            out_dtype=torch.float8_e4m3fn,
        )
    with pytest.raises(ValueError, match="K mismatch"):
        flashinfer.gemm_gated_act(a[:, :256], weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
