"""Tests for fused MoE add, residual accumulation, and RMSNorm."""

import pytest
import torch

import flashinfer
import flashinfer.jit.fused_moe_add_residual_rmsnorm as fused_moe_norm_jit
import flashinfer.norm as norm
from flashinfer.norm import get_fused_moe_add_residual_rmsnorm_sm100_module


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_HIDDEN_SIZE = 7168
_TARGET_NUM_TOKENS = (64, 96, 128, 160, 192, 224, 256)


def _skip_if_not_sm100_or_sm103() -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("the optimized kernel requires SM100 or SM103")


def _make_inputs(
    num_tokens: int, hidden_size: int = _HIDDEN_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (num_tokens, hidden_size)
    return (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16),
    )


def _reference(
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    moe_output = routed_output + shared_output
    residual_fp32 = moe_output.float() + residual.float()
    residual_out = residual_fp32.to(residual.dtype)
    inv_rms = torch.rsqrt(residual_fp32.square().mean(dim=-1, keepdim=True) + eps)
    hidden_states = (residual_fp32 * inv_rms * weight.float()).to(routed_output.dtype)
    return hidden_states, residual_out


def _assert_outputs(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual_hidden, actual_residual = actual
    expected_hidden, expected_residual = expected
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=2e-2, atol=2e-2)
    assert torch.equal(actual_residual, expected_residual)


@pytest.mark.parametrize("num_tokens", _TARGET_NUM_TOKENS)
def test_fused_moe_add_residual_rmsnorm_sm100_family(num_tokens: int) -> None:
    _skip_if_not_sm100_or_sm103()
    eps = 1e-6
    inputs = _make_inputs(num_tokens)
    expected = _reference(*inputs, eps)
    _assert_outputs(flashinfer.fused_moe_add_residual_rmsnorm(*inputs, eps), expected)


@pytest.mark.parametrize("num_tokens", (0, 1, 31, 257))
def test_fused_moe_add_residual_rmsnorm_boundary_tokens(num_tokens: int) -> None:
    _skip_if_not_sm100_or_sm103()
    eps = 1e-5
    inputs = _make_inputs(num_tokens)
    expected = _reference(*inputs, eps)
    _assert_outputs(flashinfer.fused_moe_add_residual_rmsnorm(*inputs, eps), expected)


def test_sm100_family_jit_module_entrypoint() -> None:
    _skip_if_not_sm100_or_sm103()
    eps = 1e-6
    inputs = _make_inputs(64)
    hidden_states = torch.empty_like(inputs[0])
    residual_out = torch.empty_like(inputs[0])
    module = get_fused_moe_add_residual_rmsnorm_sm100_module()
    module.fused_moe_add_residual_rmsnorm_sm100(
        *inputs, hidden_states, residual_out, eps
    )
    _assert_outputs((hidden_states, residual_out), _reference(*inputs, eps))


def test_preallocated_outputs() -> None:
    _skip_if_not_sm100_or_sm103()
    inputs = _make_inputs(64)
    hidden_states = torch.empty_like(inputs[0])
    residual_out = torch.empty_like(inputs[0])
    actual = flashinfer.fused_moe_add_residual_rmsnorm(
        *inputs, hidden_states=hidden_states, residual_out=residual_out
    )
    assert actual[0] is hidden_states
    assert actual[1] is residual_out
    _assert_outputs(actual, _reference(*inputs, 1e-6))


def test_portable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLASHINFER_DISABLE_FUSED_MOE_ADD_RESIDUAL_RMSNORM_SM100", "1")
    inputs = _make_inputs(17, hidden_size=512)
    _assert_outputs(
        flashinfer.fused_moe_add_residual_rmsnorm(*inputs),
        _reference(*inputs, 1e-6),
    )


@pytest.mark.parametrize(
    ("compute_capability", "cuda_version", "expected"),
    (
        ((10, 0), "12.8", True),
        ((10, 3), "12.9", True),
        ((10, 3), "12.8", False),
        ((10, 7), "13.0", False),
    ),
)
def test_sm100_family_dispatch_requirements(
    monkeypatch: pytest.MonkeyPatch,
    compute_capability: tuple[int, int],
    cuda_version: str,
    expected: bool,
) -> None:
    monkeypatch.delenv(
        "FLASHINFER_DISABLE_FUSED_MOE_ADD_RESIDUAL_RMSNORM_SM100", raising=False
    )
    monkeypatch.setattr(
        norm, "get_compute_capability", lambda _device: compute_capability
    )
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    inputs = _make_inputs(1)
    hidden_states = torch.empty_like(inputs[0])
    residual_out = torch.empty_like(inputs[0])
    assert (
        norm._use_fused_moe_add_residual_rmsnorm_sm100(  # pyright: ignore[reportPrivateUsage]
            *inputs, hidden_states, residual_out
        )
        is expected
    )


def test_jit_architecture_flags_by_cuda_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused_moe_norm_jit, "is_cuda_version_at_least", lambda _version: False
    )
    cuda_12_8_flags = fused_moe_norm_jit.gen_fused_moe_add_residual_rmsnorm_sm100_module().extra_cuda_cflags
    monkeypatch.setattr(
        fused_moe_norm_jit, "is_cuda_version_at_least", lambda _version: True
    )
    cuda_12_9_flags = fused_moe_norm_jit.gen_fused_moe_add_residual_rmsnorm_sm100_module().extra_cuda_cflags

    assert any("compute_100a" in flag for flag in cuda_12_8_flags)
    assert not any("compute_103a" in flag for flag in cuda_12_8_flags)
    assert any("compute_100a" in flag for flag in cuda_12_9_flags)
    assert any("compute_103a" in flag for flag in cuda_12_9_flags)


def test_cuda_graph_capture() -> None:
    _skip_if_not_sm100_or_sm103()
    inputs = _make_inputs(64)
    hidden_states = torch.empty_like(inputs[0])
    residual_out = torch.empty_like(inputs[0])

    flashinfer.fused_moe_add_residual_rmsnorm(
        *inputs, hidden_states=hidden_states, residual_out=residual_out
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashinfer.fused_moe_add_residual_rmsnorm(
            *inputs, hidden_states=hidden_states, residual_out=residual_out
        )

    for tensor in inputs[:3]:
        tensor.copy_(torch.randn_like(tensor) / 8)
    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs((hidden_states, residual_out), _reference(*inputs, 1e-6))


def test_rejects_invalid_inputs() -> None:
    inputs = _make_inputs(8, hidden_size=64)
    with pytest.raises(ValueError, match="shared_output must be bfloat16"):
        flashinfer.fused_moe_add_residual_rmsnorm(
            inputs[0], inputs[1].float(), inputs[2], inputs[3]
        )

    output = torch.empty_like(inputs[0])
    with pytest.raises(ValueError, match="must not alias"):
        flashinfer.fused_moe_add_residual_rmsnorm(
            *inputs, hidden_states=output, residual_out=output
        )
