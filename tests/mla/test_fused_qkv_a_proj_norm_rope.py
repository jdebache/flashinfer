from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import flashinfer


def _reference(
    hidden_states: torch.Tensor,
    qkv_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = F.linear(hidden_states, qkv_a_weight)
    q_latent, kv_latent, k_pe = torch.split(qkv, (1536, 512, 64), dim=-1)

    q_fp32 = q_latent.float()
    q_out = (
        q_fp32
        * torch.rsqrt(q_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * q_norm_weight.float()
    ).bfloat16()

    kv_fp32 = kv_latent.float()
    kv_out = (
        kv_fp32
        * torch.rsqrt(kv_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * kv_norm_weight.float()
    ).bfloat16()

    cos_sin = cos_sin_cache.index_select(0, positions).float()
    cosines, sines = cos_sin.chunk(2, dim=-1)
    k_fp32 = k_pe.float()
    k_even = k_fp32[..., 0::2]
    k_odd = k_fp32[..., 1::2]
    k_out = torch.stack(
        (
            k_even * cosines - k_odd * sines,
            k_odd * cosines + k_even * sines,
        ),
        dim=-1,
    ).flatten(-2)
    return q_out, kv_out, k_out.bfloat16().unsqueeze(1)


def _requires_sm100_family() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100 or SM103 is required")


@pytest.mark.parametrize("cache_dtype", (torch.bfloat16, torch.float32))
def test_fused_qkv_a_proj_norm_rope(cache_dtype: torch.dtype) -> None:
    _requires_sm100_family()
    torch.manual_seed(7)
    device = torch.device("cuda")
    hidden_states = torch.randn((96, 7168), device=device, dtype=torch.bfloat16) / 8
    qkv_a_weight = torch.randn((2112, 7168), device=device, dtype=torch.bfloat16) / 8
    q_norm_weight = torch.randn((1536,), device=device, dtype=torch.bfloat16)
    kv_norm_weight = torch.randn((512,), device=device, dtype=torch.bfloat16)
    positions = torch.randperm(256, device=device, dtype=torch.int64)[:96].contiguous()
    cos_sin_cache = torch.randn((256, 64), device=device, dtype=cache_dtype)
    eps = 1e-6

    actual = flashinfer.fused_qkv_a_proj_norm_rope(
        hidden_states,
        qkv_a_weight,
        q_norm_weight,
        kv_norm_weight,
        positions,
        cos_sin_cache,
        eps,
    )
    expected = _reference(
        hidden_states,
        qkv_a_weight,
        q_norm_weight,
        kv_norm_weight,
        positions,
        cos_sin_cache,
        eps,
    )

    assert tuple(output.shape for output in actual) == (
        (96, 1536),
        (96, 512),
        (96, 1, 64),
    )
    assert all(output.dtype == torch.bfloat16 for output in actual)
    assert all(output.is_contiguous() for output in actual)
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            output.float(),
            reference.float(),
            rtol=2e-2,
            atol=4e-2,
        )
