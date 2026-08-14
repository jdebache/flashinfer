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

from __future__ import annotations

import functools

import torch

from ..jit.fused_qkv_a_proj_norm_rope import (
    gen_fused_qkv_a_proj_norm_rope_sm100_module,
)
from ..utils import register_custom_op, register_fake_op
from ..gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    SplitKTactic,
    prepare_packed_weight,
    run_splitk_dense,
    run_splitk_dense_packed_weight,
)

_TOKENS = 96
_IN_FEATURES = 7168
_Q_FEATURES = 1536
_KV_FEATURES = 512
_K_PE_FEATURES = 64
_OUT_FEATURES = _Q_FEATURES + _KV_FEATURES + _K_PE_FEATURES
_PACKED_OUT_FEATURES = 2176
_PACKED_WEIGHT_SHAPE = (
    _IN_FEATURES // 128,
    _PACKED_OUT_FEATURES,
    128,
)
_GEMM_TACTIC = SplitKTactic(mma_m=64, mma_n=48, split_k=2, ab_stages=7)


@functools.cache
def _get_fused_qkv_a_proj_norm_rope_module():
    return gen_fused_qkv_a_proj_norm_rope_sm100_module().build_and_load()


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtypes: tuple[torch.dtype, ...],
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype not in dtypes:
        expected = ", ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype {expected}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_inputs(
    hidden_states: torch.Tensor,
    qkv_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    if hidden_states.device.type != "cuda":
        raise ValueError("hidden_states must be a CUDA tensor")
    device = hidden_states.device
    _check_tensor(
        hidden_states,
        "hidden_states",
        shape=(_TOKENS, _IN_FEATURES),
        device=device,
        dtypes=(torch.bfloat16,),
    )
    if tuple(qkv_a_weight.shape) not in (
        (_OUT_FEATURES, _IN_FEATURES),
        _PACKED_WEIGHT_SHAPE,
    ):
        raise ValueError(
            "qkv_a_weight must have row-major shape (2112, 7168) or packed "
            f"shape {_PACKED_WEIGHT_SHAPE}, got {tuple(qkv_a_weight.shape)}"
        )
    if qkv_a_weight.device != device:
        raise ValueError(
            f"qkv_a_weight must be on {device}, got {qkv_a_weight.device}"
        )
    if qkv_a_weight.dtype != torch.bfloat16:
        raise TypeError(
            "qkv_a_weight must have dtype torch.bfloat16, "
            f"got {qkv_a_weight.dtype}"
        )
    if not qkv_a_weight.is_contiguous():
        raise ValueError("qkv_a_weight must be contiguous")
    _check_tensor(
        q_norm_weight,
        "q_norm_weight",
        shape=(_Q_FEATURES,),
        device=device,
        dtypes=(torch.bfloat16,),
    )
    _check_tensor(
        kv_norm_weight,
        "kv_norm_weight",
        shape=(_KV_FEATURES,),
        device=device,
        dtypes=(torch.bfloat16,),
    )
    _check_tensor(
        positions,
        "positions",
        shape=(_TOKENS,),
        device=device,
        dtypes=(torch.int64,),
    )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != _K_PE_FEATURES:
        raise ValueError(
            "cos_sin_cache must have shape (max_position, 64), "
            f"got {tuple(cos_sin_cache.shape)}"
        )
    if cos_sin_cache.shape[0] == 0:
        raise ValueError("cos_sin_cache must contain at least one position")
    if cos_sin_cache.device != device:
        raise ValueError(
            f"cos_sin_cache must be on {device}, got {cos_sin_cache.device}"
        )
    if cos_sin_cache.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "cos_sin_cache must have dtype torch.bfloat16 or torch.float32, "
            f"got {cos_sin_cache.dtype}"
        )
    if not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous")

    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise RuntimeError(
            f"fused_qkv_a_proj_norm_rope requires SM100 or SM103, got sm_{major}{minor}"
        )


@register_custom_op(
    "flashinfer::fused_qkv_a_proj_norm_rope",
    mutates_args=("qkv_workspace", "q_out", "kv_out", "k_pe_out"),
)
def _fused_qkv_a_proj_norm_rope_impl(
    qkv_workspace: torch.Tensor,
    q_out: torch.Tensor,
    kv_out: torch.Tensor,
    k_pe_out: torch.Tensor,
    hidden_states: torch.Tensor,
    qkv_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> None:
    if qkv_a_weight.ndim == 3:
        run_splitk_dense_packed_weight(
            hidden_states,
            qkv_a_weight,
            qkv_workspace,
            _GEMM_TACTIC,
        )
    else:
        run_splitk_dense(
            hidden_states,
            qkv_a_weight.t(),
            None,
            qkv_workspace,
            False,
            _GEMM_TACTIC,
        )
    _get_fused_qkv_a_proj_norm_rope_module().fused_qkv_a_proj_norm_rope_post_sm100(
        qkv_workspace,
        q_norm_weight,
        kv_norm_weight,
        positions,
        cos_sin_cache,
        q_out,
        kv_out,
        k_pe_out,
        eps,
    )


@register_fake_op("flashinfer::fused_qkv_a_proj_norm_rope")
def _fused_qkv_a_proj_norm_rope_fake(
    qkv_workspace: torch.Tensor,
    q_out: torch.Tensor,
    kv_out: torch.Tensor,
    k_pe_out: torch.Tensor,
    hidden_states: torch.Tensor,
    qkv_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> None:
    pass


def fused_qkv_a_proj_norm_rope(
    hidden_states: torch.Tensor,
    qkv_a_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the DeepSeek-V3 QKV-A projection, latent RMSNorms, and K RoPE.

    This fixed-shape SM100/SM103 kernel computes the BF16 projection
    ``hidden_states @ qkv_a_weight.T`` with FP32 accumulation. It preserves
    the BF16 projection boundary before applying independent FP32 RMSNorms to
    the 1,536-dimensional Q latent and 512-dimensional KV latent. The final
    64 dimensions use GPT-J/interleaved RoPE, where adjacent values form a
    rotary pair.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Contiguous BF16 input with shape ``(96, 7168)``.
    qkv_a_weight : torch.Tensor
        Contiguous BF16 linear weight with row-major shape ``(2112, 7168)``
        or the packed shape ``(56, 2176, 128)`` returned by
        :func:`prepare_qkv_a_proj_weight`. The packed form is intended for a
        static model weight and avoids descriptor-strided weight tiles.
    q_norm_weight : torch.Tensor
        Contiguous BF16 Q RMSNorm weight with shape ``(1536,)``.
    kv_norm_weight : torch.Tensor
        Contiguous BF16 KV RMSNorm weight with shape ``(512,)``.
    positions : torch.Tensor
        Contiguous int64 position indices with shape ``(96,)``. Every value
        must index a valid row of ``cos_sin_cache``.
    cos_sin_cache : torch.Tensor
        Contiguous BF16 or FP32 cache with shape ``(max_position, 64)``. The
        first 32 columns contain cosines and the final 32 contain sines.
    eps : float
        Epsilon added to each RMS variance, default ``1e-6``.

    Returns
    -------
    q_latent_normed : torch.Tensor
        Contiguous BF16 tensor with shape ``(96, 1536)``.
    kv_latent_normed : torch.Tensor
        Contiguous BF16 tensor with shape ``(96, 512)``.
    k_pe_rotated : torch.Tensor
        Contiguous BF16 tensor with shape ``(96, 1, 64)``.
    """

    _check_inputs(
        hidden_states,
        qkv_a_weight,
        q_norm_weight,
        kv_norm_weight,
        positions,
        cos_sin_cache,
    )
    qkv_workspace = torch.empty(
        (_TOKENS, _OUT_FEATURES),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    q_out = torch.empty(
        (_TOKENS, _Q_FEATURES),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    kv_out = torch.empty(
        (_TOKENS, _KV_FEATURES),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    k_pe_out = torch.empty(
        (_TOKENS, 1, _K_PE_FEATURES),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    _fused_qkv_a_proj_norm_rope_impl(
        qkv_workspace,
        q_out,
        kv_out,
        k_pe_out,
        hidden_states,
        qkv_a_weight,
        q_norm_weight,
        kv_norm_weight,
        positions,
        cos_sin_cache,
        eps,
    )
    return q_out, kv_out, k_pe_out


def prepare_qkv_a_proj_weight(qkv_a_weight: torch.Tensor) -> torch.Tensor:
    """Pack a DeepSeek-V3 QKV-A projection weight for the fused operator.

    Call this once when loading the model and retain the returned tensor. The
    layout satisfies
    ``packed[k_block, n, k_inner] = weight[n, 128 * k_block + k_inner]``.

    Parameters
    ----------
    qkv_a_weight : torch.Tensor
        Contiguous BF16 weight with shape ``(2112, 7168)``.

    Returns
    -------
    torch.Tensor
        Contiguous BF16 weight with shape ``(56, 2176, 128)``. The final 64
        padded output features are zero and are not written by
        :func:`fused_qkv_a_proj_norm_rope`.
    """

    if not isinstance(qkv_a_weight, torch.Tensor):
        raise TypeError("qkv_a_weight must be a torch tensor")
    if tuple(qkv_a_weight.shape) != (_OUT_FEATURES, _IN_FEATURES):
        raise ValueError(
            "qkv_a_weight must have shape (2112, 7168), "
            f"got {tuple(qkv_a_weight.shape)}"
        )
    if qkv_a_weight.dtype != torch.bfloat16:
        raise TypeError(
            "qkv_a_weight must have dtype torch.bfloat16, "
            f"got {qkv_a_weight.dtype}"
        )
    if not qkv_a_weight.is_contiguous():
        raise ValueError("qkv_a_weight must be contiguous")
    return prepare_packed_weight(qkv_a_weight, padded_n=_PACKED_OUT_FEATURES)


__all__ = (
    "fused_qkv_a_proj_norm_rope",
    "prepare_qkv_a_proj_weight",
)
