from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from flashinfer.mla import _projection


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    mean_us: float


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


def _capture(call: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        call()
        stream.synchronize()
        graph.capture_begin()
        call()
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def _measure(
    name: str,
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
) -> BenchmarkResult:
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return BenchmarkResult(
        name=name,
        mean_us=float(start.elapsed_time(end) * 1000.0 / iterations),
    )


def benchmark(iterations: int = 200) -> tuple[BenchmarkResult, ...]:
    torch.manual_seed(7)
    device = torch.device("cuda")
    hidden_states = torch.randn((96, 7168), device=device, dtype=torch.bfloat16) / 8
    qkv_a_weight = torch.randn((2112, 7168), device=device, dtype=torch.bfloat16) / 8
    packed_qkv_a_weight = _projection.prepare_qkv_a_proj_weight(qkv_a_weight)
    q_norm_weight = torch.randn((1536,), device=device, dtype=torch.bfloat16)
    kv_norm_weight = torch.randn((512,), device=device, dtype=torch.bfloat16)
    positions = torch.arange(96, device=device, dtype=torch.int64)
    cos_sin_cache = torch.randn((256, 64), device=device, dtype=torch.bfloat16)
    eps = 1e-6

    custom_workspace = torch.empty((96, 2112), device=device, dtype=torch.bfloat16)
    custom_q = torch.empty((96, 1536), device=device, dtype=torch.bfloat16)
    custom_kv = torch.empty((96, 512), device=device, dtype=torch.bfloat16)
    custom_k = torch.empty((96, 1, 64), device=device, dtype=torch.bfloat16)

    def custom_call() -> None:
        _projection._fused_qkv_a_proj_norm_rope_impl(
            custom_workspace,
            custom_q,
            custom_kv,
            custom_k,
            hidden_states,
            qkv_a_weight,
            q_norm_weight,
            kv_norm_weight,
            positions,
            cos_sin_cache,
            eps,
        )

    packed_workspace = torch.empty((96, 2112), device=device, dtype=torch.bfloat16)
    packed_q = torch.empty((96, 1536), device=device, dtype=torch.bfloat16)
    packed_kv = torch.empty((96, 512), device=device, dtype=torch.bfloat16)
    packed_k = torch.empty((96, 1, 64), device=device, dtype=torch.bfloat16)

    def packed_call() -> None:
        _projection._fused_qkv_a_proj_norm_rope_impl(
            packed_workspace,
            packed_q,
            packed_kv,
            packed_k,
            hidden_states,
            packed_qkv_a_weight,
            q_norm_weight,
            kv_norm_weight,
            positions,
            cos_sin_cache,
            eps,
        )

    cublas_workspace = torch.empty((96, 2112), device=device, dtype=torch.bfloat16)
    cublas_q = torch.empty((96, 1536), device=device, dtype=torch.bfloat16)
    cublas_kv = torch.empty((96, 512), device=device, dtype=torch.bfloat16)
    cublas_k = torch.empty((96, 1, 64), device=device, dtype=torch.bfloat16)
    post_module = _projection._get_fused_qkv_a_proj_norm_rope_module()

    def cublas_post_call() -> None:
        torch.mm(hidden_states, qkv_a_weight.t(), out=cublas_workspace)
        post_module.fused_qkv_a_proj_norm_rope_post_sm100(
            cublas_workspace,
            q_norm_weight,
            kv_norm_weight,
            positions,
            cos_sin_cache,
            cublas_q,
            cublas_kv,
            cublas_k,
            eps,
        )

    def reference_call() -> None:
        _reference(
            hidden_states,
            qkv_a_weight,
            q_norm_weight,
            kv_norm_weight,
            positions,
            cos_sin_cache,
            eps,
        )

    graphs = (
        ("custom_row_major_and_post", _capture(custom_call)),
        ("custom_packed_and_post", _capture(packed_call)),
        ("cublas_and_fused_post", _capture(cublas_post_call)),
        ("pytorch_reference", _capture(reference_call)),
    )
    return tuple(_measure(name, graph, iterations=iterations) for name, graph in graphs)


if __name__ == "__main__":
    for result in benchmark():
        print(f"{result.name:28s} {result.mean_us:8.3f} us")
