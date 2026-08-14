"""Benchmark the SM100/SM103 fused MoE add, residual, and RMSNorm kernel."""

import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


_HIDDEN_SIZE = 7168
_DEFAULT_NUM_TOKENS = (64, 96, 128, 160, 192, 224, 256)


def _bandwidth_gb_s(num_tokens: int, latency_ms: float) -> float:
    tensor_bytes = num_tokens * _HIDDEN_SIZE * 2
    weight_bytes = _HIDDEN_SIZE * 2
    transferred_bytes = 5 * tensor_bytes + weight_bytes
    return transferred_bytes / (latency_ms * 1e-3) / 1e9


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens", nargs="+", type=int, default=_DEFAULT_NUM_TOKENS
    )
    parser.add_argument(
        "--cuda-events",
        action="store_true",
        help="measure with CUDA events instead of CUPTI",
    )
    args = parser.parse_args()

    print("tokens  fused_us  composed_us  speedup  fused_GB/s")
    for num_tokens in args.num_tokens:
        shape = (num_tokens, _HIDDEN_SIZE)
        routed_output = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        shared_output = torch.randn_like(routed_output)
        residual = torch.randn_like(routed_output)
        weight = torch.randn(_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        hidden_states = torch.empty_like(routed_output)
        residual_out = torch.empty_like(routed_output)

        def fused() -> None:
            flashinfer.fused_moe_add_residual_rmsnorm(
                routed_output,
                shared_output,
                residual,
                weight,
                hidden_states=hidden_states,
                residual_out=residual_out,
            )

        def composed() -> None:
            torch.add(routed_output, shared_output, out=hidden_states)
            residual_out.copy_(residual)
            flashinfer.fused_add_rmsnorm(hidden_states, residual_out, weight)

        fused()
        composed()
        kwargs = {
            "cold_l2_cache": True,
            "enable_cupti": not args.cuda_events,
            "use_cuda_graph": False,
            "dry_run_iters": 10,
            "repeat_iters": 100,
        }
        fused_ms = float(np.median(bench_gpu_time(fused, **kwargs)))
        composed_ms = float(np.median(bench_gpu_time(composed, **kwargs)))
        print(
            f"{num_tokens:6d}  {fused_ms * 1e3:8.2f}  "
            f"{composed_ms * 1e3:11.2f}  {composed_ms / fused_ms:7.2f}x  "
            f"{_bandwidth_gb_s(num_tokens, fused_ms):10.1f}"
        )


if __name__ == "__main__":
    main()
