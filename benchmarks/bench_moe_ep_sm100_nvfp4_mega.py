"""SM100/SM103 NVFP4 CuTeDSL MegaMoE EP token-sweep benchmark.

The default geometry matches Mistral-Large-3-675B-Instruct-2512-NVFP4:
hidden 7168, expert intermediate 4096, 128 routed experts, and top-k 4.
Four torchrun ranks therefore hold 32 experts each.

Two prestaged-input series are captured as CUDA Graphs and measured with every
replay device-drained and barrier-aligned across ranks:

* ``kernel``: the prebuilt bare MegaMoE launch thunk.
* ``e2e``: the FlashInfer compute path, including output copy.

Both include fused dispatch, expert FC1/SwiGLU/FC2, and combine. Activation
quantization/staging and the model's shared expert are outside the timed region.

Example:

    torchrun --standalone --nproc_per_node=4 \
        benchmarks/bench_moe_ep_sm100_nvfp4_mega.py \
        --output /tmp/mistral_large_3_nvfp4_megamoe_ep4.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

_BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or os.getcwd()).resolve() != _BENCHMARK_DIR
]

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from flashinfer.moe_ep import (  # noqa: E402
    BootstrapConfig,
    FleetParams,
    MoEEpTensors,
    MoEWeightPack,
    Nvfp4CutedslMegaMoeConfig,
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
    preprocess_nvfp4_cutedsl_mega_weights,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel  # noqa: E402
from flashinfer.moe_ep.core.runtime import (  # noqa: E402
    nvfp4_cutedsl_runtime_requirements,
)

DEFAULT_TOKENS = tuple(2**power for power in range(2, 15))
CSV_COLUMNS = (
    "tokens_per_rank",
    "world_size",
    "total_experts",
    "local_experts",
    "top_k",
    "hidden",
    "intermediate",
    "combine_dtype",
    "in_kernel_fc2_reduce",
    "warmup",
    "iters",
    "timing_mode",
    "routing",
    "l2_flush_mib",
    "status",
    "kernel_critical_min_us",
    "kernel_critical_median_us",
    "kernel_critical_max_us",
    "e2e_critical_min_us",
    "e2e_critical_median_us",
    "e2e_critical_max_us",
    "per_gpu_tflops_kernel",
    "per_gpu_tflops_e2e",
    "aggregate_tflops_kernel",
    "aggregate_tflops_e2e",
    "aggregate_tokens_per_s_e2e",
    "error",
)


@dataclass(frozen=True)
class PointResult:
    status: str
    kernel_samples_by_rank: tuple[tuple[float, ...], ...] = ()
    e2e_samples_by_rank: tuple[tuple[float, ...], ...] = ()
    error: str = ""


@dataclass(frozen=True)
class CapturedCall:
    graph: torch.cuda.CUDAGraph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tokens",
        default=",".join(str(tokens) for tokens in DEFAULT_TOKENS),
        help="Comma-separated tokens per rank.",
    )
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument(
        "--combine-dtype", choices=("bf16", "mxfp8", "nvfp4"), default="bf16"
    )
    parser.add_argument("--in-kernel-fc2-reduce", action="store_true")
    parser.add_argument("--gate-up-clamp", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--l2-flush-mib",
        type=int,
        default=0,
        help="MiB to flush before each aligned sample; zero measures warm-cache replay.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, world_size: int) -> tuple[int, ...]:
    tokens = tuple(int(value) for value in args.tokens.split(",") if value)
    if not tokens or any(value <= 0 for value in tokens):
        raise ValueError("--tokens must contain positive integers")
    if world_size != 4:
        raise ValueError(f"this EP4 benchmark requires 4 ranks, got {world_size}")
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by the world size")
    if args.top_k <= 0 or args.top_k > args.num_experts:
        raise ValueError("--top-k must be in [1, num_experts]")
    if args.top_k != world_size:
        raise ValueError("per-token EP-balanced routing requires top-k == world size")
    if args.hidden % 64 != 0 or args.intermediate % 64 != 0:
        raise ValueError("--hidden and --intermediate must be multiples of 64")
    if args.warmup < 0 or args.iters <= 0 or args.l2_flush_mib < 0:
        raise ValueError("warmup/flush must be non-negative and iters positive")
    if args.in_kernel_fc2_reduce and args.combine_dtype != "bf16":
        raise ValueError("in-kernel FC2 reduce requires --combine-dtype bf16")
    return tokens


def _balanced_routing(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    rank: int,
    world_size: int,
    device: Any,
) -> Any:
    local_experts = num_experts // world_size
    token_ids = torch.arange(num_tokens, device=device, dtype=torch.int64)[:, None]
    route_ids = torch.arange(top_k, device=device, dtype=torch.int64)[None, :]
    target_ranks = (rank + route_ids) % world_size
    local_ids = (token_ids * top_k + route_ids + rank) % local_experts
    return target_ranks * local_experts + local_ids


def _make_inputs(
    args: argparse.Namespace,
    tokens: int,
    rank: int,
    world_size: int,
    device: Any,
) -> Any:
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        tokens,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    topk_ids = _balanced_routing(
        tokens, args.top_k, args.num_experts, rank, world_size, device
    )
    topk_weights = torch.softmax(
        torch.randn(
            tokens,
            args.top_k,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        dim=-1,
    )
    return hidden_states, topk_ids, topk_weights


def _make_transformed_weights(
    args: argparse.Namespace,
    local_experts: int,
    rank: int,
    device: Any,
) -> Any:
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + rank)
    w13 = torch.randint(
        0,
        256,
        (local_experts, 2 * args.intermediate, args.hidden // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (local_experts, args.hidden, args.intermediate // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w13_scale = torch.ones(
        local_experts,
        2 * args.intermediate,
        args.hidden // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w2_scale = torch.ones(
        local_experts,
        args.hidden,
        args.intermediate // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    packed = MoEWeightPack(
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )
    transformed = preprocess_nvfp4_cutedsl_mega_weights(
        packed,
        intermediate_size=args.intermediate,
        hidden_size=args.hidden,
        gate_up_clamp=args.gate_up_clamp,
    )
    del packed, w13, w2, w13_scale, w2_scale
    torch.cuda.empty_cache()
    return transformed


def _capture_call(call: Callable[[], Any]) -> CapturedCall:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return CapturedCall(graph=graph)


def _prepare_aligned_sample(l2_flush: Any | None) -> None:
    if l2_flush is not None:
        l2_flush.zero_()
    torch.cuda.synchronize()
    dist.barrier()


def _time_graph(
    captured: CapturedCall,
    *,
    warmup: int,
    iters: int,
    l2_flush: Any | None,
) -> tuple[float, ...]:
    for _ in range(warmup):
        _prepare_aligned_sample(l2_flush)
        captured.graph.replay()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: tuple[float, ...] = ()
    for _ in range(iters):
        _prepare_aligned_sample(l2_flush)
        start.record()
        captured.graph.replay()
        stop.record()
        torch.cuda.synchronize()
        samples += (start.elapsed_time(stop) * 1e3,)
    dist.barrier()
    return samples


def _run_point(
    args: argparse.Namespace,
    tokens: int,
    transformed: Any,
    bootstrap: Any,
    l2_flush: Any | None,
) -> PointResult:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    fleet = FleetParams(
        num_experts=args.num_experts,
        max_tokens_per_rank=tokens,
        token_hidden_size=args.hidden,
    )
    config = Nvfp4CutedslMegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        gate_up_clamp=args.gate_up_clamp,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        combine_dtype=args.combine_dtype,
    )
    backend = create_mega_kernel(config)
    workspace = None
    try:
        backend.bind_ep_bootstrap(bootstrap)
        backend.validate_init(bootstrap, fleet)
        backend.validate_transformed_weights(transformed, bootstrap, fleet)
        workspace = backend.prepare_workspace(bootstrap, fleet)
        hidden_states, topk_ids, topk_weights = _make_inputs(
            args, tokens, rank, world_size, device
        )
        tensors = MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        backend.validate_forward(tensors, fleet, quantize_input=True)
        backend.stage_inputs(tensors, workspace, quantize_input=True)
        output = torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device=device)

        backend.compute(workspace, transformed, output=output)
        torch.cuda.synchronize()
        if not torch.isfinite(output).all().item():
            raise RuntimeError("eager output contains non-finite values")
        eager_output = output.clone()
        torch.cuda.synchronize()
        dist.barrier()

        kernel_graph = _capture_call(
            lambda: backend.compute(workspace, transformed, output=None)
        )
        dist.barrier()
        e2e_graph = _capture_call(
            lambda: backend.compute(workspace, transformed, output=output)
        )
        dist.barrier()
        workspace.output_activation[:tokens].fill_(float("nan"))
        torch.cuda.synchronize()
        _prepare_aligned_sample(None)
        kernel_graph.graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(
            workspace.output_activation[:tokens],
            eager_output,
            rtol=5e-2,
            atol=5e-2,
            msg="bare-kernel CUDA Graph replay diverged from eager output",
        )

        output.fill_(float("nan"))
        torch.cuda.synchronize()
        _prepare_aligned_sample(None)
        e2e_graph.graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(
            output,
            eager_output,
            rtol=5e-2,
            atol=5e-2,
            msg="CUDA Graph replay diverged from eager output",
        )

        kernel_samples = _time_graph(
            kernel_graph,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        e2e_samples = _time_graph(
            e2e_graph,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        local_result = ("pass", kernel_samples, e2e_samples, "")
    except Exception as error:  # noqa: BLE001
        local_result = (
            "failed",
            (),
            (),
            f"{type(error).__name__}: {error}",
        )
    finally:
        if workspace is not None:
            with contextlib.suppress(Exception):
                backend.destroy(workspace)
        gc.collect()
        torch.cuda.empty_cache()

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    dist.barrier()
    if all(result[0] == "pass" for result in gathered):
        return PointResult(
            status="pass",
            kernel_samples_by_rank=tuple(result[1] for result in gathered),
            e2e_samples_by_rank=tuple(result[2] for result in gathered),
        )
    errors = "; ".join(
        f"rank{index}:{result[3]}" for index, result in enumerate(gathered) if result[3]
    )
    return PointResult(status="failed", error=errors)


def _critical_samples(
    samples_by_rank: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    return tuple(max(samples) for samples in zip(*samples_by_rank, strict=True))


def _flops_per_rank(tokens: int, top_k: int, hidden: int, intermediate: int) -> int:
    return 6 * tokens * top_k * hidden * intermediate


def _tflops(flops: int, latency_us: float) -> float:
    return flops / latency_us / 1e6


def _row(
    args: argparse.Namespace,
    tokens: int,
    world_size: int,
    result: PointResult,
) -> dict[str, object]:
    common: dict[str, object] = {
        "tokens_per_rank": tokens,
        "world_size": world_size,
        "total_experts": args.num_experts,
        "local_experts": args.num_experts // world_size,
        "top_k": args.top_k,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "combine_dtype": args.combine_dtype,
        "in_kernel_fc2_reduce": args.in_kernel_fc2_reduce,
        "warmup": args.warmup,
        "iters": args.iters,
        "timing_mode": "aligned_cuda_graph",
        "routing": "per_token_ep_balanced",
        "l2_flush_mib": args.l2_flush_mib,
        "status": result.status,
        "error": result.error,
    }
    if result.status != "pass":
        return common

    kernel = _critical_samples(result.kernel_samples_by_rank)
    e2e = _critical_samples(result.e2e_samples_by_rank)
    kernel_median = median(kernel)
    e2e_median = median(e2e)
    flops = _flops_per_rank(tokens, args.top_k, args.hidden, args.intermediate)
    per_gpu_kernel = _tflops(flops, kernel_median)
    per_gpu_e2e = _tflops(flops, e2e_median)
    common.update(
        {
            "kernel_critical_min_us": min(kernel),
            "kernel_critical_median_us": kernel_median,
            "kernel_critical_max_us": max(kernel),
            "e2e_critical_min_us": min(e2e),
            "e2e_critical_median_us": e2e_median,
            "e2e_critical_max_us": max(e2e),
            "per_gpu_tflops_kernel": per_gpu_kernel,
            "per_gpu_tflops_e2e": per_gpu_e2e,
            "aggregate_tflops_kernel": per_gpu_kernel * world_size,
            "aggregate_tflops_e2e": per_gpu_e2e * world_size,
            "aggregate_tokens_per_s_e2e": tokens * world_size * 1e6 / e2e_median,
        }
    )
    return common


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_row(writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow(
        {column: _format_value(row.get(column, "")) for column in CSV_COLUMNS}
    )


def main() -> int:
    args = _parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tokens_list = _validate_args(args, world_size)

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        auto_bootstrap=False,
        device=local_rank,
    )
    ensure_moe_ep_cuda_device(bootstrap)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, nvfp4_cutedsl_runtime_requirements(bootstrap)
    )

    output_file = None
    writer = None
    try:
        if rank == 0:
            print(",".join(CSV_COLUMNS), flush=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                output_file = args.output.open("w", newline="")
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=CSV_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()

        local_experts = args.num_experts // world_size
        transformed = _make_transformed_weights(
            args,
            local_experts,
            rank,
            torch.device("cuda", local_rank),
        )
        l2_flush = (
            torch.empty(
                args.l2_flush_mib * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            if args.l2_flush_mib
            else None
        )
        dist.barrier()

        for tokens in tokens_list:
            if rank == 0:
                print(f"# tokens_per_rank={tokens}", flush=True)
            result = _run_point(args, tokens, transformed, bootstrap, l2_flush)
            if rank == 0:
                row = _row(args, tokens, world_size, result)
                print(
                    ",".join(
                        _format_value(row.get(column, "")) for column in CSV_COLUMNS
                    ),
                    flush=True,
                )
                if writer is not None:
                    _write_row(writer, row)
                    output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()
        finalize_moe_ep_runtime(runtime)
        with contextlib.suppress(Exception):
            dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
