"""Multi-rank benchmark for MoEEpMegaLayer (nvfp4_cutedsl backend).

Measures wall-clock e2e latency and throughput of the fused dispatch+compute+combine
mega kernel across a token-count sweep.

Launch:
    torchrun --nproc_per_node=4 benchmarks/bench_mega_moe.py [options]

Example (DeepSeek-V3 geometry, 4-GPU EP):
    torchrun --nproc_per_node=4 benchmarks/bench_mega_moe.py \\
        --num-experts 256 --hidden 7168 --intermediate 2048 --top-k 8 \\
        --tokens-per-rank 128 512 2048 4096 \\
        --warmup 3 --repeat 20

Output: one BENCH_CSV line per token count on rank 0.
"""

from __future__ import annotations

import argparse
import os
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nvfp4_cutedsl mega-MoE benchmark")
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--tokens-per-rank",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 4096],
        metavar="T",
    )
    p.add_argument("--gate-up-clamp", type=float, default=None)
    p.add_argument(
        "--combine-dtype",
        choices=["bf16", "mxfp8", "nvfp4"],
        default="bf16",
    )
    p.add_argument("--in-kernel-fc2-reduce", action="store_true")
    p.add_argument("--fast-math", action="store_true", default=True)
    p.add_argument("--no-fast-math", dest="fast_math", action="store_false")
    p.add_argument(
        "--knobs",
        default="auto",
        help="kernel tuning knobs dict, 'auto' for online autotune, or 'none' for default heuristic",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument(
        "--kernel-only",
        action="store_true",
        help="time only the kernel launch (no staging), matching the CUDA-graph capture region",
    )
    return p.parse_args()


def _timed_block(fn, *, repeat: int, device: torch.device) -> list[float]:
    """Collective-correct block timing, in microseconds.

    Every rank is barrier-aligned at the block boundary; CUDA events bracket the
    block (on-device timestamps, so host launch overhead is excluded); the result
    is MAX-reduced because the slowest rank is the real latency of a collective.
    Same convention as the mega autotune sweep in
    ``moe_ep/kernel_src/cutedsl_megamoe/shim/autotune.py``.
    """
    samples: list[float] = []
    for _ in range(repeat):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        torch.cuda.synchronize()
        ev_start.record()
        fn()
        ev_end.record()
        torch.cuda.synchronize()
        t = torch.tensor(
            [ev_start.elapsed_time(ev_end)], dtype=torch.float64, device=device
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(float(t.item()) * 1e3)  # ms -> us
    return samples


def _make_weights(
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(13 + rank)
    w13 = torch.randn(
        num_local_experts, 2 * intermediate, hidden, dtype=torch.bfloat16,
        device=device, generator=g,
    )
    w2 = torch.randn(
        num_local_experts, hidden, intermediate, dtype=torch.bfloat16,
        device=device, generator=g,
    )
    return w13, w2


def _make_inputs(
    num_tokens: int,
    *,
    hidden: int,
    num_experts: int,
    top_k: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(42 + rank)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device=device, generator=g
    )
    scores = torch.randn(num_tokens, num_experts, device=device, generator=g)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1, largest=True, sorted=False)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int64)
    return hidden_states, topk_weights, topk_ids


def main() -> int:
    args = _parse_args()

    knobs: dict | str | None = None if args.knobs.lower() == "none" else args.knobs

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        Nvfp4CutedslMegaMoeConfig,
    )

    assert args.num_experts % world_size == 0, (
        f"--num-experts ({args.num_experts}) must be divisible by world_size ({world_size})"
    )
    num_local_experts = args.num_experts // world_size
    max_tokens_per_rank = max(args.tokens_per_rank)

    if rank == 0:
        print(f"\nnvfp4_cutedsl MegaMoE Benchmark")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"World size: {world_size} | Local experts: {num_local_experts}/{args.num_experts}")
        print(f"hidden={args.hidden}, intermediate={args.intermediate}, top_k={args.top_k}")
        print(f"combine_dtype={args.combine_dtype}, fast_math={args.fast_math}")
        mode = "kernel-only" if args.kernel_only else "e2e (staging+kernel)"
        print(f"knobs={knobs!r}, warmup={args.warmup}, repeat={args.repeat}, mode={mode}\n")

    w13, w2 = _make_weights(
        num_local_experts=num_local_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        rank=rank,
        device=device,
    )

    kernel_cfg = Nvfp4CutedslMegaMoeConfig(
        intermediate_size=args.intermediate,
        top_k=args.top_k,
        gate_up_clamp=args.gate_up_clamp,
        fast_math=args.fast_math,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        combine_dtype=args.combine_dtype,
        knobs=knobs,
    )

    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=args.num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=args.hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(
            megakernel=kernel_cfg,
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    assert isinstance(layer, MoEEpMegaLayer), f"expected MoEEpMegaLayer, got {type(layer)}"

    # Warmup on all ranks (triggers weight quantization, cute.compile, autotune)
    if rank == 0:
        print("Warming up (compile + autotune)...")
    layer.warmup()
    dist.barrier()
    if rank == 0:
        print("Warmup done.\n")

    if rank == 0:
        header = (
            "BENCH_CSV,tokens_per_rank,total_tokens,gpus,latency_us,tok_per_s,"
            "hidden,intermediate,num_experts,top_k,combine_dtype,mode"
        )
        print(header)

    for num_tokens in sorted(args.tokens_per_rank):
        hidden_states, topk_weights, topk_ids = _make_inputs(
            num_tokens,
            hidden=args.hidden,
            num_experts=args.num_experts,
            top_k=args.top_k,
            rank=rank,
            device=device,
        )
        t = MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

        # Warmup at this token count (always full forward to stage inputs)
        for _ in range(args.warmup):
            layer.forward(t)
        torch.cuda.synchronize()
        dist.barrier()

        if args.kernel_only:
            # After warmup the symmetric workspace has staged inputs; pre-allocate
            # the output and time only _kernel.compute() — the region captured
            # in a CUDA graph in production.
            y = torch.empty(num_tokens, args.hidden, dtype=torch.bfloat16, device=device)

            def _block() -> None:
                layer._kernel.compute(  # pyright: ignore[reportPrivateUsage]
                    layer._workspace,  # pyright: ignore[reportPrivateUsage]
                    layer._transformed,  # pyright: ignore[reportPrivateUsage]
                    output=y,
                )
        else:

            def _block() -> None:
                layer.forward(t)

        samples = _timed_block(_block, repeat=args.repeat, device=device)

        dist.barrier()

        if rank == 0:
            latency_us = median(samples)
            total_tokens = num_tokens * world_size
            tok_per_s = total_tokens / (latency_us * 1e-6)
            mode_label = "kernel" if args.kernel_only else "e2e"
            print(
                f"BENCH_CSV,"
                f"{num_tokens},{total_tokens},{world_size},"
                f"{latency_us:.1f},{tok_per_s:.0f},"
                f"{args.hidden},{args.intermediate},{args.num_experts},{args.top_k},"
                f"{args.combine_dtype},{mode_label}"
            )

    layer.destroy()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
