"""Decompose the mega kernel's device time at a fixed weight-streaming load.

Weight streaming is batch-independent: with 128-token FC1 tiles and fewer than
128 tokens per expert, every expert has exactly one token tile regardless of the
token count, so the number of FC1/FC2 weight tiles -- and therefore the bytes
streamed from HBM -- is identical across the sweep below.  Anything that *does*
change with the token count is comm / dispatch work, not weight streaming.

That gives two readings without any new in-kernel instrumentation:

  * ``compute()`` device time vs tokens-per-rank -> the slope is comm cost, the
    intercept is (weight stream + fixed barrier/launch overhead);
  * the same shape at world_size 1 (``MEGA_NO_DIST=1``) -> cross-rank barrier
    and NVLink cost vanish, leaving the local GEMM pipeline.

Launch::

    torchrun --nproc_per_node=4 benchmarks/profile_mega_breakdown.py \\
        --num-experts 64 --hidden 7168 --intermediate 4096 --top-k 4 \\
        --tokens-per-rank 8 32 96 128
"""

from __future__ import annotations

import argparse
import os
from statistics import median

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mega kernel device-time breakdown")
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=4096)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--tokens-per-rank", type=int, nargs="+", default=[8, 32, 96, 128])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)
    return p.parse_args()


def _mega_device_us(fn, *, repeat: int) -> tuple[float, float]:
    """(mega kernel device us, all-kernel device us) per call, from the profiler."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    mega = 0.0
    total = 0.0
    for evt in prof.key_averages():
        dev = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if dev <= 0.0:
            continue
        total += dev
        if "fc1fc2_kernel_impl" in evt.key:
            mega += dev
    return mega / repeat, total / repeat


def _wall_us(fn, *, repeat: int, device: torch.device, distributed: bool) -> float:
    samples: list[float] = []
    for _ in range(repeat):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        t = torch.tensor([ev0.elapsed_time(ev1)], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(float(t.item()) * 1e3)
    return median(samples)


def main() -> int:
    args = _parse_args()
    distributed = os.environ.get("MEGA_NO_DIST", "0") != "1"

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Nvfp4CutedslMegaMoeConfig,
    )

    num_local_experts = args.num_experts // world_size
    max_tokens = max(args.tokens_per_rank)

    g = torch.Generator(device=device).manual_seed(13 + rank)
    w13 = torch.randn(
        num_local_experts, 2 * args.intermediate, args.hidden,
        dtype=torch.bfloat16, device=device, generator=g,
    )
    w2 = torch.randn(
        num_local_experts, args.hidden, args.intermediate,
        dtype=torch.bfloat16, device=device, generator=g,
    )

    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=args.num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=args.hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(
            megakernel=Nvfp4CutedslMegaMoeConfig(
                intermediate_size=args.intermediate,
                top_k=args.top_k,
                knobs=None,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    layer.warmup()
    if distributed:
        dist.barrier()

    if rank == 0:
        print(
            f"\nworld_size={world_size} local_experts={num_local_experts} "
            f"hidden={args.hidden} intermediate={args.intermediate} "
            f"top_k={args.top_k}"
        )
        print(
            "\n  tokens  pairs/rank   compute_wall   mega_device   all_device"
        )

    for n in sorted(args.tokens_per_rank):
        g = torch.Generator(device=device).manual_seed(42 + rank)
        hs = torch.randn(
            n, args.hidden, dtype=torch.bfloat16, device=device, generator=g
        )
        scores = torch.randn(n, args.num_experts, device=device, generator=g)
        tw, ti = torch.topk(scores, args.top_k, dim=-1, largest=True, sorted=False)
        tw = torch.softmax(tw, dim=-1).to(torch.float32)
        t = MoEEpTensors(
            hidden_states=hs, topk_ids=ti.to(torch.int64), topk_weights=tw
        )

        for _ in range(args.warmup):
            layer.forward(t)
        torch.cuda.synchronize()
        if distributed:
            dist.barrier()

        kernel = layer._kernel  # pyright: ignore[reportPrivateUsage]
        workspace = layer._workspace  # pyright: ignore[reportPrivateUsage]
        transformed = layer._transformed  # pyright: ignore[reportPrivateUsage]
        y = torch.empty(n, args.hidden, dtype=torch.bfloat16, device=device)

        def _compute() -> None:
            kernel.compute(workspace, transformed, output=y)

        wall = _wall_us(
            _compute, repeat=args.repeat, device=device, distributed=distributed
        )
        mega_dev, all_dev = _mega_device_us(_compute, repeat=args.repeat)

        if distributed:
            dist.barrier()
        if rank == 0:
            pairs = n * world_size * args.top_k // world_size
            print(
                f"  {n:6d}  {pairs:10d}   {wall:12.1f}   {mega_dev:11.1f}   "
                f"{all_dev:10.1f}"
            )

    layer.destroy()
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
