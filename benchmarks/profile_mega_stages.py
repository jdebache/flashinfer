"""Per-stage attribution for the nvfp4_cutedsl mega path.

``bench_mega_moe.py`` reports two scopes (``--kernel-only`` and ``forward()``)
and the v2 plan derives the staging cost by subtracting them.  A subtraction
cannot distinguish GPU work from host-side launch overhead, so this script
measures both directly:

  * event-timed wall latency of each stage, barrier-aligned and MAX-reduced
    across ranks (same convention as ``bench_mega_moe._timed_block``);
  * summed **device** time per CUDA kernel, from the torch profiler, so the
    quantization kernel's real GPU cost is visible on its own.

Launch::

    torchrun --nproc_per_node=4 benchmarks/profile_mega_stages.py \\
        --num-experts 64 --hidden 7168 --intermediate 4096 --top-k 4 \\
        --tokens-per-rank 96
"""

from __future__ import annotations

import argparse
import os
from statistics import median

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mega-MoE per-stage attribution")
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=4096)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--tokens-per-rank", type=int, default=96)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--knobs", default="none")
    return p.parse_args()


def _timed_block(fn, *, repeat: int, device: torch.device) -> float:
    """Median MAX-across-ranks event latency of ``fn``, in microseconds."""
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
        samples.append(float(t.item()) * 1e3)
    return median(samples)


def _host_only(fn, *, repeat: int) -> float:
    """Median host-side enqueue time of ``fn`` (no sync inside), microseconds."""
    from time import perf_counter

    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = perf_counter()
        fn()
        samples.append((perf_counter() - t0) * 1e6)
    torch.cuda.synchronize()
    return median(samples)


def _device_kernel_times(fn, *, repeat: int) -> list[tuple[str, float, int]]:
    """(kernel name, total device us, launch count) summed over ``repeat`` runs."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    rows: list[tuple[str, float, int]] = []
    for evt in prof.key_averages():
        dev_us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if dev_us <= 0.0:
            continue
        rows.append((evt.key, dev_us / repeat, int(evt.count) // repeat))
    rows.sort(key=lambda r: -r[1])
    return rows


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
        MoEEpTensors,
        MoEWeightPack,
        Nvfp4CutedslMegaMoeConfig,
    )

    num_local_experts = args.num_experts // world_size
    n = args.tokens_per_rank

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
            max_tokens_per_rank=n,
            token_hidden_size=args.hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(
            megakernel=Nvfp4CutedslMegaMoeConfig(
                intermediate_size=args.intermediate,
                top_k=args.top_k,
                knobs=knobs,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    layer.warmup()
    dist.barrier()

    g = torch.Generator(device=device).manual_seed(42 + rank)
    hidden_states = torch.randn(
        n, args.hidden, dtype=torch.bfloat16, device=device, generator=g
    )
    scores = torch.randn(n, args.num_experts, device=device, generator=g)
    topk_weights, topk_ids = torch.topk(
        scores, args.top_k, dim=-1, largest=True, sorted=False
    )
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int64)
    t = MoEEpTensors(
        hidden_states=hidden_states, topk_ids=topk_ids, topk_weights=topk_weights
    )

    for _ in range(args.warmup):
        layer.forward(t)
    torch.cuda.synchronize()
    dist.barrier()

    kernel = layer._kernel  # pyright: ignore[reportPrivateUsage]
    workspace = layer._workspace  # pyright: ignore[reportPrivateUsage]
    transformed = layer._transformed  # pyright: ignore[reportPrivateUsage]
    y = torch.empty(n, args.hidden, dtype=torch.bfloat16, device=device)

    def _stage() -> None:
        kernel.stage_inputs(t, workspace, quantize_input=True)

    def _compute() -> None:
        kernel.compute(workspace, transformed, output=y)

    def _fwd() -> None:
        layer.forward(t)

    stage_us = _timed_block(_stage, repeat=args.repeat, device=device)
    compute_us = _timed_block(_compute, repeat=args.repeat, device=device)
    fwd_us = _timed_block(_fwd, repeat=args.repeat, device=device)

    stage_host = _host_only(_stage, repeat=args.repeat)
    compute_host = _host_only(_compute, repeat=args.repeat)
    fwd_host = _host_only(_fwd, repeat=args.repeat)

    stage_dev = _device_kernel_times(_stage, repeat=args.repeat)
    fwd_dev = _device_kernel_times(_fwd, repeat=args.repeat)

    # Graph-captured forward(): the production configuration.  Capture records
    # without executing, so there is no cross-rank dependency during capture;
    # replay is barrier-aligned like every other block here.
    graph_us = float("nan")
    graph_dev: list[tuple[str, float, int]] = []
    try:
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                layer.forward(t)
            capture_stream.synchronize()
            with torch.cuda.graph(graph):
                layer.forward(t)
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        graph_us = _timed_block(graph.replay, repeat=args.repeat, device=device)
        graph_dev = _device_kernel_times(graph.replay, repeat=args.repeat)
    except Exception as exc:  # capture is best-effort; report and continue
        if rank == 0:
            print(f"\n[graph capture failed: {type(exc).__name__}: {exc}]")

    if rank == 0:
        print("\n=== event-timed wall latency (MAX across ranks, median) ===")
        print(f"  stage_inputs()        {stage_us:9.1f} us")
        print(f"  compute()             {compute_us:9.1f} us")
        print(f"  forward()             {fwd_us:9.1f} us")
        print(f"  forward - compute     {fwd_us - compute_us:9.1f} us")

        print("\n=== host-side enqueue time (rank 0, no sync) ===")
        print(f"  stage_inputs()        {stage_host:9.1f} us")
        print(f"  compute()             {compute_host:9.1f} us")
        print(f"  forward()             {fwd_host:9.1f} us")

        print("\n=== device kernel time, stage_inputs() only (rank 0) ===")
        tot = 0.0
        for name, us, cnt in stage_dev:
            print(f"  {us:9.2f} us  x{cnt:<3d} {name[:78]}")
            tot += us
        print(f"  {tot:9.2f} us  TOTAL device")

        print("\n=== device kernel time, forward() (rank 0) ===")
        tot = 0.0
        for name, us, cnt in fwd_dev:
            print(f"  {us:9.2f} us  x{cnt:<3d} {name[:78]}")
            tot += us
        print(f"  {tot:9.2f} us  TOTAL device")

        print("\n=== CUDA-graph replay of forward() (production config) ===")
        print(f"  graph replay wall    {graph_us:9.1f} us")
        tot = 0.0
        for name, us, cnt in graph_dev:
            print(f"  {us:9.2f} us  x{cnt:<3d} {name[:78]}")
            tot += us
        print(f"  {tot:9.2f} us  TOTAL device")

    layer.destroy()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
