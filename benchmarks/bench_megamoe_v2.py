# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the v2 split MoE-EP pipeline at a serving-shaped problem.

Default shape: hidden 7168, 64 experts of intermediate 4096, top-k 4, DP4+EP4
on one node, 96 tokens per rank (32 sequences x 3 speculative tokens).  Every
one of those is a flag; the defaults just encode that deployment.

Launch::

    NVSHMEM_DISABLE_CUDA_VMM=1 NVSHMEM_REMOTE_TRANSPORT=none \\
    torchrun --nproc_per_node=4 benchmarks/bench_megamoe_v2.py

(Both variables are container properties, not kernel ones -- see
``tests/moe_ep/test_megamoe_v2_multirank.py``.)

Two timing modes, because they answer different questions
---------------------------------------------------------

``lockstep``
    Every rank is synchronized before each timed iteration, so the measurement
    is one layer with no inherited skew.  This is the headline number.
``steady``
    ``iters`` launches back to back, total / iters.  Closer to a serving loop,
    but a free-running harness around a kernel containing cross-rank barriers
    reports the *pipelined* cost and hides per-iteration skew -- it flatters
    the kernel, so it is reported second and never alone.

What the weights are
--------------------

Random valid NVFP4 bit patterns, not a quantization of a real matrix.  Nothing
in the kernel's runtime depends on the *values*, only on the layout, and
honestly quantizing 940M weight elements would cost more than the benchmark.
Correctness lives in ``tests/moe_ep/test_megamoe_v2_*``; this file measures.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import statistics

import torch

from flashinfer.moe_ep.kernel_src.megamoe_v2 import launcher, sf_layout
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (
    NVFP4_BLOCK,
    CommConfig,
    EpilogueConfig,
    EpTopology,
    KernelConfig,
    Phase,
    ProblemShape,
    TileConfig,
)

_SEED = 2026
# NVFP4 payload plus one E4M3 block scale per 16 elements.
_BYTES_PER_WEIGHT = 0.5 + 1.0 / NVFP4_BLOCK


# --------------------------------------------------------------------------
# problem setup
# --------------------------------------------------------------------------


def make_routing(
    mode: str,
    *,
    rank: int,
    world: int,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    spec_len: int,
    spec_agree: float,
    skew_alpha: float,
) -> torch.Tensor:
    """Per-rank ``(tokens, top_k)`` expert ids, reproducible from ``rank``.

    Reproducible on purpose: any rank can regenerate any other rank's routing,
    so the expected per-expert load can be computed locally without a gather.

    The mode matters more than it looks.  Load skew decides how many token
    tiles an expert spans, and a second token tile means that expert's weights
    are streamed a second time -- and weight traffic is what this kernel is
    bound by.
    """
    g = torch.Generator(device="cuda").manual_seed(_SEED + 977 * rank)

    def draw(n: int) -> torch.Tensor:
        if mode == "skewed":
            w = torch.arange(1, num_experts + 1, device="cuda", dtype=torch.float32)
            probs = (w**-skew_alpha).expand(n, num_experts).contiguous()
            return torch.multinomial(probs, top_k, replacement=False, generator=g)
        logits = torch.rand(n, num_experts, device="cuda", generator=g)
        return logits.topk(top_k, dim=-1).indices

    if mode == "balanced":
        # Exactly uniform: consecutive global tokens walk a stride of
        # num_experts // top_k, so every expert receives the same count.
        stride = num_experts // top_k
        gt = torch.arange(num_tokens, device="cuda") + rank * num_tokens
        base = (gt % stride).unsqueeze(1)
        slots = torch.arange(top_k, device="cuda").unsqueeze(0) * stride
        ids = base + slots
    elif mode == "spec":
        # Speculative decoding: the `spec_len` tokens of one sequence are near
        # duplicates and mostly route alike.  That correlation is the whole
        # reason to model it -- it concentrates load beyond an iid draw.
        if num_tokens % spec_len:
            raise ValueError(
                f"--tokens ({num_tokens}) must be a multiple of --spec-len "
                f"({spec_len}) for the 'spec' routing mode"
            )
        seqs = num_tokens // spec_len
        head = draw(seqs)
        rows = [head]
        for _ in range(spec_len - 1):
            fresh = draw(seqs)
            keep = torch.rand(seqs, top_k, device="cuda", generator=g) < spec_agree
            rows.append(torch.where(keep, head, fresh))
        # (spec_len, seqs, top_k) -> tokens of a sequence are contiguous.
        ids = torch.stack(rows, dim=1).reshape(num_tokens, top_k)
    else:
        ids = draw(num_tokens)
    return ids.to(torch.int32).contiguous()


def scatter_scales(scales: torch.Tensor, *, num_k_atoms: int) -> torch.Tensor:
    """Place ``(rows, blocks)`` block scales into the swizzled atom buffer.

    Vectorized restatement of :func:`sf_layout.word_offset`; the scalar version
    is the specification, this one is what can run over 130M scales.
    """
    rows, blocks = scales.shape
    r = torch.arange(rows, device=scales.device).unsqueeze(1)
    b = torch.arange(blocks, device=scales.device).unsqueeze(0)
    row_block, t = r // 128, r % 128
    k_atom, k_bank = b // 4, b % 4
    atom = row_block * num_k_atoms + k_atom
    idx = atom * 512 + (t % 32) * 16 + (t // 32) * 4 + k_bank
    flat = torch.zeros(
        sf_layout.buffer_words(rows, num_k_atoms=num_k_atoms) * 4,
        dtype=torch.float32,
        device=scales.device,
    )
    flat[idx.reshape(-1)] = scales.reshape(-1)
    return flat.to(torch.float8_e4m3fn)


def random_nvfp4(
    experts: int, out_channels: int, k: int, *, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random weights already in the packed NVFP4 + swizzled-scale layout.

    Every 8-bit pattern is a valid pair of E2M1 codes, so the payload can be
    raw random bytes.  Scales are drawn in ``[0.25, 1.25)`` rather than from
    random bytes: a random E4M3 byte can be NaN, and a NaN weight would turn
    the output check into a false alarm.
    """
    g = torch.Generator(device=device).manual_seed(_SEED)
    rows = experts * out_channels
    codes = torch.randint(
        0, 256, (rows, k // 2), dtype=torch.uint8, device=device, generator=g
    ).view(torch.float4_e2m1fn_x2)
    blocks = k // NVFP4_BLOCK
    scales = (
        torch.rand(rows, blocks, dtype=torch.float32, device=device, generator=g) + 0.25
    )
    num_k_atoms = sf_layout.num_k_atoms_for(k, NVFP4_BLOCK)
    return (
        codes.reshape(experts, out_channels, k // 2).view(torch.float4_e2m1fn_x2),
        scatter_scales(scales, num_k_atoms=num_k_atoms),
    )


# --------------------------------------------------------------------------
# derived cost model
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Roofline:
    """What one rank must move and compute, given the realized routing."""

    weight_bytes: int  # if every expert is streamed exactly once
    streamed_bytes: int  # what the tile walk actually streams
    token_bytes: int  # dispatch + combine over the fabric
    flops: int
    tokens_per_expert: tuple[int, ...]
    token_blocks: int

    @property
    def amplification(self) -> float:
        """Weight re-reads: 1.0 when every expert fits in one token tile."""
        return self.streamed_bytes / self.weight_bytes


def roofline(
    config: KernelConfig, all_ids: tuple[torch.Tensor, ...], *, rank: int
) -> Roofline:
    shape, tile = config.shape, config.tile
    le = config.experts_per_rank
    counts = tuple(
        int(sum(int((ids == rank * le + e).sum()) for ids in all_ids))
        for e in range(le)
    )
    blocks = tuple(
        max(1, math.ceil(c / tile.cluster_tile_tokens)) if c else 0 for c in counts
    )
    per_expert = 3 * shape.intermediate * shape.hidden
    weight_bytes = int(le * per_expert * _BYTES_PER_WEIGHT)
    streamed = int(sum(blocks) * per_expert * _BYTES_PER_WEIGHT)
    rows = sum(counts)
    # Dispatch pulls NVFP4 rows in; combine pushes bf16 rows back out.
    token_bytes = int(
        rows * shape.hidden * (_BYTES_PER_WEIGHT + 2) + 0  # in (fp4+sf) + out (bf16)
    )
    return Roofline(
        weight_bytes=weight_bytes,
        streamed_bytes=streamed,
        token_bytes=token_bytes,
        flops=2 * rows * per_expert,
        tokens_per_expert=counts,
        token_blocks=sum(blocks),
    )


def workspace_bytes(config: KernelConfig) -> tuple[int, int]:
    from flashinfer.moe_ep.kernel_src.megamoe_v2.layout import workspace_sizes

    return workspace_sizes(config)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def _sync(dist) -> None:
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()


@dataclasses.dataclass(frozen=True)
class Runner:
    """One iteration of a pipeline, instrumented at every step boundary.

    The events are recorded *inside* the iteration, which is what lets the
    measured region start after the counter reset instead of before it.  Under
    graph capture they become event-record nodes and are re-recorded on every
    replay, so the same accessors work for both paths.

    ``measured`` reads one event pair rather than summing per-step medians:
    medians do not add, and the segments overlap under PDL.
    """

    names: tuple[str, ...]
    events: tuple[torch.cuda.Event, ...]
    launch: object  # () -> None

    def segments(self) -> tuple[float, ...]:
        return tuple(
            self.events[i].elapsed_time(self.events[i + 1])
            for i in range(len(self.names))
        )

    def measured(self, skip: int) -> float:
        return self.events[skip].elapsed_time(self.events[-1])


def make_runner(steps: tuple, *, graph: bool, stream, dist) -> Runner:
    """Build a :class:`Runner` over ``(name, callable)`` steps.

    Capturing is only legal if the kernels launch onto the capturing stream,
    which is why the whole benchmark runs on one explicit stream that the
    pipelines were compiled against -- the compiled artifacts bind their stream
    at compile time.

    Each step becomes its *own* graph rather than the pipeline becoming one,
    for two reasons.  ``cudaEventElapsedTime`` rejects events recorded by graph
    event-record nodes, so timing events have to stay outside the graphs; and
    keeping one event pair per step boundary makes the graph and eager numbers
    measure the same segments.  What capture removes either way is the host
    cost of getting the work onto the stream, which for the counter reset is
    almost all of its apparent cost.
    """
    names = tuple(n for n, _ in steps)
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(len(steps) + 1))

    def sequence_of(fns) -> object:
        def run() -> None:
            events[0].record()
            for i, fn in enumerate(fns):
                fn()
                events[i + 1].record()

        return run

    eager = sequence_of(tuple(fn for _, fn in steps))
    if not graph:
        return Runner(names=names, events=events, launch=eager)

    # One real iteration first: capture itself executes nothing, so the steps
    # must have been warmed up in their true order beforehand or a lazy
    # allocation would land inside a capture.
    _sync(dist)
    eager()
    _sync(dist)
    replays = []
    for _, fn in steps:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            fn()
        replays.append(g.replay)
    _sync(dist)
    return Runner(names=names, events=events, launch=sequence_of(tuple(replays)))


def time_lockstep(
    runner: Runner, *, dist, warmup: int, iters: int, skip: int
) -> tuple[list[float], tuple[float, ...]]:
    """Per-iteration measured milliseconds, plus the median of each segment.

    Every rank is re-synchronized before each iteration, so no iteration
    inherits the previous one's skew.
    """
    for _ in range(warmup):
        runner.launch()
    _sync(dist)
    totals, segs = [], []
    for _ in range(iters):
        _sync(dist)
        runner.launch()
        torch.cuda.synchronize()
        totals.append(runner.measured(skip))
        segs.append(runner.segments())
    return totals, tuple(statistics.median(c) for c in zip(*segs, strict=True))


def time_steady(runner: Runner, *, dist, warmup: int, iters: int) -> float:
    """Whole iterations back to back, including the reset.

    No ``skip`` here on purpose: back-to-back replays cannot exclude a step
    that sits between them, and under graph capture the reset is a pair of
    memsets whose real cost this is the honest way to see.
    """
    for _ in range(warmup):
        runner.launch()
    _sync(dist)
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        runner.launch()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def across_ranks(dist, value: float, world: int) -> tuple[float, float]:
    """``(mean, max)`` of a per-rank scalar.

    The max is the one that matters: the next collective waits for the slowest
    rank, so a mean over ranks understates the layer's real latency.
    """
    t = torch.tensor([value], device="cuda")
    s = t.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    m = t.clone()
    dist.all_reduce(m, op=dist.ReduceOp.MAX)
    return float(s.item()) / world, float(m.item())


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def nvshmem_allocator(world: int, my_pe: int):
    def alloc(nbytes: int):
        import nvshmem.core
        from nvshmem.core.interop.torch import tensor_get_buffer

        t = nvshmem.core.tensor((nbytes,), dtype=torch.uint8)
        t.zero_()
        buf, _size, _dtype = tensor_get_buffer(t)
        bases = []
        for pe in range(world):
            if pe == my_pe:
                bases.append(int(t.data_ptr()))
            else:
                peer = nvshmem.core.get_peer_buffer(buf, pe)
                bases.append(int(torch.utils.dlpack.from_dlpack(peer).data_ptr()))
        return t, tuple(bases)

    return alloc


def run_staged(pipeline) -> None:
    """The nine-launch path, for the fused/staged comparison.

    ``launcher.run`` refuses ``world > 1`` because it cannot interleave ranks;
    here each rank is its own process, so issuing the stage order *is* the
    interleaving and the barrier stage rendezvouses on its own.
    """
    launcher.reset_counters(pipeline.ws, pipeline.config)
    for name in launcher.STAGE_ORDER:
        launcher.run_stage(pipeline, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=4096)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--tokens", type=int, default=96, help="per-rank tokens this run")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="per-rank capacity the workspace is sized for (default: --tokens)",
    )
    p.add_argument("--spec-len", type=int, default=3, help="tokens per sequence")
    p.add_argument(
        "--spec-agree",
        type=float,
        default=0.8,
        help="probability a speculative token keeps its sequence head's expert",
    )
    p.add_argument("--skew-alpha", type=float, default=1.0)
    p.add_argument(
        "--routing",
        choices=["spec", "random", "balanced", "skewed"],
        default="spec",
    )
    p.add_argument("--num-clusters", type=int, default=None)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--no-pdl", action="store_true")
    p.add_argument("--staged", action="store_true", help="also time the 9-launch path")
    p.add_argument(
        "--breakdown", action="store_true", help="per-launch attribution for both paths"
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="capture each pipeline as one graph and time replays",
    )
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    max_tokens = args.max_tokens or args.tokens
    if args.tokens > max_tokens:
        raise SystemExit(f"--tokens {args.tokens} exceeds --max-tokens {max_tokens}")

    torch.cuda.set_device(rank % torch.cuda.device_count())
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.core.runtime.bootstrap import (
        NVSHMEM,
        bootstrap_moe_ep_runtime,
    )

    bootstrap_moe_ep_runtime(
        BootstrapConfig(world_size=world, rank=rank), frozenset({NVSHMEM})
    )
    import torch.distributed as dist
    import cuda.bindings.driver as cuda

    props = torch.cuda.get_device_properties(rank % torch.cuda.device_count())
    tile = TileConfig(mma_m=256, mma_n=128, mma_k=256, cluster_m=2, two_cta=True)
    # The fused kernels run a device-wide barrier, so the grid must be
    # co-resident: one block per SM is the ceiling, not a tuning choice.
    num_clusters = args.num_clusters or props.multi_processor_count // tile.cluster_m

    config = KernelConfig(
        shape=ProblemShape(
            hidden=args.hidden,
            intermediate=args.intermediate,
            num_experts=args.experts,
            top_k=args.top_k,
            max_tokens_per_rank=max_tokens,
        ),
        topology=EpTopology(world_size=world, rank=rank),
        phase=Phase.FC1,
        tile=tile,
        comm=CommConfig(invalid_expert_id=-1),
        epilogue=EpilogueConfig(gate_up_clamp=None, apply_topk_in_fc1=True),
    )
    le = config.experts_per_rank

    routing_kw = dict(
        world=world,
        num_tokens=args.tokens,
        num_experts=args.experts,
        top_k=args.top_k,
        spec_len=args.spec_len,
        spec_agree=args.spec_agree,
        skew_alpha=args.skew_alpha,
    )
    all_ids = tuple(
        make_routing(args.routing, rank=r, **routing_kw) for r in range(world)
    )
    topk_ids = all_ids[rank]
    g = torch.Generator(device="cuda").manual_seed(_SEED + rank)
    act = torch.randn(
        args.tokens, args.hidden, dtype=torch.float32, device="cuda", generator=g
    ).bfloat16()
    topk_weights = torch.rand(
        args.tokens, args.top_k, dtype=torch.float32, device="cuda", generator=g
    )

    w1, w1_sf = random_nvfp4(le, 2 * args.intermediate, args.hidden, device="cuda")
    w2, w2_sf = random_nvfp4(le, args.hidden, args.intermediate, device="cuda")
    weights = launcher.Weights(w1=w1, w1_sf=w1_sf, w2=w2, w2_sf=w2_sf)

    ws = launcher.allocate_workspaces(
        config, rank=rank, alloc_shared=nvshmem_allocator(world, rank)
    )
    views = launcher.build_views(ws, config)
    out = torch.zeros(args.tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
    # Everything runs on one explicit non-default stream, graph or not.  Graph
    # capture requires it (the legacy default stream cannot be captured), and
    # the compiled kernels bind their stream at compile time, so the choice has
    # to be made before compilation rather than at capture.
    work_stream = torch.cuda.Stream()
    torch.cuda.set_stream(work_stream)
    stream = cuda.CUstream(work_stream.cuda_stream)

    fused = launcher.compile_fused(
        config,
        rank=rank,
        ws=ws,
        views=views,
        weights=weights,
        activation=act,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        out=out,
        stream=stream,
        num_clusters=num_clusters,
        use_pdl=not args.no_pdl,
    )
    staged = (
        launcher.compile_pipeline(
            config,
            rank=rank,
            ws=ws,
            views=views,
            weights=weights,
            activation=act,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            out=out,
            stream=stream,
            num_clusters=num_clusters,
            use_pdl=not args.no_pdl,
        )
        if args.staged
        else None
    )

    # A benchmark that silently measures a no-op is the failure mode worth
    # guarding against, so check the pipeline moved real data before timing.
    dist.barrier()
    launcher.run_fused(fused)
    torch.cuda.synchronize()
    model = roofline(config, all_ids, rank=rank)
    counts = views.expert_token_count.cpu().tolist()
    if [int(c) for c in counts] != list(model.tokens_per_expert):
        raise SystemExit(
            f"rank {rank}: device counts {counts} != routed {model.tokens_per_expert}"
        )
    if not torch.isfinite(out).all() or out.abs().sum() == 0:
        raise SystemExit(f"rank {rank}: output is degenerate")

    # Step 0 of every pipeline is the counter reset, and `skip=1` puts the
    # measured region after it: the reset is *required* for correctness but is
    # not the kernel's cost, and eagerly it is ~90 us of host-side Python
    # landing on an idle queue.
    paths = [
        (
            "fused",
            (
                ("reset", lambda: launcher.reset_counters(ws, config)),
                ("kernel A", lambda: fused.kernel_a(*fused.kernel_a_args)),
                ("kernel B", lambda: fused.kernel_b(*fused.kernel_b_args)),
            ),
        )
    ]
    if staged is not None:
        paths.append(
            (
                "staged",
                (("reset", lambda: launcher.reset_counters(ws, config)),)
                + tuple(
                    (name, (lambda n=name: launcher.run_stage(staged, n)))
                    for name in launcher.STAGE_ORDER
                ),
            )
        )

    results = []
    for label, steps in paths:
        runner = make_runner(
            steps, graph=args.cuda_graph, stream=work_stream, dist=dist
        )
        totals, segments = time_lockstep(
            runner, dist=dist, warmup=args.warmup, iters=args.iters, skip=1
        )
        steady = time_steady(runner, dist=dist, warmup=5, iters=args.iters)
        results.append(
            (
                label,
                across_ranks(dist, statistics.median(totals), world),
                across_ranks(dist, steady, world),
                runner.names,
                tuple(across_ranks(dist, v, world)[1] for v in segments),
            )
        )

    if rank != 0:
        dist.barrier()
        return

    local_b, shared_b = workspace_bytes(config)
    us = lambda ms: ms * 1e3
    bw = lambda nbytes, ms: nbytes / (ms * 1e-3) / 1e12
    counts_t = model.tokens_per_expert
    print(f"\n{'=' * 72}")
    print(
        f"megamoe v2  hidden={args.hidden} inter={args.intermediate} "
        f"E={args.experts} top_k={args.top_k}  EP{world}"
    )
    print(
        f"  tokens/rank {args.tokens} ({args.tokens // args.spec_len} seq x "
        f"{args.spec_len})   routing={args.routing}   {props.name} x{world}"
    )
    print(
        f"  grid {num_clusters * tile.cluster_m} blocks / "
        f"{props.multi_processor_count} SMs   PDL={'off' if args.no_pdl else 'on'}"
    )
    print(f"{'-' * 72}")
    print(
        f"  experts/rank {le}   tokens/expert min {min(counts_t)} "
        f"med {int(statistics.median(counts_t))} max {max(counts_t)}"
    )
    print(
        f"  token blocks {model.token_blocks} over {le} experts  "
        f"-> weight re-read x{model.amplification:.2f}"
    )
    print(
        f"  weights {model.weight_bytes / 1e6:.1f} MB  streamed "
        f"{model.streamed_bytes / 1e6:.1f} MB   fabric {model.token_bytes / 1e6:.2f} MB"
    )
    print(
        f"  workspace  local {local_b / 1e6:.1f} MB   symmetric {shared_b / 1e6:.1f} MB"
    )
    print(f"{'-' * 72}")
    print("  lockstep excludes the counter reset; steady is a whole iteration")
    print(f"  {'':24s} {'mean':>10s} {'max':>10s} {'eff BW':>12s}")
    for label, (lo_mean, lo_max), (st_mean, st_max), _names, _segs in results:
        print(
            f"  {label + ' (lockstep)':24s} {us(lo_mean):9.1f}u {us(lo_max):9.1f}u "
            f"{bw(model.streamed_bytes, lo_max):10.2f} TB/s"
        )
        print(
            f"  {label + ' (steady)':24s} {us(st_mean):9.1f}u {us(st_max):9.1f}u "
            f"{bw(model.streamed_bytes, st_max):10.2f} TB/s"
        )
    fused_max = results[0][1][1]
    if len(results) > 1:
        print(f"  fused speedup (lockstep max): {results[1][1][1] / fused_max:.2f}x")
    print(f"  {model.flops / (fused_max * 1e-3) / 1e12:.1f} TFLOP/s dense-equivalent")
    if args.breakdown:
        for label, _lo, _st, names, segs in results:
            print(f"{'-' * 72}")
            print(f"  {label} breakdown (max over ranks, on-stream segments)")
            for name, v in zip(names, segs, strict=True):
                print(f"    {name:16s} {us(v):9.1f}u  {100 * v / sum(segs):5.1f}%")
    print(f"{'=' * 72}\n")
    dist.barrier()


if __name__ == "__main__":
    main()
