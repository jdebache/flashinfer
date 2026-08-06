# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Assembling the v2 pipeline: workspace views, compilation, launch order.

This is the only module that knows about torch, and the only one that knows
the *order* things run in.  Everything below it takes plain tensors and says
nothing about where they came from.

Launch order, and what orders it
--------------------------------

===============  ==========================================================
``prepare``      quantize local tokens, bucket pairs by destination expert
``push``         hand each destination its counts and pair list
``barrier``      the one true cross-rank wait
``plan``         sum the counts; build pool offsets and the tile prefix
``pull``         fetch the owed rows into the local pool
``fc1``          grouped GEMM + SwiGLU + requantize
``fc2+combine``  grouped GEMM, scattering rows back to their source rank
``barrier``      wait for every peer's scatter to land
``reduce``       sum each token's ``top_k`` slots
===============  ==========================================================

Every edge except the two barriers is ordered by the stream: a launch cannot
start before the previous one on the same stream finished.  The barriers cover
the two edges that cross ranks -- ``push -> plan`` (a rank must see *other*
ranks' counts) and ``combine -> reduce`` (a rank must see other ranks'
results).  Exactly two cross-rank waits is the payoff of the
push-metadata/pull-data split; nothing else in the pipeline needs to know that
other ranks exist.

Where allocation comes from
---------------------------

``alloc_shared`` is injected rather than hardwired to NVSHMEM, because the
symmetric heap is deployment state, not kernel logic.  Tests pass an allocator
that carves peer heaps out of one local tensor; production passes one backed by
``nvshmem.core.tensor``.  The kernels see only a base pointer and a table of
``peer_base - local_base`` offsets either way.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from . import combine as combine_mod
from . import kernel_a as kernel_a_mod
from . import kernel_b as kernel_b_mod
from . import dispatch as dispatch_mod
from . import fc1 as fc1_mod
from . import sf_layout
from .layout import local_layout, shared_layout
from .types import NVFP4_BLOCK, KernelConfig

_I32 = torch.int32
_I64 = torch.int64
_F32 = torch.float32
_U8 = torch.uint8
_BF16 = torch.bfloat16

# A shared heap allocator: bytes -> (tensor, peer base addresses).
SharedAllocator = Callable[[int], "tuple[torch.Tensor, tuple[int, ...]]"]


def region(
    workspace: torch.Tensor, wl, name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    """One named region of a workspace, typed and shaped.

    The layout's alignment guarantees every offset divides the element size, so
    the retype is always legal; an oversized ``shape`` raises here rather than
    silently running into the next region.
    """
    off = wl.offset_of(name)
    typed = workspace[off : off + wl.nbytes_of(name)].view(dtype)
    want = 1
    for s in shape:
        want *= s
    if want > typed.numel():
        raise ValueError(
            f"region {name!r} holds {typed.numel()} elements of {dtype}, but a "
            f"{shape} view needs {want}"
        )
    return typed[:want].view(*shape)


@dataclasses.dataclass(frozen=True)
class Workspaces:
    local: torch.Tensor
    shared: torch.Tensor
    peer_offset: torch.Tensor


def allocate_workspaces(
    config: KernelConfig,
    *,
    rank: int,
    alloc_shared: SharedAllocator,
    device: str = "cuda",
) -> Workspaces:
    """Allocate both workspaces and resolve the peer offset table."""
    ll, sl = local_layout(config), shared_layout(config)
    local = torch.zeros(ll.total_bytes, dtype=_U8, device=device)
    shared, peer_bases = alloc_shared(sl.total_bytes)
    if shared.numel() < sl.total_bytes:
        raise ValueError(
            f"shared allocation is {shared.numel()} B; layout needs {sl.total_bytes} B"
        )
    if len(peer_bases) != config.topology.world_size:
        raise ValueError(
            f"got {len(peer_bases)} peer bases for a world of "
            f"{config.topology.world_size}"
        )
    base = int(peer_bases[rank])
    return Workspaces(
        local=local,
        shared=shared,
        peer_offset=torch.tensor(
            [int(p) - base for p in peer_bases], dtype=_I64, device=device
        ),
    )


def reset_counters(ws: Workspaces, config: KernelConfig) -> None:
    """Zero the counter prefixes between launches.

    One fill each, because both layouts pack every resettable region into a
    contiguous prefix -- and deliberately leave the barrier phase and signal
    outside it, since a barrier that forgets its phase deadlocks the group.
    """
    ws.local[: local_layout(config).reset_prefix_bytes].zero_()
    ws.shared[: shared_layout(config).reset_prefix_bytes].zero_()


@dataclasses.dataclass(frozen=True)
class Views:
    """Every workspace region a launch touches, typed and shaped."""

    peer_expert_count: torch.Tensor
    expert_token_count: torch.Tensor
    src_token_slot: torch.Tensor
    src_topk_weight: torch.Tensor
    barrier_signal: torch.Tensor
    send_tokens_fp4: torch.Tensor
    send_tokens_i32: torch.Tensor
    send_sf_e4m3: torch.Tensor
    send_sf_i32: torch.Tensor
    combine_buf: torch.Tensor
    barrier_phase: torch.Tensor
    send_count: torch.Tensor
    send_slot: torch.Tensor
    send_weight: torch.Tensor
    rank_pool_offset: torch.Tensor
    token_block_prefix: torch.Tensor
    pool_tokens_fp4: torch.Tensor
    pool_tokens_i32: torch.Tensor
    pool_sf_e4m3: torch.Tensor
    pool_sf_i32: torch.Tensor
    pool_topk_weight: torch.Tensor
    pool_src: torch.Tensor
    fc1_out_fp4: torch.Tensor
    fc1_out_sf: torch.Tensor
    grid_sync: torch.Tensor
    grid_sync_b: torch.Tensor
    token_ready: torch.Tensor


def build_views(ws: Workspaces, config: KernelConfig) -> Views:
    """Resolve every region once.

    Two views of the token planes are deliberate: the quantizer and the GEMM
    want NVFP4/E4M3 element types, while dispatch moves whole rows and only
    cares that they are 4-byte words.  Same bytes, named by what reads them.
    """
    shape = config.shape
    world = config.topology.world_size
    le = config.experts_per_rank
    pairs = le * world * shape.max_tokens_per_rank * shape.top_k
    per_rank_pairs = shape.max_tokens_per_rank * shape.top_k
    pool_rows = config.pool_token_capacity
    ll, sl = local_layout(config), shared_layout(config)
    sh, lo = ws.shared, ws.local

    send_bytes = region(
        sh,
        sl,
        "send_tokens",
        _U8,
        (shape.max_tokens_per_rank, shape.hidden // 2),
    )
    send_sf_bytes = region(
        sh, sl, "send_token_sf", _U8, (sl.nbytes_of("send_token_sf"),)
    )
    pool_bytes = region(lo, ll, "pool_tokens", _U8, (pool_rows, shape.hidden // 2))
    pool_sf_bytes = region(
        lo, ll, "pool_token_sf", _U8, (ll.nbytes_of("pool_token_sf"),)
    )
    return Views(
        peer_expert_count=region(sh, sl, "peer_expert_count", _I64, (world * le,)),
        expert_token_count=region(sh, sl, "expert_token_count", _I64, (le,)),
        src_token_slot=region(sh, sl, "src_token_slot", _I32, (pairs,)),
        src_topk_weight=region(sh, sl, "src_topk_weight", _F32, (pairs,)),
        barrier_signal=region(sh, sl, "barrier_signal", _I64, (world,)),
        send_tokens_fp4=send_bytes.view(torch.float4_e2m1fn_x2),
        send_tokens_i32=send_bytes.view(_I32),
        send_sf_e4m3=send_sf_bytes.view(torch.float8_e4m3fn),
        send_sf_i32=send_sf_bytes.view(_I32),
        combine_buf=region(
            sh,
            sl,
            "combine_buf",
            _BF16,
            (shape.top_k * shape.max_tokens_per_rank, shape.hidden),
        ),
        barrier_phase=region(lo, ll, "barrier_phase", _I32, (1,)),
        send_count=region(lo, ll, "send_count", _I32, (shape.num_experts,)),
        send_slot=region(
            lo, ll, "send_slot", _I32, (shape.num_experts, per_rank_pairs)
        ),
        send_weight=region(
            lo, ll, "send_weight", _F32, (shape.num_experts, per_rank_pairs)
        ),
        rank_pool_offset=region(lo, ll, "rank_pool_offset", _I32, (le * world,)),
        token_block_prefix=region(lo, ll, "token_block_prefix", _I32, (le + 1,)),
        pool_tokens_fp4=pool_bytes.view(torch.float4_e2m1fn_x2),
        pool_tokens_i32=pool_bytes.view(_I32),
        pool_sf_e4m3=pool_sf_bytes.view(torch.float8_e4m3fn),
        pool_sf_i32=pool_sf_bytes.view(_I32),
        pool_topk_weight=region(lo, ll, "pool_topk_weight", _F32, (pool_rows,)),
        pool_src=region(lo, ll, "pool_src", _I64, (pool_rows,)),
        fc1_out_fp4=region(
            lo, ll, "fc1_out", _U8, (pool_rows, shape.intermediate // 2)
        ).view(torch.float4_e2m1fn_x2),
        fc1_out_sf=region(
            lo, ll, "fc1_out_sf", _U8, (ll.nbytes_of("fc1_out_sf"),)
        ).view(torch.float8_e4m3fn),
        grid_sync=region(lo, ll, "grid_sync", _I32, (2,)),
        grid_sync_b=region(lo, ll, "grid_sync_b", _I32, (2,)),
        # The readiness counters live in the first `local_experts` slots of the
        # token-tile readiness region.
        token_ready=region(lo, ll, "token_ready_count", _I32, (le,)),
    )


@dataclasses.dataclass(frozen=True)
class Weights:
    """Pre-quantized expert weights, in the layout the TMA descriptors expect."""

    w1: torch.Tensor  # (experts, 2*I, hidden) packed NVFP4
    w1_sf: torch.Tensor  # atom-swizzled E4M3
    w2: torch.Tensor  # (experts, hidden, I) packed NVFP4
    w2_sf: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """Compiled stages plus the buffers they were compiled against."""

    config: KernelConfig
    rank: int
    ws: Workspaces
    views: Views
    weights: Weights
    stages: dict[str, Any]
    num_tokens: int
    out: torch.Tensor


def compile_pipeline(
    config: KernelConfig,
    *,
    rank: int,
    ws: Workspaces,
    views: Views,
    weights: Weights,
    activation: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    stream,
    num_clusters: int = 132,
    use_pdl: bool = True,
) -> Pipeline:
    """Compile every stage against the buffers it will be launched with."""
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ct

    shape = config.shape
    world = config.topology.world_size
    le = config.experts_per_rank
    tile_tokens = config.tile.cluster_tile_tokens
    pool_rows = config.pool_token_capacity
    num_tokens = activation.shape[0]
    hidden_atoms = sf_layout.num_k_atoms_for(shape.hidden, NVFP4_BLOCK)
    inter_atoms = sf_layout.num_k_atoms_for(shape.intermediate, NVFP4_BLOCK)
    per_rank_pairs = shape.max_tokens_per_rank * shape.top_k

    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    v, rk = views, cutlass.Int32(rank)
    peer = mk(ws.peer_offset)

    def build(fn, args, **kw):
        return (cute.compile(fn, *args, **kw), args)

    stages = {}
    stages["prepare"] = build(
        dispatch_mod.dispatch_prepare,
        (
            mk(activation),
            mk(topk_ids),
            mk(topk_weights),
            mk(v.send_tokens_fp4),
            mk(v.send_sf_e4m3),
            mk(v.send_count),
            mk(v.send_slot),
            mk(v.send_weight),
            cutlass.Int32(num_tokens),
            cutlass.Float32(config.epilogue.input_norm_const),
            stream,
        ),
        hidden=shape.hidden,
        num_k_atoms=hidden_atoms,
        top_k=shape.top_k,
        num_experts=shape.num_experts,
    )
    stages["push"] = build(
        dispatch_mod.dispatch_push,
        (
            mk(v.send_count),
            mk(v.send_slot),
            mk(v.send_weight),
            mk(v.peer_expert_count),
            mk(v.src_token_slot),
            mk(v.src_topk_weight),
            peer,
            rk,
            stream,
        ),
        num_experts=shape.num_experts,
        local_experts=le,
        world=world,
        max_pairs=per_rank_pairs,
    )
    stages["barrier"] = build(
        dispatch_mod.dispatch_barrier,
        (mk(v.barrier_signal), mk(v.barrier_phase), peer, rk, stream),
        world=world,
    )
    stages["plan"] = build(
        dispatch_mod.dispatch_plan,
        (
            mk(v.peer_expert_count),
            mk(v.expert_token_count),
            mk(v.rank_pool_offset),
            mk(v.token_block_prefix),
            stream,
        ),
        local_experts=le,
        world=world,
        tile_tokens=tile_tokens,
    )
    stages["pull"] = build(
        dispatch_mod.dispatch_pull,
        (
            mk(v.send_tokens_i32),
            mk(v.send_sf_i32),
            mk(v.src_token_slot),
            mk(v.src_topk_weight),
            mk(v.peer_expert_count),
            peer,
            mk(v.expert_token_count),
            mk(v.rank_pool_offset),
            mk(v.token_block_prefix),
            mk(v.pool_tokens_i32),
            mk(v.pool_sf_i32),
            mk(v.pool_topk_weight),
            mk(v.pool_src),
            stream,
        ),
        local_experts=le,
        world=world,
        max_pairs=per_rank_pairs,
        max_tokens=shape.max_tokens_per_rank,
        top_k=shape.top_k,
        hidden=shape.hidden,
        num_k_atoms=hidden_atoms,
        tile_tokens=tile_tokens,
    )
    stages["fc1"] = build(
        fc1_mod.launch_fc1,
        (
            mk(weights.w1),
            mk(v.pool_tokens_fp4),
            mk(weights.w1_sf),
            mk(v.pool_sf_e4m3),
            mk(v.fc1_out_fp4),
            mk(v.fc1_out_sf),
            mk(v.pool_topk_weight),
            mk(v.token_block_prefix),
            stream,
        ),
        num_experts=le,
        intermediate=shape.intermediate,
        hidden=shape.hidden,
        pool_rows=pool_rows,
        num_k_atoms=inter_atoms,
        clamp=config.epilogue.gate_up_clamp,
        apply_weight=config.epilogue.apply_topk_in_fc1,
        num_clusters=num_clusters,
        use_pdl=use_pdl,
    )
    stages["fc2"] = build(
        combine_mod.launch_fc2_combine,
        (
            mk(weights.w2),
            mk(v.fc1_out_fp4),
            mk(weights.w2_sf),
            mk(v.fc1_out_sf),
            mk(v.combine_buf),
            mk(v.pool_src),
            peer,
            mk(v.token_block_prefix),
            stream,
        ),
        num_experts=le,
        intermediate=shape.intermediate,
        hidden=shape.hidden,
        pool_rows=pool_rows,
        max_tokens=shape.max_tokens_per_rank,
        top_k=shape.top_k,
        num_clusters=num_clusters,
        use_pdl=use_pdl,
    )
    stages["reduce"] = build(
        combine_mod.combine_reduce,
        (mk(v.combine_buf), mk(topk_ids), mk(out), cutlass.Int32(num_tokens), stream),
        max_tokens=shape.max_tokens_per_rank,
        top_k=shape.top_k,
        hidden=shape.hidden,
        num_experts=shape.num_experts,
    )
    return Pipeline(
        config=config,
        rank=rank,
        ws=ws,
        views=views,
        weights=weights,
        stages=stages,
        num_tokens=num_tokens,
        out=out,
    )


#: The stages a rank runs, in order.  ``None`` marks a cross-rank barrier,
#: which every rank must reach before any rank continues.
STAGE_ORDER = (
    "prepare",
    "push",
    "barrier",
    "plan",
    "pull",
    "fc1",
    "fc2",
    "barrier",
    "reduce",
)


def run_stage(pipeline: Pipeline, name: str) -> None:
    """Launch one stage onto its stream."""
    compiled, args = pipeline.stages[name]
    compiled(*args)


def run(pipeline: Pipeline) -> None:
    """Run every stage of one rank, in order.

    Correct as written only for ``world_size == 1``: with more ranks the
    barrier stage must be reached by all of them concurrently, so the caller
    drives :data:`STAGE_ORDER` across ranks rather than calling this.
    """
    if pipeline.config.topology.world_size != 1:
        raise ValueError(
            "run() serializes one rank; a multi-rank group must interleave "
            "STAGE_ORDER across ranks so the barriers can rendezvous"
        )
    reset_counters(pipeline.ws, pipeline.config)
    for name in STAGE_ORDER:
        run_stage(pipeline, name)


# --------------------------------------------------------------------------
# fused: the two-kernel form
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FusedPipeline:
    """The two compiled kernels plus the buffers they were compiled against."""

    config: KernelConfig
    rank: int
    ws: Workspaces
    views: Views
    out: torch.Tensor
    kernel_a: Any
    kernel_a_args: tuple
    kernel_b: Any
    kernel_b_args: tuple


def compile_fused(
    config: KernelConfig,
    *,
    rank: int,
    ws: Workspaces,
    views: Views,
    weights: Weights,
    activation: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    stream,
    num_clusters: int = 8,
    use_pdl: bool = True,
) -> FusedPipeline:
    """Compile the whole pipeline as two launches.

    ``num_clusters`` is load-bearing here in a way it is not for the staged
    path: both kernels run a device-wide barrier, which is only sound if every
    block is resident at once.  At this smem footprint that means one block per
    SM, so the grid must not exceed the SM count -- checked below rather than
    left to chance, because the failure mode is a hang.
    """
    import cutlass.cute as cute
    import cutlass.torch as ct

    props = torch.cuda.get_device_properties(activation.device)
    blocks = config.tile.cluster_m * num_clusters
    if blocks > props.multi_processor_count:
        raise ValueError(
            f"grid of {blocks} blocks exceeds {props.multi_processor_count} SMs; "
            "the in-kernel grid barrier requires a co-resident grid"
        )

    shape = config.shape
    world = config.topology.world_size
    le = config.experts_per_rank
    v = views
    mk = lambda t: ct.from_dlpack(t, assumed_align=16)
    dev = activation.device
    scalar = lambda x: mk(torch.tensor([x], dtype=_I32, device=dev))

    coop_a = (
        mk(activation),
        mk(topk_ids),
        mk(topk_weights),
        mk(v.send_tokens_fp4),
        mk(v.send_sf_e4m3),
        mk(v.send_count),
        mk(v.send_slot),
        mk(v.send_weight),
        mk(v.peer_expert_count),
        mk(v.src_token_slot),
        mk(v.src_topk_weight),
        mk(v.expert_token_count),
        mk(v.rank_pool_offset),
        mk(v.token_block_prefix),
        mk(v.send_tokens_i32),
        mk(v.send_sf_i32),
        mk(v.pool_tokens_i32),
        mk(v.pool_sf_i32),
        mk(v.pool_topk_weight),
        mk(v.pool_src),
        mk(ws.peer_offset),
        mk(v.grid_sync),
        mk(v.token_ready),
        scalar(activation.shape[0]),
        scalar(rank),
    )
    args_a = (
        mk(weights.w1),
        mk(weights.w1_sf),
        mk(v.fc1_out_fp4),
        mk(v.fc1_out_sf),
        coop_a,
        stream,
    )
    ka = cute.compile(
        kernel_a_mod.launch_kernel_a,
        *args_a,
        num_experts=shape.num_experts,
        local_experts=le,
        world=world,
        intermediate=shape.intermediate,
        hidden=shape.hidden,
        pool_rows=config.pool_token_capacity,
        max_tokens=shape.max_tokens_per_rank,
        top_k=shape.top_k,
        hidden_atoms=sf_layout.num_k_atoms_for(shape.hidden, NVFP4_BLOCK),
        inter_atoms=sf_layout.num_k_atoms_for(shape.intermediate, NVFP4_BLOCK),
        norm_const=config.epilogue.input_norm_const,
        clamp=config.epilogue.gate_up_clamp,
        apply_weight=config.epilogue.apply_topk_in_fc1,
        mma_m=config.tile.mma_m,
        mma_n=config.tile.mma_n,
        cluster_m=config.tile.cluster_m,
        two_cta=config.tile.two_cta,
        num_clusters=num_clusters,
        use_pdl=use_pdl,
    )

    coop_b = (
        mk(v.combine_buf),
        mk(v.pool_src),
        mk(ws.peer_offset),
        mk(topk_ids),
        mk(out),
        mk(v.grid_sync_b),
        mk(v.barrier_signal),
        mk(v.barrier_phase),
        scalar(activation.shape[0]),
        scalar(rank),
    )
    args_b = (
        mk(weights.w2),
        mk(weights.w2_sf),
        mk(v.fc1_out_fp4),
        mk(v.fc1_out_sf),
        mk(v.token_block_prefix),
        coop_b,
        stream,
    )
    kb = cute.compile(
        kernel_b_mod.launch_kernel_b,
        *args_b,
        local_experts=le,
        num_experts=shape.num_experts,
        world=world,
        intermediate=shape.intermediate,
        hidden=shape.hidden,
        pool_rows=config.pool_token_capacity,
        max_tokens=shape.max_tokens_per_rank,
        top_k=shape.top_k,
        mma_m=config.tile.mma_m,
        mma_n=config.tile.mma_n,
        cluster_m=config.tile.cluster_m,
        two_cta=config.tile.two_cta,
        num_clusters=num_clusters,
        use_pdl=use_pdl,
    )
    return FusedPipeline(
        config=config,
        rank=rank,
        ws=ws,
        views=views,
        out=out,
        kernel_a=ka,
        kernel_a_args=args_a,
        kernel_b=kb,
        kernel_b_args=args_b,
    )


def run_fused(pipeline: FusedPipeline) -> None:
    """One MoE layer: two launches, and the counter reset that precedes them."""
    reset_counters(pipeline.ws, pipeline.config)
    pipeline.kernel_a(*pipeline.kernel_a_args)
    pipeline.kernel_b(*pipeline.kernel_b_args)
