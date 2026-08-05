# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Swap-AB block-scaled NVFP4 GEMM mainloop on tcgen05.

Both v2 phases are the same GEMM with different operands, so the mainloop is
written once here and parameterized by :class:`~.types.Phase`:

===========  ==================  ==================  ==========
             A (GEMM-M)          B (GEMM-N)          K
===========  ==================  ==================  ==========
FC1          w13 (2*I, H)        tokens (T, H)       hidden
FC2          w2  (H, I)          fc1_out (T, I)      intermediate
===========  ==================  ==================  ==========

Why swap-AB
-----------

The natural orientation would put tokens on GEMM-M.  Putting *weights* there
instead buys the property the whole v2 split depends on: the A-side TMA
descriptor becomes tile-invariant and its tile space
(``expert x channel_block``) is known from launch parameters alone -- it does
not depend on how many tokens arrived.  Weights can therefore stream from cycle
0, before any count exchange has completed.  The token side keeps its
count-dependent addressing, but that is the side we are happy to have wait.

Dataflow, one CTA
-----------------

Five specialized warp roles cooperate through three pipelines::

    warp 5  TMA-A ──[a_pipeline]──┐
                                  ├──> warp 4  MMA ──[acc_pipeline]──> warps 0-3 epilogue
    warp 6  TMA-B ──[b_pipeline]──┘
    warp 7  scheduler ── work tiles ──> all of the above

**A and B get separate pipelines on purpose.**  Sharing one pool of smem
stages couples weight run-ahead to token arrival: a stage frees only when the
MMA retires it, and the MMA needs both halves, so the weight producer stalls
behind the slowest token.  With independent pipelines the weight producer runs
ahead by up to ``num_a_stages`` k-tiles regardless of what the token side is
doing, which is the point of the exercise (and matches what the upstream
FlashInfer block-scaled grouped GEMM does).

Scale factors take a different route from the data.  A and B tiles are read by
the MMA straight out of smem, but the per-16-element block scales must be in
**TMEM**: they are TMA'd to smem, then copied smem->TMEM ("S2T") by the MMA
warp before the UMMA that consumes them.  TMEM therefore holds accumulator
columns *plus* SFA columns *plus* SFB columns.
"""

from __future__ import annotations

import dataclasses

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Int32

from .types import NVFP4_BLOCK, Phase, ProblemShape, TileConfig, ceil_div

# Warp roles.  Kernel A and kernel B add comm warps *after* these, so the GEMM
# core always occupies the same eight warps and the barrier ids below stay
# valid in both.
EPILOGUE_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_A_WARP = 5
TMA_B_WARP = 6
SCHED_WARP = 7
GEMM_WARPS = 8

# Named-barrier ids reserved by the GEMM core; comm code must start above this.
TMEM_ALLOC_BARRIER = 1
TMEM_DEALLOC_BARRIER = 2
FIRST_FREE_BARRIER = 3

# Register budget split: the epilogue does the numerics (SwiGLU, requantize)
# and needs the registers; producer/scheduler warps are address arithmetic.
EPILOGUE_REGS = 232
PRODUCER_REGS = 40

# The UMMA instruction's own K extent is fixed by the operand dtype -- 64 for
# NVFP4, confirmed by querying ``tiled_mma.shape_mnk[2]`` rather than assumed.
# The tile K is a small multiple of it; 4 is the standard choice, amortizing
# per-instruction overhead while keeping one k-tile's smem footprint modest.
# 64 * 4 = 256, which is why the tuned configurations all carry mma_k=256.
_MMA_INST_K = 64
_MMA_INST_TILE_K = 4
MMA_TILE_K = _MMA_INST_K * _MMA_INST_TILE_K


@dataclasses.dataclass(frozen=True)
class GemmPlan:
    """Codegen-time constants for one phase's mainloop.

    Everything here is a Python int computed on the host, so the whole plan
    folds into immediates and can be unit-tested without a GPU.
    """

    phase: Phase
    mma_m: int
    mma_n: int
    mma_k: int
    cluster_m: int
    two_cta: bool
    out_channels: int
    k_extent: int
    num_a_stages: int
    num_b_stages: int
    num_acc_stages: int

    @property
    def cta_tile_m(self) -> int:
        return self.mma_m // (2 if self.two_cta else 1)

    @property
    def k_tiles(self) -> int:
        return ceil_div(self.k_extent, self.mma_k)

    @property
    def channel_blocks(self) -> int:
        return ceil_div(self.out_channels, self.mma_m)

    @property
    def mma_tiler(self) -> tuple[int, int, int]:
        return (self.mma_m, self.mma_n, self.mma_k)

    @property
    def cta_group(self):
        return tcgen05.CtaGroup.TWO if self.two_cta else tcgen05.CtaGroup.ONE


def plan_for(
    shape: ProblemShape,
    tile: TileConfig,
    phase: Phase,
    *,
    num_a_stages: int = 4,
    num_b_stages: int = 3,
    num_acc_stages: int = 2,
) -> GemmPlan:
    """Build the plan for one phase.

    The default stage split is biased toward A (weights): weights are the
    entire bandwidth cost at the shapes we care about, while the token side is
    tiny and latency-bound, so depth is worth more on A than on B.
    """
    if num_a_stages < 2 or num_b_stages < 2:
        raise ValueError(
            f"pipelines need >= 2 stages to overlap; got a={num_a_stages}, "
            f"b={num_b_stages}"
        )
    if tile.mma_k != MMA_TILE_K:
        raise ValueError(
            f"TileConfig.mma_k ({tile.mma_k}) must be {MMA_TILE_K} for NVFP4 "
            f"(instruction K {_MMA_INST_K} x {_MMA_INST_TILE_K})"
        )
    return GemmPlan(
        phase=phase,
        mma_m=tile.mma_m,
        mma_n=tile.mma_n,
        mma_k=MMA_TILE_K,
        cluster_m=tile.cluster_m,
        two_cta=tile.two_cta,
        out_channels=shape.out_channels_for(phase),
        k_extent=shape.k_for(phase),
        num_a_stages=num_a_stages,
        num_b_stages=num_b_stages,
        num_acc_stages=num_acc_stages,
    )


# ---------------------------------------------------------------------------
# Host-side construction of the cute objects the mainloop needs
# ---------------------------------------------------------------------------


def make_tiled_mmas(plan: GemmPlan):
    """``(tiled_mma, tiled_mma_sfb)`` for a plan.

    Two atoms, because SFB is *not* multicast across a 2-CTA pair the way the
    data operands are: it always uses ``CtaGroup.ONE`` and an N extent rounded
    up to 128.  Building it separately is what keeps the SFB TMA partitioning
    honest under ``two_cta``.
    """
    mma_inst_mn = (plan.mma_m, plan.mma_n)
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        sm100_utils.OperandMajorMode.K,
        sm100_utils.OperandMajorMode.K,
        cutlass.Float8E4M3FN,
        NVFP4_BLOCK,
        plan.cta_group,
        mma_inst_mn,
    )
    sfb_inst_mn = (
        plan.mma_m // (2 if plan.two_cta else 1),
        cute.round_up(plan.mma_n, 128),
    )
    tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        sm100_utils.OperandMajorMode.K,
        sm100_utils.OperandMajorMode.K,
        cutlass.Float8E4M3FN,
        NVFP4_BLOCK,
        tcgen05.CtaGroup.ONE,
        sfb_inst_mn,
    )
    return tiled_mma, tiled_mma_sfb


def make_smem_layouts(plan: GemmPlan, tiled_mma):
    """Staged smem layouts for A, B, SFA, SFB.

    Each is ``(..., stage)``: the mainloop slices off the stage index and the
    pipeline object owns which stage is live.
    """
    a = sm100_utils.make_smem_layout_a(
        tiled_mma, plan.mma_tiler, cutlass.Float4E2M1FN, plan.num_a_stages
    )
    b = sm100_utils.make_smem_layout_b(
        tiled_mma, plan.mma_tiler, cutlass.Float4E2M1FN, plan.num_b_stages
    )
    sfa = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, plan.mma_tiler, NVFP4_BLOCK, plan.num_a_stages
    )
    sfb = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, plan.mma_tiler, NVFP4_BLOCK, plan.num_b_stages
    )
    return a, b, sfa, sfb


def make_cluster_layouts(plan: GemmPlan, tiled_mma, tiled_mma_sfb):
    """Cluster layouts in (v, m, n, k) form, for building multicast masks.

    ``v`` is the intra-atom CTA index (2 under ``two_cta``), so dividing the
    cluster layout by ``thr_id`` separates "which CTA of the MMA pair am I"
    from "which cluster position am I".
    """
    cluster = cute.make_layout((plan.cluster_m, 1, 1))
    return (
        cute.tiled_divide(cluster, (tiled_mma.thr_id.shape,)),
        cute.tiled_divide(cluster, (tiled_mma_sfb.thr_id.shape,)),
    )


def make_tma_atoms(
    plan: GemmPlan,
    tiled_mma,
    tiled_mma_sfb,
    a_gmem: cute.Tensor,
    b_gmem: cute.Tensor,
    sfa_gmem: cute.Tensor,
    sfb_gmem: cute.Tensor,
    smem_layouts,
):
    """Build the four TMA atoms plus their gmem tensor views.

    A is multicast along the cluster's N axis and B along its M axis -- each
    CTA in a cluster needs the *other* operand's full tile, so whichever axis a
    CTA does not span is the axis its operand is broadcast over.
    """
    a_layout, b_layout, sfa_layout, sfb_layout = smem_layouts
    cluster_vmnk, cluster_sfb_vmnk = make_cluster_layouts(
        plan, tiled_mma, tiled_mma_sfb
    )

    a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        (plan.cluster_m, 1, 1), tiled_mma.thr_id
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        (plan.cluster_m, 1, 1), tiled_mma.thr_id
    )
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
        (plan.cluster_m, 1, 1), tiled_mma.thr_id
    )

    a_atom, a_view = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        a_gmem,
        cute.slice_(a_layout, (None, None, None, 0)),
        plan.mma_tiler,
        tiled_mma,
        cluster_vmnk.shape,
    )
    b_atom, b_view = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b_gmem,
        cute.slice_(b_layout, (None, None, None, 0)),
        plan.mma_tiler,
        tiled_mma,
        cluster_vmnk.shape,
    )
    sfa_atom, sfa_view = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        sfa_gmem,
        cute.slice_(sfa_layout, (None, None, None, 0)),
        plan.mma_tiler,
        tiled_mma,
        cluster_vmnk.shape,
    )
    sfb_atom, sfb_view = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op,
        sfb_gmem,
        cute.slice_(sfb_layout, (None, None, None, 0)),
        plan.mma_tiler,
        tiled_mma_sfb,
        cluster_sfb_vmnk.shape,
    )
    return (
        (a_atom, a_view),
        (b_atom, b_view),
        (sfa_atom, sfa_view),
        (sfb_atom, sfb_view),
    )


def smem_bytes(plan: GemmPlan, smem_layouts) -> int:
    """Total smem the mainloop's staged buffers occupy.

    Callers add their own (epilogue staging, comm buffers) and check the sum
    against the architecture budget before compiling.
    """
    a_layout, b_layout, sfa_layout, sfb_layout = smem_layouts
    return (
        cute.size_in_bytes(cutlass.Float4E2M1FN, a_layout)
        + cute.size_in_bytes(cutlass.Float4E2M1FN, b_layout)
        + cute.size_in_bytes(cutlass.Float8E4M3FN, sfa_layout)
        + cute.size_in_bytes(cutlass.Float8E4M3FN, sfb_layout)
    )


# ---------------------------------------------------------------------------
# Device mainloop
# ---------------------------------------------------------------------------
#
# Producer/consumer contract, stated once because it is the part that is easy
# to get subtly wrong:
#
#   * A TMA producer *acquires* a stage (waits until the MMA has retired
#     whatever was there), issues its ``cute.copy`` with the stage's mbarrier
#     as the completion target, and does NOT wait -- the TMA engine signals the
#     barrier asynchronously.
#   * The MMA consumer *waits* on the stage barrier, issues the UMMA, then
#     *releases* the stage.  Under a 2-CTA MMA only the leader CTA issues, but
#     both CTAs' TMAs must have landed, which is what the multicast mask and
#     the cluster-scoped barrier arrange.
#   * ``producer_commit`` on the accumulator pipeline is what makes a finished
#     tile visible to the epilogue warps.
#
# The peek/try_wait dance in the loops below is not decoration: it lets a warp
# discover "the next stage is already full" without paying a barrier wait, and
# is worth real time when the producer is running ahead (which, for weights, is
# the whole design goal).


@cute.jit
def tma_producer_loop(
    tma_atom_data,
    tma_atom_sf,
    g_data,  # (tile, k_tile) gmem view, already sliced to this tile
    g_sf,
    s_data,  # (..., stage) smem tensor
    s_sf,
    producer,
    *,
    k_tiles: cutlass.Constexpr[int],
    mcast_mask_data,
    mcast_mask_sf,
) -> None:
    """Stream one operand's k-tiles into its pipeline.

    Shared verbatim by the A (weight) and B (token) warps: the two differ only
    in which atoms and multicast masks they pass, so keeping one loop means the
    weight and token paths cannot drift apart in their pipeline discipline.
    """
    producer.reset()
    peek = producer.try_acquire()
    for _ in cutlass.range(0, k_tiles, 1, unroll=1):
        handle = producer.acquire_and_advance(peek)
        peek = cutlass.Boolean(1)
        if handle.count + 1 < k_tiles:
            peek = producer.try_acquire()
        cute.copy(
            tma_atom_data,
            g_data[(None, handle.count)],
            s_data[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=mcast_mask_data,
        )
        cute.copy(
            tma_atom_sf,
            g_sf[(None, handle.count)],
            s_sf[(None, handle.index)],
            tma_bar_ptr=handle.barrier,
            mcast_mask=mcast_mask_sf,
        )
    producer.tail()
