# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""FC1 with a fused SwiGLU + requantization epilogue.

This is :mod:`.gemm_kernel` with ``acc_stages=2`` and the epilogue below; the
mainloop is not forked.  Two accumulators are what makes the fusion possible at
all: SwiGLU pairs channel ``i`` of the gate half with channel ``i`` of the up
half, and under swap-AB those sit ``intermediate / mma_m`` channel blocks
apart, so a tile owning one of them alone could not activate anything.

Data flow inside one tile
-------------------------

TMEM hands each thread *one channel and ``epi_n`` tokens* (``m = 4*lane +
warp``), so with gate and up in two accumulators at the same M offset the
activation is register-local -- no shuffle, no smem.

Requantization is the opposite shape: a 16-channel NVFP4 block spans 16
threads across 4 warps.  So the activation is staged to smem as
``(channel, token)`` and re-read token-major, giving each thread 16 consecutive
channels of one token -- the same shape :mod:`.quant` already uses for the
model input, and therefore the same encode path.  Two barriers per pass bracket
the staging buffer, which is reused across subtiles.

The routing weight is applied on the *re-read* side rather than in the TMEM
fragment.  It is per token, and after the transpose a thread owns exactly one
token, so it costs one scalar load instead of ``epi_n`` broadcast loads -- and
it still lands before the block amax, which is where it has to be for the
scale to see the scaled values.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Float32, Int32

from .epilogue import quantize_channel_block, swiglu_activate
from .gemm_kernel import EPI_STAGE_BARRIER, launch_grouped_gemm
from .types import NVFP4_BLOCK

# Token width of one epilogue pass.  Halving it against the plain GEMM's 64
# halves the fp32 staging buffer, which otherwise does not fit alongside four
# weight stages at the reference tile.
FC1_EPI_N = 32
_EPI_THREADS = 128


def stage_floats(cta_tile_m: int, epi_n: int = FC1_EPI_N) -> int:
    """Staging buffer size, in fp32 words, for one epilogue pass.

    The row stride is padded by one word.  Threads write row ``m = 4*lane +
    warp``, so an unpadded stride puts all 32 lanes of a warp on one bank; the
    odd stride spreads them over 8.  That is the best available here -- the
    lane-to-row multiplier of 4 caps it at a 4-way conflict for any stride.
    """
    return cta_tile_m * (epi_n + 1)


@cute.jit
def epilogue_fc1(
    tTR_accs,  # (gate, up) TMEM-partitioned accumulators
    tiled_t2r,
    thr_t2r,
    ch_block: Int32,
    abs_tb: Int32,
    mma_v: Int32,
    tidx: Int32,
    epi_args,  # (out_codes, out_scales, weights)
    stage_smem,
    *,
    acc_stages: cutlass.Constexpr[int],
    channel_blocks: cutlass.Constexpr[int],
    cta_tile_m: cutlass.Constexpr[int],
    tile_tokens: cutlass.Constexpr[int],
    epi_n: cutlass.Constexpr[int],
    cta_per_mma: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    clamp: cutlass.Constexpr,
    apply_weight: cutlass.Constexpr[bool],
) -> None:
    """Activate one tile's paired accumulators and write NVFP4.

    ``channel_blocks`` is unused here and that is deliberate: the mainloop
    calls every epilogue with the same keyword set, and :func:`epilogue_plain`
    does need it (to step ``j * channel_blocks`` tiles to range ``j``'s output
    rows).  FC1 never makes that jump -- its partner channel is already in the
    second accumulator -- so it derives everything from the tile geometry.
    """
    if cutlass.const_expr(acc_stages != 2):
        raise ValueError("FC1 needs a gate and an up accumulator")

    out_codes, out_scales, weights = epi_args[0], epi_args[1], epi_args[2]

    epi_n_sub: cutlass.Constexpr[int] = tile_tokens // epi_n
    blocks: cutlass.Constexpr[int] = cta_tile_m // NVFP4_BLOCK
    # One (token, 16-channel block) unit per thread-step.
    steps: cutlass.Constexpr[int] = (epi_n * blocks) // _EPI_THREADS

    # This CTA's first output channel, as an index into the intermediate axis.
    ch_base = (ch_block * Int32(cta_per_mma) + mma_v) * Int32(cta_tile_m)
    tok_base = abs_tb * Int32(tile_tokens)

    stage = cute.make_tensor(
        stage_smem.iterator,
        cute.make_layout((cta_tile_m, epi_n), stride=(epi_n + 1, 1)),
    )
    # Partition the staging tile with the *same* thread-value layout the TMEM
    # load uses, so the write needs no index arithmetic and cannot disagree
    # with the fragment's channel/token mapping.
    tSt = thr_t2r.partition_D(cute.flat_divide(stage, (cta_tile_m, epi_n)))[
        (None, None, None, 0, 0)
    ]
    frag_gate = cute.make_rmem_tensor(tSt.shape, Float32)
    frag_up = cute.make_rmem_tensor(tSt.shape, Float32)

    bar = pipeline.NamedBarrier(barrier_id=EPI_STAGE_BARRIER, num_threads=_EPI_THREADS)

    for sub in cutlass.range_constexpr(epi_n_sub):
        cute.copy(tiled_t2r, tTR_accs[0][(None, None, None, 0, sub)], frag_gate)
        cute.copy(tiled_t2r, tTR_accs[1][(None, None, None, 0, sub)], frag_up)
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range_constexpr(cute.size(frag_gate)):
            frag_gate[i] = swiglu_activate(
                frag_gate[i],
                frag_up[i],
                Float32(1.0),
                clamp=clamp,
                apply_weight=False,
            )
        # Previous pass' readers must be done before the buffer is rewritten.
        bar.arrive_and_wait()
        cute.autovec_copy(frag_gate, tSt)
        bar.arrive_and_wait()

        for step in cutlass.range_constexpr(steps):
            unit = tidx + Int32(step * _EPI_THREADS)
            # Consecutive lanes take consecutive tokens, so the 16 strided
            # channel reads below are conflict-free across the warp.
            block = unit // Int32(epi_n)
            tok_local = unit % Int32(epi_n)
            token_row = tok_base + Int32(sub * epi_n) + tok_local
            w = Float32(1.0)
            if cutlass.const_expr(apply_weight):
                w = Float32(weights[token_row])
            vals = cute.make_rmem_tensor((NVFP4_BLOCK,), Float32)
            base = block * Int32(NVFP4_BLOCK)
            for i in cutlass.range_constexpr(NVFP4_BLOCK):
                vals[i] = stage[(base + Int32(i), tok_local)] * w
            quantize_channel_block(
                vals,
                token_row,
                ch_base // Int32(NVFP4_BLOCK) + block,
                out_codes,
                out_scales,
                num_k_atoms=num_k_atoms,
            )


@cute.jit
def launch_fc1(
    w1: cute.Tensor,  # (experts, 2*I, hidden) Float4E2M1FN, gate first
    tokens: cute.Tensor,  # (pool_rows, hidden) Float4E2M1FN
    sf_w1: cute.Tensor,  # flat Float8E4M3FN, atom-swizzled
    sf_tokens: cute.Tensor,
    out_codes: cute.Tensor,  # (pool_rows, I) Float4E2M1FN
    out_scales: cute.Tensor,  # flat Float8E4M3FN, atom-swizzled over I
    weights: cute.Tensor,  # (pool_rows,) Float32 routing weights
    prefix: cute.Tensor,
    total_tiles: Int32,
    stream,
    *,
    num_experts: cutlass.Constexpr[int],
    intermediate: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    pool_rows: cutlass.Constexpr[int],
    num_k_atoms: cutlass.Constexpr[int],
    clamp: cutlass.Constexpr = None,
    apply_weight: cutlass.Constexpr[bool] = True,
    mma_m: cutlass.Constexpr[int] = 256,
    mma_n: cutlass.Constexpr[int] = 128,
    cluster_m: cutlass.Constexpr[int] = 2,
    two_cta: cutlass.Constexpr[bool] = True,
    num_a_stages: cutlass.Constexpr[int] = 4,
    num_b_stages: cutlass.Constexpr[int] = 3,
    num_clusters: cutlass.Constexpr[int] = 8,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    cta_tile_m: cutlass.Constexpr[int] = mma_m // (2 if two_cta else 1)
    epilogue = functools.partial(
        epilogue_fc1,
        num_k_atoms=num_k_atoms,
        clamp=clamp,
        apply_weight=apply_weight,
    )
    launch_grouped_gemm(
        w1,
        tokens,
        sf_w1,
        sf_tokens,
        (out_codes, out_scales, weights),
        prefix,
        total_tiles,
        stream,
        num_experts=num_experts,
        out_channels=2 * intermediate,
        pool_rows=pool_rows,
        k=hidden,
        mma_m=mma_m,
        mma_n=mma_n,
        cluster_m=cluster_m,
        two_cta=two_cta,
        num_a_stages=num_a_stages,
        num_b_stages=num_b_stages,
        num_clusters=num_clusters,
        acc_stages=2,
        epilogue=epilogue,
        epi_n=FC1_EPI_N,
        epi_smem_floats=stage_floats(cta_tile_m, FC1_EPI_N),
        use_pdl=use_pdl,
    )
