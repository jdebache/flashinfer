# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Kernel B: FC2 + combine + the final reduction, in one launch.

Simpler than kernel A, and for a structural reason: kernel A had to overlap the
token exchange with the weight stream, so its extra work runs *concurrently*
with the GEMM.  Here everything extra is genuinely downstream -- a token's
``top_k`` partial results cannot be summed until every rank has finished
producing them -- so it runs *after* the persistent tile loop, on the epilogue
warps that just performed the stores.

    FC2 mainloop (combine scatter is the epilogue's store)
      -> grid barrier      every block on this rank has finished scattering
      -> rank barrier      every peer has finished scattering to us
      -> grid barrier      the rank barrier's release is visible everywhere
      -> reduce            sum each token's top_k slots

The two grid barriers bracket the one cross-rank wait: the first makes this
rank's own scatter complete before it claims to be done, the second stops the
other blocks from reducing before the rank barrier has actually released.

The reduction runs on the epilogue warps only.  They are already rendezvoused
at that point (the mainloop ends with a barrier over exactly those warps), and
the MMA and TMA warps have exited, so any block-wide sync here would hang.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import Float32, Int32, Int64

from .combine import epilogue_fc2_combine
from .dispatch import peer_view
from .fc2 import FC2_EPI_N, stage_floats, store_vector
from .gemm_kernel import launch_grouped_gemm
from .kernel_a import grid_barrier

_REDUCE_BARRIER = 7
_EPI_WARPS = 4
_EPI_THREADS = _EPI_WARPS * 32
_STORE_VEC = 8

# Index into kernel B's coop_args tuple.
(
    _COMBINE,
    _POOLSRC,
    _PEER,
    _IDS,
    _OUT,
    _GSYNC,
    _BSIG,
    _BPHASE,
    _NTOK,
    _RANK,
) = range(10)


@cute.jit
def finalize_combine(
    args,
    tidx: Int32,
    bidx: Int32,
    bidz: Int32,
    gdim_z: Int32,
    *,
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    cluster_m: cutlass.Constexpr[int],
) -> None:
    """Wait for every rank's scatter, then sum this rank's own tokens."""
    flat_block = bidz * Int32(cluster_m) + bidx
    num_blocks = gdim_z * Int32(cluster_m)
    bar = pipeline.NamedBarrier(barrier_id=_REDUCE_BARRIER, num_threads=_EPI_THREADS)
    gen = Int32(1)

    # Our own blocks first: a rank must not announce completion while one of
    # its own blocks is still writing into a peer.
    gen = grid_barrier(args[_GSYNC], gen, tidx, bar, num_blocks=num_blocks)

    if flat_block == Int32(0):
        if tidx == Int32(0):
            phase = Int64(args[_BPHASE][0]) + Int64(1)
            args[_BPHASE][0] = Int32(phase)
            cute.arch.fence_acq_rel_sys()
            for r in cutlass.range_constexpr(world):
                remote = peer_view(
                    args[_BSIG],
                    args[_PEER][r],
                    args[_BSIG].layout,
                    cutlass.Int64,
                    align=8,
                )
                cute.arch.atomic_exch(remote.iterator + args[_RANK][0], phase)
            for r in cutlass.range_constexpr(world):
                seen = cute.arch.atomic_add(args[_BSIG].iterator + r, Int64(0))
                while seen < phase:
                    seen = cute.arch.atomic_add(args[_BSIG].iterator + r, Int64(0))
            cute.arch.fence_acq_rel_sys()
    # Without this, blocks other than 0 would reduce while peers were still
    # scattering into them.
    gen = grid_barrier(args[_GSYNC], gen, tidx, bar, num_blocks=num_blocks)

    num_tokens = args[_NTOK][0]
    chunks: cutlass.Constexpr[int] = hidden // _STORE_VEC
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=_STORE_VEC * 16,
    )
    unit = flat_block * Int32(_EPI_THREADS) + tidx
    stride = num_blocks * Int32(_EPI_THREADS)
    limit = num_tokens * Int32(chunks)
    while unit < limit:
        token = unit // Int32(chunks)
        chunk = unit % Int32(chunks)
        acc = cute.make_rmem_tensor((_STORE_VEC,), Float32)
        for i in cutlass.range_constexpr(_STORE_VEC):
            acc[i] = Float32(0.0)
        # Fixed slot order, so the sum is bit-reproducible whichever rank
        # answered first.  Slots whose expert was invalid were never written
        # and must not be read: the landing buffer is never cleared.
        for slot in cutlass.range_constexpr(top_k):
            expert = args[_IDS][token, slot]
            if expert >= Int32(0) and expert < Int32(num_experts):
                row = Int32(slot * max_tokens) + token
                part = cute.make_rmem_tensor((_STORE_VEC,), cutlass.BFloat16)
                src = cute.zipped_divide(args[_COMBINE][row, None], (_STORE_VEC,))
                cute.copy(load_atom, src[(None,), (chunk,)], part)
                for i in cutlass.range_constexpr(_STORE_VEC):
                    acc[i] += Float32(part[i])
        vals = cute.make_rmem_tensor((_STORE_VEC,), cutlass.BFloat16)
        for i in cutlass.range_constexpr(_STORE_VEC):
            vals[i] = acc[i].to(cutlass.BFloat16)
        store_vector(args[_OUT][token, None], chunk, vals)
        unit += stride


@cute.jit
def launch_kernel_b(
    w2: cute.Tensor,
    sf_w2: cute.Tensor,
    fc1_out: cute.Tensor,
    sf_fc1_out: cute.Tensor,
    prefix: cute.Tensor,
    coop_args,
    stream,
    *,
    local_experts: cutlass.Constexpr[int],
    num_experts: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    intermediate: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    pool_rows: cutlass.Constexpr[int],
    max_tokens: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
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
    launch_grouped_gemm(
        w2,
        fc1_out,
        sf_w2,
        sf_fc1_out,
        (coop_args[_COMBINE], coop_args[_POOLSRC], coop_args[_PEER]),
        prefix,
        stream,
        num_experts=local_experts,
        out_channels=hidden,
        pool_rows=pool_rows,
        k=intermediate,
        mma_m=mma_m,
        mma_n=mma_n,
        cluster_m=cluster_m,
        two_cta=two_cta,
        num_a_stages=num_a_stages,
        num_b_stages=num_b_stages,
        num_clusters=num_clusters,
        acc_stages=1,
        epilogue=functools.partial(
            epilogue_fc2_combine, max_tokens=max_tokens, top_k=top_k
        ),
        epi_n=FC2_EPI_N,
        epi_smem_floats=stage_floats(cta_tile_m, FC2_EPI_N),
        use_pdl=use_pdl,
        finalize=functools.partial(
            finalize_combine,
            max_tokens=max_tokens,
            top_k=top_k,
            hidden=hidden,
            num_experts=num_experts,
            world=world,
            cluster_m=cluster_m,
        ),
        coop_args=coop_args,
    )
