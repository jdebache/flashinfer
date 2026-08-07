# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-rank test for the fused v2 pipeline, over real NVSHMEM.

Launch with::

    NVSHMEM_DISABLE_CUDA_VMM=1 NVSHMEM_REMOTE_TRANSPORT=none \\
    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_megamoe_v2_multirank.py -v -m "gpu_4 and arch_blackwell"

Both environment variables are about the container, not the kernel.  Without
``NVSHMEM_DISABLE_CUDA_VMM=1`` the symmetric heap fails in ``cuMemCreate`` and
NVSHMEM reports it as ``OUT_OF_MEMORY`` -- misleading, since the GPUs here are
idle with 284 GB free; VMM is simply not permitted.  ``REMOTE_TRANSPORT=none``
skips IB enumeration, which fails on a single-node NVLink box.

What this covers that the in-process tests cannot
-------------------------------------------------

The in-process tests emulate peers by carving every rank's heap out of one
allocation, so a peer offset is a real address delta and the *address
arithmetic* is exercised -- but three things are not:

* the symmetric heap is really symmetric (NVSHMEM guarantees the same offset on
  every PE; the in-process fixture merely arranges it);
* the two in-kernel cross-rank barriers actually rendezvous between processes
  that are running concurrently, rather than being stood in for by launch order;
* peer loads and stores cross NVLink instead of staying in one device's memory.

Every rank reconstructs *all* ranks' inputs from seeds, so the oracle can be
evaluated locally without a gather -- a gather would need collectives that
could mask a dispatch bug by moving the same data a second way.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cutlass")
pytest.importorskip("nvshmem.core")

from flashinfer.moe_ep.kernel_src.megamoe_v2 import launcher  # noqa: E402
from flashinfer.moe_ep.kernel_src.megamoe_v2.reference import (  # noqa: E402
    moe_reference,
    quantize_nvfp4,
)
from flashinfer.moe_ep.kernel_src.megamoe_v2.types import (  # noqa: E402
    CommConfig,
    EpilogueConfig,
    EpTopology,
    KernelConfig,
    Phase,
    ProblemShape,
    TileConfig,
)

_SEED = 101


def _ranks() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=codes.device,
    )
    idx = (codes.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
    nib = torch.where(codes < 0, idx | 0x8, idx).to(torch.uint8)
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()


def _scatter_scales(scales: torch.Tensor) -> torch.Tensor:
    from flashinfer.moe_ep.kernel_src.megamoe_v2 import sf_layout
    from flashinfer.moe_ep.kernel_src.megamoe_v2.types import NVFP4_BLOCK

    rows, blocks = scales.shape
    num_k_atoms = sf_layout.num_k_atoms_for(blocks * NVFP4_BLOCK, NVFP4_BLOCK)
    flat = torch.zeros(
        sf_layout.buffer_words(rows, num_k_atoms=num_k_atoms) * 4,
        dtype=torch.float32,
        device=scales.device,
    )
    for r in range(rows):
        row_block, t_in = divmod(r, 128)
        for b in range(blocks):
            k_atom, k_bank = divmod(b, 4)
            atom = row_block * num_k_atoms + k_atom
            flat[atom * 512 + (t_in % 32) * 16 + (t_in // 32) * 4 + k_bank] = scales[
                r, b
            ]
    return flat.to(torch.float8_e4m3fn)


def _quantize_weights(w: torch.Tensor):
    experts, out, k = w.shape
    q = quantize_nvfp4(w.reshape(-1, k))
    codes = _pack_fp4(q.codes).view(torch.float4_e2m1fn_x2)
    return codes, _scatter_scales(q.scales), q.dequantize().reshape(experts, out, k)


def _inputs_for(rank, *, num_tokens, hidden, num_experts, top_k):
    """Deterministic per-rank inputs; any rank can reproduce any other's."""
    g = torch.Generator(device="cuda").manual_seed(_SEED + rank)
    act = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device="cuda", generator=g
    ).bfloat16()
    logits = torch.rand(num_tokens, num_experts, device="cuda", generator=g) ** 3
    topk_ids = logits.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_ids[::9, -1] = -1
    topk_weights = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device="cuda", generator=g
    )
    return act, topk_ids, topk_weights


def _nvshmem_allocator(world: int, my_pe: int):
    """`alloc_shared` backed by the NVSHMEM symmetric heap.

    Returns the peer *base addresses*, resolved off the peer Buffer rather than
    via ``get_peer_tensor``: the tracker can hold a stale larger peer entry when
    the heap reuses an address, and only the base is needed here.  The self-PE
    is short-circuited to the local pointer, which also avoids bumping the
    parent tracker's refcount (that defers the real free to GC).
    """

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


def _run(
    rank,
    world,
    *,
    num_tokens,
    hidden,
    intermediate,
    num_experts,
    top_k,
    num_clusters=8,
    iterations=1,
):
    import cuda.bindings.driver as cuda

    le = num_experts // world
    config = KernelConfig(
        shape=ProblemShape(
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            top_k=top_k,
            max_tokens_per_rank=num_tokens,
        ),
        topology=EpTopology(world_size=world, rank=rank),
        phase=Phase.FC1,
        tile=TileConfig(mma_m=256, mma_n=128, mma_k=256, cluster_m=2, two_cta=True),
        comm=CommConfig(invalid_expert_id=-1),
        epilogue=EpilogueConfig(gate_up_clamp=None, apply_topk_in_fc1=True),
    )

    # All experts' weights from one seed, so every rank agrees on them and can
    # evaluate the oracle for the whole group.
    gw = torch.Generator(device="cuda").manual_seed(_SEED)
    w13_all = (
        torch.randn(
            num_experts,
            2 * intermediate,
            hidden,
            dtype=torch.float32,
            device="cuda",
            generator=gw,
        )
        * 0.3
    )
    w2_all = (
        torch.randn(
            num_experts,
            hidden,
            intermediate,
            dtype=torch.float32,
            device="cuda",
            generator=gw,
        )
        * 0.3
    )

    mine = slice(rank * le, (rank + 1) * le)
    w1_codes, w1_sf, _ = _quantize_weights(w13_all[mine].contiguous())
    w2_codes, w2_sf, _ = _quantize_weights(w2_all[mine].contiguous())
    weights = launcher.Weights(w1=w1_codes, w1_sf=w1_sf, w2=w2_codes, w2_sf=w2_sf)

    act, topk_ids, topk_weights = _inputs_for(
        rank,
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        top_k=top_k,
    )

    ws = launcher.allocate_workspaces(
        config, rank=rank, alloc_shared=_nvshmem_allocator(world, rank)
    )
    views = launcher.build_views(ws, config)
    out = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    pipe = launcher.compile_fused(
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
    )

    # Compilation is host-side and its cost varies per rank; without this the
    # fastest rank enters the in-kernel cross-rank barrier and spins while the
    # others are still tracing.  Correct either way, but it makes a hang here
    # mean a real bug rather than a slow compile.
    torch.distributed.barrier()
    for _ in range(iterations):
        launcher.run_fused(pipe)
    torch.cuda.synchronize()
    torch.distributed.barrier()

    # Oracle over the whole group, evaluated locally on every rank.
    acts, ids, tw = [], [], []
    for r in range(world):
        a, i, w = _inputs_for(
            r,
            num_tokens=num_tokens,
            hidden=hidden,
            num_experts=num_experts,
            top_k=top_k,
        )
        acts.append(a)
        ids.append(i)
        tw.append(w)
    w13_dq = tuple(
        _quantize_weights(w13_all[r * le : (r + 1) * le].contiguous())[2]
        for r in range(world)
    )
    w2_dq = tuple(
        _quantize_weights(w2_all[r * le : (r + 1) * le].contiguous())[2]
        for r in range(world)
    )
    expected = moe_reference(
        hidden_states=tuple(acts),
        topk_ids=tuple(ids),
        topk_weights=tuple(tw),
        w13=w13_dq,
        w2=w2_dq,
        shape=config.shape,
        topology=config.topology,
        epilogue=config.epilogue,
        invalid_expert_id=-1,
    )[rank]
    return out.float(), expected.float(), views, topk_ids


def _bootstrap(rank, world):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.core.runtime.bootstrap import (
        NVSHMEM,
        bootstrap_moe_ep_runtime,
    )

    torch.cuda.set_device(rank % torch.cuda.device_count())
    return bootstrap_moe_ep_runtime(
        BootstrapConfig(world_size=world, rank=rank), frozenset({NVSHMEM})
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("top_k", [1, 2])
def test_fused_pipeline_multirank(top_k):
    """The fused two-kernel pipeline matches the oracle across real ranks."""
    rank, world = _ranks()
    if world < 2:
        pytest.skip("needs torchrun with >=2 ranks")
    _bootstrap(rank, world)

    got, expected, _views, _ids = _run(
        rank,
        world,
        num_tokens=256,
        hidden=512,
        intermediate=256,
        num_experts=4 * world,
        top_k=top_k,
        iterations=3,
    )
    assert expected.abs().sum() > 0, "reference is degenerate"
    rel = (got - expected).norm().item() / expected.norm().item()
    assert rel < 3e-2, f"rank {rank}: relative error {rel:.4g}"


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_multirank_counts_match_global_routing():
    """Each rank's device-side counts must equal what *every* rank routed to it.

    This is the check that actually needs more than one process: at world=1 it
    is trivially the local routing, and it is the first thing a broken
    cross-rank count push would break.
    """
    rank, world = _ranks()
    if world < 2:
        pytest.skip("needs torchrun with >=2 ranks")
    _bootstrap(rank, world)

    num_experts = 4 * world
    le = num_experts // world
    _got, _exp, views, _ids = _run(
        rank,
        world,
        num_tokens=256,
        hidden=512,
        intermediate=256,
        num_experts=num_experts,
        top_k=2,
    )
    counts = views.expert_token_count.cpu()
    for local_e in range(le):
        expert = rank * le + local_e
        want = sum(
            int(
                (
                    _inputs_for(
                        r,
                        num_tokens=256,
                        hidden=512,
                        num_experts=num_experts,
                        top_k=2,
                    )[1]
                    == expert
                ).sum()
            )
            for r in range(world)
        )
        assert int(counts[local_e]) == want, (
            f"rank {rank} expert {expert}: got {int(counts[local_e])}, want {want}"
        )
