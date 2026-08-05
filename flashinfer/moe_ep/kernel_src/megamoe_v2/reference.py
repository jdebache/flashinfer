# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Torch oracle for the v2 split kernel.

Deliberately independent of the device implementation: plain torch ops on
dequantized values, so it validates the kernel's *arithmetic*, not just its
plumbing.  It is also the executable specification of the NVFP4 encoding --
:func:`quantize_nvfp4` and :func:`dequantize_nvfp4` define what the device code
must reproduce bit for bit, and every rounding decision the kernel makes is
stated here once.

The oracle models the full EP dataflow (route -> dispatch -> FC1 -> FC2 ->
combine) so a single-rank run and a multi-rank run share one reference, and so
the dispatch permutation itself is covered rather than assumed.
"""

from __future__ import annotations

import dataclasses

import torch

from .types import NVFP4_BLOCK, EpTopology, EpilogueConfig, ProblemShape

# FP4 E2M1 representable magnitudes.  The largest is 6.0, which is what the
# per-block scale normalizes against.
_FP4_MAX = 6.0
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# E4M3 finite max; block scales saturate here.
_E4M3_MAX = 448.0


def _round_to_fp4(values: torch.Tensor) -> torch.Tensor:
    """Round to the nearest representable E2M1 magnitude, ties to even.

    Done by explicit comparison against the level table rather than by a cast,
    so the reference does not inherit whatever rounding mode a torch/CUDA cast
    happens to use -- the point of an oracle is to state the intent.
    """
    sign = torch.sign(values)
    mag = values.abs()
    levels = torch.tensor(_FP4_LEVELS, dtype=values.dtype, device=values.device)
    # Midpoints between consecutive levels; searchsorted then picks the nearest.
    mids = (levels[1:] + levels[:-1]) / 2
    idx = torch.searchsorted(mids.contiguous(), mag.contiguous())
    nearest = levels[idx]
    # Ties-to-even: at an exact midpoint prefer the level with the even index.
    at_mid = torch.zeros_like(mag, dtype=torch.bool)
    for i, mid in enumerate(mids.tolist()):
        at_mid |= mag == mid
        if i % 2 == 0:
            nearest = torch.where(mag == mid, levels[i], nearest)
        else:
            nearest = torch.where(mag == mid, levels[i + 1], nearest)
    return sign * nearest


@dataclasses.dataclass(frozen=True)
class Nvfp4Tensor:
    """A dequantizable NVFP4 value: fp4 codes plus per-block E4M3 scales.

    Stored as float32 magnitudes rather than packed bytes; packing is a
    transport concern and belongs in the device code, not in the oracle.
    """

    codes: torch.Tensor  # (..., K) float32, each an exact E2M1 magnitude
    scales: torch.Tensor  # (..., K // NVFP4_BLOCK) float32, each exact E4M3
    norm_const: float

    def dequantize(self) -> torch.Tensor:
        blocks = self.codes.shape[-1] // NVFP4_BLOCK
        shaped = self.codes.reshape(*self.codes.shape[:-1], blocks, NVFP4_BLOCK)
        out = shaped * self.scales.unsqueeze(-1) / self.norm_const
        return out.reshape(self.codes.shape)


def quantize_nvfp4(values: torch.Tensor, norm_const: float = 1.0) -> Nvfp4Tensor:
    """Per-16-element-block NVFP4 quantization.

    The block scale is ``amax / FP4_MAX * norm_const`` cast to E4M3; elements
    are then divided by that scale (times ``norm_const``) and rounded to E2M1.
    An all-zero block yields a zero scale, and its elements must stay zero
    rather than becoming NaN -- that guard is the reason the multiply below is
    written as a masked reciprocal.
    """
    if values.shape[-1] % NVFP4_BLOCK != 0:
        raise ValueError(
            f"last dim ({values.shape[-1]}) must be a multiple of {NVFP4_BLOCK}"
        )
    work = values.to(torch.float32)
    blocks = work.shape[-1] // NVFP4_BLOCK
    shaped = work.reshape(*work.shape[:-1], blocks, NVFP4_BLOCK)

    amax = shaped.abs().amax(dim=-1)
    scale = (amax / _FP4_MAX * norm_const).clamp(max=_E4M3_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)

    nonzero = scale > 0
    inv = torch.where(nonzero, norm_const / scale.clamp(min=1e-30), torch.zeros_like(scale))
    codes = _round_to_fp4(shaped * inv.unsqueeze(-1))
    return Nvfp4Tensor(
        codes=codes.reshape(work.shape),
        scales=scale,
        norm_const=norm_const,
    )


def roundtrip_nvfp4(values: torch.Tensor, norm_const: float = 1.0) -> torch.Tensor:
    """``dequantize(quantize(x))`` -- the value the GEMM actually sees."""
    return quantize_nvfp4(values, norm_const).dequantize()


def swiglu(
    gate_up: torch.Tensor,
    *,
    clamp: float | None,
) -> torch.Tensor:
    """SwiGLU over a gate++up concatenation on the last axis.

    ``gate_up`` is ``(..., 2 * intermediate)`` with gate first.  The clamp is
    applied to both halves *before* the activation, matching the kernel's
    pre-activation clamp (clamping after silu would change the gradient of the
    saturation region and is a different function).
    """
    half = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if clamp is not None:
        gate = gate.clamp(-clamp, clamp)
        up = up.clamp(-clamp, clamp)
    return torch.nn.functional.silu(gate) * up


@dataclasses.dataclass(frozen=True)
class DispatchPlan:
    """Which (token, slot) pairs land on which local expert of which rank.

    ``pairs[rank]`` lists ``(src_rank, token, slot, local_expert)`` in the
    order the oracle assigns pool rows, so a test can compare the kernel's
    pool contents row for row rather than only comparing the final output.
    """

    pairs: tuple[tuple[tuple[int, int, int, int], ...], ...]

    def counts(self, rank: int, experts_per_rank: int) -> tuple[int, ...]:
        counts = [0] * experts_per_rank
        for _src, _tok, _slot, local_expert in self.pairs[rank]:
            counts[local_expert] += 1
        return tuple(counts)


def plan_dispatch(
    topk_ids: tuple[torch.Tensor, ...],
    *,
    shape: ProblemShape,
    world_size: int,
    invalid_expert_id: int,
) -> DispatchPlan:
    """Group every live (token, slot) of every rank by destination.

    Ordering within a destination expert is (source rank, token, slot).  The
    device pull uses a different interleave for bandwidth reasons, so tests
    that compare pool contents must sort; tests that compare the final output
    are order-insensitive anyway because combine is a sum.
    """
    experts_per_rank = shape.num_experts // world_size
    per_rank: list[list[tuple[int, int, int, int]]] = [[] for _ in range(world_size)]
    for src_rank, ids in enumerate(topk_ids):
        for token in range(ids.shape[0]):
            for slot in range(ids.shape[1]):
                expert = int(ids[token, slot])
                if expert == invalid_expert_id or not 0 <= expert < shape.num_experts:
                    continue
                dst = expert // experts_per_rank
                per_rank[dst].append(
                    (src_rank, token, slot, expert % experts_per_rank)
                )
    for dst in range(world_size):
        per_rank[dst].sort(key=lambda p: (p[3], p[0], p[1], p[2]))
    return DispatchPlan(pairs=tuple(tuple(p) for p in per_rank))


def moe_reference(
    *,
    hidden_states: tuple[torch.Tensor, ...],
    topk_ids: tuple[torch.Tensor, ...],
    topk_weights: tuple[torch.Tensor, ...],
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
    shape: ProblemShape,
    topology: EpTopology,
    epilogue: EpilogueConfig,
    invalid_expert_id: int = -1,
) -> tuple[torch.Tensor, ...]:
    """Full EP MoE in fp32 over NVFP4-roundtripped operands.

    Inputs are per-rank tuples: ``hidden_states[r]`` is rank ``r``'s bf16
    activations, ``w13[r]`` / ``w2[r]`` are rank ``r``'s *local* expert
    weights.  Returns one bf16 output per rank.

    The quantization round trips are applied exactly where the kernel applies
    them -- inputs before dispatch, FC1 output before FC2, weights at load --
    so the oracle carries the same precision loss and can be compared with a
    tight tolerance rather than a loose one.
    """
    world_size = topology.world_size
    experts_per_rank = shape.num_experts // world_size

    # Weights are quantized once at load time in the real pipeline.
    w13_q = tuple(roundtrip_nvfp4(w) for w in w13)
    w2_q = tuple(roundtrip_nvfp4(w) for w in w2)
    # Activations are quantized before they go on the wire.
    acts_q = tuple(
        roundtrip_nvfp4(h, epilogue.input_norm_const) for h in hidden_states
    )

    outputs = [
        torch.zeros(h.shape[0], shape.hidden, dtype=torch.float32, device=h.device)
        for h in hidden_states
    ]

    for src_rank in range(world_size):
        ids = topk_ids[src_rank]
        weights = topk_weights[src_rank]
        for token in range(ids.shape[0]):
            row = acts_q[src_rank][token]
            for slot in range(ids.shape[1]):
                expert = int(ids[token, slot])
                if expert == invalid_expert_id or not 0 <= expert < shape.num_experts:
                    continue
                owner, local = divmod(expert, experts_per_rank)
                weight = float(weights[token, slot])

                gate_up = row @ w13_q[owner][local].T
                act = swiglu(gate_up, clamp=epilogue.gate_up_clamp)
                if epilogue.apply_topk_in_fc1:
                    act = act * weight
                # FC1 output is requantized before FC2 reads it.
                act = roundtrip_nvfp4(act)
                partial = act @ w2_q[owner][local].T
                if not epilogue.apply_topk_in_fc1:
                    partial = partial * weight
                outputs[src_rank][token] += partial

    return tuple(o.to(torch.bfloat16) for o in outputs)
