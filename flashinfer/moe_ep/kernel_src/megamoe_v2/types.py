# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Value types for the v2 split MoE-EP kernel.

Everything here is a frozen product type carrying only plain data.  No kernel
object, no torch tensor, no device handle: these describe *what* the problem
is, never *how* it is executed, so they can be constructed, compared, hashed
and unit-tested on a machine with no GPU.

The split itself is expressed by :class:`Phase`.  Kernel A runs
``Phase.FC1`` (fused input quantization, cross-rank dispatch, FC1 + SwiGLU,
FC1-output requantization); kernel B runs ``Phase.FC2`` (FC2 + cross-rank
combine).  The two exchange data through a GMEM token pool and a per-expert
count vector, both described by :mod:`.layout`.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Literal

# NVFP4: 16 elements share one E4M3 block scale, 2 elements per byte.
NVFP4_BLOCK = 16
NVFP4_ELEMS_PER_BYTE = 2

# The scale-factor atom the tcgen05 MMA expects: scales are stored swizzled in
# (128 row, 4 block) atoms, so both axes get padded to these multiples.
SF_ATOM_ROWS = 128
SF_ATOM_BLOCKS = 4


class Phase(enum.Enum):
    """Which half of the MoE the launch executes."""

    FC1 = "fc1"
    FC2 = "fc2"


def ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def round_up(value: int, multiple: int) -> int:
    return ceil_div(value, multiple) * multiple


@dataclasses.dataclass(frozen=True)
class ProblemShape:
    """The MoE problem, independent of how it is tiled or distributed.

    ``intermediate`` is the *down-projection* width (the SwiGLU output width).
    FC1 therefore produces ``2 * intermediate`` channels (gate ++ up) and FC2
    consumes ``intermediate``.  v1 overloaded a single ``intermediate`` field
    with both meanings depending on call site; keeping the two derived
    properties explicit here is the whole point of the rename.
    """

    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    max_tokens_per_rank: int

    def __post_init__(self) -> None:
        if self.hidden <= 0 or self.intermediate <= 0:
            raise ValueError(
                f"hidden ({self.hidden}) and intermediate ({self.intermediate}) "
                "must be positive"
            )
        if self.hidden % NVFP4_BLOCK != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a multiple of the NVFP4 block "
                f"size ({NVFP4_BLOCK})"
            )
        if self.intermediate % NVFP4_BLOCK != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a multiple of the "
                f"NVFP4 block size ({NVFP4_BLOCK})"
            )
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be in [1, num_experts "
                f"({self.num_experts})]"
            )
        if self.max_tokens_per_rank <= 0:
            raise ValueError(
                f"max_tokens_per_rank ({self.max_tokens_per_rank}) must be positive"
            )

    @property
    def gate_up(self) -> int:
        """FC1 output width: gate and up concatenated."""
        return 2 * self.intermediate

    @property
    def hidden_bytes(self) -> int:
        """Bytes per NVFP4 token row on the hidden axis."""
        return self.hidden // NVFP4_ELEMS_PER_BYTE

    @property
    def hidden_sf_blocks(self) -> int:
        return self.hidden // NVFP4_BLOCK

    @property
    def intermediate_sf_blocks(self) -> int:
        return self.intermediate // NVFP4_BLOCK

    def k_for(self, phase: Phase) -> int:
        """GEMM reduction extent."""
        return self.hidden if phase is Phase.FC1 else self.intermediate

    def out_channels_for(self, phase: Phase) -> int:
        """GEMM output-channel extent (GEMM-M under swap-AB)."""
        return self.gate_up if phase is Phase.FC1 else self.hidden


@dataclasses.dataclass(frozen=True)
class EpTopology:
    """Expert-parallel placement of :class:`ProblemShape` across ranks."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size ({self.world_size}) must be >= 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank ({self.rank}) must be in [0, world_size "
                f"({self.world_size}))"
            )

    def experts_per_rank(self, shape: ProblemShape) -> int:
        if shape.num_experts % self.world_size != 0:
            raise ValueError(
                f"num_experts ({shape.num_experts}) must be divisible by "
                f"world_size ({self.world_size})"
            )
        return shape.num_experts // self.world_size


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """tcgen05 tiling for one GEMM phase, in swap-AB orientation.

    Swap-AB means GEMM-M is the weight/output-channel axis and GEMM-N is the
    token axis.  That is what makes the weight-side TMA descriptors
    tile-invariant, which in turn is what lets kernel A stream weights before
    any token has arrived.

    ``mma_m`` / ``mma_n`` / ``mma_k`` are the MMA atom tile; ``cluster_m`` is
    the CTA cluster extent along the channel axis.  The token axis is never
    clustered (``cluster_n == 1``) because a cluster spanning tokens would need
    the token counts before the descriptors could be built -- exactly the
    dependency the split exists to remove.
    """

    mma_m: int = 256
    mma_n: int = 128
    mma_k: int = 256
    cluster_m: int = 2
    two_cta: bool = True

    def __post_init__(self) -> None:
        if self.mma_k % (NVFP4_BLOCK * SF_ATOM_BLOCKS) != 0:
            raise ValueError(
                f"mma_k ({self.mma_k}) must be a multiple of "
                f"{NVFP4_BLOCK * SF_ATOM_BLOCKS} (SF atom granularity on K)"
            )
        if self.mma_n not in (64, 128, 256):
            raise ValueError(f"mma_n ({self.mma_n}) must be one of 64/128/256")
        per_cta_m = self.mma_m // (2 if self.two_cta else 1)
        if per_cta_m != 128:
            raise ValueError(
                f"per-CTA MMA M must be 128; got {per_cta_m} from "
                f"mma_m={self.mma_m}, two_cta={self.two_cta}"
            )
        if self.two_cta and self.cluster_m % 2 != 0:
            raise ValueError(
                f"cluster_m ({self.cluster_m}) must be even when two_cta=True"
            )
        if self.cluster_m not in (1, 2, 4):
            raise ValueError(f"cluster_m ({self.cluster_m}) must be 1, 2 or 4")

    @property
    def cta_tile_m(self) -> int:
        return self.mma_m // (2 if self.two_cta else 1)

    @property
    def cta_tile_n(self) -> int:
        """Tokens per CTA tile."""
        return self.mma_n

    @property
    def cluster_tile_tokens(self) -> int:
        """Tokens covered by one cluster tile.

        The token axis is unclustered, so this equals the CTA token tile.  It
        is the granularity at which kernel A publishes token readiness.
        """
        return self.mma_n

    @property
    def cluster_size(self) -> int:
        return self.cluster_m

    def channel_blocks(self, out_channels: int) -> int:
        """Cluster tiles along the output-channel (GEMM-M) axis."""
        return ceil_div(out_channels, self.mma_m)

    def token_blocks(self, tokens: int) -> int:
        return ceil_div(tokens, self.cluster_tile_tokens)

    def k_tiles(self, k: int) -> int:
        return ceil_div(k, self.mma_k)


@dataclasses.dataclass(frozen=True)
class CommConfig:
    """Cross-rank transfer policy for dispatch and combine."""

    # Sentinel written into topk_idx rows that carry no token.  vLLM uses
    # num_experts; gpt-oss uses -1.  Kept explicit because getting it wrong is
    # silent corruption, not a crash: a stale sentinel routes a padding row to
    # a real expert.
    invalid_expert_id: int = -1
    # Release-flag batching: how many token arrivals one warp accumulates
    # before publishing to the readiness counter.
    flag_batch: int = 4
    # Combine wire dtype.  bf16 keeps the epilogue store as the NVLink store.
    combine_dtype: Literal["bf16"] = "bf16"

    def __post_init__(self) -> None:
        if self.flag_batch < 1:
            raise ValueError(f"flag_batch ({self.flag_batch}) must be >= 1")
        if self.combine_dtype != "bf16":
            raise ValueError(
                f"combine_dtype ({self.combine_dtype!r}) -- only 'bf16' is "
                "implemented in v2"
            )


@dataclasses.dataclass(frozen=True)
class EpilogueConfig:
    """Numerics applied between and around the two GEMMs."""

    # Clamp applied to the gate and up activations before SwiGLU.
    gate_up_clamp: float | None = None
    # Fold the top-k routing weight into FC1's output rather than after FC2.
    # Doing it pre-FC2 keeps the combine a plain sum.
    apply_topk_in_fc1: bool = True
    # Per-tensor calibrated scale for the bf16 -> NVFP4 input quantization.
    input_norm_const: float = 1.0

    def __post_init__(self) -> None:
        if self.gate_up_clamp is not None and self.gate_up_clamp <= 0.0:
            raise ValueError(
                f"gate_up_clamp ({self.gate_up_clamp}) must be positive or None"
            )
        if self.input_norm_const <= 0.0:
            raise ValueError(
                f"input_norm_const ({self.input_norm_const}) must be positive"
            )


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    """Everything one launch needs, as a single hashable value.

    This is the compile cache key: two launches with equal ``KernelConfig``
    (bar ``phase``) share every codegen-time constant.
    """

    shape: ProblemShape
    topology: EpTopology
    phase: Phase
    tile: TileConfig = dataclasses.field(default_factory=TileConfig)
    comm: CommConfig = dataclasses.field(default_factory=CommConfig)
    epilogue: EpilogueConfig = dataclasses.field(default_factory=EpilogueConfig)

    @property
    def experts_per_rank(self) -> int:
        return self.topology.experts_per_rank(self.shape)

    @property
    def pool_token_capacity(self) -> int:
        """Rows the local token pool must hold, worst case.

        Every token this rank receives occupies one pool row per (token,
        expert) pair.  Each expert also owns at least one whole token tile so
        fused FC1 can issue its first weight tile before the count arrives.
        The bound assumes the adversarial case where every peer sends every
        token here.
        """
        pairs = self.shape.max_tokens_per_rank * self.topology.world_size
        pairs = min(pairs * self.shape.top_k, pairs * self.experts_per_rank)
        min_segments = self.experts_per_rank * self.tile.cluster_tile_tokens
        return pairs + min_segments

    def name(self) -> str:
        """Stable codegen cache key.

        Every field that reaches a ``const_expr`` must appear here; ``rank`` is
        deployment state, not codegen, so it is excluded on purpose (all ranks
        share one compiled artifact).
        """
        s, t, e = self.shape, self.tile, self.epilogue
        return "_".join(
            (
                "megamoe_v2",
                self.phase.value,
                f"h{s.hidden}i{s.intermediate}e{s.num_experts}k{s.top_k}",
                f"maxtok{s.max_tokens_per_rank}",
                f"ep{self.topology.world_size}",
                f"mma{t.mma_m}x{t.mma_n}x{t.mma_k}",
                f"clu{t.cluster_m}{'x2cta' if t.two_cta else ''}",
                f"clamp{e.gate_up_clamp}",
                f"topk{'fc1' if e.apply_topk_in_fc1 else 'post'}",
                f"fb{self.comm.flag_batch}",
                f"inv{self.comm.invalid_expert_id}",
                self.comm.combine_dtype,
            )
        )
