# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Split MoE-EP kernel (v2): dispatch+FC1 and FC2+combine as two launches.

Written from scratch rather than derived from ``cutedsl_megamoe``.  The parts
that carry no device dependency -- problem/tile/comm value types, the work-tile
schedule, and the workspace layout -- are importable without CUDA so they can
be unit-tested on any machine; device code is imported lazily by the launcher.
"""

from __future__ import annotations

from .layout import (
    Region,
    WorkspaceLayout,
    local_layout,
    shared_layout,
    workspace_sizes,
)
from .schedule import (
    ExpertLayout,
    WorkTile,
    build_expert_layout,
    decode_tile,
    enumerate_tiles,
    persistent_tile_indices,
)
from .types import (
    CommConfig,
    EpTopology,
    EpilogueConfig,
    KernelConfig,
    NVFP4_BLOCK,
    Phase,
    ProblemShape,
    TileConfig,
    ceil_div,
    round_up,
)

__all__ = [
    "CommConfig",
    "EpTopology",
    "EpilogueConfig",
    "ExpertLayout",
    "KernelConfig",
    "NVFP4_BLOCK",
    "Phase",
    "ProblemShape",
    "Region",
    "TileConfig",
    "WorkTile",
    "WorkspaceLayout",
    "build_expert_layout",
    "ceil_div",
    "decode_tile",
    "enumerate_tiles",
    "local_layout",
    "persistent_tile_indices",
    "round_up",
    "shared_layout",
    "workspace_sizes",
]
