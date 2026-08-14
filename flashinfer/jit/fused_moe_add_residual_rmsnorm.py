"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags
from .cpp_ext import is_cuda_version_at_least


def gen_fused_moe_add_residual_rmsnorm_sm100_module() -> JitSpec:
    """Create the SM100/SM103 JIT module."""

    sm103_flags = (
        ["-gencode=arch=compute_103a,code=sm_103a"]
        if is_cuda_version_at_least("12.9")
        else []
    )
    return gen_jit_spec(
        "fused_moe_add_residual_rmsnorm_sm100",
        [jit_env.FLASHINFER_CSRC_DIR / "fused_moe_add_residual_rmsnorm_sm100.cu"],
        extra_cuda_cflags=sm100a_nvcc_flags + sm103_flags + ["--use_fast_math"],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
