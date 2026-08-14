"""Reference correctness test for fused_moe_add_residual_rmsnorm tracing."""

import torch

from tests.trace.reference_utils import _assert_finite, _check


def test_fused_moe_add_residual_rmsnorm_reference_correctness() -> None:
    import flashinfer
    from flashinfer.trace.templates.norm import (
        fused_moe_add_residual_rmsnorm_trace,
    )

    inputs = fused_moe_add_residual_rmsnorm_trace.init(num_tokens=8, hidden_size=7168)
    expected = fused_moe_add_residual_rmsnorm_trace.reference(**inputs)
    actual = flashinfer.fused_moe_add_residual_rmsnorm(**inputs)
    _assert_finite(*inputs.values(), *expected, *actual)
    _check(fused_moe_add_residual_rmsnorm_trace, expected, actual)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
