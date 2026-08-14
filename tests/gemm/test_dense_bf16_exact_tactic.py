import pytest

from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    SplitKTactic,
    default_tactic,
    validate_tactic,
)


def test_m96_tactic_is_exact_opt_in() -> None:
    validate_tactic(
        SplitKTactic(mma_m=64, mma_n=96, split_k=4, ab_stages=3),
        96,
        2112,
        7168,
    )
    with pytest.raises(ValueError, match="exact"):
        validate_tactic(
            SplitKTactic(mma_m=64, mma_n=32, split_k=4, ab_stages=3),
            96,
            2112,
            7168,
        )
    with pytest.raises(ValueError, match="exact"):
        default_tactic(96, 2112, 7168)
