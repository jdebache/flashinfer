"""Consolidate aligned MegaMoE, split A2A, and split AGRS results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COMMON_COLUMNS = (
    "tokens_per_rank",
    "global_tokens",
    "world_size",
    "hidden",
    "intermediate",
    "total_experts",
    "local_experts",
    "top_k",
    "warmup",
    "iters",
    "timing_mode",
    "routing",
    "l2_flush_mib",
)
MEGAMOE_COLUMNS = (
    "megamoe_status",
    "megamoe_kernel_min_us",
    "megamoe_kernel_median_us",
    "megamoe_kernel_max_us",
    "megamoe_e2e_min_us",
    "megamoe_e2e_median_us",
    "megamoe_e2e_max_us",
    "megamoe_per_gpu_tflops_e2e",
    "megamoe_aggregate_tflops_e2e",
    "megamoe_aggregate_tokens_per_s_e2e",
)
A2A_COLUMNS = (
    "split_a2a_status",
    "split_a2a_backend",
    "split_a2a_stage_timing_mode",
    "split_a2a_dispatch_median_us",
    "split_a2a_cutedsl_moe_median_us",
    "split_a2a_combine_median_us",
    "split_a2a_stage_sum_median_us",
    "split_a2a_aligned_eager_e2e_median_us",
    "split_a2a_e2e_min_us",
    "split_a2a_e2e_median_us",
    "split_a2a_e2e_max_us",
    "split_a2a_per_gpu_tflops_e2e",
    "split_a2a_aggregate_tflops_e2e",
    "split_a2a_aggregate_tokens_per_s_e2e",
    "split_a2a_over_megamoe",
)
AGRS_COLUMNS = (
    "split_agrs_status",
    "split_agrs_stage_timing_mode",
    "split_agrs_all_gather_median_us",
    "split_agrs_cutedsl_moe_median_us",
    "split_agrs_reduce_scatter_median_us",
    "split_agrs_stage_sum_median_us",
    "split_agrs_aligned_eager_e2e_median_us",
    "split_agrs_e2e_min_us",
    "split_agrs_e2e_median_us",
    "split_agrs_e2e_max_us",
    "split_agrs_per_gpu_tflops_e2e",
    "split_agrs_aggregate_tflops_e2e",
    "split_agrs_aggregate_tokens_per_s_e2e",
    "split_agrs_over_megamoe",
)
CSV_COLUMNS = COMMON_COLUMNS + MEGAMOE_COLUMNS + A2A_COLUMNS + AGRS_COLUMNS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    results_dir = Path("benchmarks/results")
    parser.add_argument(
        "--megamoe",
        type=Path,
        default=results_dir / "mistral_large_3_675b_nvfp4_megamoe_ep4_gb300.csv",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=results_dir / "mistral_large_3_675b_nvfp4_unfused_ep4_gb300.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=results_dir
        / "mistral_large_3_675b_nvfp4_megamoe_vs_split_ep4_gb300.csv",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="") as input_file:
        return tuple(csv.DictReader(input_file))


def _selected_a2a_rows(path: Path) -> tuple[dict[str, str], ...]:
    return tuple(row for row in _load_rows(path) if row["selected"] == "True")


def _agrs_rows(path: Path) -> tuple[dict[str, str], ...]:
    return tuple(row for row in _load_rows(path) if row["backend"] == "agrs")


def _index_by_tokens(
    rows: tuple[dict[str, str], ...],
) -> dict[int, dict[str, str]]:
    indexed = {int(row["tokens_per_rank"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate tokens_per_rank rows")
    return indexed


def _require_equal(
    tokens: int,
    megamoe: dict[str, str],
    split: dict[str, str],
    column: str,
) -> str:
    if megamoe[column] != split[column]:
        raise ValueError(
            f"tokens_per_rank={tokens}: mismatched {column}: "
            f"{megamoe[column]} != {split[column]}"
        )
    return megamoe[column]


def _consolidate_row(
    tokens: int,
    megamoe: dict[str, str],
    split_a2a: dict[str, str],
    split_agrs: dict[str, str],
) -> dict[str, str]:
    if megamoe["timing_mode"] != split_a2a["e2e_timing_mode"]:
        raise ValueError(f"tokens_per_rank={tokens}: mismatched E2E timing modes")
    if megamoe["timing_mode"] != split_agrs["e2e_timing_mode"]:
        raise ValueError(f"tokens_per_rank={tokens}: mismatched AGRS timing mode")
    if split_a2a["global_tokens"] != split_agrs["global_tokens"]:
        raise ValueError(f"tokens_per_rank={tokens}: mismatched global token counts")
    for column in (
        "world_size",
        "hidden",
        "intermediate",
        "total_experts",
        "local_experts",
        "top_k",
        "warmup",
        "iters",
        "routing",
        "l2_flush_mib",
    ):
        _require_equal(tokens, megamoe, split_agrs, column)
    row = {
        "tokens_per_rank": str(tokens),
        "global_tokens": split_a2a["global_tokens"],
        "world_size": _require_equal(tokens, megamoe, split_a2a, "world_size"),
        "hidden": _require_equal(tokens, megamoe, split_a2a, "hidden"),
        "intermediate": _require_equal(tokens, megamoe, split_a2a, "intermediate"),
        "total_experts": _require_equal(tokens, megamoe, split_a2a, "total_experts"),
        "local_experts": _require_equal(tokens, megamoe, split_a2a, "local_experts"),
        "top_k": _require_equal(tokens, megamoe, split_a2a, "top_k"),
        "warmup": _require_equal(tokens, megamoe, split_a2a, "warmup"),
        "iters": _require_equal(tokens, megamoe, split_a2a, "iters"),
        "timing_mode": megamoe["timing_mode"],
        "routing": _require_equal(tokens, megamoe, split_a2a, "routing"),
        "l2_flush_mib": _require_equal(tokens, megamoe, split_a2a, "l2_flush_mib"),
        "megamoe_status": megamoe["status"],
        "megamoe_kernel_min_us": megamoe["kernel_critical_min_us"],
        "megamoe_kernel_median_us": megamoe["kernel_critical_median_us"],
        "megamoe_kernel_max_us": megamoe["kernel_critical_max_us"],
        "megamoe_e2e_min_us": megamoe["e2e_critical_min_us"],
        "megamoe_e2e_median_us": megamoe["e2e_critical_median_us"],
        "megamoe_e2e_max_us": megamoe["e2e_critical_max_us"],
        "megamoe_per_gpu_tflops_e2e": megamoe["per_gpu_tflops_e2e"],
        "megamoe_aggregate_tflops_e2e": megamoe["aggregate_tflops_e2e"],
        "megamoe_aggregate_tokens_per_s_e2e": megamoe["aggregate_tokens_per_s_e2e"],
        "split_a2a_status": split_a2a["status"],
        "split_a2a_backend": split_a2a["backend"],
        "split_a2a_stage_timing_mode": split_a2a["stage_timing_mode"],
        "split_a2a_dispatch_median_us": split_a2a["dispatch_critical_median_us"],
        "split_a2a_cutedsl_moe_median_us": split_a2a["compute_critical_median_us"],
        "split_a2a_combine_median_us": split_a2a["combine_critical_median_us"],
        "split_a2a_stage_sum_median_us": split_a2a["stage_sum_median_us"],
        "split_a2a_aligned_eager_e2e_median_us": split_a2a[
            "aligned_eager_e2e_critical_median_us"
        ],
        "split_a2a_e2e_min_us": split_a2a["e2e_critical_min_us"],
        "split_a2a_e2e_median_us": split_a2a["e2e_critical_median_us"],
        "split_a2a_e2e_max_us": split_a2a["e2e_critical_max_us"],
        "split_a2a_per_gpu_tflops_e2e": split_a2a["per_gpu_tflops_e2e"],
        "split_a2a_aggregate_tflops_e2e": split_a2a["aggregate_tflops_e2e"],
        "split_a2a_aggregate_tokens_per_s_e2e": split_a2a["aggregate_tokens_per_s_e2e"],
        "split_a2a_over_megamoe": split_a2a["unfused_over_fused"],
        "split_agrs_status": split_agrs["status"],
        "split_agrs_stage_timing_mode": split_agrs["stage_timing_mode"],
        "split_agrs_all_gather_median_us": split_agrs["dispatch_critical_median_us"],
        "split_agrs_cutedsl_moe_median_us": split_agrs["compute_critical_median_us"],
        "split_agrs_reduce_scatter_median_us": split_agrs["combine_critical_median_us"],
        "split_agrs_stage_sum_median_us": split_agrs["stage_sum_median_us"],
        "split_agrs_aligned_eager_e2e_median_us": split_agrs[
            "aligned_eager_e2e_critical_median_us"
        ],
        "split_agrs_e2e_min_us": split_agrs["e2e_critical_min_us"],
        "split_agrs_e2e_median_us": split_agrs["e2e_critical_median_us"],
        "split_agrs_e2e_max_us": split_agrs["e2e_critical_max_us"],
        "split_agrs_per_gpu_tflops_e2e": split_agrs["per_gpu_tflops_e2e"],
        "split_agrs_aggregate_tflops_e2e": split_agrs["aggregate_tflops_e2e"],
        "split_agrs_aggregate_tokens_per_s_e2e": split_agrs[
            "aggregate_tokens_per_s_e2e"
        ],
        "split_agrs_over_megamoe": split_agrs["unfused_over_fused"],
    }
    return row


def _consolidate(
    megamoe_path: Path,
    split_path: Path,
) -> tuple[dict[str, str], ...]:
    megamoe = _index_by_tokens(_load_rows(megamoe_path))
    split_a2a = _index_by_tokens(_selected_a2a_rows(split_path))
    split_agrs = _index_by_tokens(_agrs_rows(split_path))
    if megamoe.keys() != split_a2a.keys():
        raise ValueError("MegaMoE and selected split A2A token sets differ")
    if megamoe.keys() != split_agrs.keys():
        raise ValueError("MegaMoE and split AGRS token sets differ")
    return tuple(
        _consolidate_row(tokens, megamoe[tokens], split_a2a[tokens], split_agrs[tokens])
        for tokens in sorted(megamoe)
    )


def _write_rows(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    _write_rows(args.output, _consolidate(args.megamoe, args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
