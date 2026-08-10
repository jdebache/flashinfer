"""Plot MegaMoE, split A2A, and split AGRS latency."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "benchmarks/results/"
            "mistral_large_3_675b_nvfp4_megamoe_vs_split_ep4_gb300.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/results/"
            "mistral_large_3_675b_nvfp4_megamoe_vs_split_ep4_gb300.png"
        ),
    )
    return parser.parse_args()


def _load_latencies(
    path: Path,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    with path.open(newline="") as input_file:
        rows = tuple(csv.DictReader(input_file))
    batch_sizes = tuple(int(row["tokens_per_rank"]) for row in rows)
    megamoe_ms = tuple(float(row["megamoe_e2e_median_us"]) / 1e3 for row in rows)
    split_a2a_ms = tuple(float(row["split_a2a_e2e_median_us"]) / 1e3 for row in rows)
    split_agrs_ms = tuple(float(row["split_agrs_e2e_median_us"]) / 1e3 for row in rows)
    return batch_sizes, megamoe_ms, split_a2a_ms, split_agrs_ms


def _plot(
    batch_sizes: tuple[int, ...],
    megamoe_ms: tuple[float, ...],
    split_a2a_ms: tuple[float, ...],
    split_agrs_ms: tuple[float, ...],
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(
        batch_sizes,
        megamoe_ms,
        color="#0072B2",
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        label="MegaMoE",
    )
    axis.plot(
        batch_sizes,
        split_a2a_ms,
        color="#D55E00",
        marker="s",
        linewidth=2.4,
        markersize=5.5,
        label="Split (best one/two-sided A2A)",
    )
    axis.plot(
        batch_sizes,
        split_agrs_ms,
        color="#009E73",
        marker="^",
        linewidth=2.4,
        markersize=5.5,
        label="Split (all-gather/reduce-scatter)",
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(batch_sizes, tuple(str(value) for value in batch_sizes))
    axis.set_xlabel("Batch size (tokens per EP rank)")
    axis.set_ylabel("p50 E2E latency (ms)")
    axis.set_title(
        "Mistral Large 3 NVFP4 MoE — EP4 on 4× NVIDIA GB300\n"
        "Aligned CUDA Graph replay · EP-balanced routing · warm cache"
    )
    axis.grid(which="both", color="#B0B0B0", alpha=0.35, linestyle="--")
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    _plot(*_load_latencies(args.input), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
