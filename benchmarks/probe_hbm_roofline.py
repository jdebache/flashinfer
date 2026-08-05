"""Measure the achievable HBM read bandwidth behind the mega-MoE weight stream.

The v2 plan's 88 us "weight-streaming floor" assumes ~8 TB/s on the expert
weights (705 MB/rank at the reference shape).  That is the peak spec number,
not an achieved one.  This measures what a read-only stream of exactly that
footprint actually sustains on this device, so the kernel's 150 us can be
compared against a real roofline rather than a spec sheet.

Run on one GPU::

    python benchmarks/probe_hbm_roofline.py
"""

from __future__ import annotations

import argparse
from statistics import median

import torch


def _time_us(fn, *, warmup: int = 5, repeat: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        samples.append(ev0.elapsed_time(ev1) * 1e3)
    return median(samples)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=4096)
    p.add_argument("--local-experts", type=int, default=16)
    args = p.parse_args()

    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)

    # NVFP4 expert weights, 0.5 B/elem: fc1 is (2*I, H), fc2 is (H, I).
    fc1_elems = args.local_experts * 2 * args.intermediate * args.hidden
    fc2_elems = args.local_experts * args.hidden * args.intermediate
    fc1_bytes = fc1_elems // 2
    fc2_bytes = fc2_elems // 2
    total_bytes = fc1_bytes + fc2_bytes

    print(f"GPU: {props.name}  SMs={props.multi_processor_count}")
    print(
        f"weight footprint: fc1 {fc1_bytes / 2**20:.0f} MiB + "
        f"fc2 {fc2_bytes / 2**20:.0f} MiB = {total_bytes / 2**20:.0f} MiB"
    )

    # Read-only stream of the same byte count. uint8 so the reduction is cheap
    # relative to the load; sum() over a flat buffer is a pure streaming read.
    buf = torch.empty(total_bytes // 4, dtype=torch.float32, device=dev)
    buf.uniform_(-1.0, 1.0)
    out = torch.empty(1, dtype=torch.float32, device=dev)

    def _read() -> None:
        torch.sum(buf, dim=0, dtype=torch.float32, out=out[0])

    us = _time_us(_read)
    bw = total_bytes / (us * 1e-6) / 1e12
    print(f"\nread-only stream ({total_bytes / 2**20:.0f} MiB): {us:8.1f} us"
          f"  -> {bw:.2f} TB/s")

    # A large-buffer copy is the conventional STREAM-style upper bound; report it
    # for context (copy moves 2x the bytes: one read + one write).
    big = torch.empty(2 * 1024**3, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(big)

    def _copy() -> None:
        dst.copy_(big)

    us_c = _time_us(_copy, repeat=10)
    bw_c = 2 * big.numel() / (us_c * 1e-6) / 1e12
    print(f"2 GiB copy (r+w):                 {us_c:8.1f} us  -> {bw_c:.2f} TB/s")

    print(
        f"\nweight-stream floor at measured read bw: "
        f"{total_bytes / (bw * 1e12) * 1e6:.1f} us"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
