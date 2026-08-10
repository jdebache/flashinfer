"""SM100/SM103 unfused NVFP4 CuTeDSL MoE EP benchmark.

The default geometry matches Mistral-Large-3-675B-Instruct-2512-NVFP4:
hidden 7168, expert intermediate 4096, 128 routed experts, and top-k 4.
Four torchrun ranks therefore hold 32 experts each.

The timed pipeline is prequantized NVFP4 activation communication, local
CuTeDSL MoE, and BF16 output communication. E2E latency uses an uninstrumented
CUDA Graph replay with every sample device-drained and barrier-aligned across
ranks. Stage latencies use a separately aligned eager run so timing event nodes
do not perturb the E2E graph. It compares two split communication strategies:

* FlashInfer's NVLink one-sided/two-sided all-to-all. One-sided is measured
  through 256 tokens per rank; two-sided is measured from 128 tokens per rank.
  The faster complete pipeline is selected at the overlap points.
* NCCL all-gather of the four activation/routing payloads, followed by local
  expert compute and BF16 reduce-scatter. This is measured at every token size.

Example:

    torchrun --standalone --nproc_per_node=4 \
        benchmarks/bench_moe_ep_sm100_nvfp4_unfused.py \
        --output /tmp/mistral_large_3_nvfp4_unfused_ep4.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable

_BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or os.getcwd()).resolve() != _BENCHMARK_DIR
]

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from flashinfer.autotuner import autotune  # noqa: E402
from flashinfer.comm import (  # noqa: E402
    MoeAlltoAll,
    moe_a2a_get_workspace_size_per_rank,
)
from flashinfer.comm.mapping import Mapping  # noqa: E402
from flashinfer.comm.mnnvl import MnnvlConfig, TorchDistBackend  # noqa: E402
from flashinfer.comm.trtllm_alltoall import MnnvlMoe  # noqa: E402
from flashinfer.cute_dsl.utils import (  # noqa: E402
    convert_sf_to_mma_layout,
    get_mma_sf_shape,
)
from flashinfer.fp4_quantization import fp4_quantize  # noqa: E402
from flashinfer.fused_moe import (  # noqa: E402
    BackendOptions,
    CuteDslConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
)

DEFAULT_TOKENS = tuple(2**power for power in range(2, 15))
ONE_SIDED_MAX_TOKENS = 256
TWO_SIDED_MIN_TOKENS = 128
STRATEGY_RELATIVE_TOLERANCE = 1e-2
CSV_COLUMNS = (
    "tokens_per_rank",
    "global_tokens",
    "world_size",
    "backend",
    "selected",
    "total_experts",
    "local_experts",
    "top_k",
    "hidden",
    "intermediate",
    "warmup",
    "iters",
    "e2e_timing_mode",
    "stage_timing_mode",
    "routing",
    "l2_flush_mib",
    "status",
    "dispatch_critical_median_us",
    "compute_critical_median_us",
    "combine_critical_median_us",
    "stage_sum_median_us",
    "aligned_eager_e2e_critical_median_us",
    "e2e_critical_min_us",
    "e2e_critical_median_us",
    "e2e_critical_max_us",
    "per_gpu_tflops_e2e",
    "aggregate_tflops_e2e",
    "aggregate_tokens_per_s_e2e",
    "fused_e2e_critical_median_us",
    "unfused_over_fused",
    "error",
)


@dataclass(frozen=True)
class CandidateSpec:
    tokens: int
    backend: str


@dataclass(frozen=True)
class DispatchState:
    activations: MoEActivationPack
    context: Any = None


@dataclass(frozen=True)
class Pipeline:
    dispatch: Callable[[], DispatchState]
    compute: Callable[[MoEActivationPack], torch.Tensor]
    combine: Callable[[DispatchState, torch.Tensor], None]
    output: torch.Tensor


@dataclass(frozen=True)
class CapturedPipeline:
    graph: torch.cuda.CUDAGraph


@dataclass(frozen=True)
class RankSamples:
    dispatch_us: tuple[float, ...]
    compute_us: tuple[float, ...]
    combine_us: tuple[float, ...]
    e2e_us: tuple[float, ...]


@dataclass(frozen=True)
class CandidateResult:
    status: str
    graph_e2e_samples_by_rank: tuple[tuple[float, ...], ...] = ()
    eager_samples_by_rank: tuple[RankSamples, ...] = ()
    error: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tokens",
        default=",".join(str(tokens) for tokens in DEFAULT_TOKENS),
        help="Comma-separated tokens per rank.",
    )
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--l2-flush-mib",
        type=int,
        default=0,
        help="MiB to flush before each aligned sample; zero measures warm-cache replay.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--no-pdl", action="store_true")
    parser.add_argument("--deterministic-finalize", action="store_true")
    parser.add_argument(
        "--fused-results",
        type=Path,
        default=Path(
            "benchmarks/results/mistral_large_3_675b_nvfp4_megamoe_ep4_gb300.csv"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, world_size: int) -> tuple[int, ...]:
    tokens = tuple(int(value) for value in args.tokens.split(",") if value)
    if not tokens or any(value <= 0 for value in tokens):
        raise ValueError("--tokens must contain positive integers")
    if world_size != 4:
        raise ValueError(f"this EP4 benchmark requires 4 ranks, got {world_size}")
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by the world size")
    if args.top_k <= 0 or args.top_k > args.num_experts:
        raise ValueError("--top-k must be in [1, num_experts]")
    if args.top_k != world_size:
        raise ValueError("per-token EP-balanced routing requires top-k == world size")
    if args.hidden % 64 != 0 or args.intermediate % 64 != 0:
        raise ValueError("--hidden and --intermediate must be multiples of 64")
    if args.warmup < 0 or args.iters <= 0 or args.l2_flush_mib < 0:
        raise ValueError("warmup/flush must be non-negative and iters positive")
    return tokens


def _candidate_specs(tokens: tuple[int, ...]) -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(tokens=value, backend=backend)
        for value in tokens
        for backend in (
            *(("one_sided",) if value <= ONE_SIDED_MAX_TOKENS else ()),
            *(("two_sided",) if value >= TWO_SIDED_MIN_TOKENS else ()),
            "agrs",
        )
    )


def _balanced_routing(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    local_experts = num_experts // world_size
    token_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)[:, None]
    route_ids = torch.arange(top_k, device=device, dtype=torch.int32)[None, :]
    target_ranks = (rank + route_ids) % world_size
    local_ids = (token_ids * top_k + route_ids + rank) % local_experts
    return target_ranks * local_experts + local_ids


def _make_inputs(
    args: argparse.Namespace,
    tokens: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        tokens,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    global_scale = torch.ones(1, dtype=torch.float32, device=device)
    hidden_states_q, hidden_states_scale = fp4_quantize(
        hidden_states,
        global_scale=global_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
        backend="cute-dsl",
    )
    topk_ids = _balanced_routing(
        tokens, args.top_k, args.num_experts, rank, world_size, device
    )
    topk_weights = torch.softmax(
        torch.randn(
            tokens,
            args.top_k,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        dim=-1,
    )
    return hidden_states_q, hidden_states_scale, topk_ids, topk_weights


def _make_weight_scale(
    m: int,
    k: int,
    num_groups: int,
    device: torch.device,
) -> torch.Tensor:
    shape = get_mma_sf_shape(m, k, num_groups=num_groups, sf_vec_size=16)
    storage = torch.ones(math.prod(shape), dtype=torch.float8_e4m3fn, device=device)
    return convert_sf_to_mma_layout(
        storage,
        m=m,
        k=k,
        num_groups=num_groups,
        sf_vec_size=16,
    )


def _make_weight_pack(
    args: argparse.Namespace,
    local_experts: int,
    rank: int,
    device: torch.device,
) -> MoEWeightPack:
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + rank)
    w1_weight = torch.randint(
        0,
        256,
        (local_experts, 2 * args.intermediate, args.hidden // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    w2_weight = torch.randint(
        0,
        256,
        size=(local_experts, args.hidden, args.intermediate // 2),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    ones = torch.ones(local_experts, dtype=torch.float32, device=device)
    native_view = {
        "w1_weight": w1_weight,
        "w1_weight_sf": _make_weight_scale(
            2 * args.intermediate, args.hidden, local_experts, device
        ),
        "w1_alpha": ones,
        "fc2_input_scale": torch.ones(1, dtype=torch.float32, device=device),
        "w2_weight": w2_weight,
        "w2_weight_sf": _make_weight_scale(
            args.hidden, args.intermediate, local_experts, device
        ),
        "w2_alpha": ones,
    }
    pack = MoEWeightPack()
    pack.prepare_for("cute_dsl_nvfp4", native_view)
    return pack


def _make_moe_layer(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    max_tokens: int,
    device: torch.device,
) -> MoELayer:
    local_experts = args.num_experts // world_size
    config = MoEConfig(
        routing=RoutingConfig(num_experts=args.num_experts, top_k=args.top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4, per_token_scale=False),
        experts=ExpertConfig(
            intermediate_size=args.intermediate,
            local_expert_offset=rank * local_experts,
            local_num_experts=local_experts,
        ),
        backend=BackendOptions(candidates=(CuteDslConfig(),)),
        execution=ExecutionConfig(
            enable_pdl=not args.no_pdl,
            tune_max_num_tokens=max_tokens * world_size,
            use_fused_finalize=not args.deterministic_finalize,
        ),
    )
    return MoELayer(config, device=device)


def _make_one_sided(
    args: argparse.Namespace,
    tokens: int,
    payloads: tuple[torch.Tensor, ...],
    moe_a2a: MoeAlltoAll,
    layer: MoELayer,
    weight_pack: MoEWeightPack,
    world_size: int,
    device: torch.device,
) -> Pipeline:
    hidden_states_q, hidden_states_scale, topk_ids, topk_weights = payloads
    output = torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device=device)

    def dispatch_one_sided() -> DispatchState:
        recv = moe_a2a.dispatch(
            token_selected_experts=topk_ids,
            input_payloads=[
                hidden_states_q,
                hidden_states_scale,
                topk_ids,
                topk_weights,
            ],
            runtime_max_tokens_per_rank=tokens,
            invalid_token_expert_id=-1,
            expert_id_payload_index=2,
        )
        activations = MoEActivationPack(
            hidden_states_q=recv[0].view(world_size * tokens, -1),
            hidden_states_scale=recv[1].view(world_size * tokens, -1),
            topk_ids=recv[2].view(world_size * tokens, args.top_k),
            topk_weights=recv[3].view(world_size * tokens, args.top_k),
        )
        return DispatchState(activations=activations)

    def compute(activations: MoEActivationPack) -> torch.Tensor:
        return layer(activations, weight_pack)

    def combine_one_sided(state: DispatchState, local_output: torch.Tensor) -> None:
        del state
        moe_a2a.combine(
            local_output.view(world_size, tokens, args.hidden),
            tokens,
            output=output,
        )

    return Pipeline(
        dispatch=dispatch_one_sided,
        compute=compute,
        combine=combine_one_sided,
        output=output,
    )


def _make_two_sided(
    args: argparse.Namespace,
    tokens: int,
    payloads: tuple[torch.Tensor, ...],
    workspace: torch.Tensor,
    prepare_workspace: torch.Tensor,
    layer: MoELayer,
    weight_pack: MoEWeightPack,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Pipeline:
    hidden_states_q, hidden_states_scale, topk_ids, topk_weights = payloads
    output = torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device=device)

    def dispatch_two_sided() -> DispatchState:
        alltoall_info, local_ids, local_weights, _ = (
            MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                topk_ids,
                topk_weights,
                None,
                prepare_workspace,
                tokens,
                rank,
                world_size,
                args.num_experts,
                args.num_experts,
                args.top_k,
            )
        )
        recv_hidden = MnnvlMoe.mnnvl_moe_alltoallv(
            hidden_states_q, alltoall_info, workspace, rank, world_size
        )
        recv_scale = MnnvlMoe.mnnvl_moe_alltoallv(
            hidden_states_scale, alltoall_info, workspace, rank, world_size
        )
        activations = MoEActivationPack(
            hidden_states_q=recv_hidden,
            hidden_states_scale=recv_scale,
            topk_ids=local_ids,
            topk_weights=local_weights,
        )
        return DispatchState(activations=activations, context=alltoall_info)

    def compute(activations: MoEActivationPack) -> torch.Tensor:
        return layer(activations, weight_pack)

    def combine_two_sided(state: DispatchState, local_output: torch.Tensor) -> None:
        combined = MnnvlMoe.mnnvl_moe_alltoallv_combine(
            local_output,
            state.context,
            workspace,
            ep_rank=rank,
            ep_size=world_size,
            top_k=args.top_k,
            token_count=tokens,
        )
        output.copy_(combined)

    return Pipeline(
        dispatch=dispatch_two_sided,
        compute=compute,
        combine=combine_two_sided,
        output=output,
    )


def _make_agrs(
    args: argparse.Namespace,
    tokens: int,
    payloads: tuple[torch.Tensor, ...],
    layer: MoELayer,
    weight_pack: MoEWeightPack,
    world_size: int,
    device: torch.device,
) -> Pipeline:
    hidden_states_q, hidden_states_scale, topk_ids, topk_weights = payloads
    global_tokens = world_size * tokens
    gathered_hidden = torch.empty(
        (global_tokens, *hidden_states_q.shape[1:]),
        dtype=hidden_states_q.dtype,
        device=device,
    )
    scale_storage = hidden_states_scale.view(torch.uint8)
    gathered_scale_storage = torch.empty(
        (global_tokens, *scale_storage.shape[1:]),
        dtype=torch.uint8,
        device=device,
    )
    gathered_scale = gathered_scale_storage.view(hidden_states_scale.dtype)
    gathered_ids = torch.empty(
        (global_tokens, args.top_k),
        dtype=topk_ids.dtype,
        device=device,
    )
    gathered_weights = torch.empty(
        (global_tokens, args.top_k),
        dtype=topk_weights.dtype,
        device=device,
    )
    output = torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device=device)

    def dispatch_agrs() -> DispatchState:
        dist.all_gather_single(gathered_hidden, hidden_states_q)
        dist.all_gather_single(gathered_scale_storage, scale_storage)
        dist.all_gather_single(gathered_ids, topk_ids)
        dist.all_gather_single(gathered_weights, topk_weights)
        activations = MoEActivationPack(
            hidden_states_q=gathered_hidden,
            hidden_states_scale=gathered_scale,
            topk_ids=gathered_ids,
            topk_weights=gathered_weights,
        )
        return DispatchState(activations=activations)

    def compute(activations: MoEActivationPack) -> torch.Tensor:
        return layer(activations, weight_pack)

    def combine_agrs(state: DispatchState, local_output: torch.Tensor) -> None:
        del state
        dist.reduce_scatter_single(output, local_output, op=dist.ReduceOp.SUM)

    return Pipeline(
        dispatch=dispatch_agrs,
        compute=compute,
        combine=combine_agrs,
        output=output,
    )


def _run_pipeline(pipeline: Pipeline) -> None:
    state = pipeline.dispatch()
    local_output = pipeline.compute(state.activations)
    pipeline.combine(state, local_output)


def _tune_pipeline(pipeline: Pipeline, enable_autotune: bool) -> None:
    state = pipeline.dispatch()
    with autotune(enable_autotune):
        local_output = pipeline.compute(state.activations)
    pipeline.combine(state, local_output)


def _capture_pipeline(pipeline: Pipeline) -> CapturedPipeline:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_pipeline(pipeline)
    return CapturedPipeline(graph=graph)


def _prepare_aligned_sample(l2_flush: torch.Tensor | None) -> None:
    if l2_flush is not None:
        l2_flush.zero_()
    torch.cuda.synchronize()
    dist.barrier()


def _time_graph(
    captured: CapturedPipeline,
    *,
    warmup: int,
    iters: int,
    l2_flush: torch.Tensor | None,
) -> tuple[float, ...]:
    for _ in range(warmup):
        _prepare_aligned_sample(l2_flush)
        captured.graph.replay()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: tuple[float, ...] = ()
    for _ in range(iters):
        _prepare_aligned_sample(l2_flush)
        start.record()
        captured.graph.replay()
        stop.record()
        torch.cuda.synchronize()
        samples += (start.elapsed_time(stop) * 1e3,)
    dist.barrier()
    return samples


def _time_aligned_eager(
    pipeline: Pipeline,
    *,
    warmup: int,
    iters: int,
    l2_flush: torch.Tensor | None,
) -> RankSamples:
    for _ in range(warmup):
        _prepare_aligned_sample(l2_flush)
        _run_pipeline(pipeline)
        torch.cuda.synchronize()

    total_start = torch.cuda.Event(enable_timing=True)
    total_stop = torch.cuda.Event(enable_timing=True)
    dispatch_start = torch.cuda.Event(enable_timing=True)
    dispatch_stop = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_stop = torch.cuda.Event(enable_timing=True)
    combine_start = torch.cuda.Event(enable_timing=True)
    combine_stop = torch.cuda.Event(enable_timing=True)
    dispatch_us: tuple[float, ...] = ()
    compute_us: tuple[float, ...] = ()
    combine_us: tuple[float, ...] = ()
    e2e_us: tuple[float, ...] = ()

    for _ in range(iters):
        _prepare_aligned_sample(l2_flush)
        total_start.record()
        dispatch_start.record()
        state = pipeline.dispatch()
        dispatch_stop.record()
        compute_start.record()
        local_output = pipeline.compute(state.activations)
        compute_stop.record()
        combine_start.record()
        pipeline.combine(state, local_output)
        combine_stop.record()
        total_stop.record()
        torch.cuda.synchronize()
        dispatch_us += (dispatch_start.elapsed_time(dispatch_stop) * 1e3,)
        compute_us += (compute_start.elapsed_time(compute_stop) * 1e3,)
        combine_us += (combine_start.elapsed_time(combine_stop) * 1e3,)
        e2e_us += (total_start.elapsed_time(total_stop) * 1e3,)
    dist.barrier()

    return RankSamples(
        dispatch_us=dispatch_us,
        compute_us=compute_us,
        combine_us=combine_us,
        e2e_us=e2e_us,
    )


def _assert_strategy_outputs_close(
    actual: torch.Tensor, expected: torch.Tensor
) -> None:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    tiny = torch.finfo(torch.float32).tiny
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected_float).clamp_min(tiny)
    ).item()
    relative_max = (
        difference.abs().max() / expected_float.abs().max().clamp_min(tiny)
    ).item()
    if max(relative_l2, relative_max) > STRATEGY_RELATIVE_TOLERANCE:
        raise AssertionError(
            "split communication strategies produced different outputs: "
            f"relative_l2={relative_l2:.6g}, relative_max={relative_max:.6g}, "
            f"tolerance={STRATEGY_RELATIVE_TOLERANCE:.6g}"
        )


def _run_candidate(
    pipeline: Pipeline,
    args: argparse.Namespace,
    l2_flush: torch.Tensor | None,
    device: torch.device,
    reference_output: torch.Tensor | None,
) -> CandidateResult:
    world_size = dist.get_world_size()
    try:
        _tune_pipeline(pipeline, enable_autotune=not args.no_autotune)
        torch.cuda.synchronize()
        if not torch.isfinite(pipeline.output).all().item():
            raise RuntimeError("eager output contains non-finite values")
        if torch.count_nonzero(pipeline.output).item() == 0:
            raise RuntimeError("eager output is entirely zero")
        if reference_output is not None:
            _assert_strategy_outputs_close(pipeline.output, reference_output)
        eager_output = pipeline.output.clone()
        torch.cuda.synchronize()
        dist.barrier()

        captured = _capture_pipeline(pipeline)
        dist.barrier()
        pipeline.output.fill_(float("nan"))
        torch.cuda.synchronize()
        _prepare_aligned_sample(None)
        captured.graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(
            pipeline.output,
            eager_output,
            rtol=5e-2,
            atol=5e-2,
            msg="CUDA Graph replay diverged from eager output",
        )

        graph_e2e_samples = _time_graph(
            captured,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        eager_samples = _time_aligned_eager(
            pipeline,
            warmup=args.warmup,
            iters=args.iters,
            l2_flush=l2_flush,
        )
        if not all(
            torch.isfinite(torch.tensor(samples, device=device)).all().item()
            for samples in (
                graph_e2e_samples,
                eager_samples.dispatch_us,
                eager_samples.compute_us,
                eager_samples.combine_us,
                eager_samples.e2e_us,
            )
        ):
            raise RuntimeError("non-finite timing sample")
        local_result = ("pass", graph_e2e_samples, eager_samples, "")
    except Exception as error:  # noqa: BLE001
        local_result = (
            "failed",
            None,
            None,
            f"{type(error).__name__}: {error}",
        )

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    dist.barrier()
    if all(result[0] == "pass" for result in gathered):
        return CandidateResult(
            status="pass",
            graph_e2e_samples_by_rank=tuple(result[1] for result in gathered),
            eager_samples_by_rank=tuple(result[2] for result in gathered),
        )
    errors = "; ".join(
        f"rank{index}:{result[3]}" for index, result in enumerate(gathered) if result[3]
    )
    return CandidateResult(status="failed", error=errors)


def _critical_graph_samples(
    samples_by_rank: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    return tuple(max(samples) for samples in zip(*samples_by_rank, strict=True))


def _critical_eager_samples(
    samples_by_rank: tuple[RankSamples, ...],
) -> RankSamples:
    critical_ranks = tuple(
        max(
            range(len(samples_by_rank)),
            key=lambda rank: samples_by_rank[rank].e2e_us[iteration],
        )
        for iteration in range(len(samples_by_rank[0].e2e_us))
    )

    def select(field: str) -> tuple[float, ...]:
        return tuple(
            getattr(samples_by_rank[rank], field)[iteration]
            for iteration, rank in enumerate(critical_ranks)
        )

    return RankSamples(
        dispatch_us=select("dispatch_us"),
        compute_us=select("compute_us"),
        combine_us=select("combine_us"),
        e2e_us=select("e2e_us"),
    )


def _flops_per_rank(tokens: int, top_k: int, hidden: int, intermediate: int) -> int:
    return 6 * tokens * top_k * hidden * intermediate


def _candidate_row(
    args: argparse.Namespace,
    spec: CandidateSpec,
    world_size: int,
    result: CandidateResult,
) -> dict[str, object]:
    row: dict[str, object] = {
        "tokens_per_rank": spec.tokens,
        "global_tokens": spec.tokens * world_size,
        "world_size": world_size,
        "backend": spec.backend,
        "selected": False,
        "total_experts": args.num_experts,
        "local_experts": args.num_experts // world_size,
        "top_k": args.top_k,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "warmup": args.warmup,
        "iters": args.iters,
        "e2e_timing_mode": "aligned_cuda_graph",
        "stage_timing_mode": "aligned_eager",
        "routing": "per_token_ep_balanced",
        "l2_flush_mib": args.l2_flush_mib,
        "status": result.status,
        "error": result.error,
    }
    if result.status != "pass":
        return row

    critical_eager = _critical_eager_samples(result.eager_samples_by_rank)
    e2e_samples = _critical_graph_samples(result.graph_e2e_samples_by_rank)
    stage_sum_samples = tuple(
        dispatch + compute + combine
        for dispatch, compute, combine in zip(
            critical_eager.dispatch_us,
            critical_eager.compute_us,
            critical_eager.combine_us,
            strict=True,
        )
    )
    dispatch_median = median(critical_eager.dispatch_us)
    compute_median = median(critical_eager.compute_us)
    combine_median = median(critical_eager.combine_us)
    e2e_median = median(e2e_samples)
    flops = _flops_per_rank(spec.tokens, args.top_k, args.hidden, args.intermediate)
    per_gpu_tflops = flops / e2e_median / 1e6
    row.update(
        {
            "dispatch_critical_median_us": dispatch_median,
            "compute_critical_median_us": compute_median,
            "combine_critical_median_us": combine_median,
            "stage_sum_median_us": median(stage_sum_samples),
            "aligned_eager_e2e_critical_median_us": median(critical_eager.e2e_us),
            "e2e_critical_min_us": min(e2e_samples),
            "e2e_critical_median_us": e2e_median,
            "e2e_critical_max_us": max(e2e_samples),
            "per_gpu_tflops_e2e": per_gpu_tflops,
            "aggregate_tflops_e2e": per_gpu_tflops * world_size,
            "aggregate_tokens_per_s_e2e": (spec.tokens * world_size * 1e6 / e2e_median),
        }
    )
    return row


def _load_fused_latencies(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as input_file:
        return {
            int(row["tokens_per_rank"]): float(row["e2e_critical_median_us"])
            for row in csv.DictReader(input_file)
            if row["status"] == "pass"
        }


def _annotate_rows(
    rows: tuple[dict[str, object], ...], fused_latencies: dict[int, float]
) -> tuple[dict[str, object], ...]:
    selected_by_tokens: dict[int, dict[str, object]] = {}
    for row in rows:
        if row["status"] != "pass" or row["backend"] == "agrs":
            continue
        tokens = int(row["tokens_per_rank"])
        selected = selected_by_tokens.get(tokens)
        if selected is None or float(row["e2e_critical_median_us"]) < float(
            selected["e2e_critical_median_us"]
        ):
            selected_by_tokens[tokens] = row

    annotated: tuple[dict[str, object], ...] = ()
    for row in rows:
        updated = dict(row)
        tokens = int(row["tokens_per_rank"])
        updated["selected"] = selected_by_tokens.get(tokens) is row
        fused_latency = fused_latencies.get(tokens)
        if fused_latency is not None:
            updated["fused_e2e_critical_median_us"] = fused_latency
            if row["status"] == "pass":
                updated["unfused_over_fused"] = (
                    float(row["e2e_critical_median_us"]) / fused_latency
                )
        annotated += (updated,)
    return annotated


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_rows(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {column: _format_value(row.get(column, "")) for column in CSV_COLUMNS}
            for row in rows
        )


def main() -> int:
    args = _parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tokens_list = _validate_args(args, world_size)
    specs = _candidate_specs(tokens_list)
    if not specs:
        raise ValueError("the token list does not intersect either backend range")

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
    )
    mnnvl_config = MnnvlConfig(
        comm_backend=TorchDistBackend(dist.group.WORLD),
        fabric_page_size=1 << 29,
        allocation_granularity=0,
    )
    max_one_sided = max(
        (spec.tokens for spec in specs if spec.backend == "one_sided"), default=0
    )
    one_sided = None
    if max_one_sided:
        dispatch_bytes = args.hidden // 2 + args.hidden // 16 + args.top_k * 8
        combine_bytes = args.hidden * 2
        workspace_size = moe_a2a_get_workspace_size_per_rank(
            world_size,
            max_one_sided,
            dispatch_bytes,
            combine_bytes,
        )
        one_sided = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=max_one_sided,
            top_k=args.top_k,
            num_experts=args.num_experts,
            workspace_size_per_rank=workspace_size,
            mnnvl_config=mnnvl_config,
        )

    two_sided_workspace = None
    two_sided_prepare_workspace = None
    if any(spec.backend == "two_sided" for spec in specs):
        two_sided_workspace = MnnvlMoe.get_moe_workspaces(mapping, mnnvl_config)
        two_sided_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
            mapping, mnnvl_config
        )

    local_experts = args.num_experts // world_size
    weight_pack = _make_weight_pack(args, local_experts, rank, device)
    layer = _make_moe_layer(args, rank, world_size, max(tokens_list), device)
    l2_flush = (
        torch.empty(args.l2_flush_mib * 1024 * 1024, dtype=torch.uint8, device=device)
        if args.l2_flush_mib
        else None
    )
    dist.barrier()

    rows: tuple[dict[str, object], ...] = ()
    fused_latencies = _load_fused_latencies(args.fused_results) if rank == 0 else {}
    try:
        current_tokens = None
        payloads = None
        reference_output = None
        for spec in specs:
            if spec.tokens != current_tokens:
                del payloads
                gc.collect()
                torch.cuda.empty_cache()
                payloads = _make_inputs(args, spec.tokens, rank, world_size, device)
                current_tokens = spec.tokens
                reference_output = None
                dist.barrier()
            if rank == 0:
                print(
                    f"# tokens_per_rank={spec.tokens} backend={spec.backend}",
                    flush=True,
                )
            if spec.backend == "one_sided":
                if one_sided is None:
                    raise RuntimeError("one-sided workspace was not initialized")
                pipeline = _make_one_sided(
                    args,
                    spec.tokens,
                    payloads,
                    one_sided,
                    layer,
                    weight_pack,
                    world_size,
                    device,
                )
            elif spec.backend == "two_sided":
                if two_sided_workspace is None or two_sided_prepare_workspace is None:
                    raise RuntimeError("two-sided workspaces were not initialized")
                pipeline = _make_two_sided(
                    args,
                    spec.tokens,
                    payloads,
                    two_sided_workspace,
                    two_sided_prepare_workspace,
                    layer,
                    weight_pack,
                    rank,
                    world_size,
                    device,
                )
            else:
                pipeline = _make_agrs(
                    args,
                    spec.tokens,
                    payloads,
                    layer,
                    weight_pack,
                    world_size,
                    device,
                )
            result = _run_candidate(
                pipeline,
                args,
                l2_flush,
                device,
                reference_output,
            )
            if result.status == "pass" and reference_output is None:
                reference_output = pipeline.output.clone()
                torch.cuda.synchronize()
            if rank == 0:
                rows += (_candidate_row(args, spec, world_size, result),)
                annotated = _annotate_rows(rows, fused_latencies)
                if args.output is not None:
                    _write_rows(args.output, annotated)
                print(
                    ",".join(
                        _format_value(annotated[-1].get(column, ""))
                        for column in CSV_COLUMNS
                    ),
                    flush=True,
                )
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        annotated = _annotate_rows(rows, fused_latencies)
        if args.output is not None:
            _write_rows(args.output, annotated)
        print(",".join(CSV_COLUMNS), flush=True)
        for row in annotated:
            print(
                ",".join(_format_value(row.get(column, "")) for column in CSV_COLUMNS),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
