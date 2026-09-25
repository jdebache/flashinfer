"""NaN/Inf in paged-KV slots outside ``[0, seq_len)`` must not leak into MLA decode output.

The monolithic CuTe-DSL MLA decode kernel loads whole 128-token tiles.  Masked
scores become P == 0, but the P·V tensor-core multiply still evaluates 0 * NaN
for V rows past ``seq_len``.  The kernel keeps those rows out of every TMA box by
shifting each page's token coordinate, so the output must be finite and
bit-identical to a run over a clean cache.
"""

import math

import pytest
import torch

from flashinfer.utils import is_sm100a_supported, is_sm110a_supported

LATENT_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 128


def _skip_if_unsupported():
    device = torch.device("cuda")
    if not (is_sm100a_supported(device) or is_sm110a_supported(device)):
        pytest.skip("Requires SM100-SM110 (tcgen05)")


def _torch_reference(query, kv_cache, block_tables, seq_lens, page_size, softmax_scale):
    """FP32 reference with the monolithic kernel's MTP causal mask."""
    batch_size, q_len = query.shape[:2]
    outputs = []
    for b in range(batch_size):
        seq_len = int(seq_lens[b])
        pages = block_tables[b, : math.ceil(seq_len / page_size)].tolist()
        kv = torch.cat([kv_cache[p] for p in pages], dim=0)[:seq_len].float()
        scores = torch.einsum("qhd,kd->qhk", query[b].float(), kv) * softmax_scale
        key_pos = torch.arange(seq_len, device=query.device)
        last_visible = seq_len - q_len + torch.arange(q_len, device=query.device)
        visible = key_pos[None, :] <= last_visible[:, None]  # [q_len, seq_len]
        scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qhk,kd->qhd", probs, kv[:, :LATENT_DIM]))
    return torch.stack(outputs)


def _randn(shape, dtype, generator):
    """FP8 tensors are drawn in fp16 and scaled down before conversion."""
    if dtype == torch.float8_e4m3fn:
        values = torch.randn(
            *shape, dtype=torch.float16, device="cuda", generator=generator
        )
        return (values * 0.1).to(dtype)
    return torch.randn(*shape, dtype=dtype, device="cuda", generator=generator)


def _tolerance(dtype):
    return (
        dict(atol=0.1, rtol=0.1)
        if dtype == torch.float8_e4m3fn
        else dict(atol=2e-2, rtol=2e-2)
    )


def _make_inputs(seq_lens, page_size, q_len, dtype, seed):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = len(seq_lens)
    pages_per_request = math.ceil(max(seq_lens) / page_size)
    total_pages = batch_size * pages_per_request + 3
    query = _randn(
        (batch_size, q_len, NUM_HEADS, LATENT_DIM + ROPE_DIM), dtype, generator
    )
    kv_cache = _randn((total_pages, page_size, LATENT_DIM + ROPE_DIM), dtype, generator)
    # Reverse page order so physical page ids differ from logical order.
    block_tables = (
        torch.arange(total_pages - 3, device=device, dtype=torch.int32)
        .flip(0)
        .view(batch_size, pages_per_request)
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return query, kv_cache, block_tables, seq_lens_t


def _poison(kv_cache, block_tables, seq_lens, page_size, value):
    """Fill every slot outside each request's ``[0, seq_len)`` with ``value``."""
    poisoned = kv_cache.clone()
    referenced = set()
    for b, seq_len in enumerate(seq_lens.tolist()):
        row = block_tables[b].tolist()
        for logical, page in enumerate(row):
            referenced.add(page)
            start = logical * page_size
            valid = min(max(seq_len - start, 0), page_size)
            poisoned[page, valid:] = value
    for page in range(kv_cache.shape[0]):
        if page not in referenced:
            poisoned[page] = value
    return poisoned


def _run(query, kv_cache, block_tables, seq_lens, softmax_scale, **overrides):
    from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

    kwargs = dict(
        query=query,
        kv_cache=kv_cache,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        softmax_scale=softmax_scale,
        output_scale=1.0,
        is_var_seq=True,
        cute_dsl_impl="monolithic",
    )
    kwargs.update(overrides)
    # Host reads of seq_lens are not allowed under CUDA-graph capture; graph
    # callers pass max_seq_len and workspace_buffer explicitly.
    if "max_seq_len" not in kwargs:
        kwargs["max_seq_len"] = int(seq_lens.max())
    if "workspace_buffer" not in kwargs:
        kwargs["workspace_buffer"] = torch.zeros(
            256 << 20, dtype=torch.int8, device=query.device
        )
    return cute_dsl_mla_decode(**kwargs)


def _tail_seq_lens(page_size):
    return (
        1,
        page_size - 1,
        page_size,
        page_size + 1,
        127,
        128,
        129,
        3 * page_size + 5,
        1000,
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
@pytest.mark.parametrize("q_len", [1, 2, 4])
@pytest.mark.parametrize("poison_value", [float("nan"), float("inf")])
def test_monolithic_tail_poison_is_isolated(dtype, page_size, q_len, poison_value):
    _skip_if_unsupported()
    seq_lens = tuple(max(length, q_len) for length in _tail_seq_lens(page_size))
    query, kv_cache, block_tables, seq_lens_t = _make_inputs(
        seq_lens, page_size, q_len, dtype, seed=page_size * 10 + q_len
    )
    softmax_scale = 1.0 / math.sqrt(LATENT_DIM)

    clean = _run(query, kv_cache, block_tables, seq_lens_t, softmax_scale)
    reference = _torch_reference(
        query, kv_cache, block_tables, seq_lens_t, page_size, softmax_scale
    )
    torch.testing.assert_close(clean.float(), reference, **_tolerance(dtype))

    poisoned_cache = _poison(
        kv_cache, block_tables, seq_lens_t, page_size, poison_value
    )
    poisoned = _run(query, poisoned_cache, block_tables, seq_lens_t, softmax_scale)
    assert torch.isfinite(poisoned).all()
    torch.testing.assert_close(poisoned, clean, atol=0, rtol=0)


@pytest.mark.parametrize("page_size", [32, 128])
def test_monolithic_tail_poison_split_kv(page_size):
    """A single long request is split across CTAs; only the last split sees a tail."""
    _skip_if_unsupported()
    dtype = torch.bfloat16
    seq_lens = (16 * 1024 + 37,)
    query, kv_cache, block_tables, seq_lens_t = _make_inputs(
        seq_lens, page_size, 1, dtype, seed=7
    )
    softmax_scale = 1.0 / math.sqrt(LATENT_DIM)
    clean = _run(query, kv_cache, block_tables, seq_lens_t, softmax_scale)
    reference = _torch_reference(
        query, kv_cache, block_tables, seq_lens_t, page_size, softmax_scale
    )
    torch.testing.assert_close(clean.float(), reference, atol=2e-2, rtol=2e-2)
    poisoned_cache = _poison(
        kv_cache, block_tables, seq_lens_t, page_size, float("nan")
    )
    poisoned = _run(query, poisoned_cache, block_tables, seq_lens_t, softmax_scale)
    assert torch.isfinite(poisoned).all()
    torch.testing.assert_close(poisoned, clean, atol=0, rtol=0)


def test_monolithic_tail_poison_cuda_graph_replay():
    """Capture once, then change seq_lens and block tables between replays."""
    _skip_if_unsupported()
    dtype = torch.bfloat16
    page_size, q_len = 64, 2
    seq_lens_a = (46, 130, 64, 257)
    seq_lens_b = (257, 46, 130, 3)
    query, kv_cache, block_tables, seq_lens_t = _make_inputs(
        seq_lens_a, page_size, q_len, dtype, seed=11
    )
    softmax_scale = 1.0 / math.sqrt(LATENT_DIM)
    workspace = torch.zeros(256 << 20, dtype=torch.int8, device=query.device)
    max_seq_len = max(max(seq_lens_a), max(seq_lens_b))

    def launch(cache, out):
        return _run(
            query,
            cache,
            block_tables,
            seq_lens_t,
            softmax_scale,
            workspace_buffer=workspace,
            max_seq_len=max_seq_len,
            out=out,
        )

    poisoned_cache = _poison(
        kv_cache, block_tables, seq_lens_t, page_size, float("nan")
    )
    out = torch.empty(
        query.shape[0], q_len, NUM_HEADS, LATENT_DIM, dtype=dtype, device=query.device
    )
    launch(poisoned_cache, out)  # warm up / compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(poisoned_cache, out)

    for seq_lens in (seq_lens_a, seq_lens_b):
        seq_lens_t.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=query.device))
        block_tables.copy_(block_tables.roll(1, dims=0))
        poisoned_cache.copy_(
            _poison(kv_cache, block_tables, seq_lens_t, page_size, float("nan"))
        )
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        reference = _torch_reference(
            query, kv_cache, block_tables, seq_lens_t, page_size, softmax_scale
        )
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)
