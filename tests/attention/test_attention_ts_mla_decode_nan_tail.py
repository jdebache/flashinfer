"""NaN/Inf in paged-KV slots outside ``[0, seq_len)`` must not leak into TS MLA decode output.

The task-scheduled MLA decode kernels load whole K/V pages.  Masked scores
become P == 0, but the P·V tensor-core multiply still evaluates 0 * NaN for V
rows past ``seq_len``.  The loaders keep those rows out of every TMA box by
shifting each page's token coordinate, so the output must be finite and
bit-identical to a run over a clean cache.
"""

import math

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

LATENT_DIM = 512
ROPE_DIM = 64


def _skip_if_unsupported():
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100/SM103")


def _torch_reference(
    query, kv_cache, block_tables, seq_lens, page_size, bmm1_scale, bmm2_scale
):
    """FP32 reference with the bottom-right causal mask."""
    batch_size, q_len = query.shape[:2]
    outputs = []
    for b in range(batch_size):
        seq_len = int(seq_lens[b])
        pages = block_tables[b, : math.ceil(seq_len / page_size)].tolist()
        kv = torch.cat([kv_cache[p] for p in pages], dim=0)[:seq_len].float()
        scores = torch.einsum("qhd,kd->qhk", query[b].float(), kv) * bmm1_scale
        key_pos = torch.arange(seq_len, device=query.device)
        last_visible = seq_len - q_len + torch.arange(q_len, device=query.device)
        visible = key_pos[None, :] <= last_visible[:, None]
        scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("qhk,kd->qhd", probs, kv[:, :LATENT_DIM]) * bmm2_scale
        )
    return torch.stack(outputs)


def _randn(shape, dtype, generator):
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


def _make_inputs(seq_lens, page_size, q_len, num_heads, dtype, seed):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = len(seq_lens)
    pages_per_request = math.ceil(max(seq_lens) / page_size)
    total_pages = batch_size * pages_per_request + 3
    query = _randn(
        (batch_size, q_len, num_heads, LATENT_DIM + ROPE_DIM), dtype, generator
    )
    kv_cache = _randn((total_pages, page_size, LATENT_DIM + ROPE_DIM), dtype, generator)
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
        for logical, page in enumerate(block_tables[b].tolist()):
            referenced.add(page)
            valid = min(max(seq_len - logical * page_size, 0), page_size)
            poisoned[page, valid:] = value
    for page in range(kv_cache.shape[0]):
        if page not in referenced:
            poisoned[page] = value
    return poisoned


def _plan(query, kv_cache, seq_lens, page_size, max_kv_len=None):
    """Plan the reusable wrapper and return it with the auto-selected kernel family."""
    from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper

    batch_size, q_len, num_heads = query.shape[:3]
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        query.device,
        batch_size,
        num_heads,
        LATENT_DIM,
        ROPE_DIM,
        page_size,
        max_kv_len if max_kv_len is not None else int(seq_lens.max()),
        max_seq_len_q=q_len,
        packed_query=False,
        q_data_type=query.dtype,
        kv_data_type=kv_cache.dtype,
        o_data_type=torch.bfloat16,
        mask_type="causal",
    )
    return wrapper, str(dict(wrapper._plan_state.policy).get("kernel"))


def _run(wrapper, query, kv_cache, block_tables, seq_lens, bmm1_scale, bmm2_scale):
    return wrapper.run(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )


def _check_poison_isolated(
    query,
    kv_cache,
    block_tables,
    seq_lens_t,
    page_size,
    dtype,
    poison_value,
    max_kv_len=None,
):
    bmm1_scale = 1.0 / math.sqrt(LATENT_DIM + ROPE_DIM)
    bmm2_scale = 1.0
    wrapper, kernel = _plan(query, kv_cache, seq_lens_t, page_size, max_kv_len)
    clean = _run(
        wrapper, query, kv_cache, block_tables, seq_lens_t, bmm1_scale, bmm2_scale
    )
    reference = _torch_reference(
        query, kv_cache, block_tables, seq_lens_t, page_size, bmm1_scale, bmm2_scale
    )
    torch.testing.assert_close(clean.float(), reference, **_tolerance(dtype))
    if dtype == torch.float8_e4m3fn and kernel == "throughput_2cta":
        pytest.xfail(
            "the FP8 split-MMA 2-CTA schedule keeps MlaDecodeConfig.kv_tail_shift off "
            "(its 32-register TMA producers cannot absorb the shift math); unused "
            "cache slots must stay finite for that kernel"
        )
    poisoned_cache = _poison(
        kv_cache, block_tables, seq_lens_t, page_size, poison_value
    )
    poisoned = _run(
        wrapper,
        query,
        poisoned_cache,
        block_tables,
        seq_lens_t,
        bmm1_scale,
        bmm2_scale,
    )
    assert torch.isfinite(poisoned).all()
    torch.testing.assert_close(poisoned, clean, atol=0, rtol=0)


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
@pytest.mark.parametrize("num_heads,q_len", [(128, 1), (8, 4), (16, 2)])
@pytest.mark.parametrize("poison_value", [float("nan"), float("inf")])
def test_ts_mla_tail_poison_is_isolated(
    dtype, page_size, num_heads, q_len, poison_value
):
    _skip_if_unsupported()
    seq_lens = tuple(max(length, q_len) for length in _tail_seq_lens(page_size))
    query, kv_cache, block_tables, seq_lens_t = _make_inputs(
        seq_lens, page_size, q_len, num_heads, dtype, seed=page_size * 10 + q_len
    )
    _check_poison_isolated(
        query, kv_cache, block_tables, seq_lens_t, page_size, dtype, poison_value
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize("page_size", [32, 128])
def test_ts_mla_tail_poison_split_kv(dtype, page_size):
    """One long request is split across CTAs; only the last split sees a tail."""
    _skip_if_unsupported()
    query, kv_cache, block_tables, seq_lens_t = _make_inputs(
        (16 * 1024 + 37,), page_size, 1, 128, dtype, seed=7
    )
    _check_poison_isolated(
        query, kv_cache, block_tables, seq_lens_t, page_size, dtype, float("nan")
    )


@pytest.mark.parametrize("page_size", [64])
def test_ts_mla_tail_poison_large_plan_bound(page_size):
    """A plan bound far above the runtime lengths leaves whole pages past every request."""
    _skip_if_unsupported()
    dtype = torch.bfloat16
    query, kv_cache, block_tables, seq_lens_t = _make_inputs(
        (46, 130, 5, 257), page_size, 4, 8, dtype, seed=3
    )
    _check_poison_isolated(
        query,
        kv_cache,
        block_tables,
        seq_lens_t,
        page_size,
        dtype,
        float("nan"),
        max_kv_len=int(block_tables.shape[1] * page_size),
    )
