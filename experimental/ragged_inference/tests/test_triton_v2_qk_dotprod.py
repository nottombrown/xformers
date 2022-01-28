# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import time

import pytest
import torch

from ragged_inference.test_utils import assert_eq, bf16_cuda
from ragged_inference.triton_v2_qk_dotprod import qk_dotprod
from ragged_inference.triton_v2_ragged_qk_dotprod import (
    RaggedQkPidLookupTable,
    ragged_single_seq_qk_dotprod,
)


def _make_seq(n_ctx: int, value: int, d_model: int):
    return torch.full([n_ctx, d_model], value, **bf16_cuda())


SHAPES = [
    (3, 7),
    (384, 128),
    (784, 512),
    (1024, 1024),
    (2048, 384),
]


def qk_dotprod_pytorch(q, k):
    # attention matrix
    return torch.einsum("bqd,bkd->bqk", q, k)


def qk_dotprod_single_head_pytorch(q, k):
    # attention matrix
    return torch.einsum("qd,kd->qk", q, k)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_qk_dotprod(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda")

    out = qk_dotprod(a, b)

    torch_out = qk_dotprod_single_head_pytorch(a, b)
    assert_eq(out, torch_out, rtol=0.01, atol=0.2)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ragged_qk_dotprod(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda")

    lut = RaggedQkPidLookupTable.from_single_seq_and_head(
        n_ctx_q=shape[0], n_ctx_k=shape[0]
    )
    out = ragged_single_seq_qk_dotprod(a, b, lut)

    torch_out = qk_dotprod_single_head_pytorch(a, b)
    assert_eq(out, torch_out, rtol=0.01, atol=0.2)


def test_ragged_qk_dotprod_perf():
    active_tokens = 5
    d_head = 256
    active_and_cached_tokens = 8000 * 50
    n_iters = 10

    q = torch.randn((active_tokens, d_head), dtype=torch.bfloat16, device="cuda")
    k = torch.randn(
        (active_and_cached_tokens, d_head), dtype=torch.bfloat16, device="cuda"
    )

    lut = RaggedQkPidLookupTable.from_single_seq_and_head(
        n_ctx_q=active_tokens, n_ctx_k=active_and_cached_tokens
    )

    for _ in range(3):
        ragged_single_seq_qk_dotprod(q, k, lut)

    torch.cuda.synchronize()
    started_at = time.time()
    for _ in range(n_iters):
        ragged_single_seq_qk_dotprod(q, k, lut)
    torch.cuda.synchronize()

    elapsed_micros = (time.time() - started_at) * 1e6

    bytes_in_keys_per_seq = active_and_cached_tokens * d_head * 2  # 2 from bf16
    bytes_in_keys_total = bytes_in_keys_per_seq
    hbm_bw_bytes_per_gpu = 1555e9  # 1.5TB/s

    # If we just read the bytes directly from memory
    theor_load_micros_per_seq = bytes_in_keys_per_seq / hbm_bw_bytes_per_gpu * 1e6

    expected_micros_per_seq = theor_load_micros_per_seq

    micros_per_seq = elapsed_micros / n_iters
    print(
        f"""
# Theoretical
{bytes_in_keys_total/1e9=:.3f}GB
{bytes_in_keys_per_seq/1e6=:.2f}MB
{theor_load_micros_per_seq=:.1f}µs per seq (to just load once from memory)
{expected_micros_per_seq=:.1f}µs per seq

# Actual
{micros_per_seq=:.1f}µs per seq

{micros_per_seq/expected_micros_per_seq:.1f}x the expected HBM-bandwidth bound time
"""
    )
    assert micros_per_seq / expected_micros_per_seq < 1.5


def test_simple_qk_dotprod():
    dtype = torch.float32
    shape = (8, 8)

    # a = torch.zeros(shape, dtype=dtype, device="cuda")
    # a[0,0] = 1.0
    # b = torch.randn(shape, dtype=dtype, device="cuda")

    k = torch.zeros(shape, dtype=dtype, device="cuda")
    k[0, 0] = 1.0
    k[0, 1] = 1.0
    q = torch.randn(shape, dtype=dtype, device="cuda")

    print(f"{q=}")
    print(f"{k=}")
    out = qk_dotprod(q, k)

    torch_out = qk_dotprod_single_head_pytorch(q, k)
    assert_eq(out, torch_out, rtol=0.01, atol=0.2)


"""
pytest -vxs --tb=native tests/test_triton_v2_qk_dotprod.py -k test_ragged_qk_dotprod
"""
