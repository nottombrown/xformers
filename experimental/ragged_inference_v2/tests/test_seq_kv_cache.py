# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import time

import torch
from ragged_inference_v2.garbage_pad_ragged_acts import RaggedActivations
from ragged_inference_v2.seq_kv_cache import (
    LocalSingleSeqKVCache,
    SingleSeqKVCache,
    _new_kvs,
    calculate_scores_via_qk_dotprod,
    extend_kv_caches_in_place,
)
from ragged_inference_v2.test_utils import assert_eq, bf16_cuda


def test_extend_kv_caches_correctness():
    d_head = 6
    n_heads = 2
    n_seqs = 3
    seq_kv_caches = [SingleSeqKVCache() for _ in range(3)]
    kwargs = dict(n_heads=n_heads, d_per_head=d_head)

    seq_kv_caches[0].extend_and_return_all(*_new_kvs(n_ctx=1, value=33, **kwargs))
    seq_kv_caches[1].extend_and_return_all(*_new_kvs(n_ctx=3, value=42, **kwargs))
    seq_kv_caches[2].extend_and_return_all(*_new_kvs(n_ctx=7, value=55, **kwargs))

    n_ctx_new = 1
    active_keys = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()),
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()),
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()),
        ]
    )
    active_values = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()) * 2,
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()) * 2,
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()) * 2,
        ]
    )

    extend_kv_caches_in_place(seq_kv_caches, active_keys, active_values)

    assert_eq(seq_kv_caches[0].keys[:, 0, 0].cpu(), [33, 1])
    assert_eq(seq_kv_caches[0].values[:, 0, 0].cpu(), [33, 2])

    assert_eq(seq_kv_caches[1].keys[:, 0, 0].cpu(), [42, 42, 42, 1])
    assert_eq(seq_kv_caches[1].values[:, 0, 0].cpu(), [42, 42, 42, 2])

    assert_eq(seq_kv_caches[2].keys[:, 0, 0].cpu(), [55, 55, 55, 55, 55, 55, 55, 1])
    assert_eq(seq_kv_caches[2].values[:, 0, 0].cpu(), [55, 55, 55, 55, 55, 55, 55, 2])


def test_local_kv_caches_extend_and_return_all_correctness():
    d_head = 6
    n_heads = 2
    n_seqs = 3
    local_ctx = 4
    seq_kv_caches = [LocalSingleSeqKVCache(local_ctx=local_ctx) for _ in range(n_seqs)]
    kwargs = dict(n_heads=n_heads, d_per_head=d_head)

    seq_kv_caches[0].extend_and_return_all(*_new_kvs(n_ctx=1, value=33, **kwargs))
    seq_kv_caches[1].extend_and_return_all(*_new_kvs(n_ctx=3, value=42, **kwargs))
    seq_kv_caches[2].extend_and_return_all(*_new_kvs(n_ctx=7, value=55, **kwargs))

    assert_eq(seq_kv_caches[0].n_ctx, 1)
    assert_eq(seq_kv_caches[0].keys[:, 0, 0].cpu(), [33])
    assert_eq(seq_kv_caches[0].values[:, 0, 0].cpu(), [33])

    assert_eq(seq_kv_caches[1].n_ctx, 3)
    assert_eq(seq_kv_caches[1].keys[:, 0, 0].cpu(), [42, 42, 42])
    assert_eq(seq_kv_caches[1].values[:, 0, 0].cpu(), [42, 42, 42])

    assert_eq(seq_kv_caches[2].n_ctx, 4)
    assert_eq(seq_kv_caches[2].keys[:, 0, 0].cpu(), [55, 55, 55, 55])
    assert_eq(seq_kv_caches[2].values[:, 0, 0].cpu(), [55, 55, 55, 55])

    n_ctx_new = 1
    active_keys = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()),
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()),
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()),
        ]
    )
    active_values = RaggedActivations.from_list(
        [
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()) * 2,
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()) * 2,
            torch.ones(n_ctx_new, n_heads, d_head, **bf16_cuda()) * 2,
        ]
    )

    extend_kv_caches_in_place(seq_kv_caches, active_keys, active_values)

    assert_eq(seq_kv_caches[0].keys[:, 0, 0].cpu(), [33, 1])
    assert_eq(seq_kv_caches[0].values[:, 0, 0].cpu(), [33, 2])

    assert_eq(seq_kv_caches[1].keys[:, 0, 0].cpu(), [42, 42, 42, 1])
    assert_eq(seq_kv_caches[1].values[:, 0, 0].cpu(), [42, 42, 42, 2])

    assert_eq(seq_kv_caches[2].keys[:, 0, 0].cpu(), [55, 55, 55, 1])
    assert_eq(seq_kv_caches[2].values[:, 0, 0].cpu(), [55, 55, 55, 2])


def test_calculate_scores_via_qk_dotprod_throughput(
    n_key_ctx_per_seq=1024, n_active_query_ctx_per_seq=5
):
    n_seqs = 100
    d_per_head = 256
    n_heads = 6
    seq_kv_cache = [SingleSeqKVCache() for _ in range(n_seqs)]

    for cache in seq_kv_cache:
        cache.extend_and_return_all(
            *_new_kvs(
                n_ctx=n_key_ctx_per_seq,
                value=42,
                n_heads=n_heads,
                d_per_head=d_per_head,
            )
        )

    active_queries = RaggedActivations.from_list(
        [
            torch.ones(n_active_query_ctx_per_seq, n_heads, d_per_head, **bf16_cuda())
            * 2
            for _ in range(n_seqs)
        ]
    )
    assert n_key_ctx_per_seq > n_active_query_ctx_per_seq * 10, (
        "n_active_query_ctx_per_seq must be much larger than "
        "n_key_ctx_per_seq for our simulator to be useful because "
        "we round the HBM memory bandwidth for the active_queries and "
        "for the scores down to zero"
    )

    bytes_in_keys_per_seq = n_key_ctx_per_seq * n_heads * d_per_head * 2  # 2 from bf16
    bytes_in_keys_total = bytes_in_keys_per_seq * n_seqs
    hbm_bw_bytes_per_gpu = 1555e9  # 1.5TB/s

    # If we just read the bytes directly from memory
    theor_load_micros_per_seq = bytes_in_keys_per_seq / hbm_bw_bytes_per_gpu * 1e6

    # Doing our operation should be slower than the theoretical minimum because we
    # do the following to the items
    #
    # 1. Read them from the per-seq areas
    # 2. Write them back into the buffer
    expected_micros_per_seq = theor_load_micros_per_seq * 2

    # warmup
    calculate_scores_via_qk_dotprod(seq_kv_cache, active_queries)

    torch.cuda.synchronize()
    started_at = time.time()
    n_iters = 10
    for _ in range(n_iters):
        calculate_scores_via_qk_dotprod(seq_kv_cache, active_queries)

    torch.cuda.synchronize()
    elapsed_micros = (time.time() - started_at) * 1e6

    micros_per_mb = elapsed_micros / n_iters
    micros_per_seq = micros_per_mb / n_seqs
    print(
        f"""
# Theoretical
{bytes_in_keys_total/1e9=:.3f}GB
{bytes_in_keys_per_seq/1e6=:.2f}MB
{theor_load_micros_per_seq=:.1f}µs per seq (to just load once from memory)
{expected_micros_per_seq=:.1f}µs per seq

# Actual
{micros_per_mb=:.1f}µs per microbatch
{micros_per_seq=:.1f}µs per seq

{micros_per_seq/expected_micros_per_seq:.1f}x the expected HBM-bandwidth bound time
"""
    )


"""
# Run tests with the following
pytest -vsx tests/test_seq_kv_cache.py


# Profile with the following
pytest -vsx tests/test_seq_kv_cache.py -k test_calculate_scores_via_qk_dotprod_throughput

"""

"""
pytest -vsx tests/test_seq_kv_cache.py
"""
