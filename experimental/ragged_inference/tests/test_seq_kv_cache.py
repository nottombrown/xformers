# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
from ragged_inference.garbage_pad_ragged_acts import RaggedActivations
from ragged_inference.seq_kv_cache import (
    LocalSingleSeqKVCache,
    SingleSeqKVCache,
    _new_kvs,
    extend_kv_caches_in_place,
)
from ragged_inference.test_utils import assert_eq, bf16_cuda


def test_extend_kv_caches_correctness():
    d_head = 6
    n_heads = 2
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

    assert_eq(seq_kv_caches[0].n_ctx_in_buffer, 1)
    assert_eq(seq_kv_caches[0].n_ctx, 1)
    assert_eq(seq_kv_caches[0].keys[:, 0, 0].cpu(), [33])
    assert_eq(seq_kv_caches[0].values[:, 0, 0].cpu(), [33])

    assert_eq(seq_kv_caches[1].n_ctx_in_buffer, 3)
    assert_eq(seq_kv_caches[1].n_ctx, 3)
    assert_eq(seq_kv_caches[1].keys[:, 0, 0].cpu(), [42, 42, 42])
    assert_eq(seq_kv_caches[1].values[:, 0, 0].cpu(), [42, 42, 42])

    assert_eq(seq_kv_caches[2].n_ctx_in_buffer, 4)
    assert_eq(seq_kv_caches[2].n_ctx, 7)
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

    assert_eq(seq_kv_caches[0].n_ctx_in_buffer, 2)
    assert_eq(seq_kv_caches[0].n_ctx, 2)
    assert_eq(seq_kv_caches[0].keys[:, 0, 0].cpu(), [33, 1])
    assert_eq(seq_kv_caches[0].values[:, 0, 0].cpu(), [33, 2])

    assert_eq(seq_kv_caches[1].n_ctx_in_buffer, 4)
    assert_eq(seq_kv_caches[1].n_ctx, 4)
    assert_eq(seq_kv_caches[1].keys[:, 0, 0].cpu(), [42, 42, 42, 1])
    assert_eq(seq_kv_caches[1].values[:, 0, 0].cpu(), [42, 42, 42, 2])

    assert_eq(seq_kv_caches[2].n_ctx_in_buffer, 4)
    assert_eq(seq_kv_caches[2].n_ctx, 8)
    assert_eq(seq_kv_caches[2].keys[:, 0, 0].cpu(), [55, 55, 55, 1])
    assert_eq(seq_kv_caches[2].values[:, 0, 0].cpu(), [55, 55, 55, 2])


"""
# Run tests with the following
pytest -vsx tests/test_seq_kv_cache.py


# Profile with the following
pytest -vsx tests/test_seq_kv_cache.py -k test_calculate_scores_via_qk_dotprod_throughput

"""

"""
pytest -vsx tests/test_seq_kv_cache.py
"""
