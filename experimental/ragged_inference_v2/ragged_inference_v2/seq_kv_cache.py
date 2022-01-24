# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import math
from functools import lru_cache
from typing import List, Tuple

import torch
from ragged_inference_v2.garbage_pad_ragged_acts import RaggedActivations
from ragged_inference_v2.test_utils import assert_eq, bf16_cuda

# Whenever we need to grow the tensor, we extend by this amount extra so that we don't
# have to do expensive memcopies as frequently as we would otherwise
EXTEND_GROWTH_FACTOR = 1.3

MIN_N_CTX = 64


class SingleSeqKVCache:
    def __init__(self):
        # Tensor of shape [n_ctx, n_heads, n_dim]
        # - keys are cache[0]
        # - values are cache[1]
        self._raw_keys: torch.Tensor = None
        self._raw_values: torch.Tensor = None

        self._n_ctx = 0

    @property
    def keys(self) -> torch.Tensor:
        return self._raw_keys[: self.n_ctx]

    @property
    def values(self) -> torch.Tensor:
        return self._raw_values[: self.n_ctx]

    @property
    def n_ctx(self):
        return self._n_ctx

    @property
    def is_empty(self):
        return self._raw_keys is None

    @property
    def is_cuda(self):
        return self._raw_values.is_cuda

    @property
    def dtype(self):
        return self._raw_values.dtype

    def extend_in_place(self, new_keys: torch.Tensor, new_values: torch.Tensor):

        assert_eq(new_keys.ndim, 3)
        assert_eq(new_values.ndim, 3)
        n_ctx_from_new_keys, n_heads, d_head = new_keys.shape

        old_n_ctx = self.n_ctx
        new_n_ctx = n_ctx_from_new_keys + old_n_ctx
        minimum_n_ctx = max(new_n_ctx, MIN_N_CTX)
        n_ctx_after_grow = math.ceil(EXTEND_GROWTH_FACTOR * minimum_n_ctx)

        old_raw_keys = self._raw_keys
        old_raw_values = self._raw_values

        kwargs = dict(device=new_keys.device, dtype=new_keys.dtype)
        self._raw_keys = torch.empty(n_ctx_after_grow, n_heads, d_head, **kwargs)
        self._raw_values = torch.empty(n_ctx_after_grow, n_heads, d_head, **kwargs)

        if old_raw_keys is not None:
            self._raw_keys[:old_n_ctx] = old_raw_keys[:old_n_ctx]
            self._raw_values[:old_n_ctx] = old_raw_values[:old_n_ctx]

        self._raw_keys[old_n_ctx:new_n_ctx] = new_keys
        self._raw_values[old_n_ctx:new_n_ctx] = new_values
        self._n_ctx = new_n_ctx


def _new_kvs(n_ctx, value, n_heads, d_per_head) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.full([n_ctx, n_heads, d_per_head], value, **bf16_cuda())
    values = torch.full([n_ctx, n_heads, d_per_head], value, **bf16_cuda())
    return (keys, values)


def extend_kv_caches_in_place(
    seq_kv_caches: List[SingleSeqKVCache],
    active_keys: RaggedActivations,
    active_values: RaggedActivations,
) -> None:
    for cache, keys, values in zip(
        seq_kv_caches,
        active_keys.iter_full_tensors(),
        active_values.iter_full_tensors(),
    ):
        cache.extend_in_place(keys, values)


def garbage_pad_seq_kv_cache(
    seq_kv_cache: List[SingleSeqKVCache],
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert seq_kv_cache[0].is_cuda
    dtype = seq_kv_cache[0].dtype
    n_ctx_per_kv_cache = [seq.n_ctx for seq in seq_kv_cache]
    assert_eq(seq_kv_cache.raw_tensor.ndim, 3)

    # Create a view so that the output is (n_seqs, n_ctx_max, d_model)
    # This should not incur an extra memcopy
    n_seqs = len(n_ctx_per_kv_cache)
    n_ctx_max = max(n_ctx_per_kv_cache)

    padded_keys = torch.empty(
        n_seqs,
        n_ctx_max,
        seq_kv_cache[0].d_model_per_gpu,
        dtype=dtype,
        device="cuda",
    )

    padded_values = torch.empty(
        n_seqs,
        n_ctx_max,
        seq_kv_cache[0].d_model_per_gpu,
        dtype=dtype,
        device="cuda",
    )

    for seq_idx, seq in enumerate(seq_kv_cache):
        padded_keys[seq_idx, seq.n_ctx, :, :] = seq.keys
        padded_values[seq_idx, seq.n_ctx, :, :] = seq.values
    return (padded_keys, padded_values)


def garbage_pad_keys(
    seq_kv_cache: List[SingleSeqKVCache],
    n_heads: int,
    d_head: int,
) -> torch.Tensor:
    single_seq_kv_cache = seq_kv_cache[0]
    assert single_seq_kv_cache.is_cuda
    dtype = single_seq_kv_cache.dtype
    n_ctx_per_kv_cache = [seq.n_ctx for seq in seq_kv_cache]

    # Create a view so that the output is (n_seqs, n_ctx_max, d_model)
    # This should not incur an extra memcopy
    n_seqs = len(n_ctx_per_kv_cache)
    n_ctx_max = max(n_ctx_per_kv_cache)

    padded_keys = torch.empty(
        n_seqs,
        n_ctx_max,
        n_heads,
        d_head,
        dtype=dtype,
        device="cuda",
    )

    for seq_idx, seq in enumerate(seq_kv_cache):
        padded_keys[seq_idx, : seq.n_ctx, :, :] = seq.keys
    return padded_keys


@lru_cache(maxsize=1)  # Memoize because we repeat this for consecutive resblocks
def _create_indices(n_ctx_per_kv_cache):
    """
    We cache this because it requires some substantial CPU work and it's done multiple
    times sequentially (once per resblock)
    """
    indices_list = []
    ragged_idx = 0
    max_n_ctx = max(n_ctx_per_kv_cache)
    for n_ctx in n_ctx_per_kv_cache:
        for idx_into_seq in range(max_n_ctx):
            if idx_into_seq < n_ctx:
                indices_list.append(ragged_idx)
                ragged_idx += 1
            else:
                indices_list.append(0)  # Add a placeholder
    return torch.tensor(indices_list, device="cuda")


def calculate_scores_via_qk_dotprod(
    seq_kv_cache: List[SingleSeqKVCache],  # These have already been extended
    active_queries: RaggedActivations,
) -> torch.Tensor:
    padded_keys = garbage_pad_keys(
        seq_kv_cache,
        n_heads=active_queries.raw_tensor.shape[1],
        d_head=active_queries.raw_tensor.shape[2],
    )
    padded_active_queries = active_queries.to_garbage_padded()
    return torch.einsum("skhd,sqhd->sqhk", padded_keys, padded_active_queries)


def scores_via_qk_dotprod(
    query: RaggedActivations,
    key: RaggedActivations,
) -> torch.Tensor:
    padded_query = query.to_garbage_padded()
    padded_key = key.to_garbage_padded()
    return torch.einsum("skhd,sqhd->hsqk", padded_key, padded_query)
