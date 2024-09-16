# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import List, Tuple

import torch
from ragged_inference.garbage_pad_ragged_acts import RaggedActivations
from ragged_inference.test_utils import assert_eq, bf16_cuda

# Whenever we need to grow the tensor, we extend by this amount extra so that we don't
# have to do expensive memcopies as frequently as we would otherwise
EXTEND_GROWTH_FACTOR = 1.1

MIN_N_CTX = 32


class SingleSeqKVCache:
    def __init__(self):
        # Tensor of shape [n_ctx, n_heads, n_dim]
        # - keys are cache[0]
        # - values are cache[1]
        self._raw_keys: torch.Tensor = None
        self._raw_values: torch.Tensor = None

        self._n_ctx_total_seen = 0  # Total number of tokens appended

    @property
    def keys(self) -> torch.Tensor:
        return self._raw_keys[: self.n_ctx]

    @property
    def values(self) -> torch.Tensor:
        return self._raw_values[: self.n_ctx]

    @property
    def n_ctx(self):
        """
        Total number of tokens ever appended. Not the raw_tensor. If it's local for example,
        this could be more than the raw_tensor
        """
        return self._n_ctx_total_seen

    @property
    def n_ctx_in_buffer(self):
        """Physical amount that exists in the buffer"""
        return self._n_ctx_total_seen

    @property
    def is_empty(self):
        return self._raw_keys is None

    @property
    def is_cuda(self):
        return self._raw_values is None or self._raw_values.is_cuda

    @property
    def dtype(self):
        return self._raw_values.dtype

    def to_gpu(self):
        if self._raw_keys is not None:
            self._raw_keys.cuda()
        if self._raw_values is not None:
            self._raw_values.cuda()

    def to_cpu(self):
        if self._raw_keys is not None:
            self._raw_keys.cpu()
        if self._raw_values is not None:
            self._raw_values.cpu()

    @property
    def _buffer_size(self):
        return 0 if self.is_empty else self._raw_values.shape[0]

    def extend_and_return_all(
        self, new_keys: torch.Tensor, new_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert_eq(new_keys.ndim, 3)
        assert_eq(new_values.ndim, 3)
        n_ctx_from_new_keys, n_heads, d_head = new_keys.shape

        old_n_ctx = self.n_ctx
        new_n_ctx = n_ctx_from_new_keys + old_n_ctx
        target_new_n_ctx = max(new_n_ctx, MIN_N_CTX)

        should_grow = target_new_n_ctx > self._buffer_size
        if should_grow:
            n_ctx_after_grow = math.ceil(EXTEND_GROWTH_FACTOR * target_new_n_ctx)

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
        else:
            self._raw_keys[old_n_ctx:new_n_ctx] = new_keys
            self._raw_values[old_n_ctx:new_n_ctx] = new_values

        self._n_ctx_total_seen = new_n_ctx

        return self.keys, self.values


class LocalSingleSeqKVCache(SingleSeqKVCache):
    def __init__(self, local_ctx: int):
        super().__init__()
        self.local_ctx = local_ctx  # local ctx window

        # Slices into the raw_keys and raw_values buffer

        self._buffer_slice_start = 0
        self._buffer_slice_end = 0
        self._n_ctx_total_seen = 0  # total number of ctx tokens seen

    @property
    def keys(self) -> torch.Tensor:
        return self._raw_keys[self._buffer_slice_start : self._buffer_slice_end]

    @property
    def values(self) -> torch.Tensor:
        return self._raw_values[self._buffer_slice_start : self._buffer_slice_end]

    @property
    def n_ctx_in_buffer(self):
        """Physical amount that exists in the buffer"""
        return self._buffer_slice_end - self._buffer_slice_start

    def extend_and_return_all(
        self, new_keys: torch.Tensor, new_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert_eq(new_keys.ndim, 3)
        assert_eq(new_values.ndim, 3)
        n_ctx_from_new_keys, n_heads, d_head = new_keys.shape

        kwargs = dict(device=new_keys.device, dtype=new_keys.dtype)

        if self.is_empty:
            # Can be empty because it's sent to our triton op
            self._raw_keys = torch.empty(self.local_ctx, n_heads, d_head, **kwargs)
            # Need to use zeros for the values
            self._raw_values = torch.zeros(self.local_ctx, n_heads, d_head, **kwargs)

        old_slice = slice(self._buffer_slice_start, self._buffer_slice_end)
        all_keys = torch.cat([self._raw_keys[old_slice], new_keys])
        all_values = torch.cat([self._raw_values[old_slice], new_values])

        self._n_ctx_total_seen += n_ctx_from_new_keys
        new_buffer_slice_end = min(self._n_ctx_total_seen, self.local_ctx)

        last_keys = all_keys[-self.local_ctx :]
        self._raw_keys[:new_buffer_slice_end] = last_keys

        last_values = all_values[-self.local_ctx :]
        self._raw_values[:new_buffer_slice_end] = last_values

        self._buffer_slice_start = 0
        self._buffer_slice_end = new_buffer_slice_end

        return all_keys, all_values


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
        cache.extend_and_return_all(keys, values)


def scores_via_qk_dotprod(
    query: RaggedActivations,
    key: RaggedActivations,
) -> torch.Tensor:
    padded_query = query.to_garbage_padded()
    padded_key = key.to_garbage_padded()
    return torch.einsum("skhd,sqhd->hsqk", padded_key, padded_query)
