# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import numpy as np
import torch
from ragged_inference_v2.test_utils import assert_eq


class RaggedActivations:
    def __init__(self, raw_tensor: torch.Tensor, n_ctx_per_seq: List[int]):
        assert_eq(raw_tensor.ndim, 3)
        self.raw_tensor = raw_tensor
        self.n_ctx_per_seq = n_ctx_per_seq

    @property
    def n_seqs(self):
        return len(self.n_ctx_per_seq)

    @property
    def max_n_ctx_per_seq(self):
        return max(self.n_ctx_per_seq)

    @property
    def dtype(self):
        return self.raw_tensor.dtype

    @property
    def device(self):
        return self.raw_tensor.device

    @classmethod
    def from_list(cls, tensors: List[torch.Tensor]):
        """Tensors must all be of shape [n_ctx, d_model]."""
        return cls(
            raw_tensor=torch.cat(tensors),
            n_ctx_per_seq=[tensor.shape[0] for tensor in tensors],
        )

    def iter_full_tensors(self):
        idx_so_far = 0
        for n_ctx_in_this_seq in self.n_ctx_per_seq:
            yield self.raw_tensor[idx_so_far : idx_so_far + n_ctx_in_this_seq]
            idx_so_far += n_ctx_in_this_seq

    def to_garbage_padded(self) -> torch.Tensor:
        """
        Create a tensor of shape (n_seqs, n_ctx_max, d_model) where the
        sequences are right-padded with garbage data
        """
        n_seqs = len(self.n_ctx_per_seq)
        n_ctx_max = max(self.n_ctx_per_seq)

        assert_eq(self.raw_tensor.ndim, 3)  # (n_ctx, head, dim_per_head)
        n_ctx_across_all_seqs, n_heads, n_dim = self.raw_tensor.shape

        # TODO: flag use zeros for garbage
        padded_acts = torch.zeros(
            n_seqs,
            n_ctx_max,
            n_heads,
            n_dim,
            dtype=self.raw_tensor.dtype,
            device="cuda",
        )

        idx_so_far = 0
        for seq_idx, n_ctx_in_this_seq in enumerate(self.n_ctx_per_seq):
            this_seq = self.raw_tensor[idx_so_far : idx_so_far + n_ctx_in_this_seq]

            padded_acts[seq_idx, :n_ctx_in_this_seq, :, :] = this_seq
            idx_so_far += n_ctx_in_this_seq

        return padded_acts


def get_acts_offset_per_seq(n_ctx_per_seq):
    n_ctx_per_seq_shifted = np.array([0] + n_ctx_per_seq[:-1])
    ragged_acts_offset_per_seq = n_ctx_per_seq_shifted.cumsum(axis=0)
    return ragged_acts_offset_per_seq


print("THIS IS LOADING")
"""

# TODO: Build LUT
seq_idx = 1
ctx_idx = 0

ragged_offset = 1

# How to do a list of tensors?
#

# TODO: Add the QK dotprod to get scores
#  - Start with a ragged tensor for the keys also
#  - Using a list of tensors as the Keys
#  - Using sequences

# 16x16x256


# scores [n_seq, n_ctx_keys_max, n_ctx_queries_max]


# final_out [n_seq, n_ctx_keys_max, d_model]
"""
