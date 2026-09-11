# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FlashInfer plan must match the logical cache view used by forward_mqa."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Builder,
    FlashInferMLASparseSM90Impl,
)


@pytest.mark.parametrize(
    "storage_dtype,fp8,expected_dtype",
    [
        (torch.uint8, True, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, True, torch.float8_e4m3fn),
        (torch.bfloat16, False, torch.bfloat16),
    ],
)
def test_plan_dtype_matches_cache_view(storage_dtype, fp8, expected_dtype):
    impl = object.__new__(FlashInferMLASparseSM90Impl)
    impl.num_heads = 16
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 0
    impl.scale = 0.0625
    impl.use_fp8_kv_cache = fp8
    impl.topk_indices_buffer = torch.empty(1, 2176, dtype=torch.int32)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": SimpleNamespace(impl=impl)}
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=24, async_scheduling=False
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
    )
    spec = SimpleNamespace(dtype=storage_dtype, tokens_per_state=4)
    # Isolate the subclass contract from distributed setup and GPU allocation.
    with (
        patch.object(FlashInferMLASparseMetadataBuilder, "__init__", return_value=None),
        patch(
            "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90._SM90State"
        ) as state,
    ):
        FlashInferMLASparseSM90Builder(spec, ["layer"], config, torch.device("cpu"))
    assert state.call_count == 1
    assert state.call_args.args[2] == expected_dtype
