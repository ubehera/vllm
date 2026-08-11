# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import (
    flashinfer_mla_sparse_sm120 as sm120_module,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_forward_mqa_keeps_proven_full_topk_contract(monkeypatch) -> None:
    num_tokens = 130
    num_heads = 16
    topk_tokens = 4
    kernel_rows: list[int] = []

    def convert_indices(*args, **kwargs):  # noqa: ARG001
        assert not kwargs.get("return_valid_counts", False)
        return torch.zeros((num_tokens, topk_tokens), dtype=torch.int32)

    def sparse_decode(**kwargs):
        assert kwargs["seq_lens"] is None
        kernel_rows.append(kwargs["query"].shape[0])
        return kwargs["out"]

    monkeypatch.setattr(
        sm120_module,
        "triton_convert_req_index_to_global_index",
        convert_indices,
    )
    monkeypatch.setattr(
        fi_utils,
        "flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        sparse_decode,
    )

    impl = object.__new__(sm120_module.FlashInferMLASparseSM120Impl)
    impl.topk_indices_buffer = torch.zeros(
        (num_tokens, topk_tokens), dtype=torch.int32
    )
    impl.num_heads = num_heads
    impl.kv_lora_rank = 8
    impl.qk_nope_head_dim = 8
    impl.qk_rope_head_dim = 8
    impl.scale = 1.0
    impl.kv_scale_format = "pow2_fp32"
    impl._workspace_buffer = torch.empty(1, dtype=torch.uint8)

    q = torch.empty((num_tokens, num_heads, 16), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 1, 16), dtype=torch.uint8)
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=64,
        topk_tokens=topk_tokens,
    )

    output, lse = impl.forward_mqa(q, kv_cache, metadata, layer=None)

    assert output.shape == (num_tokens, num_heads, impl.kv_lora_rank)
    assert lse is None
    assert kernel_rows == [64, 64, 2]
