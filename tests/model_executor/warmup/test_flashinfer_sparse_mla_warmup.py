# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.warmup import flashinfer_sparse_mla_warmup as warmup


class _Backend:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name


def _runner(backend_name: str):
    group = SimpleNamespace(backend=_Backend(backend_name))
    return SimpleNamespace(attn_groups=[[group]])


def test_deepseek_sparse_swa_wrapper_selects_dsv4_flashinfer_autotune():
    label = warmup._flashinfer_sparse_mla_decode_label(
        _runner("DEEPSEEK_SPARSE_SWA"),
        warmup._DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS,
    )

    assert label == "DSv4"


def test_unrelated_deepseek_backend_does_not_select_flashinfer_autotune():
    label = warmup._flashinfer_sparse_mla_decode_label(
        _runner("FLASHMLA_SPARSE_DSV4"),
        warmup._DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS,
    )

    assert label is None
