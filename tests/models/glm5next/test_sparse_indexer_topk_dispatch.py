# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's top-k backend wiring."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform


def _require_deep_gemm() -> None:
    from vllm.utils.deep_gemm import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("kpool indexer requires DeepGEMM")


def _build(indexer_cls, backend: str):
    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        return indexer_cls(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "persistent", "cooperative", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend(backend: str) -> None:
    """The kpool indexer must read kernel_config.sparse_indexer_topk_backend
    and hand it to the shared SparseIndexerTopk dispatcher, rather than
    hard-coding its own cooperative/persistent/per_row choice."""
    _require_deep_gemm()
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "aiter", "per_row", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend_rocm(
    backend: str,
) -> None:
    """The AMD kpool indexer must go through the same dispatcher, so the
    AITER decode top-k is reachable from kernel_config."""
    from vllm.models.glm5next.amd.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize(
    "configured,full_cudagraph,expected",
    [
        ("auto", False, "aiter"),
        ("auto", True, "auto"),
        ("aiter", True, "aiter"),
        ("per_row", False, "per_row"),
    ],
)
def test_kpool_decode_topk_backend_never_auto_selects_aiter_under_full_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    full_cudagraph: bool,
    expected: str,
) -> None:
    """A FULL cudagraph bakes in one kernel for every replay, so "auto" cannot
    use the context length to pick AITER and must stay on the in-tree kernel.
    Explicit backends keep working, and outside FULL the long-context heuristic
    still narrows to AITER."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.glm5next.amd import sparse_indexer

    monkeypatch.setattr(
        rocm_aiter_ops, "is_indexer_top_k_enabled", lambda: True, raising=False
    )
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_indexer_top_k_supported",
        lambda **kwargs: True,
        raising=False,
    )

    assert (
        sparse_indexer._kpool_decode_topk_backend(
            configured,
            num_rows=8,
            max_valid_seq_len=64 * 1024,
            select_k=512,
            index_kpool=4,
            full_cudagraph=full_cudagraph,
        )
        == expected
    )


class _Props:
    def __init__(self, shared_memory_per_block_optin: int) -> None:
        self.shared_memory_per_block_optin = shared_memory_per_block_optin


@pytest.fixture
def _fake_cuda(monkeypatch):
    from vllm.models.glm5next.nvidia import sparse_indexer

    def install(shared_memory: int):
        monkeypatch.setattr(sparse_indexer.current_platform, "is_cuda", lambda: True)
        monkeypatch.setattr(
            sparse_indexer.torch.cuda,
            "get_device_properties",
            lambda index: _Props(shared_memory),
        )
        sparse_indexer._cuda_can_use_persistent_topk.cache_clear()
        return sparse_indexer

    yield install
    sparse_indexer._cuda_can_use_persistent_topk.cache_clear()


@pytest.mark.parametrize(
    ("shared_memory", "expected"), [(101_376, "per_row"), (131_072, "auto")]
)
def test_auto_topk_resolves_by_opt_in_shared_memory(
    _fake_cuda, shared_memory: int, expected: str
) -> None:
    """GB10 (101,376 B) must get the exact per-row kernel; >=128 KiB keeps auto."""
    mod = _fake_cuda(shared_memory)
    assert mod.resolve_kpool_topk_backend("auto", 512, 0) == expected


@pytest.mark.parametrize("backend", ["persistent", "cooperative", "per_row", "torch"])
def test_explicit_topk_backend_is_never_rewritten(_fake_cuda, backend: str) -> None:
    mod = _fake_cuda(101_376)
    assert mod.resolve_kpool_topk_backend(backend, 512, 0) == backend
