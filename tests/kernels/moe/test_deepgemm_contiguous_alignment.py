# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-call M alignment for DeepGEMM grouped-contiguous MoE GEMMs.

Runs without CUDA: the DeepGEMM binding and device family are stubbed so both
binding generations (nv_dev's `(expected_m, num_groups)` and vllm-project's
`(expected_m)`, which returns the legacy 128 off SM100) are covered.
"""

import pytest

import vllm.utils.deep_gemm as dg_utils
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
)

LEGACY_ALIGNMENT = 128


def _vllm_project_binding(expected_m=None):
    # vllm-project DeepGEMM: no num_groups; SM100 only honours expected_m.
    return LEGACY_ALIGNMENT


def _nv_dev_binding(expected_m=None, num_groups=None):
    per_group_m = expected_m
    if expected_m is not None and num_groups:
        per_group_m = -(-expected_m // num_groups)
    return 64 if per_group_m is not None and per_group_m <= 64 else 128


@pytest.fixture
def stub_deep_gemm(monkeypatch):
    def install(binding, family):
        monkeypatch.setattr(dg_utils, "_lazy_init", lambda: None)
        monkeypatch.setattr(
            dg_utils,
            "_get_theoretical_mk_alignment_for_contiguous_layout_impl",
            binding,
        )
        monkeypatch.setattr(
            dg_utils.current_platform,
            "is_device_capability_family",
            lambda capability, device_id=0: capability == family,
        )

    return install


@pytest.mark.parametrize("binding", [_vllm_project_binding, _nv_dev_binding])
@pytest.mark.parametrize(
    ("expected_m", "num_groups", "alignment"),
    [
        (24 * 8, 256, 64),  # MTP decode batch: under one row per expert
        (64 * 256, 256, 64),  # exactly one small tile per expert
        (64 * 256 + 1, 256, 128),  # one expert spills past 64 rows
        (8192 * 8, 256, 128),  # prefill chunk
        (None, 256, 128),  # unknown M keeps the large tile
        (48, None, 64),  # legacy callers pass per-expert M directly
        (65, None, 128),
    ],
)
def test_sm120_picks_small_tile_for_small_experts(
    stub_deep_gemm, binding, expected_m, num_groups, alignment
):
    stub_deep_gemm(binding, family=120)
    assert (
        dg_utils.get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=expected_m, num_groups=num_groups
        )
        == alignment
    )


def test_other_families_keep_binding_alignment(stub_deep_gemm):
    stub_deep_gemm(_vllm_project_binding, family=100)
    assert (
        dg_utils.get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=24 * 8, num_groups=256
        )
        == LEGACY_ALIGNMENT
    )


def test_sm120_keeps_a_smaller_binding_alignment(stub_deep_gemm):
    stub_deep_gemm(lambda expected_m=None, num_groups=None: 32, family=120)
    assert (
        dg_utils.get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=8192 * 8, num_groups=256
        )
        == 32
    )


def test_rejects_non_positive_num_groups(stub_deep_gemm):
    stub_deep_gemm(_vllm_project_binding, family=120)
    with pytest.raises(ValueError):
        dg_utils.get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=8, num_groups=0
        )


def test_sm120_decode_workspace_uses_small_tile(stub_deep_gemm):
    stub_deep_gemm(_vllm_project_binding, family=120)
    M_sum, alignment = compute_aligned_M_and_alignment(
        M=24,
        num_topk=8,
        local_num_experts=256,
        alignment=LEGACY_ALIGNMENT,
        expert_tokens_meta=None,
    )
    assert alignment == 64
    # 192 routed rows over at most 192 active experts, each padded to 64.
    assert M_sum == 192 * 64
    assert M_sum % alignment == 0
