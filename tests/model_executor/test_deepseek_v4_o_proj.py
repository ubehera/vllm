# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.models.deepseek_v4.nvidia.ops import fp8_einsum
from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability


@pytest.mark.parametrize(
    ("capability", "expected_recipe", "expected_tma_aligned"),
    [
        ((9, 0), (1, 128, 128), False),
        ((10, 0), (1, 1, 128), True),
        ((12, 0), (1, 128, 128), False),
        ((12, 1), (1, 128, 128), False),
    ],
)
def test_deepseek_v4_o_proj_recipe_is_arch_specific(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected_recipe: tuple[int, int, int],
    expected_tma_aligned: bool,
):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(*capability),
    )

    assert compute_fp8_einsum_recipe() == (expected_recipe, expected_tma_aligned)


def test_sm12x_bmm_weight_keeps_logical_scale_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_transform(*args, **kwargs):
        raise AssertionError("SM12x Triton BMM scales must not be DeepGEMM-packed")

    monkeypatch.setattr(
        fp8_utils,
        "transform_sf_into_required_layout",
        fail_transform,
    )
    weight = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones((2, 1), dtype=torch.float32)

    processed_weight, processed_scale = (
        fp8_utils.deepgemm_post_process_fp8_weight_block(
            weight,
            scale,
            (128, 128),
            use_e8m0=False,
            is_bmm=True,
            bmm_batch_size=2,
            preserve_bmm_scale_layout=True,
        )
    )

    assert processed_weight.shape == (2, 128, 128)
    assert processed_scale.shape == (2, 1, 1)
    torch.testing.assert_close(processed_scale, scale.view(2, 1, 1))


def test_sm12x_o_proj_dispatches_postprocessed_bmm_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(12, 1),
    )
    called = False

    def fake_sm12x_einsum(a, a_scale, b, b_scale, out):
        nonlocal called
        called = True
        assert b.shape == (2, 128, 128)
        assert b_scale.shape == (2, 1, 1)

    def fail_deep_gemm(*args, **kwargs):
        raise AssertionError("postprocessed SM12x BMM must use the Triton fallback")

    monkeypatch.setattr(
        fp8_einsum,
        "deepseek_v4_sm12x_fp8_einsum",
        fake_sm12x_einsum,
    )
    monkeypatch.setattr(fp8_einsum, "fp8_einsum", fail_deep_gemm)
    a = torch.empty((1, 2, 128), dtype=torch.float8_e4m3fn)
    a_scale = torch.ones((1, 2, 1), dtype=torch.float32)
    b = torch.empty((2, 128, 128), dtype=torch.float8_e4m3fn)
    b_scale = torch.ones((2, 1, 1), dtype=torch.float32)
    out = torch.empty((1, 2, 128), dtype=torch.bfloat16)

    fp8_einsum.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        out,
        "bhr,hdr->bhd",
        [1, 128, 128],
    )

    assert called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_fused_o_proj_quant_emits_ue8m0_compatible_scales():
    capability = torch.cuda.get_device_capability()
    if capability[0] != 12:
        pytest.skip(f"requires SM12x, found SM{capability[0]}{capability[1]}")

    device = torch.device("cuda")
    num_tokens, num_groups, hidden_size, out_rank = 17, 2, 128, 128
    a = torch.randn(
        (num_tokens, num_groups, hidden_size), device=device, dtype=torch.float32
    ).clamp_(-2.0, 2.0).to(torch.float8_e4m3fn)
    b = torch.randn(
        (num_groups, out_rank, hidden_size), device=device, dtype=torch.float32
    ).clamp_(-2.0, 2.0).to(torch.float8_e4m3fn)
    a_scale = torch.ones(
        (num_tokens, num_groups, hidden_size // 128),
        device=device,
        dtype=torch.float32,
    )
    b_scale = torch.ones(
        (num_groups, out_rank // 128, hidden_size // 128),
        device=device,
        dtype=torch.float32,
    )
    out_fp8 = torch.empty(
        (num_tokens, num_groups * out_rank),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    out_scale = torch.empty(
        (num_tokens, num_groups * out_rank // 128),
        device=device,
        dtype=torch.float32,
    )

    fp8_einsum.deepseek_v4_sm12x_fp8_einsum_quant(
        a,
        a_scale,
        b,
        b_scale,
        out_fp8,
        out_scale,
        use_ue8m0=True,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out_scale).all()
    assert (out_scale > 0).all()
    scale_bits = out_scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    assert ((scale_bits & 0x807FFFFF) == 0).all(), (
        "DeepGEMM UE8M0 input scales must have zero sign and mantissa bits"
    )
