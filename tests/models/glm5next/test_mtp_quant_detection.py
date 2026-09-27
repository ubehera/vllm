# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM MTP layer is quantized only when the checkpoint carries its scales."""

import json

import pytest

from vllm.models.glm5next.common.mtp import (
    _mtp_layer_is_quantized,
    mtp_layer_has_quant_scales,
)

# nvidia/GLM-5.3-Flash-NVFP4: layer 44 is NVFP4, the MTP layer 45 is BF16.
NVFP4_NAMES = [
    "model.language_model.layers.44.mlp.experts.0.down_proj.weight",
    "model.language_model.layers.44.mlp.experts.0.down_proj.weight_scale",
    "model.language_model.layers.44.mlp.experts.0.down_proj.weight_scale_2",
    "model.language_model.layers.45.mlp.experts.0.down_proj.weight",
    "model.language_model.layers.45.eh_proj.weight",
]


@pytest.mark.parametrize(
    ("names", "layer_idx", "expected"),
    [
        (NVFP4_NAMES, 45, False),
        (NVFP4_NAMES, 44, True),
        (["model.layers.45.mlp.experts.0.up_proj.weight_scale"], 45, True),
        # A layer index prefix must not match a longer one (layers.4 vs 45).
        (NVFP4_NAMES, 4, False),
        ([], 45, False),
    ],
)
def test_mtp_layer_has_quant_scales(names, layer_idx, expected):
    assert mtp_layer_has_quant_scales(names, layer_idx) is expected


def test_mtp_layer_is_quantized_reads_local_index(tmp_path):
    assert _mtp_layer_is_quantized(str(tmp_path), 45) is None
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001.safetensors" for name in NVFP4_NAMES}})
    )
    assert _mtp_layer_is_quantized(str(tmp_path), 45) is False
    assert _mtp_layer_is_quantized(str(tmp_path), 44) is True
