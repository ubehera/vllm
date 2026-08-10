# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.platforms.cuda import CudaPlatformBase


def test_deepgemm_supports_sm120_family() -> None:
    """The GB10 runtime must not silently fall back from DeepGEMM to Marlin."""
    with (
        patch.object(CudaPlatformBase, "is_device_capability", return_value=False),
        patch.object(
            CudaPlatformBase,
            "is_device_capability_family",
            side_effect=lambda family: family == 120,
        ),
    ):
        assert CudaPlatformBase.support_deep_gemm()
