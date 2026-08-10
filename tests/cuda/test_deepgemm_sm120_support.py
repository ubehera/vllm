# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from unittest.mock import patch

from vllm.platforms.cuda import CudaPlatformBase


DEEPGEMM_REPO = "https://github.com/vllm-project/DeepGEMM.git"
DEEPGEMM_COMMIT = "5f33a18079e96d26d5869c9759657eb6150f31b1"


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


def test_deepgemm_build_pins_match_proven_sm120_situ_merge() -> None:
    root = Path(__file__).parents[2]
    cmake = (root / "cmake/external_projects/deepgemm.cmake").read_text()
    installer = (root / "tools/install_deepgemm.sh").read_text()
    for source in (cmake, installer):
        assert DEEPGEMM_REPO in source
        assert DEEPGEMM_COMMIT in source
        assert "a6b593d2826719dcf4892609af7b84ee23aaf32a" not in source
