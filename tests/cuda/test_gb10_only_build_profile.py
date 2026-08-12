# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gb10_profile_omits_pre_sm12x_fallbacks() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()

    assert "option(VLLM_BUILD_GB10_ONLY" in cmake
    assert 'NOT CUDA_ARCH MATCHES "^12[.](0|1)"' in cmake
    for arch_list in (
        "MARLIN_OTHER_ARCHS",
        "SCALED_MM_2X_ARCHS",
        "HADACORE_ARCHS",
        "MARLIN_MOE_OTHER_ARCHS",
    ):
        assert f'set({arch_list} "")' in cmake


def test_bundled_flash_attention_can_be_omitted() -> None:
    setup = (ROOT / "setup.py").read_text()

    assert 'os.getenv("VLLM_BUILD_FLASH_ATTN", "1")' in setup
    assert "if should_build_flash_attn():" in setup
