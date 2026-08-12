# SPDX-License-Identifier: Apache-2.0
"""Keep the GB10 fork on the upstream DSv4 DSpark execution contract."""

from pathlib import Path


def test_deepseek_v4_uses_upstream_cache_managed_dspark_layers() -> None:
    source = (
        Path(__file__).parents[2] / "vllm/models/deepseek_v4/nvidia/dspark.py"
    ).read_text()

    assert "class DSparkDeepseekV4Model" in source
    assert "DeepseekV4DecoderLayer(" in source
    assert "_main_kv_cache" not in source
    assert "fused_deepseek_v4_kv_rope_full_cache_fp8_insert" in source
    assert "fused_deepseek_v4_kv_rope_full_cache_bf16_insert" in source


def test_deepseek_v4_dspark_keeps_v2_sampling_hooks() -> None:
    source = (
        Path(__file__).parents[2] / "vllm/models/deepseek_v4/nvidia/dspark.py"
    ).read_text()

    for hook in (
        "draft_id_to_target_id = None",
        "def compute_draft_logits(",
        "def map_draft_to_target(",
    ):
        assert hook in source
