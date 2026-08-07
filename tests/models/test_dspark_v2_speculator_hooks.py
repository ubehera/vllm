# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every DSpark model class the V2 speculator can load must expose the
sampling hooks and optional vocabulary map it probes. The sampling path calls
``map_draft_to_target`` and the rejection-logit initialization reads
``draft_id_to_target_id`` even for the full-vocabulary NVIDIA DeepSeek V4
drafter. Missing either contract is a cold-boot blocker, not an edge case."""

import inspect

from vllm.models.deepseek_v4.nvidia import dspark as nvidia_dspark


def test_nvidia_dspark_exposes_v2_speculator_hooks():
    cls = nvidia_dspark.DSparkDeepseekV4ForCausalLM
    assert hasattr(cls, "draft_id_to_target_id")
    assert cls.draft_id_to_target_id is None
    for hook in ("map_draft_to_target", "compute_draft_logits"):
        assert hasattr(cls, hook), (
            f"{cls.__name__} lacks {hook}; the V2 DSpark speculator dies in "
            "profile_run on the first cold boot"
        )


def test_map_draft_to_target_is_identity_for_full_vocab():
    src = inspect.getsource(
        nvidia_dspark.DSparkDeepseekV4ForCausalLM.map_draft_to_target
    )
    assert "return draft_ids" in src
