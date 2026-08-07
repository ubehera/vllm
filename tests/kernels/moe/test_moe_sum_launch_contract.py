# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep MoE sum CUDA launch arity aligned with its kernel declarations."""

import re
from pathlib import Path


SOURCE = (
    Path(__file__).parents[3]
    / "csrc"
    / "libtorch_stable"
    / "moe"
    / "moe_align_sum_kernels.cu"
)


def _argument_count(source: str, start: int) -> int:
    depth = 0
    commas = 0
    for char in source[start:]:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return commas + 1
        elif char == "," and depth == 1:
            commas += 1
    raise AssertionError("unterminated CUDA kernel launch")


def test_non_pad_aware_moe_sum_launches_match_kernel_arity():
    source = SOURCE.read_text().replace("\\\n", " ")
    launches = re.compile(
        r"moe_sum_(vec_kernel|vec_dynamic_kernel|scalar_kernel)"
        r"<[^>]*\bfalse>\s*<<<.*?>>>\s*\(",
        re.DOTALL,
    )
    expected = {
        "vec_kernel": 11,
        "vec_dynamic_kernel": 12,
        "scalar_kernel": 12,
    }
    observed = {}
    for match in launches.finditer(source):
        name = match.group(1)
        observed[name] = _argument_count(source, match.end() - 1)
    assert observed == expected
