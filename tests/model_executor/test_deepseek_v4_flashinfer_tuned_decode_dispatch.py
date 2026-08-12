# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source contracts for the DeepSeek V4 FlashInfer tuned decode path."""

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[2]
    / "vllm/models/deepseek_v4/nvidia/flashinfer_sm120_decode.py"
)


def _method(tree: ast.AST, name: str) -> ast.FunctionDef:
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepseekV4FlashInferSM120DecodeAttention"
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _attribute_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == name
    ]


def test_decode_uses_public_tuned_dispatcher_with_managed_scratch() -> None:
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    decode = _method(tree, "_forward_decode")

    tuned_calls = _attribute_calls(decode, "_sm120_decode_dsv4")
    assert len(tuned_calls) == 1
    assert not _attribute_calls(decode, "run")

    call = tuned_calls[0]
    assert [ast.unparse(arg) for arg in call.args[3:8]] == [
        "mid_out",
        "mid_lse",
        "output",
        "out_lse",
        "self.scale",
    ]
    assert not _attribute_calls(decode, "unsqueeze")


def test_prefill_keeps_reusable_runner_and_scratch_includes_out_lse() -> None:
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    prefill = _method(tree, "_forward_prefill")
    runner_calls = _attribute_calls(prefill, "run")
    assert len(runner_calls) == 1
    assert any(keyword.arg == "out_lse" for keyword in runner_calls[0].keywords)

    scratch = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_decode_scratch"
    )
    simultaneous = _attribute_calls(scratch, "get_simultaneous")
    assert len(simultaneous) == 1
    assert len(simultaneous[0].args) == 3
