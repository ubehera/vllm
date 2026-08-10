# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression gate: the DSv4 sparse builders must agree on the decode boundary.

The indexer, the sparse-SWA builder and the C128A builder all slice the SAME
``topk_indices_buffer`` at ``num_decode_tokens``. If they disagree about where
that boundary falls, one writes at one offset while another reads at a
different one and tokens receive each other's top-k indices. Every index stays
individually valid -- a real slot in the owning request's block table -- so
per-slot validity checks and byte-level sentinels cannot see it; it surfaces
only as garbled output, and only under concurrency, since a pure-decode or
pure-prefill batch cannot expose the disagreement.

Two independent things move that boundary, and each gets a test:

  1. ``treat_short_extends_as_decodes`` -- fixed here via
     ``sparse_short_extend_tiering()``. Asserted by inspecting the three call
     sites, because asserting on a shared helper's return value cannot fail
     when the builders do not call it.

  2. ``decode_threshold`` -- derived from the shared speculative-decode query
     bound, including the extra parallel-drafting width.
"""

import ast
import inspect
from types import SimpleNamespace

import torch

from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    get_spec_decode_max_query_len,
)
from vllm.v1.attention.backends.utils import (
    sparse_short_extend_tiering,
    split_decodes_and_prefills,
)


class _CM:
    """Minimal CommonAttentionMetadata stand-in for the split helpers."""

    def __init__(self, query_start_loc_cpu, seq_lens_cpu, is_prefilling):
        self.query_start_loc_cpu = query_start_loc_cpu
        self.seq_lens_cpu = seq_lens_cpu
        self.is_prefilling = is_prefilling
        self.num_reqs = len(seq_lens_cpu)
        diffs = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.max_query_len = int(diffs.max().item())
        self.num_actual_tokens = int(query_start_loc_cpu[-1].item())


def _mixed_batch():
    # 1 pure decode (q=1), 1 short extend (q=2), 1 real prefill (q=64)
    q = torch.tensor([0, 1, 3, 67], dtype=torch.int32)
    seq = torch.tensor([128, 130, 512], dtype=torch.int32)
    return _CM(q, seq, torch.tensor([False, False, True]))


def _tiering_call_sites() -> dict[str, str]:
    """The ``treat_short_extends_as_decodes=`` expression at each builder."""
    from vllm.models.deepseek_v4 import sparse_mla
    from vllm.v1.attention.backends.mla import indexer, sparse_swa

    found: dict[str, str] = {}
    for name, module in (
        ("indexer", indexer),
        ("sparse_swa", sparse_swa),
        ("c128a", sparse_mla),
    ):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if fname != "split_decodes_and_prefills":
                continue
            for kw in node.keywords:
                if kw.arg == "treat_short_extends_as_decodes":
                    found[name] = ast.unparse(kw.value)
    return found


def test_every_builder_derives_the_flag_from_the_shared_helper():
    """The coupling, asserted where it can actually break.

    A test that calls one helper three times with the same arguments agrees
    with itself whatever the builders do -- it passes unchanged against a tree
    where none of them were touched, so it guards nothing. What has to hold is
    that each call site *uses* the helper, so that is what is inspected.
    """
    sites = _tiering_call_sites()
    assert set(sites) == {"indexer", "sparse_swa", "c128a"}, sites
    for name, expr in sites.items():
        assert "sparse_short_extend_tiering" in expr, (
            f"{name} derives treat_short_extends_as_decodes independently "
            f"({expr!r}); it will drift from the others again"
        )


def test_tiering_is_false_when_batch_has_prefilling_rows():
    assert sparse_short_extend_tiering(_mixed_batch()) is False


def test_tiering_is_true_for_a_pure_decode_batch():
    q = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq = torch.tensor([128, 130], dtype=torch.int32)
    cm = _CM(q, seq, torch.tensor([False, False]))
    assert sparse_short_extend_tiering(cm) is True


def test_parallel_drafting_threshold_matches_all_sparse_builders():
    """Producer and consumers must split the shared top-k buffer identically."""
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5,
            parallel_drafting=True,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    threshold = get_spec_decode_max_query_len(config)
    assert threshold == 11

    from vllm.models.deepseek_v4.sparse_mla import (
        DeepseekV4FlashMLAMetadataBuilder,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadataBuilder,
    )

    assert "get_spec_decode_max_query_len" in inspect.getsource(
        DeepseekV32IndexerMetadataBuilder.__init__
    )
    assert "get_spec_decode_max_query_len" in inspect.getsource(
        DeepseekSparseSWAMetadataBuilder.__init__
    )
    assert "_init_reorder_batch_threshold" in inspect.getsource(
        DeepseekV4FlashMLAMetadataBuilder.__init__
    )

    c128a_builder = SimpleNamespace(vllm_config=config)
    AttentionMetadataBuilder._init_reorder_batch_threshold(
        c128a_builder,
        1,
        supports_spec_as_decode=True,
    )
    assert c128a_builder.reorder_batch_threshold == threshold

    qlen = 8
    q = torch.tensor([0, 1, 1 + qlen], dtype=torch.int32)
    seq = torch.tensor([128, 256], dtype=torch.int32)
    cm = _CM(q, seq, torch.tensor([False, False]))
    tiering = sparse_short_extend_tiering(cm)

    boundaries = {
        name: split_decodes_and_prefills(
            cm,
            decode_threshold=builder_threshold,
            treat_short_extends_as_decodes=tiering,
        )[2]
        for name, builder_threshold in (
            ("indexer", threshold),
            ("sparse_swa", threshold),
            ("c128a", c128a_builder.reorder_batch_threshold),
        )
    }
    assert len(set(boundaries.values())) == 1, boundaries
