# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram shards reach the device exactly through the chunked pinned copy."""

import pytest
import torch

from vllm.models.deepseek_v41.common import engram

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("rows", [1, 7, 64, 65, 1000])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_chunked_copy_matches_source(monkeypatch, rows, dtype):
    # 4 rows of 256 bytes per chunk exercises full, partial and single chunks.
    monkeypatch.setattr(engram, "_ENGRAM_COPY_CHUNK_BYTES", 4 * 256)
    src = torch.randint(0, 255, (rows, 256), dtype=torch.uint8).view(dtype)
    dst = torch.empty_like(src, device="cuda")
    engram._copy_rows_to_device(dst, src)
    assert torch.equal(dst.cpu().view(torch.uint8), src.view(torch.uint8))


def test_host_destination_keeps_plain_copy():
    src = torch.arange(32, dtype=torch.uint8).reshape(8, 4)
    dst = torch.empty_like(src)
    engram._copy_rows_to_device(dst, src)
    assert torch.equal(dst, src)
