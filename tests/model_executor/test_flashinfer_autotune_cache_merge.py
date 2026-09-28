# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The persisted FlashInfer autotune cache must cover every rank's keys.

FlashInfer MoE keys carry the TP rank, so a leader-only cache misses on the
other ranks after a reload and desynchronizes the per-tactic reduce.
"""

import json
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    merge_peer_autotune_configs,
)


def _key(tp_rank):
    return str(("trtllm::fused_moe::gemm1", "MoERunner", ((8, 2048),), (8, 4, tp_rank)))


class FakeTuner:
    def __init__(self, entries, compatible=True):
        self.entries = dict(entries)
        self.compatible = compatible

    def save_configs(self, path):
        with open(path, "w") as f:
            json.dump({"_metadata": {"gpu": "GB10"}, **self.entries}, f)

    def load_configs(self, path):
        if not self.compatible:
            return False
        with open(path) as f:
            loaded = json.load(f)
        self.entries.update({k: v for k, v in loaded.items() if not k.startswith("_")})
        return True


def _group(rank):
    return SimpleNamespace(world_size=4, rank_in_group=rank, cpu_group=object())


def _gather_from(tuners, monkeypatch):
    payloads = []
    for tuner in tuners:
        payloads.append(json.dumps({"_metadata": {}, **tuner.entries}))

    def fake_all_gather_object(output, obj, group=None):
        output[:] = payloads

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)


def test_leader_saves_union_of_every_rank(monkeypatch):
    tuners = [FakeTuner({_key(rank): ["MoERunner", rank + 1]}) for rank in range(4)]
    _gather_from(tuners, monkeypatch)
    assert merge_peer_autotune_configs(tuners[0], _group(0)) == 3
    assert set(tuners[0].entries) == {_key(rank) for rank in range(4)}
    # The leader keeps its own measured tactic for its own key.
    assert tuners[0].entries[_key(0)] == ["MoERunner", 1]


def test_non_leader_contributes_but_loads_nothing(monkeypatch):
    tuners = [FakeTuner({_key(rank): ["MoERunner", 0]}) for rank in range(4)]
    _gather_from(tuners, monkeypatch)
    assert merge_peer_autotune_configs(tuners[2], _group(2)) == 0
    assert set(tuners[2].entries) == {_key(2)}


def test_incompatible_peer_configs_fail_loudly(monkeypatch):
    tuners = [FakeTuner({_key(rank): ["MoERunner", 0]}) for rank in range(4)]
    tuners[0].compatible = False
    _gather_from(tuners, monkeypatch)
    with pytest.raises(RuntimeError, match="rank 1"):
        merge_peer_autotune_configs(tuners[0], _group(0))


def test_real_flashinfer_tuner_saves_loaded_peer_configs(tmp_path, monkeypatch):
    """FlashInfer's save_configs() must include configs added by load_configs()."""
    autotuner = pytest.importorskip("flashinfer.autotuner")
    tuner = autotuner.AutoTuner.get()
    own = tmp_path / "own.json"
    tuner.save_configs(str(own))
    metadata = json.loads(own.read_text()).get("_metadata")
    peer_key = str(("trtllm::fused_moe::gemm1", "MoERunner", ((8, 2048),), (8, 4, 3)))
    peer = tmp_path / "peer.json"
    peer.write_text(json.dumps({"_metadata": metadata, peer_key: ["MoERunner", 5]}))
    assert tuner.load_configs(str(peer))
    merged = tmp_path / "merged.json"
    tuner.save_configs(str(merged))
    assert peer_key in json.loads(merged.read_text())
