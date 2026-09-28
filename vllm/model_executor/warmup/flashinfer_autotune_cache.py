# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    config_hash = runner.vllm_config.compute_hash(include_version=False)
    return hashlib.sha256(config_hash.encode()).hexdigest()


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache: bytes | str | None = None
    if (
        group.rank_in_group == 0
        and runner.vllm_config.kernel_config.enable_flashinfer_autotune
    ):
        try:
            from vllm.platforms import current_platform
            from vllm.utils.flashinfer import has_flashinfer

            if has_flashinfer() and current_platform.has_device_capability(90):
                from flashinfer.autotuner import AutoTuner

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "autotune_configs.json"
                    AutoTuner.get().save_configs(str(path))
                    cache = path.read_bytes()
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"

    cache = group.broadcast_object(cache)
    if isinstance(cache, str):
        raise RuntimeError(f"Failed to serialize FlashInfer autotune state: {cache}")
    if cache is None or group.rank_in_group == 0:
        return

    from flashinfer.autotuner import AutoTuner

    with tempfile.NamedTemporaryFile() as f:
        f.write(cache)
        f.flush()
        if not AutoTuner.get().load_configs(f.name):
            raise RuntimeError("FlashInfer autotune cache is incompatible")


def merge_peer_autotune_configs(tuner, group: "GroupCoordinator") -> int:
    """Load every other rank's tuned configs into the leader's tuner.

    FlashInfer's MoE tuning keys carry the TP/EP rank, so a cache holding only
    the leader's configs misses on every other rank after a reload. Those
    ranks then re-profile while the leader hits, and FlashInfer's per-tactic
    reduce (which requires identical caches on all ranks) stalls. Saving the
    union makes a reload hit identically everywhere. Every rank must call
    this; returns the number of peer config files the leader loaded.
    """
    import torch.distributed as dist

    with tempfile.TemporaryDirectory() as temp_dir:
        local = Path(temp_dir) / "rank_configs.json"
        tuner.save_configs(str(local))
        payload = local.read_text()
    gathered: list[str | None] = [None] * group.world_size
    dist.all_gather_object(gathered, payload, group=group.cpu_group)
    if group.rank_in_group != 0:
        return 0
    loaded = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        for rank, text in enumerate(gathered):
            if rank == 0 or text is None:
                continue
            peer = Path(temp_dir) / f"rank{rank}_configs.json"
            peer.write_text(text)
            if not tuner.load_configs(str(peer)):
                raise RuntimeError(
                    f"FlashInfer autotune configs from rank {rank} are incompatible"
                )
            loaded += 1
    return loaded


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}."
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
