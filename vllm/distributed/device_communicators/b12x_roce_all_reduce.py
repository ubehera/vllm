# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoCEnante: adapter for the b12x one-shot RoCE all-reduce and all-gather.

The runtime is the copy of ``b12x.comm.roce`` vendored by FlashInfer
(``flashinfer.experimental.b12x.comm.roce``, which FlashInfer also exposes as
``b12x.comm.roce``).  It targets tensor parallelism across DGX Spark / GB10
nodes, one GPU per node, joined over ConnectX-7 RoCE without GPUDirect RDMA.
This module owns capability voting, construction, plan preparation, size
gating, the graph-capture hook and teardown; the protocol lives in the
runtime.  It is enabled only by ``VLLM_ENABLE_ROCE_ALLREDUCE=1`` and only for
tensor-parallel groups whose ranks span nodes; it does not depend on vLLM's
CUDA-IPC custom all-reduce, which multi-node launches disable.

Contract with the runtime (``roce.API_VERSION`` must equal
``REQUIRED_B12X_ROCE_API_VERSION``):

- Setup is all or nothing.  Every rank evaluates its capability (package, API
  version, integrated GPU with an active RDMA device, the RDMA proxy library,
  valid size limits) without communicating, then the ranks vote over the CPU
  (gloo) group.  Any rank that cannot take part disables the backend on every
  rank before the runtime's collective constructor runs, so no rank is left
  waiting in its setup exchange.  The constructor itself rejects ranks whose
  ABI, HCA count, slot geometry, spin limit or launch geometry differ, on
  every rank at once.
- Launches require a prepared plan.  One ``b12x.preparation`` plan per runtime
  declares every supported dtype and also carries the all-gather launcher.
  It is compiled, materialized (launchers plus the alignment and padded-gather
  scratch) and primed here, inside the communicator constructor: before any
  forward pass and before CUDA graph capture, under which the runtime refuses
  to compile or allocate.  Priming runs one 16-byte all-reduce per dtype and
  one all-gather in lock step on every rank.  A second vote makes the outcome
  rank-invariant, so a rank whose priming failed disables every rank.
- Dispatch is rank-invariant: eligibility depends on dtype, shape, contiguity
  and size, never on pointer values, so all ranks route the same collective.
- Failures after setup are fail-stop, never a fallback.  A wait that times out
  poisons the runtime, later launches do nothing, and ``check_health`` (which
  the worker calls once each step's output is on the host) raises, so the
  step's output never leaves the worker.  Peers starve on the stalled rank and
  raise too.
- One stream inside a capture.  Eager collectives are ordered across streams
  with an event; captured collectives must all be on one stream, which is how
  vLLM's ``graph_capture`` captures.
"""

from __future__ import annotations

import importlib
import re
import socket
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger

logger = init_logger(__name__)

REQUIRED_B12X_ROCE_API_VERSION = 1

# Every runtime dtype is prepared, named as the plan contract encodes it.
_PREPARED_DTYPES: dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
# The runtime moves 16-byte packs; smaller limits cannot route anything.
_PACK_BYTES = 16
_PLAN_NAME = "vllm.b12x_roce"

_BYTE_SIZE_MULTIPLIERS = {
    "": 1,
    "b": 1,
    "k": 1 << 10,
    "kb": 1 << 10,
    "kib": 1 << 10,
    "m": 1 << 20,
    "mb": 1 << 20,
    "mib": 1 << 20,
    "g": 1 << 30,
    "gb": 1 << 30,
    "gib": 1 << 30,
    "t": 1 << 40,
    "tb": 1 << 40,
    "tib": 1 << 40,
}


def _parse_byte_size(value: str) -> int:
    """Parse ``"2MB"``-style sizes with binary multipliers (``2MB`` = 2 MiB).

    Raises:
        ValueError: When the text is not an integer with an optional unit.

    """
    match = re.fullmatch(r"\s*([+-]?\d+)\s*([kmgt]?i?b?)?\s*", value.lower())
    if match is None:
        raise ValueError(f"invalid byte size: {value!r}")
    suffix = match.group(2) or ""
    try:
        return int(match.group(1)) * _BYTE_SIZE_MULTIPLIERS[suffix]
    except KeyError as exc:
        raise ValueError(f"invalid byte-size suffix: {suffix!r}") from exc


def _import_roce() -> ModuleType:
    """Import the runtime (a function so tests can substitute a fake)."""
    return importlib.import_module("b12x.comm.roce")


def _import_preparation() -> SimpleNamespace:
    """Import the preparation surface the vendored runtime is driven through.

    ``prepared_call`` and ``prepared_gather_call`` are the runtime's own
    priming factories, as used by FlashInfer's RoCE benchmark.
    """
    preparation = importlib.import_module("b12x.preparation")
    roce_preparation = importlib.import_module("b12x.comm.roce._preparation")
    return SimpleNamespace(
        PreparationSession=preparation.PreparationSession,
        PreparedCall=preparation.PreparedCall,
        prepared_call=roce_preparation.prepared_call,
        prepared_gather_call=roce_preparation.prepared_gather_call,
    )


def _load_proxy_library(roce: ModuleType) -> None:
    """Build (first use) and load the runtime's C RDMA proxy library.

    The runtime constructor does this before its first setup exchange, where
    a rank-local failure (no C compiler, no libibverbs) would leave the other
    ranks waiting; probing here moves that failure into the vote.
    """
    importlib.import_module(f"{roce.__name__}._proxy").load()


class B12xRoceAllReduce:
    """Route eligible tensor-parallel all-reduces and all-gathers to RoCEnante."""

    backend_name = "ROCENANTE"

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        """Vote, construct and prepare the runtime on every rank, or on none.

        Args:
            group: The tensor-parallel CPU (gloo) group; carries the votes and
                the runtime's setup exchange.
            device: This rank's CUDA device.

        """
        self.disabled = True
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.max_size = 0
        self.max_gather_size = 0
        self._runtime: Any = None
        self._plan: Any = None
        self._session: Any = None
        self._announced_reduce = False
        self._announced_gather = False

        # Both checks give the same answer on every rank.
        if self.world_size < 2:
            return
        if all(in_the_same_node_as(group, source_rank=0)):
            logger.info("RoCEnante skipped: the group is single-node.")
            return

        reason, limits, host = self._local_capability()
        verdict, hosts = self._capability_vote(reason, limits, host)
        if verdict is not None:
            logger.warning("RoCEnante disabled on every rank: %s", verdict)
            return
        assert limits is not None
        self.max_size, self.max_gather_size = limits

        roce = _import_roce()
        try:
            # The setup exchange runs over the CPU (gloo) group: the torch NCCL
            # group would create a NCCL communicator vLLM otherwise never
            # needs, costing ~3.4 GB of unified memory per rank on Spark.
            runtime = roce.AllReduce.from_exchange_group(
                exchange_group=group,
                device=device,
                max_size=self.max_size,
                max_gather_bytes=self.max_gather_size,
            )
        except Exception as exc:
            # The constructor exchanges its failures, so every rank is here.
            logger.warning("RoCEnante initialization failed on every rank: %s", exc)
            return

        error: str | None = None
        plan = session = None
        try:
            plan, session = self._prepare(roce, runtime, hosts)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        verdict = self._agreement_vote(error)
        if verdict is not None:
            logger.warning(
                "RoCEnante disabled on every rank: plan preparation failed (%s)",
                verdict,
            )
            self._release(session, runtime)
            return

        self._runtime, self._plan, self._session = runtime, plan, session
        self.disabled = False
        if self.rank == 0:
            logger.info(
                "Using RoCEnante (b12x one-shot RoCE collectives): world=%d, "
                "hcas=%s, all-reduce max=%d bytes, all-gather shard max=%d bytes, "
                "prepared dtypes=%s.",
                self.world_size,
                ",".join(getattr(runtime, "hca_names", ())),
                self.max_size,
                self.max_gather_size,
                ",".join(_PREPARED_DTYPES.values()),
            )

    # -- setup -------------------------------------------------------------------

    def _local_capability(
        self,
    ) -> tuple[str | None, tuple[int, int] | None, str]:
        """Evaluate this rank's ability to take part, without any collective.

        Returns:
            The reason this rank cannot take part (None when it can), the
            parsed ``(max_size, max_gather_size)`` limits (None on failure) and
            this rank's host name.

        """
        host = socket.gethostname()
        try:
            limits = (
                _parse_byte_size(envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE),
                _parse_byte_size(envs.VLLM_ROCE_ALLGATHER_MAX_SIZE),
            )
        except ValueError as exc:
            return f"invalid RoCEnante size limit: {exc}", None, host
        max_size, max_gather = limits
        if max_size < _PACK_BYTES:
            return (
                f"VLLM_ROCE_ALLREDUCE_MAX_SIZE={max_size} is below {_PACK_BYTES} bytes",
                None,
                host,
            )
        if max_gather != 0 and max_gather < _PACK_BYTES:
            return (
                f"VLLM_ROCE_ALLGATHER_MAX_SIZE={max_gather} must be 0 (all-gather "
                f"off) or at least {_PACK_BYTES} bytes",
                None,
                host,
            )
        config = get_current_vllm_config_or_none()
        if config is not None and config.parallel_config.use_ubatching:
            return (
                "micro-batching (DBO) issues collectives from several streams; "
                "the runtime has one channel",
                None,
                host,
            )
        try:
            roce = _import_roce()
        except Exception as exc:  # missing package or a broken native import
            return f"b12x.comm.roce is not importable: {exc}", None, host
        api = getattr(roce, "API_VERSION", None)
        if api != REQUIRED_B12X_ROCE_API_VERSION:
            return (
                f"b12x.comm.roce API version {api}, adapter needs "
                f"{REQUIRED_B12X_ROCE_API_VERSION}",
                None,
                host,
            )
        if self.world_size not in roce.SUPPORTED_WORLD_SIZES:
            return f"world size {self.world_size} is unsupported", None, host
        try:
            supported = roce.is_supported(self.device)
        except Exception as exc:
            return f"capability probe failed: {exc}", None, host
        if not supported:
            return "needs an integrated GPU with an active RDMA device", None, host
        try:
            _load_proxy_library(roce)
        except Exception as exc:
            return f"RDMA proxy library unavailable: {exc}", None, host
        return None, limits, host

    def _all_gather_object(self, obj: object) -> list[Any]:
        """Gather one picklable object from every rank over the CPU group."""
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, obj, group=self.group)
        return gathered

    def _capability_vote(
        self, reason: str | None, limits: tuple[int, int] | None, host: str
    ) -> tuple[str | None, list[str]]:
        """Exchange every rank's capability result.

        Args:
            reason: This rank's reason for not taking part, or None.
            limits: This rank's parsed ``(max_size, max_gather_size)``.
            host: This rank's host name, for the plan's peer metadata.

        Returns:
            None when every rank can proceed with identical limits, else the
            text naming the ranks that cannot or whose limits differ; and the
            host names by rank.

        """
        votes = self._all_gather_object((reason, limits, host))
        hosts = [str(vote[2]) for vote in votes]
        failures = [f"rank {i}: {vote[0]}" for i, vote in enumerate(votes) if vote[0]]
        if failures:
            return "; ".join(failures), hosts
        reference = votes[0][1]
        differing = [
            f"rank {i}: {vote[1]}"
            for i, vote in enumerate(votes)
            if vote[1] != reference
        ]
        if differing:
            return (
                f"size limits differ across ranks (rank 0: {reference}; "
                + "; ".join(differing)
                + ")"
            ), hosts
        return None, hosts

    def _agreement_vote(self, error: str | None) -> str | None:
        """Return None when no rank reported an error, else the errors by rank."""
        votes = self._all_gather_object(error)
        failures = [f"rank {i}: {e}" for i, e in enumerate(votes) if e is not None]
        return "; ".join(failures) if failures else None

    def _build_plan(self, roce: ModuleType, runtime: Any, hosts: list[str]) -> Any:
        """Declare one plan for every supported dtype and the gather launcher.

        The vendored ``plan`` materializes every declared all-reduce launcher
        and the all-gather launcher together, and ``all_gather`` accepts the
        same plan, as FlashInfer's RoCE benchmark uses it.
        """
        query = roce.query_from_runtime(
            runtime,
            surface="AllReduce.all_reduce",
            call={"dtypes": tuple(_PREPARED_DTYPES.values())},
            topology="roce_rdma",
            peer_hosts=tuple(hosts),
        )
        return roce.plan(query, runtime=runtime)

    def _prepare(
        self, roce: ModuleType, runtime: Any, hosts: list[str]
    ) -> tuple[Any, Any]:
        """Compile, materialize, prime and publish the runtime's plan.

        Materialization allocates the alignment and padded-gather scratch, so
        the runtime neither compiles nor allocates under CUDA graph capture.
        Every rank is inside this constructor, so the priming collectives run
        in lock step; they also load every native program before first use.

        Returns:
            The published plan and the preparation session that owns it.

        Raises:
            Exception: Whatever failed; the caller exchanges it in a vote.

        """
        prep = _import_preparation()
        plan = self._build_plan(roce, runtime, hosts)
        seeds = [
            torch.zeros(_PACK_BYTES // dtype.itemsize, dtype=dtype, device=self.device)
            for dtype in _PREPARED_DTYPES
        ]
        prime_gather = self.max_gather_size >= _PACK_BYTES

        def prime(state: Any) -> Any:
            calls = [prep.prepared_call(state, inp=seed) for seed in seeds]
            if prime_gather:
                calls.append(prep.prepared_gather_call(state, inp=seeds[0]))

            def run() -> None:
                for call in calls:
                    call.run()

            return prep.PreparedCall(run=run)

        # In-process compilation: vLLM workers may be daemonic processes,
        # which cannot start the compile-worker pool.
        session = prep.PreparationSession(
            device=self.device, autotune=False, compile_workers=0
        )
        try:
            session.prepare(
                (plan.request(name=_PLAN_NAME, prepare_call=prime),),
                autotune=False,
            )
            if plan.prepared is None:
                raise RuntimeError("the RoCEnante plan was not published")
            # A priming launch returns before its waits complete; a peer that
            # never launched surfaces only once the kernel times out.
            torch.cuda.synchronize(self.device)
            runtime.check_health()
        except BaseException:
            self._close_session(session)
            raise
        return plan, session

    @staticmethod
    def _close_session(session: Any) -> None:
        """Release a preparation session; it refuses calls from other threads."""
        if session is None:
            return
        try:
            session.close()
        except Exception as exc:
            logger.debug("RoCEnante preparation session close failed: %s", exc)

    def _release(self, session: Any, runtime: Any) -> None:
        """Release the plan's session, then the runtime."""
        self._close_session(session)
        if runtime is not None:
            runtime.close()

    # -- dispatch ----------------------------------------------------------------

    def check_health(self) -> None:
        """Fail-stop check of the runtime.

        Raises:
            RuntimeError: When a RoCEnante wait timed out or its proxy died.

        """
        runtime = self._runtime
        if not self.disabled and runtime is not None:
            runtime.check_health()

    def should_all_reduce(self, inp: torch.Tensor) -> bool:
        return (
            not self.disabled
            and inp.dtype in _PREPARED_DTYPES
            and self._runtime.should_allreduce(inp)
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if not self._announced_reduce:
            self._announced_reduce = True
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-reduce is live: first routed all-reduce is %d bytes "
                "(%s); larger messages stay on the next backend (above %d bytes).",
                inp.numel() * inp.element_size(),
                str(inp.dtype).removeprefix("torch."),
                self.max_size,
            )
        return self._runtime.all_reduce(inp, plan=self._plan)

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return (
            not self.disabled
            and self.max_gather_size > 0
            and self._runtime.should_all_gather(inp, dim)
        )

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate equal shards along ``dim`` (0 or last) in output layout."""
        if not self._announced_gather:
            self._announced_gather = True
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-gather is live: first routed shard is %s %s "
                "along dim %d.",
                tuple(inp.shape),
                str(inp.dtype).removeprefix("torch."),
                dim,
            )
        return self._runtime.all_gather(inp, dim=dim, plan=self._plan)

    # -- capture / lifecycle -----------------------------------------------------

    @contextmanager
    def capture(self):
        """Enter the runtime's capture context around vLLM's graph capture.

        Launchers and scratch were prepared at construction; a plan that is no
        longer prepared is refused here rather than inside a capture.
        """
        runtime = self._runtime
        if self.disabled or runtime is None:
            yield
            return
        if self._plan is None or self._plan.prepared is None:
            raise RuntimeError("RoCEnante plan is not prepared; refusing to capture")
        with runtime.capture():
            yield

    def destroy(self) -> None:
        """Disable dispatch, release the plan, then close the runtime."""
        self.disabled = True
        session, self._session = self._session, None
        runtime, self._runtime = self._runtime, None
        self._plan = None
        self._release(session, runtime)
