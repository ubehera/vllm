# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the RoCEnante adapter and its CudaCommunicator integration.

The RDMA protocol needs several DGX Spark nodes; these tests pin the adapter's
setup votes, plan preparation, rank-invariant dispatch, capture and teardown
against a scripted process group and a fake runtime, plus the call contract
with the runtime FlashInfer vendors (skipped when it is not installed).
"""

import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.envs as envs
from vllm.distributed.device_communicators import b12x_roce_all_reduce as roce_ar
from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.parallel_state import GraphCaptureContext, GroupCoordinator

RANK, WORLD = 1, 4
HOSTS = ["node-1", "node-2", "node-3", "node-4"]
LIMITS = (2 << 20, 16 << 20)


class _FakeRuntime:
    hca_names = ("rocep1s0f1", "roceP2p1s0f1")

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.events: list[object] = []
        self.eligible = True
        self.health_error: str | None = None

    def should_allreduce(self, inp):
        self.events.append("should_allreduce")
        return self.eligible

    def should_all_gather(self, inp, dim):
        self.events.append(("should_all_gather", dim))
        return self.eligible

    def all_reduce(self, inp, *, plan):
        self.events.append(("all_reduce", plan))
        return "reduced"

    def all_gather(self, inp, *, dim, plan):
        self.events.append(("all_gather", dim, plan))
        return "gathered"

    def check_health(self):
        self.events.append("check_health")
        if self.health_error is not None:
            raise RuntimeError(self.health_error)

    @contextmanager
    def capture(self):
        self.events.append("capture_enter")
        yield self
        self.events.append("capture_exit")

    def close(self):
        self.events.append("close")


def _fake_roce(*, supported=True, api=1, construct_error=None):
    runtimes: list[_FakeRuntime] = []

    def from_exchange_group(**kwargs):
        if construct_error is not None:
            raise RuntimeError(construct_error)
        runtime = _FakeRuntime(**kwargs)
        runtimes.append(runtime)
        return runtime

    return SimpleNamespace(
        __name__="fake_roce",
        API_VERSION=api,
        SUPPORTED_WORLD_SIZES=tuple(range(2, 17)),
        is_supported=lambda device: supported,
        AllReduce=SimpleNamespace(from_exchange_group=from_exchange_group),
        runtimes=runtimes,
    )


def _peer_capability(reason=None, limits=LIMITS):
    """Every rank's capability vote: this rank's own, peers OK unless told."""

    def build(local):
        votes = [(None, LIMITS, host) for host in HOSTS]
        votes[RANK] = local
        votes[3] = (reason, limits if reason is None else None, HOSTS[3])
        return votes

    return build


def _peer_agreement(error=None):
    def build(local):
        votes: list[object] = [None] * WORLD
        votes[RANK] = local
        votes[2] = error
        return votes

    return build


@pytest.fixture
def cluster(monkeypatch):
    """Rank 1 of a 4-rank TP group spanning nodes, with scripted exchanges."""
    state = SimpleNamespace(rounds=[], sent=[], roce=_fake_roce(), prepared=[])
    monkeypatch.setattr(roce_ar.dist, "get_rank", lambda group=None: RANK)
    monkeypatch.setattr(roce_ar.dist, "get_world_size", lambda group=None: WORLD)
    monkeypatch.setattr(
        roce_ar,
        "in_the_same_node_as",
        lambda group, source_rank=0: [True, False, False, False],
    )

    def all_gather_object(gathered, obj, group=None):
        assert group is state.group
        state.sent.append(obj)
        gathered[:] = state.rounds.pop(0)(obj)

    monkeypatch.setattr(roce_ar.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(roce_ar.socket, "gethostname", lambda: HOSTS[RANK])
    monkeypatch.setattr(roce_ar, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(roce_ar, "_import_roce", lambda: state.roce)
    monkeypatch.setattr(roce_ar, "_load_proxy_library", lambda roce: None)
    monkeypatch.delenv("VLLM_ROCE_ALLREDUCE_MAX_SIZE", raising=False)
    monkeypatch.delenv("VLLM_ROCE_ALLGATHER_MAX_SIZE", raising=False)
    state.group = object()
    state.plan = SimpleNamespace(prepared=object())
    state.session = Mock()

    def prepare(self, roce, runtime, hosts):
        state.prepared.append((roce, runtime, list(hosts)))
        return state.plan, state.session

    monkeypatch.setattr(roce_ar.B12xRoceAllReduce, "_prepare", prepare)
    return state


def _build(state) -> roce_ar.B12xRoceAllReduce:
    return roce_ar.B12xRoceAllReduce(group=state.group, device=torch.device("cuda:0"))


# -- size parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2MB", 2 << 20),
        ("16MB", 16 << 20),
        ("84KB", 84 << 10),
        ("1024", 1024),
        (" 4 KiB ", 4 << 10),
        ("1g", 1 << 30),
        ("0", 0),
    ],
)
def test_parse_byte_size(text, expected):
    assert roce_ar._parse_byte_size(text) == expected


@pytest.mark.parametrize("text", ["", "2XB", "1.5MB", "MB", "2 M B"])
def test_parse_byte_size_rejects(text):
    with pytest.raises(ValueError):
        roce_ar._parse_byte_size(text)


# -- setup votes ------------------------------------------------------------------


def test_single_rank_declines_without_collectives(cluster, monkeypatch):
    monkeypatch.setattr(roce_ar.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(
        roce_ar, "in_the_same_node_as", Mock(side_effect=AssertionError)
    )
    comm = _build(cluster)
    assert comm.disabled
    assert cluster.sent == [] and cluster.roce.runtimes == []


def test_single_node_group_declines_before_voting(cluster, monkeypatch):
    monkeypatch.setattr(
        roce_ar, "in_the_same_node_as", lambda group, source_rank=0: [True] * WORLD
    )
    comm = _build(cluster)
    assert comm.disabled
    assert cluster.sent == [] and cluster.roce.runtimes == []


def test_success_constructs_prepares_and_enables(cluster):
    cluster.rounds = [_peer_capability(), _peer_agreement()]
    comm = _build(cluster)
    assert not comm.disabled
    assert (comm.max_size, comm.max_gather_size) == LIMITS
    assert cluster.sent == [(None, LIMITS, HOSTS[RANK]), None]
    (runtime,) = cluster.roce.runtimes
    assert runtime.kwargs == {
        "exchange_group": cluster.group,
        "device": torch.device("cuda:0"),
        "max_size": LIMITS[0],
        "max_gather_bytes": LIMITS[1],
    }
    assert cluster.prepared == [(cluster.roce, runtime, HOSTS)]


def test_rank_without_capability_disables_every_rank(cluster):
    cluster.rounds = [_peer_capability(reason="no active RDMA device")]
    comm = _build(cluster)
    assert comm.disabled
    assert cluster.roce.runtimes == []  # nobody enters the collective constructor
    assert cluster.prepared == []


def test_this_rank_without_capability_reports_it(cluster):
    cluster.roce = _fake_roce(supported=False)
    cluster.rounds = [_peer_capability()]
    comm = _build(cluster)
    assert comm.disabled
    reason, limits, host = cluster.sent[0]
    assert "integrated GPU" in reason and limits is None and host == HOSTS[RANK]
    assert cluster.roce.runtimes == []


def test_api_version_mismatch_is_voted(cluster):
    cluster.roce = _fake_roce(api=2)
    cluster.rounds = [_peer_capability()]
    assert _build(cluster).disabled
    assert "API version 2" in cluster.sent[0][0]


def test_missing_proxy_library_is_voted(cluster, monkeypatch):
    def fail(roce):
        raise RuntimeError("no C compiler")

    monkeypatch.setattr(roce_ar, "_load_proxy_library", fail)
    cluster.rounds = [_peer_capability()]
    assert _build(cluster).disabled
    assert "no C compiler" in cluster.sent[0][0]


def test_import_failure_is_voted(cluster, monkeypatch):
    def fail():
        raise ImportError("no b12x")

    monkeypatch.setattr(roce_ar, "_import_roce", fail)
    cluster.rounds = [_peer_capability()]
    assert _build(cluster).disabled
    assert "not importable" in cluster.sent[0][0]


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("VLLM_ROCE_ALLREDUCE_MAX_SIZE", "2XB", "invalid RoCEnante size limit"),
        ("VLLM_ROCE_ALLREDUCE_MAX_SIZE", "8", "below 16 bytes"),
        ("VLLM_ROCE_ALLGATHER_MAX_SIZE", "8", "must be 0"),
    ],
)
def test_invalid_limits_are_voted_not_raised(cluster, monkeypatch, name, value, match):
    monkeypatch.setenv(name, value)
    cluster.rounds = [_peer_capability()]
    assert _build(cluster).disabled
    assert match in cluster.sent[0][0]


def test_micro_batching_is_voted(cluster, monkeypatch):
    config = SimpleNamespace(parallel_config=SimpleNamespace(use_ubatching=True))
    monkeypatch.setattr(roce_ar, "get_current_vllm_config_or_none", lambda: config)
    cluster.rounds = [_peer_capability()]
    assert _build(cluster).disabled
    assert "micro-batching" in cluster.sent[0][0]


def test_differing_limits_disable_every_rank(cluster):
    cluster.rounds = [_peer_capability(limits=(1 << 20, 16 << 20))]
    assert _build(cluster).disabled
    assert cluster.roce.runtimes == []


def test_zero_gather_limit_keeps_gathers_on_nccl(cluster, monkeypatch):
    monkeypatch.setenv("VLLM_ROCE_ALLGATHER_MAX_SIZE", "0")
    limits = (LIMITS[0], 0)

    def capability(local):
        return [(None, limits, host) for host in HOSTS]

    cluster.rounds = [capability, _peer_agreement()]
    comm = _build(cluster)
    assert not comm.disabled and comm.max_gather_size == 0
    assert not comm.should_all_gather(torch.zeros(4, 8), 1)
    assert cluster.roce.runtimes[0].kwargs["max_gather_bytes"] == 0


def test_constructor_failure_disables_without_another_vote(cluster):
    cluster.roce = _fake_roce(construct_error="connect failed")
    cluster.rounds = [_peer_capability()]
    assert _build(cluster).disabled
    assert len(cluster.sent) == 1
    assert cluster.prepared == []


def test_peer_preparation_failure_disables_and_releases(cluster):
    cluster.rounds = [_peer_capability(), _peer_agreement(error="timed out")]
    comm = _build(cluster)
    assert comm.disabled
    cluster.session.close.assert_called_once_with()
    assert cluster.roce.runtimes[0].events == ["close"]


def test_local_preparation_failure_is_exchanged(cluster, monkeypatch):
    def fail(self, roce, runtime, hosts):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(roce_ar.B12xRoceAllReduce, "_prepare", fail)
    cluster.rounds = [_peer_capability(), _peer_agreement()]
    comm = _build(cluster)
    assert comm.disabled
    assert cluster.sent[1] == "RuntimeError: compile failed"
    assert cluster.roce.runtimes[0].events == ["close"]


# -- dispatch, capture and teardown -----------------------------------------------


@pytest.fixture
def enabled(cluster):
    cluster.rounds = [_peer_capability(), _peer_agreement()]
    comm = _build(cluster)
    assert not comm.disabled
    return comm, cluster.roce.runtimes[0], cluster


def test_all_reduce_dispatch_uses_the_published_plan(enabled):
    comm, runtime, cluster = enabled
    assert comm.should_all_reduce(torch.zeros(8, dtype=torch.bfloat16))
    assert comm.all_reduce(torch.zeros(8, dtype=torch.bfloat16)) == "reduced"
    assert runtime.events[-1] == ("all_reduce", cluster.plan)
    # An unprepared dtype never reaches the runtime's own gate.
    runtime.events.clear()
    assert not comm.should_all_reduce(torch.zeros(8, dtype=torch.int32))
    assert runtime.events == []
    runtime.eligible = False
    assert not comm.should_all_reduce(torch.zeros(8, dtype=torch.float32))


def test_all_gather_dispatch_passes_dim_and_plan(enabled):
    comm, runtime, cluster = enabled
    shard = torch.zeros(4, 8, dtype=torch.int64)
    assert comm.should_all_gather(shard, 1)
    assert comm.all_gather(shard, 1) == "gathered"
    assert runtime.events[-2:] == [
        ("should_all_gather", 1),
        ("all_gather", 1, cluster.plan),
    ]


def test_capture_enters_runtime_capture(enabled):
    comm, runtime, cluster = enabled
    with comm.capture():
        runtime.events.append("body")
    assert runtime.events[-3:] == ["capture_enter", "body", "capture_exit"]


def test_capture_refuses_an_unprepared_plan(enabled):
    comm, runtime, cluster = enabled
    cluster.plan.prepared = None
    with pytest.raises(RuntimeError, match="not prepared"), comm.capture():
        pass
    assert "capture_enter" not in runtime.events


def test_check_health_propagates(enabled):
    comm, runtime, _ = enabled
    comm.check_health()
    runtime.health_error = "RoCE collective on rank 1 timed out"
    with pytest.raises(RuntimeError, match="timed out"):
        comm.check_health()


def test_destroy_disables_then_releases_session_before_runtime(enabled):
    comm, runtime, cluster = enabled
    order: list[str] = []
    cluster.session.close.side_effect = lambda: order.append("session")
    runtime.close = lambda: order.append("runtime")
    comm.destroy()
    assert comm.disabled and order == ["session", "runtime"]
    assert not comm.should_all_reduce(torch.zeros(8, dtype=torch.bfloat16))
    assert not comm.should_all_gather(torch.zeros(8), 0)
    comm.check_health()  # a destroyed adapter has nothing left to check
    with comm.capture():
        pass
    comm.destroy()  # idempotent
    assert order == ["session", "runtime"]


def test_disabled_adapter_never_consults_the_runtime(cluster):
    cluster.rounds = [_peer_capability(reason="no RDMA")]
    comm = _build(cluster)
    assert not comm.should_all_reduce(torch.zeros(8, dtype=torch.bfloat16))
    assert not comm.should_all_gather(torch.zeros(8), 0)
    comm.check_health()
    with comm.capture():
        pass
    comm.destroy()


# -- plan preparation -------------------------------------------------------------


def test_prepare_primes_every_dtype_and_gather_then_checks_health(monkeypatch):
    trace: list[object] = []
    plan = SimpleNamespace(prepared=None)
    roce = SimpleNamespace(
        query_from_runtime=lambda runtime, **kw: trace.append(("query", kw)) or "Q",
        plan=lambda query, *, runtime: trace.append(("plan", query)) or plan,
    )

    def request(*, name, prepare_call):
        return SimpleNamespace(name=name, prepare_call=prepare_call)

    plan.request = request

    class Session:
        def __init__(self, **kwargs):
            trace.append(("session", kwargs))

        def prepare(self, requests, *, autotune):
            (req,) = requests
            trace.append(("prepare", req.name, autotune))
            req.prepare_call("STATE").run()
            plan.prepared = object()

        def close(self):
            trace.append("session_close")

    prep = SimpleNamespace(
        PreparationSession=Session,
        PreparedCall=lambda run: SimpleNamespace(run=run),
        prepared_call=lambda state, *, inp: SimpleNamespace(
            run=lambda: trace.append(("reduce", state, inp.dtype, inp.numel()))
        ),
        prepared_gather_call=lambda state, *, inp: SimpleNamespace(
            run=lambda: trace.append(("gather", state, inp.dtype, inp.numel()))
        ),
    )
    monkeypatch.setattr(roce_ar, "_import_preparation", lambda: prep)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device=None: trace.append("synchronize")
    )
    runtime = Mock()
    runtime.check_health.side_effect = lambda: trace.append("check_health")

    comm = roce_ar.B12xRoceAllReduce.__new__(roce_ar.B12xRoceAllReduce)
    comm.device = torch.device("cpu")
    comm.max_gather_size = 16 << 20
    prepared_plan, session = comm._prepare(roce, runtime, HOSTS)
    assert prepared_plan is plan and isinstance(session, Session)

    assert trace == [
        (
            "query",
            {
                "surface": "AllReduce.all_reduce",
                "call": {"dtypes": ("bfloat16", "float16", "float32")},
                "topology": "roce_rdma",
                "peer_hosts": tuple(HOSTS),
            },
        ),
        ("plan", "Q"),
        ("session", {"device": comm.device, "autotune": False, "compile_workers": 0}),
        ("prepare", "vllm.b12x_roce", False),
        ("reduce", "STATE", torch.bfloat16, 8),
        ("reduce", "STATE", torch.float16, 8),
        ("reduce", "STATE", torch.float32, 4),
        ("gather", "STATE", torch.bfloat16, 8),
        "synchronize",
        "check_health",
    ]


def test_prepare_closes_its_session_when_the_runtime_is_poisoned(monkeypatch):
    sessions: list[Mock] = []
    plan = SimpleNamespace(prepared=object(), request=lambda **kw: kw)

    def make_session(**kwargs):
        session = Mock()
        sessions.append(session)
        return session

    prep = SimpleNamespace(
        PreparationSession=make_session,
        PreparedCall=Mock(),
        prepared_call=Mock(),
        prepared_gather_call=Mock(),
    )
    monkeypatch.setattr(roce_ar, "_import_preparation", lambda: prep)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    roce = SimpleNamespace(
        query_from_runtime=lambda runtime, **kw: "Q",
        plan=lambda query, *, runtime: plan,
    )
    runtime = Mock()
    runtime.check_health.side_effect = RuntimeError("timed out waiting for rank 3")

    comm = roce_ar.B12xRoceAllReduce.__new__(roce_ar.B12xRoceAllReduce)
    comm.device = torch.device("cpu")
    comm.max_gather_size = 0
    with pytest.raises(RuntimeError, match="rank 3"):
        comm._prepare(roce, runtime, HOSTS)
    sessions[0].close.assert_called_once_with()


# -- CudaCommunicator and GroupCoordinator integration ----------------------------


def _communicator(roce_comm) -> cuda_communicator.CudaCommunicator:
    communicator = cuda_communicator.CudaCommunicator.__new__(
        cuda_communicator.CudaCommunicator
    )
    communicator.roce_comm = roce_comm
    communicator.pynccl_comm = Mock(world_size=WORLD, disabled=False)
    communicator.qr_comm = None
    communicator.fi_pcie_ipc_ar_comm = None
    communicator.fi_ar_comm = Mock()
    communicator.fi_ar_comm.disabled = False
    communicator.aiter_ar_comm = None
    communicator.use_aiter_allreduce = False
    communicator.ca_comm = None
    communicator.symm_mem_comm = None
    communicator.all2all_manager = None
    communicator.world_size = WORLD
    return communicator


def test_class_default_has_no_roce_backend():
    communicator = cuda_communicator.CudaCommunicator.__new__(
        cuda_communicator.CudaCommunicator
    )
    assert communicator.roce_comm is None


def test_all_reduce_prefers_eligible_roce_before_every_other_backend(monkeypatch):
    monkeypatch.setattr(
        cuda_communicator, "should_nccl_symm_mem_allreduce", lambda *_: False
    )
    roce = Mock()
    roce.should_all_reduce.return_value = True
    communicator = _communicator(roce)
    inp = torch.zeros(8, dtype=torch.bfloat16)
    assert communicator.all_reduce(inp) is roce.all_reduce.return_value
    roce.all_reduce.assert_called_once_with(inp)
    communicator.fi_ar_comm.should_use_fi_ar.assert_not_called()
    communicator.pynccl_comm.all_reduce.assert_not_called()


def test_ineligible_all_reduce_falls_through_to_existing_chain(monkeypatch):
    monkeypatch.setattr(
        cuda_communicator, "should_nccl_symm_mem_allreduce", lambda *_: False
    )
    roce = Mock()
    roce.should_all_reduce.return_value = False
    communicator = _communicator(roce)
    communicator.fi_ar_comm.should_use_fi_ar.return_value = False
    inp = torch.zeros(1 << 20, dtype=torch.bfloat16)
    assert communicator.all_reduce(inp) is communicator.pynccl_comm.all_reduce(inp)
    roce.all_reduce.assert_not_called()


def test_all_gather_routes_eligible_shards_to_roce(monkeypatch):
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)
    roce = Mock()
    roce.should_all_gather.return_value = True
    communicator = _communicator(roce)
    shard = torch.zeros(4, 8)
    assert communicator.all_gather(shard, -1) is roce.all_gather.return_value
    roce.should_all_gather.assert_called_once_with(shard, 1)  # dim normalized
    roce.all_gather.assert_called_once_with(shard, 1)
    communicator.pynccl_comm.all_gather.assert_not_called()


def test_ineligible_all_gather_stays_on_pynccl(monkeypatch):
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)
    monkeypatch.setattr(cuda_communicator.current_platform, "is_rocm", lambda: False)
    roce = Mock()
    roce.should_all_gather.return_value = False
    communicator = _communicator(roce)
    out = communicator.all_gather(torch.zeros(4, 8), 0)
    assert out.shape == (4 * WORLD, 8)
    communicator.pynccl_comm.all_gather.assert_called_once()
    roce.all_gather.assert_not_called()


def test_all_gatherv_never_routes_to_roce(monkeypatch):
    # EP/DP all2all dispatch (allgather_reducescatter) uses all_gatherv.
    monkeypatch.setattr(cuda_communicator, "should_nccl_symm_mem_ag_rs", lambda: False)
    roce = Mock()
    communicator = _communicator(roce)
    communicator.use_aiter_ag_rs = False
    communicator.rank_in_group = RANK
    communicator.all_gatherv(torch.zeros(4, 8), 0, sizes=[4, 4, 4, 4])
    roce.should_all_gather.assert_not_called()
    communicator.pynccl_comm.all_gather.assert_called_once()


def test_destroy_releases_roce_first(monkeypatch):
    order: list[str] = []
    roce = Mock()
    roce.destroy.side_effect = lambda: order.append("roce")
    communicator = _communicator(roce)
    communicator.pynccl_comm.destroy.side_effect = lambda: order.append("pynccl")
    communicator.fi_ar_comm.destroy.side_effect = lambda: order.append("fi_ar")
    communicator.destroy()
    assert order == ["roce", "pynccl", "fi_ar"]
    assert communicator.roce_comm is None


def test_backend_selection_log_names_rocenante(monkeypatch):
    roce = Mock(disabled=False, backend_name="ROCENANTE")
    communicator = _communicator(roce)
    communicator.unique_name = "tp:0"
    communicator.pynccl_comm.disabled = True
    communicator.fi_ar_comm.disabled = True
    logged = Mock()
    monkeypatch.setattr(cuda_communicator.logger, "info_once", logged)
    communicator._log_all_reduce_backend_selection()
    enabled = logged.call_args.args[1]
    assert enabled == "['ROCENANTE']"
    assert "'ROCENANTE'" in logged.call_args.args[3]


def test_graph_capture_enters_the_roce_capture_context(monkeypatch):
    events: list[str] = []

    @contextmanager
    def capture():
        events.append("enter")
        yield
        events.append("exit")

    roce = Mock()
    roce.capture = capture
    communicator = _communicator(roce)
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.device_communicator = communicator
    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    with coordinator.graph_capture(GraphCaptureContext(stream)):
        events.append("body")
    assert events == ["enter", "body", "exit"]


# -- environment ------------------------------------------------------------------


def test_env_defaults_and_parsing(monkeypatch):
    variables = envs.environment_variables
    for name in (
        "VLLM_ENABLE_ROCE_ALLREDUCE",
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE",
        "VLLM_ROCE_ALLGATHER_MAX_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert variables["VLLM_ENABLE_ROCE_ALLREDUCE"]() is False
    assert variables["VLLM_ROCE_ALLREDUCE_MAX_SIZE"]() == "2MB"
    assert variables["VLLM_ROCE_ALLGATHER_MAX_SIZE"]() == "16MB"
    monkeypatch.setenv("VLLM_ENABLE_ROCE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_ROCE_ALLREDUCE_MAX_SIZE", "1MB")
    assert variables["VLLM_ENABLE_ROCE_ALLREDUCE"]() is True
    assert variables["VLLM_ROCE_ALLREDUCE_MAX_SIZE"]() == "1MB"
    # The defaults agree with the runtime's own defaults.
    assert roce_ar._parse_byte_size("2MB") == 2 * 1024 * 1024
    assert roce_ar._parse_byte_size("16MB") == 16 * 1024 * 1024


# -- contract with the runtime FlashInfer vendors ---------------------------------


def _vendored_roce():
    try:
        return roce_ar._import_roce(), roce_ar._import_preparation()
    except Exception as exc:  # not installed, or a native import failed
        pytest.skip(f"b12x.comm.roce unavailable: {exc}")


def test_vendored_runtime_contract():
    roce, prep = _vendored_roce()
    assert roce.API_VERSION == roce_ar.REQUIRED_B12X_ROCE_API_VERSION
    assert set(roce.SUPPORTED_DTYPES) == set(roce_ar._PREPARED_DTYPES)
    assert tuple(roce.SUPPORTED_WORLD_SIZES) == tuple(range(2, 17))
    assert roce_ar._parse_byte_size("2MB") == roce.DEFAULT_MAX_SIZE
    assert roce_ar._parse_byte_size("16MB") == roce.DEFAULT_MAX_GATHER_BYTES

    runtime_cls = roce.AllReduce
    factory = inspect.signature(runtime_cls.from_exchange_group).parameters
    assert {"exchange_group", "device", "max_size", "max_gather_bytes"} <= set(factory)
    for method, required in (
        ("all_reduce", {"plan"}),
        ("all_gather", {"plan", "dim"}),
    ):
        params = inspect.signature(getattr(runtime_cls, method)).parameters
        for name in required:
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    for method in ("should_allreduce", "should_all_gather", "capture"):
        assert callable(getattr(runtime_cls, method))
    for method in ("check_health", "close"):
        assert callable(getattr(runtime_cls, method))
    session = inspect.signature(prep.PreparationSession).parameters
    assert {"device", "autotune", "compile_workers"} <= set(session)
    assert "autotune" in inspect.signature(prep.PreparationSession.prepare).parameters
    assert "inp" in inspect.signature(prep.prepared_call).parameters
    assert "inp" in inspect.signature(prep.prepared_gather_call).parameters


def test_vendored_plan_accepts_the_adapter_query():
    """The adapter's real query builds and validates a plan without a GPU."""
    roce, _ = _vendored_roce()
    runtime = roce.AllReduce.__new__(roce.AllReduce)  # stand-in identity only
    runtime.world_size = 8
    runtime.rank = 3
    runtime.hca_names = ("rocep1s0f1", "roceP2p1s0f1")
    runtime.device = torch.device("cuda", 0)
    runtime.rail_count = 2
    runtime._threads = 512
    runtime._layout = SimpleNamespace(slots=2, flag_stride=64)
    comm = roce_ar.B12xRoceAllReduce.__new__(roce_ar.B12xRoceAllReduce)
    hosts = [f"node-{i}" for i in range(1, 9)]
    plan = comm._build_plan(roce, runtime, hosts)
    assert plan.component_id == "comm.roce"
    assert plan.prepared is None  # nothing compiled or materialized yet
    plan.contract.validate_query(plan.query, None)
    assert tuple(plan.query.call["dtypes"]) == ("bfloat16", "float16", "float32")
    assert plan.query.peer_hosts == tuple(hosts)
    assert plan.query.setup["hca_count"] == 2
    request = plan.request(name=roce_ar._PLAN_NAME, prepare_call=lambda state: None)
    assert request.plan is plan
