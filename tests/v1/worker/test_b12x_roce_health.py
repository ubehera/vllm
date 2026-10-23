# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement of the RoCEnante fail-stop health check in the GPU worker.

The runtime's GPU behaviour needs a multi-node run; this pins where the worker
runs the check relative to the step's device-to-host completion: immediately
for a synchronous output (or ``None``), and only after ``get_output()`` for an
asynchronous output.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker, _roce_guarded, _RoceCheckedAsyncOutput


class _AsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, result: ModelRunnerOutput, trace: list[str]) -> None:
        self._result = result
        self._trace = trace
        self.kv_connector_output = object()

    def get_output(self) -> ModelRunnerOutput:
        self._trace.append("get_output")
        return self._result

    def get_kv_connector_output(self):
        return self.kv_connector_output


def _roce_comm(*, disabled: bool = False) -> Mock:
    comm = Mock()
    comm.disabled = disabled
    return comm


def _tp_group(monkeypatch, group) -> None:
    monkeypatch.setattr(gpu_worker_module, "get_tp_group", lambda: group)


def _with_roce(monkeypatch, comm) -> None:
    communicator = SimpleNamespace(roce_comm=comm)
    _tp_group(monkeypatch, SimpleNamespace(device_communicator=communicator))


def _sync_output() -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        sampled_token_ids=[[1]],
    )


def test_sync_output_is_checked_immediately(monkeypatch):
    comm = _roce_comm()
    _with_roce(monkeypatch, comm)
    output = _sync_output()
    assert _roce_guarded(output) is output
    comm.check_health.assert_called_once_with()


def test_none_output_is_checked_and_passed_through(monkeypatch):
    comm = _roce_comm()
    _with_roce(monkeypatch, comm)
    assert _roce_guarded(None) is None
    comm.check_health.assert_called_once_with()


def test_async_output_is_checked_after_completion(monkeypatch):
    trace: list[str] = []
    comm = _roce_comm()
    comm.check_health.side_effect = lambda: trace.append("check")
    _with_roce(monkeypatch, comm)
    result = _sync_output()
    inner = _AsyncOutput(result, trace)
    wrapped = _roce_guarded(inner)
    assert isinstance(wrapped, _RoceCheckedAsyncOutput)
    assert isinstance(wrapped, AsyncModelRunnerOutput)
    assert trace == []  # nothing checked before the copy to host completes
    assert wrapped.get_output() is result
    assert trace == ["get_output", "check"]
    # The executor salvages KV output through the wrapper when get_output fails.
    assert wrapped.get_kv_connector_output() is inner.kv_connector_output


def test_failure_propagates_from_sync_and_async(monkeypatch):
    comm = _roce_comm()
    comm.check_health.side_effect = RuntimeError("poisoned")
    _with_roce(monkeypatch, comm)
    with pytest.raises(RuntimeError, match="poisoned"):
        _roce_guarded(_sync_output())
    wrapped = _roce_guarded(_AsyncOutput(_sync_output(), []))
    with pytest.raises(RuntimeError, match="poisoned"):
        wrapped.get_output()


@pytest.mark.parametrize(
    "group",
    [
        SimpleNamespace(),  # a group stub without a device communicator
        SimpleNamespace(device_communicator=None),
        SimpleNamespace(device_communicator=SimpleNamespace()),  # no slot
        SimpleNamespace(device_communicator=SimpleNamespace(roce_comm=None)),
        SimpleNamespace(  # built, then voted off on every rank
            device_communicator=SimpleNamespace(roce_comm=_roce_comm(disabled=True))
        ),
    ],
)
def test_inactive_roce_means_no_check_and_no_wrapping(monkeypatch, group):
    _tp_group(monkeypatch, group)
    output = _sync_output()
    assert _roce_guarded(output) is output
    async_output = _AsyncOutput(output, [])
    assert _roce_guarded(async_output) is async_output
    comm = getattr(getattr(group, "device_communicator", None), "roce_comm", None)
    if comm is not None:
        comm.check_health.assert_not_called()


def test_sample_tokens_runs_the_check(monkeypatch):
    comm = _roce_comm()
    _with_roce(monkeypatch, comm)
    worker = Worker.__new__(Worker)  # no GPU, no init
    output = _sync_output()
    worker.model_runner = Mock()
    worker.model_runner.sample_tokens.return_value = output
    assert worker.sample_tokens(None) is output
    comm.check_health.assert_called_once_with()


def _execute_model(monkeypatch, *, last_pp_rank: bool, result):
    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=last_pp_rank,
        isend_tensor_dict=lambda tensors, all_gather_group=None: [None, None],
    )
    monkeypatch.setattr(gpu_worker_module, "get_pp_group", lambda: pp_group)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            ),
        ),
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(execute_model=lambda *args: result),
        annotate_profile=lambda scheduler_output: nullcontext(),
        _pp_send_work=[],
    )
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=4)
    return Worker.execute_model(worker, scheduler_output)


def test_execute_model_checks_its_output(monkeypatch):
    comm = _roce_comm()
    _with_roce(monkeypatch, comm)
    output = _sync_output()
    assert _execute_model(monkeypatch, last_pp_rank=True, result=output) is output
    comm.check_health.assert_called_once_with()


def test_earlier_pipeline_stage_checks_its_own_tp_group(monkeypatch):
    comm = _roce_comm()
    _with_roce(monkeypatch, comm)
    hidden = IntermediateTensors({"hidden_states": torch.zeros(1)})
    assert _execute_model(monkeypatch, last_pp_rank=False, result=hidden) is None
    comm.check_health.assert_called_once_with()
