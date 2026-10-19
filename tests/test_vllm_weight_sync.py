"""The public weight-update lifecycle is identical on CUDA and Ascend."""

from types import ModuleType, SimpleNamespace
import sys
from urllib.parse import urlparse

import pytest
import torch

from cs336_alignment import vllm_utils


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize("failure", [None, "send", "receive"])
def test_official_weight_update_protocol(monkeypatch, device_type, failure):
    calls = []
    sent = []
    group = object()
    parameter = SimpleNamespace(
        device=SimpleNamespace(type=device_type), dtype=torch.bfloat16, shape=(2, 3),
    )
    policy = SimpleNamespace(
        parameters=lambda: iter([parameter]),
        named_parameters=lambda: iter([("model.weight", parameter)]),
    )

    class Engine:
        @staticmethod
        def trainer_send_weights(iterator, trainer_args):
            assert trainer_args.group is group
            assert trainer_args.packed is True
            sent.extend(iterator)
            if failure == "send":
                raise RuntimeError("send failed")

    module = ModuleType("official_engine")
    prefix = "HCCL" if device_type == "npu" else "NCCL"
    setattr(module, prefix + "WeightTransferEngine", Engine)
    setattr(module, prefix + "TrainerSendWeightsArgs", lambda **kw: SimpleNamespace(**kw))
    module_path = (
        "vllm_ascend.distributed.weight_transfer.hccl_engine" if device_type == "npu"
        else "vllm.distributed.weight_transfer.nccl_engine"
    )
    monkeypatch.setitem(sys.modules, module_path, module)
    setter = lambda device: None
    if device_type == "npu":
        monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=setter), raising=False)
    else:
        monkeypatch.setattr(torch.cuda, "set_device", setter)

    def http(method, url, payload=None, timeout=60):
        path = urlparse(url).path
        calls.append((method, path, payload))
        if path == "/update_weights" and failure == "receive":
            raise RuntimeError("receive failed")
        return {}

    monkeypatch.setattr(vllm_utils, "_http_json", http)
    if failure:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            vllm_utils.sync_policy_weights(policy, "http://localhost:18081", group)
    else:
        vllm_utils.sync_policy_weights(policy, "http://localhost:18081", group)

    paths = [path for _, path, _ in calls]
    if failure:
        # An incomplete update stays paused; it must not be committed/resumed.
        assert "/finish_weight_update" not in paths
        assert "/resume" not in paths
    else:
        assert paths == [
            "/pause", "/start_weight_update", "/update_weights",
            "/finish_weight_update", "/reset_prefix_cache", "/resume",
        ]
    assert calls[1][2] == {"is_checkpoint_format": True}
    update = next(payload["update_info"] for _, path, payload in calls if path == "/update_weights")
    assert update == {
        "names": ["model.weight"], "dtype_names": ["bfloat16"],
        "shapes": [[2, 3]], "packed": True,
    }
    assert sent == [("model.weight", parameter)]
