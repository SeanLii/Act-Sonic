from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from act.data import NormalizationStats
from act.server import MsgSerializer
from act.sonic_contract import SONIC_ACTION_FIELDS, SONIC_STATE_FIELDS
from act.sonic_policy import ACTSonicPolicy


class FakeACT(nn.Module):
    qpos_dim = 46
    action_dim = 78
    chunk_size = 3

    def __init__(self) -> None:
        super().__init__()
        self.seen_qpos: torch.Tensor | None = None
        self.seen_image: torch.Tensor | None = None

    def forward(self, qpos: torch.Tensor, image: torch.Tensor):
        self.seen_qpos = qpos.detach().clone()
        self.seen_image = image.detach().clone()
        batch = qpos.shape[0]
        values = torch.arange(78, device=qpos.device, dtype=qpos.dtype)
        actions = values.view(1, 1, 78).expand(batch, self.chunk_size, -1)
        return {
            "actions": actions,
            "pad_logits": torch.zeros(batch, self.chunk_size, device=qpos.device),
            "mu": None,
            "logvar": None,
        }


def make_stats() -> NormalizationStats:
    return NormalizationStats(
        qpos_mean=np.zeros(46, dtype=np.float32),
        qpos_std=np.ones(46, dtype=np.float32),
        action_mean=np.ones(78, dtype=np.float32),
        action_std=np.full(78, 2.0, dtype=np.float32),
    )


def make_observation(batch: int = 2) -> dict:
    state = {}
    offset = 0
    for key, width in SONIC_STATE_FIELDS:
        values = np.arange(offset, offset + width, dtype=np.float32)
        state[key] = np.broadcast_to(values, (batch, 1, width)).copy()
        offset += width
    return {
        "video": {
            "ego_view": np.zeros((batch, 1, 48, 64, 3), dtype=np.uint8),
        },
        "state": state,
        # ACT is image/state conditioned, but the official client includes this.
        "language": {"annotation.human.task_description": [["test"]] * batch},
    }


def test_sonic_output_is_denormalized_and_split() -> None:
    model = FakeACT()
    policy = ACTSonicPolicy(model, make_stats(), device="cpu", image_size=32)
    action, info = policy.get_action(make_observation())

    assert list(action) == [name for name, _ in SONIC_ACTION_FIELDS]
    assert action["motion_token"].shape == (2, 3, 64)
    assert action["left_hand_joints"].shape == (2, 3, 7)
    assert action["right_hand_joints"].shape == (2, 3, 7)
    assert all(value.dtype == np.float32 for value in action.values())

    expected = np.arange(78, dtype=np.float32) * 2.0 + 1.0
    concatenated = np.concatenate(list(action.values()), axis=-1)
    np.testing.assert_allclose(concatenated[0, 0], expected)
    assert info["is_pad"].shape == (2, 3)


def test_sonic_state_order_matches_training_vector() -> None:
    model = FakeACT()
    policy = ACTSonicPolicy(model, make_stats(), device="cpu", image_size=32)
    policy.get_action(make_observation(batch=1))

    assert model.seen_qpos is not None
    np.testing.assert_array_equal(
        model.seen_qpos.cpu().numpy()[0], np.arange(46, dtype=np.float32)
    )
    assert model.seen_image is not None
    assert model.seen_image.shape == (1, 3, 32, 32)


def test_sonic_rejects_wrong_motion_state_width() -> None:
    policy = ACTSonicPolicy(FakeACT(), make_stats(), device="cpu", image_size=32)
    observation = make_observation(batch=1)
    observation["state"]["left_arm"] = np.zeros((1, 1, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="state.left_arm"):
        policy.get_action(observation)


def test_normalization_stats_checkpoint_round_trip() -> None:
    original = make_stats()
    restored = NormalizationStats.from_dict(original.as_dict())
    np.testing.assert_array_equal(restored.qpos_mean, original.qpos_mean)
    np.testing.assert_array_equal(restored.action_std, original.action_std)


def test_policy_wire_serializer_preserves_numeric_arrays() -> None:
    action = {
        "motion_token": np.zeros((1, 3, 64), dtype=np.float32),
        "left_hand_joints": np.ones((1, 3, 7), dtype=np.float32),
    }
    restored = MsgSerializer.from_bytes(MsgSerializer.to_bytes((action, {})))
    assert isinstance(restored, list)
    np.testing.assert_array_equal(restored[0]["motion_token"], action["motion_token"])
    np.testing.assert_array_equal(restored[0]["left_hand_joints"], action["left_hand_joints"])
