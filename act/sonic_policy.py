"""ACT inference adapter for the GR00T/GEAR-SONIC PolicyServer contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from .data import NormalizationStats
from .model import ACTPolicy
from .server import WireModalityConfig
from .sonic_contract import (
    SONIC_ACTION_DIM,
    SONIC_ACTION_FIELDS,
    SONIC_LANGUAGE_KEY,
    SONIC_QPOS_DIM,
    SONIC_STATE_FIELDS,
    SONIC_VIDEO_KEY,
    contract_dict,
)


class ACTSonicPolicy:
    """Expose an ACT checkpoint through the interface used by SONIC.

    Input follows ``Gr00tPolicy`` (batched arrays with a temporal dimension),
    and output follows SONIC latent protocol v4::

        {
            "motion_token":      float32[B, T, 64],
            "left_hand_joints":  float32[B, T, 7],
            "right_hand_joints": float32[B, T, 7],
        }

    ACT predicts the concatenated, normalized 78-D vector.  This adapter is
    deliberately responsible for both inverse normalization and splitting;
    sending the raw model tensor directly to SONIC is incorrect.
    """

    def __init__(
        self,
        model: ACTPolicy,
        stats: NormalizationStats,
        *,
        device: str | torch.device = "cuda",
        image_size: int = 224,
        strict: bool = True,
    ) -> None:
        if model.qpos_dim != SONIC_QPOS_DIM:
            raise ValueError(
                f"SONIC requires qpos_dim={SONIC_QPOS_DIM}, checkpoint has {model.qpos_dim}"
            )
        if model.action_dim != SONIC_ACTION_DIM:
            raise ValueError(
                f"SONIC requires action_dim={SONIC_ACTION_DIM}, checkpoint has {model.action_dim}"
            )
        if image_size < 32:
            raise ValueError("image_size must be at least 32")
        stat_shapes = (
            stats.qpos_mean.shape,
            stats.qpos_std.shape,
            stats.action_mean.shape,
            stats.action_std.shape,
        )
        expected_shapes = (
            (SONIC_QPOS_DIM,),
            (SONIC_QPOS_DIM,),
            (SONIC_ACTION_DIM,),
            (SONIC_ACTION_DIM,),
        )
        if stat_shapes != expected_shapes:
            raise ValueError(f"normalization statistics do not match SONIC: {stat_shapes}")
        if not all(
            np.isfinite(value).all()
            for value in (stats.qpos_mean, stats.qpos_std, stats.action_mean, stats.action_std)
        ):
            raise ValueError("normalization statistics contain NaN or Inf")
        if np.any(stats.qpos_std <= 0) or np.any(stats.action_std <= 0):
            raise ValueError("normalization standard deviations must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is visible")
        self.model = model.to(self.device).eval()
        self.stats = stats
        self.image_size = int(image_size)
        self.strict = strict

        self._qpos_mean = torch.as_tensor(
            stats.qpos_mean, device=self.device, dtype=torch.float32
        )
        self._qpos_std = torch.as_tensor(
            stats.qpos_std, device=self.device, dtype=torch.float32
        )
        self._action_mean = torch.as_tensor(
            stats.action_mean, device=self.device, dtype=torch.float32
        )
        self._action_std = torch.as_tensor(
            stats.action_std, device=self.device, dtype=torch.float32
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        image_size: int | None = None,
        strict: bool = True,
    ) -> "ACTSonicPolicy":
        """Load a checkpoint written by ``train.py``."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {"model", "model_config", "normalization"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"{checkpoint_path} is missing checkpoint keys: {sorted(missing)}")

        config = dict(checkpoint["model_config"])
        saved_contract = checkpoint.get("sonic_contract")
        if saved_contract is not None:
            expected_contract = contract_dict(int(config["chunk_size"]))
            for key in ("video_key", "state_fields", "action_fields", "action_horizon"):
                if saved_contract.get(key) != expected_contract[key]:
                    raise ValueError(
                        f"{checkpoint_path} has an incompatible SONIC contract field {key!r}: "
                        f"{saved_contract.get(key)!r} (expected {expected_contract[key]!r})"
                    )
        model = ACTPolicy(**config)
        model.load_state_dict(checkpoint["model"])
        stats = NormalizationStats.from_dict(checkpoint["normalization"])
        if image_size is None:
            image_size = int(checkpoint.get("args", {}).get("image_size", 224))
        return cls(model, stats, device=device, image_size=image_size, strict=strict)

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate the subset of the GR00T observation used by ACT."""
        for modality in ("video", "state"):
            if modality not in observation or not isinstance(observation[modality], dict):
                raise ValueError(f"observation['{modality}'] must be a dictionary")

        if SONIC_VIDEO_KEY not in observation["video"]:
            raise KeyError(f"missing video field {SONIC_VIDEO_KEY!r}")
        video = observation["video"][SONIC_VIDEO_KEY]
        if not isinstance(video, np.ndarray) or video.dtype != np.uint8:
            raise TypeError(f"video.{SONIC_VIDEO_KEY} must be a uint8 numpy array")
        if video.ndim != 5 or video.shape[1] != 1 or video.shape[-1] != 3:
            raise ValueError(
                f"video.{SONIC_VIDEO_KEY} must have shape [B,1,H,W,3], got {video.shape}"
            )

        batch = video.shape[0]
        for key, width in SONIC_STATE_FIELDS:
            if key not in observation["state"]:
                raise KeyError(f"missing state field {key!r}")
            value = observation["state"][key]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise TypeError(f"state.{key} must be a float32 numpy array")
            if value.shape != (batch, 1, width):
                raise ValueError(
                    f"state.{key} must have shape [{batch},1,{width}], got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"state.{key} contains NaN or Inf")

    def _prepare_qpos(self, observation: dict[str, Any]) -> Tensor:
        values = [observation["state"][key][:, -1] for key, _ in SONIC_STATE_FIELDS]
        qpos = torch.from_numpy(np.ascontiguousarray(np.concatenate(values, axis=-1))).to(
            self.device
        )
        return (qpos - self._qpos_mean) / self._qpos_std

    def _prepare_image(self, observation: dict[str, Any]) -> Tensor:
        frames = observation["video"][SONIC_VIDEO_KEY][:, -1]
        resize_size = self.image_size + 32
        top = left = (resize_size - self.image_size) // 2
        processed: list[np.ndarray] = []
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        for frame in frames:
            image = cv2.resize(frame, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
            image = image[top : top + self.image_size, left : left + self.image_size]
            image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
            processed.append((image - mean) / std)
        return torch.from_numpy(np.stack(processed)).to(self.device)

    @torch.inference_mode()
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        outputs = self.model(self._prepare_qpos(observation), self._prepare_image(observation))
        normalized = outputs["actions"]
        pad_logits = outputs["pad_logits"]
        assert normalized is not None and pad_logits is not None

        # The checkpoint was trained in mean/std space.  SONIC consumes the
        # original token and joint values, not these normalized predictions.
        action = normalized.float() * self._action_std + self._action_mean
        action_np = np.ascontiguousarray(action.cpu().numpy(), dtype=np.float32)

        split: dict[str, np.ndarray] = {}
        start = 0
        for key, width in SONIC_ACTION_FIELDS:
            split[key] = np.ascontiguousarray(action_np[..., start : start + width])
            start += width
        info = {
            "is_pad": np.ascontiguousarray((pad_logits > 0).cpu().numpy()),
        }
        return split, info

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options)
        if self.strict:
            self.check_action(action)
        return action, info

    def check_action(self, action: dict[str, Any]) -> None:
        batch = None
        for key, width in SONIC_ACTION_FIELDS:
            if key not in action:
                raise KeyError(f"missing action field {key!r}")
            value = action[key]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise TypeError(f"action.{key} must be a float32 numpy array")
            if value.ndim != 3 or value.shape[1:] != (self.model.chunk_size, width):
                raise ValueError(
                    f"action.{key} must have shape [B,{self.model.chunk_size},{width}], "
                    f"got {value.shape}"
                )
            if batch is None:
                batch = value.shape[0]
            elif value.shape[0] != batch:
                raise ValueError("all action fields must have the same batch size")
            if not np.isfinite(value).all():
                raise ValueError(f"action.{key} contains NaN or Inf")

    def get_modality_config(self) -> dict[str, Any]:
        """Return configs serialized into GR00T ``ModalityConfig`` objects."""
        absolute = [
            {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}
            for _ in SONIC_ACTION_FIELDS
        ]
        return {
            "video": WireModalityConfig(delta_indices=[0], modality_keys=[SONIC_VIDEO_KEY]),
            "state": WireModalityConfig(
                delta_indices=[0], modality_keys=[key for key, _ in SONIC_STATE_FIELDS]
            ),
            "action": WireModalityConfig(
                delta_indices=list(range(self.model.chunk_size)),
                modality_keys=[key for key, _ in SONIC_ACTION_FIELDS],
                action_configs=absolute,
            ),
            "language": WireModalityConfig(
                delta_indices=[0], modality_keys=[SONIC_LANGUAGE_KEY]
            ),
        }

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del options
        return {}
