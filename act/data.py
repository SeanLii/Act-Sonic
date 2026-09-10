"""Episode-aware dataset loader for the LeRobot-style Sonic dataset."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .sonic_contract import SONIC_ACTION_FIELDS, SONIC_STATE_FIELDS


@dataclass(frozen=True)
class NormalizationStats:
    qpos_mean: np.ndarray
    qpos_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def from_json(cls, path: Path) -> "NormalizationStats":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        values = {}
        for output_name, source_name in (
            ("qpos_mean", ("observation.state", "mean")),
            ("qpos_std", ("observation.state", "std")),
            ("action_mean", ("action", "mean")),
            ("action_std", ("action", "std")),
        ):
            value = np.asarray(raw[source_name[0]][source_name[1]], dtype=np.float32)
            if source_name[1] == "std":
                value = np.maximum(value, 1e-6)
            values[output_name] = value
        stats = cls(**values)
        shapes = (
            stats.qpos_mean.shape,
            stats.qpos_std.shape,
            stats.action_mean.shape,
            stats.action_std.shape,
        )
        if shapes != ((46,), (46,), (78,), (78,)):
            raise ValueError(
                "dataset dimensions must be qpos=46/action=78, got "
                f"qpos mean/std={shapes[0]}/{shapes[1]}, "
                f"action mean/std={shapes[2]}/{shapes[3]}"
            )
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("normalization statistics contain NaN or Inf")
        return stats

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NormalizationStats":
        """Restore the compact normalization dictionary stored in checkpoints."""
        values: dict[str, np.ndarray] = {}
        for name in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
            if name not in raw:
                raise ValueError(f"normalization is missing {name!r}")
            value = np.asarray(raw[name], dtype=np.float32)
            if name.endswith("_std"):
                value = np.maximum(value, 1e-6)
            values[name] = value
        stats = cls(**values)
        shapes = (
            stats.qpos_mean.shape,
            stats.qpos_std.shape,
            stats.action_mean.shape,
            stats.action_std.shape,
        )
        if shapes != ((46,), (46,), (78,), (78,)):
            raise ValueError(
                "normalization dimensions must be qpos=46/action=78, got "
                f"qpos mean/std={shapes[0]}/{shapes[1]}, "
                f"action mean/std={shapes[2]}/{shapes[3]}"
            )
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("normalization statistics contain NaN or Inf")
        return stats

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "qpos_mean": self.qpos_mean.tolist(),
            "qpos_std": self.qpos_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }


@dataclass
class Episode:
    index: int
    qpos: np.ndarray
    actions: np.ndarray
    video_path: Path


def validate_sonic_layout(dataset_dir: Path) -> None:
    """Reject metadata whose concatenation order is incompatible with SONIC."""
    path = dataset_dir / "meta" / "modality.json"
    if not path.is_file():
        return  # Older converted datasets did not always retain this file.
    with path.open("r", encoding="utf-8") as handle:
        modality = json.load(handle)
    for modality_name, expected_fields in (
        ("state", SONIC_STATE_FIELDS),
        ("action", SONIC_ACTION_FIELDS),
    ):
        fields = modality.get(modality_name)
        if not isinstance(fields, dict):
            raise ValueError(f"{path}: missing {modality_name!r} layout")
        expected_start = 0
        for key, width in expected_fields:
            spec = fields.get(key)
            expected_end = expected_start + width
            if not isinstance(spec, dict) or (
                spec.get("start"), spec.get("end")
            ) != (expected_start, expected_end):
                raise ValueError(
                    f"{path}: {modality_name}.{key} must occupy "
                    f"[{expected_start}:{expected_end}] for SONIC, got {spec}"
                )
            expected_start = expected_end


def read_episode_indices(dataset_dir: Path) -> list[int]:
    indices: list[int] = []
    with (dataset_dir / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            indices.append(int(json.loads(line)["episode_index"]))
    if not indices:
        raise ValueError("no episodes found")
    return indices


def split_episodes(
    episode_indices: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(episode_indices, dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    return sorted(shuffled[val_count:].tolist()), sorted(shuffled[:val_count].tolist())


class SonicACTDataset(Dataset[dict[str, Tensor]]):
    """One item per frame: image/qpos plus a padded future action chunk."""

    def __init__(
        self,
        dataset_dir: str | Path,
        episode_indices: list[int],
        chunk_size: int,
        stats: NormalizationStats,
        image_size: int = 224,
        training: bool = True,
        max_open_videos: int = 8,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        validate_sonic_layout(self.dataset_dir)
        self.chunk_size = chunk_size
        self.stats = stats
        self.image_size = image_size
        self.training = training
        self.max_open_videos = max_open_videos
        self.episodes: list[Episode] = []
        sample_map: list[tuple[int, int]] = []
        for local_index, episode_index in enumerate(episode_indices):
            parquet_path = (
                self.dataset_dir / "data" / f"chunk-{episode_index // 1000:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            table = pq.read_table(parquet_path, columns=["observation.state", "action"])
            qpos = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if qpos.ndim != 2 or qpos.shape[1] != 46:
                raise ValueError(f"{parquet_path}: invalid qpos shape {qpos.shape}")
            if actions.shape != (len(qpos), 78):
                raise ValueError(f"{parquet_path}: invalid action shape {actions.shape}")
            if not np.isfinite(qpos).all() or not np.isfinite(actions).all():
                raise ValueError(f"{parquet_path}: contains NaN or Inf")
            video_path = (
                self.dataset_dir / "videos" / f"chunk-{episode_index // 1000:03d}"
                / "observation.images.ego_view" / f"episode_{episode_index:06d}.mp4"
            )
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            self.episodes.append(Episode(episode_index, qpos, actions, video_path))
            sample_map.extend((local_index, frame) for frame in range(len(qpos)))
        self.sample_map = np.asarray(sample_map, dtype=np.int32)
        self._captures: OrderedDict[int, cv2.VideoCapture] | None = None

    def __len__(self) -> int:
        return len(self.sample_map)

    def _capture(self, local_episode: int) -> cv2.VideoCapture:
        # Each DataLoader worker owns its own lazy capture cache.
        if self._captures is None:
            self._captures = OrderedDict()
        if local_episode in self._captures:
            capture = self._captures.pop(local_episode)
            self._captures[local_episode] = capture
            return capture
        capture = cv2.VideoCapture(str(self.episodes[local_episode].video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {self.episodes[local_episode].video_path}")
        self._captures[local_episode] = capture
        while len(self._captures) > self.max_open_videos:
            _, old = self._captures.popitem(last=False)
            old.release()
        return capture

    def _read_image(self, local_episode: int, frame_index: int) -> Tensor:
        capture = self._capture(local_episode)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if not ok:
            # Re-open once: some ffmpeg backends become stale after many seeks.
            assert self._captures is not None
            self._captures.pop(local_episode).release()
            capture = self._capture(local_episode)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
        if not ok:
            raise RuntimeError(
                f"failed to decode episode {self.episodes[local_episode].index}, frame {frame_index}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_size = self.image_size + 32
        image = cv2.resize(image, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        if self.training:
            top = int(np.random.randint(0, resize_size - self.image_size + 1))
            left = int(np.random.randint(0, resize_size - self.image_size + 1))
            # Mild photometric jitter; horizontal flipping would invalidate robot semantics.
            alpha = float(np.random.uniform(0.9, 1.1))
            beta = float(np.random.uniform(-8.0, 8.0))
            image = np.clip(image.astype(np.float32) * alpha + beta, 0.0, 255.0)
        else:
            top = left = (resize_size - self.image_size) // 2
        image = image[top : top + self.image_size, left : left + self.image_size]
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        image = (image - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None])
        image = image / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return torch.from_numpy(image)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        local_episode, frame_index = (int(v) for v in self.sample_map[item])
        episode = self.episodes[local_episode]
        qpos = (episode.qpos[frame_index] - self.stats.qpos_mean) / self.stats.qpos_std
        end = min(frame_index + self.chunk_size, len(episode.actions))
        valid_count = end - frame_index
        actions = np.zeros((self.chunk_size, 78), dtype=np.float32)
        actions[:valid_count] = (
            episode.actions[frame_index:end] - self.stats.action_mean
        ) / self.stats.action_std
        is_pad = np.ones(self.chunk_size, dtype=np.bool_)
        is_pad[:valid_count] = False
        return {
            "qpos": torch.from_numpy(np.asarray(qpos, dtype=np.float32)),
            "image": self._read_image(local_episode, frame_index),
            "actions": torch.from_numpy(actions),
            "is_pad": torch.from_numpy(is_pad),
            "episode_index": torch.tensor(episode.index, dtype=torch.int64),
            "frame_index": torch.tensor(frame_index, dtype=torch.int64),
        }

    def __del__(self) -> None:
        if self._captures:
            for capture in self._captures.values():
                capture.release()


def dataset_summary(dataset: SonicACTDataset) -> dict[str, Any]:
    lengths = [len(episode.qpos) for episode in dataset.episodes]
    return {
        "episodes": len(lengths),
        "frames": len(dataset),
        "min_episode_frames": min(lengths),
        "max_episode_frames": max(lengths),
    }
