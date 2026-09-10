from pathlib import Path

import torch

from act.data import NormalizationStats, SonicACTDataset, read_episode_indices, split_episodes
from act.model import ACTPolicy
from act.training import compute_loss


DATASET = Path(__file__).parents[1] / "io_g1_box_cotransport2_sonic_joint"


def test_dataset_contract() -> None:
    stats = NormalizationStats.from_json(DATASET / "meta" / "stats.json")
    train, _ = split_episodes(read_episode_indices(DATASET), 0.1, 42)
    dataset = SonicACTDataset(DATASET, train[:1], chunk_size=8, stats=stats, image_size=64)
    sample = dataset[0]
    assert sample["qpos"].shape == (46,)
    assert sample["image"].shape == (3, 64, 64)
    assert sample["actions"].shape == (8, 78)
    assert sample["is_pad"].shape == (8,)


def test_model_forward_backward() -> None:
    model = ACTPolicy(
        chunk_size=8,
        hidden_dim=64,
        latent_dim=8,
        nheads=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=128,
    )
    qpos = torch.randn(2, 46)
    image = torch.randn(2, 3, 64, 64)
    actions = torch.randn(2, 8, 78)
    is_pad = torch.zeros(2, 8, dtype=torch.bool)
    outputs = model(qpos, image, actions, is_pad)
    assert outputs["actions"].shape == (2, 8, 78)
    losses = compute_loss(model, outputs, actions, is_pad, kl_weight=10.0, pad_weight=0.01)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


def test_inference_uses_prior() -> None:
    model = ACTPolicy(
        chunk_size=4,
        hidden_dim=64,
        latent_dim=8,
        nheads=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=128,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 46), torch.zeros(1, 3, 64, 64))
    assert output["actions"].shape == (1, 4, 78)
    assert output["mu"] is None
