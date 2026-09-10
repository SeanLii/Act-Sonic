"""A compact implementation of the classic Action Chunking Transformer (ACT).

The policy is a conditional VAE during training.  Its posterior encoder consumes
the current robot state and the ground-truth action chunk.  A ResNet image
backbone, the current state, and the sampled latent are fused by a Transformer;
learned action queries decode the complete future action chunk in parallel.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torchvision.models import resnet18


def _sincos_2d(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return DETR-style 2-D sine/cosine positions with shape [1, H*W, D]."""
    if dim % 4:
        raise ValueError(f"hidden_dim must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))
    x = x.flatten()[:, None] * omega[None]
    y = y.flatten()[:, None] * omega[None]
    pos = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pos[None].to(dtype=dtype)


class ACTPolicy(nn.Module):
    """Image-and-qpos ACT policy producing ``chunk_size x action_dim`` outputs."""

    def __init__(
        self,
        qpos_dim: int = 46,
        action_dim: int = 78,
        chunk_size: int = 30,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        nheads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 7,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % nheads:
            raise ValueError("hidden_dim must be divisible by nheads")
        self.qpos_dim = qpos_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim

        backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # CVAE posterior q(z | qpos, action chunk).
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.posterior_pos = nn.Parameter(torch.zeros(1, chunk_size + 2, hidden_dim))
        self.posterior_qpos_proj = nn.Linear(qpos_dim, hidden_dim)
        self.posterior_action_proj = nn.Linear(action_dim, hidden_dim)
        posterior_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nheads,
            dim_feedforward,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior_encoder = nn.TransformerEncoder(
            posterior_layer, encoder_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.latent_stats = nn.Linear(hidden_dim, latent_dim * 2)

        # Observation encoder and parallel action decoder.
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.qpos_proj = nn.Linear(qpos_dim, hidden_dim)
        self.condition_type = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        observation_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nheads,
            dim_feedforward,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.observation_encoder = nn.TransformerEncoder(
            observation_layer, encoder_layers, norm=nn.LayerNorm(hidden_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim,
            nheads,
            dim_feedforward,
            dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, decoder_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.pad_head = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.posterior_cls, std=0.02)
        nn.init.normal_(self.posterior_pos, std=0.02)
        nn.init.normal_(self.condition_type, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)
        for module in (self.image_proj, self.action_head, self.pad_head, self.latent_stats):
            if getattr(module, "weight", None) is not None:
                nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _posterior(
        self, qpos: Tensor, actions: Tensor, is_pad: Tensor, sample: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = qpos.shape[0]
        tokens = torch.cat(
            (
                self.posterior_cls.expand(batch, -1, -1),
                self.posterior_qpos_proj(qpos).unsqueeze(1),
                self.posterior_action_proj(actions),
            ),
            dim=1,
        )
        tokens = tokens + self.posterior_pos[:, : tokens.shape[1]]
        prefix_pad = torch.zeros(batch, 2, dtype=torch.bool, device=qpos.device)
        encoded = self.posterior_encoder(
            tokens, src_key_padding_mask=torch.cat((prefix_pad, is_pad), dim=1)
        )
        mu, logvar = self.latent_stats(encoded[:, 0]).chunk(2, dim=-1)
        logvar = logvar.clamp(-10.0, 10.0)
        if sample:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, logvar

    def forward(
        self,
        qpos: Tensor,
        image: Tensor,
        actions: Tensor | None = None,
        is_pad: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Run ACT.

        Training calls should pass normalized ``actions`` and ``is_pad``.  At
        inference, omit them and the prior mean z=0 is used, as in classic ACT.
        """
        if qpos.ndim != 2 or qpos.shape[-1] != self.qpos_dim:
            raise ValueError(f"expected qpos [B,{self.qpos_dim}], got {tuple(qpos.shape)}")
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image [B,3,H,W], got {tuple(image.shape)}")
        batch = qpos.shape[0]
        mu: Tensor | None = None
        logvar: Tensor | None = None
        if actions is not None:
            if actions.shape[1:] != (self.chunk_size, self.action_dim):
                raise ValueError(
                    f"expected actions [B,{self.chunk_size},{self.action_dim}], "
                    f"got {tuple(actions.shape)}"
                )
            if is_pad is None:
                is_pad = torch.zeros(
                    batch, self.chunk_size, dtype=torch.bool, device=qpos.device
                )
            z, mu, logvar = self._posterior(qpos, actions, is_pad, sample=self.training)
        else:
            z = torch.zeros(batch, self.latent_dim, device=qpos.device, dtype=qpos.dtype)

        features = self.image_proj(self.backbone(image))
        height, width = features.shape[-2:]
        image_tokens = features.flatten(2).transpose(1, 2)
        image_tokens = image_tokens + _sincos_2d(
            height, width, image_tokens.shape[-1], image.device, image_tokens.dtype
        )
        condition = torch.stack((self.latent_proj(z), self.qpos_proj(qpos)), dim=1)
        condition = condition + self.condition_type
        memory = self.observation_encoder(torch.cat((condition, image_tokens), dim=1))
        queries = self.action_queries.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return {
            "actions": self.action_head(decoded),
            "pad_logits": self.pad_head(decoded).squeeze(-1),
            "mu": mu,
            "logvar": logvar,
        }

    @staticmethod
    def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        """Mean KL(q(z|x)||N(0,I)), summed over latent dimensions."""
        return (-0.5 * (1.0 + logvar - mu.square() - logvar.exp())).sum(-1).mean()

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
