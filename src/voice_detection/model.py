"""Model components for LeJEPA speaker verification."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from torchvision.ops import MLP


class SIGReg(nn.Module):
    """Sliced Independence testing via Gaussian REGularization.

    Prevents representation collapse by penalising deviation from
    a Gaussian distribution along random 1-D projections.
    """

    def __init__(self, knots: int = 17, num_projections: int = 256) -> None:
        super().__init__()
        self.num_projections = num_projections
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (V, N, proj_dim) projected embeddings.
        """
        A = torch.randn(proj.size(-1), self.num_projections, device=proj.device)
        A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    """ViT-Small backbone + projection head for spectrogram embeddings.

    Input spectrograms are (1, 80, 300) — mono log-mel with 80 bins and
    300 time frames (3 s @ 16 kHz, hop 160).  Patch size (16, 20) yields
    5 x 15 = 75 patches.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        proj_dim: int = 128,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=embed_dim,
            in_chans=1,
            img_size=(80, 300),
            patch_size=(16, 20),
            drop_path_rate=drop_path_rate,
        )
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, V, 1, 80, 300) multi-view spectrograms, or
               (N, 1, 80, 300) single-view spectrograms.
        Returns:
            emb:  (N*V, embed_dim) backbone embeddings.
            proj: (V, N, proj_dim) projected embeddings.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))  # (N*V, embed_dim)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)  # (V, N, proj_dim)
        return emb, proj

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Single-view embedding: (N, 1, H, W) -> (N, embed_dim)."""
        return self.backbone(x)
