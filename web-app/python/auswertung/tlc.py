"""


TLC ( ML Gruppierer )




"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PAD_MULTIPLE = 32


@dataclass
class TLCConfig:
    list: str = "parallel"
    groups: int = 16
    pretrained: bool = False
    feature_stage: int = 3
    context_channels: int = 32
    pool_points: int = 16
    position_points: int = 16
    direction_bins: int = 36
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1


def _canonical(lines: torch.Tensor) -> torch.Tensor:
    """Endpunkte so ordnen, dass dx > 0 (bzw. dx == 0 und dy > 0)."""
    P, Q = lines[..., :2], lines[..., 2:]
    d = Q - P
    swap = (d[..., 0] < 0) | ((d[..., 0] == 0) & (d[..., 1] < 0))
    return torch.where(swap[..., None], torch.cat([Q, P], -1), lines)


def _points_along(lines: torch.Tensor, n: int) -> torch.Tensor:
    t = torch.linspace(0, 1, n, device=lines.device, dtype=lines.dtype)
    P, Q = lines[..., None, :2], lines[..., None, 2:]
    return P + (Q - P) * t[:, None]


class TLC(nn.Module):
    def __init__(self, cfg: TLCConfig):
        super().__init__()
        import torchvision
        self.cfg = cfg
        net = torchvision.models.resnet18(weights=None)
        stages = [net.layer1, net.layer2, net.layer3, net.layer4][:cfg.feature_stage]
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, *stages)
        in_ch = (64, 128, 256, 512)[cfg.feature_stage - 1]
        self.reduce = nn.Conv2d(in_ch, cfg.context_channels, 1)
        self.pool_weights = nn.Parameter(torch.full((cfg.pool_points,), 1.0 / cfg.pool_points))
        self.proj = nn.Linear(cfg.context_channels + 2 * cfg.position_points + cfg.direction_bins, cfg.d_model)

        def branch():
            layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.heads, 4 * cfg.d_model, cfg.dropout, batch_first=True)
            return nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)

        self.t_cluster, self.t_score = branch(), branch()
        self.cluster_head = nn.Linear(cfg.d_model, cfg.groups)
        self.score_head = nn.Linear(cfg.d_model, 1)

    def forward(self, image: torch.Tensor, lines: torch.Tensor, size: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """image (1, 3, H, W), lines (1, L, 4) Pixel, size (1, 2) = Breite, Höhe ohne Rand."""
        cfg = self.cfg
        lines = _canonical(lines)
        fmap = self.reduce(self.backbone(image))
        B, C = fmap.shape[:2]
        L = lines.shape[1]
        H, W = image.shape[-2:]
        pts = _points_along(lines, cfg.pool_points)
        grid = torch.stack([(pts[..., 0] + 0.5) / W * 2 - 1, (pts[..., 1] + 0.5) / H * 2 - 1], -1)
        s = F.grid_sample(fmap, grid.reshape(B, L * cfg.pool_points, 1, 2), mode="bilinear",
                          padding_mode="border", align_corners=False).reshape(B, C, L, cfg.pool_points)
        f_con = (s * self.pool_weights).sum(-1).transpose(1, 2)
        c = size[:, None, None, :] / 2
        scale = size.max(1).values[:, None, None, None] / 2
        f_pos = ((_points_along(lines, cfg.position_points) - c) / scale).reshape(B, L, -1)
        d = lines[..., 2:] - lines[..., :2]
        ang = torch.remainder(torch.atan2(d[..., 1], d[..., 0]), math.pi)
        idx = torch.clamp((ang / math.pi * cfg.direction_bins).long(), max=cfg.direction_bins - 1)
        f = self.proj(torch.cat([f_con, f_pos, F.one_hot(idx, cfg.direction_bins).to(f_con.dtype)], -1))
        return self.cluster_head(self.t_cluster(f))[0], self.score_head(self.t_score(f)).squeeze(-1)[0]


class TLCPredictor:
    def __init__(self, checkpoint: Path, device: torch.device):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.cfg = TLCConfig(**{**ck["modelConfig"], "pretrained": False})
        self.image_size = ck["dataConfig"]["image_size"]
        self.device = device
        self.model = TLC(self.cfg)
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, lines: np.ndarray) -> list[list[int]]:
        """-> Gruppen als Linienindizes (größte zuerst)."""
        if len(lines) < 2:
            return []
        H, W = image_bgr.shape[:2]
        s = self.image_size / max(H, W)
        img = cv2.resize(image_bgr, (round(W * s), round(H * s)),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        rgb = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        h, w = rgb.shape[:2]
        x = np.zeros((3, -(-h // PAD_MULTIPLE) * PAD_MULTIPLE, -(-w // PAD_MULTIPLE) * PAD_MULTIPLE), np.float32)
        x[:, :h, :w] = rgb.transpose(2, 0, 1)
        cluster, score = self.model(torch.from_numpy(x)[None].to(self.device),
                                    torch.from_numpy(lines * s)[None].to(self.device),
                                    torch.tensor([[w, h]], dtype=torch.float32, device=self.device))
        return decode(cluster.float().cpu().numpy(), score.float().cpu().numpy(), self.cfg.list != "parallel")


def decode(cluster_logits: np.ndarray, score_logits: np.ndarray, multi_label: bool, threshold: float = 0.5,
           min_lines: int = 2) -> list[list[int]]:
    """Linien mit Konfidenz <= threshold gehören zu keiner Gruppe; Gruppen ohne Duplikate, größte zuerst."""
    inl = 1 / (1 + np.exp(-score_logits)) > threshold
    if multi_label:
        member = (1 / (1 + np.exp(-cluster_logits)) > threshold) & inl[:, None]
    else:
        member = np.zeros_like(cluster_logits, bool)
        member[np.arange(len(cluster_logits)), cluster_logits.argmax(1)] = True
        member &= inl[:, None]
    groups = {tuple(np.where(member[:, k])[0].tolist()) for k in range(member.shape[1])}
    return [list(g) for g in sorted(groups, key=lambda g: (-len(g), g)) if len(g) >= min_lines]
