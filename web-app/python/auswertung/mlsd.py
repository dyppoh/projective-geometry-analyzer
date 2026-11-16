"""


M-LSD "512 large" in PyTorch - mit den Gewichten des bisher verwendeten ONNX-Modells
(mlsd-512-large.onnx, PINTO-Konvertierung des TensorFlow-Originals navervision/mlsd,
Apache 2.0), damit die Linien exakt dieselben bleiben.

Architektur wie MobileV2_MLSD_Large (lhwcv/mlsd_pytorch), aber wie
im TensorFlow-Original: BatchNorm in die Faltungen eingerechnet (Faltung mit Bias) und Upsampling mit
align_corners=False. Die Gewichte von lhwcv (mlsd_large_512_fp32.pth) sind ein anderer Stand und ergeben andere
Linien - deshalb nicht verwendet.

    Bild ──► 512×512 (RGB + Alpha, auf [-1, 1]) ──► MobileNetV2 (bis 96 Kanäle) ──► FPN-Decoder ──► (9, 256, 256)
    Kanal 0: Mittelpunkt-Heatmap, 1-4: Verschiebung Mittelpunkt -> Start (x, y) und Ende (x, y) im 256er-Grid

Einmalig aus dem ONNX erzeugt (Paket onnx nur dafür nötig):
    python -m auswertung.mlsd <pfad>/mlsd-512-large.onnx models/mlsd-512-large.pt


    
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT = 512
GRID = 256
SCORE_THRESHOLD = 0.1
DIST_THRESHOLD = 20.0           # Grid-Einheiten
TOPK = 200


class ConvAct(nn.Module):
    """Faltung (BatchNorm eingerechnet) + Aktivierung; Stride 2 mit TensorFlow-"SAME"-Rand (rechts/unten)."""

    def __init__(self, c_in, c_out, k=1, stride=1, groups=1, dilation=1, act: str | None = "relu6"):
        super().__init__()
        pad = 0 if stride == 2 else dilation * (k - 1) // 2
        self.conv = nn.Conv2d(c_in, c_out, k, stride, pad, dilation=dilation, groups=groups, bias=True)
        self.stride, self.act = stride, act

    def forward(self, x):
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        if self.act == "relu6":
            return F.relu6(x)
        return F.relu(x) if self.act == "relu" else x


class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, stride, expand):
        super().__init__()
        hidden = c_in * expand
        layers = [ConvAct(c_in, hidden)] if expand != 1 else []
        layers += [ConvAct(hidden, hidden, 3, stride, groups=hidden), ConvAct(hidden, c_out, act=None)]
        self.conv = nn.Sequential(*layers)
        self.residual = stride == 1 and c_in == c_out

    def forward(self, x):
        return x + self.conv(x) if self.residual else self.conv(x)


class BlockA(nn.Module):
    """b: (2× hochskaliert, dann) 1×1 + ReLU, a: 1×1 + ReLU, dann [a, b] zusammenfügen. Wie im TensorFlow-Original
    erst hochskalieren, dann falten (lhwcv faltet zuerst - wegen ReLU nicht dasselbe)."""

    def __init__(self, c_a, c_b, out_a, out_b, upscale=True):
        super().__init__()
        self.conv_b, self.conv_a, self.upscale = ConvAct(c_b, out_b, act="relu"), ConvAct(c_a, out_a, act="relu"), upscale

    def forward(self, a, b):
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=False)
        return torch.cat((self.conv_a(a), self.conv_b(b)), 1)


class BlockB(nn.Module):
    def __init__(self, c, c_out):
        super().__init__()
        self.conv1, self.conv2 = ConvAct(c, c, 3, act="relu"), ConvAct(c, c_out, 3, act="relu")

    def forward(self, x):
        return self.conv2(self.conv1(x) + x)


class BlockC(nn.Module):
    def __init__(self, c, c_out):
        super().__init__()
        self.conv1 = ConvAct(c, c, 3, dilation=5, act="relu")
        self.conv2 = ConvAct(c, c, 3, act="relu")
        self.conv3 = ConvAct(c, c_out, act=None)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class MLSDLarge(nn.Module):
    def __init__(self):
        super().__init__()
        features, c = [ConvAct(4, 32, 3, 2)], 32
        for t, c_out, n, s in [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1]]:
            for i in range(n):
                features.append(InvertedResidual(c, c_out, s if i == 0 else 1, t))
                c = c_out
        self.features = nn.ModuleList(features)
        self.decoder = nn.ModuleList([BlockA(64, 96, 64, 64, upscale=False), BlockB(128, 64),
                                      BlockA(32, 64, 64, 64), BlockB(128, 64),
                                      BlockA(24, 64, 64, 64), BlockB(128, 64),
                                      BlockA(16, 64, 64, 64), BlockB(128, 64), BlockC(64, 16)])

    def forward(self, x):
        skips = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in (1, 3, 6, 10, 13):
                skips.append(x)
        c1, c2, c3, c4, c5 = skips
        d = self.decoder
        x = d[1](d[0](c4, c5))
        x = d[3](d[2](c3, x))
        x = d[5](d[4](c2, x))
        x = d[7](d[6](c1, x))
        return d[8](x)[:, 7:]


class MLSD:
    def __init__(self, weights: Path, device: torch.device):
        self.net = MLSDLarge()
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR-Bild -> (N, 4) Linien x1 y1 x2 y2 in Pixeln des Originalbildes (wie pred_lines())."""
        import cv2

        H, W = image_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image_bgr, (INPUT, INPUT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float()
        x = torch.cat([x, torch.full_like(x[:1], 255.0)]) / 127.5 - 1.0
        out = self.net(x[None].to(self.device))[0].float().cpu()
        heat = torch.sigmoid(out[0])
        heat = heat * (F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0] == heat)
        scores, idx = heat.flatten().topk(TOPK)
        y, x = idx // GRID, idx % GRID
        d = out[1:5, y, x].T                                                       # (TOPK, 4)
        keep = (scores > SCORE_THRESHOLD) & (torch.hypot(d[:, 0] - d[:, 2], d[:, 1] - d[:, 3]) > DIST_THRESHOLD)
        seg = torch.stack([x + d[:, 0], y + d[:, 1], x + d[:, 2], y + d[:, 3]], 1)[keep].numpy()
        lines = (seg * (INPUT / GRID) * np.array([W, H, W, H], np.float32) / INPUT).astype(np.float32)
        return lines[(lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2 > 0]


def convert(onnx_path: str, out_path: str) -> None:
    """Gewichte aus dem ONNX-Modell übernehmen: die 58 Faltungen in Ausführungsreihenfolge, je Form zugeordnet."""
    import onnx
    from onnx import numpy_helper

    g = onnx.load(onnx_path).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    convs = [(inits[n.input[1]], inits[n.input[2]]) for n in g.node if n.op_type == "Conv"]
    net = MLSDLarge()
    order: list[nn.Conv2d] = []
    hooks = [m.register_forward_hook(lambda m, i, o: order.append(m)) for m in net.modules() if isinstance(m, nn.Conv2d)]
    net(torch.zeros(1, 4, INPUT, INPUT))
    for h in hooks:
        h.remove()
    used = set()
    for m in order:                        # früheste noch freie ONNX-Faltung mit gleicher Gewichtsform
        k = next(k for k, (w, _) in enumerate(convs) if k not in used and w.shape == tuple(m.weight.shape))
        used.add(k)
        m.weight.data = torch.from_numpy(convs[k][0].copy())
        m.bias.data = torch.from_numpy(convs[k][1].copy())
    if len(used) != len(convs):
        raise ValueError(f"{len(convs) - len(used)} Faltungen nicht zugeordnet.")
    torch.save(net.state_dict(), out_path)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
