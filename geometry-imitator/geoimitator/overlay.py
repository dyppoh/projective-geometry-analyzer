"""Kontrollbilder: Linienbild über der Generierung und gelabelte DeepLSD-Linien"""

from __future__ import annotations

import cv2
import numpy as np

from .render import SHIFT, _poly_px, line_image, palette


def _line(img, p, q, color, t):
    cv2.line(img, tuple(_poly_px(np.asarray(p, float))), tuple(_poly_px(np.asarray(q, float))), color, t,
             cv2.LINE_AA, SHIFT)


def overlay_control(image: np.ndarray, scene_json: dict) -> np.ndarray:
    """Generiertes Bild (abgedunkelt) + Original-Linienbild in Cyan."""
    H, W = image.shape[:2]
    W0, H0 = scene_json["image"]["width"], scene_json["image"]["height"]
    lines = line_image(scene_json["segments2d"], W, H, max(1, round(W / W0)), (W / W0, H / H0))
    out = (image.astype(np.float32) * 0.6)
    a = lines.astype(np.float32)[..., None] / 255.0
    out = out * (1 - a) + np.array([255, 255, 0], np.float32) * a
    return out.clip(0, 255).astype(np.uint8)


def overlay_labels(image: np.ndarray, sample: dict, gt: list[dict]) -> np.ndarray:
    """Graues Bild; Original dünn weiß, zugeordnete Linien in der Farbe ihrer Richtung,
    mehrdeutige orange, unbekannte dunkelrot."""
    gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = (gray.astype(np.float32) * 0.45).astype(np.uint8)
    for s in gt:
        _line(out, s["p1"], s["p2"], (200, 200, 200), 1)
    dir_ids = sample["training"]["directionIds"]
    cols = dict(zip(dir_ids, palette(len(dir_ids))))
    for l in sample["lines"]:
        p, q = (l["point1"]["x"], l["point1"]["y"]), (l["point2"]["x"], l["point2"]["y"])
        if l["status"] == "matched":
            _line(out, p, q, cols.get(l["label"]["direction"], (255, 255, 255)), 2)
        elif l["status"] == "ambiguous":
            _line(out, p, q, (0, 140, 255), 1)
        else:
            _line(out, p, q, (40, 40, 170), 1)
    m = sample["match"]
    txt = (f"{'OK' if m['accepted'] else 'REJECTED'}  recall lines {m.get('recallLines', 0):.2f}  "
           f"matched {m['matchedLines']}/{m['detectedLines']}  groups {m['matchedGroups']}/{m['gtGroups']}")
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (120, 255, 120) if m["accepted"] else (80, 80, 255), 1, cv2.LINE_AA)
    return out
