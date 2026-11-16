"""Absichtlich gestörte Projektion (gegen Zirkularität beim späteren Training)

Ein Teil der Szenen bekommt für einzelne Richtungen Linien, die in 2D NICHT
mehr exakt in einem Fluchtpunkt zusammenlaufen - das Label bleibt trotzdem
"parallel". Ein Sortierungsmodell kann dann nicht einfach die Konvergenz
nachrechnen, sondern muss den Bildinhalt nutzen; außerdem sehen die
Trainingsdaten so aus wie die Fehler generierter Bilder.

Modi (pro gestörter Richtung):
* ``line_jitter``  - jede Linie wird um ihren Mittelpunkt um N(0, sigma) gedreht
* ``camera_split`` - ein Teil der 3D-Linien wird mit leicht gedrehter Kamera
                     projiziert (zwei Fluchtpunkte statt einem)
"""

from __future__ import annotations

import math

import numpy as np

from .camera import Camera
from .geom import rot_axis
from .render import clip_to_image


def perturb_segments(segments: list[dict], cam: Camera, rng: np.random.Generator, cfg: dict) -> list[dict]:
    """Verändert `segments` in place und gibt die Beschreibung der Störungen zurück."""
    if rng.random() >= cfg.get("probability", 0.0):
        return []
    counts: dict[str, int] = {}
    for s in segments:
        counts[s["direction"]] = counts.get(s["direction"], 0) + 1
    candidates = sorted([d for d, c in counts.items() if c >= cfg.get("minLines", 3)], key=lambda d: int(d[1:]))
    if not candidates:
        return []
    k = int(rng.integers(1, min(cfg.get("maxDirections", 2), len(candidates)) + 1))
    chosen = [str(d) for d in rng.choice(candidates, size=k, replace=False)]
    report = []
    W, H = cam.width, cam.height
    for d in chosen:
        mode = str(rng.choice(cfg.get("modes", ["line_jitter", "camera_split"])))
        segs = [s for s in segments if s["direction"] == d]
        if mode == "line_jitter":
            sigma = float(rng.uniform(*cfg.get("jitterSigmaDeg", [0.6, 2.0])))
            for s in segs:
                ang = float(rng.normal(0, sigma))
                _rotate_2d(s, ang)
                s["perturbed"] = {"mode": mode, "angleDeg": ang}
            report.append({"direction": d, "mode": mode, "sigmaDeg": sigma, "segments": [s["id"] for s in segs]})
        else:
            angle = float(rng.uniform(*cfg.get("cameraSplitDeg", [1.0, 3.0]))) * (1 if rng.random() < 0.5 else -1)
            axis = rng.normal(size=3)
            Hm = cam.homography_rotation(rot_axis(axis, angle))
            line_ids = sorted({s["line3d"] for s in segs})
            n_moved = max(1, len(line_ids) // 2)
            moved = set(rng.choice(line_ids, size=n_moved, replace=False).tolist())
            for s in segs:
                if s["line3d"] in moved:
                    s["p1"], s["p2"] = _apply_h(Hm, s["p1"]), _apply_h(Hm, s["p2"])
                    s["perturbed"] = {"mode": mode, "angleDeg": angle}
            report.append({"direction": d, "mode": mode, "angleDeg": angle, "axis": axis.tolist(),
                           "segments": [s["id"] for s in segs if s["perturbed"]]})
    # wieder auf das Bild beschneiden, zu kurze verwerfen
    min_len = cfg.get("minLengthPx", 8.0)
    keep = []
    for s in segments:
        if s["perturbed"]:
            c = clip_to_image(s["p1"], s["p2"], W, H)
            if c is None or np.linalg.norm(c[1] - c[0]) < min_len:
                continue
            s["p1"], s["p2"] = c[0].tolist(), c[1].tolist()
        keep.append(s)
    segments[:] = keep
    return report


def _rotate_2d(s: dict, deg: float) -> None:
    p, q = np.array(s["p1"]), np.array(s["p2"])
    m = (p + q) / 2
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    s["p1"], s["p2"] = (m + R @ (p - m)).tolist(), (m + R @ (q - m)).tolist()


def _apply_h(Hm: np.ndarray, p) -> list[float]:
    v = Hm @ np.array([p[0], p[1], 1.0])
    return [float(v[0] / v[2]), float(v[1] / v[2])]
