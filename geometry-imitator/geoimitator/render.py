"""
Projektion der Szene

* Tiefenpuffer: jede Fläche wird (an der Nahebene geschnitten) gefüllt, die
  Tiefe pro Pixel kommt exakt aus dem Strahl-Ebene-Schnitt.
* Sichtbarkeit: Punkte entlang jeder 3D-Linie werden gegen den Tiefenpuffer
  getestet -> verdeckte Teile (Rückseiten von Kisten, Möbel vor der Wand, ...)
  fallen weg. Eine 3D-Linie kann so in mehrere 2D-Segmente zerfallen.
* Linienbild: weiße Linien auf Schwarz (wie die M-LSD-Bilder, auf denen
  ControlNet-MLSD trainiert wurde).
* Tiefenbild: inverse Tiefe, nah = hell (wie MiDaS-Bilder für ControlNet-Depth)
"""

from __future__ import annotations

import colorsys
import math

import cv2
import numpy as np

from .camera import Camera
from .geom import clip_polygon_near, clip_segment_near, clip_segment_rect
from .scene import Scene

NEAR = 0.05
SHIFT = 4  # Subpixel-Bits für cv2.fillPoly / cv2.line


def _poly_px(uv: np.ndarray) -> np.ndarray:
    return np.round(uv * (1 << SHIFT)).astype(np.int64).clip(-(2 ** 30), 2 ** 30).astype(np.int32)


def render_buffers(scene: Scene, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Tiefe (z in Kamerakoordinaten, inf = nichts) und Flächen-Index (-1 = nichts)."""
    W, H = cam.width, cam.height
    zbuf = np.full((H, W), np.inf, np.float32)
    fbuf = np.full((H, W), -1, np.int32)
    Kinv = np.linalg.inv(cam.K)
    for fi, face in enumerate(scene.faces):
        Pc = clip_polygon_near(cam.to_cam(face.polygon), NEAR)
        if len(Pc) < 3:
            continue
        uv = cam.project_cam(Pc)
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x0 > x1 or y0 > y1:
            continue
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        off = np.array([x0, y0])
        cv2.fillPoly(mask, [_poly_px(uv - off)], 1, cv2.LINE_8, SHIFT)
        for hole in face.holes:
            Hc = clip_polygon_near(cam.to_cam(hole), NEAR)
            if len(Hc) >= 3:
                cv2.fillPoly(mask, [_poly_px(cam.project_cam(Hc) - off)], 0, cv2.LINE_8, SHIFT)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        # Ebene in Kamerakoordinaten: n . X = d
        n = cam.R @ face.normal
        d = float(n @ Pc[0])
        u, v = xs + x0, ys + y0
        rx = Kinv[0, 0] * u + Kinv[0, 1] * v + Kinv[0, 2]
        ry = Kinv[1, 1] * v + Kinv[1, 2]
        denom = n[0] * rx + n[1] * ry + n[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = d / denom
        ok = np.isfinite(z) & (z > NEAR * 0.5)
        u, v, z = u[ok], v[ok], z[ok].astype(np.float32)
        closer = z < zbuf[v, u]
        zbuf[v[closer], u[closer]] = z[closer]
        fbuf[v[closer], u[closer]] = fi
    return zbuf, fbuf


def visible_segments(scene: Scene, cam: Camera, zbuf: np.ndarray, min_length_px: float = 8.0,
                     rel_tol: float = 0.015, abs_tol: float = 0.03, max_gap_px: float = 3.0) -> list[dict]:
    """Sichtbare 2D-Segmente aller 3D-Linien (mit den Eigenschaften der 3D-Linie)."""
    W, H = cam.width, cam.height
    # großzügig: Maximum der Tiefe in der 3x3-Nachbarschaft (Kanten liegen zwischen Flächen)
    zfin = np.where(np.isfinite(zbuf), zbuf, 1e9).astype(np.float32)
    zmax = cv2.dilate(zfin, np.ones((3, 3), np.uint8))
    segs = []
    for line in scene.lines:
        a, b = cam.to_cam(line.p1)[0], cam.to_cam(line.p2)[0]
        clipped = clip_segment_near(a, b, NEAR)
        if clipped is None:
            continue
        a, b, _, _ = clipped
        uv_a, uv_b = cam.project_cam(np.array([a, b]))
        L = float(np.linalg.norm(uv_b - uv_a))
        if not np.isfinite(L) or L < 1:
            continue
        n = int(min(6000, max(8, math.ceil(L * 1.5))))
        t = np.linspace(0, 1, n)
        P = a[None] + t[:, None] * (b - a)[None]
        uv = cam.project_cam(P)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        vis = np.zeros(n, bool)
        idx = np.where(inside)[0]
        if len(idx):
            px = np.clip(np.round(uv[idx]).astype(int), 0, [W - 1, H - 1])
            zb = zmax[px[:, 1], px[:, 0]]
            vis[idx] = P[idx, 2] <= zb * (1 + rel_tol) + abs_tol
        for i0, i1 in _runs(vis, uv, max_gap_px):
            p, q = uv[i0], uv[i1]
            if np.linalg.norm(q - p) < min_length_px:
                continue
            segs.append({
                "id": f"s{len(segs)}", "line3d": line.id,
                "p1": [float(p[0]), float(p[1])], "p2": [float(q[0]), float(q[1])],
                "direction": line.direction_id, "planes": list(line.plane_ids),
                "planeFamilies": list(line.family_ids), "object": line.object_id, "kind": line.kind,
                "depth": float(np.mean(P[[i0, i1], 2])), "perturbed": None,
            })
    return segs


def _runs(vis: np.ndarray, uv: np.ndarray, max_gap_px: float):
    """Zusammenhängende sichtbare Abschnitte; kleine Lücken (Aliasing) werden überbrückt."""
    idx = np.where(vis)[0]
    if len(idx) == 0:
        return
    start = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1 and np.linalg.norm(uv[i] - uv[prev]) > max_gap_px:
            yield start, prev
            start = i
        prev = i
    yield start, prev


# ---------------------------------------------------------------------------
# Bilder
# ---------------------------------------------------------------------------

def line_image(segments: list[dict], width: int, height: int, thickness: int = 1,
               scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Weiße Linien auf Schwarz (uint8, HxW). `scale` skaliert die Koordinaten
    (Szenen-Auflösung -> Generator-Auflösung)."""
    img = np.zeros((height, width), np.uint8)
    sx, sy = scale
    for s in segments:
        p = np.array([s["p1"][0] * sx, s["p1"][1] * sy])
        q = np.array([s["p2"][0] * sx, s["p2"][1] * sy])
        cv2.line(img, tuple(_poly_px(p)), tuple(_poly_px(q)), 255, thickness, cv2.LINE_AA, SHIFT)
    return img


def depth_image(zbuf: np.ndarray) -> np.ndarray:
    """Inverse Tiefe auf 0..255 (nah = hell), Hintergrund (Himmel) = 0."""
    valid = np.isfinite(zbuf)
    out = np.zeros(zbuf.shape, np.uint8)
    if not valid.any():
        return out
    disp = 1.0 / zbuf[valid]
    lo, hi = np.percentile(disp, 1), np.percentile(disp, 99.5)
    norm = np.clip((disp - lo) / max(hi - lo, 1e-9), 0, 1)
    out[valid] = (20 + 235 * norm).astype(np.uint8)
    return out


def palette(n: int) -> list[tuple[int, int, int]]:
    """Gut unterscheidbare BGR-Farben (goldener Schnitt im Farbton). Rot/Orange
    (Farbton < 0.12) bleibt frei - im Overlay heißt das unbekannt/mehrdeutig."""
    cols = []
    for i in range(n):
        h = 0.12 + ((0.2 + i * 0.618033988749895) % 1.0) * 0.83
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


def colored_line_image(segments: list[dict], direction_ids: list[str], width: int, height: int,
                       background: np.ndarray | None = None, thickness: int = 2) -> np.ndarray:
    """Vorschau: Linien nach 3D-Richtung eingefärbt (Einzellinien grau)."""
    img = np.zeros((height, width, 3), np.uint8) if background is None else background.copy()
    counts: dict[str, int] = {}
    for s in segments:
        counts[s["direction"]] = counts.get(s["direction"], 0) + 1
    multi = [d for d in direction_ids if counts.get(d, 0) >= 2]
    cols = dict(zip(multi, palette(len(multi))))
    for s in segments:
        c = cols.get(s["direction"], (140, 140, 140))
        p, q = np.array(s["p1"]), np.array(s["p2"])
        cv2.line(img, tuple(_poly_px(p)), tuple(_poly_px(q)), c, thickness, cv2.LINE_AA, SHIFT)
    return img


def clip_to_image(p, q, w: int, h: int):
    return clip_segment_rect(np.asarray(p, float), np.asarray(q, float), w - 1, h - 1)
