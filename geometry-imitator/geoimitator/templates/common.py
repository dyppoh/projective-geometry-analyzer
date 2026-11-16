"""Gemein same Vorlage der Szenen: Öffnungen, Möbel, Gebäude, Kamera """

from __future__ import annotations

import math

import numpy as np

from ..geom import rot_z
from ..scene import SceneBuilder

Z = np.array([0, 0, 1.0])


def camera_rng(rng: np.random.Generator) -> dict:
    """Realistische Kameraparameter (Grad)."""
    return {
        "pitch": lambda lo, hi: float(np.clip(rng.normal((lo + hi) / 2, (hi - lo) / 4), lo, hi)),
        "roll": lambda: float(np.clip(rng.normal(0, 1.2), -4, 4)) if rng.random() < 0.7 else 0.0,
        "hfov": lambda lo, hi: float(rng.uniform(lo, hi)),
    }


# ---------------------------------------------------------------------------
# Öffnungen
# ---------------------------------------------------------------------------

def place_openings(rng, L: float, H: float, door_prob: float = 0.4, max_windows: int = 3,
                   interior: bool = True) -> list[dict]:
    """Zufällige, nicht überlappende Tür + Fenster für eine Innenwand."""
    ops: list[dict] = []
    margin = 0.2

    def free(x, w):
        return all(x + w + margin < o["x"] or x > o["x"] + o["w"] + margin for o in ops)

    if rng.random() < door_prob and L > 1.6:
        w = rng.uniform(0.8, 1.0) if rng.random() < 0.8 else rng.uniform(1.4, 1.8)
        h = min(H - 0.2, rng.uniform(2.0, 2.15))
        x = rng.uniform(margin, L - w - margin)
        ops.append({"x": x, "y": 0.0, "w": w, "h": h, "recess": rng.uniform(0.08, 0.2), "kind": "door",
                    "panels": rng.random() < 0.5})
    for _ in range(rng.integers(0, max_windows + 1)):
        w = rng.uniform(0.7, min(2.4, L - 2 * margin))
        y = rng.uniform(0.7, 1.05)
        h = min(H - y - 0.25, rng.uniform(0.9, 1.7))
        if h < 0.5 or L - w - margin <= margin:
            continue
        for _try in range(10):
            x = rng.uniform(margin, L - w - margin)
            if free(x, w):
                ops.append({"x": x, "y": y, "w": w, "h": h, "recess": rng.uniform(0.12, 0.35), "kind": "window",
                            "mullions": rng.random() < 0.5, "transom": rng.uniform(0.55, 0.75),
                            "sill": interior and rng.random() < 0.5})
                break
    return ops


def facade_openings(rng, L: float, floors: int, floor_h: float, style: dict) -> list[dict]:
    """Fensterraster einer Fassade (pro Geschoss eine Reihe), EG mit Tür/Schaufenster."""
    ww, gap = style["windowWidth"], style["windowGap"]
    pitch = ww + gap
    n = int((L - 2 * style["margin"] + gap) // pitch)
    if n < 1:
        return []
    x0 = (L - (n * pitch - gap)) / 2
    ops = []
    door_col = int(rng.integers(n)) if style["door"] else -1
    for f in range(floors):
        base = f * floor_h
        for c in range(n):
            x = x0 + c * pitch
            if f == 0 and c == door_col:
                dw = min(pitch - 0.2, rng.uniform(1.0, 2.0))
                ops.append({"x": x + (ww - dw) / 2, "y": 0.0, "w": dw, "h": min(floor_h - 0.3, 2.3),
                            "recess": 0.25, "kind": "door", "panels": rng.random() < 0.5})
                continue
            if f == 0 and style["shop"]:
                ops.append({"x": x, "y": 0.35, "w": ww, "h": min(floor_h - 0.7, 2.4), "recess": 0.2,
                            "kind": "shopwindow", "mullions": rng.random() < 0.3})
                continue
            if rng.random() < style["skip"]:
                continue
            h = min(style["windowHeight"], floor_h - style["sillHeight"] - 0.3)
            ops.append({"x": x, "y": base + style["sillHeight"], "w": ww, "h": h, "recess": style["recess"],
                        "kind": "window", "mullions": style["mullions"], "transom": style["transom"],
                        "sill": style["sills"]})
    return ops


def facade_style(rng) -> dict:
    ww = rng.uniform(0.8, 1.8)
    return {"windowWidth": ww, "windowGap": rng.uniform(0.7, 2.6), "windowHeight": rng.uniform(1.1, 2.0),
            "sillHeight": rng.uniform(0.6, 1.0), "margin": rng.uniform(0.4, 1.5), "recess": rng.uniform(0.1, 0.35),
            "mullions": rng.random() < 0.5, "transom": rng.uniform(0.55, 0.75), "sills": rng.random() < 0.5,
            "skip": 0.05 if rng.random() < 0.8 else 0.25, "door": rng.random() < 0.8, "shop": rng.random() < 0.3,
            "cornices": rng.random() < 0.6}


# ---------------------------------------------------------------------------
# Gebäude
# ---------------------------------------------------------------------------

def add_building(b: SceneBuilder, rng, base_center, yaw_deg: float, w: float, d: float, floors: int,
                 floor_h: float, window_sides=("front",), roof: str = "flat", detail: bool = True) -> dict:
    """Quaderförmiges Gebäude (Vorderseite = lokal -y) mit Fensterraster und Dach."""
    obj = b.new_object("building")
    R = rot_z(yaw_deg)
    ex, ey = R[:, 0], R[:, 1]
    c = np.asarray(base_center, float)
    Ht = floors * floor_h
    style = facade_style(rng)
    sides = {
        "front": (c - ex * w / 2 - ey * d / 2, ex, w, ey),
        "right": (c + ex * w / 2 - ey * d / 2, ey, d, -ex),
        "back": (c + ex * w / 2 + ey * d / 2, -ex, w, -ey),
        "left": (c - ex * w / 2 + ey * d / 2, -ey, d, ex),
    }
    for name, (o, u, L, into) in sides.items():
        ops = facade_openings(rng, L, floors, floor_h, style) if (detail and name in window_sides) else []
        b.wall(o, u, L, Ht, into, ops, obj, f"facade-{name}")
        if detail and style["cornices"] and name in window_sides:
            for f in range(1, floors):
                b.line(o + Z * f * floor_h, o + u * L + Z * f * floor_h, obj, "cornice")
        if detail and name in window_sides and rng.random() < 0.25 and floors > 1:
            _balconies(b, rng, o, u, L, into, floors, floor_h, obj)

    corners = [c - ex * w / 2 - ey * d / 2, c + ex * w / 2 - ey * d / 2, c + ex * w / 2 + ey * d / 2,
               c - ex * w / 2 + ey * d / 2]
    top = [p + Z * Ht for p in corners]
    if roof == "gable":
        rise = rng.uniform(0.25, 0.7) * d / 2
        r0, r1 = c - ex * w / 2 + Z * (Ht + rise), c + ex * w / 2 + Z * (Ht + rise)
        b.face([top[0], top[1], r1, r0], obj=obj, kind="roof-slope")
        b.face([top[3], top[2], r1, r0], obj=obj, kind="roof-slope")
        b.face([top[0], top[3], r0], obj=obj, kind="roof-gable")
        b.face([top[1], top[2], r1], obj=obj, kind="roof-gable")
        if detail and rng.random() < 0.5:  # Dachziegel-Reihen parallel zur Traufe
            for t in np.linspace(0, 1, int(rng.integers(4, 10)))[1:-1]:
                b.line(top[0] + (r0 - top[0]) * t, top[1] + (r1 - top[1]) * t, obj, "roof-rows")
    else:
        b.face(top, obj=obj, kind="roof-flat")
        if detail and rng.random() < 0.5:  # Attika
            hh = rng.uniform(0.3, 0.9)
            for i in range(4):
                b.line(corners[i] + Z * (Ht - hh), corners[(i + 1) % 4] + Z * (Ht - hh), obj, "parapet")
    return {"id": obj, "height": Ht, "style": style}


def _balconies(b, rng, o, u, L, into, floors, floor_h, obj):
    depth = rng.uniform(0.8, 1.5)
    bw = rng.uniform(1.6, min(4.0, L / 2))
    x = rng.uniform(0.3, max(0.31, L - bw - 0.3))
    yaw = math.degrees(math.atan2(u[1], u[0]))
    for f in range(1, floors):
        if rng.random() < 0.3:
            continue
        c = o + u * (x + bw / 2) + Z * (f * floor_h - 0.15) - into * (depth / 2)
        b.box(c, bw, depth, 0.15, yaw, obj, "balcony-slab")
        rail = c + Z * 0.15
        h = 1.0
        front = rail - into * (depth / 2)
        b.line(front - u * bw / 2 + Z * h, front + u * bw / 2 + Z * h, obj, "balcony-rail")
        for s in (-1, 1):
            p = front + u * s * bw / 2
            b.line(p, p + Z * h, obj, "balcony-rail")
            b.line(p + Z * h, p + Z * h + into * depth, obj, "balcony-rail")


def add_pole(b: SceneBuilder, rng, base, height: float, arm_dir=None) -> None:
    obj = b.new_object("pole")
    b.box(base, 0.14, 0.14, height, off_axis(float(rng.uniform(0, 90))) if arm_dir is None else 0.0, obj, "pole")
    if arm_dir is not None:
        top = np.asarray(base, float) + Z * (height - 0.1)
        b.line(top, top + np.asarray(arm_dir) * rng.uniform(1.0, 2.5), obj, "pole-arm")


# ---------------------------------------------------------------------------
# Möbel / Objekte
# ---------------------------------------------------------------------------

def off_axis(yaw: float, min_deg: float = 4.0) -> float:
    """Gedrehte Objekte nicht "fast" achsparallel: sonst wären die Richtungen im Bild
    ununterscheidbar, aber unterschiedlich gelabelt."""
    r = yaw % 90
    if r < min_deg:
        return yaw - r + min_deg
    if r > 90 - min_deg:
        return yaw - r + 90 - min_deg
    return yaw


def front_yaw(pos, target, aligned: bool) -> float:
    """Gierwinkel, bei dem die Vorderseite (lokal -y) Richtung target zeigt."""
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = math.degrees(math.atan2(d[0], -d[1]))
    return round(yaw / 90) * 90.0 if aligned else yaw


def _front_lines(b, info, kind, obj, heights=(), splits=()):
    c, ex, ey, w, d = info["center"], info["ex"], info["ey"], info["w"], info["d"]
    f0 = c - ey * d / 2
    for hh in heights:
        b.line(f0 - ex * w / 2 + Z * hh, f0 + ex * w / 2 + Z * hh, obj, kind)
    for (s, h0, h1) in splits:
        p = f0 + ex * (s - 0.5) * w
        b.line(p + Z * h0, p + Z * h1, obj, kind)


def furniture_piece(b: SceneBuilder, rng, kind: str, pos, yaw: float) -> float:
    """Baut ein Möbelstück, gibt den Radius der Grundfläche zurück."""
    obj = b.new_object(kind)
    R = rot_z(yaw)
    ex, ey = R[:, 0], R[:, 1]
    pos = np.asarray(pos, float)
    if kind == "table":
        w, d, h = rng.uniform(0.8, 2.0), rng.uniform(0.6, 1.1), rng.uniform(0.7, 0.78)
        b.box(pos + Z * (h - 0.04), w, d, 0.04, yaw, obj, "table-top")
        for sx in (-1, 1):
            for sy in (-1, 1):
                b.box(pos + ex * sx * (w / 2 - 0.06) + ey * sy * (d / 2 - 0.06), 0.05, 0.05, h - 0.04, yaw, obj, "table-leg")
    elif kind == "cabinet":
        w, d, h = rng.uniform(0.5, 2.0), rng.uniform(0.35, 0.6), rng.uniform(0.6, 2.2)
        info = b.box(pos, w, d, h, yaw, obj, "cabinet")
        if rng.random() < 0.5:
            n = int(rng.integers(2, 5))
            _front_lines(b, info, "drawer", obj, heights=list(np.linspace(0, h, n + 1)[1:-1]))
        else:
            _front_lines(b, info, "cabinet-door", obj, splits=[(0.5, 0.0, h)])
    elif kind == "shelf":
        w, d, h = rng.uniform(0.6, 2.0), rng.uniform(0.25, 0.45), rng.uniform(1.0, 2.2)
        info = b.box(pos, w, d, h, yaw, obj, "shelf")
        n = int(rng.integers(2, 6))
        _front_lines(b, info, "shelf-board", obj, heights=list(np.linspace(0, h, n + 1)[1:-1]),
                     splits=[(0.5, 0.0, h)] if rng.random() < 0.4 else [])
    elif kind == "bed":
        w, d = rng.uniform(1.0, 1.9), rng.uniform(1.9, 2.2)
        b.box(pos, w, d, 0.45, yaw, obj, "bed")
        b.box(pos + ey * (d / 2 - 0.04), w, 0.08, rng.uniform(0.8, 1.2), yaw, obj, "bed-headboard")
    elif kind == "sofa":
        w, d = rng.uniform(1.6, 2.6), rng.uniform(0.8, 1.0)
        b.box(pos, w - 0.4, d - 0.2, 0.42, yaw, obj, "sofa-seat")
        b.box(pos + ey * (d / 2 - 0.1), w, 0.2, 0.85, yaw, obj, "sofa-back")
        for s in (-1, 1):
            b.box(pos + ex * s * (w / 2 - 0.1) - ey * 0.1, 0.2, d - 0.2, 0.62, yaw, obj, "sofa-arm")
    elif kind == "rug":
        w, d = rng.uniform(1.2, 3.0), rng.uniform(0.8, 2.2)
        p0 = pos - ex * w / 2 - ey * d / 2 + Z * 0.004
        b.face(b.rect(p0, ex, ey, w, d), obj=obj, kind="rug")
        if rng.random() < 0.5:
            m = rng.uniform(0.08, 0.2)
            b.polyline(b.rect(p0 + ex * m + ey * m, ex, ey, w - 2 * m, d - 2 * m), obj, "rug-border", True)
        return 0.0  # liegt flach, darf überlappt werden
    else:  # crate / box
        w, d, h = rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.0)
        b.box(pos, w, d, h, yaw, obj, "crate")
        if rng.random() < 0.4:
            b.box(pos + Z * h, w * rng.uniform(0.5, 0.9), d * rng.uniform(0.5, 0.9), rng.uniform(0.2, 0.6),
                  yaw if rng.random() < 0.5 else off_axis(yaw + rng.uniform(-40, 40)), obj, "crate")
        return math.hypot(w, d) / 2
    return math.hypot(w, d) / 2


FURNITURE = ["table", "cabinet", "shelf", "bed", "sofa", "rug", "crate"]


def random_furniture(b: SceneBuilder, rng, xr, yr, cam_pos, n_max: int = 6, room_center=None) -> None:
    placed: list[tuple[np.ndarray, float]] = []
    for _ in range(int(rng.integers(0, n_max + 1))):
        kind = str(rng.choice(FURNITURE, p=[0.18, 0.2, 0.14, 0.08, 0.1, 0.12, 0.18]))
        for _try in range(15):
            pos = np.array([rng.uniform(*xr), rng.uniform(*yr), 0.0])
            r = 1.3  # grobe Obergrenze, das Möbel selbst bestimmt die echte Größe
            if np.linalg.norm(pos[:2] - cam_pos[:2]) < r + 0.5:
                continue
            if any(np.linalg.norm(pos[:2] - p[:2]) < r * 0.8 + q for p, q in placed):
                continue
            aligned = rng.random() < 0.65
            target = room_center if room_center is not None else cam_pos
            yaw = front_yaw(pos, target, aligned) if aligned else off_axis(front_yaw(pos, target, False))
            rr = furniture_piece(b, rng, kind, pos, yaw)
            if rr > 0:
                placed.append((pos, rr))
            break
