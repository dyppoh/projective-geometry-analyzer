"""Innenraum: Boden, Decke, Wände mit Türen und Fenstern, Bodenmuster, Möbel"""

from __future__ import annotations

import math

import numpy as np

from ..camera import Camera
from ..scene import SceneBuilder
from .common import camera_rng, place_openings, random_furniture


def generate(rng: np.random.Generator, width: int, height: int):
    b = SceneBuilder("room", rng)
    W, D, H = rng.uniform(3.0, 8.0), rng.uniform(3.5, 10.0), rng.uniform(2.4, 3.4)
    X, Y, Z = np.eye(3)
    b.meta.update(room={"width": W, "depth": D, "height": H})

    # Boden + Muster
    b.face(b.rect([0, 0, 0], X, Y, W, D), obj="floor", kind="floor")
    pattern = rng.choice(["tiles", "planks", "diagonal", "none"], p=[0.35, 0.3, 0.15, 0.2])
    if pattern == "tiles":
        s = rng.uniform(0.3, 0.9)
        b.grid([0, 0, 0], X, Y, W, D, s, s, "floor", "floor-tiles")
    elif pattern == "planks":
        s = rng.uniform(0.12, 0.3)
        if rng.random() < 0.5:
            b.grid([0, 0, 0], X, Y, W, D, s, None, "floor", "floor-planks")
        else:
            b.grid([0, 0, 0], X, Y, W, D, None, s, "floor", "floor-planks")
    elif pattern == "diagonal":
        b.rotated_grid([0, 0, 0], X, Y, W, D, rng.uniform(20, 70), rng.uniform(0.4, 0.9), "floor", "floor-diagonal")
    b.meta["floorPattern"] = str(pattern)

    # Decke
    b.face(b.rect([0, 0, H], X, Y, W, D), obj="ceiling", kind="ceiling")
    if rng.random() < 0.35:
        s = rng.uniform(0.6, 1.2)
        b.grid([0, 0, H], X, Y, W, D, s, s, "ceiling", "ceiling-panels")
    elif rng.random() < 0.3:  # Deckenbalken
        s = rng.uniform(0.8, 1.6)
        b.grid([0, 0, H], X, Y, W, D, s, None, "ceiling", "ceiling-beams")

    # Wände: (Ursprung, Richtung entlang der Wand, Länge, "into" = nach außen)
    walls = [
        (np.array([0, 0, 0.0]), X, W, -Y),
        (np.array([W, 0, 0.0]), Y, D, X),
        (np.array([W, D, 0.0]), -X, W, Y),
        (np.array([0, D, 0.0]), -Y, D, -X),
    ]
    baseboard = rng.uniform(0.06, 0.14) if rng.random() < 0.7 else None
    crown = rng.uniform(0.04, 0.12) if rng.random() < 0.35 else None
    for wi, (o, u, L, into) in enumerate(walls):
        obj = f"wall{wi}"
        openings = place_openings(rng, L, H, door_prob=0.45, max_windows=3, interior=True)
        b.wall(o, u, L, H, into, openings, obj, "wall")
        doors = [(op["x"], op["x"] + op["w"]) for op in openings if op["kind"] == "door"]
        if baseboard:
            for x0, x1 in _gaps(0, L, doors):
                b.line(o + u * x0 + Z * baseboard, o + u * x1 + Z * baseboard, obj, "baseboard")
        if crown:
            b.line(o + Z * (H - crown), o + u * L + Z * (H - crown), obj, "crown-molding")
        # Bilder / Regalbretter / Fliesenspiegel an der Wand (eigene, parallele Ebene)
        for _ in range(rng.integers(0, 3)):
            w, h = rng.uniform(0.3, 1.4), rng.uniform(0.3, 1.0)
            x = rng.uniform(0.1, max(0.11, L - w - 0.1))
            y = rng.uniform(1.0, max(1.01, H - h - 0.2))
            if any(op["x"] - w < x < op["x"] + op["w"] and op["y"] - h < y < op["y"] + op["h"] for op in openings):
                continue
            p0 = o + u * x + Z * y - into * rng.uniform(0.015, 0.05)
            b.face(b.rect(p0, u, Z, w, h), obj=b.new_object("picture"), kind="picture")
            if rng.random() < 0.4:  # Passepartout
                m = min(w, h) * 0.12
                b.polyline(b.rect(p0 + u * m + Z * m, u, Z, w - 2 * m, h - 2 * m), "picture", "picture-inner", True)

    # Kamera nahe einer Wand, Blick in den Raum
    side = rng.integers(4)
    o, u, L, into = walls[side]
    along = rng.uniform(0.25, 0.75) * L
    dist = rng.uniform(0.3, 1.2)
    cam_pos = o + u * along - into * dist
    cam_pos[2] = rng.uniform(1.1, 1.8)
    center = np.array([W / 2, D / 2, cam_pos[2]])
    look = center - cam_pos
    yaw = math.degrees(math.atan2(look[1], look[0])) + rng.normal(0, 18)
    cr = camera_rng(rng)
    cam = Camera.from_pose(cam_pos, yaw, cr["pitch"](-14, 8), cr["roll"](), cr["hfov"](55, 90), width, height)

    # Möbel (nicht direkt vor der Kamera)
    random_furniture(b, rng, (min(0.9, W / 2), max(W / 2, W - 0.9)), (min(0.9, D / 2), max(D / 2, D - 0.9)), cam_pos, n_max=6, room_center=center)
    b.clutter_on_faces(int(rng.integers(0, 8)), (0.2, 1.5), ("wall", "floor"))
    return b.build(), cam


def _gaps(a: float, b: float, blocked: list[tuple[float, float]]):
    cur = a
    for x0, x1 in sorted(blocked):
        if x0 > cur:
            yield cur, x0
        cur = max(cur, x1)
    if cur < b:
        yield cur, b
