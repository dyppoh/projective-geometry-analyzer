"""Objekte auf einer Fläche: Kisten, Treppe, Rampe, Wände"""

from __future__ import annotations

import math

import numpy as np

from ..camera import Camera
from ..geom import rot_z
from ..scene import SceneBuilder
from .common import Z, camera_rng, furniture_piece, off_axis


def generate(rng: np.random.Generator, width: int, height: int):
    b = SceneBuilder("objects", rng)
    X, Y, _ = np.eye(3)
    b.face(b.rect([-15, -15, 0], X, Y, 30, 30), obj="ground", kind="ground")
    if rng.random() < 0.4:
        s = rng.uniform(0.3, 1.0)
        b.grid([-6, -6, 0], X, Y, 12, 12, s, s, "ground", "floor-tiles")
    if rng.random() < 0.5:  # Hintergrund-Wand(e)
        wy = rng.uniform(2.0, 4.0)
        b.face(b.rect([-15, wy, 0], X, Z, 30, 6), obj="backwall", kind="wall")
        if rng.random() < 0.5:
            wx = -rng.uniform(2.0, 4.0)
            b.face(b.rect([wx, -15, 0], Y, Z, 15 + wy, 6), obj="sidewall", kind="wall")

    shared_yaw = off_axis(rng.uniform(0, 90)) if rng.random() < 0.5 else None
    centers = []
    for _ in range(int(rng.integers(2, 9))):
        for _try in range(20):
            p = np.array([rng.uniform(-1.8, 1.8), rng.uniform(-1.5, 1.8), 0])
            if all(np.linalg.norm(p[:2] - c[:2]) > 0.8 for c in centers):
                break
        centers.append(p)
        yaw = shared_yaw if shared_yaw is not None else off_axis(rng.uniform(0, 90))
        kind = str(rng.choice(["crate", "crate", "crate", "cabinet", "shelf", "table"]))
        furniture_piece(b, rng, kind, p, yaw)

    if rng.random() < 0.3:
        _stairs(b, rng, np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), 0]), off_axis(rng.uniform(0, 90)))
    if rng.random() < 0.3:
        _ramp(b, rng, np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), 0]), off_axis(rng.uniform(0, 90)))

    target = np.mean(centers, axis=0) + Z * rng.uniform(0.2, 0.6)
    cr = camera_rng(rng)
    dist = rng.uniform(2.5, 7.0)
    elev = math.radians(rng.uniform(8, 45))
    azim = math.radians(rng.uniform(-150, -30))
    cam_pos = target + dist * np.array([math.cos(azim) * math.cos(elev), math.sin(azim) * math.cos(elev), math.sin(elev)])
    cam_pos[2] = max(cam_pos[2], 0.3)
    cam = Camera.look_at(cam_pos, target + rng.normal(0, 0.2, 3), cr["roll"](), cr["hfov"](40, 75), width, height)
    b.clutter_on_faces(int(rng.integers(0, 6)), (0.2, 1.0), ("ground", "crate"))
    return b.build(), cam


def _stairs(b: SceneBuilder, rng, base, yaw: float) -> None:
    """Treppe aus Stufen-Quadern + schräge Wangen-Linien (schräge, parallele Richtung)."""
    obj = b.new_object("stairs")
    R = rot_z(yaw)
    ex, ey = R[:, 0], R[:, 1]
    n = int(rng.integers(3, 9))
    w, tread, rise = rng.uniform(0.8, 1.6), rng.uniform(0.25, 0.32), rng.uniform(0.15, 0.2)
    for i in range(n):
        c = base + ey * (i * tread + tread / 2)
        b.box(c, w, tread, (i + 1) * rise, yaw, obj, "stair-step")
    for s in (-1, 1):  # Handlauf / Wange
        p0 = base + ex * s * w / 2 + Z * (rise + 0.9)
        p1 = base + ex * s * w / 2 + ey * ((n - 1) * tread) + Z * (n * rise + 0.9)
        b.line(p0, p1, obj, "stair-handrail")
        b.line(p0 - Z * 0.9, p0, obj, "stair-post")
        b.line(p1 - Z * 0.9, p1, obj, "stair-post")


def _ramp(b: SceneBuilder, rng, base, yaw: float) -> None:
    """Keil: schräge Fläche mit parallelen Kanten."""
    obj = b.new_object("ramp")
    R = rot_z(yaw)
    ex, ey = R[:, 0], R[:, 1]
    w, l, h = rng.uniform(0.5, 1.5), rng.uniform(0.8, 2.0), rng.uniform(0.3, 0.9)
    a0, a1 = base - ex * w / 2, base + ex * w / 2
    b0, b1 = a0 + ey * l, a1 + ey * l
    t0, t1 = b0 + Z * h, b1 + Z * h
    b.face([a0, a1, t1, t0], obj=obj, kind="ramp-slope")
    b.face([b0, b1, t1, t0], obj=obj, kind="ramp-back")
    b.face([a0, b0, t0], obj=obj, kind="ramp-side")
    b.face([a1, b1, t1], obj=obj, kind="ramp-side")
    if rng.random() < 0.6:
        for t in np.linspace(0, 1, int(rng.integers(3, 7)))[1:-1]:
            b.line(a0 + (t0 - a0) * t, a1 + (t1 - a1) * t, obj, "ramp-rows")
