"""Außenansicht: Hauptgebäude, Boden, Gehweg, Masten """

from __future__ import annotations

import numpy as np

from ..camera import Camera
from ..geom import rot_z
from ..scene import SceneBuilder
from .common import add_building, add_pole, camera_rng


def generate(rng: np.random.Generator, width: int, height: int):
    b = SceneBuilder("facade", rng)
    X, Y, Z = np.eye(3)
    b.face(b.rect([-200, -200, 0], X, Y, 400, 400), obj="ground", kind="ground")

    fh = rng.uniform(2.8, 3.6)
    w, d = rng.uniform(10, 32), rng.uniform(8, 16)
    floors = int(rng.integers(1, 8))
    roof = "gable" if (floors <= 4 and rng.random() < 0.45) else "flat"
    main = add_building(b, rng, [0, d / 2, 0], 0.0, w, d, floors, fh,
                        window_sides=("front", "left", "right") if rng.random() < 0.5 else ("front",), roof=roof)
    buildings = [main]

    # direkte Nachbarn in einer Flucht
    x_left, x_right = -w / 2, w / 2
    for side in (-1, 1):
        for _ in range(int(rng.integers(0, 3))):
            nw, nd = rng.uniform(6, 22), rng.uniform(8, 16)
            nf = int(np.clip(floors + rng.integers(-2, 3), 1, 9))
            setback = 0.0 if rng.random() < 0.7 else rng.uniform(-2, 4)
            if side < 0:
                cx = x_left - nw / 2
                x_left -= nw
            else:
                cx = x_right + nw / 2
                x_right += nw
            buildings.append(add_building(b, rng, [cx, setback + nd / 2, 0], 0.0, nw, nd, nf, rng.uniform(2.8, 3.6),
                                          window_sides=("front",), roof="gable" if rng.random() < 0.3 else "flat"))
    # gedrehtes Gebäude im Hintergrund -> zusätzliche horizontale Richtungen
    if rng.random() < 0.35:
        yaw = rng.uniform(15, 75) * (1 if rng.random() < 0.5 else -1)
        pos = [rng.uniform(-30, 30), rng.uniform(25, 60), 0]
        add_building(b, rng, pos, yaw, rng.uniform(10, 25), rng.uniform(10, 20), int(rng.integers(2, 10)),
                     rng.uniform(2.8, 3.5), window_sides=("front", "right", "left"), detail=rng.random() < 0.7)

    # Gehweg / Bordstein parallel zur Fassade
    curb = -rng.uniform(2.5, 6.0)
    b.line([x_left - 30, curb, 0], [x_right + 30, curb, 0], "street", "curb")
    if rng.random() < 0.6:
        b.line([x_left - 30, curb - 0.3, 0], [x_right + 30, curb - 0.3, 0], "street", "curb")
    if rng.random() < 0.7:  # Gehwegplatten
        s = rng.uniform(0.6, 1.5)
        b.grid([x_left - 10, curb, 0], X, Y, (x_right - x_left) + 20, -curb, s, s, "sidewalk", "sidewalk-tiles")
    if rng.random() < 0.6:  # Fahrbahn vor dem Haus mit Markierungen
        rw = rng.uniform(6, 12)
        y_far = curb - 0.3 - rw
        b.line([x_left - 30, y_far, 0], [x_right + 30, y_far, 0], "street", "curb")
        yc = curb - 0.3 - rw / 2
        dash, gap = rng.uniform(2, 5), rng.uniform(3, 8)
        t = x_left - 30
        while t < x_right + 30:
            b.polyline(b.rect([t, yc - 0.06, 0], X, Y, dash, 0.12), "marking", "lane-dash", True)
            t += dash + gap
        if rng.random() < 0.5:  # Zebrastreifen quer zur Fahrbahn
            xc = rng.uniform(x_left, x_right)
            for x in np.arange(xc, xc + rng.uniform(3, 6), 1.0):
                b.polyline(b.rect([x, y_far, 0], X, Y, 0.5, rw), "marking", "crosswalk", True)
    for _ in range(int(rng.integers(0, 4))):
        add_pole(b, rng, [rng.uniform(x_left, x_right), curb + 0.5, 0], rng.uniform(3.5, 7.0),
                 arm_dir=-Y if rng.random() < 0.5 else None)

    # Kamera: auf Augenhöhe gegenüber oder erhöht (Drohne/Fenster gegenüber)
    cr = camera_rng(rng)
    dist = rng.uniform(8, 40)
    elevated = rng.random() < 0.15
    cam_pos = np.array([rng.uniform(-w * 0.6, w * 0.6), -dist, rng.uniform(4, 18) if elevated else rng.uniform(1.4, 1.9)])
    target = np.array([rng.uniform(-w / 3, w / 3), 0.0, main["height"] * rng.uniform(0.25, 0.6)])
    # seitlicher Blickwinkel
    yaw_off = rng.normal(0, 20)
    target = cam_pos + rot_z(yaw_off) @ (target - cam_pos)
    cam = Camera.look_at(cam_pos, target, cr["roll"](), cr["hfov"](45, 85), width, height)

    b.clutter_on_faces(int(rng.integers(0, 6)), (0.5, 3.0), ("facade", "ground"))
    b.meta.update(buildings=len(buildings), floors=floors, roof=roof)
    return b.build(), cam
