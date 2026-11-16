"""Straßenschlucht: Fahrbahn mit Markierungen, Bordsteine, Gehwege, Häuserzeilen, Masten"""

from __future__ import annotations

import numpy as np

from ..camera import Camera
from ..scene import SceneBuilder
from .common import add_building, add_pole, camera_rng, off_axis


def generate(rng: np.random.Generator, width: int, height: int):
    b = SceneBuilder("street", rng)
    X, Y, Z = np.eye(3)
    rw = rng.uniform(6, 16)        # Fahrbahnbreite
    sw = rng.uniform(1.8, 5.0)     # Gehwegbreite
    ch = rng.uniform(0.1, 0.18)    # Bordsteinhöhe
    x0, x1 = -20.0, rng.uniform(90, 160)
    L = x1 - x0
    # Kreuzung: Lücke in den Häuserzeilen + Querstraße
    cross = (rng.uniform(15, 50), rng.uniform(8, 14)) if rng.random() < 0.35 else None

    b.face(b.rect([x0, -rw / 2, 0], X, Y, L, rw), obj="road", kind="road")
    for s in (-1, 1):
        y_curb = s * rw / 2
        y_out = s * (rw / 2 + sw)
        b.face([[x0, y_curb, 0], [x1, y_curb, 0], [x1, y_curb, ch], [x0, y_curb, ch]], obj="curb", kind="curb")
        b.face([[x0, y_curb, ch], [x1, y_curb, ch], [x1, y_out, ch], [x0, y_out, ch]], obj="sidewalk", kind="sidewalk")
        if rng.random() < 0.6:
            s_t = rng.uniform(0.8, 2.0)
            o = np.array([x0, min(y_curb, y_out), ch])
            b.grid(o, X, Y, L, sw, s_t, None, "sidewalk", "sidewalk-tiles")
        # Häuserzeile
        x = x0
        while x < x1:
            bw = rng.uniform(6, 22)
            if cross and x < cross[0] < x + bw:
                x = cross[0] + cross[1]
                continue
            bd = rng.uniform(10, 16)
            floors = int(rng.integers(2, 9))
            detail = x < 35 + rng.uniform(0, 30)  # weit entfernte Häuser ohne Fenster
            cy = s * (rw / 2 + sw + bd / 2)
            # Vorderseite (lokal -y) zur Straße: links yaw 0 wäre -y -> Straße liegt bei y<0 für s>0
            yaw = 0.0 if s > 0 else 180.0
            add_building(b, rng, [x + bw / 2, cy, 0], yaw, bw, bd, floors, rng.uniform(2.8, 3.5),
                         window_sides=("front",), roof="flat", detail=detail)
            x += bw + (rng.uniform(0, 3) if rng.random() < 0.2 else 0)

    # Fahrbahnmarkierungen (liegen auf der Fahrbahn-Ebene)
    if rng.random() < 0.8:
        dash, gap, mw = rng.uniform(2, 6), rng.uniform(3, 9), rng.uniform(0.1, 0.18)
        t = x0
        while t < x1:
            b.polyline(b.rect([t, -mw / 2, 0], X, Y, min(dash, x1 - t), mw), "marking", "lane-dash", True)
            t += dash + gap
    if rng.random() < 0.6:
        for s in (-1, 1):
            y = s * (rw / 2 - rng.uniform(0.3, 0.6))
            b.line([x0, y, 0], [x1, y, 0], "marking", "lane-edge")
    if rng.random() < 0.5:  # Zebrastreifen
        xc = cross[0] - 4 if cross else rng.uniform(8, 40)
        sl, swd = rng.uniform(2.5, 4.0), 0.5
        for y in np.arange(-rw / 2 + 0.4, rw / 2 - 0.4, 1.0):
            b.polyline(b.rect([xc, y, 0], X, Y, sl, swd), "marking", "crosswalk", True)
    if cross:  # Querstraße als Linien auf dem Boden
        for xx in (cross[0], cross[0] + cross[1]):
            for s in (-1, 1):
                b.line([xx, s * rw / 2, 0], [xx, s * (rw / 2 + 60), 0], "road", "cross-curb")

    # Masten am Bordstein
    t = x0 + rng.uniform(0, 10)
    spacing = rng.uniform(12, 30)
    while t < x1 and rng.random() < 0.85:
        s = 1 if rng.random() < 0.5 else -1
        add_pole(b, rng, [t, s * (rw / 2 + 0.5), ch], rng.uniform(4, 8), arm_dir=-s * Y if rng.random() < 0.6 else None)
        t += spacing
    # parkende Autos (Kisten, fast ausgerichtet)
    for _ in range(int(rng.integers(0, 6))):
        s = 1 if rng.random() < 0.5 else -1
        pos = np.array([rng.uniform(5, 60), s * (rw / 2 - 1.2), 0])
        yaw = 0.0 if rng.random() < 0.7 else off_axis(rng.uniform(-30, 30))
        obj = b.new_object("car")
        cl, cw = rng.uniform(3.8, 5.0), rng.uniform(1.7, 2.0)
        b.box(pos + Z * 0.3, cl, cw, 0.8, yaw, obj, "car-body")
        b.box(pos + Z * 1.1, cl * 0.55, cw * 0.9, 0.5, yaw, obj, "car-cabin")

    cr = camera_rng(rng)
    on_sidewalk = rng.random() < 0.35
    cam_y = (rng.choice([-1, 1]) * rng.uniform(rw / 2 + 0.5, rw / 2 + sw - 0.3)) if on_sidewalk else rng.uniform(-rw / 3, rw / 3)
    cam_pos = np.array([rng.uniform(-5, 5), cam_y, rng.uniform(1.3, 1.9) + (ch if on_sidewalk else 0)])
    yaw = rng.normal(0, 22)
    cam = Camera.from_pose(cam_pos, yaw, cr["pitch"](-6, 14), cr["roll"](), cr["hfov"](50, 90), width, height)
    b.clutter_on_faces(int(rng.integers(0, 6)), (0.5, 3.0), ("road", "facade"))
    b.meta.update(roadWidth=rw, sidewalkWidth=sw, crossStreet=bool(cross))
    return b.build(), cam
