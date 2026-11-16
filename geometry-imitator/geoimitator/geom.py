"""

Welt-Koordinaten: Meter, Z zeigt nach oben, der Boden liegt bei z = 0
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Nullvektor kann nicht normiert werden.")
    return v / n


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_axis(axis, deg: float) -> np.ndarray:
    """Rodrigues-Rotation um eine beliebige Achse."""
    k = unit(axis)
    a = math.radians(deg)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def canonical(v) -> np.ndarray:
    """Richtung bzw. Normale ohne Vorzeichen: erste deutlich von 0 verschiedene
    Komponente in der Reihenfolge z, x, y wird positiv gemacht."""
    v = unit(v)
    for i in (2, 0, 1):
        if abs(v[i]) > 1e-6:
            return v if v[i] > 0 else -v
    return v


def angle_between_dirs_deg(a, b) -> float:
    """Winkel zwischen zwei ungerichteten Richtungen (0..90°)."""
    c = abs(float(np.dot(unit(a), unit(b))))
    return math.degrees(math.acos(min(1.0, c)))


def clip_polygon_near(poly_cam: np.ndarray, near: float) -> np.ndarray:
    """Sutherland-Hodgman gegen die Ebene z = near (Kamerakoordinaten)."""
    out = []
    n = len(poly_cam)
    for i in range(n):
        a, b = poly_cam[i], poly_cam[(i + 1) % n]
        a_in, b_in = a[2] >= near, b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if out else np.zeros((0, 3))


def clip_segment_near(a: np.ndarray, b: np.ndarray, near: float):
    """Schneidet ein 3D-Segment (Kamerakoordinaten) auf z >= near zu.
    Gibt (a', b', t0, t1) oder None zurück (t: Parameter auf dem Original)."""
    za, zb = a[2], b[2]
    if za < near and zb < near:
        return None
    t0, t1 = 0.0, 1.0
    if za < near:
        t0 = (near - za) / (zb - za)
    elif zb < near:
        t1 = (near - za) / (zb - za)
    return a + t0 * (b - a), a + t1 * (b - a), t0, t1


def clip_segment_rect(p: np.ndarray, q: np.ndarray, w: float, h: float):
    """Liang-Barsky: 2D-Segment auf [0, w] x [0, h] zuschneiden. None wenn außerhalb."""
    x0, y0 = p
    dx, dy = q[0] - x0, q[1] - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0), (dx, w - x0), (-dy, y0), (dy, h - y0)):
        if abs(pp) < EPS:
            if qq < 0:
                return None
            continue
        r = qq / pp
        if pp < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return np.array([x0 + t0 * dx, y0 + t0 * dy]), np.array([x0 + t1 * dx, y0 + t1 * dy])


def clip_line_to_rect_2d(origin, direction, w: float, h: float):
    """Unendliche 2D-Gerade (origin + t*direction) auf das Rechteck [0,w]x[0,h]
    zuschneiden (lokale Koordinaten, z.B. für schräge Fliesenraster)."""
    d = np.asarray(direction, float)
    o = np.asarray(origin, float)
    big = 1e6
    return clip_segment_rect(o - big * d, o + big * d, w, h)


def homogeneous_line(p1, p2) -> np.ndarray:
    l = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    return l / (np.linalg.norm(l[:2]) + EPS)


def point_line_distance(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Abstand von 2D-Punkten (N,2) zur normierten homogenen Geraden."""
    return np.abs(points @ line[:2] + line[2])
