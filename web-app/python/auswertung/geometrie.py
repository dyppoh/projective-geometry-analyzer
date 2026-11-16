"""


Punkte, Linien und die Geometrie der Auswertung: Fluchtpunkt (kleinste Quadrate), Orthozentrum,
Ausgleichsgerade (Fluchthorizont). Alle Abstände in Pixeln -> Bildbreite



"""

from __future__ import annotations

import math
from itertools import combinations
from typing import NamedTuple

import numpy as np


class GeometryError(ValueError):
    """Geometrisch nicht lösbar (z.B. exakt parallele Linien -> Fluchtpunkt im Unendlichen)."""


class Point(NamedTuple):
    x: float
    y: float

    def dist(self, other: "Point") -> float:
        return math.dist(self, other)

    def to_dict(self) -> dict:
        return {"x": float(self.x), "y": float(self.y)}


class Line(NamedTuple):
    point1: Point
    point2: Point

    @classmethod
    def from_dict(cls, d: dict) -> "Line":
        return cls(Point(float(d["point1"]["x"]), float(d["point1"]["y"])),
                   Point(float(d["point2"]["x"]), float(d["point2"]["y"])))

    def to_dict(self) -> dict:
        return {"point1": self.point1.to_dict(), "point2": self.point2.to_dict()}


# Unter dieser Determinante der Normalengleichung gelten die Linien als exakt parallel
_PARALLEL_DET_EPS = 1e-14


def _unit_normal(line: Line) -> np.ndarray:
    dx = line.point2.x - line.point1.x
    dy = line.point2.y - line.point1.y
    length = math.hypot(dx, dy)
    if length == 0:
        raise GeometryError("Linie mit Länge 0.")
    return np.array([-dy / length, dx / length])


def angle_spread(lines: list[Line]) -> float:
    """Größter Richtungsunterschied (Grad, 0-90) zwischen den Linien, richtungsunabhängig."""
    if len(lines) < 2:
        return 0.0
    a = np.sort([math.degrees(math.atan2(l.point2.y - l.point1.y, l.point2.x - l.point1.x)) % 180 for l in lines])
    gaps = np.diff(np.append(a, a[0] + 180))
    return float(180 - gaps.max())


def intersection(lines: list[Line]) -> tuple[Point, list[float]]:
    """Fluchtpunkt: minimiert die Summe der quadrierten Senkrecht-Abstände zu allen (verlängerten)
    Linien -> (Punkt, Abstand jeder Linie). Bei zwei Linien der exakte Schnittpunkt."""
    if len(lines) < 2:
        raise GeometryError("Für einen Schnittpunkt werden mindestens zwei Linien benötigt.")
    normals = [_unit_normal(line) for line in lines]
    a = np.zeros((2, 2))
    b = np.zeros(2)
    for n, line in zip(normals, lines):
        nn = np.outer(n, n)
        a += nn
        b += nn @ np.array([line.point1.x, line.point1.y])
    if abs(np.linalg.det(a)) < _PARALLEL_DET_EPS:
        raise GeometryError("Linien sind exakt parallel - Fluchtpunkt liegt im Unendlichen.")
    v = np.linalg.solve(a, b)
    distances = [float(abs(n @ (v - np.array([line.point1.x, line.point1.y])))) for n, line in zip(normals, lines)]
    return Point(float(v[0]), float(v[1])), distances


def orthocenter(points: list[Point]) -> tuple[Point, list[Line], list[Line]]:
    """Orthozentrum des Dreiecks aus drei Fluchtpunkten -> (Punkt, Höhen, Dreiecksseiten)."""
    p1, p2, p3 = points
    sides = [Line(p2, p3), Line(p1, p3), Line(p1, p2)]       # jeweils gegenüber dem Eckpunkt der Höhe
    altitudes = []
    for apex, side in zip(points, sides):
        n = _unit_normal(side)
        altitudes.append(Line(apex, Point(apex.x + n[0], apex.y + n[1])))
    v, _ = intersection(altitudes)
    return v, altitudes, sides


def pair_distances(points: list[Point]) -> list[float]:
    return [a.dist(b) for a, b in combinations(points, 2)]


def fit_line(points: list[Point]) -> tuple[Line, list[float]]:
    """Ausgleichsgerade (totale kleinste Quadrate) -> (Gerade, Abstand jedes Punkts)."""
    pts = np.array([[p.x, p.y] for p in points], dtype=float)
    mean = pts.mean(axis=0)
    d = pts - mean
    sxx, syy, sxy = (d[:, 0] ** 2).sum(), (d[:, 1] ** 2).sum(), (d[:, 0] * d[:, 1]).sum()
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)            # Hauptachse der Punktwolke
    normal = np.array([-math.sin(theta), math.cos(theta)])
    c = Point(float(mean[0]), float(mean[1]))
    return Line(c, Point(c.x + math.cos(theta), c.y + math.sin(theta))), [float(abs(row @ normal)) for row in d]
