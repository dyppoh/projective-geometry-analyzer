"""
3D-Szene aus Flächen

Die Generatoren (templates/) legen nur Flächen und Linien an. Alle
Eigenschaften werden danach rein geometrisch bestimmt, sie müssen also nicht
von Hand gepflegt werden:

* Richtungen   - Linien mit gleicher 3D-Richtung (bis auf Vorzeichen) -> parallel
* Ebenen       - unendliche Ebenen der Flächen; eine Linie liegt auf einer Ebene,
                 wenn beide Endpunkte (fast) darauf liegen -> koplanar
* Ebenen-Familien - Ebenen mit gleicher Normale -> parallele Ebenen
* senkrecht    - Paare/Tripel von Richtungen mit Skalarprodukt ~ 0

Doppelte bzw. überlappende kollineare Linien (z.B. die gemeinsame Kante zweier
Flächen) werden zusammengeführt.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .geom import angle_between_dirs_deg, canonical, rot_z, unit

DIR_TOL_DEG = 0.05          # gleiche Richtung (Generatoren erzeugen exakt parallele Linien)
PERP_TOL_DEG = 0.05         # senkrecht
PLANE_OFFSET_TOL = 0.012    # m: gleiche Ebene / Linie liegt auf Ebene
COLLINEAR_TOL = 0.004       # m: kollinear (zum Zusammenführen)
MIN_LINE_LENGTH = 0.02      # m


@dataclass
class Face:
    id: str
    polygon: np.ndarray       
    holes: list[np.ndarray] = field(default_factory=list)
    object_id: str = ""
    kind: str = ""

    @property
    def normal(self) -> np.ndarray:
        p = self.polygon
        n = np.zeros(3)
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        return unit(n)

    def to_dict(self) -> dict:
        return {"id": self.id, "polygon": np.round(self.polygon, 5).tolist(),
                "holes": [np.round(h, 5).tolist() for h in self.holes], "object": self.object_id, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: dict) -> "Face":
        return cls(d["id"], np.array(d["polygon"], float), [np.array(h, float) for h in d.get("holes", [])],
                   d.get("object", ""), d.get("kind", ""))


@dataclass
class Line3D:
    id: str
    p1: np.ndarray
    p2: np.ndarray
    object_id: str = ""
    kind: str = ""
    direction_id: str = ""
    plane_ids: list[str] = field(default_factory=list)
    family_ids: list[str] = field(default_factory=list)

    @property
    def vector(self) -> np.ndarray:
        return self.p2 - self.p1

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.vector))

    def to_dict(self) -> dict:
        return {"id": self.id, "p1": np.round(self.p1, 5).tolist(), "p2": np.round(self.p2, 5).tolist(),
                "object": self.object_id, "kind": self.kind, "direction": self.direction_id,
                "planes": self.plane_ids, "planeFamilies": self.family_ids}

    @classmethod
    def from_dict(cls, d: dict) -> "Line3D":
        return cls(d["id"], np.array(d["p1"], float), np.array(d["p2"], float), d.get("object", ""),
                   d.get("kind", ""), d.get("direction", ""), list(d.get("planes", [])), list(d.get("planeFamilies", [])))


@dataclass
class Direction:
    id: str
    vector: np.ndarray
    line_ids: list[str]

    @property
    def kind(self) -> str:
        z = abs(self.vector[2])
        if z > math.cos(math.radians(1)):
            return "vertical"
        if z < math.sin(math.radians(1)):
            return "horizontal"
        return "oblique"

    def to_dict(self) -> dict:
        return {"id": self.id, "vector": np.round(self.vector, 7).tolist(), "kind": self.kind, "lines": self.line_ids}


@dataclass
class Plane:
    id: str
    family_id: str
    normal: np.ndarray      # = Normale
    offset: float         
    face_ids: list[str]
    line_ids: list[str] = field(default_factory=list)
    direction_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "family": self.family_id, "normal": np.round(self.normal, 7).tolist(),
                "offset": round(float(self.offset), 5), "faces": self.face_ids, "lines": self.line_ids,
                "directions": self.direction_ids}


@dataclass
class PlaneFamily:
    id: str
    normal: np.ndarray
    plane_ids: list[str]
    # alle Richtungen senkrecht zur Normale (ihre Fluchtpunkte liegen auf der Fluchtgeraden)
    direction_ids: list[str] = field(default_factory=list)
    # davon die Richtungen, von denen wirklich Linien auf Ebenen dieser Gruppe liegen
    supporting_direction_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "normal": np.round(self.normal, 7).tolist(), "planes": self.plane_ids,
                "directions": self.direction_ids, "supportingDirections": self.supporting_direction_ids}


class SceneBuilder:
    """Sammelt Flächen und Linien. Die Generatoren benutzen nur diese Methoden."""

    def __init__(self, scene_type: str, rng: np.random.Generator):
        self.scene_type = scene_type
        self.rng = rng
        self.faces: list[Face] = []
        self.lines: list[Line3D] = []
        self.meta: dict = {}
        self._obj_counter = itertools.count()

    def new_object(self, kind: str) -> str:
        return f"{kind}{next(self._obj_counter)}"

    # --- Grundbausteine ----------------------------------------------------
    def face(self, polygon, holes=(), obj: str = "", kind: str = "", edges: bool = True) -> Face:
        poly = np.asarray(polygon, float)
        f = Face(f"f{len(self.faces)}", poly, [np.asarray(h, float) for h in holes], obj, kind)
        self.faces.append(f)
        if edges:
            self.polyline(poly, obj, kind, closed=True)
            for h in f.holes:
                self.polyline(h, obj, kind + "-opening", closed=True)
        return f

    def line(self, p1, p2, obj: str = "", kind: str = "") -> None:
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        if np.linalg.norm(p2 - p1) >= MIN_LINE_LENGTH:
            self.lines.append(Line3D(f"l{len(self.lines)}", p1, p2, obj, kind))

    def polyline(self, pts, obj: str = "", kind: str = "", closed: bool = False) -> None:
        pts = np.asarray(pts, float)
        n = len(pts)
        for i in range(n if closed else n - 1):
            self.line(pts[i], pts[(i + 1) % n], obj, kind)

    # --- zusammengesetzte Bausteine ----------------------------------------
    @staticmethod
    def rect(o, u, v, w: float, h: float) -> np.ndarray:
        o, u, v = (np.asarray(x, float) for x in (o, u, v))
        return np.array([o, o + u * w, o + u * w + v * h, o + v * h])

    def box(self, base_center, w: float, d: float, h: float, yaw_deg: float = 0.0, obj: str = "",
            kind: str = "box", bottom: bool = False) -> dict:
        """Quader: Grundfläche zentriert bei base_center, w entlang lokal x, d entlang lokal y."""
        R = rot_z(yaw_deg)
        c = np.asarray(base_center, float)
        ex, ey, ez = R[:, 0], R[:, 1], np.array([0, 0, 1.0])
        p = lambda sx, sy, sz: c + ex * (sx * w / 2) + ey * (sy * d / 2) + ez * (sz * h)  # noqa: E731
        corners = {(sx, sy, sz): p(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)}
        q = lambda *keys: [corners[k] for k in keys]  # noqa: E731
        faces = {
            "front": q((-1, -1, 0), (1, -1, 0), (1, -1, 1), (-1, -1, 1)),
            "back": q((1, 1, 0), (-1, 1, 0), (-1, 1, 1), (1, 1, 1)),
            "left": q((-1, 1, 0), (-1, -1, 0), (-1, -1, 1), (-1, 1, 1)),
            "right": q((1, -1, 0), (1, 1, 0), (1, 1, 1), (1, -1, 1)),
            "top": q((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
        }
        if bottom:
            faces["bottom"] = q((-1, -1, 0), (-1, 1, 0), (1, 1, 0), (1, -1, 0))
        for name, poly in faces.items():
            self.face(poly, obj=obj, kind=f"{kind}-{name}")
        return {"center": c, "ex": ex, "ey": ey, "w": w, "d": d, "h": h, "yaw": yaw_deg}

    def grid(self, o, u, v, w: float, h: float, spacing_u: float | None, spacing_v: float | None,
             obj: str = "", kind: str = "grid") -> None:
        """Innere Rasterlinien eines Rechtecks (o, u*w, v*h)."""
        o, u, v = (np.asarray(x, float) for x in (o, u, v))
        if spacing_u:
            for t in np.arange(spacing_u, w - 1e-6, spacing_u):
                self.line(o + u * t, o + u * t + v * h, obj, kind)
        if spacing_v:
            for t in np.arange(spacing_v, h - 1e-6, spacing_v):
                self.line(o + v * t, o + v * t + u * w, obj, kind)

    def rotated_grid(self, o, u, v, w: float, h: float, angle_deg: float, spacing: float,
                     obj: str = "", kind: str = "grid-rotated", both: bool = True) -> None:
        """Raster unter einem Winkel innerhalb eines Rechtecks (z.B. diagonale Fliesen)."""
        from .geom import clip_line_to_rect_2d
        o, u, v = (np.asarray(x, float) for x in (o, u, v))
        angles = [angle_deg, angle_deg + 90] if both else [angle_deg]
        diag = math.hypot(w, h)
        for a in angles:
            d = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            nrm = np.array([-d[1], d[0]])
            for s in np.arange(-diag, diag, spacing):
                seg = clip_line_to_rect_2d(np.array([w / 2, h / 2]) + nrm * s, d, w, h)
                if seg is None:
                    continue
                (a0, b0), (a1, b1) = seg
                self.line(o + u * a0 + v * b0, o + u * a1 + v * b1, obj, kind)

    def wall(self, o, u, length: float, height: float, into, openings: list[dict], obj: str = "",
             kind: str = "wall", up=(0, 0, 1)) -> Face:
        """Wand mit Öffnungen (Fenster/Türen) inkl. Laibung und zurückgesetzter Fläche.

        o: untere Ecke, u: Richtung entlang der Wand, into: zeigt vom Betrachter
        weg in die Wand hinein (dort liegt die Laibung).
        openings: {"x", "y", "w", "h", "recess", "kind", "mullions": bool}
        """
        o, u, into, up = (np.asarray(x, float) for x in (o, u, into, up))
        holes = []
        for op in openings:
            holes.append(self.rect(o + u * op["x"] + up * op["y"], u, up, op["w"], op["h"]))
        face = self.face(self.rect(o, u, up, length, height), holes, obj, kind)
        for op, hole in zip(openings, holes):
            r = op.get("recess", 0.12)
            okind = op.get("kind", "window")
            inner = hole + into * r
            for i in range(4):  # Laibung
                self.face([hole[i], hole[(i + 1) % 4], inner[(i + 1) % 4], inner[i]], obj=obj, kind=f"{okind}-reveal")
            self.face(inner, obj=obj, kind=f"{okind}-pane")
            if op.get("mullions"):
                mid_u = inner[0] + u * op["w"] / 2
                self.line(mid_u, mid_u + up * op["h"], obj, f"{okind}-mullion")
                t = inner[0] + up * op["h"] * op.get("transom", 0.65)
                self.line(t, t + u * op["w"], obj, f"{okind}-transom")
            if op.get("panels"):  # Türfüllungen
                m = 0.12
                p0 = inner[0] + u * m + up * m
                self.polyline(self.rect(p0, u, up, op["w"] - 2 * m, op["h"] * 0.42 - m), obj, f"{okind}-panel", True)
                p1 = inner[0] + u * m + up * op["h"] * 0.5
                self.polyline(self.rect(p1, u, up, op["w"] - 2 * m, op["h"] * 0.5 - m), obj, f"{okind}-panel", True)
            if op.get("sill"):
                sill_w = op["w"] + 0.12
                c = o + u * (op["x"] + op["w"] / 2) + up * (op["y"] - 0.04) - into * 0.03
                yaw = math.degrees(math.atan2(u[1], u[0]))
                self.box(c, sill_w, 0.06, 0.04, yaw, obj, f"{okind}-sill")
        return face

    def clutter_on_faces(self, n: int, length_range=(0.3, 2.0), kinds: tuple[str, ...] = ()) -> None:
        """Linien ohne Partner: zufällige Richtung innerhalb einer Fläche (Risse, Deko, Kabel)."""
        faces = [f for f in self.faces if not kinds or any(f.kind.startswith(k) for k in kinds)]
        if not faces:
            return
        for _ in range(n):
            f = faces[self.rng.integers(len(faces))]
            wts = self.rng.dirichlet(np.ones(len(f.polygon)))
            p = wts @ f.polygon
            nrm = f.normal
            a = unit(np.cross(nrm, [0.3, 0.5, 0.8]) + 1e-9)
            b = np.cross(nrm, a)
            ang = self.rng.uniform(0, math.pi)
            d = math.cos(ang) * a + math.sin(ang) * b
            L = self.rng.uniform(*length_range)
            self.line(p - d * L / 2, p + d * L / 2, "clutter", "clutter")

    def build(self) -> "Scene":
        return Scene.from_parts(self.scene_type, self.faces, self.lines, self.meta)


# ---------------------------------------------------------------------------
# Szene + geometrische Eigenschaften
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    scene_type: str
    faces: list[Face]
    lines: list[Line3D]
    directions: list[Direction]
    families: list[PlaneFamily]
    planes: list[Plane]
    perpendicular_pairs: list[list[str]]
    perpendicular_triples: list[list[str]]
    meta: dict = field(default_factory=dict)

    @property
    def line_by_id(self) -> dict[str, Line3D]:
        return {l.id: l for l in self.lines}

    @property
    def direction_by_id(self) -> dict[str, Direction]:
        return {d.id: d for d in self.directions}

    @classmethod
    def from_parts(cls, scene_type: str, faces: list[Face], lines: list[Line3D], meta: dict | None = None) -> "Scene":
        lines = merge_collinear(lines)
        directions = cluster_directions(lines)
        families, planes = cluster_planes(faces)
        assign_planes(lines, planes, families, directions)
        pairs = [[a.id, b.id] for a, b in itertools.combinations(directions, 2)
                 if abs(90 - angle_between_dirs_deg(a.vector, b.vector)) < PERP_TOL_DEG]
        pair_set = {tuple(p) for p in pairs}
        triples = [[a, b, c] for a, b, c in itertools.combinations([d.id for d in directions], 3)
                   if {(a, b), (a, c), (b, c)} <= pair_set]
        return cls(scene_type, faces, lines, directions, families, planes, pairs, triples, meta or {})

    def to_dict(self) -> dict:
        return {
            "sceneType": self.scene_type,
            "meta": self.meta,
            "directions": [d.to_dict() for d in self.directions],
            "planeFamilies": [f.to_dict() for f in self.families],
            "planes": [p.to_dict() for p in self.planes],
            "relations": {"perpendicularPairs": self.perpendicular_pairs,
                          "perpendicularTriples": self.perpendicular_triples},
            "lines3d": [l.to_dict() for l in self.lines],
            "faces": [f.to_dict() for f in self.faces],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        faces = [Face.from_dict(f) for f in d["faces"]]
        lines = [Line3D.from_dict(l) for l in d["lines3d"]]
        dirs = [Direction(x["id"], np.array(x["vector"], float), list(x["lines"])) for x in d["directions"]]
        fams = [PlaneFamily(x["id"], np.array(x["normal"], float), list(x["planes"]), list(x["directions"]),
                            list(x.get("supportingDirections", []))) for x in d["planeFamilies"]]
        planes = [Plane(x["id"], x["family"], np.array(x["normal"], float), float(x["offset"]), list(x["faces"]),
                        list(x["lines"]), list(x["directions"])) for x in d["planes"]]
        rel = d.get("relations", {})
        return cls(d["sceneType"], faces, lines, dirs, fams, planes, rel.get("perpendicularPairs", []),
                   rel.get("perpendicularTriples", []), d.get("meta", {}))


def merge_collinear(lines: list[Line3D]) -> list[Line3D]:
    """Führt identische/überlappende kollineare Segmente zusammen (gemeinsame Kanten)."""
    lines = [l for l in lines if l.length >= MIN_LINE_LENGTH]
    if not lines:
        return []
    P1 = np.array([l.p1 for l in lines])
    P2 = np.array([l.p2 for l in lines])
    D = P2 - P1
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    cos_tol = math.cos(math.radians(DIR_TOL_DEG))
    rep_of = np.full(len(lines), -1)
    reps: list[np.ndarray] = []
    for i in range(len(lines)):
        if rep_of[i] >= 0:
            continue
        d = canonical(D[i])
        hit = (rep_of < 0) & (np.abs(D @ d) > cos_tol)
        rep_of[hit] = len(reps)
        reps.append(d)
    # Schlüssel: Fußpunkt der Geraden (Lot vom Ursprung), quantisiert
    buckets: dict[tuple, list[int]] = {}
    quantum = 0.00173  # "krummer" Wert, damit runde Koordinaten nicht auf Rundungsgrenzen fallen
    for k, d in enumerate(reps):
        idx = np.where(rep_of == k)[0]
        foot = P1[idx] - np.outer(P1[idx] @ d, d)
        keys = np.round(foot / quantum).astype(np.int64)
        for i, key in zip(idx, map(tuple, keys)):
            buckets.setdefault((k, *key), []).append(int(i))
    out: list[Line3D] = []
    for (k, *_), members in buckets.items():
        d = reps[k]
        base = P1[members[0]] - (P1[members[0]] @ d) * d
        iv = sorted((min(P1[j] @ d, P2[j] @ d), max(P1[j] @ d, P2[j] @ d), j) for j in members)
        groups: list[list] = []
        for a, b, j in iv:
            if groups and a <= groups[-1][1] + COLLINEAR_TOL:
                groups[-1][1] = max(groups[-1][1], b)
                groups[-1][2].append(j)
            else:
                groups.append([a, b, [j]])
        for a, b, js in groups:
            src = lines[max(js, key=lambda j: lines[j].length)]
            out.append(Line3D("", base + d * a, base + d * b, src.object_id, src.kind))
    for k, l in enumerate(out):
        l.id = f"L{k}"
    return out


def cluster_directions(lines: list[Line3D]) -> list[Direction]:
    cos_tol = math.cos(math.radians(DIR_TOL_DEG))
    reps: list[np.ndarray] = []
    members: list[list[str]] = []
    for l in lines:
        d = canonical(l.vector)
        for k, r in enumerate(reps):
            if abs(r @ d) > cos_tol:
                members[k].append(l.id)
                l.direction_id = f"d{k}"
                break
        else:
            reps.append(d)
            members.append([l.id])
            l.direction_id = f"d{len(reps) - 1}"
    # Mittelwert als Repräsentant (Vorzeichen angleichen)
    by_id = {l.id: l for l in lines}
    dirs = []
    for k, (r, ids) in enumerate(zip(reps, members)):
        acc = np.zeros(3)
        for i in ids:
            v = unit(by_id[i].vector)
            acc += v if v @ r >= 0 else -v
        dirs.append(Direction(f"d{k}", canonical(acc), ids))
    return dirs


def cluster_planes(faces: list[Face]) -> tuple[list[PlaneFamily], list[Plane]]:
    cos_tol = math.cos(math.radians(DIR_TOL_DEG))
    fam_normals: list[np.ndarray] = []
    face_fam: list[int] = []
    for f in faces:
        n = canonical(f.normal)
        for k, r in enumerate(fam_normals):
            if abs(r @ n) > cos_tol:
                face_fam.append(k)
                break
        else:
            fam_normals.append(n)
            face_fam.append(len(fam_normals) - 1)
    families = [PlaneFamily(f"F{k}", n, []) for k, n in enumerate(fam_normals)]
    planes: list[Plane] = []
    for f, k in zip(faces, face_fam):
        n = fam_normals[k]
        off = float(np.mean(f.polygon @ n))
        for p in planes:
            if p.family_id == f"F{k}" and abs(p.offset - off) < PLANE_OFFSET_TOL:
                p.face_ids.append(f.id)
                break
        else:
            planes.append(Plane(f"P{len(planes)}", f"F{k}", n, off, [f.id]))
    for p in planes:
        families[int(p.family_id[1:])].plane_ids.append(p.id)
    return families, planes


def assign_planes(lines: list[Line3D], planes: list[Plane], families: list[PlaneFamily],
                  directions: list[Direction]) -> None:
    if planes:
        N = np.array([p.normal for p in planes])
        O = np.array([p.offset for p in planes])
        fam_by_id = {f.id: f for f in families}
        for l in lines:
            on = (np.abs(N @ l.p1 - O) < PLANE_OFFSET_TOL) & (np.abs(N @ l.p2 - O) < PLANE_OFFSET_TOL)
            l.plane_ids = [planes[i].id for i in np.where(on)[0]]
            l.family_ids = sorted({planes[i].family_id for i in np.where(on)[0]}, key=lambda s: int(s[1:]))
            for i in np.where(on)[0]:
                planes[i].line_ids.append(l.id)
                if l.direction_id not in planes[i].direction_ids:
                    planes[i].direction_ids.append(l.direction_id)
            for fid in l.family_ids:
                fam = fam_by_id[fid]
                if l.direction_id not in fam.supporting_direction_ids:
                    fam.supporting_direction_ids.append(l.direction_id)
    sin_tol = math.sin(math.radians(PERP_TOL_DEG))
    for fam in families:
        fam.direction_ids = [d.id for d in directions if abs(d.vector @ fam.normal) < sin_tol]
        fam.supporting_direction_ids.sort(key=lambda s: int(s[1:]))
