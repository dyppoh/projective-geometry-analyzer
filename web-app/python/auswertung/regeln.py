"""

Abstandsmaße nach den Regeln der projektiven Geometrie

Jede Prüfgruppe liefert je Kategorie den Ø, den Median und jeden Einzelabstand - als Anteil der
Bildbreite. Ob ein Wert zu groß ist, entscheidet erst bewertung.py.

Rule 1     Richtung: Abstand jeder Linie zum gemeinsamen Fluchtpunkt ("rule1").
Rule 2     Richtungen mit >= 2 Untergruppen: wie Rule 1 je Untergruppe ("rule2") und Abstand der
           Fluchtpunkte der Untergruppen zueinander ("rule2cross").
Rule 3/4   Fluchtpunkte einer Ebenenschar (mit allen Ebenen der beteiligten Richtungen) liegen auf einem
           Horizont: Abstand jedes Fluchtpunkts zur Ausgleichsgeraden ("rule3"); ab 3 Fluchtpunkten aus
           >= 2 Richtungen.
Rule 5     Je Kombination aus einer Untergruppe dreier senkrechter Richtungen ein Orthozentrum (höchstens
           OZ_MAX_PER_GROUP je Senkrecht-Gruppe); alle Orthozentren des Bildes in einer Prüfgruppe:
           Abstand zueinander ("rule5cross") und zur Bildmitte ("rule5center").

Fluchtpunkte, deren Linien sich um weniger als die Winkelgrenze der Regel unterscheiden (MIN_ANGLE),
entfallen für diese Regel. Gruppen mit genau 2 Linien: der Fluchtpunkt ist ihr exakter Schnittpunkt, ihr
eigener Abstand (immer 0) wird nicht geprüft, für die weiteren Regeln zählt er. Untergruppen mit einer
Linie nehmen für Rule 3/4 und 5 den Fluchtpunkt ihrer Richtung, wenn diese sonst keinen beiträgt.
Jede Prüfung, die entfällt, erscheint als Prüfgruppe ohne Messwerte mit Begründung ("reason") - ebenso
Gruppen aus dem Editor, die gar nicht in die Prüfung eingehen (modell.Modell.verworfen).
"""

from __future__ import annotations

from itertools import combinations, product
from math import prod
from statistics import median

from .geometrie import GeometryError, Line, Point, angle_spread, fit_line, intersection, orthocenter, pair_distances
from .modell import Ebene, Modell

# Winkelgrenze je Regel (Grad) - Rule 1 ist die Basis des Grundwerts und sollte die kleinste haben
MIN_ANGLE = {"1": 1.5, "2": 1.5, "2-cross": 1.5, "3/4": 1.5, "5": 0.5}
OZ_MAX_PER_GROUP = 10
KIND_RULE = {"richtung": "1", "ebene": "2", "horizont": "3/4", "senkrecht": "5"}

COLOR_LINE = "#3b82f6"
COLOR_POINT = "#214d94"
COLOR_TRIANGLE = "#000000"
COLOR_ORTHOCENTER_LINE = "#a855f7"
COLOR_ORTHOCENTER_POINT = "#551d8a"
COLOR_HORIZON_LINE = "#1e3a8a"

TWO_LINES = ("Nur zwei Linien: Der Fluchtpunkt ist ihr exakter Schnittpunkt, der Abstand (immer 0) wird nicht "
             "geprüft. Für die weiteren Regeln zählt der Fluchtpunkt.")


def layer(label: str, line_color: str, point_color: str, lines: list[Line] = (), points: list[Point] = (),
          extend: bool = True) -> dict:
    d = {"label": label, "lineColor": line_color, "pointColor": point_color,
         "lines": [l.to_dict() for l in lines], "points": [p.to_dict() for p in points]}
    if not extend:
        d["extend"] = False
    return d


def _deg(v: float) -> str:
    return f"{v:.1f}".replace(".", ",") + "°"


class Regeln:
    def __init__(self, m: Modell):
        self.m = m
        self.width = m.width
        self.center = Point(m.width / 2, m.height / 2)
        self.ebene = {e.id: e for e in m.ebenen}
        self.richtung = {r.id: r for r in m.richtungen}
        self.owner = {e.id: r.id for r in m.richtungen for e in r.ebenen}
        self.gruppen: list[dict] = []
        self.plane_vp: dict[str, Point] = {}
        self.plane_spread: dict[str, float] = {}
        self.group_vp: dict[str, Point] = {}
        self.group_spread: dict[str, float] = {}
        for e in m.ebenen:
            if len(e.lines) >= 2:
                try:
                    self.plane_vp[e.id], _ = intersection(e.lines)
                    self.plane_spread[e.id] = angle_spread(e.lines)
                except GeometryError:
                    pass

    # --- Hilfen ---------------------------------------------------------------
    def _plane_vp(self, pid: str, rule: str) -> Point | None:
        vp = self.plane_vp.get(pid)
        return vp if vp is not None and self.plane_spread[pid] >= MIN_ANGLE[rule] else None

    def _group_vp(self, gid: str, rule: str) -> Point | None:
        vp = self.group_vp.get(gid)
        return vp if vp is not None and self.group_spread[gid] >= MIN_ANGLE[rule] else None

    def _vp_points(self, plane_ids: list[str], rule: str) -> list[tuple[str, Point]]:
        """Fluchtpunkte für Rule 3/4 und 5: je Untergruppe der eigene; Untergruppen mit einer Linie nehmen
        den ihrer Richtung - nur, wenn die Richtung hier sonst keinen beiträgt."""
        own = {self.owner[i] for i in plane_ids if self._plane_vp(i, rule) is not None}
        out, used_dir = [], set()
        for i in dict.fromkeys(plane_ids):
            if (vp := self._plane_vp(i, rule)) is not None:
                out.append((i, vp))
            elif len(self.ebene[i].lines) == 1:
                g = self.owner[i]
                if g not in own and g not in used_dir and (vp := self._group_vp(g, rule)) is not None:
                    out.append((i, vp))
                    used_dir.add(g)
        return out

    def _all_planes(self, plane_ids: list[str]) -> list[str]:
        """Die angegebenen Untergruppen und alle weiteren Ebenen ihrer Richtungen (gleicher Fluchtpunkt)."""
        out = list(dict.fromkeys(plane_ids))
        for g in dict.fromkeys(self.owner[i] for i in plane_ids):
            out += [e.id for e in self.richtung[g].ebenen if e.id not in out]
        return out

    def _why(self, pid: str, rule: str) -> str:
        """Warum eine Untergruppe für die Regel keinen eigenen Fluchtpunkt hat."""
        e = self.ebene[pid]
        if len(e.lines) == 1:
            return "nur eine Linie"
        if pid not in self.plane_vp:
            return "Linien exakt parallel"
        return f"Linien zu parallel ({_deg(self.plane_spread[pid])} < {_deg(MIN_ANGLE[rule])})"

    def _missing(self, plane_ids: list[str], used: list[str], rule: str) -> str:
        """Welche Untergruppen keinen Fluchtpunkt beitragen und warum."""
        why = [f"{self.ebene[i].label} ("
               + ("eine Linie, Richtung schon vertreten oder ohne Fluchtpunkt" if len(self.ebene[i].lines) == 1
                  else self._why(i, rule)) + ")"
               for i in dict.fromkeys(plane_ids) if i not in used]
        return f" Ohne Fluchtpunkt: {'; '.join(why)}." if why else ""

    def _werte(self, px: list[float], refs: list[str]) -> dict:
        return {"avg": sum(px) / len(px) / self.width, "median": median(px) / self.width,
                "singles": [{"ref": r, "value": v / self.width, "valuePx": v} for r, v in zip(refs, px)]}

    def _add(self, id: str, rule: str, title: str, subject: str, teile: dict, layers: list[dict],
             spread: float | None = None) -> None:
        self.gruppen.append({"id": id, "rule": rule, "title": title, "subject": subject, "teile": teile,
                             "layers": layers, "angleSpread": spread, "minAngle": MIN_ANGLE[rule]})

    def _skip(self, id: str, rule: str, title: str, subject: str, reason: str, layers: list[dict]) -> None:
        """Prüfung entfällt - als Prüfgruppe ohne Messwerte mit Begründung."""
        self.gruppen.append({"id": id, "rule": rule, "title": title, "subject": subject, "teile": {},
                             "layers": layers, "angleSpread": None, "minAngle": MIN_ANGLE[rule], "reason": reason})

    def _plane_layers(self, plane_ids: list[str], points: list[Point]) -> list[dict]:
        lines = [l for i in dict.fromkeys(plane_ids) for l in self.ebene[i].lines]
        return [layer("Fluchtlinien & Fluchtpunkte", COLOR_LINE, COLOR_POINT, lines, points)]

    # --- Editor-Gruppen ohne Wirkung --------------------------------------------
    def verworfen(self) -> None:
        for k, v in enumerate(self.m.verworfen):
            self._skip(f"verworfen-{k}", KIND_RULE[v["kind"]], "Gruppe nicht verwendet", v["name"], v["reason"],
                       [layer("Linien der Gruppe", COLOR_LINE, COLOR_POINT, v["lines"])])

    # --- Rule 1 ---------------------------------------------------------------
    def rule1(self) -> None:
        for index, r in enumerate(self.m.richtungen):
            refs = r.line_refs()
            ids, lines = [i for i, _ in refs], [l for _, l in refs]
            gid, title = f"rule1-{index}", f"Fluchtpunkt von {r.name}"
            try:
                vp, distances = intersection(lines)
            except GeometryError:
                why = ("Nur eine Linie - kein Fluchtpunkt." if len(lines) == 1 else
                       "Linien exakt parallel - der Fluchtpunkt liegt im Unendlichen.")
                self._skip(gid, "1", title, r.name, why, [layer("Fluchtlinien", COLOR_LINE, COLOR_POINT, lines)])
                continue
            self.group_vp[r.id] = vp
            self.group_spread[r.id] = spread = angle_spread(lines)
            layers = [layer("Fluchtlinien & Fluchtpunkt", COLOR_LINE, COLOR_POINT, lines, [vp])]
            if spread < MIN_ANGLE["1"]:
                self._skip(gid, "1", title, r.name,
                           f"Linien zu parallel: Ihre Richtungen unterscheiden sich um höchstens {_deg(spread)} "
                           f"(Grenze {_deg(MIN_ANGLE['1'])}) - der Fluchtpunkt entfällt.", layers)
                continue
            if len(lines) == 2:
                self._skip(gid, "1", title, r.name, TWO_LINES, layers)
                continue
            self._add(gid, "1", title, r.name, {"rule1": self._werte(distances, ids)}, layers, spread)

    # --- Rule 2 ---------------------------------------------------------------
    def rule2(self) -> None:
        index = cross = 0
        for r in self.m.richtungen:
            if len(r.ebenen) < 2:
                continue  # identisch zu Rule 1
            own: list[tuple[Ebene, Point]] = []
            for e in r.ebenen:
                gid, title = f"rule2-{index}", f"Fluchtpunkt von {e.label}"
                index += 1
                vp = self._plane_vp(e.id, "2")
                raw = self.plane_vp.get(e.id)
                layers = [layer("Fluchtlinien & Fluchtpunkt der Ebene", COLOR_LINE, COLOR_POINT, e.lines,
                                [raw] if raw is not None else [])]
                if vp is not None:
                    own.append((e, vp))
                if vp is None:
                    reason = ("Nur eine Linie - kein eigener Fluchtpunkt (für Rule 3/4 und 5 zählt der der Richtung)."
                              if len(e.lines) == 1 else f"{self._why(e.id, '2')} - kein Fluchtpunkt.")
                    self._skip(gid, "2", title, e.label, reason, layers)
                    continue
                if len(e.lines) == 2:
                    self._skip(gid, "2", title, e.label, TWO_LINES, layers)
                    continue
                _, distances = intersection(e.lines)
                self._add(gid, "2", title, e.label, {"rule2": self._werte(distances, e.line_ids)}, layers)
            gid, title, subject = f"rule2cross-{cross}", f"Fluchtpunkte der Ebenen von {r.name}", f"Ebenen von {r.name}"
            cross += 1
            plane_ids = [e.id for e in r.ebenen]
            if len(own) < 2:
                self._skip(gid, "2-cross", title, subject,
                           f"{'Keine' if not own else 'Nur eine'} Ebene von {r.name} hat einen eigenen Fluchtpunkt - "
                           "für einen Vergleich braucht es mindestens 2." + self._missing(plane_ids, [e.id for e, _ in own], "2"),
                           self._plane_layers(plane_ids, [vp for _, vp in own]))
                continue
            refs = [f"{a.label} ↔ {b.label}" for (a, _), (b, _) in combinations(own, 2)]
            self._add(gid, "2-cross", title, subject,
                      {"rule2cross": self._werte(pair_distances([vp for _, vp in own]), refs)},
                      [layer("Fluchtlinien & Fluchtpunkte der Ebenen", COLOR_LINE, COLOR_POINT,
                             [l for e, _ in own for l in e.lines], [vp for _, vp in own])])

    # --- Rule 3 & 4 -----------------------------------------------------------
    def rule3(self) -> None:
        seen: set[frozenset[str]] = set()
        for index, h in enumerate(self.m.horizonte):
            plane_ids = self._all_planes(h.ebenen)
            if frozenset(plane_ids) in seen:
                continue
            seen.add(frozenset(plane_ids))
            pts = self._vp_points(plane_ids, "3/4")
            refs, vps = [self.ebene[i].label for i, _ in pts], [vp for _, vp in pts]
            gid, title = f"rule3-{index}", "Fluchthorizont paralleler Ebenen"
            subject = f"Horizont von {h.name}"
            if len(pts) < 3 or len({self.owner[i] for i, _ in pts}) < 2:
                why = (f"{['Kein gültiger Fluchtpunkt', 'Nur ein gültiger Fluchtpunkt', 'Nur zwei gültige Fluchtpunkte'][len(pts)]}"
                       " - ein Horizont braucht mindestens 3 (durch 2 Punkte geht immer eine Gerade)." if len(pts) < 3 else
                       "Alle Fluchtpunkte stammen aus derselben Richtung - ein Horizont ist nicht bestimmbar.")
                self._skip(gid, "3/4", title, subject, why + self._missing(plane_ids, [i for i, _ in pts], "3/4"),
                           self._plane_layers(plane_ids, vps))
                continue
            horizon, distances = fit_line(vps)
            self._add(gid, "3/4", title, subject, {"rule3": self._werte(distances, refs)},
                      [layer("Fluchtlinien der Ebenen", COLOR_LINE, COLOR_POINT,
                             [l for i, _ in pts for l in self.ebene[i].lines]),
                       layer("Fluchthorizont & Fluchtpunkte", COLOR_HORIZON_LINE, COLOR_POINT, [horizon], vps)])

    # --- Rule 5 ---------------------------------------------------------------
    def rule5(self) -> None:
        triangles = []   # (ref, Orthozentrum, Fluchtpunkte, Seiten, Höhen)
        failed: list[str] = []
        seen: set[frozenset[str]] = set()
        for index, s in enumerate(self.m.senkrecht):
            plane_ids = self._all_planes(s.ebenen)
            directions = list(dict.fromkeys(self.owner[i] for i in plane_ids))
            pts = self._vp_points(plane_ids, "5")
            gid, title = f"rule5-entfallen-{index}", "Orthozentren"
            subject = f"{s.name} ({', '.join(self.richtung[g].name for g in directions)})"
            layers = self._plane_layers(plane_ids, [vp for _, vp in pts])
            if len(directions) != 3:
                self._skip(gid, "5", title, subject,
                           f"Die Linien stammen aus {len(directions)} Richtung(en) - für ein Orthozentrum braucht es "
                           "genau 3 zueinander senkrechte.", layers)
                continue
            by_dir: dict[str, list[tuple[str, Point]]] = {g: [] for g in directions}
            for i, vp in pts:
                by_dir[self.owner[i]].append((i, vp))
            missing = [self.richtung[g].name for g, v in by_dir.items() if not v]
            if missing:
                self._skip(gid, "5", title, subject,
                           f"Für {', '.join(missing)} gibt es keinen gültigen Fluchtpunkt - kein Dreieck möglich."
                           + self._missing(plane_ids, [i for i, _ in pts], "5"), layers)
                continue
            if prod(len(v) for v in by_dir.values()) > OZ_MAX_PER_GROUP:
                # je Richtung nach Linienzahl sortieren, die kleinste Ebene der Richtung mit den meisten
                # Ebenen weglassen, bis die Zahl der Kombinationen passt
                for g in by_dir:
                    by_dir[g].sort(key=lambda t: -len(self.ebene[t[0]].lines))
                while prod(len(v) for v in by_dir.values()) > OZ_MAX_PER_GROUP:
                    max(by_dir.values(), key=len).pop()
            for combo in product(*by_dir.values()):
                ids, vps = [i for i, _ in combo], [vp for _, vp in combo]
                if frozenset(ids) in seen:
                    continue
                seen.add(frozenset(ids))
                try:
                    pp, altitudes, sides = orthocenter(vps)
                except GeometryError:
                    failed.append("×".join(ids))
                    continue
                triangles.append(("×".join(ids), pp, vps, sides, altitudes))
        if not triangles:
            if failed:
                self._skip("rule5", "5", "Orthozentren", "Orthozentren",
                           f"Kein Orthozentrum berechenbar ({', '.join(failed)}).", [])
            return
        centers = [t[1] for t in triangles]
        refs = [t[0] for t in triangles]
        teile = {}
        if len(triangles) >= 2:
            teile["rule5cross"] = self._werte(pair_distances(centers), [f"{a} ↔ {b}" for a, b in combinations(refs, 2)])
        teile["rule5center"] = self._werte([c.dist(self.center) for c in centers], refs)
        self._add("rule5", "5", f"Orthozentren aus {len(triangles)} Fluchtpunkt-Dreieck(en)", "Orthozentren", teile,
                  [layer("Fluchtpunkte", COLOR_LINE, COLOR_POINT, [], [v for t in triangles for v in t[2]]),
                   layer("Dreiecke der Fluchtpunkte", COLOR_TRIANGLE, COLOR_TRIANGLE,
                         [s for t in triangles for s in t[3]], extend=False),
                   layer("Höhenlinien & Orthozentren", COLOR_ORTHOCENTER_LINE, COLOR_ORTHOCENTER_POINT,
                         [a for t in triangles for a in t[4]], centers),
                   layer("Bildmitte", COLOR_HORIZON_LINE, COLOR_HORIZON_LINE, [], [self.center])])


def pruefgruppen(m: Modell) -> list[dict]:
    """Alle Prüfgruppen (Rule 1, 2, 2-cross, 3/4, 5) - entfallene ohne Messwerte mit Begründung."""
    r = Regeln(m)
    r.rule1()
    r.rule2()
    r.rule3()
    r.rule5()
    r.verworfen()
    return r.gruppen
