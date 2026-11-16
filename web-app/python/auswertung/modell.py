"""

Gruppenstruktur der Auswertung und der Sortieralgorithmus, der sie aus drei Beziehungslisten baut.

Gruppenstruktur (= Scene zwischen Webserver und Service, im Editor bearbeitet):

    richtungen   3D-Richtungen (Rule 1), je mit untergeordneten Ebenen = Linien der Richtung auf einer
                 Ebene(nschar) (Rule 2). Eine Linie darf in mehreren Ebenen stehen (Kante).
    horizonte    Ebenen-Gruppen (Rule 3/4): Ebenen auf derselben bzw. parallelen Ebenen - ihre Fluchtpunkte
                 liegen auf einem Horizont.
    senkrecht    Senkrecht-Gruppen (Rule 5): Ebenen aus drei zueinander senkrechten Richtungen.

Beziehungslisten (Ausgabe von TLC bzw. MLLM) -> Gruppenstruktur :
  1. Richtungen = Parallel-Gruppen (identische zusammengelegt), nach Größe sortiert.
  2. Ebenen einer Richtung = Richtung ∩ Ebenen-Gruppe, dazu der Rest ohne Ebene; gleiche Schnittmengen werden
     EINE Ebene.
  3. Ebenenscharen: Ebenen-Gruppen mit >= 2 gemeinsamen Richtungen sind parallel (jedes Mitglied mit jedem
     direkt, keine Verkettung). Horizont-Gruppe = die Ebenen aller Ebenen-Gruppen einer Schar.
  4. Senkrecht-Gruppe = die Ebenen der (höchstens drei) getroffenen Richtungen.
Sortiert wird nach Größe und kleinstem Linienindex - das Ergebnis hängt nicht von der Reihenfolge ab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .geometrie import Line


@dataclass
class Ebene:
    """Linien einer Richtung auf einer Ebene(nschar) (Rule 2)."""
    id: str
    name: str
    label: str                  # "<Richtung> · <Ebene>"
    line_ids: list[str]
    lines: list[Line]


@dataclass
class Richtung:
    """Eine 3D-Richtung, ggf. auf mehreren Ebenen (Rule 1)."""
    id: str
    name: str
    ebenen: list[Ebene]

    def line_refs(self) -> list[tuple[str, Line]]:
        """Jede Linie der Richtung einmal (eine Kante in zwei Ebenen zählt einfach)."""
        return list({lid: (lid, l) for e in self.ebenen for lid, l in zip(e.line_ids, e.lines)}.values())


@dataclass
class Gruppe:
    """Ebenen-Gruppe (Rule 3/4) bzw. Senkrecht-Gruppe (Rule 5) - Verweise auf Ebenen-IDs."""
    id: str
    name: str
    ebenen: list[str]


@dataclass
class Modell:
    width: float
    height: float
    richtungen: list[Richtung] = field(default_factory=list)
    horizonte: list[Gruppe] = field(default_factory=list)
    senkrecht: list[Gruppe] = field(default_factory=list)
    # Gruppen, die nicht in die Prüfung eingehen: {"kind", "name", "reason", "lines"}
    verworfen: list[dict] = field(default_factory=list)

    @property
    def ebenen(self) -> list[Ebene]:
        return [e for r in self.richtungen for e in r.ebenen]


# ---------------------------------------------------------------------------
# Gruppenstruktur <-> Modell
# ---------------------------------------------------------------------------

def from_scene(scene: dict) -> Modell:
    """Gruppenstruktur aus dem Editor -> Modell. Leere Ebenen/Richtungen/Gruppen gehen nicht in die Prüfung ein
    (verworfen), Verweise auf unbekannte Linien bzw. Ebenen werden ignoriert."""
    by_id = {str(l["id"]): Line.from_dict(l) for l in scene["lines"]}
    verworfen: list[dict] = []
    richtungen, seen = [], set()

    def check_id(i: str) -> str:
        if i in seen:
            raise ValueError(f"Doppelte ID '{i}'.")
        seen.add(i)
        return i

    for k, r in enumerate(scene.get("richtungen", [])):
        name = r.get("name") or f"Richtung {k + 1}"
        ebenen = []
        for j, e in enumerate(r.get("ebenen", [])):
            ename = e.get("name") or f"Ebene {j + 1}"
            ids = list(dict.fromkeys(str(i) for i in e.get("lineIds", []) if str(i) in by_id))
            if not ids:
                verworfen.append({"kind": "ebene", "name": f"{name} · {ename}", "lines": [],
                                  "reason": "Die Ebene hat keine Linien."})
                continue
            ebenen.append(Ebene(check_id(str(e["id"])), ename, f"{name} · {ename}", ids, [by_id[i] for i in ids]))
        if not ebenen:
            verworfen.append({"kind": "richtung", "name": name, "lines": [], "reason": "Die Richtung hat keine Linien."})
            continue
        richtungen.append(Richtung(check_id(str(r["id"])), name, ebenen))

    known = {e.id for r in richtungen for e in r.ebenen}
    lines_of = {e.id: e.lines for r in richtungen for e in r.ebenen}

    def gruppen(key: str, kind: str, item: str) -> list[Gruppe]:
        out = []
        for k, g in enumerate(scene.get(key, [])):
            name = g.get("name") or f"{item} {k + 1}"
            ids = list(dict.fromkeys(str(i) for i in g.get("ebenen", []) if str(i) in known))
            if ids:
                out.append(Gruppe(str(g.get("id") or f"{kind}{k + 1}"), name, ids))
            else:
                verworfen.append({"kind": kind, "name": name, "lines": [],
                                  "reason": "Die Gruppe enthält keine Ebene mit Linien."})
        return out

    return Modell(float(scene["imageWidth"]), float(scene["imageHeight"]), richtungen,
                  gruppen("horizonte", "horizont", "Ebenen-Gruppe"), gruppen("senkrecht", "senkrecht", "Senkrecht"),
                  verworfen)


def to_scene(m: Modell, lines: list[dict]) -> dict:
    """Modell -> Gruppenstruktur für den Editor (alle Linien, auch nicht zugeordnete)."""
    return {"imageWidth": m.width, "imageHeight": m.height, "lines": lines,
            "richtungen": [{"id": r.id, "name": r.name,
                            "ebenen": [{"id": e.id, "name": e.name, "lineIds": e.line_ids} for e in r.ebenen]}
                           for r in m.richtungen],
            "horizonte": [{"id": g.id, "name": g.name, "ebenen": g.ebenen} for g in m.horizonte],
            "senkrecht": [{"id": g.id, "name": g.name, "ebenen": g.ebenen} for g in m.senkrecht]}


# ---------------------------------------------------------------------------
# Sortieralgorithmus: Beziehungslisten -> Modell
# ---------------------------------------------------------------------------

def _key(idx: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(idx)))


def _order(sets: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    return sorted(sets, key=lambda s: (-len(s), s))


def build(width: float, height: float, lines: list[dict], relations: dict) -> Modell:
    """Linien + Beziehungslisten {"parallel"|"plane"|"perpendicular": [([Linienindex, ...], Name), ...]} -> Modell."""
    objs = [Line.from_dict(l) for l in lines]
    lid = [str(l["id"]) for l in lines]
    n = len(lines)

    def of(kind: str) -> tuple[list[tuple[int, ...]], list[str]]:
        groups = relations.get(kind) or []
        return [_key(i for i in g if 0 <= i < n) for g, _ in groups], [name or "" for _, name in groups]

    par_in, par_names = of("parallel")
    planes_in, plane_names = of("plane")
    perp_in, perp_names = of("perpendicular")
    planes = [set(p) for p in planes_in]

    # 1. Richtungen
    directions = _order({p for p in par_in if p})
    gid = {p: f"g{k + 1}" for k, p in enumerate(directions)}

    # 2. Ebenen der Richtungen
    on_any_plane = set().union(*planes) if planes else set()
    sub_id: dict[tuple[int, int | None], str] = {}       # (Richtung k, Ebenen-Gruppe e) -> Ebenen-ID
    richtungen = []
    for k, p in enumerate(directions):
        pset = set(p)
        parts: dict[tuple[int, ...], list[int | None]] = {}
        for e, plane in enumerate(planes):
            if pset & plane:
                parts.setdefault(_key(pset & plane), []).append(e)
        rest = _key(pset - on_any_plane)
        if rest:
            parts.setdefault(rest, []).append(None)
        name = next((par_names[j] for j, q in enumerate(par_in) if q == p and par_names[j]), f"Richtung {k + 1}")
        ebenen = []
        for j, part in enumerate(_order(parts), 1):
            pid = f"{gid[p]}.{j}"
            for e in parts[part]:
                sub_id[(k, e)] = pid
            names = [plane_names[e] or f"Ebene {e + 1}" for e in parts[part] if e is not None]
            ename = ", ".join(names) if names else "ohne Ebene"
            ebenen.append(Ebene(pid, ename, f"{name} · {ename}", [lid[i] for i in part], [objs[i] for i in part]))
        richtungen.append(Richtung(gid[p], name, ebenen))

    # 3. Ebenenscharen -> Ebenen-Gruppen (Horizont)
    dirs_of = [{k for k in range(len(directions)) if (k, e) in sub_id} for e in range(len(planes))]
    found: list[list[int]] = []
    for e in sorted(range(len(planes)), key=lambda e: (-len(dirs_of[e]), sorted(planes[e]))):
        for members in found:
            if all(len(dirs_of[e] & dirs_of[m]) >= 2 for m in members):
                members.append(e)
                break
        else:
            found.append([e])

    def pid_key(pid: str) -> tuple[int, ...]:
        return tuple(int(x) for x in pid[1:].split("."))

    horizonte: dict[tuple[str, ...], str] = {}
    for members in sorted(found, key=min):
        ids = tuple(sorted({sub_id[(k, e)] for k in range(len(directions)) for e in members if (k, e) in sub_id},
                           key=pid_key))
        if ids and ids not in horizonte:
            horizonte[ids] = ", ".join(plane_names[e] or f"Ebene {e + 1}" for e in sorted(members))

    # 4. Senkrecht-Gruppen: je getroffener Richtung die ganz enthaltenen Ebenen, sonst die berührten
    subs_of = {k: sorted({pid for (kk, _e), pid in sub_id.items() if kk == k}, key=pid_key)
               for k in range(len(directions))}
    index = {i: k for k, i in enumerate(lid)}
    sub_lines = {e.id: {index[i] for i in e.line_ids} for r in richtungen for e in r.ebenen}

    def chosen(k: int, sset: set[int]) -> list[str]:
        whole = [pid for pid in subs_of[k] if sub_lines[pid] <= sset]
        return whole or [pid for pid in subs_of[k] if sub_lines[pid] & sset]

    senkrecht: dict[tuple[str, ...], str] = {}
    for j, s in enumerate(perp_in):
        sset = set(s)
        hits = sorted(((len(sset & set(p)), k) for k, p in enumerate(directions) if sset & set(p)),
                      key=lambda h: (-h[0], h[1]))
        ids = tuple(pid for _, k in sorted(hits[:3], key=lambda h: h[1]) for pid in chosen(k, sset))
        if ids and ids not in senkrecht:
            senkrecht[ids] = perp_names[j]

    def gruppen(d: dict[tuple[str, ...], str], prefix: str, item: str) -> list[Gruppe]:
        return [Gruppe(f"{prefix}{k}", name or f"{item} {k}", list(ids))
                for k, (ids, name) in enumerate(sorted(d.items()), 1)]

    return Modell(float(width), float(height), richtungen, gruppen(horizonte, "h", "Ebenen-Gruppe"),
                  gruppen(senkrecht, "s", "Senkrecht"))
