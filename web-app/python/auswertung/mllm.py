"""


Gruppierung durch ein MLLM über HTTP 

    POST /mllm                     {"image": data:-URL} -> {"token", ...}: neue Aufgabe
    GET  /group/<token>            Aufgabenseite (HTML): Anleitung, Bilder, Linientabelle, wie abgeben
    GET  /group/<token>/bild.jpg   Bild unverändert          /linien.jpg   mit nummerierten Linien
    GET  /group/<token>/ausschnitt-<1-4>.jpg   Viertel (oben links, oben rechts, unten links, unten rechts)
    GET  /group/<token>/ausschnitt.jpg?x=&y=&w=&h=   beliebiger Ausschnitt (Pixel des Originalbildes)
    POST /group/<token>            Gruppierung als JSON -> Rückmeldung zur Struktur (erneut abgeben überschreibt)
    GET  /mllm/<token>             Stand für die Weboberfläche (offen | abgegeben + Scene)


"""

from __future__ import annotations

import html
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .annotieren import DETECTORS, detect, line_dicts, load_image
from .modell import build, to_scene

MAX_LINES = 100              # so viele (die längsten) Linien werden nummeriert
RENDER_SIZE = 1024          # lange Seite der Bilder für das MLLM
TTL = 3600
PALETTE_BGR = [(0, 0, 230), (0, 160, 0), (230, 80, 0), (0, 170, 230), (200, 0, 200), (0, 120, 120),
               (120, 0, 200), (40, 40, 40), (0, 90, 255), (200, 120, 0)]
VIERTEL = {1: "oben links", 2: "oben rechts", 3: "unten links", 4: "unten rechts"}

GROUP_RULES = """\
Gib drei Arten von Gruppen an, jeweils als Listen von Linien-Nummern. Alle drei sind wichtig.
1. parallel: Linien, die in der dargestellten Szene in dieselbe 3D-Richtung verlaufen, z.B. Kanten von Wänden, Türen, Möbeln oder weiteren Objekten.
2. ebenen: Linien, die auf derselben Fläche oder auf zueinander parallelen Flächen liegen, z.B. die Linien auf eines Fußbodens können mit denen einer Tischplatte und der Decke auf parallelen Ebenen liegen. Zwei Gegenüberliegende Wände möglicherweise auch.
   Gib jede erkennbare Fläche an, auch wenn sie wenige Linien darauf liegen. Eine Linie kann auch wie eine Kante an zwei Flächen liegen.
3. senkrecht: Linien, die in der Szene wie Koordinatenachsen jeweils senkrecht zueinander stehen. Zum Beispiel eine Raumecke oder der Eckpunkt einer Box

Linien aus ebenen und senkrecht sollten auch in einer parallel-Gruppe stehen.

Wichtig
- Entscheide nach der Szene: Rechne keine Schnittpunkte oder Winkel
  nach und gruppiere nicht danach, ob Linien im Bild zusammenlaufen oder ähnlich ausgerichtet sind.
  Die Gruppen werden anschließend geometrisch geprüft und das funktioniert nur, wenn deine Gruppierung
  unabhängig davon entsteht.
- Es geht also um Geometrie und Struktur, wie gerade Kanten von Gegenständen oder Architektur. Linien in Textur, Muster, Falten, Schatten, Pflanzen, Haare, Schrift und gekrümmte Konturen sind wahrscheinlich nicht relevant sein.
- Unsichere Linien lieber weglassen als raten. Nicht jede Linie muss vorkommen.
- Gib den Gruppen bitte einen kurzen, passenden Namen wie z.B. "Box vorne", "Fußboden".
- Nach dem Abgeben bekommst du Hinweise zur Struktur wie z.B. fehlendes senkrecht-Tripel. Prüfe sie und gib bei Bedarf gerne erneut ab (überschreibt).
"""


@dataclass
class Aufgabe:
    token: str
    image: np.ndarray
    lines: np.ndarray                       # (N, 4) Pixel des Originalbildes
    shown: list[int]                        # nummerierte Linien (Index, Nummer = Index + 1)
    detector: str = "mlsd"                  # Linienerkennung: mlsd | hough
    created: float = field(default_factory=time.time)
    renders: dict[str, bytes] = field(default_factory=dict)
    abgabe: dict | None = None              # letzte Gruppierung
    version: int = 0


_lock = threading.Lock()
_aufgaben: dict[str, Aufgabe] = {}


# ---------------------------------------------------------------------------
# Aufgaben
# ---------------------------------------------------------------------------

def create(image: str, detector: str = "mlsd") -> dict:
    img = load_image(image)
    lines = detect(img, detector)
    if not len(lines):
        raise ValueError(f"{DETECTORS[detector]} hat keine Linien erkannt.")
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    shown = sorted(np.argsort(-lengths, kind="stable")[:MAX_LINES].tolist())
    token = secrets.token_urlsafe(9)
    with _lock:
        now = time.time()
        for t in [t for t, a in _aufgaben.items() if now - a.created > TTL]:
            del _aufgaben[t]
        _aufgaben[token] = Aufgabe(token, img, lines, shown, detector=detector)
    return {"token": token, "lines": len(lines), "shown": len(shown), "detector": DETECTORS[detector]}


def get(token: str) -> Aufgabe:
    with _lock:
        a = _aufgaben.get(token)
    if a is None:
        raise LookupError("Unbekannte oder abgelaufene Aufgabe.")
    return a


def status(token: str) -> dict:
    a = get(token)
    if a.abgabe is None:
        return {"status": "offen", "version": 0}
    return {"status": "abgegeben", "version": a.version, **a.abgabe}


# ---------------------------------------------------------------------------
# Darstellung
# ---------------------------------------------------------------------------

def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Bild konnte nicht kodiert werden.")
    return buf.tobytes()


def _draw_numbered(img: np.ndarray, lines: np.ndarray, shown: list[int], x0: float, y0: float, s: float) -> None:
    def pt(x, y):
        return int(round((x - x0) * s)), int(round((y - y0) * s))

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    labels = []
    for k, i in enumerate(shown):
        color = PALETTE_BGR[k % len(PALETTE_BGR)]
        a, b = pt(*lines[i][:2]), pt(*lines[i][2:4])
        cv2.line(img, a, b, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        labels.append((i, color, a, b))
    for i, color, a, b in labels:                      # Nummern über allen Linien, etwas neben der Linie
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1
        tx, ty = int(mx - dy / n * 10), int(my + dx / n * 10)
        text = str(i + 1)
        (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
        if not (-tw < tx < img.shape[1] and 0 < ty < img.shape[0] + th):
            continue
        cv2.rectangle(img, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 3), (255, 255, 255), -1)
        cv2.rectangle(img, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 3), color, 1)
        cv2.putText(img, text, (tx, ty), font, fs, (0, 0, 0), 1, cv2.LINE_AA)


def render(a: Aufgabe, numbered: bool, crop: tuple[float, float, float, float] | None = None) -> bytes:
    """JPEG: Bild(ausschnitt) auf lange Seite RENDER_SIZE skaliert, optional mit nummerierten Linien."""
    H, W = a.image.shape[:2]
    x0, y0, cw, ch = crop if crop else (0, 0, W, H)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(W, int(math.ceil(x0 + cw))), min(H, int(math.ceil(y0 + ch)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("Ausschnitt zu klein oder außerhalb des Bildes.")
    s = RENDER_SIZE / max(x1 - x0, y1 - y0)
    img = cv2.resize(a.image[y0:y1, x0:x1], (round((x1 - x0) * s), round((y1 - y0) * s)),
                     interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    if numbered:
        _draw_numbered(img, a.lines, a.shown, x0, y0, s)
    return _encode(img)


def viertel(a: Aufgabe, k: int) -> tuple[float, float, float, float]:
    """Viertel k (1-4) mit Überlappung: je 60 % von Breite und Höhe."""
    if k not in VIERTEL:
        raise LookupError("Ausschnitt 1-4.")
    H, W = a.image.shape[:2]
    return ((k - 1) % 2 * 0.4 * W, (k - 1) // 2 * 0.4 * H, 0.6 * W, 0.6 * H)


def image(token: str, name: str, query: dict) -> bytes:
    a = get(token)
    if name == "ausschnitt.jpg":
        try:
            crop = tuple(float(query[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, ValueError):
            raise ValueError("Ausschnitt: x, y, w, h in Pixeln des Originalbildes angeben.") from None
        return render(a, True, crop)
    if name not in a.renders:
        if name == "bild.jpg":
            a.renders[name] = render(a, False)
        elif name == "linien.jpg":
            a.renders[name] = render(a, True)
        elif name.startswith("ausschnitt-") and name.endswith(".jpg") and name[11:-4].isdigit():
            a.renders[name] = render(a, True, viertel(a, int(name[11:-4])))
        else:
            raise LookupError(f"Unbekanntes Bild '{name}'.")
    return a.renders[name]


# ---------------------------------------------------------------------------
# Aufgabenseite
# ---------------------------------------------------------------------------

def page(token: str, origin: str) -> str:
    a = get(token)
    H, W = a.image.shape[:2]
    url = html.escape(f"{origin}/group/{token}")      # origin kommt aus einem Header
    rows = "\n".join(f"{i + 1} | ({l[0]:.0f}, {l[1]:.0f}) | ({l[2]:.0f}, {l[3]:.0f}) | "
                     f"{math.hypot(l[2] - l[0], l[3] - l[1]):.0f}" for i, l in ((i, a.lines[i]) for i in a.shown))
    beispiel = json.dumps({"parallel": [[3, 8, 15], [1, 4, 22]], "richtungsnamen": ["vertikal", "Tiefe"],
                           "ebenen": [[4, 22, 31]], "ebenennamen": ["Fußboden"], "senkrecht": [[3, 4, 9]],
                           "modell": "dein Modellname", "notiz": "wo du unsicher warst"}, ensure_ascii=False)
    viertel_links = "\n".join(f'<li><a href="{url}/ausschnitt-{k}.jpg">{url}/ausschnitt-{k}.jpg</a> ({v})</li>'
                              for k, v in VIERTEL.items())
    e = html.escape
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linien gruppieren</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #14171c; }}
h1, h2 {{ letter-spacing: .02em; }} h2 {{ border-bottom: 1px solid #14171c; padding-bottom: .2rem; margin-top: 2rem; }}
pre {{ background: #f2f1ec; padding: .8rem; overflow-x: auto; }} img {{ max-width: 100%; border: 1px solid #ccc; }}
textarea {{ width: 100%; height: 12rem; font: 13px monospace; }} button {{ padding: .5rem 1.2rem; }}
</style></head><body>
<h1>Aufgabe: Liniensegmente gruppieren</h1>
<p>Moin liebes MLLM, du siehst gleich ein Bild mit den Maßen {W}x{H} px. Die Linienerkennung ({DETECTORS[a.detector]}) hat {len(a.lines)} Linien gefunden, die {len(a.shown)} längsten
sind im Bild nummeriert. Die Farben dienen nur zur Unterscheidung. Die Linientabelle unten nennt die Endpunkte in
Pixeln des Originalbildes.</p>
<p>Ordne die nummerierten Liniensegmente nach ihrer räumlichen 3D-Eigenschaften in der Szene in gemeinsame Gruppen. 
Sieh dir dazu beide Bilder genau an und zoome bei dichten Bereichen in die Ausschnitte.</p>
<pre>{e(GROUP_RULES)}</pre>

<h2>Bilder</h2>
<ul>
<li>Bild unverändert: <a href="{url}/bild.jpg">{url}/bild.jpg</a></li>
<li>Bild mit nummerierten Linien: <a href="{url}/linien.jpg">{url}/linien.jpg</a></li>
<li>Vergrößerte Viertel mit Nummern:<ul>
{viertel_links}
</ul></li>
<li>Beliebiger Ausschnitt mit Nummern: <code>{url}/ausschnitt.jpg?x=…&amp;y=…&amp;w=…&amp;h=…</code>
(Pixel des Originalbildes)</li>
</ul>
<p><img src="{url}/bild.jpg" alt="Bild unverändert"></p>
<p><img src="{url}/linien.jpg" alt="Bild mit nummerierten Linien"></p>

<h2>Linientabelle</h2>
<pre>Nr | Endpunkt 1 | Endpunkt 2 | Länge (px)
{rows}</pre>

<h2>Abgeben</h2>
<p>Schicke die Gruppierung bitte als JSON per <b>POST</b> an <code>{url}</code> (Content-Type: application/json).
Gruppen sind Listen von Linien-Nummern wie im Bild; Namen in derselben Reihenfolge wie die Gruppen:</p>
<pre>{e(beispiel)}</pre>
<pre>curl -X POST {url} -H "Content-Type: application/json" -d '{e(beispiel)}'</pre>
<p>Die Antwort enthält Hinweise zur Struktur. Erneut abgeben überschreibt die vorige Gruppierung.
Ohne HTTP-POST: JSON hier einfügen und abschicken.</p>
<form id="abgabe"><textarea name="daten" placeholder='{e(beispiel)}'></textarea>
<p><button type="submit">Gruppierung abgeben</button></p></form>
<pre id="antwort"></pre>
<script>
document.getElementById("abgabe").addEventListener("submit", async (ev) => {{
  ev.preventDefault();
  const out = document.getElementById("antwort");
  try {{
    const res = await fetch({json.dumps(url)}, {{ method: "POST", headers: {{ "Content-Type": "application/json" }},
                                          body: ev.target.daten.value }});
    out.textContent = JSON.stringify(await res.json(), null, 1);
  }} catch (e) {{ out.textContent = String(e); }}
}});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Abgabe
# ---------------------------------------------------------------------------

def _groups(value, names) -> tuple[list[list[int]], list[str]]:
    """Gruppen als [[1, 2], ...] oder [{"name": ..., "linien": [...]}, ...] -> (Gruppen, Namen)."""
    groups, out_names = [], list(names or [])
    for k, g in enumerate(value or []):
        if isinstance(g, dict):
            if g.get("name"):
                out_names += [""] * (k + 1 - len(out_names))
                out_names[k] = str(g["name"])
            g = g.get("linien", g.get("lines", []))
        if not isinstance(g, list):
            raise ValueError("Gruppen sind Listen von Linien-Nummern.")
        groups.append(g)
    return groups, out_names


def submit(token: str, body: dict) -> dict:
    a = get(token)
    n = len(a.lines)
    shown = set(a.shown)
    problems: list[str] = []

    def pick(*keys):
        return next((body[k] for k in keys if body.get(k) is not None), None)

    parallel, par_names = _groups(pick("parallel"), pick("richtungsnamen", "parallelNames"))
    planes, plane_names = _groups(pick("ebenen", "planes"), pick("ebenennamen", "planeNames"))
    perpendicular, _ = _groups(pick("senkrecht", "perpendicular"), None)

    def convert(groups, name) -> list[list[int]]:
        out = []
        for j, g in enumerate(groups):
            idx = []
            for nr in g:
                try:
                    i = int(nr) - 1
                except (TypeError, ValueError):
                    problems.append(f"{name} {j + 1}: '{nr}' ist keine Linien-Nummer.")
                    continue
                if not 0 <= i < n:
                    problems.append(f"{name} {j + 1}: Linie {nr} gibt es nicht.")
                    continue
                if i not in shown:
                    problems.append(f"{name} {j + 1}: Linie {nr} war nicht nummeriert (trotzdem übernommen).")
                idx.append(i)
            out.append(sorted(set(idx)))
        return out

    rel = {"parallel": convert(parallel, "parallel"), "plane": convert(planes, "ebenen"),
           "perpendicular": convert(perpendicular, "senkrecht")}
    if not any(rel["parallel"]):
        raise ValueError("Keine gültige parallel-Gruppe angegeben.")
    names = {"parallel": par_names, "plane": plane_names, "perpendicular": []}
    relations = {kind: [(g, names[kind][k] if k < len(names[kind]) else "") for k, g in enumerate(gs) if g]
                 for kind, gs in rel.items()}

    structure = []                                    # Rückmeldung zur Struktur, keine Abstände
    in_parallel = {i for g in rel["parallel"] for i in g}
    for name, gs in (("ebenen", rel["plane"]), ("senkrecht", rel["perpendicular"])):
        for j, g in enumerate(gs):
            missing = [i + 1 for i in g if i not in in_parallel]
            if missing:
                structure.append(f"{name} {j + 1}: Linie(n) {missing} stehen in keiner parallel-Gruppe und werden "
                                 "dafür nicht verwendet - wenn möglich einer Richtung zuordnen.")
    if not any(rel["plane"]):
        structure.append("Keine Ebenen angegeben. Ebenen teilen die Richtungen nach Flächen auf (z.B. die Tiefe "
                         "auf Boden und auf Decke getrennt) - jede erkennbare Fläche angeben, auch mit wenigen "
                         "Linien, und erneut abgeben.")
    if sum(map(bool, rel["parallel"])) >= 3 and not any(rel["perpendicular"]):
        structure.append("Keine senkrecht-Gruppe angegeben. Wenn in der Szene Linien in drei paarweise senkrechten "
                         "Richtungen verlaufen (z.B. an einer Raum- oder Kastenecke: vertikal, Breite, Tiefe), "
                         "diese unter senkrecht angeben und erneut abgeben.")
    H, W = a.image.shape[:2]
    ld = line_dicts(a.lines)
    m = build(W, H, ld, relations)                   # Sortieralgorithmus -> Gruppenstruktur
    scene = to_scene(m, ld)
    hinweise = problems + structure
    with _lock:
        a.version += 1
        a.abgabe = {"modell": str(pick("modell", "model") or ""), "notiz": str(pick("notiz", "notes") or ""),
                    "hinweise": hinweise, "abgegeben": time.strftime("%H:%M:%S"), "scene": scene}
    return {"abgegeben": True, "version": a.version, "richtungen": len(m.richtungen),
            "ebenen": len(m.ebenen), "ebenenGruppen": len(m.horizonte), "senkrechtGruppen": len(m.senkrecht),
            "zugeordneteLinien": len(in_parallel), "hinweise": hinweise}
