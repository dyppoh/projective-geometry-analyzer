"""


Urteil aus den Abstandsmaßen

Grundwert = Messungenauigkeit im Bild = Median aller Rule-1-Einzelabstände (Linie <-> Fluchtpunkt ihrer
Richtung), je Winkelgrenze einer Prüfgruppe aus den Rule-1-Richtungen, deren Linien sich mindestens um diese
Grenze unterscheiden; erst ab GRUNDWERT_MIN_ABSTAENDE Abständen. Rule 1 selbst wird nicht bewertet.

Teilprüfung = Prüfgruppe * Kategorie. Je Kategorie in GRUNDWERT_REGELN ein Kriterium:

    median   Median der Abstände der Teilprüfung über Faktor × Grundwert -> Befund

Urteil:

    keine geprüfte Prüfgruppe (kein Grundwert bzw. keine Prüfgruppe)   -> nicht bewertbar
    mindestens ein Befund                                              -> unecht
    sonst                                                              -> echt (Annahme, da keine Befunde)
"""

from __future__ import annotations

import statistics

GRUNDWERT_REGELN = {        # Kategorie -> {Kriterium: Faktor * Grundwert}
    "rule2":        {"median": 2.13},
    "rule2cross":   {"median": 64.00},
    "rule3":        {"median": 10.04},
    "rule5cross":   {"median": 180.80},
    "rule5center":  {"median": 401.88},
}
GRUNDWERT_MIN_ABSTAENDE = 5     # Grundwert erst ab so vielen Rule-1-Einzelabständen
ECHT_TEXT = "da keine Befunde identifiziert werden konnten"

# Kategorie -> (Regel, was nicht eingehalten ist, Bezeichnung eines Elements)
KATEGORIEN = {
    "rule1": ("Rule 1", "Fluchtlinien laufen nicht in einem Fluchtpunkt zusammen.", "Linie"),
    "rule2": ("Rule 2", "Fluchtlinien einer Ebene laufen nicht in einem Fluchtpunkt zusammen.", "Linie"),
    "rule2cross": ("Rule 2", "Die Fluchtpunkte von nach Ebenen getrenntren Fluchtlinien fallen nicht zusammen.", "Fluchtpunkte"),
    "rule3": ("Rule 3/4", "Die Fluchtpunkte paralleler Ebenen liegen nicht auf einem gemeinsamen Horizont.",
              "Fluchtpunkt von"),
    "rule5cross": ("Rule 5", "Die Orthozentren von senkrechten Fluchtlinien fallen nicht zusammen", "Orthozentren"),
    "rule5center": ("Rule 5", "Das Orthozentrum von senkrechten Fluchtlinien liegt nicht nahe der Bildmitte",
                    "Orthozentrum"),
}


def _pct(v: float) -> str:
    return f"{v * 100:.1f}".replace(".", ",") + " %"


def _zahl(v: float) -> str:
    return f"{v:g}".replace(".", ",")


def _gw(v: float, grundwert: float) -> str:
    """Wert in % der Bildbreite und als Vielfaches des Grundwerts, z.B. "5,2 % (3,1 × Grundwert)"."""
    r = v / grundwert
    return f"{_pct(v)} ({(f'{r:.1f}' if r < 100 else f'{r:.0f}').replace('.', ',')} × Grundwert)"


def _grenze(k: dict) -> str:
    return f"der Grenze {_pct(k['threshold'])} ({_zahl(k['faktor'])} × Grundwert)"


def _grundwert(gruppen: list[dict], winkel: float) -> dict:
    r1 = [g["teile"]["rule1"] for g in gruppen if "rule1" in g["teile"] and g["angleSpread"] >= winkel]
    werte = [s["value"] for t in r1 for s in t["singles"]]
    value = statistics.median(werte) if len(werte) >= GRUNDWERT_MIN_ABSTAENDE else None
    return {"value": value, "n": len(werte), "richtungen": len(r1), "winkel": winkel}


def _grundwert_text(gw: dict) -> str:
    wo = f"{gw['richtungen']} Richtung{'' if gw['richtungen'] == 1 else 'en'} mit mindestens {_zahl(gw['winkel'])}°"
    if gw["value"] is None:
        return f"nicht bestimmbar - nur {gw['n']} Rule-1-Einzelabstände aus {wo} (nötig: {GRUNDWERT_MIN_ABSTAENDE})"
    return f"{_pct(gw['value'])} (Median von {gw['n']} Rule-1-Einzelabständen aus {wo})"


def _finding_text(f: dict) -> str:
    regel, was, _ = KATEGORIEN[f["category"]]
    teile = []
    for k in f["kriterien"]:
        if not k["befund"]:
            continue
        gw = k["threshold"] / k["faktor"]
        teile.append(f"Median der Abstände {_gw(k['value'], gw)} liegt über {_grenze(k)} "
                     f"({f['nSingles']} Einzelabstände)")
    return f"{regel} ({was}), {f['subject']}: {'; '.join(teile)}."


def bewerten(gruppen: list[dict]) -> tuple[dict, dict[str, str]]:
    """Prüfgruppen (regeln.py) -> (Urteil, {Prüfgruppe: {"status", "reason"}}). Status: befund | ok |
    grundwert (Rule 1, nicht bewertet) | ungeprueft (kein Grundwert) | entfallen (keine Messwerte)."""
    grundwerte: dict[float, dict] = {}
    regeln_je_winkel: dict[float, list[str]] = {}
    status = {g["id"]: ({"status": "entfallen", "reason": g.get("reason")} if not g["teile"] else
                        {"status": "grundwert", "reason": "Rule 1 liefert den Grundwert der Messungenauigkeit und "
                                                          "wird selbst nicht bewertet."} if g["rule"] == "1" else
                        {"status": "ungeprueft", "reason": None}) for g in gruppen}
    findings, kriterien_gesamt, ratios, geprueft = [], 0, [], set()
    for g in gruppen:
        for kat, t in g["teile"].items():
            cfg = GRUNDWERT_REGELN.get(kat)
            if not cfg:
                continue                        # Rule 1
            winkel = g["minAngle"]
            if winkel not in grundwerte:
                grundwerte[winkel] = _grundwert(gruppen, winkel)
            gw = grundwerte[winkel]
            regel = KATEGORIEN[kat][0]
            if regel not in regeln_je_winkel.setdefault(winkel, []):
                regeln_je_winkel[winkel].append(regel)
            if gw["value"] is None:              # kein Grundwert -> nicht prüfbar
                status[g["id"]]["reason"] = f"Nicht prüfbar: Grundwert {_grundwert_text(gw)}."
                continue
            geprueft.add(g["id"])
            kriterien = []
            for wert, faktor in cfg.items():
                grenze = faktor * gw["value"]
                kriterien.append({"wert": wert, "faktor": faktor, "threshold": grenze, "value": t[wert],
                                  "befund": t[wert] > grenze, "over": []})
            kriterien_gesamt += len(kriterien)
            ratios += [(k["value"] / k["threshold"], g, kat, k) for k in kriterien if k["threshold"]]
            n_befunde = sum(k["befund"] for k in kriterien)
            if n_befunde:
                findings.append({"group": g["id"], "rule": g["rule"], "category": kat, "subject": g["subject"],
                                 "title": KATEGORIEN[kat][1], "nSingles": len(t["singles"]), "points": n_befunde,
                                 "kriterien": kriterien})
                status[g["id"]] = {"status": "befund", "reason": None}
            elif status[g["id"]]["status"] != "befund":
                status[g["id"]] = {"status": "ok", "reason": None}
    findings.sort(key=lambda f: max(k["value"] / k["threshold"] for k in f["kriterien"] if k["befund"]),
                  reverse=True)

    zeilen = [f"Grundwert der Messungenauigkeit für {', '.join(regeln_je_winkel[w])}: {_grundwert_text(gw)}."
              for w, gw in grundwerte.items()]
    gw_anzeige = next((gw for gw in grundwerte.values() if gw["value"] is not None),
                      next(iter(grundwerte.values()), None))
    base = {"grundwert": gw_anzeige, "pruefgruppen": len(geprueft)}
    if not geprueft:
        reason = "kein Grundwert" if grundwerte else "keine Prüfgruppen"
        text = ("Bewertung nicht möglich: kein Grundwert aus Rule 1 bestimmbar." if grundwerte else
                "Bewertung nicht möglich: keine Prüfgruppe von Rule 2-5 auswertbar (zu wenige Linien bzw. "
                "Fluchtpunkte).")
        return ({**base, "decision": "nicht bewertbar", "reason": reason, "explanation": "\n".join([text, *zeilen]),
                 "befunde": 0, "kriterien": 0, "findings": []}, status)

    n_bef = sum(f["points"] for f in findings)
    if findings:
        decision = "unecht"
        top = findings[0]
        reason = (f"{n_bef} Befund{'' if n_bef == 1 else 'e'}, stärkster: {KATEGORIEN[top['category']][0]}, "
                  f"{top['subject']}")
        kopf = (f"Unecht: {n_bef} von {kriterien_gesamt} geprüften Kriterien mit Befund - diese Abstandsmaße sind "
                "im Verhältnis zur Messungenauigkeit (Grundwert) womöglich zu groß.")
    else:
        decision = "echt"
        reason = ECHT_TEXT
        regeln = list(dict.fromkeys(KATEGORIEN[k][0] for g in gruppen if g["id"] in geprueft for k in g["teile"]))
        kopf = (f"Annahme: echt, {ECHT_TEXT}. Geprüft wurden {kriterien_gesamt} Kriterien in {len(geprueft)} "
                f"Prüfgruppen ({', '.join(regeln)}).")
    zeilen = [kopf, *zeilen, *(f"- {_finding_text(f)}" for f in findings)]
    if decision == "echt" and ratios:
        r, g, kat, k = max(ratios, key=lambda x: x[0])
        zeilen.append(f"Größte Abweichung: {KATEGORIEN[kat][0]}, {g['subject']}, Median der Abstände "
                      f"{_gw(k['value'], k['threshold'] / k['faktor'])} = {_zahl(round(r, 2))} × {_grenze(k)}.")
    return ({**base, "decision": decision, "reason": reason, "explanation": "\n".join(zeilen),
             "befunde": n_bef, "kriterien": kriterien_gesamt, "findings": findings}, status)
