"""
    Auswertung eines Bildes nach den Regeln der projektiven Geometrie

    annotieren.annotate(bild)   Bild -> Linien + Gruppen (M-LSD + TLC)
    auswerten(scene)            Linien + Gruppen -> Abstandsmaße (regeln.py) -> Urteil (bewertung.py)
"""

from __future__ import annotations

from .bewertung import bewerten
from .modell import from_scene
from .regeln import pruefgruppen


def auswerten(scene: dict) -> dict:
    """Gruppenstruktur (imageWidth, imageHeight, lines, richtungen, horizonte, senkrecht) -> + verdict + checks."""
    gruppen = pruefgruppen(from_scene(scene))
    verdict, status = bewerten(gruppen)
    checks = [{"id": g["id"], "rule": g["rule"], "title": g["title"], "subject": g["subject"], **status[g["id"]],
               "layers": g["layers"]} for g in gruppen]
    return {**scene, "verdict": verdict, "checks": checks}
