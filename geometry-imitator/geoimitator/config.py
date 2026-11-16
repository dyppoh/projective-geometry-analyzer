"""Einstellungen aller Stufen"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# Modellordner (Generatoren und Linienerkennung), siehe models/README.md
MODELS_DIR = Path(os.environ.get("GEOIMITATOR_MODELS", Path(__file__).resolve().parents[1] / "models"))


@dataclass
class SceneConfig:
    # Szenentypen und ihre Gewichte
    types: dict = field(default_factory=lambda: {"room": 0.35, "facade": 0.25, "street": 0.25, "objects": 0.15})
    # Bildgrößen (Breite, Höhe) der Szene - werden zufällig gewählt. Vielfache von 64.
    sizes: list = field(default_factory=lambda: [[768, 512], [512, 768], [640, 640], [768, 576]])
    min_segment_px: float = 10.0          # kürzere sichtbare Segmente werden verworfen
    min_segments: int = 25                # Szene neu würfeln, wenn weniger sichtbar
    min_groups: int = 2                   # ... oder weniger Richtungen mit >= 2 Segmenten
    max_segments: int = 900               # ... oder zu unruhig
    max_directions: int = 16              # ... oder mehr sichtbare 3D-Richtungen (= Ausgabegruppen der TLC-Modelle)
    max_attempts: int = 25
    # Basis-Seed der Szenen. None -> beim Anlegen des Runs zufällig gezogen und in config.json
    # gespeichert (sonst wären alle Runs identisch). Ältere Runs ohne Eintrag: 0.
    seed: int | None = None
    line_thickness: list = field(default_factory=lambda: [1, 2])   # zufällig pro Szene
    # Gauß-Rauschen auf ALLEN Endpunkten (σ in px der Szenengröße, pro Szene gezogen; [0, 0] = aus).
    # Die unverrauschte Lage bleibt in p1Exact/p2Exact.
    noise_sigma_px: list = field(default_factory=lambda: [0.3, 2.0])
    # Gestörte Projektion (siehe perturb.py)
    perturbation: dict = field(default_factory=lambda: {
        "probability": 0.25, "maxDirections": 2, "minLines": 3, "modes": ["line_jitter", "camera_split"],
        "jitterSigmaDeg": [0.6, 2.0], "cameraSplitDeg": [1.0, 3.0], "minLengthPx": 10.0})


@dataclass
class GenerationConfig:
    per_scene: int = 2                    # Generierungen pro Szene
    generators: list = field(default_factory=lambda: ["sd15-realvis", "sd15-dreamshaper", "sd15-base",
                                                      "sdxl-juggernaut", "sdxl-t2i-juggernaut"])
    device: str | None = None             # None -> cuda / mps / cpu
    dtype: str = "auto"                   # "auto" | "float16" | "float32"
    cpu_offload: bool = False             # bei wenig VRAM
    steps: int | None = None              # None -> Default des Generators
    seed: int | None = None               # None -> beim Anlegen des Runs zufällig (siehe pipeline.init_run)
    # Anteil Prompts ohne Geometrie-Motiv bzw. mit Render-/Kunststil (siehe prompts.py)
    prompt: dict = field(default_factory=lambda: {"offTopic": 0.04, "nonPhoto": 0.08})


@dataclass
class MatchConfig:
    # "all" (alle aus `detectors` mit Gewicht > 0, das beste angenommene wird sample.json) |
    # "random" (eins nach `detectors`) | deeplsd | mlsd | classical | opencv-lsd
    detector: str = "all"
    detectors: dict = field(default_factory=lambda: {"deeplsd": 1.0, "mlsd": 1.0, "classical": 1.0})
    deeplsd_max_dimension: int = 1024
    min_det_length_px: float = 8.0
    # Linien-Zuordnung (Schwellen relativ zur Bildbreite bzw. in Grad)
    max_angle_deg: float = 3.0
    max_dist_ratio: float = 0.01         # max. Abstand der Endpunkte zur Original-Geraden (Anteil der Bildbreite)
    min_dist_px: float = 0
    min_endpoint_hits: int = 2            # Endpunkte (beider Linien), deren Lot die andere Strecke trifft
    # Annahme einer Generierung: Anteil der Originalsegmente mit mindestens einer zugeordneten Linie > min_recall_lines
    min_recall_lines: float = 0.2
    segment_coverage: float = 0.5         # nur Kennzahl recallSegments: Anteil der Segmente mit Abdeckung >= diesem Wert
    # Bestes Verfahren bei detector "all" (immer erst angenommen vor abgelehnt):
    # "recall" = recallLines, dann precision | "precision" = precision, dann recallLines
    best_by: str = "recall"


@dataclass
class Config:
    scene: SceneConfig = field(default_factory=SceneConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    match: MatchConfig = field(default_factory=MatchConfig)

    def to_dict(self) -> dict:
        return _camel_keys(asdict(self))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, *paths: str | Path | None) -> "Config":
        """Defaults, dann jede vorhandene Datei der Reihe nach darüber (spätere gewinnen)."""
        cfg = cls()
        for path in paths:
            if path and Path(path).exists():
                update(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
        return cfg


# entfernte Einstellungen, die in config.json älterer Runs noch stehen
REMOVED = {"min_overlap", "ambiguity_margin", "min_recall_length", "min_recall_segments", "min_matched_lines",
           "min_matched_groups"}


def update(obj, d: dict) -> None:
    known = {f.name: f for f in fields(obj)}
    for k, v in d.items():
        name = _snake(k)
        if name in REMOVED:
            continue
        if name not in known:
            raise ValueError(f"Unbekannte Einstellung '{k}' in {type(obj).__name__}")
        cur = getattr(obj, name)
        if is_dataclass(cur):
            update(cur, v)
        elif isinstance(cur, dict) and isinstance(v, dict):
            cur.update(v)
        else:
            setattr(obj, name, v)


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


def _snake(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in name)


def _camel_keys(d):
    if isinstance(d, dict):
        # verschachtelte freie dicts (types, perturbation) behalten ihre Schlüssel
        return {_camel(k): (_camel_keys(v) if isinstance(v, dict) and k in ("scene", "generation", "match") else v)
                for k, v in d.items()}
    return d
