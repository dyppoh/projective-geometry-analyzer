"""
    Befehle

    python -m geoimitator scenes   --run runs/demo --count 100
    python -m geoimitator generate --run runs/demo --per-scene 2 [--generators sd15-realvis,sdxl-canny-depth]
    python -m geoimitator detect   --run runs/demo [--detector all|random|deeplsd|mlsd|classical|opencv-lsd] [--best-by recall|precision] [--rematch]
    python -m geoimitator all      --run runs/demo --count 20
    python -m geoimitator index    --run runs/demo
    python -m geoimitator negatives --run runs/neg --count 500   
    python -m geoimitator preview  --type street --seed 3 -o preview.png
    python -m geoimitator generators
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Umlaute in der Windows-Konsole / in Log-Dateien
        sys.stdout.reconfigure(encoding="utf-8")
    ap =argparse.ArgumentParser(prog="geoimitator", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run=True):
        if run:
            p.add_argument("--run", type=Path, required=True, help="Run-Ordner")
        p.add_argument("--config", type=Path, help="JSON mit abweichenden Einstellungen")

    p = sub.add_parser("scenes", help="3D-Szenen erzeugen + projizieren")
    common(p)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, help="Basis-Seed der Szenen (Default: scene.seed aus config.json)")
    p.add_argument("--type", choices=["room", "facade", "street", "objects"])

    p = sub.add_parser("generate", help="Bilder mit ControlNet generieren")
    common(p)
    p.add_argument("--per-scene", type=int)
    p.add_argument("--generators", help="Komma-getrennt, z.B. sd15-realvis,sd15-dreamshaper (mock = ohne Modell)")
    p.add_argument("--generators-file", type=Path)
    p.add_argument("--device")
    p.add_argument("--steps", type=int)
    p.add_argument("--limit", type=int, help="höchstens so viele neue Bilder")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("detect", help="DeepLSD + Zuordnung + Labels")
    common(p)
    p.add_argument("--detector", choices=["all", "random", "deeplsd", "mlsd", "classical", "opencv-lsd"])
    p.add_argument("--device")
    p.add_argument("--best-by", choices=["recall", "precision"],
                   help="bestes Verfahren bei --detector all: erst recallLines oder erst precision")
    p.add_argument("--rematch", action="store_true", help="nur neu zuordnen (gespeicherte Detektionen)")
    p.add_argument("--force", action="store_true", help="Detektor neu laufen lassen")

    p = sub.add_parser("all", help="scenes + generate + detect (erneuter Aufruf setzt fort)")
    common(p)
    p.add_argument("--count", type=int, default=10, help="Szenen INSGESAMT im Run (fehlende werden ergänzt)")
    p.add_argument("--seed", type=int, help="Basis-Seed der Szenen (Default: scene.seed aus config.json)")
    p.add_argument("--per-scene", type=int)
    p.add_argument("--generators")
    p.add_argument("--detector", choices=["all", "random", "deeplsd", "mlsd", "classical", "opencv-lsd"])
    p.add_argument("--best-by", choices=["recall", "precision"])
    p.add_argument("--device")

    p = sub.add_parser("index", help="index.jsonl / dataset.jsonl / summary.json neu schreiben")
    common(p)

    p = sub.add_parser("negatives", help="Negativ-Datensatz: gleiche Basismodelle ohne ControlNet, Motive ohne "
                                         "Geometrie, Linienerkennung, alle Linien ohne Gruppe")
    common(p)
    p.add_argument("--count", type=int, required=True, help="Bilder INSGESAMT im Run (fehlende werden ergänzt)")
    p.add_argument("--generators", help="Komma-getrennt (Default: generation.generators, je Basismodell einmal)")
    p.add_argument("--detector", choices=["all", "random", "deeplsd", "mlsd", "classical", "opencv-lsd"])
    p.add_argument("--device")
    p.add_argument("--steps", type=int)

    p = sub.add_parser("preview", help="eine Szene schnell ansehen (Linien farbig + Tiefe)")
    common(p, run=False)
    p.add_argument("--type", choices=["room", "facade", "street", "objects"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", type=Path, default=Path("preview.png"))

    sub.add_parser("generators", help="registrierte Generatoren anzeigen")

    a = ap.parse_args(argv)
    # config.json des Runs (u.a. dessen Seeds), darüber --config mit den abweichenden Feldern
    cfg = Config.load(a.run / "config.json" if getattr(a, "run", None) else None, getattr(a, "config", None))
    if getattr(a, "device", None):
        cfg.generation.device = a.device
    if getattr(a, "steps", None):
        cfg.generation.steps = a.steps
    if getattr(a, "detector", None):
        cfg.match.detector = a.detector
    if getattr(a, "best_by", None):
        cfg.match.best_by = a.best_by
    gens = a.generators.split(",") if getattr(a, "generators", None) else None

    from . import pipeline as pl
    if a.cmd == "scenes":
        pl.run_scenes(a.run, cfg, a.count, a.seed, a.type)
    elif a.cmd == "generate":
        pl.run_generate(a.run, cfg, gens, a.per_scene, a.force, a.generators_file, a.limit)
    elif a.cmd == "detect":
        pl.run_detect(a.run, cfg, a.force, a.rematch, a.device)
    elif a.cmd == "all":
        # --count ist hier die Ziel-Anzahl -> nach einem Abbruch denselben Befehl einfach neu starten
        missing = a.count - len(pl.scene_dirs(a.run))
        if missing > 0:
            pl.run_scenes(a.run, cfg, missing, a.seed)
        pl.run_generate(a.run, cfg, gens, a.per_scene)
        pl.run_detect(a.run, cfg, device=a.device)
        print(json.dumps(json.loads((a.run / "summary.json").read_text()), indent=2))
    elif a.cmd == "negatives":
        from .negatives import run_negatives
        print(json.dumps(run_negatives(a.run, cfg, a.count, gens, a.device), indent=2))
    elif a.cmd == "index":
        print(json.dumps(pl.rebuild_index(a.run), indent=2))
    elif a.cmd == "preview":
        _preview(cfg, a.type, a.seed, a.out)
    elif a.cmd == "generators":
        from .diffusion import GENERATORS
        for name, spec in GENERATORS.items():
            u = spec.get("controlnetUnion")
            cns = ", ".join([f"{c['input']}:{c['model']}" for c in spec.get("controlnets", []) + spec.get("adapters", [])]
                            + ([f"{'+'.join(i['input'] for i in u['inputs'])}:{u['model']}"] if u else []))
            print(f"{name:18s} {spec['family']:5s} {spec.get('base', '-')}  [{cns}]")


def _preview(cfg: Config, scene_type, seed: int, out: Path) -> None:
    import cv2
    import numpy as np

    from .render import colored_line_image, depth_image
    from .scenes import make_scene

    s = make_scene("preview", seed, cfg.scene, scene_type)
    cam = s.camera
    a = colored_line_image(s.segments, [d.id for d in s.scene.directions], cam.width, cam.height)
    d = cv2.cvtColor(depth_image(s.zbuf), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out), np.hstack([a, d]))
    print(f"{s.scene.scene_type}: {len(s.segments)} Segmente, {len(s.scene.directions)} Richtungen -> {out}")


if __name__ == "__main__":
    main()
