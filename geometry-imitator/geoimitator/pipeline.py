"""
    runs/<name>/
      config.json
      scenes/<scene>/scene.json  lines.png  depth.png  lines_colored.png
      scenes/<scene>/generations/<gen>/
          image.png             generiertes Bild
          meta.json             Generator, Prompt, Seed, Status, Match-Kennzahlen
          overlay_control.png   Linienbild (cyan) über der Generierung
          detections_<v>.json   erkannte Linien je Verfahren v (Cache für neues Matching; ältere Runs: detections.json)
          overlay_labels_<v>.png  erkannte Linien nach Richtung eingefärbt, je Verfahren
          sample_<v>.json       je angenommenem Verfahren: gelabelte Linien + Beziehungen
          overlay_labels.png, sample.json   dasselbe für das beste Verfahren (sample.json nur wenn angenommen)
      index.jsonl               eine Zeile pro Generierung (alle; Kennzahlen des besten Verfahrens + variants)
      dataset.jsonl             eine Zeile pro angenommener Generierung (mit Split; sample = bestes, variants = alle)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .scenes import load_scene_json, make_scene


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _try_read(p: Path) -> dict | None:
    """None, wenn die Datei fehlt oder kaputt ist (z.B. leer nach einem Abbruch mit Strg+C)."""
    try:
        return _read(p)
    except (OSError, ValueError):
        return None


# Atomar schreiben (temporäre Datei + Umbenennen): ein Abbruch hinterlässt nie halbe Dateien
def _write(p: Path, d: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _imwrite(p: Path, image: np.ndarray) -> None:
    tmp = p.with_name(f"{p.stem}.tmp{p.suffix}")
    if not cv2.imwrite(str(tmp), image):
        raise OSError(f"Bild konnte nicht geschrieben werden: {p}")
    os.replace(tmp, p)


def scene_dirs(run: Path) -> list[Path]:
    return sorted(p.parent for p in (run / "scenes").glob("*/scene.json"))


def generation_dirs(run: Path) -> list[Path]:
    return sorted(p.parent for p in (run / "scenes").glob("*/generations/*/image.png"))


def _log(msg: str) -> None:
    print(msg, flush=True)


# 




# Szenen erstellen

def init_run(run: Path, cfg: Config) -> None:
    """Legt config.json an, falls noch nicht vorhanden. Fehlende Seeds werden dabei zufällig gezogen
    und gespeichert: Jeder neue Run bekommt andere Szenen, Prompts und Generatoren, ein
    fortgesetzter Run liest seine Seeds wieder aus config.json."""
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        return
    import secrets
    if cfg.scene.seed is None:
        cfg.scene.seed = secrets.randbelow(2 ** 31)
    if cfg.generation.seed is None:
        cfg.generation.seed = secrets.randbelow(2 ** 31)
    cfg.save(run / "config.json")
    _log(f"Neuer Run {run}: scene.seed {cfg.scene.seed}, generation.seed {cfg.generation.seed}")


def _seed_or_zero(v: int | None) -> int:
    return 0 if v is None else int(v)   # Runs von vor init_run hatten fest 0


def run_scenes(run: Path, cfg: Config, count: int, seed: int | None = None, scene_type: str | None = None) -> list[str]:
    init_run(run, cfg)
    seed = _seed_or_zero(cfg.scene.seed if seed is None else seed)
    existing = {p.name for p in scene_dirs(run)}
    start = len(existing)
    made = []
    t0 = time.perf_counter()
    for i in range(start, start + count):
        sid = f"s{i:05d}"
        if sid in existing:
            continue
        s_seed = int.from_bytes(hashlib.sha256(f"{seed}/{i}".encode()).digest()[:4], "little")
        try:
            sample = make_scene(sid, s_seed, cfg.scene, scene_type)
        except RuntimeError as e:
            _log(f"  {sid}: übersprungen ({e})")
            continue
        sample.save(run / "scenes" / sid)
        made.append(sid)
        counts = sample.direction_counts
        _log(f"  {sid} {sample.scene.scene_type:8s} {sample.camera.width}x{sample.camera.height} "
             f"{len(sample.segments):4d} Segmente, {sum(c >= 2 for c in counts.values())} Richtungen"
             + (f", gestört: {[p['direction'] for p in sample.perturbation]}" if sample.perturbation else ""))
    _log(f"{len(made)} Szenen in {time.perf_counter() - t0:.1f} s -> {run / 'scenes'}")
    return made


# generieren

def run_generate(run: Path, cfg: Config, generators: list[str] | None = None, per_scene: int | None = None,
                 force: bool = False, generators_file: Path | None = None, limit: int | None = None) -> int:
    from .diffusion import DiffusionGenerator, load_generators
    from .overlay import overlay_control
    from .prompts import make_prompt

    g = cfg.generation
    specs = load_generators(generators_file or (run / "generators.json"))
    names = generators or g.generators
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise ValueError(f"Unbekannte Generatoren {unknown}. Verfügbar: {sorted(specs)}")
    per_scene = per_scene or g.per_scene

    # Plan: (Generator -> [(scene_dir, gen_id, seed)])
    jobs: dict[str, list] = defaultdict(list)
    n_jobs = 0
    for si, sdir in enumerate(scene_dirs(run)):
        for k in range(per_scene):
            gid = f"g{k:02d}"
            gdir = sdir / "generations" / gid
            if (gdir / "image.png").exists() and not force:
                continue
            seed = int.from_bytes(hashlib.sha256(f"{_seed_or_zero(g.seed)}/{sdir.name}/{gid}".encode()).digest()[:4], "little")
            gen = names[np.random.default_rng(seed).integers(len(names))]
            jobs[gen].append((sdir, gid, seed))
            n_jobs += 1
            if limit and n_jobs >= limit:
                break
        if limit and n_jobs >= limit:
            break
    done = 0
    for name, items in jobs.items():
        _log(f"Generator {name}: {len(items)} Bilder")
        model = DiffusionGenerator(name, specs[name], g.device, g.dtype, g.cpu_offload)
        for sdir, gid, seed in items:
            scene_json = load_scene_json(sdir)
            prompt = make_prompt(scene_json["sceneType"], np.random.default_rng(seed + 1),
                                 g.prompt.get("offTopic", 0.0), g.prompt.get("nonPhoto", 0.0))
            res = model.generate(scene_json, sdir, prompt, seed, g.steps)
            gdir = sdir / "generations" / gid
            gdir.mkdir(parents=True, exist_ok=True)
            for pattern in ("sample*.json", "detections*.json", "overlay_labels*.png"):
                for stale in gdir.glob(pattern):
                    stale.unlink()
            _imwrite(gdir / "overlay_control.png", overlay_control(res.image, scene_json))
            meta = {"id": gid, "sceneId": scene_json["id"], "sceneType": scene_json["sceneType"], "generator": name,
                    "generatorSpec": specs[name], **prompt, "seed": seed, "params": res.params,
                    "durationMs": round(res.duration_ms, 1), "status": "generated"}
            _write(gdir / "meta.json", meta)
            _imwrite(gdir / "image.png", res.image)   # zuletzt: image.png heißt "Generierung fertig"
            done += 1
            _log(f"  {sdir.name}/{gid} [{name}] {res.duration_ms / 1000:.1f} s  {prompt['subject']}"
                 + (" [off-topic]" if prompt["offTopic"] else "") + (f" [{prompt['style']}]" if prompt["style"] != "photo" else ""))
        model.unload()
    _log(f"{done} Bilder generiert.")
    return done



# Erkennen + Zuordnen

def _cached_detections(gdir: Path) -> dict[str, dict]:
    """Gespeicherte Linien je Verfahren: detections_<verfahren>.json, dazu detections.json älterer Runs.
    Kaputte Dateien (z.B. leer nach einem Abbruch) fehlen -> werden neu erkannt."""
    out = {}
    legacy = _try_read(gdir / "detections.json")
    if legacy and "lines" in legacy:
        out[legacy.get("detector", {}).get("method", "deeplsd")] = legacy
    for p in gdir.glob("detections_*.json"):
        d = _try_read(p)
        if d and "lines" in d:
            out[p.stem.removeprefix("detections_")] = d
    return out


def _best_variant(variants: dict[str, dict], best_by: str = "recall") -> str:
    """Bestes Verfahren: angenommen vor abgelehnt, dann nach `best_by`:
    "recall" = recallLines, bei Gleichstand precision; "precision" = precision, bei Gleichstand recallLines."""
    if best_by not in ("recall", "precision"):
        raise ValueError(f"match.bestBy muss 'recall' oder 'precision' sein, nicht {best_by!r}")

    def key(k):
        m = variants[k]["match"]
        order = (m["recallLines"], m["precision"]) if best_by == "recall" else (m["precision"], m["recallLines"])
        return (variants[k]["status"] == "accepted", *order)
    return max(variants, key=key)


def run_detect(run: Path, cfg: Config, force: bool = False, rematch: bool = False, device: str | None = None) -> Counter:
    """Pro Generierung jedes Verfahren (match.detector, "all" = alle) erkennen und zuordnen. Jede angenommene
    Variante -> sample_<verfahren>.json; die beste (angenommen, höchster recallLines) zusätzlich als
    sample.json. --rematch nutzt nur gespeicherte Linien (fehlende Verfahren werden ausgelassen)."""
    from .labels import build_sample, scaled_segments
    from .matching import match_lines
    from .overlay import overlay_labels

    from .detection import LineDetector, choose_detectors

    m = cfg.match
    detectors: dict[str, LineDetector] = {}
    stats: Counter = Counter()
    for gdir in generation_dirs(run):
        meta_p = gdir / "meta.json"
        meta = _try_read(meta_p) or {"id": gdir.name, "sceneId": gdir.parent.parent.name}
        methods = choose_detectors(f"{meta['sceneId']}/{meta['id']}", m.detector, m.detectors)
        done = set(meta.get("variants") or ([meta["detector"]] if meta.get("detector") else []))
        if meta.get("status") in ("accepted", "rejected") and set(methods) <= done and not (force or rematch):
            stats["skipped"] += 1
            continue
        scene_json = load_scene_json(gdir.parent.parent)
        image = cv2.imread(str(gdir / "image.png"))
        if image is None:
            _log(f"  {gdir.parent.parent.name}/{gdir.name} übersprungen: image.png nicht lesbar "
                 f"(neu erzeugen mit 'generate --force' oder Ordner löschen)")
            stats["broken"] += 1
            continue
        H, W = image.shape[:2]
        cached = {} if force else _cached_detections(gdir)
        if rematch and cached:
            # nur gespeicherte Linien
            methods = [k for k in methods if k in cached] or sorted(cached)
        gt = scaled_segments(scene_json, W, H)
        gt_arr = np.array([[*s["p1"], *s["p2"]] for s in gt], float).reshape(-1, 4)
        variants, samples, overlays = {}, {}, {}
        for method in methods:
            det_info = cached.get(method)
            if det_info is None:
                if method not in detectors:
                    detectors[method] = LineDetector(method, m.deeplsd_max_dimension, m.min_det_length_px, device)
                detector = detectors[method]
                t0 = time.perf_counter()
                det = detector.detect(gdir / "image.png")
                det_info = {"detector": detector.info, "durationMs": round((time.perf_counter() - t0) * 1000, 1),
                            "imageWidth": W, "imageHeight": H, "lines": np.round(det, 3).tolist()}
                _write(gdir / f"detections_{method}.json", det_info)
            det = np.array(det_info["lines"], float).reshape(-1, 4)
            res = match_lines(gt_arr, [s["direction"] for s in gt], det, W, m, [s["line3d"] for s in gt])
            samples[method] = build_sample(scene_json, meta, det, gt, res, W, H, det_info["detector"])
            overlays[method] = overlay_labels(image, samples[method], gt)
            _imwrite(gdir / f"overlay_labels_{method}.png", overlays[method])
            if res.accepted:
                _write(gdir / f"sample_{method}.json", samples[method])
            else:
                (gdir / f"sample_{method}.json").unlink(missing_ok=True)
            variants[method] = {"status": "accepted" if res.accepted else "rejected", "match": samples[method]["match"]}
            mm = res.metrics
            _log(f"  {gdir.parent.parent.name}/{gdir.name} {method:10s} {variants[method]['status']:8s} "
                 f"recall lines {mm['recallLines']:.2f} matched {mm['matchedLines']}/{mm['detectedLines']} "
                 f"groups {mm['matchedGroups']}/{mm['gtGroups']}" + (f"  ({'; '.join(res.reasons)})" if res.reasons else ""))
        best = _best_variant(variants, cfg.match.best_by)
        _imwrite(gdir / "overlay_labels.png", overlays[best])
        if variants[best]["status"] == "accepted":
            _write(gdir / "sample.json", samples[best])
        else:
            (gdir / "sample.json").unlink(missing_ok=True)
        meta.update(status=variants[best]["status"], match=variants[best]["match"], detector=best, variants=variants)
        _write(meta_p, meta)
        stats[meta["status"]] += 1
        if len(variants) > 1:
            _log(f"  {gdir.parent.parent.name}/{gdir.name} -> {best} ({meta['status']})")
    rebuild_index(run)
    _log(f"Erkennung: {dict(stats)}")
    return stats




def split_of(scene_id: str, val: float = 0.1, test: float = 0.1) -> str:
    """Split nach Szene (alle Generierungen einer Szene im selben Split -> kein Leck)."""
    h = int.from_bytes(hashlib.sha256(scene_id.encode()).digest()[:4], "little") / 2 ** 32
    return "test" if h < test else "val" if h < test + val else "train"


def rebuild_index(run: Path) -> dict:
    rows, data = [], []
    for gdir in generation_dirs(run):
        meta = _try_read(gdir / "meta.json")
        if meta is None:
            continue
        rel = gdir.relative_to(run).as_posix()
        mt = meta.get("match", {})
        row = {"path": rel, "sceneId": meta.get("sceneId"), "generationId": meta.get("id"),
               "sceneType": meta.get("sceneType"), "generator": meta.get("generator"), "status": meta.get("status"),
               "split": split_of(meta.get("sceneId", "")), "prompt": meta.get("prompt"),
               "offTopic": bool(meta.get("offTopic")), "style": meta.get("style", "photo"),
               "detector": meta.get("detector", "deeplsd"),
               **{k: mt.get(k) for k in ("recallLines", "recallLength", "recallSegments", "matchedLines", "detectedLines",
                                         "ambiguousLines", "unknownLines", "matchedGroups", "gtGroups",
                                         "meanAngleDeg", "meanDistPx")},
               # alle Verfahren dieser Generierung (ältere Runs: nur das eine)
               "variants": {k: {"status": v["status"], **{x: v["match"].get(x) for x in
                                                          ("recallLines", "recallLength", "matchedLines", "detectedLines")}}
                            for k, v in (meta.get("variants") or {}).items()}
                           or {meta.get("detector", "deeplsd"): {"status": meta.get("status"),
                                                                 **{x: mt.get(x) for x in ("recallLines", "recallLength",
                                                                                           "matchedLines", "detectedLines")}}}}
        rows.append(row)
        if meta.get("status") == "accepted":
            scene = load_scene_json(gdir.parent.parent)
            variants = {k: f"{rel}/sample_{k}.json" for k, v in row["variants"].items()
                        if v["status"] == "accepted" and (gdir / f"sample_{k}.json").exists()}
            data.append({"sample": f"{rel}/sample.json", "image": f"{rel}/image.png", "sceneId": row["sceneId"],
                         "sceneType": row["sceneType"], "generator": row["generator"], "split": row["split"],
                         "perturbed": bool(scene.get("perturbation")), "offTopic": row["offTopic"],
                         "style": row["style"], "detector": row["detector"],
                         "noiseSigmaPx": scene.get("noiseSigmaPx", 0.0), "lines": row["detectedLines"],
                         "labelledLines": row["matchedLines"],
                         "variants": variants or {row["detector"]: f"{rel}/sample.json"}})
    (run / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (run / "dataset.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    summary = summarize(rows)
    _write(run / "summary.json", summary)
    return summary


def summarize(rows: list[dict]) -> dict:
    out: dict = {"generations": len(rows), "byStatus": dict(Counter(r["status"] for r in rows)), "byGenerator": {},
                 "bySceneType": {}, "byStyle": {}, "byOffTopic": {}, "bestDetector": {}, "byDetector": {}}
    for key, name in (("generator", "byGenerator"), ("sceneType", "bySceneType"), ("style", "byStyle"),
                      ("offTopic", "byOffTopic"), ("detector", "bestDetector")):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for k, rs in groups.items():
            judged = [r for r in rs if r["status"] in ("accepted", "rejected")]
            out[name][k] = {"generations": len(rs), "judged": len(judged),
                            "acceptRate": (sum(r["status"] == "accepted" for r in judged) / len(judged)) if judged else None,
                            "meanRecallLength": float(np.mean([r["recallLength"] for r in judged])) if judged else None}
    # jedes Verfahren für sich (bei detector "all" mehrere pro Generierung)
    per: dict[str, list] = defaultdict(list)
    for r in rows:
        for k, v in r.get("variants", {}).items():
            if v["status"] in ("accepted", "rejected"):
                per[k].append(v)
    for k, vs in per.items():
        out["byDetector"][k] = {"judged": len(vs), "acceptRate": sum(v["status"] == "accepted" for v in vs) / len(vs),
                                "meanRecallLines": float(np.mean([v["recallLines"] or 0 for v in vs]))}
    return out
