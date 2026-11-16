"""Negativ-Datensatz: Bilder OHNE vorgegebene Geometrie

Dieselben Basismodelle wie die Generatoren (nur Text-zu-Bild, kein ControlNet,
kein Adapter), Prompts ohne Bauwerke/Gegenstände (prompts.make_negative_prompt),
danach dieselbe zufällige Linienerkennung. Alle erkannten Linien bekommen den
Status "unknown" und kein Label -> sample.json hat keine Richtungen/Familien.
Ein Gruppierungsmodell lernt daraus: alle Queries "kein Objekt", Kopf "strukturell" = 0.

    runs/<name>/
      config.json  index.jsonl  dataset.jsonl  summary.json
      images/<id>/
        image.png  meta.json  detections.json  sample.json  overlay.png

    python -m geoimitator negatives --run runs/neg1 --count 500

`--count` ist die Zielzahl insgesamt; Abbrechen und mit demselben Befehl neu
starten setzt fort. Generatoren mit demselben Basismodell (z.B. sdxl-juggernaut
und sdxl-t2i-juggernaut) zählen einmal.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .detection import LineDetector, choose_detector
from .diffusion import _draw, load_generators, model_path, output_size
from .pipeline import _imwrite, _log, _seed_or_zero, _try_read, _write, init_run, split_of
from .prompts import make_negative_prompt


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "little")


class Text2Image:
    """Basismodell eines Generators ohne ControlNet/Adapter."""

    def __init__(self, spec: dict, device: str | None, dtype: str, cpu_offload: bool):
        self.spec, self.device_pref, self.dtype_pref, self.cpu_offload = spec, device, dtype, cpu_offload
        self.pipe = None

    def load(self) -> None:
        if self.pipe is not None or self.spec["family"] == "mock":
            return
        import torch
        from diffusers import AutoencoderKL, AutoPipelineForText2Image, UniPCMultistepScheduler

        device = self.device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (torch.float16 if device == "cuda" else torch.float32) if self.dtype_pref == "auto" \
            else getattr(torch, self.dtype_pref)
        spec, kwargs = self.spec, {"torch_dtype": dtype}
        if spec.get("variant"):
            kwargs["variant"] = spec["variant"]
        if spec.get("vae"):
            kwargs["vae"] = AutoencoderKL.from_pretrained(model_path(spec["vae"]), torch_dtype=dtype)
        if spec["family"] == "sd15":
            kwargs.update(safety_checker=None, requires_safety_checker=False)
        pipe = AutoPipelineForText2Image.from_pretrained(model_path(spec["base"]), **kwargs)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if (self.cpu_offload or spec.get("cpuOffload")) and device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def generate(self, prompt: dict, W: int, H: int, seed: int, steps: int | None) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        if self.spec["family"] == "mock":
            return mock_image(W, H, rng), {"size": [W, H], "durationMs": round((time.perf_counter() - t0) * 1000)}
        import torch

        self.load()
        t0 = time.perf_counter()
        params = {"size": [W, H], "steps": steps or self.spec.get("steps", 30),
                  "guidance": _draw(rng, self.spec.get("guidance", 7.0))}
        out = self.pipe(prompt=prompt["prompt"], negative_prompt=prompt["negativePrompt"] or None, width=W, height=H,
                        num_inference_steps=params["steps"], guidance_scale=params["guidance"],
                        generator=torch.Generator("cpu").manual_seed(int(seed))).images[0]
        params["durationMs"] = round((time.perf_counter() - t0) * 1000)
        return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR), params

    def unload(self) -> None:
        self.pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def mock_image(W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    """Weiche Farbflecken + Rauschen, ohne Kanten (Tests ohne Diffusionsmodell)."""
    small = rng.uniform(0, 255, (int(rng.integers(3, 7)), int(rng.integers(3, 7)), 3)).astype(np.float32)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC) + rng.normal(0, 6, (H, W, 3))
    return cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), 1.0)


def distinct_bases(names: list[str], specs: dict) -> list[str]:
    """Pro Basismodell (+ VAE) nur der erste Generator."""
    seen, out = set(), []
    for n in names:
        key = (specs[n].get("base", n), specs[n].get("vae"))
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def run_negatives(run: Path, cfg: Config, count: int, generators: list[str] | None = None,
                  device: str | None = None) -> dict:
    init_run(run, cfg)
    g = cfg.generation
    gseed = _seed_or_zero(g.seed)
    specs = load_generators(run / "generators.json")
    names = distinct_bases(generators or g.generators, specs)
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise ValueError(f"Unbekannte Generatoren {unknown}. Verfügbar: {sorted(specs)}")

    # 1) Generieren, nach Generator gebündelt (jedes Modell nur einmal laden)
    jobs: dict[str, list[str]] = defaultdict(list)
    for i in range(count):
        nid = f"n{i:05d}"
        if not (run / "images" / nid / "image.png").exists():
            jobs[names[np.random.default_rng(_seed(gseed, "neg", nid)).integers(len(names))]].append(nid)
    for name, ids in jobs.items():
        _log(f"Generator {name}: {len(ids)} Bilder")
        model = Text2Image(specs[name], device or g.device, g.dtype, g.cpu_offload)
        for nid in ids:
            seed = _seed(gseed, "neg", nid)
            rng = np.random.default_rng(seed)
            w, h = cfg.scene.sizes[int(rng.integers(len(cfg.scene.sizes)))]
            W, H = output_size(specs[name], int(w), int(h))
            prompt = make_negative_prompt(rng, g.prompt.get("nonPhoto", 0.08))
            img, params = model.generate(prompt, W, H, seed, g.steps)
            d = run / "images" / nid
            d.mkdir(parents=True, exist_ok=True)
            _write(d / "meta.json", {"id": nid, "generator": name, "mode": "txt2img", **prompt, "seed": seed,
                                     "params": params})
            _imwrite(d / "image.png", img)   # zuletzt: image.png heißt "Bild fertig"
            _log(f"  {nid} [{name}] {params['durationMs'] / 1000:.1f} s  {prompt['subject']}")
        model.unload()

    # 2) Linien erkennen, alle ohne Gruppe
    m = cfg.match
    detectors: dict[str, LineDetector] = {}
    for d in sorted(p.parent for p in (run / "images").glob("*/image.png")):
        meta = _try_read(d / "meta.json")
        if meta is None or (meta.get("status") == "done" and (d / "sample.json").exists()):
            continue
        method = choose_detector(f"neg/{d.name}", m.detector, m.detectors)
        if method not in detectors:
            detectors[method] = LineDetector(method, m.deeplsd_max_dimension, m.min_det_length_px, device)
        image = cv2.imread(str(d / "image.png"))
        if image is None:
            _log(f"  {d.name} übersprungen: image.png nicht lesbar (Ordner löschen und neu starten)")
            continue
        H, W = image.shape[:2]
        det = detectors[method].detect(d / "image.png")
        _write(d / "detections.json", {"detector": detectors[method].info, "imageWidth": W, "imageHeight": H,
                                       "lines": np.round(det, 3).tolist()})
        _write(d / "sample.json", build_negative_sample(meta, det, W, H, detectors[method].info))
        ov = (image * 0.5).astype(np.uint8)
        for x1, y1, x2, y2 in det:
            cv2.line(ov, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1, cv2.LINE_AA)
        _imwrite(d / "overlay.png", ov)
        meta.update(status="done", detector=method, detectedLines=len(det))
        _write(d / "meta.json", meta)
        _log(f"  {d.name} {method:10s} {len(det):4d} Linien")
    return rebuild_negative_index(run)


def build_negative_sample(meta: dict, det: np.ndarray, W: int, H: int, detector_info: dict) -> dict:
    """sample.json im Format von labels.build_sample, aber ohne Labels und Beziehungen."""
    pt = [({"x": round(float(x1), 3), "y": round(float(y1), 3)}, {"x": round(float(x2), 3), "y": round(float(y2), 3)})
          for x1, y1, x2, y2 in det]
    lines = [{"index": i, "point1": a, "point2": b, "length": round(float(np.hypot(b["x"] - a["x"], b["y"] - a["y"])), 3),
              "status": "unknown", "label": None, "match": None} for i, (a, b) in enumerate(pt)]
    return {
        "version": 1, "id": meta["id"], "sceneId": meta["id"], "generationId": "g00", "sceneType": "negative",
        "image": {"path": "image.png", "width": W, "height": H},
        "source": {k: meta.get(k) for k in ("generator", "prompt", "negativePrompt", "offTopic", "style", "seed",
                                            "params")},
        "detector": detector_info, "camera": None, "noiseSigmaPx": 0.0,
        "match": {"detectedLines": len(lines), "matchedLines": 0, "accepted": True, "reasons": []},
        "perturbation": [], "lines": lines, "directions": [], "planeFamilies": [], "planes": [],
        "relations": {"parallel": {}, "coplanar": {}, "parallelPlanes": {}, "perpendicularPairs": [],
                      "perpendicularTriples": [], "vanishingLines": {}},
        "training": {"directionIds": [], "directionLabel": [-1] * len(lines), "ignore": [True] * len(lines),
                     "perturbed": [False] * len(lines)},
    }


def rebuild_negative_index(run: Path) -> dict:
    rows, data = [], []
    for d in sorted(p.parent for p in (run / "images").glob("*/image.png")):
        meta = _try_read(d / "meta.json")
        if meta is None:
            continue
        rel = d.relative_to(run).as_posix()
        row = {"path": rel, "id": meta["id"], "generator": meta["generator"], "status": meta.get("status"),
               "split": split_of(meta["id"]), "subject": meta.get("subject"), "style": meta.get("style"),
               "detector": meta.get("detector"), "detectedLines": meta.get("detectedLines")}
        rows.append(row)
        if meta.get("status") == "done":
            data.append({"sample": f"{rel}/sample.json", "image": f"{rel}/image.png", "sceneId": meta["id"],
                         "sceneType": "negative", "generator": row["generator"], "split": row["split"],
                         "perturbed": False, "offTopic": True, "style": row["style"], "detector": row["detector"],
                         "noiseSigmaPx": 0.0, "lines": row["detectedLines"], "labelledLines": 0})
    (run / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (run / "dataset.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data),
                                       encoding="utf-8")
    summary = {"images": len(rows), "done": len(data),
               **{f"by{k.capitalize()}": dict(Counter(r[k] for r in rows)) for k in ("generator", "detector", "split")},
               "meanDetectedLines": float(np.mean([r["lines"] for r in data])) if data else None}
    _write(run / "summary.json", summary)
    return summary
