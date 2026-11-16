from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .config import SceneConfig
from .perturb import perturb_segments
from .render import colored_line_image, depth_image, line_image, render_buffers, visible_segments
from .scene import Scene
from .templates import TEMPLATES

FORMAT_VERSION = 1


@dataclass
class SceneSample:
    id: str
    seed: int
    scene: Scene
    camera: Camera
    segments: list[dict]
    zbuf: np.ndarray
    fbuf: np.ndarray
    perturbation: list[dict]
    thickness: int
    attempts: int
    duration_ms: float
    noise_sigma: float = 0.0

    @property
    def direction_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for s in self.segments:
            c[s["direction"]] = c.get(s["direction"], 0) + 1
        return c

    # --- Export 
    def to_dict(self) -> dict:
        cam = self.camera
        counts = self.direction_counts
        dirs = []
        for d in self.scene.directions:
            dd = d.to_dict()
            dd["vanishingPoint"] = vanishing_point_dict(cam, d.vector)
            dd["segmentCount"] = counts.get(d.id, 0)
            dirs.append(dd)
        fams = []
        for f in self.scene.families:
            fd = f.to_dict()
            fd["vanishingLine"] = np.round(cam.vanishing_line(f.normal), 9).tolist()
            fams.append(fd)
        sd = self.scene.to_dict()
        return {
            "version": FORMAT_VERSION,
            "id": self.id,
            "seed": self.seed,
            "sceneType": self.scene.scene_type,
            "image": {"width": cam.width, "height": cam.height},
            "lineThickness": self.thickness,
            "noiseSigmaPx": self.noise_sigma,
            "camera": cam.to_dict(),
            "directions": dirs,
            "planeFamilies": fams,
            "planes": sd["planes"],
            "relations": sd["relations"],
            "perturbation": self.perturbation,
            "segments2d": [_round_seg(s) for s in self.segments],
            "lines3d": sd["lines3d"],
            "faces": sd["faces"],
            "meta": {**sd["meta"], "attempts": self.attempts, "durationMs": round(self.duration_ms, 1)},
        }

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cam = self.camera
        cv2.imwrite(str(out_dir / "lines.png"), line_image(self.segments, cam.width, cam.height, self.thickness))
        cv2.imwrite(str(out_dir / "depth.png"), depth_image(self.zbuf))
        cv2.imwrite(str(out_dir / "lines_colored.png"),
                    colored_line_image(self.segments, [d.id for d in self.scene.directions], cam.width, cam.height))
        # zuletzt: scene.json heißt "Szene fertig" (Abbruch hinterlässt keine halbe Szene)
        tmp = out_dir / "scene.json.tmp"
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out_dir / "scene.json")


def vanishing_point_dict(cam: Camera, direction) -> dict:
    v = cam.vanishing_point(direction)
    h = (v / np.linalg.norm(v)).tolist()
    if abs(v[2]) < 1e-9 or max(abs(v[0] / v[2]), abs(v[1] / v[2])) > 1e7:
        return {"x": None, "y": None, "atInfinity": True, "homogeneous": h}
    return {"x": float(v[0] / v[2]), "y": float(v[1] / v[2]), "atInfinity": False, "homogeneous": h}


def _round_seg(s: dict) -> dict:
    s = dict(s)
    # Koordinaten NICHT runden: Fluchtpunkte fast paralleler Linien (weit entfernt) reagieren
    # schon auf 1e-4 px so empfindlich, dass die exakte Auswertung sonst Inkonsistenzen meldet.
    s["depth"] = round(s["depth"], 4)
    return s


def choose_type(rng: np.random.Generator, cfg: SceneConfig) -> str:
    names = list(cfg.types)
    p = np.array([cfg.types[n] for n in names], float)
    return str(rng.choice(names, p=p / p.sum()))


def make_scene(scene_id: str, seed: int, cfg: SceneConfig, scene_type: str | None = None,
               size: tuple[int, int] | None = None) -> SceneSample:
    """Würfelt so lange neu, bis genug sichtbare Linien und Gruppen vorhanden sind."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    best = None
    for attempt in range(1, cfg.max_attempts + 1):
        stype = scene_type or choose_type(rng, cfg)
        w, h = size or tuple(cfg.sizes[rng.integers(len(cfg.sizes))])
        scene, cam = TEMPLATES[stype](rng, int(w), int(h))
        zbuf, fbuf = render_buffers(scene, cam)
        segs = visible_segments(scene, cam, zbuf, cfg.min_segment_px)
        counts: dict[str, int] = {}
        for s in segs:
            counts[s["direction"]] = counts.get(s["direction"], 0) + 1
        groups = sum(c >= 2 for c in counts.values())
        ok = (cfg.min_segments <= len(segs) <= cfg.max_segments and groups >= cfg.min_groups
              and len(counts) <= cfg.max_directions)
        cand = (scene, cam, segs, zbuf, fbuf)
        if ok:
            best = cand
            break
        if (best is None or len(segs) > len(best[2])) and len(counts) <= cfg.max_directions:
            best = cand
    else:
        raise RuntimeError(f"Szene {scene_id}: nach {cfg.max_attempts} Versuchen keine brauchbare Szene "
                           f"(zuletzt {len(best[2]) if best else 0} Segmente).")
    scene, cam, segs, zbuf, fbuf = best
    perturbation = perturb_segments(segs, cam, rng, cfg.perturbation)
    thickness = int(rng.choice(cfg.line_thickness))
    sigma = add_noise(segs, cam.width, cam.height, rng, cfg.noise_sigma_px)
    return SceneSample(scene_id, seed, scene, cam, segs, zbuf, fbuf, perturbation, thickness, attempt,
                       (time.perf_counter() - t0) * 1000, sigma)


def add_noise(segments: list[dict], W: int, H: int, rng: np.random.Generator, sigma_range) -> float:
    """Gauß-Rauschen auf alle Endpunkte (nach der Störung). p1/p2 = verrauscht (Linienbild, Zuordnung),
    p1Exact/p2Exact = exakte (bzw. absichtlich gestörte) Projektion."""
    sigma = float(rng.uniform(*sigma_range))
    for s in segments:
        s["p1Exact"], s["p2Exact"] = list(s["p1"]), list(s["p2"])
        if sigma > 0:
            for k in ("p1", "p2"):
                x, y = np.asarray(s[k]) + rng.normal(0, sigma, 2)
                s[k] = [float(np.clip(x, 0, W - 1)), float(np.clip(y, 0, H - 1))]
    return sigma


def load_scene_json(scene_dir: Path) -> dict:
    return json.loads((Path(scene_dir) / "scene.json").read_text(encoding="utf-8"))
