"""Bildgenerierung mit ControlNet (Linienbild + optional Tiefenbild)

Generatoren sind unten in GENERATORS registriert (eigene über
`generators.json` im Run-Ordner bzw. `--generators-file`). torch/diffusers
werden erst beim ersten echten Generieren importiert.

``mock`` erzeugt ohne Diffusionsmodell ein grob schattiertes Bild aus der
Szene (zum Testen der Pipeline ohne GPU).

Modelle: ``{models}`` ist ein Platzhalter für den Modellordner (``models/`` neben
``geoimitator/`` oder ``GEOIMITATOR_MODELS``). In diesen Ordner müssen die Modelle
(diffusers-Format), Unterordner siehe ``models/README.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .config import MODELS_DIR
from .render import line_image, render_buffers
from .scene import Scene

def model_path(ref: str) -> str:
    """'{models}/sd15/...' -> Pfad im Modellordner. Das Modell muss lokal liegen."""
    path = Path(ref.replace("{models}", str(MODELS_DIR)))
    if not path.exists():
        raise FileNotFoundError(f"Modell fehlt: {path} - siehe models/README.md (oder GEOIMITATOR_MODELS setzen)")
    return str(path)

# Werte in [lo, hi] werden pro Generierung zufällig gezogen
GENERATORS: dict[str, dict] = {
    # SD 1.5 + M-LSD + Depth (Multi-ControlNet)
    "sd15-base": {
        "family": "sd15", "base": "{models}/sd15/stable-diffusion-v1-5",
        "controlnets": [
            {"model": "{models}/sd15/controlnet-mlsd", "input": "lines", "scale": [0.9, 1.2]},
            {"model": "{models}/sd15/controlnet-depth", "input": "depth", "scale": [0.3, 0.7]},
        ],
        "steps": 30, "guidance": [6.0, 8.5], "controlEnd": [0.8, 1.0], "resolutionScale": 1.0,
    },
    "sd15-realvis": {
        "family": "sd15", "base": "{models}/sd15/realistic-vision-v5.1", "vae": "{models}/sd15/vae-ft-mse",
        "controlnets": [
            {"model": "{models}/sd15/controlnet-mlsd", "input": "lines", "scale": [0.9, 1.2]},
            {"model": "{models}/sd15/controlnet-depth", "input": "depth", "scale": [0.3, 0.7]},
        ],
        "steps": 30, "guidance": [5.0, 7.5], "controlEnd": [0.8, 1.0], "resolutionScale": 1.0,
    },
    "sd15-dreamshaper": {
        "family": "sd15", "base": "{models}/sd15/dreamshaper-8",
        "controlnets": [
            {"model": "{models}/sd15/controlnet-mlsd", "input": "lines", "scale": [0.9, 1.2]},
            {"model": "{models}/sd15/controlnet-depth", "input": "depth", "scale": [0.3, 0.7]},
        ],
        "steps": 28, "guidance": [5.0, 7.5], "controlEnd": [0.8, 1.0], "resolutionScale": 1.0,
    },
    # nur Linien 
    "sd15-mlsd-only": {
        "family": "sd15", "base": "{models}/sd15/realistic-vision-v5.1", "vae": "{models}/sd15/vae-ft-mse",
        "controlnets": [{"model": "{models}/sd15/controlnet-mlsd", "input": "lines", "scale": [1.0, 1.3]}],
        "steps": 30, "guidance": [5.0, 7.5], "controlEnd": [0.85, 1.0], "resolutionScale": 1.0,
    },
    # SDXL: Canny-ControlNet
    "sdxl-canny-depth": {
        "family": "sdxl", "base": "{models}/sdxl/stable-diffusion-xl-base-1.0", "vae": "{models}/sdxl/vae-fp16-fix",
        "controlnets": [
            {"model": "{models}/sdxl/controlnet-canny", "input": "lines", "scale": [0.7, 1.0]},
            {"model": "{models}/sdxl/controlnet-depth", "input": "depth", "scale": [0.3, 0.6]},
        ],
        "steps": 30, "guidance": [5.0, 7.0], "controlEnd": [0.8, 1.0], "resolutionScale": 1.5,
    },
    # SDXL-Finetune + Union-ControlNet 
    "sdxl-juggernaut": {
        "family": "sdxl", "base": "{models}/sdxl/juggernaut-xl-v9", "variant": "fp16",
        "vae": "{models}/sdxl/vae-fp16-fix",
        "controlnetUnion": {"model": "{models}/sdxl/controlnet-union", "scale": [0.7, 1.0],
                            "inputs": [{"input": "lines", "mode": 3}, {"input": "depth", "mode": 1}]},
        "steps": 30, "guidance": [4.0, 6.5], "controlEnd": [0.8, 1.0], "resolutionScale": 1.5,
    },
    # SDXL-Finetune + T2I-Adapter 
    "sdxl-t2i-juggernaut": {
        "family": "sdxl", "base": "{models}/sdxl/juggernaut-xl-v9", "variant": "fp16",
        "vae": "{models}/sdxl/vae-fp16-fix",
        "adapters": [
            {"model": "{models}/sdxl/t2i-adapter-lineart", "input": "lines", "scale": [0.6, 0.9]},
            {"model": "{models}/sdxl/t2i-adapter-depth", "input": "depth", "scale": [0.2, 0.5]},
        ],
        "steps": 30, "guidance": [4.0, 6.5], "controlEnd": [0.5, 0.9], "resolutionScale": 1.5,
    },
    "mock": {"family": "mock", "resolutionScale": 1.0},
}


def load_generators(path: Path | None) -> dict[str, dict]:
    gens = dict(GENERATORS)
    if path and Path(path).exists():
        gens.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return gens


def output_size(spec: dict, width: int, height: int) -> tuple[int, int]:
    s = spec.get("resolutionScale", 1.0)
    return int(round(width * s / 8)) * 8, int(round(height * s / 8)) * 8


def _draw(rng, v):
    return float(rng.uniform(*v)) if isinstance(v, (list, tuple)) else v


@dataclass
class GenerationResult:
    image: np.ndarray        # BGR uint8
    params: dict
    duration_ms: float


class DiffusionGenerator:
    """Hält eine Pipeline geladen (ein Generator nach dem anderen)."""

    def __init__(self, name: str, spec: dict, device: str | None = None, dtype: str = "auto",
                 cpu_offload: bool = False):
        self.name, self.spec = name, spec
        self.device_pref, self.dtype_pref, self.cpu_offload = device, dtype, cpu_offload
        self.pipe = None

    def load(self):
        if self.pipe is not None or self.spec["family"] == "mock":
            return
        import torch
        from diffusers import (AutoencoderKL, ControlNetModel, ControlNetUnionModel, MultiAdapter,
                               StableDiffusionControlNetPipeline, StableDiffusionXLAdapterPipeline,
                               StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetUnionPipeline,
                               T2IAdapter, UniPCMultistepScheduler)

        device = self.device_pref or ("cuda" if torch.cuda.is_available()
                                      else "mps" if torch.backends.mps.is_available() else "cpu")
        if self.dtype_pref == "auto":
            dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            dtype = getattr(torch, self.dtype_pref)
        spec = self.spec
        kwargs = {"torch_dtype": dtype}
        if spec.get("variant"):
            kwargs["variant"] = spec["variant"]
        if spec.get("vae"):
            kwargs["vae"] = AutoencoderKL.from_pretrained(model_path(spec["vae"]), torch_dtype=dtype)
        if spec.get("adapters"):
            ads = [T2IAdapter.from_pretrained(model_path(a["model"]), torch_dtype=dtype) for a in spec["adapters"]]
            kwargs["adapter"] = MultiAdapter(ads) if len(ads) > 1 else ads[0]
            pipe = StableDiffusionXLAdapterPipeline.from_pretrained(model_path(spec["base"]), **kwargs)
        elif spec.get("controlnetUnion"):
            kwargs["controlnet"] = ControlNetUnionModel.from_pretrained(model_path(spec["controlnetUnion"]["model"]),
                                                                        torch_dtype=dtype)
            pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(model_path(spec["base"]), **kwargs)
        else:
            cns = [ControlNetModel.from_pretrained(model_path(c["model"]), torch_dtype=dtype) for c in spec["controlnets"]]
            kwargs["controlnet"] = cns if len(cns) > 1 else cns[0]
            if spec["family"] == "sdxl":
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model_path(spec["base"]), **kwargs)
            else:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_path(spec["base"]), safety_checker=None, requires_safety_checker=False, **kwargs)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if (self.cpu_offload or spec.get("cpuOffload")) and device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        if device == "mps":
            pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.device = pipe, device

    def generate(self, scene_json: dict, scene_dir: Path, prompt: dict, seed: int, steps: int | None = None) -> GenerationResult:
        rng = np.random.default_rng(seed)
        W0, H0 = scene_json["image"]["width"], scene_json["image"]["height"]
        W, H = output_size(self.spec, W0, H0)
        t0 = time.perf_counter()
        if self.spec["family"] == "mock":
            img = mock_image(scene_json, W, H, rng)
            return GenerationResult(img, {"size": [W, H]}, (time.perf_counter() - t0) * 1000)

        import torch
        from PIL import Image

        self.load()
        t0 = time.perf_counter()  # ohne Laden des Modells
        spec = self.spec
        union = spec.get("controlnetUnion")
        inputs = union["inputs"] if union else spec.get("adapters") or spec["controlnets"]
        conds = [Image.fromarray(control_image(scene_json, scene_dir, c["input"], W, H)).convert("RGB")
                 for c in inputs]
        scales = [_draw(rng, union["scale"])] if union else [_draw(rng, c["scale"]) for c in inputs]
        params = {
            "size": [W, H], "steps": steps or spec.get("steps", 30),
            "guidance": _draw(rng, spec.get("guidance", 7.0)),
            "controlScales": scales, "controlEnd": _draw(rng, spec.get("controlEnd", 1.0)),
        }
        multi = len(conds) > 1
        if spec.get("adapters"):
            control = {"image": conds if multi else conds[0],
                       "adapter_conditioning_scale": scales if multi else scales[0],
                       "adapter_conditioning_factor": params["controlEnd"]}
        elif union:
            control = {"control_image": conds, "control_mode": [c["mode"] for c in inputs],
                       "controlnet_conditioning_scale": scales[0], "control_guidance_end": params["controlEnd"]}
        else:
            control = {"image": conds if multi else conds[0],
                       "controlnet_conditioning_scale": scales if multi else scales[0],
                       "control_guidance_end": params["controlEnd"]}
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt["prompt"], negative_prompt=prompt["negativePrompt"], **control,
            num_inference_steps=params["steps"], guidance_scale=params["guidance"],
            width=W, height=H, generator=gen,
        ).images[0]
        img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
        return GenerationResult(img, params, (time.perf_counter() - t0) * 1000)

    def unload(self):
        self.pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def control_image(scene_json: dict, scene_dir: Path, kind: str, W: int, H: int) -> np.ndarray:
    W0, H0 = scene_json["image"]["width"], scene_json["image"]["height"]
    if kind == "lines":  # neu zeichnen statt hochskalieren -> scharfe Linien
        t = max(1, int(round(scene_json.get("lineThickness", 1) * W / W0)))
        return line_image(scene_json["segments2d"], W, H, t, (W / W0, H / H0))
    if kind == "depth":
        d = cv2.imread(str(Path(scene_dir) / "depth.png"), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unbekannter ControlNet-Eingang '{kind}'")


def mock_image(scene_json: dict, W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    """Schattierte Flächen + Kanten + Rauschen - nur für Tests ohne Diffusionsmodell."""
    scene = Scene.from_dict(scene_json)
    cam = Camera.from_dict(scene_json["camera"]).scaled(W, H)
    zbuf, fbuf = render_buffers(scene, cam)
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    base = rng.uniform(60, 200, size=(len(scene.faces) + 1, 3))
    shade = np.array([0.55 + 0.45 * abs(f.normal @ light) for f in scene.faces] + [1.0])
    cols = (base * shade[:, None]).astype(np.float32)
    img = cols[fbuf].copy()
    img[fbuf < 0] = (235, 206, 170)  # Himmel
    # Detail-Linien (Fliesen, Fenstersprossen, ...) leicht dunkler einzeichnen
    sx, sy = W / scene_json["image"]["width"], H / scene_json["image"]["height"]
    for s in scene_json["segments2d"]:
        if rng.random() < 0.8:
            p = (int(s["p1"][0] * sx), int(s["p1"][1] * sy))
            q = (int(s["p2"][0] * sx), int(s["p2"][1] * sy))
            c = float(rng.uniform(20, 70))
            cv2.line(img, p, q, (c, c, c), 1, cv2.LINE_AA)
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0.8)
    return img
