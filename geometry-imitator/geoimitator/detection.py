"""Linienerkennung im generierten Bild

* ``deeplsd``    - DeepLSD (Pautrat et al., CVPR 2023), Paket ``deeplsd`` + Gewichte
* ``mlsd``       - M-LSD "512 large" (ONNX)
* ``classical``  - Canny + probabilistische Hough-Transformation (OpenCV)
* ``opencv-lsd`` - klassischer LSD aus OpenCV (ohne GPU, zum Testen)

Mit ``match.detector = "random"`` wird pro Generierung eine nach ``match.detectors``
gewählt (``choose_detector``, reproduzierbar über die Generierungs-ID).

Modelle: ``deeplsd`` und ``mlsd`` laden ihre Gewichte aus dem Modellordner
(``models/``, siehe ``models/README.md``)
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import MODELS_DIR

DEEPLSD_WEIGHTS = MODELS_DIR / "detection" / "deeplsd_md.tar"
MLSD_MODEL = MODELS_DIR / "detection" / "mlsd-512-large.onnx"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Modell fehlt: {path} - siehe models/README.md (oder GEOIMITATOR_MODELS setzen)")
    return path


class LineDetector:
    def __init__(self, method: str = "deeplsd", max_dimension: int = 1024, min_length_px: float = 8.0,
                 device: str | None = None):
        self.method = method
        self.min_length_px = min_length_px
        self._det = None
        if method == "deeplsd":
            self._det = DeepLSD(max_dimension, device)
        elif method == "mlsd":
            self._det = MLSD()
        elif method == "classical":
            self._det = Classical()
        elif method == "opencv-lsd":
            self._det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        else:
            raise ValueError(f"Unbekannter Detektor '{method}'")

    @property
    def info(self) -> dict:
        d = {"method": self.method, "minLengthPx": self.min_length_px}
        if self.method == "deeplsd" and self._det.loaded:
            d.update(weights=DEEPLSD_WEIGHTS.stem, device=self._det.device)
        return d

    def detect(self, image_path: Path) -> np.ndarray:
        """(N, 4) x1, y1, x2, y2 in Pixeln des Bildes."""
        if self.method == "opencv-lsd":
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            out = self._det.detect(gray)[0]
            segs = np.zeros((0, 4)) if out is None else out.reshape(-1, 4).astype(float)
        else:
            segs = np.asarray(self._det.detect(cv2.imread(str(image_path), cv2.IMREAD_COLOR)), float)
        segs = segs.reshape(-1, 4)
        if len(segs) == 0:
            return np.zeros((0, 4))
        L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        return segs[L >= self.min_length_px]


# DeepLSD braucht extra Backbones

class DeepLSD:
    SIZE_MULTIPLE = 16  

    def __init__(self, max_dimension: int = 1024, device: str | None = None):
        self.max_dimension, self.device_pref = max_dimension, device
        self.net = self.device = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.net is not None

    def load(self):
        if self.net is not None:
            return self.net
        import torch
        from deeplsd.models.deeplsd_inference import DeepLSD as Net

        device = self.device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        conf = {"detect_lines": True,
                "line_detection_params": {"merge": False, "filtering": "normal", "grad_thresh": 3, "grad_nfa": True}}
        ckpt = torch.load(_require(DEEPLSD_WEIGHTS), map_location="cpu", weights_only=False)
        net = Net(conf)
        net.load_state_dict(ckpt["model"])
        self.net, self.device = net.to(device).eval(), device
        return self.net

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # verkleinern auf max_dimension, dann auf Vielfache von 16 abrunden (x/y getrennt zurückskaliert)
        m = self.SIZE_MULTIPLE
        scale = min(1.0, self.max_dimension / max(w, h))
        ww = max(m, int(round(w * scale)) // m * m)
        wh = max(m, int(round(h * scale)) // m * m)
        work = gray if (ww, wh) == (w, h) else cv2.resize(
            gray, (ww, wh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        with self._lock:
            net = self.load()
            x = torch.from_numpy(work).float()[None, None].to(self.device) / 255.0
            with torch.no_grad():
                out = net({"image": x})
        segs = np.asarray(out["lines"][0], dtype=np.float64).reshape(-1, 4)
        return segs * [w / ww, h / wh, w / ww, h / wh]


# MLSD aber als ONNX, weil das Originale Python hier nicht ging

class MLSD:
    INPUT, GRID = 512, 256
    SCORE_THRESHOLD = 0.1
    DIST_THRESHOLD = 20.0   # Grid-Einheiten

    def __init__(self):
        self.session = None

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.session is None:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(_require(MLSD_MODEL)), providers=["CPUExecutionProvider"])
        n = self.INPUT
        H, W = image_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image_bgr, (n, n), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = np.concatenate([rgb, np.full((n, n, 1), 255, np.uint8)], -1).astype(np.float32)[None]
        outs = dict(zip([o.name for o in self.session.get_outputs()],
                        self.session.run(None, {"input_image_with_alpha:0": x})))
        return self.decode(outs["Identity"], outs["Identity_1"], outs["Identity_2"], W, H)

    @classmethod
    def decode(cls, pts, scores, vmap, width: int, height: int) -> np.ndarray:
        """pts (1,200,2) [y,x] im 256er-Grid, scores (1,200), vmap (1,256,256,4) -> Linien im Originalbild."""
        pts = np.asarray(pts).reshape(-1, 2)
        scores = np.asarray(scores).reshape(-1)
        vmap = np.asarray(vmap).reshape(cls.GRID, cls.GRID, 4)
        scale = cls.INPUT / cls.GRID
        sx, sy = width / cls.INPUT * scale, height / cls.INPUT * scale
        out = []
        for (y, x), s in zip(pts, scores):
            if s <= cls.SCORE_THRESHOLD or not (0 <= y < cls.GRID and 0 <= x < cls.GRID):
                continue
            dx1, dy1, dx2, dy2 = vmap[y, x]
            if np.hypot(dx1 - dx2, dy1 - dy2) <= cls.DIST_THRESHOLD:
                continue
            out.append([(x + dx1) * sx, (y + dy1) * sy, (x + dx2) * sx, (y + dy2) * sy])
        return np.array(out, np.float32).reshape(-1, 4)


# Canny und HoughLinesP

class Classical:
    MAX_WORKING_DIMENSION = 1600
    CANNY_LOW, CANNY_HIGH = 50, 150
    HOUGH_VOTES, HOUGH_MIN_LENGTH, HOUGH_MAX_GAP = 60, 40, 10

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        H, W = image_bgr.shape[:2]
        scale = min(1.0, self.MAX_WORKING_DIMENSION / max(W, H))
        img = image_bgr if scale >= 1 else cv2.resize(
            image_bgr, (max(1, round(W * scale)), max(1, round(H * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), self.CANNY_LOW, self.CANNY_HIGH)
        found = cv2.HoughLinesP(edges, 1, np.pi / 180, self.HOUGH_VOTES, minLineLength=self.HOUGH_MIN_LENGTH,
                                maxLineGap=self.HOUGH_MAX_GAP)
        return np.zeros((0, 4), np.float32) if found is None else found.reshape(-1, 4).astype(np.float32) / scale


def choose_detectors(key: str, method: str, weights: dict) -> list[str]:
    """Verfahren für eine Generierung: "all" -> alle mit Gewicht > 0, sonst eins (siehe choose_detector)."""
    if method == "all":
        return [n for n, w in weights.items() if w > 0]
    return [choose_detector(key, method, weights)]


def choose_detector(key: str, method: str, weights: dict) -> str:
    """`method` fest oder "random" (bzw. "all", z.B. bei Negativ-Bildern): gewichtet nach `weights`,
    reproduzierbar über `key`."""
    if method not in ("random", "all"):
        return method
    names = list(weights)
    p = np.array([weights[n] for n in names], float)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    return names[int(rng.choice(len(names), p=p / p.sum()))]
