"""

Automatische Annotation: Linien mit M-LSD (512 large, mlsd.py) oder Canny + Hough (hough), Gruppierung mit
TLC v2 (best, tlc.py) - je ein Modell für Richtungen, Ebenen und senkrechte Richtungen. M-LSD und TLC in PyTorch,
auf der GPU, falls vorhanden. Die Modelle liegen in python/models/


"""

from __future__ import annotations

import base64
import re
import threading
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
MLSD_MODEL = MODELS / "mlsd-512-large.pt"
TLC_MODELS = {"parallel": MODELS / "tlc-v2-parallel.pt", "plane": MODELS / "tlc-v2-planes.pt",
              "perpendicular": MODELS / "tlc-v2-perpendicular.pt"}

# Hough: größere Bilder werden vorher auf diese Seitenlänge verkleinert (Tempo), Ergebnis zurückskaliert
HOUGH_MAX_DIM = 1600
CANNY_LOW, CANNY_HIGH = 50, 150         # Hysterese-Schwellen
HOUGH_RHO = 1                           # px
HOUGH_THETA = np.pi / 180               # 1°
HOUGH_VOTES = 60
HOUGH_MIN_LENGTH = 30                   # px
HOUGH_MAX_GAP = 10                      # px

DETECTORS = {"mlsd": "M-LSD", "hough": "Hough"}

_lock = threading.Lock()
_models: dict = {}


def load_image(data_url: str) -> np.ndarray:
    """data:-URL bzw. base64 -> BGR-Bild."""
    m = re.match(r"^data:[^;,]*;base64,(.*)$", data_url, re.S)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(m.group(1) if m else data_url), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Bild nicht lesbar.")
    return img


def load() -> dict:
    """Modelle einmal laden."""
    with _lock:
        if not _models:
            import torch

            from .mlsd import MLSD
            from .tlc import TLCPredictor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _models["mlsd"] = MLSD(MLSD_MODEL, device)
            _models["tlc"] = {kind: TLCPredictor(path, device) for kind, path in TLC_MODELS.items()}
        return _models


def mlsd(img: np.ndarray) -> np.ndarray:
    """BGR-Bild -> (N, 4) Linien x1 y1 x2 y2 in Pixeln des Originalbildes. Ohne TF32: mit TF32 (PyTorch-Standard
    für Faltungen auf der GPU) kippen einzelne Linien an den Schwellen - ohne sind sie dieselben wie mit dem
    ursprünglichen ONNX-Modell. TLC rechnet weiter wie gewohnt (mit TF32)."""
    import torch

    model = load()["mlsd"]
    with _lock:
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            return model(img)
        finally:
            torch.backends.cudnn.allow_tf32 = tf32


def hough(img: np.ndarray) -> np.ndarray:
    """BGR-Bild -> (N, 4) Linien x1 y1 x2 y2 (float32) in Pixeln des Originalbildes.

    Graustufen + 5×5-Gauß gegen Rauschkanten, Canny (Hysterese 50/150, 1 px dünne Kanten), probabilistische
    Hough-Transformation (ρ = 1 px, θ = 1°, ab HOUGH_VOTES Stimmen, Segmente ab HOUGH_MIN_LENGTH px, Lücken bis
    HOUGH_MAX_GAP px überbrückt) - auf dem auf HOUGH_MAX_DIM verkleinerten Bild, danach durch scale geteilt."""
    H, W = img.shape[:2]
    scale = min(1.0, HOUGH_MAX_DIM / max(H, W))
    work = cv2.resize(img, (max(1, round(W * scale)), max(1, round(H * scale))), interpolation=cv2.INTER_AREA) \
        if scale < 1 else img
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), CANNY_LOW, CANNY_HIGH)
    found = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_VOTES, minLineLength=HOUGH_MIN_LENGTH,
                            maxLineGap=HOUGH_MAX_GAP)
    if found is None:
        return np.zeros((0, 4), np.float32)
    return (found.reshape(-1, 4).astype(np.float32) / scale).astype(np.float32)


def detect(img: np.ndarray, detector: str = "mlsd") -> np.ndarray:
    """Linienerkennung nach Verfahren ("mlsd" | "hough")."""
    if detector == "hough":
        return hough(img)
    if detector == "mlsd":
        return mlsd(img)
    raise ValueError(f"Unbekannte Linienerkennung {detector!r} (mlsd | hough).")


def line_dicts(lines: np.ndarray) -> list[dict]:
    return [{"id": f"l{i + 1}", "point1": {"x": float(l[0]), "y": float(l[1])},
             "point2": {"x": float(l[2]), "y": float(l[3])}} for i, l in enumerate(lines)]


def annotate(image: str, detector: str = "mlsd") -> dict:
    """Bild (data:-URL) -> Gruppenstruktur: TLC liefert die drei Beziehungslisten, der Sortieralgorithmus
    (modell.build) macht daraus Richtungen mit Ebenen, Ebenen-Gruppen und Senkrecht-Gruppen."""
    from .modell import build, to_scene

    img = load_image(image)
    lines = detect(img, detector)
    if not len(lines):
        raise ValueError(f"{DETECTORS[detector]} hat keine Linien erkannt.")
    tlc = load()["tlc"]
    with _lock:
        relations = {kind: [(idx, "") for idx in model.predict(img, lines)] for kind, model in tlc.items()}
    H, W = img.shape[:2]
    ld = line_dicts(lines)
    return to_scene(build(W, H, ld, relations), ld)
