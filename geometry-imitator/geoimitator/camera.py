"""

Kamera


"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geom import rot_x, rot_z, unit

# Welt -> Kamera für yaw = pitch = roll = 0: Blick entlang +X, rechts = -Y, unten = -Z
_BASE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


@dataclass
class Camera:
    K: np.ndarray        # 3x3 Intrinsik
    R: np.ndarray        # 3x3 Rotation Welt -> Kamera
    C: np.ndarray        # Kamerazentrum in Weltkoordinaten
    width: int
    height: int
    pose: dict | None = None  # yaw/pitch/roll/hfov für die Nachvollziehbarkeit

    @classmethod
    def from_pose(cls, position, yaw_deg: float, pitch_deg: float, roll_deg: float,
                  hfov_deg: float, width: int, height: int) -> "Camera":
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        K = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])
        # Kamera-Achsen: erst yaw (Welt), dann pitch um Kamera-x, dann roll um Kamera-z.
        # pitch > 0 hebt den Blick (Kamera-y zeigt nach unten -> negativer Winkel um Kamera-x).
        R = rot_z_cam(roll_deg) @ rot_x(-pitch_deg) @ _BASE @ rot_z(yaw_deg).T
        pose = {"position": [float(v) for v in position], "yawDeg": yaw_deg, "pitchDeg": pitch_deg,
                "rollDeg": roll_deg, "hfovDeg": hfov_deg}
        return cls(K, R, np.asarray(position, float), width, height, pose)

    @classmethod
    def look_at(cls, position, target, roll_deg: float, hfov_deg: float, width: int, height: int) -> "Camera":
        d = unit(np.asarray(target, float) - np.asarray(position, float))
        yaw = math.degrees(math.atan2(d[1], d[0]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
        return cls.from_pose(position, yaw, pitch, roll_deg, hfov_deg, width, height)

    # --- Projektion --------------------------------------------------------
    @property
    def forward(self) -> np.ndarray:
        return self.R[2]

    def to_cam(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return (X - self.C) @ self.R.T

    def project_cam(self, Xc: np.ndarray) -> np.ndarray:
        Xc = np.atleast_2d(Xc)
        uvw = Xc @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3]

    def project(self, X: np.ndarray) -> np.ndarray:
        return self.project_cam(self.to_cam(X))

    def vanishing_point(self, direction) -> np.ndarray:
        """Homogener Fluchtpunkt K R d (nicht normiert, w ~ 0 -> unendlich fern)."""
        return self.K @ self.R @ unit(direction)

    def vanishing_line(self, normal) -> np.ndarray:
        """Fluchtgerade einer Ebenen-Orientierung: l = K^-T R n (normiert auf |l[:2]| = 1)."""
        l = np.linalg.inv(self.K).T @ (self.R @ unit(normal))
        return l / (np.linalg.norm(l[:2]) + 1e-12)

    def scaled(self, width: int, height: int) -> "Camera":
        """Dieselbe Kamera für eine andere Bildauflösung (gleiches Seitenverhältnis erwartet)."""
        sx, sy = width / self.width, height / self.height
        S = np.diag([sx, sy, 1.0])
        K = S @ self.K
        # Pixelmitten-Konvention (cx = (w-1)/2) beibehalten
        K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        return Camera(K, self.R.copy(), self.C.copy(), width, height, self.pose)

    def homography_rotation(self, R_delta: np.ndarray) -> np.ndarray:
        """Bild-Homographie einer reinen Kameradrehung um R_delta."""
        return self.K @ R_delta @ np.linalg.inv(self.K)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "K": self.K.tolist(), "R": self.R.tolist(),
                "C": self.C.tolist(), "pose": self.pose}

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(np.array(d["K"], float), np.array(d["R"], float), np.array(d["C"], float),
                   int(d["width"]), int(d["height"]), d.get("pose"))


def rot_z_cam(deg: float) -> np.ndarray:
    """Rollen um die optische Achse."""
    return rot_z(deg)
