"""Gelabelte Linien + Beziehungen als sample.json"""

from __future__ import annotations

import numpy as np

from .camera import Camera
from .matching import MatchResult
from .scenes import vanishing_point_dict



def scaled_segments(scene_json: dict, width: int, height: int) -> list[dict]:
    W0, H0 = scene_json["image"]["width"], scene_json["image"]["height"]
    sx, sy = width / W0, height / H0
    out = []
    for s in scene_json["segments2d"]:
        s = dict(s)
        for k in ("p1", "p2", "p1Exact", "p2Exact"):
            if k in s:
                s[k] = [s[k][0] * sx, s[k][1] * sy]
        out.append(s)
    return out


def _pt(x, y, ndigits: int | None = 3) -> dict:
    if ndigits is None:
        return {"x": float(x), "y": float(y)}
    return {"x": round(float(x), ndigits), "y": round(float(y), ndigits)}


def build_sample(scene_json: dict, gen_meta: dict, det: np.ndarray, gt: list[dict], match: MatchResult,
                 width: int, height: int, detector_info: dict) -> dict:
    cam = Camera.from_dict(scene_json["camera"]).scaled(width, height)
    lines = []
    for m, (x1, y1, x2, y2) in enumerate(det):
        st = match.status[m]
        entry = {"index": m, "point1": _pt(x1, y1), "point2": _pt(x2, y2),
                 "length": round(float(np.hypot(x2 - x1, y2 - y1)), 3), "status": st, "label": None, "match": None}
        g = match.gt_index[m]
        if g >= 0:
            entry["match"] = {"segment": gt[g]["id"], "angleDeg": round(float(match.angle_deg[m]), 3),
                              "distPx": round(float(match.dist_px[m]), 3), "overlap": round(float(match.overlap[m]), 3),
                              "cost": round(float(match.cost[m]), 4)}
        if st == "matched":
            s = gt[g]
            entry["label"] = {"segment": s["id"], "line3d": s["line3d"], "direction": s["direction"],
                              "planes": s["planes"], "planeFamilies": s["planeFamilies"], "object": s["object"],
                              "kind": s["kind"], "perturbed": bool(s.get("perturbed"))}
        lines.append(entry)

    labelled = [l for l in lines if l["label"]]
    by_dir: dict[str, list[int]] = {}
    by_plane: dict[str, list[int]] = {}
    by_fam: dict[str, list[int]] = {}
    for l in labelled:
        lab = l["label"]
        by_dir.setdefault(lab["direction"], []).append(l["index"])
        for p in lab["planes"]:
            by_plane.setdefault(p, []).append(l["index"])
        for f in lab["planeFamilies"]:
            by_fam.setdefault(f, []).append(l["index"])

    perturbed_dirs = sorted({p["direction"] for p in scene_json.get("perturbation", [])})
    directions = []
    for d in scene_json["directions"]:
        directions.append({"id": d["id"], "vector": d["vector"], "kind": d["kind"],
                           "vanishingPoint": vanishing_point_dict(cam, d["vector"]),
                           "lineIndices": by_dir.get(d["id"], []), "perturbed": d["id"] in perturbed_dirs})
    families = []
    for f in scene_json["planeFamilies"]:
        families.append({"id": f["id"], "normal": f["normal"], "planes": f["planes"], "directions": f["directions"],
                         "supportingDirections": f["supportingDirections"],
                         "vanishingLine": np.round(cam.vanishing_line(f["normal"]), 9).tolist(),
                         "lineIndices": by_fam.get(f["id"], [])})
    planes = [{"id": p["id"], "family": p["family"], "normal": p["normal"], "offset": p["offset"],
               "directions": p["directions"], "lineIndices": by_plane.get(p["id"], [])} for p in scene_json["planes"]]

    present = {d for d, idx in by_dir.items() if len(idx) >= 2}
    rel = scene_json["relations"]
    relations = {
        "parallel": {d: idx for d, idx in by_dir.items() if len(idx) >= 2},
        "coplanar": {p: idx for p, idx in by_plane.items() if len(idx) >= 2},
        "parallelPlanes": {f: idx for f, idx in by_fam.items() if len(idx) >= 2},
        "perpendicularPairs": [p for p in rel["perpendicularPairs"] if set(p) <= present],
        "perpendicularTriples": [t for t in rel["perpendicularTriples"] if set(t) <= present],
        "vanishingLines": {f["id"]: [d for d in f["directions"] if d in present] for f in scene_json["planeFamilies"]
                           if sum(d in present for d in f["directions"]) >= 2},
    }

    dir_ids = sorted(present, key=lambda s: int(s[1:]))
    dir_index = {d: i for i, d in enumerate(dir_ids)}
    training = {
        "directionIds": dir_ids,
        # Ziel für das Sortierungsmodell: Cluster-Index pro Linie, -1 = ignorieren (unknown/ambiguous/Einzellinie)
        "directionLabel": [dir_index.get(l["label"]["direction"], -1) if l["label"] else -1 for l in lines],
        "ignore": [l["status"] != "matched" for l in lines],
        "perturbed": [bool(l["label"] and l["label"]["perturbed"]) for l in lines],
    }

    return {
        "version": 1,
        "id": f"{scene_json['id']}/{gen_meta['id']}",
        "sceneId": scene_json["id"], "generationId": gen_meta["id"],
        "sceneType": scene_json["sceneType"],
        "image": {"path": "image.png", "width": width, "height": height},
        "source": {k: gen_meta.get(k) for k in ("generator", "prompt", "negativePrompt", "offTopic", "style",
                                                    "seed", "params")},
        "detector": detector_info,
        "camera": cam.to_dict(),
        "match": {**match.metrics, "accepted": match.accepted, "reasons": match.reasons},
        "perturbation": scene_json.get("perturbation", []),
        "noiseSigmaPx": scene_json.get("noiseSigmaPx", 0.0),
        "lines": lines,
        "directions": directions,
        "planeFamilies": families,
        "planes": planes,
        "relations": relations,
        "training": training,
    }

