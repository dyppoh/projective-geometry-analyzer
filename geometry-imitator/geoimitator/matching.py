"""Erkannte Linien den Originalsegmenten zuordnen.

Zuordnung erkannt -> original ist n:1 (DeepLSD zerlegt Linien oft in mehrere
Stücke, generierte Objekte verdecken Teile). Jede erkannte Linie bekommt
höchstens ein Originalsegment, wenn

* beide Endpunkte <= max_dist von der Original-Geraden entfernt sind,
* mindestens min_endpoint_hits der vier Endpunkte (egal von welcher Linie) mit
  ihrer Senkrechten die jeweils andere, nicht verlängerte Strecke treffen.

Unter mehreren Kandidaten gewinnt der mit den geringsten Kosten aus Winkel,
Abstand und Längenunterschied. Alles ohne Kandidat ist "unknown" (vom Generator
erfundene Linien: Möbel, Textur, ...) und gehört beim Training nicht in den
Loss - als "nicht parallel" gewertet wären es falsche Negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MatchConfig


@dataclass
class MatchResult:
    status: list[str]               # pro erkannter Linie: matched | unknown
    gt_index: np.ndarray            # pro erkannter Linie: Index des Originalsegments oder -1
    angle_deg: np.ndarray
    dist_px: np.ndarray
    overlap: np.ndarray
    cost: np.ndarray
    coverage: np.ndarray            # pro Originalsegment: wiedergefundener Längenanteil
    metrics: dict
    accepted: bool
    reasons: list[str]


def match_lines(gt: np.ndarray, gt_dirs: list[str], det: np.ndarray, image_width: int,
                cfg: MatchConfig, gt_lines: list[str] | None = None) -> MatchResult:
    """gt: (N,4), det: (M,4) - gleiche Pixelkoordinaten. gt_lines: 3D-Linien-ID je
    Originalsegment (Bruchstücke derselben 3D-Linie zählen für eine Gruppe nur einmal)."""
    N, M = len(gt), len(det)
    max_dist = max(cfg.min_dist_px, cfg.max_dist_ratio * image_width)
    status = ["unknown"] * M
    gt_index = np.full(M, -1)
    nan = np.full(M, np.nan)
    angle_o, dist_o, ov_o, cost_o = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    coverage = np.zeros(N)
    if N and M:
        P, Q = gt[:, :2], gt[:, 2:]
        Lg = np.linalg.norm(Q - P, axis=1)
        U = (Q - P) / Lg[:, None]
        Nrm = np.stack([-U[:, 1], U[:, 0]], 1)
        A, B = det[:, :2], det[:, 2:]
        Ld = np.linalg.norm(B - A, axis=1)
        Ud = (B - A) / Ld[:, None]
        # (N, M)
        cosang = np.abs(U @ Ud.T)
        ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
        dA = A[None] - P[:, None]
        dB = B[None] - P[:, None]
        da = np.abs(np.einsum("nmk,nk->nm", dA, Nrm))
        db = np.abs(np.einsum("nmk,nk->nm", dB, Nrm))
        dist = np.maximum(da, db)
        ta = np.einsum("nmk,nk->nm", dA, U)
        tb = np.einsum("nmk,nk->nm", dB, U)
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        ov_len = np.clip(np.minimum(hi, Lg[:, None]) - np.maximum(lo, 0), 0, None)
        ov = ov_len / Ld[None]
        # Endpunkt-Treffer: Lot eines Endpunkts trifft die andere, nicht verlängerte Strecke
        sa = np.einsum("nmk,mk->nm", P[:, None] - A[None], Ud)
        sb = np.einsum("nmk,mk->nm", Q[:, None] - A[None], Ud)
        auf = lambda t, n: ((t >= 0) & (t <= n)).astype(int)  # noqa: E731
        hits = auf(ta, Lg[:, None]) + auf(tb, Lg[:, None]) + auf(sa, Ld[None]) + auf(sb, Ld[None])
        valid = (ang <= cfg.max_angle_deg) & (dist <= max_dist) & (hits >= cfg.min_endpoint_hits)
        len_diff = np.abs(Ld[None] - Lg[:, None]) / np.maximum(Ld[None], Lg[:, None])
        cost = np.where(valid, ang / cfg.max_angle_deg + dist / max_dist + len_diff, np.inf)

        intervals: list[list[tuple[float, float]]] = [[] for _ in range(N)]
        gt_dirs_arr = np.array(gt_dirs)
        for m in range(M):
            col = cost[:, m]
            best = int(np.argmin(col))
            if not np.isfinite(col[best]):
                continue
            gt_index[m] = best
            angle_o[m], dist_o[m], ov_o[m], cost_o[m] = ang[best, m], dist[best, m], ov[best, m], col[best]
            status[m] = "matched"
            # Abdeckung auch für alle anderen gültigen Kandidaten derselben Richtung: innerhalb
            # der Abstandstoleranz sind z.B. Fensteröffnung und Laibungskante nicht unterscheidbar
            for n in np.where(valid[:, m] & (gt_dirs_arr == gt_dirs_arr[best]))[0]:
                intervals[n].append((max(lo[n, m], 0), min(hi[n, m], Lg[n])))
        for n in range(N):
            coverage[n] = _union_length(intervals[n]) / Lg[n]
    else:
        Lg = np.linalg.norm(gt[:, 2:] - gt[:, :2], axis=1) if N else np.zeros(0)

    matched = np.array([s == "matched" for s in status], bool)
    gt_lines = gt_lines or [str(i) for i in range(N)]
    dirs_matched: dict[str, set] = {}
    for m in np.where(matched)[0]:
        dirs_matched.setdefault(gt_dirs[gt_index[m]], set()).add(gt_lines[gt_index[m]])
    gt_groups: dict[str, set] = {}
    for d, l in zip(gt_dirs, gt_lines):
        gt_groups.setdefault(d, set()).add(l)
    metrics = {
        "gtSegments": N, "detectedLines": M,
        "matchedLines": int(matched.sum()),
        "ambiguousLines": int(sum(s == "ambiguous" for s in status)),
        "unknownLines": int(sum(s == "unknown" for s in status)),
        "precision": float(matched.mean()) if M else 0.0,
        # Anteil der Originalsegmente, denen mindestens eine erkannte Linie zugeordnet ist
        "recallLines": len(set(gt_index[matched].tolist())) / N if N else 0.0,
        "recallLength": float((coverage * Lg).sum() / Lg.sum()) if N else 0.0,
        "recallSegments": float((coverage >= cfg.segment_coverage).mean()) if N else 0.0,
        "gtGroups": int(sum(len(c) >= 2 for c in gt_groups.values())),
        "matchedGroups": int(sum(len(c) >= 2 for c in dirs_matched.values())),
        "meanAngleDeg": float(np.nanmean(angle_o[matched])) if matched.any() else None,
        "meanDistPx": float(np.nanmean(dist_o[matched])) if matched.any() else None,
        "maxDistPx": max_dist,
    }
    reasons = []
    if metrics["recallLines"] <= cfg.min_recall_lines:
        reasons.append(f"recallLines {metrics['recallLines']:.2f} <= {cfg.min_recall_lines}")
    return MatchResult(status, gt_index, angle_o, dist_o, ov_o, cost_o, coverage, metrics, not reasons, reasons)


def _union_length(iv: list[tuple[float, float]]) -> float:
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(iv):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total
