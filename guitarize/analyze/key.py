"""调性估计：Krumhansl–Schmuckler 模板与平均色度做相关。"""
from __future__ import annotations

import numpy as np

from .. import theory

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_STEPS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}


def estimate_key(chroma_mean) -> tuple[str, str, float]:
    c = np.asarray(chroma_mean, dtype=float)
    best: tuple[str, str, float] | None = None
    for mode, profile in (("major", MAJOR), ("minor", MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(c, np.roll(profile, tonic))[0, 1])
            if best is None or r > best[2]:
                best = (theory.pc_name(tonic), mode, r)
    return best


def scale_pcs(tonic: str, mode: str) -> set[int]:
    root = theory.pc(tonic)
    return {(root + s) % 12 for s in _STEPS[mode]}


def parse_key(text: str) -> tuple[str, str]:
    """'B' → ('B', 'major')；'G#m' → ('G#', 'minor')。"""
    t = text.strip()
    if t.endswith("m"):
        return theory.pc_name(theory.pc(t[:-1])), "minor"
    return theory.pc_name(theory.pc(t)), "major"
