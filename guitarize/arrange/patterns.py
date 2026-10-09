"""弹唱节奏型：8 个八分格。D = 下扫，U = 上扫，x = 闷音下扫，- = 延续。按与原曲起音的相关度选型。"""
from __future__ import annotations

import numpy as np

PATTERNS = {
    "whole": "D-------",
    "half": "D---D---",
    "quarter": "D-D-D-D-",
    "push": "D-DUD-DU",
    "pop": "D-DU-UDU",
    "eighth": "DUDUDUDU",
}
DEFAULT = "push"
_WEIGHT = {"D": 1.0, "x": 0.8, "U": 0.6, "-": 0.0}


def slots_per_beat(feel: str) -> int:
    return 3 if feel == "shuffle" else 2


def accent_vector(pattern: str, feel: str) -> np.ndarray:
    """节奏型 → 每拍 slots_per_beat 格的重音向量；反拍八分落在 swing 位置（shuffle 为第 3 格）。"""
    spb = slots_per_beat(feel)
    v = np.zeros(4 * spb)
    for k, ch in enumerate(pattern):
        pos = (k // 2) * spb + (0 if k % 2 == 0 else spb - 1)
        v[pos] = max(v[pos], _WEIGHT[ch])
    return v


def choose_pattern(onset: np.ndarray | None, feel: str) -> str:
    """与原曲起音向量相关度最高的节奏型；相关度相同取扫弦次数少的；无数据时用默认型。"""
    if onset is None or np.std(onset) == 0:
        return DEFAULT
    best = None
    for name, pattern in PATTERNS.items():
        r = float(np.corrcoef(accent_vector(pattern, feel), onset)[0, 1])
        key = (round(r, 3), -sum(c != "-" for c in pattern))
        if best is None or key > best[0]:
            best = (key, name)
    return best[1]
