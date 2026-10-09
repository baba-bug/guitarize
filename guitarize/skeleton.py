"""旋律骨架规则：指弹生成器和旋律评分共用，保证"保留哪些音"和"考核哪些音"一致。"""
from __future__ import annotations

from typing import Sequence

EPS = 1e-3


def is_on_beat(start: float) -> bool:
    frac = start % 1.0
    return frac < EPS or frac > 1 - EPS


def skeleton_mask(starts: Sequence[float], durs: Sequence[float]) -> list[bool]:
    """按开始时间排好序的音符 → 是否为骨架音。

    骨架音 = 落在拍点上，或时值 ≥ 2/3 拍，或前面休止 ≥ 1 拍（乐句开头）。
    """
    out = []
    prev_end = None
    for s, d in zip(starts, durs):
        after_rest = prev_end is None or s - prev_end >= 1 - EPS
        out.append(is_on_beat(s) or d >= 2 / 3 - EPS or after_rest)
        prev_end = s + d
    return out
